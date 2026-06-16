"""DuckDB-backed persistent coverage store.

Materialises CoverageEntry rows into a queryable columnar database,
enriched with Sprint-1 atlas IDs (UBERON, Allen CCF, NCBITaxon).

Tables
------
datasets          — one row per corpus record
coverage_entries  — exploded (dataset × dimension × value) rows
ontology_regions  — 106-region atlas with UBERON + CCF IDs
ontology_species  — 10 species with NCBITaxon IDs

Typical usage
-------------
    store = CoverageStore("data/coverage/ledger.duckdb")
    store.build(corpus_path="data/corpus/normalized/combined_corpus.jsonl")
    store.gap_matrix("brain_regions", "modalities").show()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import yaml

from neural_search.coverage.ledger import infer_access_tier, load_dataset_mappings
from neural_search.ontology.loader import get_brain_regions

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/coverage/ledger.duckdb")
ONTOLOGY_REGIONS_PATH = Path("data/ontology/brain_regions.yaml")
ONTOLOGY_SPECIES_PATH = Path("data/ontology/species_taxonomy.yaml")

# ─── DDL ─────────────────────────────────────────────────────────────────────

_CREATE_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id   TEXT PRIMARY KEY,
    source       TEXT,
    source_id    TEXT,
    title        TEXT,
    access_tier  TEXT,
    has_behavior BOOLEAN,
    has_raw_data BOOLEAN,
    has_standard_format BOOLEAN,
    first_seen   TEXT,
    ingested_at  TEXT,
    snapshot_id  TEXT
)
"""

_CREATE_COVERAGE_ENTRIES = """
CREATE TABLE IF NOT EXISTS coverage_entries (
    entry_id          TEXT PRIMARY KEY,
    dataset_id        TEXT,
    source            TEXT,
    dimension         TEXT,
    value_id          TEXT,
    label             TEXT,
    confidence        DOUBLE,
    evidence_tier     TEXT,
    access_tier       TEXT,
    analysis_level    TEXT,
    source_field      TEXT,
    evidence_text     TEXT,
    snapshot_id       TEXT,
    -- Sprint 1 ontology enrichment
    uberon_id         TEXT,
    allen_ccf_mouse_id TEXT,
    ncbitaxon_id      TEXT
)
"""

_CREATE_ONTOLOGY_REGIONS = """
CREATE TABLE IF NOT EXISTS ontology_regions (
    id                TEXT PRIMARY KEY,
    label             TEXT,
    uberon_id         TEXT,
    allen_ccf_mouse_id TEXT,
    waxholm_rat_id    TEXT,
    allen_human_id    TEXT,
    parents           TEXT,   -- JSON array
    species_scope     TEXT,   -- JSON array
    system            TEXT
)
"""

_CREATE_ONTOLOGY_SPECIES = """
CREATE TABLE IF NOT EXISTS ontology_species (
    id             TEXT PRIMARY KEY,
    label          TEXT,
    ncbitaxon_id   TEXT,
    primary_atlas  TEXT,
    is_model_organism BOOLEAN
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ce_dataset ON coverage_entries (dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_dimension ON coverage_entries (dimension)",
    "CREATE INDEX IF NOT EXISTS idx_ce_value ON coverage_entries (value_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_dim_val ON coverage_entries (dimension, value_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_uberon ON coverage_entries (uberon_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_ccf ON coverage_entries (allen_ccf_mouse_id)",
]

# ─── NCBITaxon lookup ─────────────────────────────────────────────────────────
_NCBI_MAP: dict[str, str] = {
    "mus_musculus": "NCBITaxon:10090",
    "mouse": "NCBITaxon:10090",
    "rattus_norvegicus": "NCBITaxon:10116",
    "rat": "NCBITaxon:10116",
    "homo_sapiens": "NCBITaxon:9606",
    "human": "NCBITaxon:9606",
    "macaca_mulatta": "NCBITaxon:9544",
    "macaque": "NCBITaxon:9544",
    "macaca_fascicularis": "NCBITaxon:9541",
    "callithrix_jacchus": "NCBITaxon:9483",
    "marmoset": "NCBITaxon:9483",
    "danio_rerio": "NCBITaxon:7955",
    "zebrafish": "NCBITaxon:7955",
    "drosophila_melanogaster": "NCBITaxon:7227",
    "drosophila": "NCBITaxon:7227",
    "caenorhabditis_elegans": "NCBITaxon:6239",
    "c_elegans": "NCBITaxon:6239",
}


def _ncbitaxon(species_id: str) -> str | None:
    norm = species_id.casefold().replace(" ", "_").replace("-", "_")
    return _NCBI_MAP.get(norm)


# ─── Ontology loaders ─────────────────────────────────────────────────────────

def _load_ontology_region_rows() -> list[dict[str, Any]]:
    regions = get_brain_regions()
    rows = []
    for r in regions:
        refs = r.atlas_refs
        rows.append({
            "id": r.id,
            "label": r.label,
            "uberon_id": refs.get("uberon"),
            "allen_ccf_mouse_id": refs.get("allen_ccf_mouse"),
            "waxholm_rat_id": refs.get("waxholm_rat"),
            "allen_human_id": refs.get("allen_human"),
            "parents": json.dumps(r.parents),
            "species_scope": json.dumps(r.species_scope),
            "system": r.system,
        })
    return rows


def _load_ontology_species_rows() -> list[dict[str, Any]]:
    if not ONTOLOGY_SPECIES_PATH.exists():
        return []
    raw = yaml.safe_load(ONTOLOGY_SPECIES_PATH.read_text())
    rows = []
    for s in raw.get("species", []):
        rows.append({
            "id": s["id"],
            "label": s["label"],
            "ncbitaxon_id": s["ncbitaxon_id"],
            "primary_atlas": s.get("primary_atlas"),
            "is_model_organism": s.get("is_model_organism", False),
        })
    return rows


# ─── Fast entry builder (skips NLP text matching) ────────────────────────────

_DIRECT_DIMS: dict[str, float] = {
    "brain_regions": 0.75,
    "modalities": 0.80,
    "species": 0.80,
    "tasks": 0.75,
    "behavioral_events": 0.70,
}

_MODALITY_TO_SCALE: dict[str, str] = {
    "bold": "bold_voxel_timeseries",
    "fmri": "bold_voxel_timeseries",
    "calcium_imaging": "calcium_roi_fluorescence",
    "two_photon": "calcium_roi_fluorescence",
    "calcium_widefield": "widefield_fluorescence",
    "lfp": "local_field_potential",
    "eeg": "eeg_sensor_timeseries",
    "meg": "meg_sensor_timeseries",
    "ecog": "ecog_surface_potential",
    "extracellular_ephys": "raw_extracellular_voltage",
    "neuropixels": "raw_extracellular_voltage",
    "single_nucleus_rnaseq": "transcriptomic_cell_profile",
    "single_cell_rnaseq": "transcriptomic_cell_profile",
    "connectomics": "connectomic_edge",
    "neuron_morphology": "connectomic_edge",
}


def _norm_val(v: Any) -> str:
    return str(v or "").casefold().replace("-", "_").replace(" ", "_").strip("_")


_MODALITY_TO_ANALYSIS_LEVEL: dict[str, str] = {
    "fmri": "voxelwise_imaging",
    "bold": "voxelwise_imaging",
    "mri": "voxelwise_imaging",
    "eeg": "mesoscale_field_potential",
    "meg": "mesoscale_field_potential",
    "lfp": "mesoscale_field_potential",
    "ecog": "mesoscale_field_potential",
    "extracellular_ephys": "unit_activity",
    "neuropixels": "unit_activity",
    "calcium_imaging": "population_dynamics",
    "two_photon": "population_dynamics",
    "widefield_imaging": "population_dynamics",
    "single_nucleus_rnaseq": "molecular_profile",
    "single_cell_rnaseq": "molecular_profile",
    "connectomics": "circuit_structure",
    "neuron_morphology": "circuit_structure",
}


def _fast_analysis_level(rec: dict[str, Any]) -> str:
    """Infer primary analysis level from modalities only — no NLP."""
    modalities = rec.get("modalities") or []
    for mod in modalities:
        m = _norm_val(mod if isinstance(mod, str) else mod.get("id", ""))
        level = _MODALITY_TO_ANALYSIS_LEVEL.get(m)
        if level:
            return level
    flags = rec.get("usability_flags") or {}
    if flags.get("has_raw_data"):
        return "raw_signal"
    if flags.get("has_behavior") or rec.get("tasks") or rec.get("behavioral_events"):
        return "behavior_correlation"
    return "unspecified_analysis_level"


def _fast_entry_rows(
    records: list[dict[str, Any]],
    *,
    snapshot_id: str,
    region_atlas: dict[str, dict[str, str]],
    existing_entries: set[str],
) -> list[tuple]:
    """Build coverage entry rows from declared fields only — no NLP text matching."""
    from neural_search.coverage.ledger import MODALITY_TO_RECORDING_SCALE
    from neural_search.species import canonical_species_id

    rows: list[tuple] = []
    for rec in records:
        did = str(
            rec.get("dataset_id")
            or f"dataset:{rec.get('source')}:{rec.get('source_id')}"
        )
        source = str(rec.get("source") or "unknown")
        access = infer_access_tier(rec)
        primary_level = _fast_analysis_level(rec)
        prov = _norm_val(rec.get("brain_regions_provenance") or "")

        for field_name, base_conf in _DIRECT_DIMS.items():
            raw_list = rec.get(field_name) or []
            if isinstance(raw_list, str):
                raw_list = [raw_list]
            for raw in raw_list:
                if isinstance(raw, dict):
                    vid = _norm_val(raw.get("id") or raw.get("value") or raw.get("label") or "")
                    # label:brain_region:X:source format → use X as canonical id
                    if vid.startswith("label:"):
                        parts = vid.split(":")
                        vid = parts[2] if len(parts) >= 3 else _norm_val(raw.get("label") or "")
                    label = str(raw.get("label") or vid)
                    conf = float(raw.get("confidence", base_conf) or base_conf)
                    src_field = raw.get("source_field") or field_name
                else:
                    vid = _norm_val(raw)
                    label = str(raw)
                    conf = base_conf
                    src_field = field_name
                if not vid:
                    continue
                if field_name == "species":
                    vid = canonical_species_id(label) or vid
                    ncbi = _ncbitaxon(vid)
                    uberon, ccf = None, None
                elif field_name == "brain_regions":
                    refs = region_atlas.get(vid, {})
                    uberon = refs.get("uberon")
                    ccf = refs.get("allen_ccf_mouse")
                    ncbi = None
                else:
                    uberon, ccf, ncbi = None, None, None

                # Silver label cap
                if any(t in prov for t in ("llm", "gemini", "silver")):
                    conf = min(conf, 0.70)
                    evidence_tier = "silver_inferred"
                elif field_name == "brain_regions" and any(
                    t in prov for t in ("electrode", "nwb", "bids")
                ):
                    evidence_tier = "structured_metadata"
                else:
                    evidence_tier = "declared_metadata"

                entry_id = f"coverage:{snapshot_id}:{_norm_val(did)}:{field_name}:{vid}"
                if entry_id in existing_entries:
                    continue
                rows.append((
                    entry_id, did, source, field_name, vid, label,
                    conf, evidence_tier, access, primary_level,
                    src_field, None, snapshot_id,
                    uberon, ccf, ncbi,
                ))

        # Modality-inferred recording scales
        for raw in (rec.get("modalities") or []):
            mod = _norm_val(raw if isinstance(raw, str) else raw.get("id", ""))
            for scale_id in MODALITY_TO_RECORDING_SCALE.get(mod, ()):
                entry_id = (
                    f"coverage:{snapshot_id}:{_norm_val(did)}:recording_scales:{scale_id}"
                )
                if entry_id in existing_entries:
                    continue
                rows.append((
                    entry_id, did, source, "recording_scales", scale_id, scale_id,
                    0.72, "inferred_metadata", access, primary_level,
                    "modalities", f"inferred from modality {mod!r}", snapshot_id,
                    None, None, None,
                ))

    return rows


# ─── Core store ───────────────────────────────────────────────────────────────

class CoverageStore:
    """Persistent DuckDB coverage store."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(_CREATE_DATASETS)
        self._conn.execute(_CREATE_COVERAGE_ENTRIES)
        self._conn.execute(_CREATE_ONTOLOGY_REGIONS)
        self._conn.execute(_CREATE_ONTOLOGY_SPECIES)
        for idx in _INDEXES:
            self._conn.execute(idx)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CoverageStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Build / refresh ──────────────────────────────────────────────────────

    def upsert_ontology(self) -> None:
        """Load/replace ontology dimension tables from YAML sources."""
        region_rows = _load_ontology_region_rows()
        species_rows = _load_ontology_species_rows()

        self._conn.execute("DELETE FROM ontology_regions")
        if region_rows:
            self._conn.executemany(
                """INSERT INTO ontology_regions
                   (id, label, uberon_id, allen_ccf_mouse_id, waxholm_rat_id,
                    allen_human_id, parents, species_scope, system)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (r["id"], r["label"], r["uberon_id"], r["allen_ccf_mouse_id"],
                     r["waxholm_rat_id"], r["allen_human_id"], r["parents"],
                     r["species_scope"], r["system"])
                    for r in region_rows
                ],
            )

        self._conn.execute("DELETE FROM ontology_species")
        if species_rows:
            self._conn.executemany(
                """INSERT INTO ontology_species
                   (id, label, ncbitaxon_id, primary_atlas, is_model_organism)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (s["id"], s["label"], s["ncbitaxon_id"],
                     s["primary_atlas"], s["is_model_organism"])
                    for s in species_rows
                ],
            )
        log.info("Ontology: %d regions, %d species", len(region_rows), len(species_rows))

    def build(
        self,
        corpus_path: str | Path,
        *,
        snapshot_id: str = "current",
        replace: bool = False,
    ) -> dict[str, int]:
        """Build (or incrementally extend) the store from a corpus path."""
        if replace:
            self._conn.execute("DELETE FROM coverage_entries")
            self._conn.execute("DELETE FROM datasets")

        self.upsert_ontology()

        records = load_dataset_mappings(corpus_path)
        log.info("Loaded %d records from %s", len(records), corpus_path)

        # Build atlas-ref lookup for enrichment
        region_atlas: dict[str, dict[str, str]] = {}
        for r in get_brain_regions():
            region_atlas[r.id] = dict(r.atlas_refs)

        existing_datasets: set[str] = {
            row[0] for row in self._conn.execute("SELECT dataset_id FROM datasets").fetchall()
        }
        existing_entries: set[str] = {
            row[0] for row in self._conn.execute("SELECT entry_id FROM coverage_entries").fetchall()
        }

        new_records = [
            rec for rec in records
            if _dataset_id(rec) not in existing_datasets
        ]
        log.info("New records: %d (skipping %d already stored)", len(new_records),
                 len(records) - len(new_records))

        dataset_rows = []
        for rec in new_records:
            did = _dataset_id(rec)
            flags = rec.get("usability_flags") or {}
            dataset_rows.append((
                did,
                str(rec.get("source") or ""),
                str(rec.get("source_id") or ""),
                str(rec.get("title") or "")[:500],
                infer_access_tier(rec),
                bool(flags.get("has_behavior")),
                bool(flags.get("has_raw_data")),
                bool(flags.get("has_standard_format")),
                str(rec.get("created_at") or ""),
                "",
                snapshot_id,
            ))

        if dataset_rows:
            self._conn.executemany(
                """INSERT OR IGNORE INTO datasets
                   (dataset_id, source, source_id, title, access_tier,
                    has_behavior, has_raw_data, has_standard_format,
                    first_seen, ingested_at, snapshot_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                dataset_rows,
            )

        # Fast direct-parse: read declared fields without NLP text matching.
        # The full build_coverage_entries() path runs match_recording_scales(text)
        # on every record's title+description (~1-2s/record for 7K records = hours).
        # For gap analysis we only need the declared structured labels.
        entry_rows = _fast_entry_rows(
            new_records,
            snapshot_id=snapshot_id,
            region_atlas=region_atlas,
            existing_entries=existing_entries,
        )

        if entry_rows:
            self._conn.executemany(
                """INSERT OR IGNORE INTO coverage_entries
                   (entry_id, dataset_id, source, dimension, value_id, label,
                    confidence, evidence_tier, access_tier, analysis_level,
                    source_field, evidence_text, snapshot_id,
                    uberon_id, allen_ccf_mouse_id, ncbitaxon_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                entry_rows,
            )

        stats = {
            "new_datasets": len(dataset_rows),
            "new_entries": len(entry_rows),
            "total_datasets": self._conn.execute(
                "SELECT COUNT(*) FROM datasets"
            ).fetchone()[0],
            "total_entries": self._conn.execute(
                "SELECT COUNT(*) FROM coverage_entries"
            ).fetchone()[0],
        }
        log.info("Store: %d datasets, %d entries", stats["total_datasets"], stats["total_entries"])
        return stats

    # ── Analytical queries ───────────────────────────────────────────────────

    _ALLOWED_DIMS = frozenset({
        "brain_regions", "modalities", "species", "tasks", "recording_scales",
    })

    @classmethod
    def _require_dim(cls, value: str) -> str:
        if value not in cls._ALLOWED_DIMS:
            raise ValueError(f"Invalid dimension {value!r}; must be one of {sorted(cls._ALLOWED_DIMS)}")
        return value

    def gap_matrix(
        self,
        row_dim: str = "brain_regions",
        col_dim: str = "modalities",
        *,
        species_filter: str | None = None,
        min_confidence: float = 0.65,
    ) -> duckdb.DuckDBPyRelation:
        """Cross-tabulation: row_dim × col_dim dataset counts."""
        row_dim = self._require_dim(row_dim)
        col_dim = self._require_dim(col_dim)

        # species_filter is a user-supplied value — bind via parameter, not f-string
        params: list[Any] = []
        species_clause = ""
        if species_filter:
            species_clause = """
              AND ce_r.dataset_id IN (
                  SELECT dataset_id FROM coverage_entries
                  WHERE dimension = 'species' AND value_id = ?
                  AND confidence >= ?
              )"""
            params = [species_filter, min_confidence]

        sql = f"""
        SELECT
            ce_r.value_id  AS {row_dim.rstrip('s')},
            ce_c.value_id  AS {col_dim.rstrip('s')},
            COUNT(DISTINCT ce_r.dataset_id) AS n_datasets
        FROM coverage_entries ce_r
        JOIN coverage_entries ce_c ON ce_r.dataset_id = ce_c.dataset_id
        WHERE ce_r.dimension = '{row_dim}'
          AND ce_c.dimension = '{col_dim}'
          AND ce_r.confidence >= {min_confidence}
          AND ce_c.confidence >= {min_confidence}
          {species_clause}
        GROUP BY 1, 2
        ORDER BY 3 DESC
        """
        return self._conn.sql(sql, params=params) if params else self._conn.sql(sql)

    def uncovered_regions(self, *, min_confidence: float = 0.65) -> duckdb.DuckDBPyRelation:
        """Ontology regions with zero corpus datasets."""
        sql = f"""
        SELECT o.id, o.label, o.uberon_id, o.allen_ccf_mouse_id, o.parents
        FROM ontology_regions o
        LEFT JOIN (
            SELECT DISTINCT value_id
            FROM coverage_entries
            WHERE dimension = 'brain_regions' AND confidence >= {min_confidence}
        ) ce ON ce.value_id = o.id
        WHERE ce.value_id IS NULL
        ORDER BY o.id
        """
        return self._conn.sql(sql)

    def source_coverage_rates(self, *, min_confidence: float = 0.65) -> duckdb.DuckDBPyRelation:
        """Per-source coverage rates across brain_regions, modalities, species."""
        sql = f"""
        WITH totals AS (
            SELECT source, COUNT(*) AS n_total FROM datasets GROUP BY source
        ),
        covered AS (
            SELECT source, dimension, COUNT(DISTINCT dataset_id) AS n_covered
            FROM coverage_entries
            WHERE confidence >= {min_confidence}
              AND dimension IN ('brain_regions', 'modalities', 'species', 'tasks')
            GROUP BY source, dimension
        )
        SELECT
            t.source,
            t.n_total,
            MAX(CASE WHEN c.dimension = 'brain_regions' THEN c.n_covered ELSE 0 END) AS regions_covered,
            ROUND(100.0 * MAX(CASE WHEN c.dimension = 'brain_regions' THEN c.n_covered ELSE 0 END) / t.n_total, 1) AS regions_pct,
            MAX(CASE WHEN c.dimension = 'modalities' THEN c.n_covered ELSE 0 END) AS modalities_covered,
            ROUND(100.0 * MAX(CASE WHEN c.dimension = 'modalities' THEN c.n_covered ELSE 0 END) / t.n_total, 1) AS modalities_pct,
            MAX(CASE WHEN c.dimension = 'species' THEN c.n_covered ELSE 0 END) AS species_covered,
            ROUND(100.0 * MAX(CASE WHEN c.dimension = 'species' THEN c.n_covered ELSE 0 END) / t.n_total, 1) AS species_pct,
            MAX(CASE WHEN c.dimension = 'tasks' THEN c.n_covered ELSE 0 END) AS tasks_covered,
            ROUND(100.0 * MAX(CASE WHEN c.dimension = 'tasks' THEN c.n_covered ELSE 0 END) / t.n_total, 1) AS tasks_pct
        FROM totals t
        LEFT JOIN covered c ON c.source = t.source
        GROUP BY t.source, t.n_total
        ORDER BY t.n_total DESC
        """
        return self._conn.sql(sql)

    def atlas_coverage(self, atlas: str = "allen_ccf_mouse_id") -> duckdb.DuckDBPyRelation:
        """Coverage per atlas structure ID."""
        sql = f"""
        SELECT
            o.{atlas},
            o.id    AS canonical_id,
            o.label,
            COUNT(DISTINCT ce.dataset_id) AS n_datasets
        FROM ontology_regions o
        LEFT JOIN coverage_entries ce
            ON ce.value_id = o.id AND ce.dimension = 'brain_regions'
        WHERE o.{atlas} IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 4 DESC
        """
        return self._conn.sql(sql)

    def dark_pairs(
        self,
        dim_a: str = "brain_regions",
        dim_b: str = "modalities",
        *,
        top_n: int = 30,
        min_confidence: float = 0.65,
    ) -> duckdb.DuckDBPyRelation:
        """Highest-opportunity (dim_a × dim_b) pairs with zero observed datasets."""
        dim_a = self._require_dim(dim_a)
        dim_b = self._require_dim(dim_b)
        sql = f"""
        WITH a_counts AS (
            SELECT value_id, COUNT(DISTINCT dataset_id) AS n
            FROM coverage_entries
            WHERE dimension = '{dim_a}' AND confidence >= {min_confidence}
            GROUP BY value_id
        ),
        b_counts AS (
            SELECT value_id, COUNT(DISTINCT dataset_id) AS n
            FROM coverage_entries
            WHERE dimension = '{dim_b}' AND confidence >= {min_confidence}
            GROUP BY value_id
        ),
        observed AS (
            SELECT ce_a.value_id AS a_val, ce_b.value_id AS b_val,
                   COUNT(DISTINCT ce_a.dataset_id) AS n_observed
            FROM coverage_entries ce_a
            JOIN coverage_entries ce_b ON ce_a.dataset_id = ce_b.dataset_id
            WHERE ce_a.dimension = '{dim_a}' AND ce_b.dimension = '{dim_b}'
              AND ce_a.confidence >= {min_confidence} AND ce_b.confidence >= {min_confidence}
            GROUP BY 1, 2
        )
        SELECT
            a.value_id AS {dim_a.rstrip('s')},
            b.value_id AS {dim_b.rstrip('s')},
            COALESCE(o.n_observed, 0) AS n_observed,
            a.n AS a_marginal,
            b.n AS b_marginal,
            a.n + b.n AS opportunity_score
        FROM a_counts a
        CROSS JOIN b_counts b
        LEFT JOIN observed o ON o.a_val = a.value_id AND o.b_val = b.value_id
        WHERE COALESCE(o.n_observed, 0) = 0
        ORDER BY opportunity_score DESC
        LIMIT {top_n}
        """
        return self._conn.sql(sql)

    def coverage_summary(self, *, min_confidence: float = 0.65) -> dict[str, Any]:
        """High-level coverage summary dict."""
        total = self._conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        if not total:
            return {"total_datasets": 0}

        dim_sql = f"""
        SELECT dimension,
               COUNT(DISTINCT dataset_id) AS datasets_with_dim,
               ROUND(100.0 * COUNT(DISTINCT dataset_id) / {total}, 1) AS pct
        FROM coverage_entries
        WHERE confidence >= {min_confidence}
          AND dimension IN ('brain_regions','modalities','species','tasks','recording_scales')
        GROUP BY dimension
        ORDER BY dimension
        """
        rows = self._conn.execute(dim_sql).fetchall()
        return {
            "total_datasets": total,
            "total_entries": self._conn.execute(
                "SELECT COUNT(*) FROM coverage_entries"
            ).fetchone()[0],
            "dimension_coverage": {
                row[0]: {"datasets": row[1], "pct": row[2]} for row in rows
            },
        }

    def export_parquet(self, out_dir: str | Path = "data/coverage") -> dict[str, Path]:
        """Export tables to Parquet for downstream data science workflows."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for table in ("coverage_entries", "datasets", "ontology_regions", "ontology_species"):
            p = out / f"{table}.parquet"
            self._conn.execute(f"COPY {table} TO '{p}' (FORMAT PARQUET)")
            paths[table] = p
        return paths

    def region_dataset_counts(self, *, min_confidence: float = 0.65) -> list[dict[str, Any]]:
        """All brain regions with their dataset counts. Used to color the Brain Atlas."""
        sql = f"""
        SELECT
            ce.value_id          AS region_id,
            MAX(o.label)         AS region_label,
            COUNT(DISTINCT ce.dataset_id) AS n_datasets
        FROM coverage_entries ce
        LEFT JOIN ontology_regions o ON o.id = ce.value_id
        WHERE ce.dimension = 'brain_regions'
          AND ce.confidence >= {min_confidence}
        GROUP BY ce.value_id
        ORDER BY n_datasets DESC
        """
        rows = self._conn.sql(sql).fetchall()
        return [
            {"region_id": r[0], "region_label": r[1] or r[0], "n_datasets": r[2]}
            for r in rows
        ]

    def datasets_for_region(
        self,
        region_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        min_confidence: float = 0.65,
    ) -> list[dict[str, Any]]:
        """Datasets tagged with a specific brain region, ordered by confidence then title."""
        sql = """
        SELECT
            d.dataset_id,
            d.source,
            d.title,
            d.access_tier,
            ce.confidence
        FROM coverage_entries ce
        JOIN datasets d ON d.dataset_id = ce.dataset_id
        WHERE ce.dimension = 'brain_regions'
          AND ce.value_id = ?
          AND ce.confidence >= ?
        ORDER BY ce.confidence DESC, d.title
        LIMIT ? OFFSET ?
        """
        rows = self._conn.sql(
            sql, params=[region_id, min_confidence, limit, offset]
        ).fetchall()
        return [
            {
                "dataset_id": r[0],
                "source": r[1],
                "title": r[2],
                "access_tier": r[3],
                "confidence": round(float(r[4]), 3),
            }
            for r in rows
        ]

    def get_uncovered_datasets(self, dimension: str = "tasks") -> list[tuple[str, str]]:
        """Return (dataset_id, title) pairs that have no coverage entry for dimension."""
        return self._conn.execute(
            """
            SELECT d.dataset_id, d.title
            FROM datasets d
            WHERE d.dataset_id NOT IN (
                SELECT DISTINCT dataset_id FROM coverage_entries WHERE dimension = ?
            )
            ORDER BY d.dataset_id
            """,
            [dimension],
        ).fetchall()

    def add_coverage_entries_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> int:
        """Bulk-insert coverage entries, skipping duplicates.

        Each entry dict must have: dataset_id, dimension, value_id.
        Optional: confidence (default 0.5), provenance (default 'nlp_enrichment').
        Returns number of rows actually inserted.
        """
        if not entries:
            return 0
        rows = []
        for e in entries:
            did = e["dataset_id"]
            dim = e["dimension"]
            vid = e["value_id"]
            conf = float(e.get("confidence", 0.5))
            prov = e.get("provenance", "nlp_enrichment")
            entry_id = f"coverage:nlp:{did}:{dim}:{vid}"
            rows.append((
                entry_id, did, "nlp_enrichment", dim, vid, vid,
                conf, "silver", "open", "dataset",
                prov, prov, "nlp_enrichment_v1",
                None, None, None,
            ))
        before = self._conn.execute(
            "SELECT COUNT(*) FROM coverage_entries"
        ).fetchone()[0]
        self._conn.executemany(
            """INSERT OR IGNORE INTO coverage_entries
               (entry_id, dataset_id, source, dimension, value_id, label,
                confidence, evidence_tier, access_tier, analysis_level,
                source_field, evidence_text, snapshot_id,
                uberon_id, allen_ccf_mouse_id, ncbitaxon_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        after = self._conn.execute(
            "SELECT COUNT(*) FROM coverage_entries"
        ).fetchone()[0]
        return after - before

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        return self._conn.sql(query)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _dataset_id(record: dict[str, Any]) -> str:
    return str(
        record.get("dataset_id")
        or f"dataset:{record.get('source')}:{record.get('source_id')}"
    )
