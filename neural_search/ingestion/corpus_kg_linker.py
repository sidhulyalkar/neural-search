"""Link corpus records to KG concept nodes via label/alias matching.

Reads the corpus JSONL and the DuckDB KG store, then produces a
``dataset_concept_index.jsonl`` file mapping each dataset_id to the
concept node IDs it matches.  Also exposes fast in-memory helpers for
use during search scoring.

Usage (CLI)::

    python neural_search/ingestion/corpus_kg_linker.py
    python neural_search/ingestion/corpus_kg_linker.py \\
        --corpus  data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl \\
        --db      data/kg/neural_search_kg.duckdb \\
        --output  data/kg/dataset_concept_index.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import duckdb

log = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path(
    "data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl"
)
DEFAULT_DB_PATH = Path("data/kg/neural_search_kg.duckdb")
DEFAULT_INDEX_PATH = Path("data/kg/dataset_concept_index.jsonl")

LINK_CONFIDENCE = 0.7


def extract_str_list(items: Any) -> list[str]:
    """Extract string values from a field that may be ``str | dict`` items."""
    result: list[str] = []
    for item in items or []:
        if isinstance(item, str):
            result.append(item.strip())
        elif isinstance(item, dict):
            v = item.get("label") or item.get("id") or item.get("name") or ""
            if v:
                result.append(str(v).strip())
    return [x for x in result if x]


def _norm(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _tokenset(value: str) -> set[str]:
    return set(_norm(value).split())


def _load_concept_lookup(
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, list[str]]]:
    rows = conn.execute(
        "SELECT node_id, label, properties FROM kg_nodes WHERE node_type = 'concept'"
    ).fetchall()

    lookup: list[tuple[str, list[str]]] = []
    for node_id, label, props_raw in rows:
        terms: list[str] = [_norm(str(label))]
        try:
            props = json.loads(props_raw) if props_raw else {}
            aliases_top = props.get("aliases", []) or []
            aliases_nested = (props.get("properties") or {}).get("aliases", []) or []
            for alias in list(aliases_top) + list(aliases_nested):
                if isinstance(alias, str) and alias.strip():
                    terms.append(_norm(alias))
        except Exception:
            pass
        lookup.append((node_id, list(dict.fromkeys(terms))))

    log.info("Loaded %d concept nodes for matching", len(lookup))
    return lookup


def _record_terms(record: dict[str, Any]) -> list[str]:
    raw: list[str] = []
    for field in ("modalities", "tasks", "species", "brain_regions", "behaviors"):
        raw.extend(extract_str_list(record.get(field)))
    for item in record.get("behavioral_events") or []:
        if isinstance(item, dict):
            lbl = item.get("label") or item.get("id") or ""
            if lbl:
                raw.append(str(lbl).strip())
    return raw


def _match_concept(record_terms: list[str], concept_terms: list[str]) -> bool:
    norm_record = [_norm(t) for t in record_terms]
    for ct in concept_terms:
        ct_tokens = _tokenset(ct)
        for rt in norm_record:
            if ct == rt:
                return True
            rt_tokens = set(rt.split())
            if ct_tokens and ct_tokens <= rt_tokens:
                return True
            if rt_tokens and rt_tokens <= ct_tokens:
                return True
    return False


def load_dataset_concept_index(
    path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, list[str]]:
    """Return ``{dataset_id: [concept_id, ...]}`` from the on-disk index file."""
    path = Path(path)
    if not path.exists():
        log.warning("Dataset concept index not found at %s", path)
        return {}
    index: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                dataset_id = rec.get("dataset_id", "")
                concept_ids = rec.get("concept_ids", [])
                if dataset_id:
                    index[dataset_id] = concept_ids
            except json.JSONDecodeError:
                pass
    return index


def concept_expansion_score(
    dataset_id: str,
    query_concepts: list[str],
    index: dict[str, list[str]],
) -> float:
    """Return [0, 1] overlap score between query concepts and dataset's concept set."""
    if not query_concepts:
        return 0.0

    dataset_concepts = index.get(dataset_id, [])
    if not dataset_concepts:
        bare = dataset_id.removeprefix("dataset:")
        dataset_concepts = index.get(bare, [])
    if not dataset_concepts:
        return 0.0

    def _slug_norm(s: str) -> str:
        return s.casefold().replace("concept:", "").replace("-", "_").replace(" ", "_")

    ds_slugs = {_slug_norm(c) for c in dataset_concepts}
    q_slugs = [_slug_norm(c) for c in query_concepts]
    matched = sum(1 for q in q_slugs if q in ds_slugs)
    return round(matched / max(len(q_slugs), 1), 4)


_FIELD_SLUG_MAP = {
    "brain_regions": "region",
    "modalities": "modality",
    "tasks": "task",
    "species": "species",
    "behaviors": "behavior",
}


def _field_slugs(record: dict[str, Any]) -> list[str]:
    """Return vocab slugs from corpus metadata fields (e.g. 'region:hippocampus').

    These slugs are stored alongside KG concept IDs so concept_overlap_score
    can match them against query expected_regions_any / expected_modalities_any
    slugs (which share the same normalization).
    """
    slugs: list[str] = []
    for field, prefix in _FIELD_SLUG_MAP.items():
        for raw in extract_str_list(record.get(field)):
            slug = raw.strip().casefold().replace(" ", "_").replace("-", "_").replace("/", "_")
            if slug:
                slugs.append(f"{prefix}:{slug}")

    # behavioral_events is a list of dicts with "id" or "label"
    for item in (record.get("behavioral_events") or []):
        lbl = (item.get("label") or item.get("id") or "") if isinstance(item, dict) else str(item)
        slug = lbl.strip().casefold().replace(" ", "_").replace("-", "_").replace("/", "_")
        if slug:
            slugs.append(f"behavior:{slug}")

    return list(dict.fromkeys(slugs))  # deduplicate, preserve order


def build_dataset_concept_index(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    db_path: Path | None = DEFAULT_DB_PATH,
    output_path: Path = DEFAULT_INDEX_PATH,
) -> dict[str, list[str]]:
    """Build and save the dataset→concept index.

    Combines two sources:
    - **KG concept nodes** from DuckDB (abstract theoretical concepts like
      "hebbian_learning", "predictive_coding") — skipped if db_path is None or missing.
    - **Vocabulary field slugs** from corpus metadata (concrete terms like
      "region:hippocampus", "modality:electrophysiology", "species:mouse") — always
      included, gives high coverage (~70%+) so concept_overlap_score can fire on
      nearly all relevant queries.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    concept_lookup: list[tuple[str, list[str]]] = []
    if db_path and Path(db_path).exists():
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            concept_lookup = _load_concept_lookup(conn)
        finally:
            conn.close()
    else:
        log.warning("DuckDB not available — skipping KG concept matching")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    index: dict[str, list[str]] = {}
    total_records = 0
    total_links = 0

    with corpus_path.open(encoding="utf-8") as fh, output_path.open(
        "w", encoding="utf-8"
    ) as out_fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record: dict[str, Any] = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            total_records += 1
            dataset_id = record.get("dataset_id", "")
            if not dataset_id:
                # Fall back to source:source_id (stable_id format) for dataset-like records
                # that lack an explicit dataset_id (e.g. buzsaki, crcns, ibl sources).
                src = record.get("source", "")
                sid = str(record.get("source_id") or "")
                if src and sid and (record.get("brain_regions") or record.get("modalities") or record.get("tasks")):
                    dataset_id = f"{src}:{sid}"
            if not dataset_id:
                continue

            # KG concept matches (abstract theoretical concepts)
            record_terms = _record_terms(record)
            kg_concepts: list[str] = []
            if concept_lookup and record_terms:
                kg_concepts = [
                    concept_id
                    for concept_id, concept_terms in concept_lookup
                    if _match_concept(record_terms, concept_terms)
                ]

            # Vocabulary field slugs (concrete metadata terms)
            vocab_slugs = _field_slugs(record)

            all_concepts = list(dict.fromkeys(kg_concepts + vocab_slugs))

            if all_concepts:
                total_links += len(all_concepts)
                index[dataset_id] = all_concepts
                out_fh.write(
                    json.dumps(
                        {
                            "dataset_id": dataset_id,
                            "concept_ids": all_concepts,
                            "confidence": LINK_CONFIDENCE,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    log.info(
        "Done. %d records -> %d datasets linked -> %d concept edges",
        total_records,
        len(index),
        total_links,
    )
    print(
        f"Dataset concept index: {len(index):,} datasets linked, "
        f"{total_links:,} concept edges -> {output_path}"
    )
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Link corpus records to KG concept nodes.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    build_dataset_concept_index(
        corpus_path=args.corpus,
        db_path=args.db,
        output_path=args.output,
    )
