"""Coverage rarity boost — upweights datasets covering underrepresented dimensions.

When a query asks for brain regions, modalities, or species that are sparsely
covered in the corpus, any matching result should be scored higher because it
is genuinely hard to find. This is an inverse-frequency signal: rare → valuable.

Usage
-----
    booster = CoverageGapBooster.from_db("data/coverage/ledger.duckdb")
    boost = booster.score(
        region_ids={"barrel_cortex"},
        modality_ids={"fmri"},
        species_ids={"mus_musculus"},
    )
    # Returns float in [0.0, MAX_BOOST]

The boost is additive and capped at MAX_BOOST (0.10) so it cannot dominate
the existing ontology/semantic scores.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_BOOST = 0.10  # additive ceiling — never overwhelms ontology/semantic scores
_MIN_DATASETS = 1  # floor to avoid division by zero
DEFAULT_DB_PATH = Path("data/coverage/ledger.duckdb")


class CoverageGapBooster:
    """Computes a rarity-based coverage boost for retrieval scoring.

    Rarity formula (per value_id):
        rarity(v) = 1 - log(n_v + 1) / log(total + 1)

    where n_v is the number of datasets covering value v.
    Ranges from ~0 (common, e.g. fMRI with 2,473 datasets) to ~1 (never seen).
    The boost is the average rarity of matched dimension values, scaled to MAX_BOOST.
    """

    def __init__(
        self,
        region_counts: dict[str, int],
        modality_counts: dict[str, int],
        species_counts: dict[str, int],
        total_datasets: int,
    ) -> None:
        self._region_counts = region_counts
        self._modality_counts = modality_counts
        self._species_counts = species_counts
        self._total = max(total_datasets, _MIN_DATASETS)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_db(cls, db_path: str | Path = DEFAULT_DB_PATH) -> CoverageGapBooster:
        """Load counts from the live DuckDB coverage ledger."""
        import duckdb

        path = Path(db_path)
        if not path.exists():
            log.warning("Coverage ledger not found at %s — gap boost disabled", path)
            return cls._empty()

        conn = duckdb.connect(str(path), read_only=True)
        try:
            total = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
            region_counts = _fetch_counts(conn, "brain_regions")
            modality_counts = _fetch_counts(conn, "modalities")
            species_counts = _fetch_counts(conn, "species")
        finally:
            conn.close()

        log.info(
            "CoverageGapBooster loaded: %d regions, %d modalities, %d species, %d total datasets",
            len(region_counts), len(modality_counts), len(species_counts), total,
        )
        return cls(region_counts, modality_counts, species_counts, total)

    @classmethod
    def _empty(cls) -> CoverageGapBooster:
        inst = cls({}, {}, {}, 0)
        inst._disabled = True
        return inst

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        region_ids: set[str] | None = None,
        modality_ids: set[str] | None = None,
        species_ids: set[str] | None = None,
    ) -> float:
        """Return additive coverage rarity boost in [0.0, MAX_BOOST].

        Only matched dimensions contribute — passing empty sets returns 0.0.
        """
        rarities: list[float] = []

        if region_ids:
            rarities.extend(
                self._rarity(v, self._region_counts) for v in region_ids
            )
        if modality_ids:
            rarities.extend(
                self._rarity(v, self._modality_counts) for v in modality_ids
            )
        if species_ids:
            rarities.extend(
                self._rarity(v, self._species_counts) for v in species_ids
            )

        if not rarities or getattr(self, "_disabled", False):
            return 0.0
        avg_rarity = sum(rarities) / len(rarities)
        return round(min(avg_rarity * MAX_BOOST, MAX_BOOST), 4)

    def rarity(self, dimension: str, value_id: str) -> float:
        """Return rarity score [0, 1] for a single value in a given dimension."""
        counts = {
            "brain_regions": self._region_counts,
            "modalities": self._modality_counts,
            "species": self._species_counts,
        }.get(dimension, {})
        return self._rarity(value_id, counts)

    def coverage_stats(self) -> dict[str, Any]:
        """Summary of loaded coverage counts for debugging."""
        return {
            "total_datasets": self._total,
            "regions_tracked": len(self._region_counts),
            "modalities_tracked": len(self._modality_counts),
            "species_tracked": len(self._species_counts),
            "rarest_regions": _top_rarest(self._region_counts, self._total, n=10),
            "rarest_modalities": _top_rarest(self._modality_counts, self._total, n=5),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rarity(self, value_id: str, counts: dict[str, int]) -> float:
        n = counts.get(value_id.casefold().replace("-", "_"), 0)
        return 1.0 - math.log(n + 1) / math.log(self._total + 1)


def _fetch_counts(conn: Any, dimension: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT value_id, COUNT(DISTINCT dataset_id) AS n
        FROM coverage_entries
        WHERE dimension = ? AND confidence >= 0.65
        GROUP BY value_id
        """,
        [dimension],
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _top_rarest(counts: dict[str, int], total: int, n: int = 10) -> list[dict[str, Any]]:
    log_total = math.log(total + 1)
    scored = [
        {"value_id": k, "n_datasets": v,
         "rarity": round(1.0 - math.log(v + 1) / log_total, 3)}
        for k, v in counts.items()
        if v < 10  # only show genuinely rare ones
    ]
    scored.sort(key=lambda x: x["rarity"], reverse=True)
    return scored[:n]


@lru_cache(maxsize=1)
def _global_booster() -> CoverageGapBooster:
    """Module-level singleton — loaded once, shared across requests."""
    return CoverageGapBooster.from_db()


def get_global_booster() -> CoverageGapBooster:
    return _global_booster()
