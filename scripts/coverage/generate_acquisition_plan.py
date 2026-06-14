"""Generate a prioritized corpus acquisition plan from the DuckDB coverage ledger.

Outputs a ranked list of what to acquire next, ordered by expected coverage impact.
Each entry carries: target type, description, estimated n_new_datasets, priority,
and the specific gap it closes.

Usage
-----
    python scripts/coverage/generate_acquisition_plan.py
    python scripts/coverage/generate_acquisition_plan.py --output data/reports/coverage/acquisition_plan.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.coverage.duckdb_store import CoverageStore

DB_PATH = ROOT / "data" / "coverage" / "ledger.duckdb"
REPORTS_DIR = ROOT / "data" / "reports" / "coverage"


def _priority(score: float) -> str:
    if score >= 0.75:
        return "P0"
    if score >= 0.50:
        return "P1"
    if score >= 0.25:
        return "P2"
    return "P3"


def _impact_score(marginal_a: int, marginal_b: int, total: int) -> float:
    """Normalised opportunity score in [0, 1]."""
    raw = marginal_a + marginal_b
    return min(raw / max(total * 0.1, 1), 1.0)


def _rarity(n: int, total: int) -> float:
    return 1.0 - math.log(n + 1) / math.log(total + 1)


def generate_plan(db_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    with CoverageStore(db_path) as store:
        summary = store.coverage_summary()
        total = summary["total_datasets"]

        # ── 1. Zero-coverage ontology regions ────────────────────────────────
        uncovered = store.uncovered_regions().fetchall()
        for row in uncovered:
            rid, label, uberon_id, ccf_id, parents_json = row
            parents = json.loads(parents_json or "[]")
            # Estimate impact by how connected the region is (many parents = well-studied system)
            connectedness = len(parents)
            impact = 0.60 + min(connectedness * 0.05, 0.35)
            items.append({
                "type": "missing_region",
                "id": rid,
                "label": label,
                "uberon_id": uberon_id,
                "allen_ccf_mouse_id": ccf_id,
                "description": f"Zero datasets cover {label}. Target repositories with explicit {label} recording metadata.",
                "recommended_sources": _recommend_sources_for_region(rid),
                "estimated_new_datasets": _estimate_region_yield(rid),
                "impact_score": round(impact, 3),
                "priority": _priority(impact),
                "gap_type": "uncovered_region",
            })

        # ── 2. Low-coverage sources ───────────────────────────────────────────
        source_rows = store.source_coverage_rates().fetchall()
        for row in source_rows:
            src, n_total, regions_cov, regions_pct = row[0], row[1], row[2], row[3]
            if n_total < 10:
                continue
            if regions_pct < 30.0:
                deficit = 100.0 - regions_pct
                impact = deficit / 100.0 * min(n_total / 500, 1.0)
                items.append({
                    "type": "low_coverage_source",
                    "source": src,
                    "current_datasets": n_total,
                    "current_region_coverage_pct": regions_pct,
                    "description": (
                        f"{src} has {n_total:,} datasets but only {regions_pct:.0f}% "
                        f"have brain region labels. Structured metadata enrichment or "
                        f"additional adapter fields could unlock {int(n_total * (1 - regions_pct/100)):,} more."
                    ),
                    "recommended_action": _recommend_action_for_source(src, regions_pct),
                    "estimated_new_datasets": int(n_total * (100 - regions_pct) / 100 * 0.4),
                    "impact_score": round(impact, 3),
                    "priority": _priority(impact),
                    "gap_type": "low_coverage_source",
                })

        # ── 3. High-opportunity dark region × modality pairs ─────────────────
        dark = store.dark_pairs("brain_regions", "modalities", top_n=20).fetchall()
        for row in dark:
            region, modality, n_obs, a_marg, b_marg, opp = row
            if n_obs > 0:
                continue
            impact = _impact_score(a_marg, b_marg, total)
            items.append({
                "type": "dark_pair",
                "region": region,
                "modality": modality,
                "region_n_datasets": a_marg,
                "modality_n_datasets": b_marg,
                "n_observed": n_obs,
                "description": (
                    f"No datasets combine {region} with {modality}. "
                    f"{a_marg} {region} datasets and {b_marg} {modality} datasets exist separately — "
                    f"cross-species or protocol bridging could close this gap."
                ),
                "recommended_action": f"Search {_dark_pair_sources(region, modality)} for {region}+{modality} co-recorded datasets.",
                "estimated_new_datasets": max(1, min(a_marg, b_marg) // 5),
                "impact_score": round(impact, 3),
                "priority": _priority(impact),
                "gap_type": "dark_region_modality_pair",
            })

        # ── 4. Species gaps ───────────────────────────────────────────────────
        species_rows = store.sql(
            """
            SELECT value_id, COUNT(DISTINCT dataset_id) AS n
            FROM coverage_entries
            WHERE dimension = 'species' AND confidence >= 0.65
            GROUP BY value_id ORDER BY n DESC
            """
        ).fetchall()
        species_counts = {r[0]: r[1] for r in species_rows}
        underrep_species = [
            (sp, n) for sp, n in species_counts.items()
            if n < 50 and sp not in ("unknown", "unspecified")
        ]
        for sp, n in underrep_species[:5]:
            rarity = _rarity(n, total)
            items.append({
                "type": "underrepresented_species",
                "species": sp,
                "current_datasets": n,
                "description": (
                    f"{sp} has only {n} datasets. "
                    f"Targeted crawl of species-specific repositories could expand coverage."
                ),
                "recommended_sources": _recommend_sources_for_species(sp),
                "estimated_new_datasets": max(10, n * 2),
                "impact_score": round(rarity * 0.6, 3),
                "priority": _priority(rarity * 0.6),
                "gap_type": "underrepresented_species",
            })

    items.sort(key=lambda x: x["impact_score"], reverse=True)
    return items


def _recommend_sources_for_region(region_id: str) -> list[str]:
    region_source_hints: dict[str, list[str]] = {
        "dlPFC": ["openneuro", "harvard_dataverse", "osf"],
        "ACC": ["openneuro", "gin", "dandi"],
        "OFC": ["dandi", "crcns", "gin"],
        "mPFC": ["dandi", "crcns", "gin"],
        "lateral_hypothalamus": ["dandi", "crcns"],
        "median_raphe": ["dandi", "gin"],
        "red_nucleus": ["gin", "zenodo"],
        "thalamic_reticular_nucleus": ["dandi", "gin"],
        "ventral_posterolateral_thalamus": ["dandi", "openneuro"],
        "wernicke_area": ["openneuro", "osf"],
        "lumbar_spinal_cord": ["osf", "zenodo", "figshare"],
    }
    return region_source_hints.get(region_id, ["dandi", "openneuro", "gin"])


def _estimate_region_yield(region_id: str) -> int:
    high_yield = {"ACC", "OFC", "mPFC", "dlPFC", "lateral_hypothalamus"}
    medium_yield = {"median_raphe", "red_nucleus", "thalamic_reticular_nucleus"}
    if region_id in high_yield:
        return 50
    if region_id in medium_yield:
        return 20
    return 10


def _recommend_action_for_source(source: str, coverage_pct: float) -> str:
    actions = {
        "brain_image_library": (
            "BIL datasets use a structured JSON manifest. "
            "Parse `general/subject/species` and `general/technique` fields to extract "
            "brain_regions and modalities. Expected yield: ~180 additional structured records."
        ),
        "openneuro": (
            "OpenNeuro BIDS datasets include `participants.tsv` (species) and "
            "electrode location files. Run NWB/BIDS electrode extractor on the "
            f"{int(300 * (1 - coverage_pct / 100))} uncovered datasets."
        ),
        "gin": (
            "GIN repositories often link to publications. "
            "Run CrossRef DOI resolver on linked papers to extract region mentions. "
            "Estimate +100-150 region labels."
        ),
        "osf": (
            "OSF projects frequently include README files with experimental details. "
            "LLM inference on README/description fields. Estimate +60-80 region labels."
        ),
        "neurovault": (
            "NeuroVault whole-brain fMRI maps rarely specify recording regions — "
            "by design. Region coverage here represents ROI analyses only. "
            "Parse `contrast_definition` field for region mentions instead."
        ),
    }
    return actions.get(source, f"Enrich {source} metadata via structured field extraction or LLM inference.")


def _dark_pair_sources(region: str, modality: str) -> str:
    modality_sources = {
        "fmri": "OpenNeuro, NeuroVault",
        "calcium_imaging": "DANDI, GIN",
        "extracellular_ephys": "DANDI, CRCNS, IBL",
        "eeg": "OpenNeuro, DANDI",
        "two_photon": "DANDI, GIN",
    }
    return modality_sources.get(modality, "DANDI, OpenNeuro")


def _recommend_sources_for_species(species: str) -> list[str]:
    hints = {
        "macaca_mulatta": ["dandi", "crcns", "gin", "openneuro"],
        "macaque": ["dandi", "crcns", "gin", "openneuro"],
        "callithrix_jacchus": ["dandi", "gin", "zenodo"],
        "marmoset": ["dandi", "gin", "zenodo"],
        "danio_rerio": ["zenodo", "osf", "figshare"],
        "drosophila_melanogaster": ["dandi", "zenodo", "figshare"],
        "caenorhabditis_elegans": ["zenodo", "osf"],
    }
    return hints.get(species, ["dandi", "zenodo", "figshare"])


def render_markdown(items: list[dict[str, Any]], total: int) -> str:
    lines = [
        "# Corpus Acquisition Plan",
        "",
        f"Generated from DuckDB coverage ledger ({total:,} datasets).",
        "Items ranked by estimated coverage impact.",
        "",
        f"**Total action items:** {len(items)}",
        "",
    ]

    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_type.setdefault(item["gap_type"], []).append(item)

    section_titles = {
        "missing_region": "## Uncovered Ontology Regions",
        "low_coverage_source": "## Low-Coverage Sources",
        "dark_region_modality_pair": "## Dark Region × Modality Pairs",
        "underrepresented_species": "## Underrepresented Species",
    }

    for gap_type, title in section_titles.items():
        group = by_type.get(gap_type, [])
        if not group:
            continue
        lines += [title, ""]
        for item in group:
            prio = item["priority"]
            lines.append(f"### [{prio}] {item.get('label') or item.get('source') or item.get('region', '') + ' × ' + item.get('modality', '') or item.get('species', '')}")
            lines.append(f"**Impact:** {item['impact_score']:.3f} | **Est. new datasets:** {item['estimated_new_datasets']:,}")
            lines.append("")
            lines.append(item["description"])
            if item.get("recommended_action"):
                lines.append(f"\n**Action:** {item['recommended_action']}")
            if item.get("recommended_sources"):
                lines.append(f"\n**Sources:** {', '.join(item['recommended_sources'])}")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "acquisition_plan.json")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"Ledger not found: {args.db}. Run build_duckdb_ledger.py first.", file=sys.stderr)
        return 1

    print(f"Generating acquisition plan from {args.db}…")
    items = generate_plan(args.db)

    with CoverageStore(args.db) as store:
        total = store.coverage_summary()["total_datasets"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2))
    print(f"JSON → {args.output} ({len(items)} items)")

    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_markdown(items, total))
    print(f"Markdown → {md_path}")

    print(f"\nTop 5 by impact:")
    for item in items[:5]:
        desc = item.get("label") or item.get("source") or f"{item.get('region')} × {item.get('modality')}"
        print(f"  [{item['priority']}] {desc} — impact={item['impact_score']:.3f}, est. {item['estimated_new_datasets']} datasets")

    return 0


if __name__ == "__main__":
    sys.exit(main())
