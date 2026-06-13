#!/usr/bin/env python3
"""Verify a dataset is suitable for starter notebook development.

Checks:
  1. Dataset exists in normalized corpus
  2. Has source archive URL
  3. Has usable modality and species metadata
  4. Has at least one documented loading route (NWB, BIDS, or direct URL)
  5. Passes a minimum metadata quality threshold

Usage:
    python scripts/eval/verify_demo_dataset.py dandi:000039
    python scripts/eval/verify_demo_dataset.py --list-candidates
    python scripts/eval/verify_demo_dataset.py --min-score 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
PASS_SCORE = 0.6   # minimum quality score to pass
MAX_CANDIDATES = 20  # candidates to show in --list-candidates mode


def _load_corpus(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_key(rec: dict) -> str:
    return f"{rec.get('source', '')}:{rec.get('source_id', '')}"


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------

def check_corpus_presence(rec: dict | None, dataset_id: str) -> tuple[bool, str]:
    if rec is None:
        return False, f"Dataset '{dataset_id}' not found in corpus."
    return True, f"Found: {rec.get('title', '(no title)')[:80]}"


def check_source_url(rec: dict) -> tuple[bool, str]:
    url = rec.get("url") or ""
    if not url:
        return False, "No source archive URL."
    return True, f"URL: {url}"


def check_modality_metadata(rec: dict) -> tuple[bool, str]:
    mods = rec.get("modalities") or []
    if not mods:
        return False, "No modality metadata."
    return True, f"Modalities: {', '.join(str(m) for m in mods)}"


def check_species_metadata(rec: dict) -> tuple[bool, str]:
    species = rec.get("species") or []
    if not species:
        return False, "No species metadata."
    return True, f"Species: {', '.join(str(s) for s in species)}"


def check_loading_route(rec: dict) -> tuple[bool, str]:
    """Check if the dataset has a documented loading route."""
    standards = rec.get("data_standards") or []
    url = rec.get("url") or ""
    source = rec.get("source") or ""

    if "NWB" in standards:
        return True, "Loading route: NWB (via pynwb or DANDI client)"
    if "BIDS" in standards:
        return True, "Loading route: BIDS (via pybids)"
    if source in ("dandi", "openneuro", "zenodo", "osf", "neurovault", "allen"):
        if url:
            return True, f"Loading route: direct from {source} archive at {url}"
    return False, "No documented loading route (no NWB/BIDS standard and no recognized archive URL)"


def check_notebook_suitability(rec: dict) -> tuple[bool, str]:
    """Score overall notebook suitability based on metadata richness."""
    score = 0.0
    notes = []

    # Title quality
    title = rec.get("title") or ""
    if len(title) > 20:
        score += 0.15
        notes.append("+title")

    # Description
    desc = rec.get("description") or ""
    if desc and len(desc) > 50:
        score += 0.15
        notes.append("+description")

    # Modalities and species
    if rec.get("modalities"):
        score += 0.15
        notes.append("+modalities")
    if rec.get("species"):
        score += 0.15
        notes.append("+species")

    # Data format
    standards = rec.get("data_standards") or []
    if "NWB" in standards:
        score += 0.20
        notes.append("+NWB")
    elif "BIDS" in standards:
        score += 0.15
        notes.append("+BIDS")

    # Affordances
    if rec.get("has_raw_data"):
        score += 0.10
        notes.append("+raw_data")
    if rec.get("has_processed_data"):
        score += 0.05
        notes.append("+processed")
    if rec.get("has_trials") or rec.get("has_behavior"):
        score += 0.05
        notes.append("+behavioral")

    passed = score >= PASS_SCORE
    status = "PASS" if passed else "FAIL"
    return passed, f"Notebook suitability score: {score:.2f} [{status}]  ({', '.join(notes)})"


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

def verify_dataset(dataset_id: str, corpus: list[dict]) -> dict:
    corpus_map = {_record_key(r): r for r in corpus}
    rec = corpus_map.get(dataset_id)

    checks = []
    all_pass = True

    check_fns = [
        ("corpus_presence", lambda: check_corpus_presence(rec, dataset_id)),
    ]

    if rec:
        check_fns += [
            ("source_url", lambda: check_source_url(rec)),
            ("modality_metadata", lambda: check_modality_metadata(rec)),
            ("species_metadata", lambda: check_species_metadata(rec)),
            ("loading_route", lambda: check_loading_route(rec)),
            ("notebook_suitability", lambda: check_notebook_suitability(rec)),
        ]

    for name, fn in check_fns:
        passed, msg = fn()
        checks.append({"check": name, "passed": passed, "message": msg})
        if not passed:
            all_pass = False

    return {
        "dataset_id": dataset_id,
        "title": rec.get("title") if rec else None,
        "url": rec.get("url") if rec else None,
        "source": rec.get("source") if rec else None,
        "overall": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "record": rec,
    }


def list_candidates(corpus: list[dict], min_score: float = PASS_SCORE, limit: int = MAX_CANDIDATES) -> list[dict]:
    """Find datasets that pass the notebook suitability threshold."""
    candidates = []
    for rec in corpus:
        passed_suit, _ = check_notebook_suitability(rec)
        if not passed_suit:
            continue
        passed_url, _ = check_source_url(rec)
        if not passed_url:
            continue
        passed_route, _ = check_loading_route(rec)
        if not passed_route:
            continue
        candidates.append({
            "dataset_id": _record_key(rec),
            "title": (rec.get("title") or "")[:70],
            "url": rec.get("url", ""),
            "modalities": rec.get("modalities") or [],
            "species": rec.get("species") or [],
            "data_standards": rec.get("data_standards") or [],
        })
        if len(candidates) >= limit:
            break
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a dataset for starter notebook development.")
    parser.add_argument("dataset_id", nargs="?", help="Dataset ID to verify (e.g. dandi:000039)")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--list-candidates", action="store_true", help="List all notebook-ready datasets")
    parser.add_argument("--min-score", type=float, default=PASS_SCORE)
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    corpus = _load_corpus(args.corpus)

    if args.list_candidates:
        candidates = list_candidates(corpus, min_score=args.min_score)
        if args.json:
            print(json.dumps(candidates, indent=2))
        else:
            print(f"Found {len(candidates)} notebook-ready datasets:\n")
            for c in candidates:
                mods = ", ".join(c["modalities"][:3])
                specs = ", ".join(c["species"][:2])
                stds = ", ".join(c["data_standards"][:2])
                print(f"  {c['dataset_id']:<25}  {c['title'][:50]}")
                print(f"    modalities={mods}  species={specs}  standards={stds}")
                print(f"    URL: {c['url']}")
                print()
        return

    if not args.dataset_id:
        parser.error("Provide a dataset_id or --list-candidates")

    result = verify_dataset(args.dataset_id, corpus)

    if args.json:
        out = {k: v for k, v in result.items() if k != "record"}
        print(json.dumps(out, indent=2))
    else:
        status_sym = "✓" if result["overall"] == "PASS" else "✗"
        print(f"\n{'='*60}")
        print(f" {status_sym} {result['dataset_id']}  [{result['overall']}]")
        print(f"{'='*60}")
        if result["title"]:
            print(f"  Title: {result['title']}")
        if result["url"]:
            print(f"  URL:   {result['url']}")
        print()
        for c in result["checks"]:
            sym = "✓" if c["passed"] else "✗"
            print(f"  [{sym}] {c['check']:<25}  {c['message']}")
        print()
        if result["overall"] == "PASS":
            print("  VERDICT: Dataset is suitable for starter notebook development.")
        else:
            print("  VERDICT: Dataset does not meet requirements for starter notebook.")

    sys.exit(0 if result["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
