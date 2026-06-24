"""Paper-dataset link audit sampler.

Samples from artifacts/literature/paper_dataset_links.jsonl:
  - 50 DOI-exact matches
  - 50 title-fuzzy matches
  - 50 "not found" corpus records (false-negative audit)

Produces:
  - reports/eval/paper_link_audit_template.csv
  - reports/eval/paper_link_audit_instructions.md

Usage:
    python scripts/eval/sample_paper_links_for_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINKS_PATH = ROOT / "artifacts/literature/paper_dataset_links.jsonl"
CORPUS_PATH = ROOT / "data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl"
AUDIT_CSV = ROOT / "reports/eval/paper_link_audit_template.csv"
AUDIT_INSTRUCTIONS = ROOT / "reports/eval/paper_link_audit_instructions.md"

AUDIT_COLUMNS = [
    "link_type",           # doi_exact / title_fuzzy / not_found
    "dataset_record_id",
    "dataset_title",
    "paper_doi",
    "paper_title",
    "paper_year",
    "match_method",
    "confidence",
    "link_correct",        # TRUE / FALSE / UNCERTAIN
    "error_type",          # wrong_paper / no_relationship / ambiguous / none
    "notes",
]

INSTRUCTIONS_TEXT = """# Paper-Dataset Link Audit Instructions

## Purpose
Validate whether the paper-dataset linker correctly associates papers with datasets.
This audit determines whether linker precision is sufficient for whitepaper claims.

## Files
- Template: `reports/eval/paper_link_audit_template.csv`
- Links source: `artifacts/literature/paper_dataset_links.jsonl`

## Columns

| Column | Description | Values |
|---|---|---|
| `link_type` | How the link was found | `doi_exact`, `title_fuzzy`, `not_found` |
| `dataset_record_id` | Neural Search dataset ID | — |
| `dataset_title` | Dataset title | — |
| `paper_doi` | Paper DOI (if available) | — |
| `paper_title` | Paper title | — |
| `paper_year` | Publication year | — |
| `match_method` | Matching strategy used | — |
| `confidence` | Linker confidence score | — |
| `link_correct` | Your judgment | `TRUE` / `FALSE` / `UNCERTAIN` |
| `error_type` | If not TRUE | `wrong_paper`, `no_relationship`, `ambiguous`, `none` |
| `notes` | Free text | any |

## For "not_found" rows
These are dataset records where the linker found **no** paper link.
Assess whether you can find a paper for this dataset manually.
If yes, mark `link_correct=FALSE` (false negative) and add the paper in `notes`.

## Target Precision
- DOI-exact: expect ≥95% precision.
- Title-fuzzy: expect 60–85% precision.
- Not-found recall: measure how many have discoverable papers.
"""


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-each", type=int, default=50, help="Samples per method (default 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    links = load_jsonl(LINKS_PATH)
    if not links:
        print(f"✗ No links found at {LINKS_PATH}")
        return

    doi_links = [r for r in links if r.get("match_method") == "doi_exact"]
    fuzzy_links = [r for r in links if r.get("match_method") == "title_fuzzy_local"]
    not_found = [r for r in links if r.get("match_method") not in ("doi_exact", "title_fuzzy_local")]

    print(f"Links: doi_exact={len(doi_links)}  title_fuzzy={len(fuzzy_links)}  other/not_found={len(not_found)}")

    # If there are few not_found entries in the links file, sample from corpus records
    corpus_records: list[dict] = []
    if len(not_found) < args.n_each:
        print("Loading corpus for not-found sampling ...")
        corpus_records = load_jsonl(CORPUS_PATH)
        linked_ids = {r["dataset_record_id"] for r in links}
        not_found_corpus = [r for r in corpus_records if r.get("source_id", r.get("dataset_id", "")) not in linked_ids]
        rng.shuffle(not_found_corpus)
        not_found = [
            {
                "dataset_record_id": r.get("source_id", r.get("dataset_id", "")),
                "dataset_title": r.get("title", ""),
                "paper_doi": "",
                "paper_title": "",
                "paper_year": "",
                "match_method": "not_found",
                "confidence": 0.0,
            }
            for r in not_found_corpus[:args.n_each]
        ]

    def sample(pool: list[dict], n: int) -> list[dict]:
        rng.shuffle(pool)
        return pool[:n]

    doi_sample = sample(doi_links, args.n_each)
    fuzzy_sample = sample(fuzzy_links, args.n_each)
    nf_sample = sample(not_found, args.n_each)

    rows = (
        [{"link_type": "doi_exact", **r} for r in doi_sample]
        + [{"link_type": "title_fuzzy", **r} for r in fuzzy_sample]
        + [{"link_type": "not_found", **r} for r in nf_sample]
    )

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "link_type": r.get("link_type", ""),
                "dataset_record_id": r.get("dataset_record_id", ""),
                "dataset_title": (r.get("dataset_title") or r.get("title", ""))[:200],
                "paper_doi": r.get("paper_doi", ""),
                "paper_title": (r.get("paper_title") or "")[:200],
                "paper_year": r.get("paper_year", ""),
                "match_method": r.get("match_method", ""),
                "confidence": r.get("confidence", ""),
                "link_correct": "",
                "error_type": "",
                "notes": "",
            })

    with open(AUDIT_INSTRUCTIONS, "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS_TEXT)

    print(f"CSV written          -> {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Instructions written -> {AUDIT_INSTRUCTIONS.relative_to(ROOT)}")
    print(f"  doi_exact={len(doi_sample)}  title_fuzzy={len(fuzzy_sample)}  not_found={len(nf_sample)}")


if __name__ == "__main__":
    main()
