"""Qrels annotation progress report.

Generates reports/eval/qrels_progress_report.json and
reports/eval/qrels_progress_report.md showing exactly what has been
labelled, what remains, and what to prioritize next.

Usage:
    python scripts/eval/qrels_progress_report.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = ROOT / "artifacts/qrels_gold.jsonl"
SILVER_PATH = ROOT / "artifacts/qrels_silver.jsonl"
BRONZE_PATH = ROOT / "artifacts/qrels_bronze.jsonl"
ADJ_PATH = ROOT / "artifacts/field_state/adjudicated_qrels.jsonl"
REVIEWS_PATH = ROOT / "artifacts/field_state/qrels_reviews.jsonl"
POOL_PATH = ROOT / "reports/eval/benchmark_pool.jsonl"

# Canonical LLM-silver qrels (merged in from claude/latent-usefulness-v08,
# 2026-06-23): 317 queries / 13,654 non-error single-LLM-judge labels, used
# for the ablation ladder + regression gate. Single-judge, not human-adjudicated
# — kept separate from the gold/adjudicated/reviews counts above, which track
# the human-labelling pipeline specifically.
CANONICAL_SILVER_PATH = ROOT / "data/qrels/qrels.canonical.jsonl"

REPORT_JSON = ROOT / "reports/eval/qrels_progress_report.json"
REPORT_MD = ROOT / "reports/eval/qrels_progress_report.md"

PUBLISHABLE_TARGET = {
    "min_queries": 100,
    "min_pairs": 1500,
    "min_annotators": 2,
    "agreement_required": True,
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def summarize_qrel_set(records: list[dict], name: str) -> dict:
    if not records:
        return {"name": name, "count": 0}
    label_key = "label" if "label" in records[0] else "relevance"
    labels = Counter(str(r.get(label_key, "?")) for r in records)
    query_ids = {r.get("query_id") for r in records}
    hn = sum(1 for r in records if r.get("hard_negative_triggered") or r.get("hard_negative_violation"))
    return {
        "name": name,
        "count": len(records),
        "queries_covered": len(query_ids),
        "pairs_per_query": round(len(records) / max(len(query_ids), 1), 1),
        "label_distribution": dict(labels),
        "hard_negative_count": hn,
    }


def next_priority_pairs(pool: list[dict], labelled_ids: set[str], n: int = 50) -> list[dict]:
    """Return up to n pool pairs that have not yet been labelled, highest score first."""
    unlabelled = [
        r for r in pool
        if r.get("candidate_id") not in labelled_ids
        and r.get("query_id") + "_" + r.get("dataset_id", r.get("record_id", "")) not in labelled_ids
    ]
    unlabelled.sort(key=lambda r: float(r.get("score", 0)), reverse=True)
    return [
        {
            "query_id": r.get("query_id"),
            "query_text": r.get("query_text", r.get("query", "")),
            "dataset_id": r.get("dataset_id", r.get("record_id", "")),
            "dataset_title": r.get("dataset_title", r.get("title", "")),
            "score": round(float(r.get("score", 0)), 4),
        }
        for r in unlabelled[:n]
    ]


def build_report() -> dict:
    gold = load_jsonl(GOLD_PATH)
    silver = load_jsonl(SILVER_PATH)
    bronze = load_jsonl(BRONZE_PATH)
    adj = load_jsonl(ADJ_PATH)
    reviews = load_jsonl(REVIEWS_PATH)
    pool = load_jsonl(POOL_PATH)
    canonical_silver = load_jsonl(CANONICAL_SILVER_PATH)

    labelled_ids: set[str] = set()
    for r in gold + adj + reviews:
        cid = r.get("candidate_id")
        if cid:
            labelled_ids.add(cid)
        qid = r.get("query_id", "")
        did = r.get("dataset_id", r.get("record_id", ""))
        labelled_ids.add(f"{qid}_{did}")

    adj_summary = summarize_qrel_set(adj, "adjudicated (field_state)")
    reviews_summary = summarize_qrel_set(reviews, "human_reviews (field_state)")
    silver_summary = summarize_qrel_set(silver, "silver_diagnostic")
    bronze_summary = summarize_qrel_set(bronze, "bronze_diagnostic")
    canonical_silver_summary = summarize_qrel_set(
        canonical_silver, "canonical_llm_silver (single-judge, ablation ladder)"
    )

    priority_pairs = next_priority_pairs(pool, labelled_ids, n=50)

    gold_count = count_lines(GOLD_PATH)
    human_pairs = len(adj) + len(reviews)
    unique_human_queries = {r.get("query_id") for r in adj + reviews}

    publishable_checklist = {
        "min_queries_100": len(unique_human_queries) >= 100,
        "min_pairs_1500": human_pairs >= 1500,
        "min_annotators_2": False,  # cannot be auto-determined without annotator list
        "gold_qrels_non_empty": gold_count > 0,
        "agreement_stats_available": False,
    }

    gaps = []
    remaining_pairs = max(0, PUBLISHABLE_TARGET["min_pairs"] - human_pairs)
    remaining_queries = max(0, PUBLISHABLE_TARGET["min_queries"] - len(unique_human_queries))
    if remaining_queries > 0:
        gaps.append(f"Need {remaining_queries} more unique queries annotated (target: 100)")
    if remaining_pairs > 0:
        gaps.append(f"Need {remaining_pairs} more labelled pairs (target: 1,500)")
    if not publishable_checklist["min_annotators_2"]:
        gaps.append("Need >=2 annotators with inter-annotator agreement statistics")
    if gold_count == 0:
        gaps.append("Gold qrels are empty — adjudication workflow must produce gold labels")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "gold_qrels_rows": gold_count,
            "human_adjudicated_rows": len(adj),
            "human_review_rows": len(reviews),
            "silver_diagnostic_rows": len(silver),
            "bronze_diagnostic_rows": len(bronze),
            "canonical_llm_silver_rows": len(canonical_silver),
            "canonical_llm_silver_queries": canonical_silver_summary.get("queries_covered", 0),
            "total_human_labelled_pairs": human_pairs,
            "unique_queries_human_covered": len(unique_human_queries),
            "pool_size": len(pool),
            "unlabelled_pool_pairs": len(pool) - len(labelled_ids.intersection(
                {r.get("candidate_id", "") for r in pool}
            )),
        },
        "publishable_target": PUBLISHABLE_TARGET,
        "publishable_checklist": publishable_checklist,
        "gaps_to_publishable": gaps,
        "sets": {
            "adjudicated": adj_summary,
            "reviews": reviews_summary,
            "silver": silver_summary,
            "bronze": bronze_summary,
            "canonical_llm_silver": canonical_silver_summary,
        },
        "next_50_priority_pairs": priority_pairs,
    }


def render_markdown(report: dict) -> str:
    s = report["summary"]
    checklist = report["publishable_checklist"]
    gaps = report["gaps_to_publishable"]
    lines = [
        "# Qrels Annotation Progress Report",
        f"\n_Generated: {report['generated_at']}_\n",
        "## Current State\n",
        "| Layer | Count |",
        "|---|---|",
        f"| Gold qrels | {s['gold_qrels_rows']} |",
        f"| Field-state adjudicated | {s['human_adjudicated_rows']} |",
        f"| Field-state reviews | {s['human_review_rows']} |",
        f"| Silver diagnostic | {s['silver_diagnostic_rows']} |",
        f"| Bronze diagnostic | {s['bronze_diagnostic_rows']} |",
        f"| **Total human-labelled pairs** | **{s['total_human_labelled_pairs']}** |",
        f"| Unique queries covered (human) | {s['unique_queries_human_covered']} |",
        f"| Pool size | {s['pool_size']} |",
        f"| Canonical LLM-silver qrels (single judge) | {s['canonical_llm_silver_rows']} |",
        f"| Canonical LLM-silver queries | {s['canonical_llm_silver_queries']} |",
        "",
        "_The canonical LLM-silver row is a separate, much larger pool"
        " (`data/qrels/qrels.canonical.jsonl`, merged from `claude/latent-usefulness-v08`"
        " on 2026-06-23) used for the ablation ladder and regression gate. It is"
        " single-LLM-judged, not human-adjudicated, and does not count toward the"
        " human-labelled totals or the publishable checklist below — those still"
        " require human/gold labels per the existing benchmark spec._",
        "",
        "## Publishable Benchmark Checklist\n",
        "Target: 100 queries, 1,500 pairs, 2 annotators, agreement stats.\n",
    ]
    for key, val in checklist.items():
        icon = "✅" if val else "❌"
        label = key.replace("_", " ")
        lines.append(f"- {icon} {label}")
    if gaps:
        lines += ["", "### Gaps to Publishable\n"]
        for g in gaps:
            lines.append(f"- {g}")
    # Per-set detail
    for name, details in report["sets"].items():
        lines += ["", f"## {name.replace('_', ' ').title()}\n"]
        if details["count"] == 0:
            lines.append("_No records._")
            continue
        lines += [
            f"- **Pairs:** {details['count']}",
            f"- **Queries covered:** {details.get('queries_covered', '?')}",
            f"- **Pairs per query:** {details.get('pairs_per_query', '?')}",
            f"- **Hard-negative count:** {details.get('hard_negative_count', 0)}",
            f"- **Label distribution:** {details.get('label_distribution', {})}",
        ]
    # Next priority pairs
    pairs = report.get("next_50_priority_pairs", [])
    if pairs:
        lines += ["", "## Next Priority Pairs to Label\n",
                  "_Top pairs from pool not yet human-labelled, highest score first._\n",
                  "| # | Query | Dataset | Score |",
                  "|---|---|---|---|"]
        for i, p in enumerate(pairs[:50], 1):
            q = (p.get("query_text") or p.get("query_id") or "")[:60]
            d = (p.get("dataset_title") or p.get("dataset_id") or "")[:60]
            lines.append(f"| {i} | {q} | {d} | {p.get('score', 0):.4f} |")
    else:
        lines += ["", "## Next Priority Pairs\n",
                  "_Pool file not found or all pairs already labelled._"]
    return "\n".join(lines)


def main():
    report = build_report()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = render_markdown(report)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"JSON written -> {REPORT_JSON.relative_to(ROOT)}")
    print(f"MD written   -> {REPORT_MD.relative_to(ROOT)}")
    s = report["summary"]
    print(f"\n  gold={s['gold_qrels_rows']}  adj={s['human_adjudicated_rows']}  "
          f"reviews={s['human_review_rows']}  silver={s['silver_diagnostic_rows']}  "
          f"bronze={s['bronze_diagnostic_rows']}")
    print(f"  human pairs={s['total_human_labelled_pairs']}  "
          f"queries covered={s['unique_queries_human_covered']}")
    print(f"  canonical_llm_silver={s['canonical_llm_silver_rows']} "
          f"queries={s['canonical_llm_silver_queries']} (single-judge, not human)")
    gaps = report["gaps_to_publishable"]
    if gaps:
        print("\n  Gaps to publishable benchmark:")
        for g in gaps:
            print(f"    [gap] {g}")
    else:
        print("\n  [ok] Publishable checklist complete!")


if __name__ == "__main__":
    main()
