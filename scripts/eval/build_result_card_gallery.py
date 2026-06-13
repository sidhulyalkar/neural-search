"""Build a result card gallery for the Neural Search benchmark.

Generates a Markdown gallery with:
- 8 success cases (top-ranked results that are highly relevant, relevance=3)
- 8 failure cases (top-ranked results that are irrelevant, relevance=0)
- 4 hard-negative examples (results matching stated hard-negative patterns)

Usage:
    python scripts/eval/build_result_card_gallery.py \\
        --qrels artifacts/qrels.jsonl \\
        --queries artifacts/benchmark_queries.jsonl \\
        --corpus data/corpus/normalized/combined_corpus.jsonl \\
        --runs-dir reports/eval/runs \\
        --out reports/eval/result_card_gallery.md \\
        [--variant usefulness]

If qrels are absent, writes a placeholder. If corpus is absent, omits
dataset titles/descriptions but still writes the gallery structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ResultCard:
    query_id: str
    query_text: str
    intent: str
    record_id: str
    rank: int
    score: float
    relevance: int
    label: str
    rationale: str
    hard_negative_violation: bool
    dataset_title: str = ""
    dataset_description: str = ""
    variant: str = ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_queries(path: Path) -> dict[str, dict]:
    """Returns {query_id: record}"""
    queries: dict[str, dict] = {}
    for row in _load_jsonl(path):
        qid = str(row.get("query_id", ""))
        # accept both query_text and query field names
        if "query_text" not in row and "query" in row:
            row["query_text"] = row["query"]
        if qid:
            queries[qid] = row
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, dict]]:
    """Returns {query_id: {dataset_id: {relevance, rationale, hn_violation}}}"""
    qrels: dict[str, dict[str, dict]] = {}
    for row in _load_jsonl(path):
        qid = str(row.get("query_id", ""))
        did = str(row.get("dataset_id") or row.get("record_id", ""))
        if not qid or not did:
            continue
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = {
            "relevance": int(row.get("relevance", row.get("label", 0))),
            "label": str(row.get("label", "")),
            "rationale": str(row.get("rationale", "")),
            "hard_negative_violation": bool(row.get("hard_negative_violation", False)),
        }
    return qrels


def _load_corpus_index(path: Path) -> dict[str, dict]:
    """Returns {record_id: {title, description}} — loads lazily with stable IDs."""
    index: dict[str, dict] = {}
    if not path.exists():
        return index
    for row in _load_jsonl(path):
        source = str(row.get("source", "unknown"))
        source_id = str(
            row.get("source_id") or row.get("dataset_id") or row.get("id", "unknown")
        )
        record_id = f"{source}:{source_id}"
        index[record_id] = {
            "title": str(row.get("title", row.get("name", ""))),
            "description": str(row.get("description", row.get("abstract", "")))[:300],
        }
    return index


def _load_run(
    path: Path, depth: int = 50
) -> dict[str, list[tuple[int, str, float]]]:
    """Returns {query_id: [(rank, record_id, score)]}"""
    runs: dict[str, list[tuple[int, str, float]]] = {}
    for row in _load_jsonl(path):
        rank = int(row.get("rank", 10**9))
        if rank > depth:
            continue
        qid = str(row["query_id"])
        rid = str(row.get("record_id", row.get("dataset_id", "")))
        score = float(row.get("score", 0.0))
        if qid not in runs:
            runs[qid] = []
        runs[qid].append((rank, rid, score))
    for qid in runs:
        runs[qid].sort(key=lambda x: x[0])
    return runs


# ---------------------------------------------------------------------------
# Card collection
# ---------------------------------------------------------------------------


def _collect_cards(
    run: dict[str, list[tuple[int, str, float]]],
    queries: dict[str, dict],
    qrels: dict[str, dict[str, dict]],
    corpus_index: dict[str, dict],
    variant: str,
    top_k: int = 10,
) -> tuple[list[ResultCard], list[ResultCard], list[ResultCard]]:
    """Returns (successes, failures, hn_examples) as sorted card lists."""
    successes: list[ResultCard] = []
    failures: list[ResultCard] = []
    hn_examples: list[ResultCard] = []

    for query_id, rows in run.items():
        q = queries.get(query_id, {})
        query_text = str(q.get("query_text", q.get("query", query_id)))
        intent = str(q.get("intent", "UNKNOWN"))
        q_qrels = qrels.get(query_id, {})

        for rank, record_id, score in rows[:top_k]:
            if record_id not in q_qrels:
                continue
            qrel = q_qrels[record_id]
            corpus_meta = corpus_index.get(record_id, {})
            card = ResultCard(
                query_id=query_id,
                query_text=query_text,
                intent=intent,
                record_id=record_id,
                rank=rank,
                score=score,
                relevance=qrel["relevance"],
                label=qrel["label"],
                rationale=qrel["rationale"],
                hard_negative_violation=qrel["hard_negative_violation"],
                dataset_title=corpus_meta.get("title", ""),
                dataset_description=corpus_meta.get("description", ""),
                variant=variant,
            )
            if qrel["hard_negative_violation"]:
                hn_examples.append(card)
            elif qrel["relevance"] == 3:
                successes.append(card)
            elif qrel["relevance"] == 0:
                failures.append(card)

    # Sort for gallery: best successes first, worst failures first
    successes.sort(key=lambda c: (c.rank, -c.score))
    failures.sort(key=lambda c: (c.rank, c.score))
    hn_examples.sort(key=lambda c: c.rank)

    return successes[:8], failures[:8], hn_examples[:4]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_card(card: ResultCard, card_type: str) -> str:
    icon = {"success": "✓", "failure": "✗", "hard_negative": "⚠"}.get(card_type, "?")
    lines = [
        f"### {icon} [{card_type.upper()}] `{card.record_id}`\n",
        f"**Query:** {card.query_text}  ",
        f"**Intent:** `{card.intent}`  ",
        f"**Rank:** {card.rank}  **Score:** {card.score:.4f}  "
        f"**Relevance:** {card.relevance}  ",
    ]
    if card.dataset_title:
        lines.append(f"**Dataset title:** {card.dataset_title}  ")
    if card.dataset_description:
        lines.append(f"**Description (excerpt):** {card.dataset_description}  ")
    if card.rationale:
        lines.append(f"**Rationale:** {card.rationale}  ")
    if card.hard_negative_violation:
        lines.append("**Hard-negative violation: YES**  ")
    lines.append("")
    return "\n".join(lines)


def _render_placeholder(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Result Card Gallery\n\n"
        "**Status:** No adjudicated qrels found.\n\n"
        "This gallery requires `artifacts/qrels.jsonl` with annotated pairs.\n"
        "Re-run after completing annotation:\n"
        "```\n"
        "python scripts/eval/build_result_card_gallery.py \\\n"
        "    --qrels artifacts/qrels.jsonl \\\n"
        "    --queries artifacts/benchmark_queries.jsonl \\\n"
        "    --runs-dir reports/eval/runs\n"
        "```\n",
        encoding="utf-8",
    )


def _render_gallery(
    successes: list[ResultCard],
    failures: list[ResultCard],
    hn_examples: list[ResultCard],
    variant: str,
) -> str:
    lines = [
        f"# Result Card Gallery — `{variant}`\n",
        f"Successes: {len(successes)}  Failures: {len(failures)}  "
        f"Hard-negative examples: {len(hn_examples)}\n",
        "---\n",
        "## Successes (relevance = 3, top-ranked)\n",
    ]
    if successes:
        for card in successes:
            lines.append(_render_card(card, "success"))
    else:
        lines.append("_No success examples with relevance=3 found in judged pairs._\n")

    lines.append("---\n")
    lines.append("## Failures (relevance = 0, top-ranked)\n")
    if failures:
        for card in failures:
            lines.append(_render_card(card, "failure"))
    else:
        lines.append("_No failure examples found in judged pairs._\n")

    lines.append("---\n")
    lines.append("## Hard-Negative Examples\n")
    lines.append(
        "> These results look relevant to the query but are explicitly excluded "
        "by a stated hard-negative pattern.\n"
    )
    if hn_examples:
        for card in hn_examples:
            lines.append(_render_card(card, "hard_negative"))
    else:
        lines.append("_No hard-negative violations found in judged pairs._\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_gallery(
    qrels_path: Path,
    queries_path: Path,
    runs_dir: Path,
    corpus_path: Path | None,
    out_path: Path,
    variant: str = "",
    top_k: int = 10,
) -> int:
    qrels = _load_qrels(qrels_path)
    if not qrels:
        print(
            f"[INFO] No qrels found at {qrels_path} — writing placeholder gallery.",
            file=sys.stderr,
        )
        _render_placeholder(out_path)
        print(f"Placeholder written to {out_path}")
        return 0

    queries = _load_queries(queries_path)
    corpus_index = _load_corpus_index(corpus_path) if corpus_path else {}

    if not runs_dir.exists():
        print(f"[WARN] Runs dir not found: {runs_dir}", file=sys.stderr)

    # Pick variant — default to first available run file
    run_files = sorted(runs_dir.glob("*.jsonl")) if runs_dir.exists() else []
    if not run_files:
        _render_placeholder(out_path)
        print("No run files — writing placeholder gallery.")
        return 0

    chosen_path = None
    if variant:
        candidates = [f for f in run_files if f.stem == variant]
        chosen_path = candidates[0] if candidates else run_files[0]
    else:
        chosen_path = run_files[0]

    chosen_variant = chosen_path.stem
    run = _load_run(chosen_path, depth=top_k * 5)

    successes, failures, hn_examples = _collect_cards(
        run=run,
        queries=queries,
        qrels=qrels,
        corpus_index=corpus_index,
        variant=chosen_variant,
        top_k=top_k,
    )

    markdown = _render_gallery(successes, failures, hn_examples, variant=chosen_variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(out_path)
    print(
        f"Gallery written to {out_path} "
        f"({len(successes)} successes, {len(failures)} failures, "
        f"{len(hn_examples)} HN examples)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build result card gallery")
    parser.add_argument("--qrels", type=Path, default=Path("artifacts/qrels.jsonl"))
    parser.add_argument(
        "--queries", type=Path, default=Path("artifacts/benchmark_queries.jsonl")
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional: corpus JSONL for dataset titles",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("reports/eval/runs"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/eval/result_card_gallery.md")
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help="Run variant to show (default: first available)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    return build_gallery(
        qrels_path=args.qrels,
        queries_path=args.queries,
        runs_dir=args.runs_dir,
        corpus_path=args.corpus,
        out_path=args.out,
        variant=args.variant,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    sys.exit(main())
