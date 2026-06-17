#!/usr/bin/env python3
"""Build pair_evidence.jsonl from the benchmark pool, queries, and corpus.

Usage:
    python scripts/eval/build_evidence.py \
        --pool reports/eval/benchmark_pool.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out artifacts/eval/pair_evidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import dataset_evidence_from_record, PairEvidence
from neural_search.eval.query_decomposition import load_query_specs


def _load_pool(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_corpus_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = f"{rec.get('source', '')}:{rec.get('source_id', '')}"
            index[rid] = rec
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair evidence JSONL.")
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    specs = {s.query_id: s for s in load_query_specs(args.queries)}
    corpus = _load_corpus_index(args.corpus)
    pool_rows = _load_pool(args.pool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    missing_queries = missing_corpus = written = 0

    with args.out.open("w", encoding="utf-8") as out_fh:
        for row in pool_rows:
            qid = row["query_id"]
            rid = row["record_id"]
            if qid not in specs:
                missing_queries += 1
                continue
            if rid not in corpus:
                missing_corpus += 1
                continue
            pair = PairEvidence(
                query_id=qid, record_id=rid,
                query=specs[qid],
                dataset=dataset_evidence_from_record(corpus[rid]),
                pooled_from=row.get("pooled_from") or [],
                min_rank=int(row.get("min_rank", 1000)),
                priority=str(row.get("priority", "normal")),
            )
            out_fh.write(json.dumps(pair.to_dict()) + "\n")
            written += 1

    print(f"Written: {written} pairs")
    if missing_queries:
        print(f"Warning: {missing_queries} pool rows had no matching query spec")
    if missing_corpus:
        print(f"Warning: {missing_corpus} pool rows had no matching corpus record")


if __name__ == "__main__":
    main()
