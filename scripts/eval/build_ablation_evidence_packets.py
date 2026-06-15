#!/usr/bin/env python3
"""Build EvidencePacket JSONL from ablation ladder run files.

Reads run files from all completed ablation rungs, pools the top-N candidates
per query across rungs, deduplicates by (query_id, record_id), and writes one
EvidencePacket per pair for consumption by run_parallel_llm_qrels.py.

Usage
-----
    python scripts/eval/build_ablation_evidence_packets.py

    # Custom pool size or run dir
    python scripts/eval/build_ablation_evidence_packets.py \\
        --top-per-rung 20 \\
        --runs-dir reports/eval/runs \\
        --out artifacts/ablation_judge/evidence_packets.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.evidence_packet import (  # noqa: E402
    AffordanceMatch,
    EvidencePacket,
    LinkedPaper,
)

DEFAULT_RUNS_DIR = Path("reports/eval/runs")
DEFAULT_QUERIES = Path("data/eval/benchmark_queries_canonical.yaml")
DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
DEFAULT_OUT = Path("artifacts/ablation_judge/evidence_packets.jsonl")
DEFAULT_TOP_PER_RUNG = 20

RUNG_FILES = [
    "bm25.jsonl",
    "bm25_structured.jsonl",
    "dense_bge.jsonl",
    "hybrid_rrf.jsonl",
    "hybrid_graph.jsonl",
    "full.jsonl",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_queries(path: Path) -> dict[str, dict[str, Any]]:
    """Return {query_id: query_dict}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries: list[dict] = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and "query" in v[0]:
                queries = v
                break
    elif isinstance(data, list):
        queries = data
    return {str(q.get("id", q.get("query_id", ""))): q for q in queries}


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    """Return {stable_id: record}.

    Corpus record_id is 'dataset:source:id'; run record_id is 'source:id'.
    We index under both so lookups always work.
    """
    by_id: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        full_id = str(r.get("dataset_id", ""))
        short_id = full_id.removeprefix("dataset:")
        if full_id:
            by_id[full_id] = r
        if short_id and short_id != full_id:
            by_id[short_id] = r
        # Also index as source:source_id
        source = str(r.get("source", ""))
        sid = str(r.get("source_id", ""))
        if source and sid:
            by_id[f"{source}:{sid}"] = r
    return by_id


def _load_run_pool(
    runs_dir: Path,
    top_per_rung: int,
) -> dict[str, set[str]]:
    """Return {query_id: {record_id, ...}} pooled from top-N of each rung."""
    pool: dict[str, set[str]] = {}
    for fname in RUNG_FILES:
        path = runs_dir / fname
        if not path.exists():
            print(f"  Skipping missing rung: {fname}")
            continue
        by_query: dict[str, list[tuple[int, str]]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec["query_id"])
            rank = int(rec["rank"])
            rid = str(rec["record_id"])
            by_query.setdefault(qid, []).append((rank, rid))

        count = 0
        for qid, ranked in by_query.items():
            ranked.sort()
            top = [rid for _, rid in ranked[:top_per_rung]]
            pool.setdefault(qid, set()).update(top)
            count += len(top)
        print(f"  {fname}: {count} candidates pooled")
    return pool


# ---------------------------------------------------------------------------
# EvidencePacket builder
# ---------------------------------------------------------------------------

def _list_field(record: dict, *keys: str) -> list[str]:
    for k in keys:
        v = record.get(k)
        if isinstance(v, list):
            return [str(x) for x in v if x]
    return []


def _str_field(record: dict, *keys: str) -> str:
    for k in keys:
        v = record.get(k)
        if v:
            return str(v)
    return ""


def _hard_negatives(query: dict) -> list[str]:
    parts: list[str] = []
    for field in ("hard_negative_modalities", "hard_negative_tasks", "hard_negative_regions"):
        for v in (query.get(field) or []):
            parts.append(f"{field.replace('hard_negative_', '')}: {v}")
    return parts


def _affordance_matches(record: dict) -> list[AffordanceMatch]:
    raw = record.get("analysis_affordances") or []
    if isinstance(raw, list):
        return [
            AffordanceMatch(
                affordance=str(a) if isinstance(a, str) else str(a.get("name", a)),
                matched=True,
                confidence=0.7,
            )
            for a in raw
        ]
    return []


def _linked_papers(record: dict) -> list[LinkedPaper]:
    raw = record.get("linked_papers") or []
    papers: list[LinkedPaper] = []
    for p in raw:
        if isinstance(p, dict):
            papers.append(LinkedPaper(
                title=str(p.get("title", "")),
                abstract=str(p.get("abstract", "")),
                doi=str(p.get("doi", "")),
            ))
        elif isinstance(p, str):
            papers.append(LinkedPaper(title=p))
    return papers


def _usability(record: dict, key: str) -> bool | None:
    flags = record.get("usability_flags")
    if isinstance(flags, dict):
        v = flags.get(key)
        if isinstance(v, bool):
            return v
    return None


def build_packet(
    query_id: str,
    query: dict[str, Any],
    record_id: str,
    record: dict[str, Any],
) -> EvidencePacket:
    return EvidencePacket(
        # query side
        query_id=query_id,
        query_text=str(query.get("query", "")),
        query_intent=str(query.get("intent", "EXPLORATION")),
        hard_negatives=_hard_negatives(query),
        expected_species=_list_field(query, "expected_species"),
        expected_modalities=_list_field(query, "expected_modalities_any", "expected_modalities"),
        expected_brain_regions=_list_field(query, "expected_regions_any", "expected_brain_regions"),
        expected_tasks=_list_field(query, "expected_tasks"),
        expected_analysis_affordances=_list_field(query, "expected_analysis_any"),
        # dataset side
        dataset_id=record_id,
        title=_str_field(record, "title"),
        source_archive=_str_field(record, "source"),
        source_url=_str_field(record, "url"),
        description=_str_field(record, "description"),
        dataset_modalities=_list_field(record, "modalities"),
        dataset_species=_list_field(record, "species"),
        dataset_brain_regions=_list_field(record, "brain_regions"),
        dataset_tasks=_list_field(record, "tasks"),
        data_standards=_list_field(record, "data_standards"),
        license=_str_field(record, "license"),
        # derived evidence
        linked_papers=_linked_papers(record),
        affordance_matches=_affordance_matches(record),
        # raw data signals
        has_raw_data=_usability(record, "has_raw_data"),
        has_processed_data=_usability(record, "has_neural_data"),
        file_format_evidence=_list_field(record, "file_formats"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EvidencePackets from ablation run pool.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-per-rung", type=int, default=DEFAULT_TOP_PER_RUNG)
    args = parser.parse_args(argv)

    print(f"Loading queries from {args.queries} ...")
    queries = _load_queries(args.queries)
    print(f"  {len(queries)} queries")

    print(f"Loading corpus from {args.corpus} ...")
    corpus = _load_corpus(args.corpus)
    print(f"  {len(corpus)} index entries")

    print(f"\nPooling top-{args.top_per_rung} candidates per rung ...")
    pool = _load_run_pool(args.runs_dir, args.top_per_rung)

    total_pairs = sum(len(v) for v in pool.values())
    print(f"\nTotal unique (query, candidate) pairs: {total_pairs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    built = 0
    missing_corpus = 0
    missing_query = 0

    with args.out.open("w", encoding="utf-8") as out_f:
        for query_id, record_ids in pool.items():
            query = queries.get(query_id)
            if query is None:
                missing_query += 1
                continue
            for record_id in record_ids:
                record = corpus.get(record_id)
                if record is None:
                    missing_corpus += 1
                    continue
                packet = build_packet(query_id, query, record_id, record)
                out_f.write(packet.model_dump_json() + "\n")
                built += 1

    print(f"\nDone: {built} packets written → {args.out}")
    if missing_corpus:
        print(f"  Warning: {missing_corpus} record IDs not found in corpus")
    if missing_query:
        print(f"  Warning: {missing_query} query IDs not found in canonical queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
