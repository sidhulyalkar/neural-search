"""Build evidence packets for neuro_judge from the pooled candidate pool.

Usage::

    python scripts/eval/build_neuro_judge_evidence.py \
        --candidates artifacts/field_state/qrels_candidates_pooled.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out artifacts/field_state/neuro_judge_evidence_packets.jsonl

Outputs one EvidencePacket per (query, dataset) pair in the pool.
Missing dataset metadata is enriched from the corpus when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve repo root
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.evidence_retriever import build_evidence_packet
from scripts.eval.benchmark_schema import BenchmarkQueryV1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_queries(path: Path) -> dict[str, BenchmarkQueryV1]:
    queries: dict[str, BenchmarkQueryV1] = {}
    for raw in _load_jsonl(path):
        try:
            q = BenchmarkQueryV1.model_validate(raw)
            queries[q.query_id] = q
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skipping malformed query: {exc}", file=sys.stderr)
    return queries


def _load_concept_results(path: Path | None) -> dict[tuple[str, str], dict]:
    """Index concept rerank results by (query_id, dataset_id)."""
    if path is None or not path.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for rec in _load_jsonl(path):
        qid = str(rec.get("query_id") or "")
        did = str(rec.get("dataset_id") or "")
        if qid and did:
            index[(qid, did)] = rec
    return index


def _load_corpus_index(path: Path | None) -> dict[str, dict]:
    """Index corpus records by source:source_id for metadata enrichment."""
    if path is None or not path.exists():
        return {}
    idx: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = f"{r.get('source','').strip()}:{r.get('source_id','').strip()}"
            idx[key] = r
    return idx


def _enrich_candidate(cand: dict, corpus_idx: dict[str, dict]) -> dict:
    """Fill missing fields in a candidate record from the corpus."""
    rec = corpus_idx.get(cand.get("dataset_id", ""))
    if rec is None:
        return cand
    enriched = dict(cand)
    # Map corpus fields to candidate fields
    field_map = {
        "dataset_description": "description",
        "dataset_species": "species",
        "dataset_modalities": "modalities",
        "dataset_brain_regions": "brain_regions",
        "dataset_tasks": "tasks",
        "dataset_data_standards": "data_standards",
    }
    # Also enrich the flat fields used by evidence_retriever
    if not enriched.get("dataset_description") and rec.get("description"):
        enriched["dataset_description"] = rec["description"]
    for cand_field, corpus_field in [
        ("modalities", "modalities"),
        ("species", "species"),
        ("brain_regions", "brain_regions"),
        ("tasks", "tasks"),
        ("data_standards", "data_standards"),
    ]:
        if not enriched.get(cand_field) and rec.get(corpus_field):
            enriched[cand_field] = rec[corpus_field]
    if not enriched.get("dataset_source_url") and rec.get("url"):
        enriched["dataset_source_url"] = rec["url"]
    if not enriched.get("license") and rec.get("license"):
        enriched["license"] = rec["license"]
    return enriched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_evidence_packets(
    candidates: list[dict],
    queries: dict[str, BenchmarkQueryV1],
    concept_results: dict[tuple[str, str], dict],
    corpus_idx: dict[str, dict] | None = None,
) -> tuple[list[dict], dict]:
    """Build packets and return (packets, quality_stats)."""
    packets = []
    skipped = 0
    enriched_count = 0
    still_missing_desc = 0

    for cand in candidates:
        qid = str(cand.get("query_id") or "")
        did = str(cand.get("dataset_id") or cand.get("source_id") or "")
        query = queries.get(qid)
        if query is None:
            skipped += 1
            continue
        # Enrich from corpus if available
        if corpus_idx:
            orig_desc = cand.get("dataset_description", "")
            cand = _enrich_candidate(cand, corpus_idx)
            if not orig_desc and cand.get("dataset_description"):
                enriched_count += 1
        if not cand.get("dataset_description"):
            still_missing_desc += 1

        cr = concept_results.get((qid, did))
        packet = build_evidence_packet(query, cand, concept_result=cr)
        packets.append(packet.model_dump(mode="json"))

    if skipped:
        print(f"[WARN] skipped {skipped} candidates with unknown query_id", file=sys.stderr)

    stats = {
        "total": len(packets),
        "skipped": skipped,
        "enriched_from_corpus": enriched_count,
        "missing_description": still_missing_desc,
        "missing_pct": round(100 * still_missing_desc / max(len(packets), 1), 1),
    }
    return packets, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build neuro_judge evidence packets")
    parser.add_argument(
        "--candidates",
        default="artifacts/field_state/qrels_candidates_pooled.jsonl",
        help="Pooled qrels candidates JSONL",
    )
    parser.add_argument(
        "--queries",
        default="artifacts/benchmark_queries.jsonl",
        help="Benchmark queries JSONL",
    )
    parser.add_argument(
        "--corpus",
        default="data/corpus/normalized/combined_corpus.jsonl",
        help="Corpus JSONL for metadata enrichment",
    )
    parser.add_argument(
        "--concept-rerank",
        default=None,
        help="Optional concept rerank results JSONL",
    )
    # --out and --output are both accepted
    parser.add_argument(
        "--out", "--output",
        dest="output",
        default="artifacts/field_state/neuro_judge_evidence_packets.jsonl",
        help="Output path for evidence packets JSONL",
    )
    args = parser.parse_args(argv)

    cand_path = _REPO / args.candidates
    query_path = _REPO / args.queries
    corpus_path = _REPO / args.corpus
    cr_path = Path(args.concept_rerank) if args.concept_rerank else None
    out_path = _REPO / args.output

    if not cand_path.exists():
        sys.exit(f"[ERROR] Candidates file not found: {cand_path}")
    if not query_path.exists():
        sys.exit(f"[ERROR] Queries file not found: {query_path}")

    print(f"Loading candidates from {cand_path}...")
    candidates = _load_jsonl(cand_path)
    print(f"Loading queries from {query_path}...")
    queries = _load_queries(query_path)
    concept_results = _load_concept_results(cr_path)
    corpus_idx = _load_corpus_index(corpus_path)
    if corpus_idx:
        print(f"Loaded corpus index: {len(corpus_idx)} records")

    print(f"Building evidence packets for {len(candidates)} candidates...")
    packets, stats = build_evidence_packets(candidates, queries, concept_results, corpus_idx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for p in packets:
            fh.write(json.dumps(p) + "\n")

    print(f"Wrote {len(packets)} evidence packets to {out_path}")
    print(f"  Enriched from corpus: {stats['enriched_from_corpus']}")
    print(f"  Missing description: {stats['missing_description']} ({stats['missing_pct']}%)")

    if stats["missing_pct"] > 10:
        print(
            f"[WARN] {stats['missing_pct']}% of packets missing descriptions "
            "— do NOT run real LLM judge until descriptions are sourced.",
            file=sys.stderr,
        )
    return stats


if __name__ == "__main__":
    main()
