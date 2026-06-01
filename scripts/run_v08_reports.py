#!/usr/bin/env python3
"""Generate v0.8 usefulness reports from seed data."""
from pathlib import Path
from neural_search.evaluation.usefulness_benchmark import load_seed_pairs, run_usefulness_benchmark
from neural_search.evaluation.ablation_runner import (
    AblationConfig, CandidatePool, run_ablation
)
from neural_search.retrieval.usefulness_scorer import DatasetContext

SEED_PATH = Path("data/eval/usefulness_seed_pairs.jsonl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Load seed pairs
queries, labels = load_seed_pairs(SEED_PATH)
print(f"Loaded {len(queries)} queries, {len(labels)} labels")

# Build synthetic candidate pool from seed data
pool = CandidatePool(candidates={})
for lbl in labels:
    cid = lbl.candidate_id
    if cid not in pool.candidates:
        pool.candidates[cid] = DatasetContext(dataset_id=cid)
for q in queries:
    for cid in q.candidate_ids:
        if cid not in pool.candidates:
            pool.candidates[cid] = DatasetContext(dataset_id=cid)

# Run benchmark with identity run (alphabetical baseline)
identity_run = {q.query_id: sorted(q.candidate_ids) for q in queries}
bench_report = run_usefulness_benchmark(queries, labels, identity_run, k=5)
bench_md = REPORTS_DIR / "usefulness_benchmark_v08.md"
bench_md.write_text(bench_report.to_markdown(), encoding="utf-8")
print(f"Wrote {bench_md}")

# Run ablation
config = AblationConfig(
    queries=queries,
    labels=labels,
    pool=pool,
    k=5,
    out_path=REPORTS_DIR / "ablation_v08.md",
)
ablation_report = run_ablation(config)
print(f"Wrote {REPORTS_DIR / 'ablation_v08.md'}")
print("\nAblation Results:")
for variant, metrics in ablation_report.variant_metrics.items():
    print(f"  {variant}: NDCG={metrics['ndcg_at_k']:.4f} MRR={metrics['mrr']:.4f}")
