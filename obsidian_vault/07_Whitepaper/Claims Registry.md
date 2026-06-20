# Whitepaper Claim Registry

_Auto-generated. Edit claim status in `scripts/obsidian/export_claim_registry.py`._

| ID | Claim | Metric | Artifact | Status | Notes |
|---|---|---|---|---|---|
| C001 | Latent-usefulness retrieval achieves NDCG@10 > 0.40 on the gold qrels benchmark. | `NDCG@10` | `artifacts/qrels_gold.jsonl` | 🔴 Unsupported | Gold qrels pending human audit completion. |
| C002 | Hard-negative violation rate < 5% on gold qrels. | `hard_negative_violation_rate` | `artifacts/qrels_gold.jsonl` | 🔴 Unsupported | Pending gold qrels. |
| C003 | Weak supervision silver qrels cover ≥ 80% of pooled pairs. | `silver_coverage` | `artifacts/qrels_silver.jsonl` | 🔴 Unsupported | Run build_qrels_from_votes.py to check. |
