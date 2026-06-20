#!/usr/bin/env python3
"""Export a whitepaper claim registry to Obsidian and reports/.

Usage:
    python scripts/obsidian/export_claim_registry.py \
        --vault obsidian_vault \
        --out reports/eval/whitepaper_claims_status.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

CLAIMS = [
    {
        "id": "C001",
        "text": "Latent-usefulness retrieval achieves NDCG@10 > 0.40 on the gold qrels benchmark.",
        "metric": "NDCG@10",
        "required_artifact": "artifacts/qrels_gold.jsonl",
        "status": "unsupported",
        "notes": "Gold qrels pending human audit completion.",
    },
    {
        "id": "C002",
        "text": "Hard-negative violation rate < 5% on gold qrels.",
        "metric": "hard_negative_violation_rate",
        "required_artifact": "artifacts/qrels_gold.jsonl",
        "status": "unsupported",
        "notes": "Pending gold qrels.",
    },
    {
        "id": "C003",
        "text": "Weak supervision silver qrels cover ≥ 80% of pooled pairs.",
        "metric": "silver_coverage",
        "required_artifact": "artifacts/qrels_silver.jsonl",
        "status": "unsupported",
        "notes": "Run build_qrels_from_votes.py to check.",
    },
]


def _status_badge(status: str) -> str:
    badges = {
        "unsupported": "🔴 Unsupported",
        "weakly_supported": "🟡 Weakly supported",
        "supported": "🟢 Supported",
        "contradicted": "❌ Contradicted",
    }
    return badges.get(status, status)


def _render_md(claims: list[dict]) -> str:
    lines = [
        "# Whitepaper Claim Registry\n",
        "_Auto-generated. Edit claim status in `scripts/obsidian/export_claim_registry.py`._\n",
        "| ID | Claim | Metric | Artifact | Status | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for c in claims:
        lines.append(
            f"| {c['id']} | {c['text']} | `{c['metric']}` | "
            f"`{c['required_artifact']}` | {_status_badge(c['status'])} | {c['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    content = _render_md(CLAIMS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Claim registry written → {args.out}")

    vault_path = args.vault / "07_Whitepaper" / "Claims Registry.md"
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    vault_path.write_text(content, encoding="utf-8")
    print(f"Claim registry written → {vault_path}")


if __name__ == "__main__":
    main()
