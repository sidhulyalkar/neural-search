"""Export claim cards to obsidian_vault/10_Claims/.

Usage: python scripts/obsidian/export_claims.py [--vault PATH] [--claims PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import claim_card_body, claim_card_frontmatter

DEFAULT_CLAIMS = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"
DEFAULT_VAULT = REPO_ROOT / "obsidian_vault"


def _safe_filename(claim_id: str) -> str:
    return claim_id.replace("node:claim:", "cl_").replace("/", "_").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    args = parser.parse_args()

    out_dir = args.vault / "10_Claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.claims.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claim = json.loads(line)
            claim_id = claim.get("claim_id")
            if not claim_id:
                continue
            filename = _safe_filename(claim_id) + ".md"
            path = out_dir / filename
            fm = claim_card_frontmatter(claim)
            body = claim_card_body(claim)
            safe_write_note(path, fm, body)
            count += 1

    print(f"Done. {count} claim cards written to {out_dir}")


if __name__ == "__main__":
    main()
