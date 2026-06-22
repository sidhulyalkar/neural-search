"""Export paper cards to obsidian_vault/09_Literature/.

Usage: python scripts/obsidian/export_literature.py [--vault PATH] [--findings PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import paper_card_body, paper_card_frontmatter

DEFAULT_FINDINGS = REPO_ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
DEFAULT_LINKS = REPO_ROOT / "artifacts/literature/paper_dataset_links.jsonl"
DEFAULT_VAULT = REPO_ROOT / "obsidian_vault"


def _aggregate_findings(findings_path: Path) -> dict[str, dict]:
    """Return {paper_id: {title, authors, year, n_findings, finding_ids, regions, species, modalities}}."""
    papers: dict[str, dict] = {}
    with findings_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("paper_id")
            if not pid:
                continue
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "doi": rec.get("paper_doi"),
                    "title": rec.get("paper_title") or pid,
                    "authors": [],
                    "year": rec.get("year"),
                    "finding_ids": [],
                    "regions": [],
                    "species": [],
                    "modalities": [],
                    "extraction_model": rec.get("extraction_model"),
                    "extraction_prompt_version": "extraction_v2",
                    "linked_datasets": [],
                }
            p = papers[pid]
            fid = rec.get("finding_id")
            if fid and fid not in p["finding_ids"]:
                p["finding_ids"].append(fid)
            for r in rec.get("regions") or []:
                if r not in p["regions"]:
                    p["regions"].append(r)
            for s in rec.get("species") or []:
                if s not in p["species"]:
                    p["species"].append(s)
            for m in rec.get("modalities") or []:
                if m not in p["modalities"]:
                    p["modalities"].append(m)

    for p in papers.values():
        p["n_findings"] = len(p["finding_ids"])
    return papers


def _load_links(links_path: Path) -> dict[str, list[str]]:
    """Return {paper_id: [dataset_id, ...]}."""
    links: dict[str, list[str]] = defaultdict(list)
    if not links_path.exists():
        return links
    with links_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("paper_id")
            did = rec.get("dataset_id")
            if pid and did and did not in links[pid]:
                links[pid].append(did)
    return links


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    args = parser.parse_args()

    out_dir = args.vault / "09_Literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Aggregating findings by paper...")
    papers = _aggregate_findings(args.findings)
    links = _load_links(args.links)

    for pid, paper in papers.items():
        paper["linked_datasets"] = links.get(pid, [])
        safe_id = pid.replace(":", "_").replace("/", "_")
        path = out_dir / f"paper_{safe_id}.md"
        fm = paper_card_frontmatter(paper)
        body = paper_card_body(paper)
        safe_write_note(path, fm, body)

    print(f"Done. {len(papers)} paper cards written to {out_dir}")


if __name__ == "__main__":
    main()
