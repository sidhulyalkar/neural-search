# Task 07 — Obsidian Vault I/O + All Scripts

**Files:**
- Create: `neural_search/obsidian/__init__.py`
- Create: `neural_search/obsidian/templates.py`
- Create: `neural_search/obsidian/io.py`
- Create: `scripts/obsidian/init_vault.py`
- Create: `scripts/obsidian/export_dataset_cards.py`
- Create: `scripts/obsidian/export_query_cards.py`
- Create: `scripts/obsidian/export_audit_queue.py`
- Create: `scripts/obsidian/import_audits.py`
- Create: `scripts/obsidian/compile_vault.py`
- Create: `scripts/obsidian/export_claim_registry.py`
- Test: `tests/test_obsidian_io.py`
- Test: `tests/test_obsidian_exports.py`

---

## Part A — templates.py + io.py

- [ ] **Step 1: Create `tests/test_obsidian_io.py`**

```python
"""Tests for Obsidian vault I/O — frontmatter roundtrip and write-safety."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_search.obsidian.io import parse_note, safe_write_note, read_note


class TestParseNote:
    def test_parses_frontmatter_and_body(self):
        content = "---\ntype: dataset\ndataset_id: dandi:1\n---\n\n# Title\n\nBody text."
        fm, body = parse_note(content)
        assert fm["type"] == "dataset"
        assert fm["dataset_id"] == "dandi:1"
        assert "Body text." in body

    def test_no_frontmatter_returns_empty_dict(self):
        content = "Just a body with no frontmatter."
        fm, body = parse_note(content)
        assert fm == {}
        assert "Just a body" in body

    def test_malformed_frontmatter_returns_empty_dict(self):
        content = "---\n: broken: yaml: \n---\nBody."
        fm, body = parse_note(content)
        assert isinstance(fm, dict)


class TestSafeWriteNote:
    def test_creates_new_file(self, tmp_path):
        note_path = tmp_path / "test.md"
        safe_write_note(note_path, {"type": "dataset", "label": None}, "# Body")
        assert note_path.exists()
        content = note_path.read_text()
        assert "type: dataset" in content

    def test_preserves_human_label_on_overwrite(self, tmp_path):
        note_path = tmp_path / "annot.md"
        # First write — no human label
        safe_write_note(note_path, {"type": "annotation", "label": None, "audit_status": "pending"}, "")
        # Human edits the file
        content = note_path.read_text().replace("label:", "label: 2")
        note_path.write_text(content)
        # Second write by export script — must NOT overwrite human label
        safe_write_note(note_path, {"type": "annotation", "label": None, "audit_status": "pending"}, "")
        fm, _ = read_note(note_path)
        assert fm.get("label") == 2

    def test_preserves_audit_status_done(self, tmp_path):
        note_path = tmp_path / "annot2.md"
        safe_write_note(note_path, {"audit_status": "pending"}, "")
        content = note_path.read_text().replace("audit_status: pending", "audit_status: done")
        note_path.write_text(content)
        # Re-export with pending
        safe_write_note(note_path, {"audit_status": "pending"}, "")
        fm, _ = read_note(note_path)
        assert fm.get("audit_status") == "done"

    def test_creates_parent_dirs(self, tmp_path):
        note_path = tmp_path / "deep" / "nested" / "note.md"
        safe_write_note(note_path, {"x": 1}, "")
        assert note_path.exists()
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_obsidian_io.py -v 2>&1 | head -10
```

- [ ] **Step 3: Create `neural_search/obsidian/__init__.py`**

```python
"""Obsidian vault I/O — templates and safe read/write."""
```

- [ ] **Step 4: Create `neural_search/obsidian/templates.py`**

```python
"""Frontmatter dataclasses and Markdown renderers for the Obsidian vault."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import yaml


def render_frontmatter(data: dict) -> str:
    """Render a dict as a YAML frontmatter block."""
    # Remove None values to keep notes clean
    clean = {k: v for k, v in data.items() if v is not None}
    return "---\n" + yaml.dump(clean, default_flow_style=False, allow_unicode=True) + "---\n"


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def dataset_card_frontmatter(ev) -> dict:
    """Build frontmatter dict from a DatasetEvidence object."""
    today = date.today().isoformat()
    return {
        "type": "dataset",
        "dataset_id": ev.record_id,
        "source": ev.source,
        "species": ev.species or None,
        "modalities": ev.modalities or None,
        "data_levels": ev.data_levels or None,
        "tasks": ev.tasks or None,
        "regions": ev.regions or None,
        "license": ev.license,
        "doi": ev.doi,
        "url": ev.url,
        "raw_data_available": ev.raw_data_available,
        "metadata_completeness": round(ev.metadata_completeness, 2),
        "last_synced": today,
        "tags": ["dataset", ev.source],
    }


def dataset_card_body(ev) -> str:
    desc = ev.description or "_No description available._"
    return f"# {ev.title or ev.record_id}\n\n{desc}\n"


# ---------------------------------------------------------------------------
# Query card
# ---------------------------------------------------------------------------

def query_card_frontmatter(spec) -> dict:
    return {
        "type": "query",
        "query_id": spec.query_id,
        "intent": spec.intent,
        "required_modalities": spec.required_modalities or None,
        "required_data_levels": spec.data_level_requirements or None,
        "species_constraints": spec.required_species or None,
        "region_constraints": spec.brain_regions or None,
        "task_constraints": spec.task_constraints or None,
        "hard_negatives": spec.hard_negatives or None,
        "status": "active",
        "tags": ["query", spec.intent.lower()],
    }


def query_card_body(spec) -> str:
    hn_lines = "\n".join(f"- {h}" for h in spec.hard_negatives) or "_None defined_"
    return (
        f"# {spec.query_text}\n\n"
        f"**Scientific goal:** {spec.scientific_goal}\n\n"
        f"## Hard Negatives\n{hn_lines}\n"
    )


# ---------------------------------------------------------------------------
# Annotation (audit) card
# ---------------------------------------------------------------------------

def annotation_card_frontmatter(
    annotation_id: str,
    query_id: str,
    record_id: str,
    label: int | None,
    confidence: float | None,
    source: str,
    audit_status: str = "pending",
    judge_version: str = "lf_v1",
) -> dict:
    from datetime import datetime, timezone
    return {
        "type": "annotation",
        "annotation_id": annotation_id,
        "query_id": query_id,
        "dataset_id": record_id,
        "label": label,
        "confidence": confidence,
        "source": source,
        "audit_status": audit_status,
        "judge_version": judge_version,
        "created": datetime.now(timezone.utc).date().isoformat(),
        "tags": ["annotation", "audit"],
    }


def annotation_card_body(
    query_text: str,
    scientific_goal: str,
    hard_negatives: list[str],
    dataset_title: str,
    dataset_desc: str | None,
    lf_votes: list[dict],
    ensemble_label: int | None,
    ensemble_confidence: float | None,
    llm_judgment: dict | None = None,
) -> str:
    hn_lines = "\n".join(f"- {h}" for h in hard_negatives) or "_None_"
    vote_lines = "\n".join(
        f"- **{v['lf_name']}**: label={v['label']}, conf={v['confidence']:.2f} — {v['rationale']}"
        for v in lf_votes if not v.get("abstain")
    ) or "_No active votes_"

    llm_section = ""
    if llm_judgment:
        llm_section = (
            f"\n## LLM Judgment\n"
            f"- Label: {llm_judgment.get('label')}, Confidence: {llm_judgment.get('confidence')}\n"
            f"- Rationale: {llm_judgment.get('rationale', '')}\n"
        )

    return (
        f"## Query\n**{query_text}**\n\n"
        f"**Scientific goal:** {scientific_goal}\n\n"
        f"## Hard Negatives\n{hn_lines}\n\n"
        f"## Dataset\n**{dataset_title}**\n\n{dataset_desc or '_No description_'}\n"
        f"\n## Rule Votes\n{vote_lines}\n"
        f"{llm_section}"
        f"\n## Ensemble\n"
        f"- Label: **{ensemble_label}**, Confidence: {ensemble_confidence}\n\n"
        f"## Human Audit Checklist\n"
        f"- [ ] Reviewed query intent and hard negatives\n"
        f"- [ ] Checked dataset modality / species\n"
        f"- [ ] Verified or corrected label in frontmatter\n"
        f"- [ ] Set `audit_status: done` in frontmatter\n\n"
        f"> **Edit in frontmatter:** `label`, `confidence`, `audit_status`\n"
    )
```

- [ ] **Step 5: Create `neural_search/obsidian/io.py`**

```python
"""Safe Obsidian vault note reading and writing.

Write-safety rule: fields listed in HUMAN_OWNED_FIELDS are NEVER overwritten
by export scripts if a human has already set them to a non-None value.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from neural_search.obsidian.templates import render_frontmatter

HUMAN_OWNED_FIELDS = {"label", "confidence", "audit_status"}

_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_note(content: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_str) from a Markdown note string."""
    match = _FM_RE.match(content)
    if not match:
        return {}, content
    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    body = content[match.end():]
    return fm, body


def read_note(path: Path) -> tuple[dict, str]:
    """Read and parse an existing vault note."""
    return parse_note(path.read_text(encoding="utf-8"))


def safe_write_note(path: Path, frontmatter: dict, body: str) -> None:
    """Write a note, preserving any human-owned frontmatter fields."""
    if path.exists():
        existing_fm, existing_body = read_note(path)
        for field in HUMAN_OWNED_FIELDS:
            existing_val = existing_fm.get(field)
            if existing_val is not None:
                frontmatter[field] = existing_val
        # If the human wrote body content, keep it
        if existing_body.strip():
            body = existing_body

    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_frontmatter(frontmatter) + "\n" + body
    path.write_text(content, encoding="utf-8")
```

- [ ] **Step 6: Run io tests — expect green**

```bash
pytest tests/test_obsidian_io.py -v
```

---

## Part B — Vault scripts

- [ ] **Step 7: Create `tests/test_obsidian_exports.py`**

```python
"""Tests for Obsidian vault export scripts."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from neural_search.eval.evidence import DatasetEvidence, QuerySpec
from neural_search.obsidian.io import read_note
from neural_search.obsidian.templates import (
    dataset_card_body,
    dataset_card_frontmatter,
    query_card_body,
    query_card_frontmatter,
)


class TestDatasetCardTemplate:
    def _make_ev(self) -> DatasetEvidence:
        return DatasetEvidence(
            record_id="dandi:000004",
            source="dandi",
            title="Human ephys",
            description="A human single-neuron dataset",
            species=["human"],
            modalities=["extracellular_ephys"],
            data_levels=["raw"],
            tasks=[],
            regions=[],
            license="CC-BY-4.0",
            doi=None,
            url="https://dandiarchive.org/dandiset/000004",
            raw_data_available=True,
            metadata_completeness=0.7,
        )

    def test_frontmatter_has_required_fields(self):
        ev = self._make_ev()
        fm = dataset_card_frontmatter(ev)
        assert fm["type"] == "dataset"
        assert fm["dataset_id"] == "dandi:000004"
        assert fm["source"] == "dandi"
        assert "dataset" in fm["tags"]

    def test_body_contains_title(self):
        ev = self._make_ev()
        body = dataset_card_body(ev)
        assert "Human ephys" in body


class TestQueryCardTemplate:
    def _make_spec(self) -> QuerySpec:
        return QuerySpec(
            query_id="q_0001",
            query_text="human fMRI reward",
            intent="META_ANALYSIS",
            scientific_goal="Find datasets for meta-analysis.",
            required_modalities=["fmri"],
            required_species=["human"],
            hard_negatives=["resting-state fMRI"],
        )

    def test_frontmatter_has_query_id(self):
        spec = self._make_spec()
        fm = query_card_frontmatter(spec)
        assert fm["query_id"] == "q_0001"
        assert fm["type"] == "query"

    def test_body_contains_hard_negatives(self):
        spec = self._make_spec()
        body = query_card_body(spec)
        assert "resting-state fMRI" in body


class TestInitVaultScript:
    def test_creates_vault_folders(self, tmp_path):
        vault = tmp_path / "vault"
        result = subprocess.run(
            [sys.executable, "scripts/obsidian/init_vault.py", "--vault", str(vault)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (vault / "00_Project").exists()
        assert (vault / "05_Annotations" / "Human Audits").exists()
        assert (vault / "99_Templates").exists()
```

- [ ] **Step 8: Create `scripts/obsidian/init_vault.py`**

```python
#!/usr/bin/env python3
"""Scaffold the Obsidian vault folder structure.

Usage:
    python scripts/obsidian/init_vault.py --vault obsidian_vault
"""
from __future__ import annotations

import argparse
from pathlib import Path

VAULT_FOLDERS = [
    "00_Project",
    "01_Rubrics",
    "02_Ontology",
    "03_Datasets",
    "04_Queries",
    "05_Annotations/Human Audits",
    "06_Evaluations",
    "07_Whitepaper",
    "08_Dashboards",
    "99_Templates",
]


def init_vault(vault: Path) -> None:
    vault.mkdir(parents=True, exist_ok=True)
    for folder in VAULT_FOLDERS:
        folder_path = vault / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        gitkeep = folder_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"Vault initialised at {vault} ({len(VAULT_FOLDERS)} folders)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()
    init_vault(args.vault)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Create `scripts/obsidian/export_dataset_cards.py`**

```python
#!/usr/bin/env python3
"""Export corpus records as Obsidian dataset cards.

Usage:
    python scripts/obsidian/export_dataset_cards.py \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import dataset_evidence_from_record
from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import dataset_card_body, dataset_card_frontmatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "03_Datasets"
    dest.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.corpus.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ev = dataset_evidence_from_record(record)
            # Sanitise filename
            safe_name = ev.record_id.replace(":", "_").replace("/", "_")
            note_path = dest / f"{safe_name}.md"
            safe_write_note(note_path, dataset_card_frontmatter(ev), dataset_card_body(ev))
            written += 1

    print(f"Exported {written} dataset cards → {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Create `scripts/obsidian/export_query_cards.py`**

```python
#!/usr/bin/env python3
"""Export benchmark queries as Obsidian query cards.

Usage:
    python scripts/obsidian/export_query_cards.py \
        --queries artifacts/benchmark_queries.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.query_decomposition import load_query_specs
from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import query_card_body, query_card_frontmatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "04_Queries"
    dest.mkdir(parents=True, exist_ok=True)
    specs = load_query_specs(args.queries)

    for spec in specs:
        note_path = dest / f"{spec.query_id}.md"
        safe_write_note(note_path, query_card_frontmatter(spec), query_card_body(spec))

    print(f"Exported {len(specs)} query cards → {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Create `scripts/obsidian/export_audit_queue.py`**

```python
#!/usr/bin/env python3
"""Export high-priority audit items to Obsidian annotation notes.

Usage:
    python scripts/obsidian/export_audit_queue.py \
        --audit-queue artifacts/eval/audit_queue.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.obsidian.io import safe_write_note
from neural_search.obsidian.templates import annotation_card_body, annotation_card_frontmatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-queue", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()

    dest = args.vault / "05_Annotations" / "Human Audits"
    dest.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.audit_queue.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            rid = row["record_id"]
            annotation_id = f"{qid}__{rid.replace(':', '_')}"

            pair_ev = row.get("pair_evidence", {})
            q = pair_ev.get("query", {})
            d = pair_ev.get("dataset", {})
            votes = []  # LF votes not stored in audit queue by default

            fm = annotation_card_frontmatter(
                annotation_id=annotation_id,
                query_id=qid,
                record_id=rid,
                label=row.get("label"),
                confidence=row.get("confidence"),
                source=row.get("source", "bronze"),
                audit_status="pending",
            )
            body = annotation_card_body(
                query_text=q.get("query_text", ""),
                scientific_goal=q.get("scientific_goal", ""),
                hard_negatives=q.get("hard_negatives", []),
                dataset_title=d.get("title", rid),
                dataset_desc=d.get("description"),
                lf_votes=votes,
                ensemble_label=row.get("label"),
                ensemble_confidence=row.get("confidence"),
            )
            note_path = dest / f"{annotation_id}.md"
            safe_write_note(note_path, fm, body)
            written += 1

    print(f"Exported {written} audit notes → {dest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12: Create `scripts/obsidian/import_audits.py`**

```python
#!/usr/bin/env python3
"""Import completed human audits from Obsidian into qrels_gold.jsonl.

Only notes with audit_status: done are imported.

Usage:
    python scripts/obsidian/import_audits.py \
        --vault obsidian_vault \
        --out artifacts/qrels_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.obsidian.io import read_note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    audit_dir = args.vault / "05_Annotations" / "Human Audits"
    if not audit_dir.exists():
        print(f"No audit directory found at {audit_dir}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load existing gold qrels to avoid duplicates
    existing: dict[tuple[str, str], dict] = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing[(r["query_id"], r["record_id"])] = r

    imported = skipped = 0
    for note_path in sorted(audit_dir.glob("*.md")):
        fm, _ = read_note(note_path)
        if fm.get("audit_status") != "done":
            skipped += 1
            continue
        if fm.get("label") is None:
            skipped += 1
            continue

        qid = fm.get("query_id", "")
        rid = fm.get("dataset_id", "")
        if not qid or not rid:
            skipped += 1
            continue

        existing[(qid, rid)] = {
            "query_id": qid,
            "record_id": rid,
            "label": int(fm["label"]),
            "confidence": float(fm.get("confidence") or 1.0),
            "source": "gold",
            "provenance": ["human_audit"],
            "hard_negative_triggered": False,
            "disagreement": 0.0,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        imported += 1

    with args.out.open("w", encoding="utf-8") as fh:
        for record in existing.values():
            fh.write(json.dumps(record) + "\n")

    print(f"Imported {imported} gold labels (skipped {skipped}) → {args.out}")
    print(f"Total gold qrels: {len(existing)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13: Create `scripts/obsidian/compile_vault.py`**

```python
#!/usr/bin/env python3
"""Run the full vault export pipeline in sequence.

Usage:
    python scripts/obsidian/compile_vault.py \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --audit-queue artifacts/eval/audit_queue.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--audit-queue", required=True)
    parser.add_argument("--vault", required=True)
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/obsidian/init_vault.py", "--vault", args.vault])
    run([py, "scripts/obsidian/export_dataset_cards.py",
         "--corpus", args.corpus, "--vault", args.vault])
    run([py, "scripts/obsidian/export_query_cards.py",
         "--queries", args.queries, "--vault", args.vault])
    run([py, "scripts/obsidian/export_audit_queue.py",
         "--audit-queue", args.audit_queue, "--vault", args.vault])
    print("Vault compilation complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 14: Create `scripts/obsidian/export_claim_registry.py`**

```python
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

    # Write to reports
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Claim registry written → {args.out}")

    # Also write to vault
    vault_path = args.vault / "07_Whitepaper" / "Claims Registry.md"
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    vault_path.write_text(content, encoding="utf-8")
    print(f"Claim registry written → {vault_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 15: Run obsidian export tests**

```bash
pytest tests/test_obsidian_io.py tests/test_obsidian_exports.py -v
```

Expected: all tests pass.

- [ ] **Step 16: Smoke-test the full vault pipeline on real data**

```bash
mkdir -p scripts/obsidian
python scripts/obsidian/init_vault.py --vault obsidian_vault
python scripts/obsidian/export_dataset_cards.py \
    --corpus data/corpus/normalized/combined_corpus.jsonl --vault obsidian_vault
python scripts/obsidian/export_query_cards.py \
    --queries artifacts/benchmark_queries.jsonl --vault obsidian_vault
python scripts/obsidian/export_audit_queue.py \
    --audit-queue artifacts/eval/audit_queue.jsonl --vault obsidian_vault
```

Verify:
```bash
ls obsidian_vault/03_Datasets/ | head -5
ls obsidian_vault/04_Queries/ | head -5
ls "obsidian_vault/05_Annotations/Human Audits/" | head -5
```

- [ ] **Step 17: Commit**

```bash
git add neural_search/obsidian/ scripts/obsidian/ \
    tests/test_obsidian_io.py tests/test_obsidian_exports.py \
    obsidian_vault/
git commit -m "feat(obsidian): vault templates, safe I/O, and all export/import scripts"
```
