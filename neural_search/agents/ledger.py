"""Append-only run ledger for the agent orchestration loop.

Every agent-class run (see `artifacts/agents/registry.yaml`) appends exactly
one row to `artifacts/agents/ledger.jsonl` when it finishes -- never
overwritten, never rewritten, so the ledger is a durable history of what
ran, what it cost, and whether it passed its gate. This mirrors the "never
clobber a partial result" lesson from the literature-linking incidents
(see the `literature_source_expansion` project memory) applied to the
meta-level of tracking the agents themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "artifacts" / "agents" / "ledger.jsonl"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "agents" / "registry.yaml"

Outcome = Literal["ok", "regression", "error", "no_op"]


@dataclass
class LedgerEntry:
    agent: str
    outcome: Outcome
    started_at: str
    finished_at: str
    cost: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    eval_delta: dict[str, Any] = field(default_factory=dict)
    gate_result: str | None = None
    notes: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cost": self.cost,
            "findings": self.findings,
            "eval_delta": self.eval_delta,
            "gate_result": self.gate_result,
            "notes": self.notes,
        }


def append_ledger_entry(entry: LedgerEntry, ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append one run's outcome. Never overwrites -- see module docstring."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_row()) + "\n")


def read_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Read every recorded run, oldest first. `[]` if the ledger doesn't exist yet."""

    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_run_for_agent(
    agent: str, ledger_path: Path = DEFAULT_LEDGER_PATH
) -> dict[str, Any] | None:
    """Most recent ledger row for a given agent name, or `None` if it has never run."""

    matches = [row for row in read_ledger(ledger_path) if row["agent"] == agent]
    return matches[-1] if matches else None


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load `artifacts/agents/registry.yaml` as a plain dict."""

    import yaml

    if not registry_path.exists():
        return {"version": 1, "agents": []}
    with registry_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {"version": 1, "agents": []}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
