"""Evaluation snapshot report for field-state validation memory."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from neural_search.field_state.eval_memory.qrels_schema import (
    AdjudicatedQrel,
    QrelsCandidate,
    QrelsReview,
)
from neural_search.field_state.store import (
    ADJUDICATED_QRELS_PATH,
    EVAL_SNAPSHOT_PATH,
    EVAL_SNAPSHOT_REPORT,
    QRELS_AGREEMENT_PATH,
    QRELS_CANDIDATES_PATH,
    QRELS_REVIEWS_PATH,
    read_jsonl,
    resolve_path,
)


def _load_json(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = resolve_path(path, root)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _available_eval_artifacts(root: Path | None = None) -> list[str]:
    base = root or Path.cwd()
    paths = [
        base / "artifacts/qrels.jsonl",
        base / "artifacts/field_state/adjudicated_qrels.jsonl",
        base / "artifacts/field_state/qrels_agreement.json",
    ]
    eval_dir = base / "reports/eval"
    if eval_dir.exists():
        paths.extend(sorted(eval_dir.glob("*.json")))
        paths.extend(sorted(eval_dir.glob("*.md")))
    return [str(path.relative_to(base)) for path in paths if path.exists()]


def build_eval_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Build an evaluation snapshot from available artifacts."""
    candidates = read_jsonl(QRELS_CANDIDATES_PATH, QrelsCandidate, root)
    reviews = read_jsonl(QRELS_REVIEWS_PATH, QrelsReview, root)
    adjudicated = read_jsonl(ADJUDICATED_QRELS_PATH, AdjudicatedQrel, root)
    agreement = _load_json(QRELS_AGREEMENT_PATH, root)
    reviewed_candidate_ids = {
        review.candidate_id for review in reviews if review.relevance_score is not None
    }
    needs_adjudication = [
        qrel for qrel in adjudicated if qrel.adjudication_status == "needs_adjudication"
    ]
    relevance_distribution = Counter(str(qrel.final_relevance_score) for qrel in adjudicated)
    usefulness_distribution = Counter(
        str(qrel.final_usefulness_score)
        for qrel in adjudicated
        if qrel.final_usefulness_score is not None
    )
    hard_negative_violations = sum(
        1 for qrel in adjudicated if qrel.final_hard_negative_violation
    )
    return {
        "schema_version": "0.3",
        "qrels_status": {
            "candidates_exported": len(candidates),
            "candidates_reviewed": len(reviewed_candidate_ids),
            "candidates_adjudicated": len(adjudicated),
            "candidates_needing_adjudication": len(needs_adjudication),
        },
        "label_distribution": {
            "relevance": dict(sorted(relevance_distribution.items())),
            "usefulness": dict(sorted(usefulness_distribution.items())),
            "hard_negative_violations": hard_negative_violations,
        },
        "agreement": agreement,
        "available_eval_artifacts": _available_eval_artifacts(root),
        "claim_evidence_update_suggestions": [
            "Strengthen qrels-related claims only after adjudicated labels and metric reports exist.",
            "Keep dense retrieval, hard-negative, and affordance claims caveated until qrels-backed metrics are generated.",
        ],
        "recommended_next_actions": [
            "Review unreviewed qrels candidates.",
            "Adjudicate disagreements.",
            "Run retrieval metrics against adjudicated qrels.",
            "Add source-skew and calibration reports before publication-grade claims.",
        ],
    }


def render_eval_snapshot(snapshot: dict[str, Any]) -> str:
    """Render eval snapshot Markdown."""
    status = snapshot["qrels_status"]
    labels = snapshot["label_distribution"]
    agreement = snapshot.get("agreement") or {}
    lines = [
        "# Field-State Evaluation Snapshot",
        "",
        "## Qrels Status",
        "",
        f"- Candidates exported: {status['candidates_exported']}",
        f"- Candidates reviewed: {status['candidates_reviewed']}",
        f"- Candidates adjudicated: {status['candidates_adjudicated']}",
        f"- Candidates needing adjudication: {status['candidates_needing_adjudication']}",
        "",
        "## Label Distribution",
        "",
        "- Relevance:",
    ]
    for score, count in labels["relevance"].items():
        lines.append(f"  - {score}: {count}")
    lines.append("- Usefulness:")
    for score, count in labels["usefulness"].items():
        lines.append(f"  - {score}: {count}")
    lines.append(f"- Hard-negative violations: {labels['hard_negative_violations']}")
    lines.extend(
        [
            "",
            "## Agreement",
            "",
            f"- Exact agreement rate: {agreement.get('exact_agreement_rate')}",
            f"- Disagreement count: {agreement.get('disagreement_count')}",
            "",
            "## Available Evaluation Artifacts",
            "",
        ]
    )
    lines.extend(
        f"- `{artifact}`" for artifact in snapshot.get("available_eval_artifacts", [])
    )
    if not snapshot.get("available_eval_artifacts"):
        lines.append("- none")
    lines.extend(["", "## Claim Evidence Update Suggestions", ""])
    lines.extend(f"- {item}" for item in snapshot["claim_evidence_update_suggestions"])
    lines.extend(["", "## Recommended Next Actions", ""])
    lines.extend(f"- {item}" for item in snapshot["recommended_next_actions"])
    return "\n".join(lines)


def write_eval_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Write eval snapshot JSON and Markdown."""
    snapshot = build_eval_snapshot(root)
    json_path = resolve_path(EVAL_SNAPSHOT_PATH, root)
    md_path = resolve_path(EVAL_SNAPSHOT_REPORT, root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_eval_snapshot(snapshot), encoding="utf-8")
    return snapshot
