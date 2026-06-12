"""Whitepaper-ready validation status report."""

from __future__ import annotations

from pathlib import Path

from neural_search.field_state.eval_memory.eval_snapshot import build_eval_snapshot
from neural_search.field_state.store import WHITEPAPER_VALIDATION_REPORT, resolve_path


def render_whitepaper_validation_report(snapshot: dict[str, object]) -> str:
    """Render a concise whitepaper validation report."""
    qrels_status = snapshot.get("qrels_status", {})
    if not isinstance(qrels_status, dict):
        qrels_status = {}
    available = snapshot.get("available_eval_artifacts", [])
    if not isinstance(available, list):
        available = []
    return "\n".join(
        [
            "# Whitepaper Validation Report",
            "",
            "## Current Methodology Status",
            "",
            "Neural Search has engineering validation for corpus construction, indexing, and artifact generation. Scientific retrieval validation remains gated by human-reviewed qrels, adjudication, metric reports, source-skew diagnostics, and calibration.",
            "",
            "## Engineering Validation",
            "",
            "- Corpus and evaluation artifacts are generated locally.",
            "- Field-state reports and Obsidian review memory are reproducible from JSONL artifacts.",
            "",
            "## Scientific Retrieval Validation",
            "",
            f"- Qrels candidates exported: {qrels_status.get('candidates_exported', 0)}",
            f"- Qrels candidates reviewed: {qrels_status.get('candidates_reviewed', 0)}",
            f"- Qrels candidates adjudicated: {qrels_status.get('candidates_adjudicated', 0)}",
            f"- Candidates needing adjudication: {qrels_status.get('candidates_needing_adjudication', 0)}",
            "",
            "## Human Relevance Validation",
            "",
            "Gold retrieval claims should remain caveated until at least two human reviews are available where possible and disagreements have been adjudicated.",
            "",
            "## Future Usefulness Validation",
            "",
            "Future usefulness claims require longitudinal reuse proxies, content-validated affordances, and calibrated confidence estimates. Current field-state artifacts should treat those claims as hypotheses or preliminary evidence.",
            "",
            "## Available Supporting Artifacts",
            "",
            *(f"- `{artifact}`" for artifact in available),
            "",
            "## Publication-Grade Work Remaining",
            "",
            "- Complete human qrels review and adjudication.",
            "- Compute nDCG/MRR/Recall metrics from adjudicated qrels.",
            "- Track hard-negative violation rate.",
            "- Add source-skew and calibration reports.",
            "- Validate analysis affordances against human or content-level evidence.",
        ]
    )


def write_whitepaper_validation_report(root: Path | None = None) -> Path:
    """Write the whitepaper validation report."""
    snapshot = build_eval_snapshot(root)
    path = resolve_path(WHITEPAPER_VALIDATION_REPORT, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_whitepaper_validation_report(snapshot), encoding="utf-8")
    return path
