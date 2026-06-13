import json
from pathlib import Path

from neural_search.field_state.cli import main
from neural_search.field_state.memory.diff import compute_memory_diff
from neural_search.field_state.memory.review_overlay import OVERLAY_PATHS
from neural_search.field_state.obsidian.frontmatter import (
    compose_note,
    render_frontmatter,
)
from neural_search.field_state.obsidian.frontmatter import (
    parse_frontmatter as parse_note_frontmatter,
)
from neural_search.field_state.obsidian.paths import (
    ensure_vault_structure,
    field_state_id,
)
from neural_search.field_state.obsidian.templates import (
    HUMAN_BEGIN,
    HUMAN_END,
    render_benchmark_gap_note,
    render_claim_note,
    render_opportunity_note,
    render_snapshot_note,
)


def _init_and_export(tmp_path: Path) -> Path:
    vault = tmp_path / "test-vault"
    assert main(["--root", str(tmp_path), "init"]) == 0
    assert main(["--root", str(tmp_path), "report"]) == 0
    assert main(["--root", str(tmp_path), "export-obsidian", "--vault", str(vault)]) == 0
    return vault


def _opportunity_note(vault: Path) -> Path:
    return next(
        path
        for path in vault.rglob("*.md")
        if "Human qrels benchmark for dataset-method compatibility" in path.name
    )


def test_frontmatter_parse_render_and_compose_round_trip():
    rendered = render_frontmatter(
        {
            "type": "opportunity",
            "field_state_id": "opportunity:test",
            "tags": ["field-state", "opportunity"],
        }
    )
    frontmatter, body = parse_note_frontmatter(f"{rendered}\n# Body\n")

    assert frontmatter["type"] == "opportunity"
    assert frontmatter["tags"] == ["field-state", "opportunity"]
    assert body.strip() == "# Body"
    assert compose_note(frontmatter, body).startswith("---\n")


def test_template_rendering_for_required_note_types():
    claim = render_claim_note(
        title="Dense retrieval",
        claim_text="Dense retrieval helps.",
        claim_type="scientific_retrieval_claim",
        evidence_level="plausible",
        confidence=0.6,
        status="needs_validation",
        missing_tests=["human qrels"],
        supporting_artifacts=["reports/eval"],
        related_benchmark_gaps=["gap:qrels"],
        related_opportunities=["opp:qrels"],
    )
    gap = render_benchmark_gap_note(
        title="Human qrels",
        description="Need labels.",
        gap_type="benchmark_gap",
        why_it_matters="Metrics need labels.",
        required_validation=["label pool"],
        minimum_viable_benchmark="20 queries",
        related_claims=["claim:qrels"],
        related_opportunities=["opp:qrels"],
        source_artifacts=[],
    )
    opportunity = render_opportunity_note(
        title="Build qrels",
        hypothesis="Labels improve rigor.",
        opportunity_type="benchmark",
        rationale="Highest leverage.",
        novelty_score=7,
        feasibility_score=7,
        impact_score=10,
        uncertainty_reduction_score=10,
        personal_fit_score=9,
        risk_score=3,
        total_score=7.7,
        minimum_viable_experiment="Label a small pool.",
        next_action="Draft queries.",
        related_claims=[],
        related_benchmark_gaps=[],
        related_artifacts=[],
        codex_task_stub_or_link="export-task",
    )
    snapshot = render_snapshot_note(
        field="neuroscience_dataset_reuse",
        date="2026-06-10",
        summary="Snapshot",
        top_opportunities=["qrels"],
        weak_claims=["dense"],
        benchmark_gaps=["labels"],
        recommended_next_actions=["label"],
        snapshot_diff="none",
    )

    for rendered in [claim, gap, opportunity, snapshot]:
        assert "<!-- FIELDSTATE:BEGIN generated -->" in rendered
        assert HUMAN_BEGIN in rendered
        assert HUMAN_END in rendered


def test_folder_structure_and_field_state_id_are_stable(tmp_path: Path):
    vault = tmp_path / "vault"
    created = ensure_vault_structure(vault)

    assert (vault / "Field-State/00_Dashboard").exists()
    assert (vault / "Field-State/90_System").exists()
    assert created
    assert field_state_id("claim", "claim_dense_semantic_retrieval") == (
        "claim:dense-semantic-retrieval"
    )


def test_export_creates_notes_dashboard_and_memory_index(tmp_path: Path):
    vault = _init_and_export(tmp_path)
    index_path = vault / "Field-State/90_System/memory_index.json"

    assert (vault / "Field-State/00_Dashboard/Field-State Dashboard.md").exists()
    assert (vault / "Field-State/10_Snapshots/latest_snapshot.md").exists()
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    note_types = {entry["type"] for entry in index["notes"]}
    assert {"claim", "benchmark_gap", "opportunity", "field_snapshot"} <= note_types


def test_roundtrip_preserves_human_block_and_human_frontmatter(tmp_path: Path):
    vault = _init_and_export(tmp_path)
    note_path = _opportunity_note(vault)
    text = note_path.read_text(encoding="utf-8")
    frontmatter, body = parse_note_frontmatter(text)
    human_block = (
        f"{HUMAN_BEGIN}\n\n"
        "## Human review\n\n"
        "- [x] Reviewed\n"
        "- [x] Worth pursuing\n\n"
        "## Reviewer notes\n\n"
        "This is the next validation hinge.\n\n"
        f"{HUMAN_END}"
    )
    old_human_block = body[body.index(HUMAN_BEGIN) : body.index(HUMAN_END) + len(HUMAN_END)]
    body = body.replace(old_human_block, human_block)
    frontmatter["review_status"] = "reviewed"
    frontmatter["status"] = "active"
    frontmatter["human_priority"] = "high"
    note_path.write_text(compose_note(frontmatter, body), encoding="utf-8")

    assert main(["--root", str(tmp_path), "import-obsidian", "--vault", str(vault)]) == 0
    assert main(["--root", str(tmp_path), "report"]) == 0
    assert main(["--root", str(tmp_path), "export-obsidian", "--vault", str(vault)]) == 0

    updated = note_path.read_text(encoding="utf-8")
    updated_frontmatter, updated_body = parse_note_frontmatter(updated)
    assert human_block in updated_body
    assert updated_frontmatter["review_status"] == "reviewed"
    assert updated_frontmatter["status"] == "active"
    assert updated_frontmatter["human_priority"] == "high"

    overlay_path = tmp_path / OVERLAY_PATHS["opportunity"]
    overlays = [
        json.loads(line)
        for line in overlay_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record["review_status"] == "reviewed" for record in overlays)
    assert any(record["human_priority"] == "high" for record in overlays)

    report = tmp_path / "reports/field_state/top_opportunities.md"
    report_text = report.read_text(encoding="utf-8")
    assert "reviewed" in report_text
    assert "high" in report_text


def test_validation_success_and_failure_cases(tmp_path: Path):
    vault = _init_and_export(tmp_path)

    assert main(["memory-validate", "--vault", str(vault)]) == 0
    broken = vault / "Field-State/20_Claims/weak/Broken.md"
    broken.write_text("# Missing frontmatter\n", encoding="utf-8")

    assert main(["memory-validate", "--vault", str(vault)]) == 1
    report = vault / "Field-State/90_System/validation_report.md"
    assert "missing frontmatter" in report.read_text(encoding="utf-8")


def test_duplicate_field_state_id_is_detected(tmp_path: Path):
    vault = _init_and_export(tmp_path)
    first = next((vault / "Field-State/20_Claims/weak").glob("*.md"))
    duplicate = vault / "Field-State/20_Claims/weak/Duplicate.md"
    duplicate.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    assert main(["memory-validate", "--vault", str(vault)]) == 1
    report = vault / "Field-State/90_System/validation_report.md"
    assert "duplicate field_state_id" in report.read_text(encoding="utf-8")


def test_memory_diff_detects_human_edits(tmp_path: Path):
    vault = _init_and_export(tmp_path)
    note_path = _opportunity_note(vault)
    text = note_path.read_text(encoding="utf-8")
    note_path.write_text(
        text.replace("<!-- Add notes here. -->", "Human note added."),
        encoding="utf-8",
    )

    diff = compute_memory_diff(vault, "neuroscience_dataset_reuse")
    assert diff.human_edited
    assert main(["memory-diff", "--vault", str(vault)]) == 0
    assert (vault / "Field-State/90_System/memory_diff.md").exists()


def test_export_task_and_decision_add_commands(tmp_path: Path):
    vault = _init_and_export(tmp_path)

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "export-task",
                "--opportunity-id",
                "opp_human_qrels_benchmark",
                "--vault",
                str(vault),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "decision-add",
                "--vault",
                str(vault),
                "--title",
                "Prioritize qrels benchmark before paper ingestion",
            ]
        )
        == 0
    )

    task_notes = list((vault / "Field-State/70_Codex_Tasks/todo").glob("*.md"))
    decision_notes = list((vault / "Field-State/80_Decision_Log/decisions").glob("*.md"))
    assert task_notes
    assert decision_notes
    assert "Codex Task" in task_notes[0].read_text(encoding="utf-8")
    assert "Decision:" in decision_notes[0].read_text(encoding="utf-8")


def test_existing_field_state_commands_still_work(tmp_path: Path):
    assert main(["--root", str(tmp_path), "init"]) == 0
    assert main(["--root", str(tmp_path), "report"]) == 0
    assert main(["--root", str(tmp_path), "opportunities"]) == 0
    assert main(["--root", str(tmp_path), "snapshot"]) == 0
