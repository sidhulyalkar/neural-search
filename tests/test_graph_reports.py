from neural_search.graph import (
    build_graph_from_records,
    read_graph_json,
    write_graph_json,
)
from neural_search.graph.reports import generate_graph_reports, write_graph_reports
from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.5.0",
    )


def _graph():
    dataset = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("dandi", "000026"),
        source="dandi",
        source_id="000026",
        title="Mouse OFC reversal learning",
        tasks=[_label("task", "reversal_learning")],
        modalities=[_label("modality", "Neuropixels")],
        brain_regions=[_label("brain_region", "orbitofrontal_cortex")],
        missing_fields=["license"],
    )
    return build_graph_from_records([dataset], [])


def test_generate_graph_reports_returns_required_markdown_reports():
    reports = generate_graph_reports(_graph())

    assert set(reports) == {
        "graph_summary_report.md",
        "graph_scientific_coverage_report.md",
        "graph_requirement_report.md",
        "graph_linking_report.md",
        "graph_gap_report.md",
    }
    assert "Graph Summary Report" in reports["graph_summary_report.md"]
    assert "reversal_learning" in reports["graph_scientific_coverage_report.md"]
    assert "analysis_requires_modality" in reports["graph_requirement_report.md"]
    assert "license" in reports["graph_gap_report.md"]


def test_write_graph_reports_and_cli_inputs_roundtrip(tmp_path):
    graph = _graph()
    graph_path = write_graph_json(graph, tmp_path / "graph.json")
    assert read_graph_json(graph_path) == graph

    paths = write_graph_reports(graph, tmp_path / "reports")

    assert len(paths) == 5
    assert (tmp_path / "reports" / "graph_linking_report.md").exists()
