import json

from neural_search.graph import build_graph_from_records, write_graph_json
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
    write_jsonl,
)
from neural_search.reports.scientific_readiness import (
    build_scientific_readiness_report,
    render_scientific_readiness_markdown,
    write_scientific_readiness_reports,
)
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.8.0",
    )


def test_scientific_readiness_report_summarizes_corpus_graph_and_warnings(tmp_path):
    dataset = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("demo", "MAC"),
        source="demo",
        source_id="MAC",
        title="Macaque visual cortex Neuropixels",
        species=[_label("species", "macaque")],
        modalities=[_label("modality", "neuropixels")],
        data_standards=[_label("data_standard", "NWB")],
        linked_papers=[make_paper_id("demo", "P1")],
    )
    paper = NormalizedPaperRecord(
        paper_id=make_paper_id("demo", "P1"),
        source="demo",
        source_id="P1",
        title="Macaque paper",
        linked_datasets=[dataset.dataset_id],
    )
    corpus_path = write_jsonl([dataset, paper], tmp_path / "records.jsonl")
    graph_path = write_graph_json(
        build_graph_from_records([dataset], [paper]),
        tmp_path / "graph.json",
    )

    report = build_scientific_readiness_report(
        corpus_path=corpus_path,
        graph_path=graph_path,
    )

    assert report["corpus"]["dataset_records"] == 1
    assert report["corpus"]["paper_records"] == 1
    assert report["corpus"]["canonical_species_counts"] == {"macaque": 1}
    assert report["graph"]["species_context_edge_count"] > 0
    assert "Corpus is still small" in " ".join(report["warnings"])

    markdown = render_scientific_readiness_markdown(report)
    assert "# Scientific Readiness Report" in markdown
    assert "Species context edges" in markdown

    paths = write_scientific_readiness_reports(report, tmp_path / "reports")
    assert json.loads((tmp_path / "reports" / "scientific_readiness_report.json").read_text(
        encoding="utf-8"
    ))["version"] == "v0.8.0"
    assert paths["markdown"].endswith("scientific_readiness_report.md")
