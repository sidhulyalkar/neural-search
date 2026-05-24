from neural_search.graph import (
    build_dataset_subgraph,
    build_graph_from_records,
    build_paper_subgraph,
    dataset_node_id,
    make_edge_id,
    make_node_id,
    paper_node_id,
    read_graph_json,
)
from neural_search.graph.build_graph import main as build_graph_main
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
    write_jsonl,
)
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
    UsabilityFlags,
)


def _label(label_type: str, label: str, confidence: float = 0.9) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=confidence,
        evidence_text=label,
        source_field=label_type,
        source_value=label,
        extractor_name="test",
        extractor_version="v0.5.0",
    )


def _dataset(**overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", "000026"),
        "source": "dandi",
        "source_id": "000026",
        "title": "Mouse OFC reversal learning electrophysiology",
        "description": "Choice, reward, and trial outcome events in OFC.",
        "species": [_label("species", "mouse")],
        "modalities": [_label("modality", "Neuropixels")],
        "brain_regions": [_label("brain_region", "orbitofrontal_cortex")],
        "tasks": [_label("task", "reversal_learning")],
        "behavioral_events": [
            _label("behavioral_event", "choice"),
            _label("behavioral_event", "reward"),
        ],
        "data_standards": [_label("data_standard", "NWB")],
        "file_formats": [_label("file_format", "nwb")],
        "linked_papers": [make_paper_id("openalex", "W123")],
        "usability_flags": UsabilityFlags(has_neural_data=True, has_behavior=True),
        "analysis_affordances": [
            AnalysisAffordance(
                analysis_id="q_learning_modeling",
                support_level="high",
                confidence=0.91,
                required_fields_present=["choice", "reward"],
                evidence=["choice", "reward"],
                detector_name="test_affordance",
                detector_version="v0.5.0",
            )
        ],
        "created_at": "2026-05-24T00:00:00+00:00",
        "extractor_version": "v0.5.0",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _paper(**overrides) -> NormalizedPaperRecord:
    payload = {
        "paper_id": make_paper_id("openalex", "W123"),
        "source": "openalex",
        "source_id": "W123",
        "title": "Reversal learning in orbitofrontal cortex",
        "abstract": "Neuropixels recordings during reversal learning.",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "linked_datasets": [make_dataset_id("dandi", "000026")],
        "extracted_labels": [
            _label("task", "reversal_learning", 0.85),
            _label("modality", "Neuropixels", 0.82),
            _label("brain_region", "orbitofrontal_cortex", 0.88),
        ],
        "created_at": "2026-05-24T00:00:00+00:00",
        "extractor_version": "v0.5.0",
    }
    payload.update(overrides)
    return NormalizedPaperRecord(**payload)


def test_dataset_subgraph_contains_expected_concept_edges():
    dataset = _dataset()
    graph = build_dataset_subgraph(dataset)
    dataset_id = dataset_node_id(dataset)
    task_id = make_node_id("task", "reversal_learning")
    analysis_id = make_node_id("analysis_affordance", "q_learning_modeling")
    paper_id = paper_node_id(make_paper_id("openalex", "W123"))

    assert dataset_id in graph.nodes
    assert task_id in graph.nodes
    assert analysis_id in graph.nodes
    assert paper_id in graph.nodes
    assert make_edge_id(dataset_id, "dataset_has_task", task_id) in graph.edges
    assert make_edge_id(dataset_id, "dataset_supports_analysis", analysis_id) in graph.edges
    assert make_edge_id(paper_id, "paper_mentions_dataset", dataset_id) in graph.edges
    assert graph.edges[make_edge_id(dataset_id, "dataset_has_task", task_id)].evidence


def test_paper_subgraph_contains_authors_labels_and_dataset_links():
    paper = _paper()
    graph = build_paper_subgraph(paper)
    paper_id = paper_node_id(paper)
    author_id = make_node_id("author", "ada_lovelace")
    dataset_id = dataset_node_id(make_dataset_id("dandi", "000026"))
    task_id = make_node_id("task", "reversal_learning")

    assert author_id in graph.nodes
    assert make_edge_id(paper_id, "paper_has_author", author_id) in graph.edges
    assert make_edge_id(paper_id, "paper_uses_dataset", dataset_id) in graph.edges
    assert make_edge_id(paper_id, "paper_studies_task", task_id) in graph.edges


def test_build_graph_merges_placeholders_and_filters_low_confidence_labels():
    low_confidence_region = _label("brain_region", "hippocampus", confidence=0.1)
    dataset = _dataset(brain_regions=[low_confidence_region])
    paper = _paper()

    graph = build_graph_from_records([dataset], [paper], min_confidence=0.5)
    dataset_id = dataset_node_id(dataset)
    paper_id = paper_node_id(paper)
    low_region_id = make_node_id("brain_region", "hippocampus")

    assert graph.nodes[dataset_id].label == dataset.title
    assert graph.nodes[paper_id].label == paper.title
    assert low_region_id not in graph.nodes
    assert graph.metadata["dataset_count"] == 1
    assert graph.metadata["paper_count"] == 1


def test_build_graph_cli_writes_readable_graph(tmp_path):
    datasets_path = write_jsonl([_dataset()], tmp_path / "datasets.jsonl")
    papers_path = write_jsonl([_paper()], tmp_path / "papers.jsonl")
    output_path = tmp_path / "graph.json"

    exit_code = build_graph_main(
        [
            "--datasets",
            str(datasets_path),
            "--papers",
            str(papers_path),
            "--out",
            str(output_path),
        ]
    )

    graph = read_graph_json(output_path)

    assert exit_code == 0
    assert graph.metadata["record_count"] == 2
    assert dataset_node_id(make_dataset_id("dandi", "000026")) in graph.nodes
