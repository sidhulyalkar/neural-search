from neural_search.graph import (
    build_graph_from_records,
    dataset_node_id,
    explain_connection,
    find_datasets_for_analysis,
    find_datasets_for_paper,
    find_datasets_for_task,
    find_datasets_with_constraints,
    find_nodes_by_label,
    find_nodes_by_type,
    find_papers_for_dataset,
    find_paths,
    get_neighbors,
    get_node,
    make_node_id,
    paper_node_id,
    rank_related_datasets,
    rank_related_papers,
)
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
)
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)


def _label(label_type: str, label: str, confidence: float = 0.9) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=confidence,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.5.0",
    )


def _affordance(analysis_id: str) -> AnalysisAffordance:
    return AnalysisAffordance(
        analysis_id=analysis_id,
        support_level="high",
        confidence=0.9,
        evidence=[analysis_id],
        detector_name="test",
        detector_version="v0.5.0",
    )


def _dataset(source_id: str, task: str, modality: str, linked_paper: str) -> NormalizedDatasetRecord:
    return NormalizedDatasetRecord(
        dataset_id=make_dataset_id("dandi", source_id),
        source="dandi",
        source_id=source_id,
        title=f"{task} {modality} dataset",
        species=[_label("species", "mouse")],
        modalities=[_label("modality", modality)],
        brain_regions=[_label("brain_region", "orbitofrontal_cortex")],
        tasks=[_label("task", task)],
        behavioral_events=[_label("behavioral_event", "choice")],
        analysis_affordances=[_affordance("q_learning_modeling")],
        linked_papers=[make_paper_id("openalex", linked_paper)],
    )


def _paper(source_id: str, dataset_id: str, task: str) -> NormalizedPaperRecord:
    return NormalizedPaperRecord(
        paper_id=make_paper_id("openalex", source_id),
        source="openalex",
        source_id=source_id,
        title=f"{task} paper",
        authors=["Demo Author"],
        linked_datasets=[make_dataset_id("dandi", dataset_id)],
        extracted_labels=[_label("task", task), _label("modality", "Neuropixels")],
    )


def _graph():
    datasets = [
        _dataset("000026", "reversal_learning", "Neuropixels", "W123"),
        _dataset("000027", "reversal_learning", "Neuropixels", "W123"),
        _dataset("000028", "visual_decision_making", "calcium_imaging", "W999"),
    ]
    papers = [
        _paper("W123", "000026", "reversal_learning"),
        _paper("W999", "000028", "visual_decision_making"),
    ]
    return build_graph_from_records(datasets, papers)


def test_basic_lookup_helpers_find_nodes_and_neighbors():
    graph = _graph()
    dataset_id = dataset_node_id(make_dataset_id("dandi", "000026"))

    assert get_node(graph, dataset_id) is not None
    assert find_nodes_by_type(graph, "dataset")
    assert find_nodes_by_label(graph, "Neuropixels")
    assert any(node.node_type == "task" for node in get_neighbors(graph, dataset_id, direction="out"))


def test_domain_specific_dataset_and_paper_queries():
    graph = _graph()
    dataset_id = dataset_node_id(make_dataset_id("dandi", "000026"))
    paper_id = paper_node_id(make_paper_id("openalex", "W123"))

    assert {node.node_id for node in find_datasets_for_task(graph, "reversal_learning")} >= {
        dataset_id
    }
    assert dataset_id in {
        node.node_id for node in find_datasets_for_analysis(graph, "q_learning_modeling")
    }
    assert paper_id in {node.node_id for node in find_papers_for_dataset(graph, dataset_id)}
    assert dataset_id in {node.node_id for node in find_datasets_for_paper(graph, paper_id)}


def test_constraint_query_respects_required_and_excluded_concepts():
    graph = _graph()
    results = find_datasets_with_constraints(
        graph,
        required_tasks=["reversal_learning"],
        required_modalities=["Neuropixels"],
        excluded_modalities=["calcium_imaging"],
    )

    assert {node.node_id for node in results} == {
        dataset_node_id(make_dataset_id("dandi", "000026")),
        dataset_node_id(make_dataset_id("dandi", "000027")),
    }


def test_paths_and_explanations_connect_dataset_to_paper_via_task():
    graph = _graph()
    dataset_id = dataset_node_id(make_dataset_id("dandi", "000027"))
    task_id = make_node_id("task", "reversal_learning")
    paper_id = paper_node_id(make_paper_id("openalex", "W123"))

    paths = find_paths(graph, dataset_id, paper_id, max_depth=3)
    explanation = explain_connection(graph, dataset_id, paper_id, max_depth=3)

    assert [dataset_id, task_id, paper_id] in [path for path in paths if len(path) == 3]
    assert explanation["connected"] is True
    assert explanation["steps"]


def test_related_dataset_and_paper_ranking_uses_shared_graph_context():
    graph = _graph()
    dataset_id = dataset_node_id(make_dataset_id("dandi", "000026"))
    paper_id = paper_node_id(make_paper_id("openalex", "W123"))

    related_datasets = rank_related_datasets(graph, dataset_id)
    related_papers = rank_related_papers(graph, paper_id)

    assert related_datasets[0].node_id == dataset_node_id(make_dataset_id("dandi", "000027"))
    assert related_datasets[0].score > 0
    assert related_papers
    assert related_papers[0].node_id == paper_node_id(make_paper_id("openalex", "W999"))
