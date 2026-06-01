from pathlib import Path

from neural_search.corpus.convert_demo_seed import convert_demo_seed
from neural_search.graph.builder import build_graph_from_records, split_records
from neural_search.normalized import load_normalized_records
from neural_search.schemas import (
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)

ROOT = Path(__file__).resolve().parents[1]


def test_demo_seed_converter_writes_full_deterministic_normalized_corpus(tmp_path):
    paths = convert_demo_seed(
        ROOT / "data" / "seed" / "demo_datasets.yaml",
        ROOT / "data" / "seed" / "demo_papers.yaml",
        tmp_path,
    )
    first_dataset_text = paths["datasets"].read_text(encoding="utf-8")
    first_paper_text = paths["papers"].read_text(encoding="utf-8")

    paths = convert_demo_seed(
        ROOT / "data" / "seed" / "demo_datasets.yaml",
        ROOT / "data" / "seed" / "demo_papers.yaml",
        tmp_path,
    )

    assert paths["datasets"].read_text(encoding="utf-8") == first_dataset_text
    assert paths["papers"].read_text(encoding="utf-8") == first_paper_text

    datasets, papers = split_records(load_normalized_records(paths["records"]))
    assert len(datasets) > 2
    assert len(papers) > 2
    assert all(isinstance(record, NormalizedDatasetRecord) for record in datasets)
    assert all(isinstance(record, NormalizedPaperRecord) for record in papers)

    reversal = next(record for record in datasets if record.source_id == "DEMO_REVERSAL_EPHYS")
    assert {label.label for label in reversal.modalities} >= {"extracellular_ephys", "spikes"}
    assert {label.label for label in reversal.behavioral_events} >= {"choice", "reward"}
    assert reversal.linked_papers == ["paper:demo:PAPER_REVERSAL_EPHYS"]
    assert reversal.analysis_affordances


def test_graph_artifact_can_be_built_from_normalized_demo_records(tmp_path):
    paths = convert_demo_seed(
        ROOT / "data" / "seed" / "demo_datasets.yaml",
        ROOT / "data" / "seed" / "demo_papers.yaml",
        tmp_path,
    )
    datasets, papers = split_records(load_normalized_records(paths["records"]))

    graph = build_graph_from_records(datasets, papers)

    node_types = {node.node_type for node in graph.nodes.values()}
    assert {"dataset", "paper", "task", "modality", "species", "analysis_affordance"} <= node_types
    assert graph.metadata["dataset_count"] == len(datasets)
    assert graph.metadata["paper_count"] == len(papers)
