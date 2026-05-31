import json
from collections import Counter
from pathlib import Path

from neural_search.graph import (
    build_graph_from_records,
    find_datasets_for_experimental_design,
    read_graph_json,
)
from neural_search.graph.build_graph import main as build_graph_main
from neural_search.graph.reports import main as reports_main
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord, NormalizedPaperRecord

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graph"


def _summary(graph):
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_counts": dict(
            sorted(Counter(node.node_type for node in graph.nodes.values()).items())
        ),
        "edge_counts": dict(
            sorted(Counter(edge.edge_type for edge in graph.edges.values()).items())
        ),
    }


def test_graph_fixtures_build_expected_summary():
    datasets = load_normalized_records(FIXTURE_DIR / "normalized_datasets.jsonl")
    papers = load_normalized_records(FIXTURE_DIR / "normalized_papers.jsonl")
    expected = json.loads((FIXTURE_DIR / "expected_graph_summary.json").read_text())

    graph = build_graph_from_records(
        [record for record in datasets if isinstance(record, NormalizedDatasetRecord)],
        [record for record in papers if isinstance(record, NormalizedPaperRecord)],
    )

    assert _summary(graph) == expected


def test_graph_quality_gate_commands_work_on_fixtures(tmp_path):
    graph_path = tmp_path / "fixture_graph.json"
    reports_dir = tmp_path / "graph_reports"

    build_exit = build_graph_main(
        [
            "--datasets",
            str(FIXTURE_DIR / "normalized_datasets.jsonl"),
            "--papers",
            str(FIXTURE_DIR / "normalized_papers.jsonl"),
            "--out",
            str(graph_path),
        ]
    )
    reports_exit = reports_main(["--graph", str(graph_path), "--out", str(reports_dir)])

    graph = read_graph_json(graph_path)

    assert build_exit == 0
    assert reports_exit == 0
    assert _summary(graph)["node_count"] == 146
    assert (reports_dir / "graph_summary_report.md").exists()


def test_fixture_experimental_design_seed_matches_reversal_dataset():
    datasets = load_normalized_records(FIXTURE_DIR / "normalized_datasets.jsonl")
    graph = build_graph_from_records(
        [record for record in datasets if isinstance(record, NormalizedDatasetRecord)],
        [],
    )

    matches = find_datasets_for_experimental_design(
        graph,
        "fixture_q_learning_design",
        path=FIXTURE_DIR / "experimental_design_seeds.yaml",
        min_score=0.8,
    )

    assert matches
    assert matches[0].dataset_id == "node:dataset:dandi:000026"
