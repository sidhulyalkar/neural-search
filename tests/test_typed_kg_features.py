import json
from pathlib import Path

from neural_search.graph.typed_kg_features import TypedKGIndex, typed_kg_score


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _links_file(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "paper_dataset_links.jsonl",
        [
            {
                "dataset_record_id": "dandi:000003",
                "paper_openalex_id": "W1111111111",
                "match_method": "title_fuzzy_local",
                "confidence": 0.95,
            },
            {
                "dataset_record_id": "dandi:000004",
                "paper_openalex_id": "",
                "match_method": "not_found",
                "confidence": 0.0,
            },
            {
                "dataset_record_id": "dandi:000005",
                "paper_openalex_id": "W2222222222",
                "match_method": "doi_exact",
                "confidence": 1.0,
            },
        ],
    )


def _edges_file(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "finding_edges.jsonl",
        [
            {
                "edge_type": "supports",
                "paper_id_a": "paper:openalex:W1111111111",
                "paper_id_b": "paper:openalex:W9999999999",
            },
            {
                "edge_type": "supports",
                "paper_id_a": "paper:openalex:W8888888888",
                "paper_id_b": "paper:openalex:W1111111111",
            },
            {
                "edge_type": "contradicts",
                "paper_id_a": "paper:openalex:W1111111111",
                "paper_id_b": "paper:openalex:W7777777777",
            },
        ],
    )


def _qualified_consensus_file(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "consensus_summaries_qualified.jsonl",
        [
            {"region": "Hippocampus", "direction": "increase", "n_papers": 3, "facet_fields": ["species"]},
            {"region": "striatum", "direction": "decrease", "n_papers": 1, "facet_fields": ["species"]},
        ],
    )


def test_dataset_with_no_linked_paper_scores_zero(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(_links_file(tmp_path), _edges_file(tmp_path))
    assert typed_kg_score("dandi:000004", index) == 0.0


def test_unlinked_dataset_not_in_links_file_scores_zero(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(_links_file(tmp_path), _edges_file(tmp_path))
    assert typed_kg_score("dandi:999999", index) == 0.0


def test_linked_dataset_with_no_edges_scores_zero(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(_links_file(tmp_path), _edges_file(tmp_path))
    assert typed_kg_score("dandi:000005", index) == 0.0


def test_linked_dataset_with_supports_and_contradicts_scores_above_zero(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(_links_file(tmp_path), _edges_file(tmp_path))
    score = typed_kg_score("dandi:000003", index)
    # 2 supports * 0.03 + 1 contradicts * 0.02 = 0.08
    assert score == 0.08


def test_score_is_bounded_by_max(tmp_path: Path) -> None:
    many_edges = [
        {"edge_type": "supports", "paper_id_a": "paper:openalex:W1111111111", "paper_id_b": f"paper:openalex:W{i}"}
        for i in range(50)
    ]
    edges_path = _write_jsonl(tmp_path / "finding_edges.jsonl", many_edges)
    index = TypedKGIndex.from_files(_links_file(tmp_path), edges_path)
    score = typed_kg_score("dandi:000003", index)
    # capped at MAX_EDGES_COUNTED=5 supports * 0.03 = 0.15, well under MAX_TYPED_KG_SCORE
    assert score == 0.15


def test_qualified_bonus_applied_when_region_matches_consensus(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(
        _links_file(tmp_path),
        _edges_file(tmp_path),
        _qualified_consensus_file(tmp_path),
    )
    base_score = typed_kg_score(
        "dandi:000003", index, record={"brain_regions": ["hippocampus"]}, qualified=False
    )
    qualified_score = typed_kg_score(
        "dandi:000003", index, record={"brain_regions": ["hippocampus"]}, qualified=True
    )
    assert qualified_score == round(base_score + 0.05, 4)


def test_qualified_bonus_not_applied_when_region_unmatched(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(
        _links_file(tmp_path),
        _edges_file(tmp_path),
        _qualified_consensus_file(tmp_path),
    )
    score = typed_kg_score(
        "dandi:000003", index, record={"brain_regions": ["motor_cortex"]}, qualified=True
    )
    base_score = typed_kg_score("dandi:000003", index, qualified=False)
    assert score == base_score


def test_qualified_bonus_requires_min_papers_two(tmp_path: Path) -> None:
    """consensus_summaries_qualified rows with n_papers<2 are dropped at load time."""
    index = TypedKGIndex.from_files(
        _links_file(tmp_path),
        _edges_file(tmp_path),
        _qualified_consensus_file(tmp_path),
    )
    assert "striatum" not in index.region_to_qualified_consensus
    assert "hippocampus" in index.region_to_qualified_consensus


def test_missing_qualified_consensus_path_is_safe(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(_links_file(tmp_path), _edges_file(tmp_path))
    assert index.has_qualified_consensus is False
    score = typed_kg_score(
        "dandi:000003", index, record={"brain_regions": ["hippocampus"]}, qualified=True
    )
    # qualified=True but no consensus index loaded -> behaves like base score
    assert score == typed_kg_score("dandi:000003", index, qualified=False)


def test_missing_files_produce_empty_index(tmp_path: Path) -> None:
    index = TypedKGIndex.from_files(
        tmp_path / "does_not_exist_links.jsonl",
        tmp_path / "does_not_exist_edges.jsonl",
    )
    assert typed_kg_score("dandi:000003", index) == 0.0
