"""Tests for the Blue Brain Open Data ingestion adapter."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from neural_search.ingestion.bluebrain import (
    _bundle_title,
    _child_asset,
    _extract_extensions,
    _has_data_files,
    _infer_data_standards,
    _load_checkpoint_prefixes,
    _parse_path,
    _source_id,
    normalize_bluebrain_bundle,
    fetch_bluebrain,
)


# --- Unit tests: path parsing ---

def test_parse_path_experimental_morphology_mouse_cortex():
    meta = _parse_path("Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/")
    assert meta["category"] == "experimental"
    assert "morphology" in meta["modalities"]
    assert "mouse" in meta["species"]
    assert "neocortex" in meta["brain_regions"]


def test_parse_path_model_ephys_rat_hippocampus():
    meta = _parse_path("Model_Data/Electrophysiological_models/Hippocampus/")
    assert meta["category"] == "model"
    assert "patch_clamp" in meta["modalities"]
    assert "hippocampus" in meta["brain_regions"]


def test_parse_path_simulation_campaigns():
    meta = _parse_path("Simulation_data/Simulation_Campaigns/Rat/SSCx/")
    assert meta["category"] == "simulation"
    assert "simulation" in meta["modalities"]
    assert "rat" in meta["species"]
    assert "somatosensory_cortex" in meta["brain_regions"]


def test_parse_path_images_videos_brain_regions():
    meta = _parse_path("Images_Videos/Brain_Regions/Hippocampus/")
    assert meta["category"] == "image_video"
    assert "hippocampus" in meta["brain_regions"]


def test_parse_path_transcriptomics():
    meta = _parse_path("Experimental_Data/Transcriptomics/Whole_brain/aibs_10x_mouse_wholebrain/")
    assert meta["category"] == "experimental"
    assert "transcriptomics" in meta["modalities"]
    assert "mouse" in meta["species"]
    assert "whole_brain" in meta["brain_regions"]


def test_parse_path_embedded_species_token():
    meta = _parse_path("Experimental_Data/Neuron_density/rat_P14/LayerBoundariesProject/")
    assert "rat" in meta["species"]


# --- Unit tests: file utilities ---

def test_has_data_files_true_for_h5():
    files = [("Experimental_Data/morphologies/cell1.h5", 1024)]
    assert _has_data_files(files) is True


def test_has_data_files_true_for_swc():
    files = [("Model_Data/morph/neuron.swc", 512)]
    assert _has_data_files(files) is True


def test_has_data_files_false_for_metadata_only():
    files = [
        ("dir/.DS_Store", 100),
        ("dir/README.txt", 200),
        ("dir/COMMIT_EDITMSG", 50),
    ]
    # .txt and misc files should still count - they're not in _DATA_EXTS exclusions
    # Only .DS_Store is filtered
    assert _has_data_files(files) is False  # none of these are in _DATA_EXTS


def test_extract_extensions_dedupes():
    files = [
        ("a/b/cell1.h5", 100),
        ("a/b/cell2.h5", 200),
        ("a/b/morph.swc", 300),
    ]
    exts = _extract_extensions(files)
    assert exts == ["h5", "swc"]


def test_infer_data_standards_nwb():
    assert "NWB" in _infer_data_standards(["nwb"])


def test_infer_data_standards_hdf5():
    assert "HDF5" in _infer_data_standards(["h5", "hdf5"])


def test_infer_data_standards_multiple():
    stds = _infer_data_standards(["swc", "abf", "nwb"])
    assert "NWB" in stds
    assert "SWC" in stds
    assert "ABF" in stds


# --- Unit tests: bundle title ---

def test_bundle_title_meaningful():
    title = _bundle_title("Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/")
    assert "Mouse" in title or "mouse" in title.lower()
    assert "Cortex" in title or "cortex" in title.lower()
    assert "\u2014" not in title


def test_bundle_title_strips_generic_tokens():
    title = _bundle_title("Model_Data/Ion_Channel_models/Others/Striatum/")
    assert "Striatum" in title or "striatum" in title.lower()
    # "others" should be filtered out (it's in the skip list)
    assert "others" not in title.lower()


# --- Unit tests: normalization ---

def test_normalize_bluebrain_bundle_fields():
    prefix = "Experimental_Data/Electrophysiological_recordings/Single-cell_recordings/Mouse/Cortex/"
    files = [
        (prefix + "recording1.abf", 204800),
        (prefix + "recording2.h5", 1048576),
        (prefix + "metadata.json", 1024),
    ]
    rec = normalize_bluebrain_bundle(prefix, files)

    assert rec["source"] == "bluebrain"
    assert rec["license"] == "CC-BY-4.0"
    assert "patch_clamp" in rec["modalities"]
    assert "mouse" in rec["species"]
    assert "neocortex" in rec["brain_regions"]
    assert rec["has_raw_data"] is True
    assert "abf" in rec["formats"]
    assert "h5" in rec["formats"]
    assert rec["identifier"] == prefix.strip("/")
    assert rec["storage_url"].startswith("s3://openbluebrain/")
    assert rec["source_type"] == "canonical_dataset"
    assert rec["record_type"] == "dataset_bundle"
    assert len(rec["assets"]) == 3
    assert rec["assets"][0]["record_type"] == "child_asset"
    assert rec["metadata_json"]["child_assets"] == rec["assets"]


def test_normalize_bluebrain_bundle_model_record():
    prefix = "Model_Data/Electrophysiological_models/Hippocampus/"
    files = [(prefix + "model.hoc", 8192), (prefix + "model.mod", 4096)]
    rec = normalize_bluebrain_bundle(prefix, files)

    assert rec["data_category"] == "model"
    assert rec["has_processed_data"] is True
    assert "HOC/MOD" in rec["data_standards"]
    assert "electrophysiology_feature_extraction" in rec["analysis_affordances"]


def test_normalize_bluebrain_bundle_stable_source_id():
    prefix = "Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/"
    id1 = _source_id(prefix)
    id2 = _source_id(prefix)
    assert id1 == id2
    assert len(id1) == 16


def test_child_asset_metadata():
    prefix = "Model_Data/Electrophysiological_models/Hippocampus/"
    key = prefix + "mechanisms/na.mod"
    asset = _child_asset(prefix, key, 1234)
    assert asset["path"] == key
    assert asset["relative_path"] == "mechanisms/na.mod"
    assert asset["storage_url"] == "s3://openbluebrain/" + key
    assert asset["file_format"] == "mod"
    assert asset["data_standard"] == "HOC/MOD"
    assert asset["asset_type"] == "neuron_model_file"


def test_load_checkpoint_prefixes(tmp_path):
    checkpoint = tmp_path / "real_bluebrain.jsonl"
    records = [
        {"source": "bluebrain", "identifier": "Model_Data/Circuits/Rat"},
        {
            "source": "bluebrain",
            "metadata_json": {"s3_prefix": "Experimental_Data/Neuron_density/rat_P14/"},
        },
        {"source": "openneuro", "identifier": "ds000001"},
    ]
    checkpoint.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    prefixes = _load_checkpoint_prefixes(checkpoint, limit=10)

    assert prefixes == [
        "Model_Data/Circuits/Rat/",
        "Experimental_Data/Neuron_density/rat_P14/",
    ]


# --- Integration test: fetch with mocked S3 ---

def _make_s3_mock(prefix_tree: dict[str, tuple[list[str], list[tuple[str, int]]]]):
    """Build a mock S3 client where list_prefix returns from the given tree."""
    mock_client = MagicMock()
    paginator = MagicMock()
    mock_client.get_paginator.return_value = paginator

    def paginate_side_effect(Bucket, Prefix, Delimiter):
        subdirs, files = prefix_tree.get(Prefix, ([], []))
        page = {
            "CommonPrefixes": [{"Prefix": s} for s in subdirs],
            "Contents": [{"Key": k, "Size": sz} for k, sz in files],
        }
        return iter([page])

    paginator.paginate.side_effect = paginate_side_effect
    return mock_client


def test_fetch_bluebrain_returns_bundle_records():
    prefix_tree = {
        "Experimental_Data/": (
            ["Experimental_Data/Reconstructed_morphologies/"],
            [],
        ),
        "Experimental_Data/Reconstructed_morphologies/": (
            ["Experimental_Data/Reconstructed_morphologies/Categorized/"],
            [],
        ),
        "Experimental_Data/Reconstructed_morphologies/Categorized/": (
            ["Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/"],
            [],
        ),
        "Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/": (
            ["Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/"],
            [],
        ),
        "Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/": (
            ["Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/"],
            [],
        ),
        "Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/": (
            [],
            [
                ("Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/cell1.swc", 1024),
                ("Experimental_Data/Reconstructed_morphologies/Categorized/Neurons/Mouse/Cortex/cell2.swc", 2048),
            ],
        ),
        # All other roots return empty
        **{root: ([], []) for root in [
            "Model_Data/", "Simulation_data/", "Images_Videos/",
            "Brain_Systems/", "Circuits/", "Simulatable_Circuit/",
        ]},
    }

    mock_client = _make_s3_mock(prefix_tree)

    with patch("neural_search.ingestion.bluebrain._s3_client", return_value=mock_client):
        records = fetch_bluebrain(limit=50)

    assert len(records) == 1
    rec = records[0]
    assert rec["source"] == "bluebrain"
    assert "morphology" in rec["modalities"]
    assert "mouse" in rec["species"]
    assert "swc" in rec["formats"]
    assert rec["license"] == "CC-BY-4.0"
    assert rec["n_files"] == 2


def test_fetch_bluebrain_skips_git_dirs():
    prefix_tree = {
        "Experimental_Data/": (
            ["Experimental_Data/Models/"],
            [],
        ),
        "Experimental_Data/Models/": (
            [
                "Experimental_Data/Models/.git/",
                "Experimental_Data/Models/Rat/",
            ],
            [],
        ),
        "Experimental_Data/Models/.git/": (
            [],
            [("Experimental_Data/Models/.git/HEAD", 40)],
        ),
        "Experimental_Data/Models/Rat/": (
            [],
            [("Experimental_Data/Models/Rat/data.h5", 10240)],
        ),
        **{root: ([], []) for root in [
            "Model_Data/", "Simulation_data/", "Images_Videos/",
            "Brain_Systems/", "Circuits/", "Simulatable_Circuit/",
        ]},
    }

    mock_client = _make_s3_mock(prefix_tree)

    with patch("neural_search.ingestion.bluebrain._s3_client", return_value=mock_client):
        records = fetch_bluebrain(limit=50)

    # .git dir should be skipped; Rat/ should be indexed
    assert len(records) == 1
    assert "Rat" in records[0]["identifier"]


def test_fetch_bluebrain_respects_limit():
    # Build a tree with 5 leaf bundles
    prefix_tree: dict[str, tuple[list[str], list[tuple[str, int]]]] = {
        "Experimental_Data/": (
            [f"Experimental_Data/Morph{i}/" for i in range(5)],
            [],
        ),
    }
    for i in range(5):
        prefix_tree[f"Experimental_Data/Morph{i}/"] = (
            [],
            [(f"Experimental_Data/Morph{i}/cell.swc", 1024)],
        )
    prefix_tree.update({root: ([], []) for root in [
        "Model_Data/", "Simulation_data/", "Images_Videos/",
        "Brain_Systems/", "Circuits/", "Simulatable_Circuit/",
    ]})

    mock_client = _make_s3_mock(prefix_tree)

    with patch("neural_search.ingestion.bluebrain._s3_client", return_value=mock_client):
        records = fetch_bluebrain(limit=3)

    assert len(records) <= 3
