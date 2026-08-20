from neural_search.runtime import (
    ARTIFACTS,
    PROFILES,
    ArtifactSpec,
    artifact_status,
    build_reproducibility_manifest,
    profile_status,
)


def test_every_profile_references_registered_artifacts():
    for profile in PROFILES.values():
        referenced = (
            profile.required_artifacts
            + profile.recommended_artifacts
            + profile.produced_artifacts
        )
        assert set(referenced) <= set(ARTIFACTS), profile.name


def test_demo_profile_is_portable_from_source_checkout():
    status = profile_status("demo")

    assert status["ready"] is True
    assert status["source_checkout"] is True
    assert status["missing_modules"] == []
    assert all(item["usable"] for item in status["required_artifacts"])
    assert {
        item["kind"] for item in status["required_artifacts"]
    } == {"committed_fixture"}


def test_generated_artifacts_only_offer_repair_when_it_is_self_contained():
    corpus = artifact_status("full_corpus_v09")
    graph = artifact_status("production_graph")
    embeddings = artifact_status("dense_field_embeddings")

    assert corpus["kind"] == "generated_local"
    assert corpus["repair_command"] is None
    assert "source acquisition" in corpus["producer"]
    assert graph["repair_command"] == "python scripts/rebuild_full_corpus_graph.py"
    assert embeddings["repair_command"] == (
        "python scripts/recompute_embeddings.py --provider dense"
    )


def test_required_content_directory_cannot_false_green(tmp_path, monkeypatch):
    empty_directory = tmp_path / "empty-artifact"
    empty_directory.mkdir()
    artifact_id = "test_empty_generated_directory"
    monkeypatch.setitem(
        ARTIFACTS,
        artifact_id,
        ArtifactSpec(
            id=artifact_id,
            path=str(empty_directory),
            kind="generated_local",
            description="Synthetic empty artifact for readiness testing.",
            requires_content=True,
        ),
    )

    status = artifact_status(artifact_id)

    assert status["exists"] is True
    assert status["usable"] is False
    assert status["state"] == "empty_generated_asset"


def test_reproducibility_manifest_captures_portable_inputs():
    manifest = build_reproducibility_manifest("demo")

    assert manifest["schema_version"] == 1
    assert manifest["profile"] == "demo"
    assert manifest["profile_ready"] is True
    assert manifest["python"]
    assert {item["id"] for item in manifest["artifacts"]} == {
        "behavioral_ontology",
        "demo_datasets",
        "demo_papers",
    }
    assert all(item.get("sha256") for item in manifest["artifacts"])
