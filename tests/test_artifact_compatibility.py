from neural_search.runtime import (
    ARTIFACTS,
    ArtifactSpec,
    compatibility_status,
    write_lineage,
)


def test_declared_parent_can_be_nonlocal_without_false_incompatibility(tmp_path, monkeypatch):
    child_path = tmp_path / "child.bin"
    child_path.write_bytes(b"derived")
    parent_path = tmp_path / "missing-parent.bin"
    parent_id = "test_parent_nonlocal"
    child_id = "test_child_nonlocal"
    monkeypatch.setitem(
        ARTIFACTS,
        parent_id,
        ArtifactSpec(
            id=parent_id,
            path=str(parent_path),
            kind="generated_local",
            description="Synthetic parent",
            version="1",
        ),
    )
    monkeypatch.setitem(
        ARTIFACTS,
        child_id,
        ArtifactSpec(
            id=child_id,
            path=str(child_path),
            kind="generated_local",
            description="Synthetic child",
            version="1",
            derived_from=(parent_id,),
        ),
    )
    write_lineage(
        child_path,
        artifact_id=child_id,
        artifact_version="1",
        derived_from={parent_id: "sha256:remote-parent-lineage"},
    )

    result = compatibility_status([child_id, parent_id])

    assert result["compatible"] is True
    assert result["state"] == "unknown"
    assert any(
        item["reason"] == f"parent_not_local:{parent_id}"
        for item in result["unknown"]
    )


def test_installed_parent_lineage_mismatch_is_incompatible(tmp_path, monkeypatch):
    parent_path = tmp_path / "parent.bin"
    child_path = tmp_path / "child.bin"
    parent_path.write_bytes(b"parent")
    child_path.write_bytes(b"child")
    parent_id = "test_parent_mismatch"
    child_id = "test_child_mismatch"
    monkeypatch.setitem(
        ARTIFACTS,
        parent_id,
        ArtifactSpec(
            id=parent_id,
            path=str(parent_path),
            kind="generated_local",
            description="Synthetic parent",
            version="1",
        ),
    )
    monkeypatch.setitem(
        ARTIFACTS,
        child_id,
        ArtifactSpec(
            id=child_id,
            path=str(child_path),
            kind="generated_local",
            description="Synthetic child",
            version="1",
            derived_from=(parent_id,),
        ),
    )
    write_lineage(parent_path, artifact_id=parent_id, artifact_version="1")
    write_lineage(
        child_path,
        artifact_id=child_id,
        artifact_version="1",
        derived_from={parent_id: "sha256:not-the-installed-parent"},
    )

    result = compatibility_status([child_id, parent_id])

    assert result["compatible"] is False
    assert result["state"] == "incompatible"
    assert any(
        f"parent_lineage_mismatch:{parent_id}" in item["issues"]
        for item in result["incompatible"]
    )
