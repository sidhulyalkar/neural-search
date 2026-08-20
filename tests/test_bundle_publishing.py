import json

import pytest

from neural_search.runtime import ARTIFACTS, ArtifactSpec, write_lineage
from neural_search.runtime.publishing import (
    add_release_index_entry,
    build_bundle_manifest,
    write_bundle_manifest,
)


def test_bundle_publisher_requires_stamped_artifact(tmp_path, monkeypatch):
    artifact_id = "publish_untracked"
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"payload")
    monkeypatch.setitem(
        ARTIFACTS,
        artifact_id,
        ArtifactSpec(
            id=artifact_id,
            path=str(path),
            kind="generated_local",
            description="Synthetic publish artifact",
            version="1",
            compatibility_group="test:v1",
        ),
    )

    with pytest.raises(ValueError, match="no lineage sidecar"):
        build_bundle_manifest(
            name="test",
            version="1.0.0",
            compatibility_group="test:v1",
            artifact_ids=[artifact_id],
            source_base_url="https://example.org/releases/1.0.0",
        )


def test_bundle_publisher_carries_parent_lineage(tmp_path, monkeypatch):
    parent_id = "publish_parent"
    child_id = "publish_child"
    parent_path = tmp_path / "parent.bin"
    child_path = tmp_path / "child.bin"
    parent_path.write_bytes(b"parent")
    child_path.write_bytes(b"child")
    monkeypatch.setitem(
        ARTIFACTS,
        parent_id,
        ArtifactSpec(
            id=parent_id,
            path=str(parent_path),
            kind="generated_local",
            description="Parent",
            version="1",
            compatibility_group="test:v1",
        ),
    )
    monkeypatch.setitem(
        ARTIFACTS,
        child_id,
        ArtifactSpec(
            id=child_id,
            path=str(child_path),
            kind="generated_local",
            description="Child",
            version="1",
            compatibility_group="test:v1",
            derived_from=(parent_id,),
        ),
    )
    parent = write_lineage(
        parent_path,
        artifact_id=parent_id,
        artifact_version="1",
        compatibility_group="test:v1",
    )
    write_lineage(
        child_path,
        artifact_id=child_id,
        artifact_version="1",
        compatibility_group="test:v1",
        derived_from={parent_id: parent.lineage_id},
    )

    bundle = build_bundle_manifest(
        name="test",
        version="1.0.0",
        compatibility_group="test:v1",
        artifact_ids=[child_id],
        source_base_url="https://example.org/releases/1.0.0",
    )

    assert bundle.artifacts[0].derived_from[parent_id] == parent.lineage_id
    assert bundle.artifacts[0].source_url.startswith("https://example.org/")


def test_bundle_publisher_rehashes_bytes_after_lineage_stamp(tmp_path, monkeypatch):
    artifact_id = "publish_changed"
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    monkeypatch.setitem(
        ARTIFACTS,
        artifact_id,
        ArtifactSpec(
            id=artifact_id,
            path=str(path),
            kind="generated_local",
            description="Changed after stamp",
            version="1",
            compatibility_group="test:v1",
        ),
    )
    write_lineage(
        path,
        artifact_id=artifact_id,
        artifact_version="1",
        compatibility_group="test:v1",
    )
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed after lineage was stamped"):
        build_bundle_manifest(
            name="test",
            version="1",
            compatibility_group="test:v1",
            artifact_ids=[artifact_id],
            source_base_url="https://example.org/releases/1",
        )


def test_release_index_pins_manifest_and_ref_is_append_only(tmp_path, monkeypatch):
    artifact_id = "publish_release"
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"payload")
    monkeypatch.setitem(
        ARTIFACTS,
        artifact_id,
        ArtifactSpec(
            id=artifact_id,
            path=str(path),
            kind="generated_local",
            description="Release artifact",
            version="1",
            compatibility_group="test:v1",
        ),
    )
    write_lineage(
        path,
        artifact_id=artifact_id,
        artifact_version="1",
        compatibility_group="test:v1",
    )
    bundle = build_bundle_manifest(
        name="release",
        version="1.0.0",
        compatibility_group="test:v1",
        artifact_ids=[artifact_id],
        source_base_url="https://example.org/releases/1.0.0",
    )
    manifest = tmp_path / "release.json"
    index = tmp_path / "index.json"
    write_bundle_manifest(manifest, bundle)

    first = add_release_index_entry(
        index_path=index,
        manifest_path=manifest,
        manifest_url="https://example.org/releases/release.json",
    )
    assert len(first.manifest_sha256 or "") == 64
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["bundles"][0]["manifest_sha256"] == first.manifest_sha256

    # Exact repeats are idempotent.
    repeated = add_release_index_entry(
        index_path=index,
        manifest_path=manifest,
        manifest_url="https://example.org/releases/release.json",
    )
    assert repeated == first

    with pytest.raises(ValueError, match="already exists"):
        add_release_index_entry(
            index_path=index,
            manifest_path=manifest,
            manifest_url="https://other.example.org/releases/release.json",
        )
