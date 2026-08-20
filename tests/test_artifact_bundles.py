import hashlib
from pathlib import Path

import pytest

from neural_search.runtime import (
    ArtifactBundle,
    BundleArtifact,
    fetch_bundle,
    load_bundle_manifest,
    make_lineage_id,
    resolve_locked_artifact,
    verify_locked_artifacts,
)


def _artifact(
    source: Path,
    *,
    artifact_id: str = "example",
    relative_path: str = "data/example.bin",
    sha256: str | None = None,
) -> BundleArtifact:
    content = source.read_bytes()
    digest = sha256 or hashlib.sha256(content).hexdigest()
    return BundleArtifact(
        artifact_id=artifact_id,
        relative_path=relative_path,
        source_url=source.as_uri(),
        sha256=digest,
        size_bytes=len(content),
        artifact_version="1.0",
        lineage_id=make_lineage_id(artifact_id, "1.0", digest),
    )


def test_fetch_bundle_verifies_checksum_and_reuses_cache(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"neural-search-bundle")
    bundle = ArtifactBundle(
        name="test-bundle",
        version="1.0.0",
        compatibility_group="test:v1",
        artifacts=(_artifact(source),),
    )
    cache = tmp_path / "cache"
    lock = tmp_path / "artifact-lock.json"

    first = fetch_bundle(
        bundle,
        cache_dir=cache,
        lock_path=lock,
        allow_local_files=True,
    )
    assert first["installed"][0]["reused"] is False
    installed = Path(first["installed"][0]["path"])
    assert installed.read_bytes() == source.read_bytes()

    second = fetch_bundle(
        bundle,
        cache_dir=cache,
        lock_path=lock,
        allow_local_files=True,
    )
    assert second["installed"][0]["reused"] is True
    assert verify_locked_artifacts(lock_path=lock)["valid"] is True
    assert resolve_locked_artifact("example", lock_path=lock) == installed

    installed.chmod(0o644)
    installed.write_bytes(b"tampered")
    assert resolve_locked_artifact("example", lock_path=lock) is None
    report = verify_locked_artifacts(lock_path=lock)
    assert report["valid"] is False
    assert report["artifacts"][0]["actual_sha256"] != report["artifacts"][0]["expected_sha256"]


def test_bundle_rejects_path_traversal(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    bundle = ArtifactBundle(
        name="unsafe",
        version="1",
        compatibility_group="test:v1",
        artifacts=(_artifact(source, relative_path="../escape.bin"),),
    )

    with pytest.raises(ValueError, match="Unsafe artifact relative_path"):
        bundle.validate()


def test_bundle_rejects_wrong_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"correct")
    bundle = ArtifactBundle(
        name="bad-checksum",
        version="1",
        compatibility_group="test:v1",
        artifacts=(_artifact(source, sha256="0" * 64),),
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_bundle(
            bundle,
            cache_dir=tmp_path / "cache",
            lock_path=tmp_path / "lock.json",
            allow_local_files=True,
        )


def test_bundle_rejects_lineage_id_not_derived_from_manifest_content(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    artifact = _artifact(source)
    inconsistent = BundleArtifact(
        **{
            **artifact.__dict__,
            "lineage_id": "sha256:" + "0" * 64,
        }
    )
    bundle = ArtifactBundle(
        name="bad-lineage",
        version="1",
        compatibility_group="test:v1",
        artifacts=(inconsistent,),
    )

    with pytest.raises(ValueError, match="Lineage identity mismatch"):
        bundle.validate()


def test_manifest_can_be_pinned_by_its_own_sha256(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    bundle = ArtifactBundle(
        name="pinned",
        version="1",
        compatibility_group="test:v1",
        artifacts=(_artifact(source),),
    )
    manifest = tmp_path / "manifest.json"
    import json

    manifest.write_text(json.dumps(bundle.to_dict(), sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    loaded = load_bundle_manifest(manifest, expected_sha256=digest)
    assert loaded.ref == "pinned@1"

    with pytest.raises(ValueError, match="Manifest checksum mismatch"):
        load_bundle_manifest(manifest, expected_sha256="0" * 64)


def test_reinstalling_bundle_replaces_stale_artifact_pins(tmp_path):
    source_a = tmp_path / "a.bin"
    source_b = tmp_path / "b.bin"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    lock = tmp_path / "lock.json"
    cache = tmp_path / "cache"

    first = ArtifactBundle(
        name="researcher",
        version="1",
        compatibility_group="test:v1",
        artifacts=(
            _artifact(source_a, artifact_id="a", relative_path="a.bin"),
            _artifact(source_b, artifact_id="b", relative_path="b.bin"),
        ),
    )
    second = ArtifactBundle(
        name="researcher",
        version="2",
        compatibility_group="test:v1",
        artifacts=(_artifact(source_a, artifact_id="a", relative_path="a.bin"),),
    )
    fetch_bundle(first, cache_dir=cache, lock_path=lock, allow_local_files=True)
    fetch_bundle(second, cache_dir=cache, lock_path=lock, allow_local_files=True)

    import json

    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert set(payload["artifacts"]) == {"a"}
    assert payload["bundles"]["researcher"]["version"] == "2"
