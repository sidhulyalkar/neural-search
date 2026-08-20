import hashlib
from pathlib import Path

import pytest

from neural_search.runtime import (
    ArtifactBundle,
    BundleArtifact,
    fetch_bundle,
    make_lineage_id,
    verify_locked_artifacts,
)


def _artifact(source: Path, *, relative_path: str = "data/example.bin") -> BundleArtifact:
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return BundleArtifact(
        artifact_id="example",
        relative_path=relative_path,
        source_url=source.as_uri(),
        sha256=digest,
        size_bytes=len(content),
        artifact_version="1.0",
        lineage_id=make_lineage_id("example", "1.0", digest),
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

    installed.write_bytes(b"tampered")
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
    artifact = _artifact(source)
    bad = BundleArtifact(
        **{
            **artifact.__dict__,
            "sha256": "0" * 64,
        }
    )
    bundle = ArtifactBundle(
        name="bad-checksum",
        version="1",
        compatibility_group="test:v1",
        artifacts=(bad,),
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_bundle(
            bundle,
            cache_dir=tmp_path / "cache",
            lock_path=tmp_path / "lock.json",
            allow_local_files=True,
        )
