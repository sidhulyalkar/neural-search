"""Build and index immutable artifact bundle manifests.

This module does not upload scientific data. Storage is backend agnostic: a lab
can publish bytes to GitHub Releases, S3, Zenodo, institutional object storage,
or another HTTPS origin, then generate a manifest that pins every object by
size, SHA-256, version, and parent lineage.

Publishing is deliberately conservative. A lineage sidecar is not trusted just
because it exists: the current bytes are re-hashed before a manifest is built.
Likewise, a release-index entry pins the manifest itself by SHA-256 so a stable
``name@version`` cannot be silently redefined by replacing a remote JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from neural_search.runtime.bundles import (
    ArtifactBundle,
    BundleArtifact,
    ReleaseIndex,
    ReleaseIndexEntry,
)
from neural_search.runtime.catalog import ARTIFACTS, artifact_status
from neural_search.runtime.lineage import hash_path, make_lineage_id, read_lineage


def _source_url(base_url: str, relative_path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https":
        raise ValueError("Bundle publication base URL must use HTTPS")
    encoded = "/".join(
        urllib.parse.quote(part, safe="")
        for part in Path(relative_path).as_posix().split("/")
    )
    return base_url.rstrip("/") + "/" + encoded


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def build_bundle_manifest(
    *,
    name: str,
    version: str,
    compatibility_group: str,
    artifact_ids: Iterable[str],
    source_base_url: str,
    source_commit: str | None = None,
    description: str | None = None,
    allow_untracked: bool = False,
) -> ArtifactBundle:
    """Build a manifest only after checking local artifact identity and lineage."""

    artifacts: list[BundleArtifact] = []
    requested = list(dict.fromkeys(artifact_ids))
    if not requested:
        raise ValueError("At least one artifact_id is required")

    for artifact_id in requested:
        if artifact_id not in ARTIFACTS:
            raise ValueError(f"Unknown artifact: {artifact_id}")
        spec = ARTIFACTS[artifact_id]
        status = artifact_status(artifact_id)
        if not status["usable"]:
            raise ValueError(f"Artifact is not usable: {artifact_id}")
        path = Path(status["absolute_path"])
        if not path.is_file():
            raise ValueError(
                "Artifact bundle publishing currently requires file artifacts; "
                f"archive directory artifact {artifact_id!r} before publishing"
            )

        lineage = read_lineage(path)
        if lineage is None and not allow_untracked:
            raise ValueError(
                f"Artifact {artifact_id} has no lineage sidecar. Stamp it before publishing."
            )

        actual_sha256 = hash_path(path)
        if lineage is not None:
            if lineage.artifact_id != artifact_id:
                raise ValueError(
                    f"Lineage artifact_id mismatch for {artifact_id}: "
                    f"{lineage.artifact_id}"
                )
            if lineage.content_sha256 != actual_sha256:
                raise ValueError(
                    f"Artifact {artifact_id} changed after lineage was stamped; "
                    "rebuild/review it and stamp a new lineage before publishing"
                )
            expected_lineage_id = make_lineage_id(
                lineage.artifact_id,
                lineage.artifact_version,
                lineage.content_sha256,
            )
            if lineage.lineage_id != expected_lineage_id:
                raise ValueError(f"Invalid lineage identity for {artifact_id}")
            artifact_version = lineage.artifact_version
            lineage_id = lineage.lineage_id
            sha256 = lineage.content_sha256
            derived_from = dict(lineage.derived_from)
            if (
                lineage.compatibility_group
                and lineage.compatibility_group != compatibility_group
            ):
                raise ValueError(
                    f"Compatibility group mismatch for {artifact_id}: "
                    f"{lineage.compatibility_group} != {compatibility_group}"
                )
        else:
            sha256 = actual_sha256
            artifact_version = spec.version or version
            lineage_id = make_lineage_id(
                artifact_id,
                artifact_version,
                sha256,
            )
            derived_from = {}

        if spec.version and artifact_version != spec.version and not allow_untracked:
            raise ValueError(
                f"Artifact version mismatch for {artifact_id}: "
                f"{artifact_version} != registry {spec.version}"
            )

        if spec.derived_from:
            missing_parent_declarations = [
                parent_id
                for parent_id in spec.derived_from
                if parent_id not in derived_from
            ]
            if missing_parent_declarations and not allow_untracked:
                raise ValueError(
                    f"Artifact {artifact_id} lineage does not declare expected parents: "
                    + ", ".join(missing_parent_declarations)
                )

        relative_path = spec.path
        artifacts.append(
            BundleArtifact(
                artifact_id=artifact_id,
                relative_path=relative_path,
                source_url=_source_url(source_base_url, relative_path),
                sha256=sha256,
                size_bytes=path.stat().st_size,
                artifact_version=artifact_version,
                lineage_id=lineage_id,
                derived_from=derived_from,
                metadata={
                    "capability": spec.capability,
                    "producer": spec.producer,
                    "registry_path": spec.path,
                },
            )
        )

    bundle = ArtifactBundle(
        name=name,
        version=version,
        compatibility_group=compatibility_group,
        artifacts=tuple(artifacts),
        created_at=datetime.now(UTC).isoformat(),
        source_commit=source_commit,
        description=description,
    )
    bundle.validate()
    return bundle


def write_bundle_manifest(path: Path, bundle: ArtifactBundle) -> None:
    """Atomically write an immutable manifest JSON candidate."""

    bundle.validate()
    _atomic_json_write(path, bundle.to_dict())


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_release_index_entry(
    *,
    index_path: Path,
    manifest_path: Path,
    manifest_url: str,
) -> ReleaseIndexEntry:
    """Pin a manifest in a local release index without allowing ref mutation.

    The manifest must already be written and must declare the same bundle ref as
    the index entry. If the ref is already present with a different URL, digest,
    or compatibility group the operation fails rather than mutating history.
    """

    parsed = urllib.parse.urlparse(manifest_url)
    if parsed.scheme != "https":
        raise ValueError("Published manifest URL must use HTTPS")
    if not manifest_path.is_file():
        raise ValueError(f"Bundle manifest does not exist: {manifest_path}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("Bundle manifest root must be a JSON object")
    bundle = ArtifactBundle.from_dict(raw_manifest)
    entry = ReleaseIndexEntry(
        name=bundle.name,
        version=bundle.version,
        manifest_url=manifest_url,
        manifest_sha256=_manifest_sha256(manifest_path),
        compatibility_group=bundle.compatibility_group,
        deprecated=False,
    )
    entry.validate()

    if index_path.is_file():
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(raw_index, dict):
            raise ValueError("Release index root must be a JSON object")
        index = ReleaseIndex.from_dict(raw_index)
    else:
        index = ReleaseIndex(
            bundles=(),
            schema_version=2,
            description=(
                "Immutable Neural Search artifact bundle index. Each release pins "
                "the manifest and every artifact by SHA-256."
            ),
        )

    for existing in index.bundles:
        if existing.ref != entry.ref:
            continue
        if existing == entry:
            return existing
        raise ValueError(
            f"Release ref {entry.ref} already exists with different immutable metadata; "
            "publish a new version instead"
        )

    updated = ReleaseIndex(
        bundles=tuple(
            sorted(
                (*index.bundles, entry),
                key=lambda item: (item.name, item.version),
            )
        ),
        schema_version=2,
        description=index.description,
    )
    _atomic_json_write(index_path, updated.to_dict())
    return entry


def release_index_entry_dict(entry: ReleaseIndexEntry) -> dict[str, object]:
    """Stable serializable representation for CLI output/tests."""

    return asdict(entry)
