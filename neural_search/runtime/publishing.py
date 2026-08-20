"""Build immutable bundle manifests from locally generated artifact lineages.

This module does not upload scientific data. Storage is deliberately backend
agnostic: a lab can publish bytes to GitHub Releases, S3, Zenodo, institutional
object storage, or another HTTPS origin, then generate an immutable manifest
that pins each object by size, SHA-256, version, and parent lineage.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from neural_search.runtime.bundles import ArtifactBundle, BundleArtifact
from neural_search.runtime.catalog import ARTIFACTS, artifact_status
from neural_search.runtime.lineage import make_lineage_id, read_lineage


def _source_url(base_url: str, relative_path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https":
        raise ValueError("Bundle publication base URL must use HTTPS")
    encoded = "/".join(
        urllib.parse.quote(part, safe="") for part in Path(relative_path).as_posix().split("/")
    )
    return base_url.rstrip("/") + "/" + encoded


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
                f"Artifact bundle publishing currently requires file artifacts; "
                f"archive directory artifact {artifact_id!r} before publishing"
            )

        lineage = read_lineage(path)
        if lineage is None and not allow_untracked:
            raise ValueError(
                f"Artifact {artifact_id} has no lineage sidecar. Stamp it before publishing."
            )

        if lineage is not None:
            artifact_version = lineage.artifact_version
            lineage_id = lineage.lineage_id
            sha256 = lineage.content_sha256
            derived_from = dict(lineage.derived_from)
            if lineage.compatibility_group and lineage.compatibility_group != compatibility_group:
                raise ValueError(
                    f"Compatibility group mismatch for {artifact_id}: "
                    f"{lineage.compatibility_group} != {compatibility_group}"
                )
        else:
            sha256 = status.get("sha256")
            if not sha256:
                import hashlib

                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                sha256 = digest.hexdigest()
            artifact_version = spec.version or version
            lineage_id = make_lineage_id(artifact_id, artifact_version, sha256)
            derived_from = {}

        if spec.derived_from:
            missing_parent_declarations = [
                parent_id for parent_id in spec.derived_from if parent_id not in derived_from
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
                sha256=str(sha256),
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
    """Atomically write the immutable manifest JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(bundle.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
