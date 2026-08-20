"""Content-addressed lineage metadata for Neural Search artifacts.

Lineage sidecars make generated scientific assets auditable.  A file existing at
an expected path is not sufficient evidence that it belongs to the corpus or
configuration currently in use; the sidecar records the content digest and the
exact lineage IDs of its parents.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

SIDECAR_FILENAME = ".neural-search-artifact.json"
SIDECAR_SUFFIX = ".neural-search.json"


@dataclass(frozen=True)
class ArtifactLineage:
    """Immutable identity and derivation metadata for one artifact."""

    artifact_id: str
    artifact_version: str
    lineage_id: str
    content_sha256: str
    created_at: str
    compatibility_group: str | None = None
    producer: str | None = None
    derived_from: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactLineage":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_version=str(payload["artifact_version"]),
            lineage_id=str(payload["lineage_id"]),
            content_sha256=str(payload["content_sha256"]),
            created_at=str(payload["created_at"]),
            compatibility_group=(
                str(payload["compatibility_group"])
                if payload.get("compatibility_group") is not None
                else None
            ),
            producer=str(payload["producer"]) if payload.get("producer") else None,
            derived_from={
                str(key): str(value)
                for key, value in dict(payload.get("derived_from") or {}).items()
            },
            metadata=dict(payload.get("metadata") or {}),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sidecar_path(path: Path) -> Path:
    """Return the deterministic metadata sidecar path for a file or directory."""

    if path.is_dir() or (not path.exists() and path.suffix == ""):
        return path / SIDECAR_FILENAME
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    """Hash a file or a directory tree deterministically.

    Directory identities include relative paths and bytes, and deliberately
    exclude lineage sidecars so writing metadata cannot change the content
    identity it describes.
    """

    if path.is_file():
        return _hash_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    for candidate in sorted(p for p in path.rglob("*") if p.is_file()):
        if candidate.name == SIDECAR_FILENAME or candidate.name.endswith(SIDECAR_SUFFIX):
            continue
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def make_lineage_id(artifact_id: str, artifact_version: str, content_sha256: str) -> str:
    """Build a stable lineage identifier from artifact identity and content."""

    payload = f"{artifact_id}\0{artifact_version}\0{content_sha256}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_lineage(path: Path) -> ArtifactLineage | None:
    metadata_path = sidecar_path(path)
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ArtifactLineage.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def write_lineage(
    path: Path,
    *,
    artifact_id: str,
    artifact_version: str,
    compatibility_group: str | None = None,
    producer: str | None = None,
    derived_from: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactLineage:
    """Hash an artifact and atomically write its lineage sidecar."""

    content_sha256 = hash_path(path)
    lineage = ArtifactLineage(
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        lineage_id=make_lineage_id(artifact_id, artifact_version, content_sha256),
        content_sha256=content_sha256,
        created_at=datetime.now(UTC).isoformat(),
        compatibility_group=compatibility_group,
        producer=producer,
        derived_from=dict(derived_from or {}),
        metadata=dict(metadata or {}),
    )
    _atomic_json_write(sidecar_path(path), lineage.to_dict())
    return lineage


def write_lineage_record(path: Path, lineage: ArtifactLineage) -> None:
    """Write a pre-verified lineage record, used by immutable bundle installs."""

    _atomic_json_write(sidecar_path(path), lineage.to_dict())


def validate_lineage(
    path: Path,
    lineage: ArtifactLineage,
    *,
    parent_lineages: Mapping[str, ArtifactLineage | None] | None = None,
    verify_content: bool = False,
) -> dict[str, Any]:
    """Validate content identity and declared parent relationships."""

    issues: list[str] = []
    if verify_content:
        try:
            actual_sha = hash_path(path)
        except FileNotFoundError:
            actual_sha = None
        if actual_sha != lineage.content_sha256:
            issues.append("content_digest_mismatch")

    parent_states: dict[str, str] = {}
    for parent_id, expected_lineage_id in lineage.derived_from.items():
        parent = (parent_lineages or {}).get(parent_id)
        if parent is None:
            parent_states[parent_id] = "missing_or_untracked"
            issues.append(f"parent_unavailable:{parent_id}")
        elif parent.lineage_id != expected_lineage_id:
            parent_states[parent_id] = "lineage_mismatch"
            issues.append(f"parent_lineage_mismatch:{parent_id}")
        else:
            parent_states[parent_id] = "compatible"

    return {
        "compatible": not issues,
        "state": "compatible" if not issues else "incompatible",
        "lineage_id": lineage.lineage_id,
        "content_sha256": lineage.content_sha256,
        "parent_states": parent_states,
        "issues": issues,
    }
