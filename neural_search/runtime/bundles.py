"""Versioned, checksum-verified artifact bundle distribution.

Bundles provide a transport layer for large scientific assets that should not be
committed to Git.  The implementation deliberately uses the standard library so
a fresh Neural Search installation can fetch and verify a bundle before any
heavy analysis extras are available.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from neural_search.runtime.lineage import ArtifactLineage, write_lineage_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "releases" / "index.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / ".neural-search" / "artifact-lock.json"
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class BundleArtifact:
    artifact_id: str
    relative_path: str
    source_url: str
    sha256: str
    size_bytes: int
    artifact_version: str
    lineage_id: str
    derived_from: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundleArtifact":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            relative_path=str(payload["relative_path"]),
            source_url=str(payload["source_url"]),
            sha256=str(payload["sha256"]).lower(),
            size_bytes=int(payload["size_bytes"]),
            artifact_version=str(payload["artifact_version"]),
            lineage_id=str(payload["lineage_id"]),
            derived_from={
                str(key): str(value)
                for key, value in dict(payload.get("derived_from") or {}).items()
            },
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ArtifactBundle:
    name: str
    version: str
    compatibility_group: str
    artifacts: tuple[BundleArtifact, ...]
    created_at: str | None = None
    source_commit: str | None = None
    description: str | None = None
    schema_version: int = 1

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactBundle":
        artifacts = tuple(
            BundleArtifact.from_dict(item) for item in list(payload.get("artifacts") or [])
        )
        bundle = cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            compatibility_group=str(payload["compatibility_group"]),
            artifacts=artifacts,
            created_at=str(payload["created_at"]) if payload.get("created_at") else None,
            source_commit=str(payload["source_commit"]) if payload.get("source_commit") else None,
            description=str(payload["description"]) if payload.get("description") else None,
            schema_version=int(payload.get("schema_version", 1)),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported bundle schema_version={self.schema_version}")
        if not self.name or "@" in self.name:
            raise ValueError("Bundle name must be non-empty and cannot contain '@'")
        if not self.version:
            raise ValueError("Bundle version must be non-empty")
        seen: set[str] = set()
        for artifact in self.artifacts:
            if artifact.artifact_id in seen:
                raise ValueError(f"Duplicate artifact_id in bundle: {artifact.artifact_id}")
            seen.add(artifact.artifact_id)
            _validate_relative_path(artifact.relative_path)
            if len(artifact.sha256) != 64 or any(c not in "0123456789abcdef" for c in artifact.sha256):
                raise ValueError(f"Invalid SHA-256 for {artifact.artifact_id}")
            if artifact.size_bytes < 0:
                raise ValueError(f"Invalid size for {artifact.artifact_id}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseIndexEntry:
    name: str
    version: str
    manifest_url: str
    compatibility_group: str | None = None
    deprecated: bool = False

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class ReleaseIndex:
    bundles: tuple[ReleaseIndexEntry, ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReleaseIndex":
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported release index schema")
        entries = tuple(
            ReleaseIndexEntry(
                name=str(item["name"]),
                version=str(item["version"]),
                manifest_url=str(item["manifest_url"]),
                compatibility_group=(
                    str(item["compatibility_group"])
                    if item.get("compatibility_group") is not None
                    else None
                ),
                deprecated=bool(item.get("deprecated", False)),
            )
            for item in list(payload.get("bundles") or [])
        )
        return cls(bundles=entries)


def default_cache_dir() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ARTIFACT_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "neural-search" / "artifacts"


def default_lock_path() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ARTIFACT_LOCK")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_LOCK_PATH


def parse_bundle_ref(value: str) -> tuple[str, str]:
    name, separator, version = value.partition("@")
    if not separator or not name.strip() or not version.strip():
        raise ValueError("Bundle reference must look like 'name@version'")
    return name.strip(), version.strip()


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.strip() in {"", "."}:
        raise ValueError(f"Unsafe artifact relative_path: {value!r}")


def _safe_destination(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Artifact path escapes cache root: {relative_path}")
    return candidate


def _read_url_bytes(
    source: str,
    *,
    max_bytes: int,
    allow_local_files: bool,
) -> bytes:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"", "file"}:
        if not allow_local_files:
            raise ValueError("Local file sources are disabled; pass allow_local_files=True explicitly")
        path = Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else source)
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"Source exceeds maximum allowed size: {source}")
        return data
    if parsed.scheme != "https":
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only HTTPS is allowed")

    request = urllib.request.Request(source, headers={"User-Agent": "neural-search-artifacts/1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - HTTPS enforced above
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Source exceeds maximum allowed size: {source}")
    return data


def _load_json_source(
    source: str | Path,
    *,
    allow_local_files: bool = False,
    max_bytes: int = MAX_MANIFEST_BYTES,
) -> dict[str, Any]:
    text_source = str(source)
    if isinstance(source, Path) or urllib.parse.urlparse(text_source).scheme == "":
        if not allow_local_files and isinstance(source, Path):
            # A caller passing an explicit Path is an intentional local operation.
            allow_local_files = True
    payload = json.loads(
        _read_url_bytes(
            text_source,
            max_bytes=max_bytes,
            allow_local_files=allow_local_files,
        ).decode("utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("Manifest/index root must be a JSON object")
    return payload


def load_bundle_manifest(
    source: str | Path,
    *,
    allow_local_files: bool = False,
) -> ArtifactBundle:
    return ArtifactBundle.from_dict(
        _load_json_source(source, allow_local_files=allow_local_files)
    )


def load_release_index(
    source: str | Path = DEFAULT_INDEX_PATH,
    *,
    allow_local_files: bool = False,
) -> ReleaseIndex:
    return ReleaseIndex.from_dict(
        _load_json_source(source, allow_local_files=allow_local_files)
    )


def resolve_bundle_manifest_source(
    bundle_ref: str,
    *,
    index_source: str | Path = DEFAULT_INDEX_PATH,
    allow_local_files: bool = False,
) -> str:
    name, version = parse_bundle_ref(bundle_ref)
    index = load_release_index(index_source, allow_local_files=allow_local_files)
    for entry in index.bundles:
        if entry.name == name and entry.version == version and not entry.deprecated:
            return entry.manifest_url
    raise ValueError(f"Bundle not found in release index: {bundle_ref}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_cached_file(path: Path, artifact: BundleArtifact) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == artifact.size_bytes
        and _file_sha256(path) == artifact.sha256
    )


def _download_artifact(
    artifact: BundleArtifact,
    destination: Path,
    *,
    allow_local_files: bool,
    max_artifact_bytes: int,
) -> None:
    if artifact.size_bytes > max_artifact_bytes:
        raise ValueError(
            f"Artifact {artifact.artifact_id} declares {artifact.size_bytes} bytes, "
            f"above configured limit {max_artifact_bytes}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(artifact.source_url)
    if parsed.scheme in {"", "file"}:
        if not allow_local_files:
            raise ValueError("Local file artifact sources require explicit allow_local_files=True")
        source_path = Path(
            urllib.request.url2pathname(parsed.path)
            if parsed.scheme == "file"
            else artifact.source_url
        )
        source = source_path.open("rb")
        expected_length = source_path.stat().st_size
    elif parsed.scheme == "https":
        request = urllib.request.Request(
            artifact.source_url,
            headers={"User-Agent": "neural-search-artifacts/1"},
        )
        response = urllib.request.urlopen(request, timeout=120)  # noqa: S310 - HTTPS enforced
        source = response
        expected_length = int(response.headers.get("Content-Length") or 0)
    else:
        raise ValueError(f"Unsupported artifact URL scheme: {parsed.scheme}")

    if expected_length and expected_length != artifact.size_bytes:
        source.close()
        raise ValueError(
            f"Content-Length mismatch for {artifact.artifact_id}: "
            f"expected {artifact.size_bytes}, got {expected_length}"
        )

    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    digest = hashlib.sha256()
    written = 0
    try:
        with source, os.fdopen(fd, "wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > artifact.size_bytes or written > max_artifact_bytes:
                    raise ValueError(f"Artifact exceeded declared/allowed size: {artifact.artifact_id}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if written != artifact.size_bytes:
            raise ValueError(
                f"Size mismatch for {artifact.artifact_id}: expected {artifact.size_bytes}, got {written}"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != artifact.sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact.artifact_id}: expected {artifact.sha256}, got {actual_sha}"
            )
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        return {"schema_version": 1, "bundles": {}, "artifacts": {}}
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "bundles": {}, "artifacts": {}}
    if not isinstance(payload, dict) or int(payload.get("schema_version", 1)) != 1:
        raise ValueError(f"Unsupported artifact lock schema at {lock_path}")
    payload.setdefault("bundles", {})
    payload.setdefault("artifacts", {})
    return payload


def _atomic_lock_write(lock_path: Path, payload: Mapping[str, Any]) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{lock_path.name}.", dir=lock_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, lock_path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def fetch_bundle(
    bundle: ArtifactBundle,
    *,
    cache_dir: Path | None = None,
    lock_path: Path | None = None,
    allow_local_files: bool = False,
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Install a verified bundle into the local content cache and pin it."""

    bundle.validate()
    cache_root = (cache_dir or default_cache_dir()).expanduser().resolve()
    bundle_root = cache_root / bundle.name / bundle.version
    installed: list[dict[str, Any]] = []

    for artifact in bundle.artifacts:
        destination = _safe_destination(bundle_root, artifact.relative_path)
        reused = _verify_cached_file(destination, artifact)
        if not reused:
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            _download_artifact(
                artifact,
                destination,
                allow_local_files=allow_local_files,
                max_artifact_bytes=max_artifact_bytes,
            )

        lineage = ArtifactLineage(
            artifact_id=artifact.artifact_id,
            artifact_version=artifact.artifact_version,
            lineage_id=artifact.lineage_id,
            content_sha256=artifact.sha256,
            created_at=bundle.created_at or datetime.now(UTC).isoformat(),
            compatibility_group=bundle.compatibility_group,
            producer=f"artifact-bundle:{bundle.ref}",
            derived_from=dict(artifact.derived_from),
            metadata={**artifact.metadata, "bundle_ref": bundle.ref},
        )
        write_lineage_record(destination, lineage)
        installed.append(
            {
                "artifact_id": artifact.artifact_id,
                "path": str(destination),
                "sha256": artifact.sha256,
                "lineage_id": artifact.lineage_id,
                "reused": reused,
            }
        )

    resolved_lock = (lock_path or default_lock_path()).expanduser().resolve()
    lock = _load_lock(resolved_lock)
    lock["bundles"][bundle.name] = {
        "version": bundle.version,
        "compatibility_group": bundle.compatibility_group,
        "source_commit": bundle.source_commit,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    for item in installed:
        lock["artifacts"][item["artifact_id"]] = {
            "path": item["path"],
            "sha256": item["sha256"],
            "lineage_id": item["lineage_id"],
            "bundle": bundle.name,
            "bundle_version": bundle.version,
            "compatibility_group": bundle.compatibility_group,
        }
    _atomic_lock_write(resolved_lock, lock)

    return {
        "bundle": bundle.ref,
        "compatibility_group": bundle.compatibility_group,
        "cache_root": str(cache_root),
        "lock_path": str(resolved_lock),
        "installed": installed,
    }


def fetch_bundle_ref(
    bundle_ref: str,
    *,
    index_source: str | Path = DEFAULT_INDEX_PATH,
    manifest_source: str | Path | None = None,
    cache_dir: Path | None = None,
    lock_path: Path | None = None,
    allow_local_files: bool = False,
) -> dict[str, Any]:
    source = manifest_source or resolve_bundle_manifest_source(
        bundle_ref,
        index_source=index_source,
        allow_local_files=allow_local_files,
    )
    bundle = load_bundle_manifest(source, allow_local_files=allow_local_files)
    if bundle.ref != bundle_ref:
        raise ValueError(
            f"Manifest identity mismatch: requested {bundle_ref}, manifest declares {bundle.ref}"
        )
    return fetch_bundle(
        bundle,
        cache_dir=cache_dir,
        lock_path=lock_path,
        allow_local_files=allow_local_files,
    )


def resolve_locked_artifact(
    artifact_id: str,
    *,
    lock_path: Path | None = None,
) -> Path | None:
    lock = _load_lock((lock_path or default_lock_path()).expanduser().resolve())
    item = dict(lock.get("artifacts") or {}).get(artifact_id)
    if not isinstance(item, dict) or not item.get("path"):
        return None
    path = Path(str(item["path"])).expanduser()
    return path if path.exists() else None


def lock_snapshot(*, lock_path: Path | None = None) -> dict[str, Any]:
    """Return the local lock without exposing unrelated environment data."""

    resolved = (lock_path or default_lock_path()).expanduser().resolve()
    payload = _load_lock(resolved)
    return {"lock_path": str(resolved), **payload}


def verify_locked_artifacts(*, lock_path: Path | None = None) -> dict[str, Any]:
    resolved = (lock_path or default_lock_path()).expanduser().resolve()
    lock = _load_lock(resolved)
    results: list[dict[str, Any]] = []
    for artifact_id, raw in dict(lock.get("artifacts") or {}).items():
        item = dict(raw)
        path = Path(str(item.get("path") or ""))
        expected_sha = str(item.get("sha256") or "")
        exists = path.is_file()
        actual_sha = _file_sha256(path) if exists else None
        valid = exists and actual_sha == expected_sha
        results.append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "exists": exists,
                "valid": valid,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "lineage_id": item.get("lineage_id"),
                "bundle": item.get("bundle"),
                "bundle_version": item.get("bundle_version"),
            }
        )
    return {
        "lock_path": str(resolved),
        "valid": all(item["valid"] for item in results),
        "artifacts": results,
    }
