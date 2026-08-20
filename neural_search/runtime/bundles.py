"""Versioned, checksum-verified artifact bundle distribution.

Bundles provide a transport layer for large scientific assets that should not be
committed to Git. The implementation deliberately uses the standard library so
a fresh Neural Search installation can fetch and verify a bundle before any
heavy analysis extras are available.

A release reference is immutable only when BOTH levels are content-addressed:

* the release index pins the bundle manifest by SHA-256; and
* the bundle manifest pins every artifact by byte size and SHA-256.

The local lock records the verified stat state of installed files. Runtime
resolution performs a cheap stat check so a file changed after verification is
not silently consumed. A full ``artifacts verify`` re-hashes bytes and refreshes
that stat checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from neural_search.runtime.lineage import (
    ArtifactLineage,
    make_lineage_id,
    write_lineage_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "artifacts" / "releases" / "index.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / ".neural-search" / "artifact-lock.json"
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024 * 1024
_SHA256_HEX_LENGTH = 64
_LINEAGE_PREFIX = "sha256:"


def _valid_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _valid_lineage_id(value: str) -> bool:
    if not value.startswith(_LINEAGE_PREFIX):
        return False
    return _valid_sha256(value[len(_LINEAGE_PREFIX) :])


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
    def from_dict(cls, payload: Mapping[str, Any]) -> BundleArtifact:
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

    def validate(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("Bundle artifact_id must be non-empty")
        if not self.artifact_version.strip():
            raise ValueError(f"Artifact version is empty: {self.artifact_id}")
        _validate_relative_path(self.relative_path)
        if not _valid_sha256(self.sha256):
            raise ValueError(f"Invalid SHA-256 for {self.artifact_id}")
        if self.size_bytes < 0:
            raise ValueError(f"Invalid size for {self.artifact_id}")
        expected_lineage = make_lineage_id(
            self.artifact_id,
            self.artifact_version,
            self.sha256,
        )
        if self.lineage_id != expected_lineage:
            raise ValueError(
                f"Lineage identity mismatch for {self.artifact_id}: manifest lineage "
                "does not match artifact id/version/content digest"
            )
        for parent_id, parent_lineage in self.derived_from.items():
            if not parent_id.strip() or not _valid_lineage_id(parent_lineage):
                raise ValueError(
                    f"Invalid parent lineage declaration for {self.artifact_id}: {parent_id!r}"
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
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactBundle:
        artifacts = tuple(
            BundleArtifact.from_dict(item) for item in list(payload.get("artifacts") or [])
        )
        bundle = cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            compatibility_group=str(payload["compatibility_group"]),
            artifacts=artifacts,
            created_at=str(payload["created_at"]) if payload.get("created_at") else None,
            source_commit=str(payload["source_commit"])
            if payload.get("source_commit")
            else None,
            description=str(payload["description"])
            if payload.get("description")
            else None,
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
        if not self.compatibility_group:
            raise ValueError("Bundle compatibility_group must be non-empty")
        seen: set[str] = set()
        for artifact in self.artifacts:
            artifact.validate()
            if artifact.artifact_id in seen:
                raise ValueError(
                    f"Duplicate artifact_id in bundle: {artifact.artifact_id}"
                )
            seen.add(artifact.artifact_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseIndexEntry:
    name: str
    version: str
    manifest_url: str
    manifest_sha256: str | None = None
    compatibility_group: str | None = None
    deprecated: bool = False

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def validate(self) -> None:
        if not self.name or "@" in self.name or not self.version:
            raise ValueError("Release index entry has invalid name/version")
        parsed = urllib.parse.urlparse(self.manifest_url)
        if parsed.scheme != "https":
            raise ValueError(
                f"Published manifest URL must use HTTPS: {self.manifest_url}"
            )
        if self.manifest_sha256 is not None and not _valid_sha256(
            self.manifest_sha256
        ):
            raise ValueError(f"Invalid manifest SHA-256 for {self.ref}")


@dataclass(frozen=True)
class ReleaseIndex:
    bundles: tuple[ReleaseIndexEntry, ...]
    schema_version: int = 2
    description: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReleaseIndex:
        schema_version = int(payload.get("schema_version", 1))
        if schema_version not in {1, 2}:
            raise ValueError("Unsupported release index schema")
        entries = tuple(
            ReleaseIndexEntry(
                name=str(item["name"]),
                version=str(item["version"]),
                manifest_url=str(item["manifest_url"]),
                manifest_sha256=(
                    str(item["manifest_sha256"]).lower()
                    if item.get("manifest_sha256") is not None
                    else None
                ),
                compatibility_group=(
                    str(item["compatibility_group"])
                    if item.get("compatibility_group") is not None
                    else None
                ),
                deprecated=bool(item.get("deprecated", False)),
            )
            for item in list(payload.get("bundles") or [])
        )
        seen: set[str] = set()
        for entry in entries:
            entry.validate()
            if entry.ref in seen:
                raise ValueError(f"Duplicate release reference: {entry.ref}")
            seen.add(entry.ref)
        return cls(
            bundles=entries,
            schema_version=schema_version,
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "description": self.description,
            "bundles": [asdict(entry) for entry in self.bundles],
        }


def default_cache_dir() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ARTIFACT_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "neural-search" / "artifacts"


def default_lock_path() -> Path:
    configured = os.getenv("NEURAL_SEARCH_ARTIFACT_LOCK")
    return (
        Path(configured).expanduser().resolve() if configured else DEFAULT_LOCK_PATH
    )


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
            raise ValueError(
                "Local file sources are disabled; pass allow_local_files=True explicitly"
            )
        path = Path(
            urllib.request.url2pathname(parsed.path)
            if parsed.scheme == "file"
            else source
        )
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"Source exceeds maximum allowed size: {source}")
        return data
    if parsed.scheme != "https":
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; only HTTPS is allowed"
        )

    request = urllib.request.Request(
        source,
        headers={"User-Agent": "neural-search-artifacts/2"},
    )
    with urllib.request.urlopen(  # noqa: S310 - HTTPS enforced above
        request,
        timeout=60,
    ) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Source exceeds maximum allowed size: {source}")
    return data


def _load_json_bytes(data: bytes) -> dict[str, Any]:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest/index root must be a JSON object")
    return payload


def _source_bytes(
    source: str | Path,
    *,
    allow_local_files: bool,
    max_bytes: int = MAX_MANIFEST_BYTES,
) -> bytes:
    text_source = str(source)
    explicit_path = isinstance(source, Path)
    return _read_url_bytes(
        text_source,
        max_bytes=max_bytes,
        allow_local_files=allow_local_files or explicit_path,
    )


def load_bundle_manifest(
    source: str | Path,
    *,
    allow_local_files: bool = False,
    expected_sha256: str | None = None,
) -> ArtifactBundle:
    data = _source_bytes(source, allow_local_files=allow_local_files)
    if expected_sha256 is not None:
        expected = expected_sha256.lower()
        if not _valid_sha256(expected):
            raise ValueError("Invalid expected manifest SHA-256")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Manifest checksum mismatch: expected {expected}, got {actual}"
            )
    return ArtifactBundle.from_dict(_load_json_bytes(data))


def load_release_index(
    source: str | Path = DEFAULT_INDEX_PATH,
    *,
    allow_local_files: bool = False,
) -> ReleaseIndex:
    data = _source_bytes(source, allow_local_files=allow_local_files)
    return ReleaseIndex.from_dict(_load_json_bytes(data))


def resolve_release_entry(
    bundle_ref: str,
    *,
    index_source: str | Path = DEFAULT_INDEX_PATH,
    allow_local_files: bool = False,
) -> ReleaseIndexEntry:
    name, version = parse_bundle_ref(bundle_ref)
    index = load_release_index(index_source, allow_local_files=allow_local_files)
    for entry in index.bundles:
        if entry.name == name and entry.version == version and not entry.deprecated:
            return entry
    raise ValueError(f"Bundle not found in release index: {bundle_ref}")


def resolve_bundle_manifest_source(
    bundle_ref: str,
    *,
    index_source: str | Path = DEFAULT_INDEX_PATH,
    allow_local_files: bool = False,
) -> str:
    return resolve_release_entry(
        bundle_ref,
        index_source=index_source,
        allow_local_files=allow_local_files,
    ).manifest_url


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
            raise ValueError(
                "Local file artifact sources require explicit allow_local_files=True"
            )
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
            headers={"User-Agent": "neural-search-artifacts/2"},
        )
        response = urllib.request.urlopen(  # noqa: S310 - HTTPS enforced
            request,
            timeout=120,
        )
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

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
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
                    raise ValueError(
                        f"Artifact exceeded declared/allowed size: {artifact.artifact_id}"
                    )
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if written != artifact.size_bytes:
            raise ValueError(
                f"Size mismatch for {artifact.artifact_id}: "
                f"expected {artifact.size_bytes}, got {written}"
            )
        actual_sha = digest.hexdigest()
        if actual_sha != artifact.sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact.artifact_id}: "
                f"expected {artifact.sha256}, got {actual_sha}"
            )
        os.replace(temp_name, destination)
        try:
            destination.chmod(0o444)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _empty_lock() -> dict[str, Any]:
    return {"schema_version": 2, "bundles": {}, "artifacts": {}}


def _load_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        return _empty_lock()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_lock()
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid artifact lock at {lock_path}")
    schema = int(payload.get("schema_version", 1))
    if schema not in {1, 2}:
        raise ValueError(f"Unsupported artifact lock schema at {lock_path}")
    payload["schema_version"] = 2
    payload.setdefault("bundles", {})
    payload.setdefault("artifacts", {})
    return payload


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


def _artifact_lock_record(
    *,
    bundle: ArtifactBundle,
    artifact: BundleArtifact,
    path: Path,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "lineage_id": artifact.lineage_id,
        "bundle": bundle.name,
        "bundle_version": bundle.version,
        "compatibility_group": bundle.compatibility_group,
        "verified_at": datetime.now(UTC).isoformat(),
        "verified_size_bytes": stat.st_size,
        "verified_mtime_ns": stat.st_mtime_ns,
        "integrity_state": "verified",
    }


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
    artifact_records: dict[str, dict[str, Any]] = {}

    for artifact in bundle.artifacts:
        destination = _safe_destination(bundle_root, artifact.relative_path)
        reused = _verify_cached_file(destination, artifact)
        if not reused:
            if destination.exists():
                try:
                    destination.chmod(0o644)
                except OSError:
                    pass
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
        record = _artifact_lock_record(
            bundle=bundle,
            artifact=artifact,
            path=destination,
        )
        artifact_records[artifact.artifact_id] = record
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
    lock["artifacts"] = {
        artifact_id: item
        for artifact_id, item in dict(lock.get("artifacts") or {}).items()
        if not isinstance(item, dict) or item.get("bundle") != bundle.name
    }
    lock["bundles"][bundle.name] = {
        "version": bundle.version,
        "compatibility_group": bundle.compatibility_group,
        "source_commit": bundle.source_commit,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    lock["artifacts"].update(artifact_records)
    _atomic_json_write(resolved_lock, lock)

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
    """Fetch a bundle by immutable release ref or an explicit test manifest.

    Normal release-index resolution requires ``manifest_sha256``. Passing an
    explicit ``manifest_source`` is intentionally treated as a direct/test mode:
    the manifest still pins all artifact bytes, but the caller is responsible
    for pinning the manifest itself.
    """

    if manifest_source is not None:
        bundle = load_bundle_manifest(
            manifest_source,
            allow_local_files=allow_local_files,
        )
    else:
        entry = resolve_release_entry(
            bundle_ref,
            index_source=index_source,
            allow_local_files=allow_local_files,
        )
        if not entry.manifest_sha256:
            raise ValueError(
                f"Release {bundle_ref} is not immutable: manifest_sha256 is missing"
            )
        bundle = load_bundle_manifest(
            entry.manifest_url,
            allow_local_files=allow_local_files,
            expected_sha256=entry.manifest_sha256,
        )
        if (
            entry.compatibility_group
            and bundle.compatibility_group != entry.compatibility_group
        ):
            raise ValueError(
                f"Release compatibility mismatch: index declares "
                f"{entry.compatibility_group}, manifest declares "
                f"{bundle.compatibility_group}"
            )

    if bundle.ref != bundle_ref:
        raise ValueError(
            f"Manifest identity mismatch: requested {bundle_ref}, "
            f"manifest declares {bundle.ref}"
        )
    return fetch_bundle(
        bundle,
        cache_dir=cache_dir,
        lock_path=lock_path,
        allow_local_files=allow_local_files,
    )


def _lightweight_lock_state(item: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(item.get("path") or ""))
    if not path.is_file():
        return {"usable": False, "state": "missing"}
    stat = path.stat()
    expected_size = int(item.get("size_bytes") or item.get("verified_size_bytes") or -1)
    if expected_size >= 0 and stat.st_size != expected_size:
        return {"usable": False, "state": "size_changed"}
    verified_mtime = item.get("verified_mtime_ns")
    if verified_mtime is not None and stat.st_mtime_ns != int(verified_mtime):
        return {"usable": False, "state": "changed_since_verification"}
    return {"usable": True, "state": "verified_stat"}


def resolve_locked_artifact(
    artifact_id: str,
    *,
    lock_path: Path | None = None,
) -> Path | None:
    lock = _load_lock((lock_path or default_lock_path()).expanduser().resolve())
    item = dict(lock.get("artifacts") or {}).get(artifact_id)
    if not isinstance(item, dict) or not item.get("path"):
        return None
    state = _lightweight_lock_state(item)
    if not state["usable"]:
        return None
    return Path(str(item["path"])).expanduser()


def lock_snapshot(*, lock_path: Path | None = None) -> dict[str, Any]:
    """Return the local lock without exposing unrelated environment data."""

    resolved = (lock_path or default_lock_path()).expanduser().resolve()
    payload = _load_lock(resolved)
    return {"lock_path": str(resolved), **payload}


def locked_artifact_integrity(*, lock_path: Path | None = None) -> dict[str, Any]:
    """Return cheap integrity state without re-hashing potentially huge files."""

    resolved = (lock_path or default_lock_path()).expanduser().resolve()
    lock = _load_lock(resolved)
    results: list[dict[str, Any]] = []
    for artifact_id, raw in dict(lock.get("artifacts") or {}).items():
        if not isinstance(raw, dict):
            results.append(
                {"artifact_id": artifact_id, "usable": False, "state": "invalid_lock"}
            )
            continue
        state = _lightweight_lock_state(raw)
        results.append(
            {
                "artifact_id": artifact_id,
                "path": str(raw.get("path") or ""),
                "bundle": raw.get("bundle"),
                "bundle_version": raw.get("bundle_version"),
                **state,
            }
        )
    return {
        "lock_path": str(resolved),
        "valid": all(item["usable"] for item in results),
        "verification_mode": "stat_checkpoint",
        "artifacts": results,
    }


def verify_locked_artifacts(*, lock_path: Path | None = None) -> dict[str, Any]:
    """Fully re-hash locked artifact bytes and refresh verification checkpoints."""

    resolved = (lock_path or default_lock_path()).expanduser().resolve()
    lock = _load_lock(resolved)
    results: list[dict[str, Any]] = []
    changed = False
    artifacts = dict(lock.get("artifacts") or {})
    for artifact_id, raw in artifacts.items():
        item = dict(raw) if isinstance(raw, dict) else {}
        path = Path(str(item.get("path") or ""))
        expected_sha = str(item.get("sha256") or "")
        exists = path.is_file()
        actual_sha = _file_sha256(path) if exists else None
        valid = exists and bool(expected_sha) and actual_sha == expected_sha
        if valid:
            stat = path.stat()
            item["verified_at"] = datetime.now(UTC).isoformat()
            item["verified_size_bytes"] = stat.st_size
            item["verified_mtime_ns"] = stat.st_mtime_ns
            item["size_bytes"] = stat.st_size
            item["integrity_state"] = "verified"
            artifacts[artifact_id] = item
            changed = True
        elif item:
            item["integrity_state"] = "invalid"
            artifacts[artifact_id] = item
            changed = True
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
    if changed:
        lock["artifacts"] = artifacts
        _atomic_json_write(resolved, lock)
    return {
        "lock_path": str(resolved),
        "valid": all(item["valid"] for item in results),
        "verification_mode": "sha256",
        "artifacts": results,
    }
