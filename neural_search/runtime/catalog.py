"""Machine-readable execution profiles and artifact contracts.

The repository contains two very different classes of assets:

* portable files that should exist in every clone (fixtures, benchmark inputs), and
* large/generated research assets that are intentionally local (corpora, graphs,
  embedding caches, extracted literature).

This module makes that distinction explicit so CLIs, CI, the API, and external
labs all ask the same question: what does this profile require, what is missing,
and how was the current environment produced?
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ArtifactKind = Literal[
    "committed_fixture",
    "frozen_evaluation",
    "generated_local",
    "derived_report",
]


@dataclass(frozen=True)
class ArtifactSpec:
    """Contract for one repository artifact or artifact directory."""

    id: str
    path: str
    kind: ArtifactKind
    description: str
    producer: str | None = None
    repair_command: str | None = None
    requires_content: bool = False


@dataclass(frozen=True)
class ExecutionProfile:
    """A supported way to operate Neural Search."""

    name: str
    description: str
    install_extra: str
    required_modules: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    recommended_artifacts: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    commands: tuple[str, ...]
    resource_notes: tuple[str, ...]


ARTIFACTS: dict[str, ArtifactSpec] = {
    "behavioral_ontology": ArtifactSpec(
        id="behavioral_ontology",
        path="behavioral_task_ontology.yaml",
        kind="committed_fixture",
        description="Behavioral task ontology used by the portable search path.",
    ),
    "demo_datasets": ArtifactSpec(
        id="demo_datasets",
        path="data/seed/demo_datasets.yaml",
        kind="committed_fixture",
        description="Curated dataset fixtures for demos, tests, and onboarding.",
    ),
    "demo_papers": ArtifactSpec(
        id="demo_papers",
        path="data/seed/demo_papers.yaml",
        kind="committed_fixture",
        description="Curated literature fixtures linked to demo datasets.",
    ),
    "canonical_benchmark_queries": ArtifactSpec(
        id="canonical_benchmark_queries",
        path="data/eval/benchmark_queries_canonical.yaml",
        kind="frozen_evaluation",
        description="Canonical retrieval benchmark queries committed with the repository.",
    ),
    "canonical_qrels": ArtifactSpec(
        id="canonical_qrels",
        path="data/qrels/qrels.canonical.jsonl",
        kind="frozen_evaluation",
        description="Canonical graded relevance judgments used by evaluation tooling.",
    ),
    "ablation_corpus": ArtifactSpec(
        id="ablation_corpus",
        path="data/eval/ablation_corpus_from_packets.jsonl",
        kind="generated_local",
        description=(
            "Evaluation corpus derived from evidence packets for the full retrieval "
            "ablation ladder; not expected in a fresh clone."
        ),
        producer="evaluation evidence-packet pipeline",
    ),
    "raw_corpus_inputs": ArtifactSpec(
        id="raw_corpus_inputs",
        path="data/raw",
        kind="generated_local",
        description="Downloaded source payloads used to rebuild the full normalized corpus.",
        producer="source-specific acquisition scripts under scripts/corpus/",
        requires_content=True,
    ),
    "full_corpus_v09": ArtifactSpec(
        id="full_corpus_v09",
        path="data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl",
        kind="generated_local",
        description=(
            "Current flat normalized multi-source dataset corpus. Building it requires "
            "source payloads to have been acquired first."
        ),
        producer="scripts/corpus/build_full_corpus.py after source acquisition",
    ),
    "production_graph": ArtifactSpec(
        id="production_graph",
        path="data/graph/neural_search_graph.real_corpus.json",
        kind="generated_local",
        description="Knowledge graph derived from the current full corpus.",
        producer="scripts/rebuild_full_corpus_graph.py",
        repair_command="python scripts/rebuild_full_corpus_graph.py",
    ),
    "dense_field_embeddings": ArtifactSpec(
        id="dense_field_embeddings",
        path="data/embeddings/real_all.dense.field_embeddings.jsonl",
        kind="generated_local",
        description="Dense field-level embedding cache for semantic retrieval.",
        producer="scripts/recompute_embeddings.py --provider dense",
        repair_command="python scripts/recompute_embeddings.py --provider dense",
    ),
    "literature_findings": ArtifactSpec(
        id="literature_findings",
        path="artifacts/literature/findings_v1.jsonl",
        kind="generated_local",
        description="Structured findings extracted from the literature corpus.",
        producer="literature extraction pipeline",
    ),
    "paper_dataset_links": ArtifactSpec(
        id="paper_dataset_links",
        path="artifacts/literature/paper_dataset_links.jsonl",
        kind="generated_local",
        description="Dataset-to-paper linkage evidence used by search and the graph.",
        producer="literature linking pipeline",
    ),
    "current_artifact_manifest": ArtifactSpec(
        id="current_artifact_manifest",
        path="reports/eval/current_artifact_manifest.json",
        kind="derived_report",
        description="Computed current-state scientific artifact summary.",
        producer="scripts/build_artifact_manifest.py",
        repair_command="python scripts/build_artifact_manifest.py",
    ),
}


_CORE_MODULES = ("fastapi", "pydantic", "yaml", "numpy", "pandas")

PROFILES: dict[str, ExecutionProfile] = {
    "demo": ExecutionProfile(
        name="demo",
        description=(
            "Portable, deterministic onboarding path. Uses committed fixtures and does "
            "not require production corpora, GPUs, databases, or external APIs."
        ),
        install_extra="dev",
        required_modules=_CORE_MODULES,
        required_artifacts=("behavioral_ontology", "demo_datasets", "demo_papers"),
        recommended_artifacts=(),
        produced_artifacts=(),
        commands=(
            "python -m pip install -e '.[dev]'",
            "cd apps/web && npm ci",
            "neural-search profile check demo",
            "make demo",
        ),
        resource_notes=(
            "CPU-only is supported.",
            "No production corpus or external service is required.",
        ),
    ),
    "researcher": ExecutionProfile(
        name="researcher",
        description=(
            "Interactive dataset/literature discovery against the real local corpus, with "
            "semantic retrieval and reusable analysis outputs."
        ),
        install_extra="researcher",
        required_modules=_CORE_MODULES + ("sentence_transformers", "nbformat"),
        required_artifacts=("behavioral_ontology", "full_corpus_v09"),
        recommended_artifacts=(
            "production_graph",
            "dense_field_embeddings",
            "literature_findings",
            "paper_dataset_links",
        ),
        produced_artifacts=(),
        commands=(
            "python -m pip install -e '.[researcher]'",
            "neural-search profile check researcher",
            "NEURAL_SEARCH_PROFILE=researcher make api",
            "make web",
        ),
        resource_notes=(
            "CPU search is supported; a GPU materially reduces dense-index build time.",
            "Expect local generated research assets that are intentionally not committed.",
        ),
    ),
    "corpus-builder": ExecutionProfile(
        name="corpus-builder",
        description=(
            "Ingest, normalize, deduplicate, enrich, embed, and graph multi-source neural "
            "dataset metadata."
        ),
        install_extra="corpus-builder",
        required_modules=_CORE_MODULES + ("sentence_transformers", "networkx"),
        required_artifacts=("behavioral_ontology", "raw_corpus_inputs"),
        recommended_artifacts=(),
        produced_artifacts=(
            "full_corpus_v09",
            "production_graph",
            "dense_field_embeddings",
        ),
        commands=(
            "python -m pip install -e '.[corpus-builder]'",
            "python scripts/corpus/build_full_corpus.py",
            "python scripts/rebuild_full_corpus_graph.py",
            "python scripts/recompute_embeddings.py --provider dense",
        ),
        resource_notes=(
            "Network access is needed to refresh raw source payloads.",
            "Dense embedding generation is GPU-friendly and can be slow on CPU-only hosts.",
            "Keep downloaded/raw data and generated indexes outside version control.",
        ),
    ),
    "evaluator": ExecutionProfile(
        name="evaluator",
        description=(
            "Reproduce portable retrieval checks and, when local assets are available, "
            "run full qrels metrics, ablations, confidence intervals, calibration, and "
            "scientific regression gates."
        ),
        install_extra="evaluator",
        required_modules=_CORE_MODULES + ("pytest", "networkx"),
        required_artifacts=("canonical_benchmark_queries", "canonical_qrels"),
        recommended_artifacts=(
            "ablation_corpus",
            "production_graph",
            "dense_field_embeddings",
        ),
        produced_artifacts=("current_artifact_manifest",),
        commands=(
            "python -m pip install -e '.[evaluator]'",
            "neural-search profile check evaluator",
            "make benchmark",
            "python scripts/build_artifact_manifest.py",
        ),
        resource_notes=(
            "Canonical benchmark queries and qrels are committed for a portable baseline.",
            "Full dense/graph ablations require generated local evaluation artifacts.",
            "Do not promote silver/LLM judgments to human-gold claims.",
        ),
    ),
    "full-stack": ExecutionProfile(
        name="full-stack",
        description=(
            "Maintainer environment spanning the web/API product, real corpus, graph, "
            "literature tooling, evaluation, databases, and optional spectral analysis."
        ),
        install_extra="full-stack",
        required_modules=_CORE_MODULES
        + ("sentence_transformers", "networkx", "redis", "psycopg"),
        required_artifacts=(
            "behavioral_ontology",
            "full_corpus_v09",
            "production_graph",
            "dense_field_embeddings",
        ),
        recommended_artifacts=(
            "literature_findings",
            "paper_dataset_links",
            "canonical_benchmark_queries",
            "canonical_qrels",
            "ablation_corpus",
            "current_artifact_manifest",
        ),
        produced_artifacts=(),
        commands=(
            "python -m pip install -e '.[full-stack]'",
            "cd apps/web && npm ci",
            "neural-search profile check full-stack",
            "NEURAL_SEARCH_PROFILE=full-stack make api",
            "make web",
        ),
        resource_notes=(
            "Designed for maintainers and infrastructure hosts, not first-time users.",
            "Postgres/Redis are optional for many read-only paths but part of this profile.",
            "GPU acceleration is recommended for rebuilding dense embeddings.",
        ),
    ),
}


def list_profiles() -> list[dict[str, Any]]:
    """Return compact metadata for all supported profiles."""

    return [
        {
            "name": profile.name,
            "description": profile.description,
            "install_extra": profile.install_extra,
        }
        for profile in PROFILES.values()
    ]


def get_profile(name: str) -> ExecutionProfile:
    """Return one profile or raise a useful error."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(
            f"Unknown execution profile {name!r}; choose one of: {choices}"
        ) from exc


def list_artifacts() -> list[dict[str, Any]]:
    """Return every registered artifact contract."""

    return [asdict(spec) for spec in ARTIFACTS.values()]


def _artifact_path(spec: ArtifactSpec) -> Path:
    return PROJECT_ROOT / spec.path


def _directory_has_file(path: Path) -> bool:
    return any(candidate.is_file() for candidate in path.rglob("*"))


def artifact_status(artifact_id: str) -> dict[str, Any]:
    """Inspect one registered artifact without mutating the checkout."""

    try:
        spec = ARTIFACTS[artifact_id]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact: {artifact_id}") from exc

    path = _artifact_path(spec)
    exists = path.exists()
    usable = exists
    payload: dict[str, Any] = {
        **asdict(spec),
        "exists": exists,
        "usable": usable,
        "absolute_path": str(path),
    }

    if exists:
        stat = path.stat()
        is_directory = path.is_dir()
        payload["is_directory"] = is_directory
        payload["size_bytes"] = stat.st_size if path.is_file() else None
        payload["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

        if path.is_file() and stat.st_size == 0:
            usable = False
        elif is_directory and spec.requires_content and not _directory_has_file(path):
            usable = False

        payload["usable"] = usable
        if usable:
            payload["state"] = "present"
        elif spec.kind in {"committed_fixture", "frozen_evaluation"}:
            payload["state"] = "empty_portable_asset"
        else:
            payload["state"] = "empty_generated_asset"
    elif spec.kind in {"committed_fixture", "frozen_evaluation"}:
        payload["state"] = "missing_portable_asset"
    else:
        payload["state"] = "missing_generated_asset"

    return payload


def _module_status(module_names: tuple[str, ...]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in module_names}


def profile_status(name: str) -> dict[str, Any]:
    """Return readiness and remediation information for one execution profile."""

    profile = get_profile(name)
    modules = _module_status(profile.required_modules)
    required = [artifact_status(artifact_id) for artifact_id in profile.required_artifacts]
    recommended = [
        artifact_status(artifact_id) for artifact_id in profile.recommended_artifacts
    ]
    produced = [artifact_status(artifact_id) for artifact_id in profile.produced_artifacts]
    required_ready = all(item["usable"] for item in required)
    dependencies_ready = all(modules.values())
    missing_required = [item for item in required if not item["usable"]]
    missing_modules = [module for module, present in modules.items() if not present]
    remediation = [
        item["repair_command"]
        for item in missing_required
        if item.get("repair_command")
    ]
    return {
        "profile": asdict(profile),
        "ready": required_ready and dependencies_ready,
        "dependencies_ready": dependencies_ready,
        "required_artifacts_ready": required_ready,
        "required_modules": modules,
        "missing_modules": missing_modules,
        "required_artifacts": required,
        "recommended_artifacts": recommended,
        "produced_artifacts": produced,
        "remediation_commands": remediation,
        "source_checkout": (PROJECT_ROOT / "pyproject.toml").is_file(),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reproducibility_manifest(
    name: str,
    *,
    checksum_limit_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    """Build a portable manifest for reproducing one profile's environment.

    Files up to ``checksum_limit_bytes`` are SHA-256 hashed. Larger generated
    artifacts record size and modification time without forcing an expensive
    full-file read on every profile check.
    """

    status = profile_status(name)
    ordered_ids: list[str] = []
    for key in ("required_artifacts", "recommended_artifacts", "produced_artifacts"):
        for artifact_id in status["profile"][key]:
            if artifact_id not in ordered_ids:
                ordered_ids.append(artifact_id)

    artifacts: list[dict[str, Any]] = []
    for artifact_id in ordered_ids:
        item = artifact_status(artifact_id)
        path = PROJECT_ROOT / item["path"]
        if item["exists"] and path.is_file():
            size = int(item.get("size_bytes") or 0)
            if size <= checksum_limit_bytes:
                item["sha256"] = _sha256(path)
            else:
                item["sha256"] = None
                item["checksum_note"] = (
                    f"Skipped because file exceeds {checksum_limit_bytes} bytes"
                )
        artifacts.append(item)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": name,
        "profile_ready": status["ready"],
        "git_commit": _git_commit(),
        "neural_search_profile_env": os.getenv("NEURAL_SEARCH_PROFILE"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "required_modules": status["required_modules"],
        "artifacts": artifacts,
        "commands": status["profile"]["commands"],
        "resource_notes": status["profile"]["resource_notes"],
    }
