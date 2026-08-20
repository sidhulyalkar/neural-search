"""Machine-readable execution profiles, artifact contracts, and compatibility health."""

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

from neural_search.runtime.bundles import lock_snapshot, resolve_locked_artifact
from neural_search.runtime.lineage import ArtifactLineage, read_lineage

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
    version: str | None = None
    compatibility_group: str | None = None
    derived_from: tuple[str, ...] = ()
    capability: str | None = None
    bundle_name: str | None = None


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
        version="repo",
        capability="behavioral_ontology",
    ),
    "demo_datasets": ArtifactSpec(
        id="demo_datasets",
        path="data/seed/demo_datasets.yaml",
        kind="committed_fixture",
        description="Curated dataset fixtures for demos, tests, and onboarding.",
        version="repo",
        capability="demo_search",
    ),
    "demo_papers": ArtifactSpec(
        id="demo_papers",
        path="data/seed/demo_papers.yaml",
        kind="committed_fixture",
        description="Curated literature fixtures linked to demo datasets.",
        version="repo",
        capability="demo_literature",
    ),
    "method_registry": ArtifactSpec(
        id="method_registry",
        path="data/methods/method_registry.yaml",
        kind="committed_fixture",
        description="Analysis-family to neuroscience-method registry used by reanalysis planning.",
        version="2.0",
        capability="method_registry",
    ),
    "methods_taxonomy": ArtifactSpec(
        id="methods_taxonomy",
        path="data/methods/methods_taxonomy.yaml",
        kind="committed_fixture",
        description="Method assumptions, limitations, mathematical basis, and related methods.",
        version="repo",
        capability="method_taxonomy",
    ),
    "canonical_benchmark_queries": ArtifactSpec(
        id="canonical_benchmark_queries",
        path="data/eval/benchmark_queries_canonical.yaml",
        kind="frozen_evaluation",
        description="Canonical retrieval benchmark queries committed with the repository.",
        version="canonical-v1",
        compatibility_group="evaluation:canonical-v1",
        capability="canonical_evaluation",
    ),
    "canonical_qrels": ArtifactSpec(
        id="canonical_qrels",
        path="data/qrels/qrels.canonical.jsonl",
        kind="frozen_evaluation",
        description="Canonical graded relevance judgments used by evaluation tooling.",
        version="canonical-v1",
        compatibility_group="evaluation:canonical-v1",
        capability="canonical_evaluation",
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
        version="canonical-v1",
        compatibility_group="evaluation:canonical-v1",
        derived_from=("canonical_benchmark_queries",),
        capability="full_ablation_corpus",
        bundle_name="neural-search-evaluator",
    ),
    "raw_corpus_inputs": ArtifactSpec(
        id="raw_corpus_inputs",
        path="data/raw",
        kind="generated_local",
        description="Downloaded source payloads used to rebuild the full normalized corpus.",
        producer="source-specific acquisition scripts under scripts/corpus/",
        requires_content=True,
        capability="raw_source_payloads",
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
        version="0.9",
        compatibility_group="corpus:v09",
        derived_from=("raw_corpus_inputs",),
        capability="real_dataset_corpus",
        bundle_name="neural-search-researcher",
    ),
    "production_graph": ArtifactSpec(
        id="production_graph",
        path="data/graph/neural_search_graph.real_corpus.json",
        kind="generated_local",
        description="Knowledge graph derived from the current full corpus.",
        producer="scripts/rebuild_full_corpus_graph.py",
        repair_command="python scripts/rebuild_full_corpus_graph.py",
        version="0.9",
        compatibility_group="corpus:v09",
        derived_from=("full_corpus_v09",),
        capability="knowledge_graph",
        bundle_name="neural-search-researcher",
    ),
    "dense_field_embeddings": ArtifactSpec(
        id="dense_field_embeddings",
        path="data/embeddings/real_all.dense.field_embeddings.jsonl",
        kind="generated_local",
        description="Dense field-level embedding cache for semantic retrieval.",
        producer="scripts/recompute_embeddings.py --provider dense",
        repair_command="python scripts/recompute_embeddings.py --provider dense",
        version="0.9",
        compatibility_group="corpus:v09",
        derived_from=("full_corpus_v09",),
        capability="dense_semantic_index",
        bundle_name="neural-search-researcher",
    ),
    "specter2_embeddings": ArtifactSpec(
        id="specter2_embeddings",
        path="data/embeddings/specter2_corpus.jsonl",
        kind="generated_local",
        description="Scientific-text SPECTER2 embedding cache for corpus comparisons.",
        producer="scripts/eval/run_specter2_comparison.py --build-embeddings",
        repair_command="python scripts/eval/run_specter2_comparison.py --build-embeddings",
        version="0.9",
        compatibility_group="corpus:v09",
        derived_from=("full_corpus_v09",),
        capability="specter2_index",
        bundle_name="neural-search-researcher",
    ),
    "literature_findings": ArtifactSpec(
        id="literature_findings",
        path="artifacts/literature/findings_v1.jsonl",
        kind="generated_local",
        description="Structured findings extracted from the literature corpus.",
        producer="literature extraction pipeline",
        version="1",
        compatibility_group="literature:v1",
        capability="literature_findings",
        bundle_name="neural-search-literature",
    ),
    "paper_dataset_links": ArtifactSpec(
        id="paper_dataset_links",
        path="artifacts/literature/paper_dataset_links.jsonl",
        kind="generated_local",
        description="Dataset-to-paper linkage evidence used by search and the graph.",
        producer="literature linking pipeline",
        version="1",
        compatibility_group="literature:v1",
        derived_from=("full_corpus_v09",),
        capability="paper_dataset_links",
        bundle_name="neural-search-literature",
    ),
    "ner_method_graph": ArtifactSpec(
        id="ner_method_graph",
        path="artifacts/ner/ner_kg.jsonl",
        kind="generated_local",
        description="Paper-to-method evidence extracted into the NER knowledge graph.",
        producer="neural_search.ingestion.ner_builder pipeline",
        version="1",
        compatibility_group="literature:v1",
        capability="paper_method_evidence",
        bundle_name="neural-search-literature",
    ),
    "coverage_ledger": ArtifactSpec(
        id="coverage_ledger",
        path="data/coverage/ledger.duckdb",
        kind="generated_local",
        description="DuckDB coverage ledger used for source/corpus gap-aware retrieval signals.",
        producer="scripts/coverage/build_duckdb_ledger.py",
        repair_command="python scripts/coverage/build_duckdb_ledger.py",
        version="0.9",
        compatibility_group="corpus:v09",
        derived_from=("full_corpus_v09",),
        capability="coverage_gap_boost",
        bundle_name="neural-search-researcher",
    ),
    "neurosynth_raw": ArtifactSpec(
        id="neurosynth_raw",
        path="data/neurosynth",
        kind="generated_local",
        description="NeuroSynth coordinate/term source data for neuroimaging enrichment.",
        producer="scripts/ingestion/download_neurosynth.py",
        repair_command="python scripts/ingestion/download_neurosynth.py",
        requires_content=True,
        version="upstream",
        capability="neurosynth_enrichment",
        bundle_name="neural-search-neuroimaging",
    ),
    "reanalysis_affordance_report": ArtifactSpec(
        id="reanalysis_affordance_report",
        path="reports/eval/reanalysis_affordance_report.json",
        kind="derived_report",
        description="Corpus-level analysis-affordance and reanalysis feasibility report.",
        producer="scripts/eval/build_reanalysis_affordance_report.py",
        repair_command="python scripts/eval/build_reanalysis_affordance_report.py",
        version="1",
        compatibility_group="corpus:v09",
        derived_from=("full_corpus_v09",),
        capability="reanalysis_affordance_report",
    ),
    "current_artifact_manifest": ArtifactSpec(
        id="current_artifact_manifest",
        path="reports/eval/current_artifact_manifest.json",
        kind="derived_report",
        description="Computed current-state scientific artifact summary.",
        producer="scripts/build_artifact_manifest.py",
        repair_command="python scripts/build_artifact_manifest.py",
        version="2",
        derived_from=(
            "full_corpus_v09",
            "production_graph",
            "canonical_benchmark_queries",
            "canonical_qrels",
        ),
        capability="scientific_manifest",
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
        recommended_artifacts=("method_registry", "methods_taxonomy"),
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
            "method_registry",
            "methods_taxonomy",
            "production_graph",
            "dense_field_embeddings",
            "literature_findings",
            "paper_dataset_links",
            "ner_method_graph",
            "coverage_ledger",
            "specter2_embeddings",
            "neurosynth_raw",
        ),
        produced_artifacts=(),
        commands=(
            "python -m pip install -e '.[researcher]'",
            "neural-search artifacts fetch neural-search-researcher@<version>",
            "neural-search profile check researcher",
            "NEURAL_SEARCH_PROFILE=researcher make api",
            "make web",
        ),
        resource_notes=(
            "CPU search is supported; a GPU materially reduces dense-index build time.",
            "Generated assets may be installed from a verified bundle or built locally.",
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
        recommended_artifacts=("method_registry", "methods_taxonomy"),
        produced_artifacts=(
            "full_corpus_v09",
            "production_graph",
            "dense_field_embeddings",
            "coverage_ledger",
        ),
        commands=(
            "python -m pip install -e '.[corpus-builder]'",
            "python scripts/corpus/build_full_corpus.py",
            "python scripts/rebuild_full_corpus_graph.py",
            "python scripts/recompute_embeddings.py --provider dense",
            "python scripts/coverage/build_duckdb_ledger.py",
        ),
        resource_notes=(
            "Network access is needed to refresh raw source payloads.",
            "Dense embedding generation is GPU-friendly and can be slow on CPU-only hosts.",
            "Stamp generated artifacts with parent lineage before publishing a bundle.",
        ),
    ),
    "evaluator": ExecutionProfile(
        name="evaluator",
        description=(
            "Reproduce portable retrieval checks and, when local assets are available, "
            "run qrels metrics, ablations, calibration, and scientific regression gates."
        ),
        install_extra="evaluator",
        required_modules=_CORE_MODULES + ("pytest", "networkx"),
        required_artifacts=("canonical_benchmark_queries", "canonical_qrels"),
        recommended_artifacts=(
            "ablation_corpus",
            "production_graph",
            "dense_field_embeddings",
            "reanalysis_affordance_report",
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
            "Maintainer environment spanning web/API, real corpus, graph, literature, "
            "reanalysis, evaluation, databases, and optional scientific enrichments."
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
        recommended_artifacts=tuple(
            artifact_id
            for artifact_id in ARTIFACTS
            if artifact_id
            not in {
                "behavioral_ontology",
                "full_corpus_v09",
                "production_graph",
                "dense_field_embeddings",
                "demo_datasets",
                "demo_papers",
                "raw_corpus_inputs",
            }
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
    return [
        {
            "name": profile.name,
            "description": profile.description,
            "install_extra": profile.install_extra,
        }
        for profile in PROFILES.values()
    ]


def get_profile(name: str) -> ExecutionProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(
            f"Unknown execution profile {name!r}; choose one of: {choices}"
        ) from exc


def list_artifacts() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in ARTIFACTS.values()]


def _artifact_path(spec: ArtifactSpec) -> tuple[Path, str]:
    locked = resolve_locked_artifact(spec.id)
    if locked is not None:
        return locked, "bundle_lock"
    return PROJECT_ROOT / spec.path, "repository_path"


def _directory_has_file(path: Path) -> bool:
    return any(candidate.is_file() for candidate in path.rglob("*"))


def _lineage_payload(lineage: ArtifactLineage | None) -> dict[str, Any] | None:
    return lineage.to_dict() if lineage else None


def artifact_status(artifact_id: str) -> dict[str, Any]:
    """Inspect one registered artifact and its content-addressed identity."""

    try:
        spec = ARTIFACTS[artifact_id]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact: {artifact_id}") from exc

    path, resolution_source = _artifact_path(spec)
    exists = path.exists()
    usable = exists
    payload: dict[str, Any] = {
        **asdict(spec),
        "exists": exists,
        "usable": usable,
        "absolute_path": str(path),
        "resolution_source": resolution_source,
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

    lineage = read_lineage(path) if usable else None
    payload["lineage"] = _lineage_payload(lineage)
    payload["lineage_state"] = "tracked" if lineage else ("untracked" if usable else "absent")
    if lineage and spec.version and lineage.artifact_version != spec.version:
        payload["lineage_state"] = "version_mismatch"
    if (
        lineage
        and spec.compatibility_group
        and lineage.compatibility_group
        and lineage.compatibility_group != spec.compatibility_group
    ):
        payload["lineage_state"] = "compatibility_group_mismatch"
    return payload


def _module_status(module_names: tuple[str, ...]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in module_names}


def compatibility_status(artifact_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Validate known lineage relationships without pretending untracked assets are valid."""

    statuses = {artifact_id: artifact_status(artifact_id) for artifact_id in artifact_ids}
    lineages: dict[str, ArtifactLineage | None] = {}
    for artifact_id, item in statuses.items():
        raw = item.get("lineage")
        lineages[artifact_id] = ArtifactLineage.from_dict(raw) if isinstance(raw, dict) else None

    incompatible: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []

    for artifact_id, item in statuses.items():
        if not item["usable"]:
            continue
        spec = ARTIFACTS[artifact_id]
        lineage = lineages[artifact_id]
        if lineage is None:
            if spec.derived_from:
                unknown.append(
                    {
                        "artifact_id": artifact_id,
                        "reason": "lineage_sidecar_missing",
                        "expected_parents": list(spec.derived_from),
                    }
                )
            continue

        issues: list[str] = []
        if spec.version and lineage.artifact_version != spec.version:
            issues.append(
                f"artifact_version:{lineage.artifact_version}!={spec.version}"
            )
        if (
            spec.compatibility_group
            and lineage.compatibility_group
            and lineage.compatibility_group != spec.compatibility_group
        ):
            issues.append(
                "compatibility_group:"
                f"{lineage.compatibility_group}!={spec.compatibility_group}"
            )

        for parent_id in spec.derived_from:
            expected_parent_lineage = lineage.derived_from.get(parent_id)
            parent_status = statuses.get(parent_id)
            if parent_status is None:
                parent_status = artifact_status(parent_id)
                statuses[parent_id] = parent_status
                raw_parent = parent_status.get("lineage")
                lineages[parent_id] = (
                    ArtifactLineage.from_dict(raw_parent)
                    if isinstance(raw_parent, dict)
                    else None
                )
            parent_lineage = lineages.get(parent_id)
            if not parent_status["usable"]:
                issues.append(f"parent_missing:{parent_id}")
            elif expected_parent_lineage is None:
                issues.append(f"parent_not_declared:{parent_id}")
            elif parent_lineage is None:
                unknown.append(
                    {
                        "artifact_id": artifact_id,
                        "reason": f"parent_lineage_untracked:{parent_id}",
                    }
                )
            elif parent_lineage.lineage_id != expected_parent_lineage:
                issues.append(f"parent_lineage_mismatch:{parent_id}")

        result = {
            "artifact_id": artifact_id,
            "lineage_id": lineage.lineage_id,
            "issues": issues,
        }
        checked.append(result)
        if issues:
            incompatible.append(result)

    state = "incompatible" if incompatible else ("unknown" if unknown else "compatible")
    return {
        "state": state,
        "compatible": not incompatible,
        "checked": checked,
        "unknown": unknown,
        "incompatible": incompatible,
    }


def capability_status(profile_name: str | None = None) -> dict[str, Any]:
    """Expose user-facing scientific capabilities from the artifact contract."""

    profile = PROFILES.get(profile_name) if profile_name else None
    visible_ids = set(ARTIFACTS)
    if profile:
        visible_ids = set(
            profile.required_artifacts
            + profile.recommended_artifacts
            + profile.produced_artifacts
        )

    capabilities: list[dict[str, Any]] = []
    for artifact_id, spec in ARTIFACTS.items():
        if artifact_id not in visible_ids or not spec.capability:
            continue
        status = artifact_status(artifact_id)
        lineage = status.get("lineage") or {}
        capabilities.append(
            {
                "capability": spec.capability,
                "artifact_id": artifact_id,
                "available": bool(status["usable"]),
                "state": status["state"],
                "version": lineage.get("artifact_version") or spec.version,
                "lineage_id": lineage.get("lineage_id"),
                "resolution_source": status["resolution_source"],
                "description": spec.description,
            }
        )
    return {
        "profile": profile_name,
        "available": sum(1 for item in capabilities if item["available"]),
        "total": len(capabilities),
        "capabilities": capabilities,
    }


def profile_status(name: str) -> dict[str, Any]:
    """Return readiness, compatibility, and remediation for one profile."""

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
    all_ids = list(
        dict.fromkeys(
            profile.required_artifacts
            + profile.recommended_artifacts
            + profile.produced_artifacts
        )
    )
    compatibility = compatibility_status(all_ids)
    required_incompatible = {
        item["artifact_id"]
        for item in compatibility["incompatible"]
        if item["artifact_id"] in profile.required_artifacts
    }
    ready = required_ready and dependencies_ready and not required_incompatible
    recommended_missing = [item["id"] for item in recommended if not item["usable"]]
    health = "unhealthy" if not ready else (
        "degraded"
        if recommended_missing or compatibility["state"] == "unknown"
        else "ready"
    )
    return {
        "profile": asdict(profile),
        "ready": ready,
        "health": health,
        "dependencies_ready": dependencies_ready,
        "required_artifacts_ready": required_ready,
        "required_modules": modules,
        "missing_modules": missing_modules,
        "required_artifacts": required,
        "recommended_artifacts": recommended,
        "recommended_missing": recommended_missing,
        "produced_artifacts": produced,
        "remediation_commands": remediation,
        "compatibility": compatibility,
        "capabilities": capability_status(name),
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
    """Build a portable manifest including bundle pins and artifact lineages."""

    status = profile_status(name)
    ordered_ids: list[str] = []
    for key in ("required_artifacts", "recommended_artifacts", "produced_artifacts"):
        for artifact_id in status["profile"][key]:
            if artifact_id not in ordered_ids:
                ordered_ids.append(artifact_id)

    artifacts: list[dict[str, Any]] = []
    for artifact_id in ordered_ids:
        item = artifact_status(artifact_id)
        path = Path(item["absolute_path"])
        if item["exists"] and path.is_file() and not item.get("lineage"):
            size = int(item.get("size_bytes") or 0)
            if size <= checksum_limit_bytes:
                item["sha256"] = _sha256(path)
            else:
                item["sha256"] = None
                item["checksum_note"] = (
                    f"Skipped because file exceeds {checksum_limit_bytes} bytes"
                )
        elif item.get("lineage"):
            item["sha256"] = item["lineage"]["content_sha256"]
        artifacts.append(item)

    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": name,
        "profile_ready": status["ready"],
        "profile_health": status["health"],
        "git_commit": _git_commit(),
        "neural_search_profile_env": os.getenv("NEURAL_SEARCH_PROFILE"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "required_modules": status["required_modules"],
        "compatibility": status["compatibility"],
        "capabilities": status["capabilities"],
        "artifact_lock": lock_snapshot(),
        "artifacts": artifacts,
        "commands": status["profile"]["commands"],
        "resource_notes": status["profile"]["resource_notes"],
    }
