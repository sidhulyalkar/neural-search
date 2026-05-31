"""Release-readiness summary generation for Neural Search."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_search.graph.quality import audit_graph_quality

ROOT = Path(__file__).resolve().parents[2]
RELEASE_REPORT_DIR = ROOT / "data" / "reports" / "release"
READINESS_REPORT = (
    ROOT / "data" / "reports" / "readiness" / "scientific_readiness_report.json"
)

ARTIFACTS = {
    "demo_datasets": ROOT / "data" / "corpus" / "normalized" / "demo_v05.datasets.jsonl",
    "demo_papers": ROOT / "data" / "corpus" / "normalized" / "demo_v05.papers.jsonl",
    "demo_graph": ROOT / "data" / "graph" / "neural_search_graph.demo_v05.json",
    "demo_embeddings": ROOT / "data" / "embeddings" / "demo_v05.field_embeddings.jsonl",
    "real_datasets": ROOT / "data" / "corpus" / "normalized" / "real_v07.datasets.jsonl",
    "real_papers": ROOT / "data" / "corpus" / "normalized" / "real_v07.papers.jsonl",
    "real_claims": ROOT / "data" / "corpus" / "claims" / "real_v07.claims.jsonl",
    "real_graph": ROOT / "data" / "graph" / "neural_search_graph.real_v07.json",
    "real_embeddings": ROOT / "data" / "embeddings" / "real_v07.field_embeddings.jsonl",
}

ARTIFACT_INPUTS = {
    "demo_datasets": [ROOT / "data" / "seed" / "demo_datasets.yaml"],
    "demo_papers": [ROOT / "data" / "seed" / "demo_papers.yaml"],
    "demo_graph": [
        ROOT / "data" / "corpus" / "normalized" / "demo_v05.datasets.jsonl",
        ROOT / "data" / "corpus" / "normalized" / "demo_v05.papers.jsonl",
    ],
    "demo_embeddings": [ROOT / "data" / "corpus" / "normalized" / "demo_v05.records.jsonl"],
    "real_datasets": [
        ROOT / "data" / "corpus" / "manifests" / "real_v07.yaml",
        ROOT / "data" / "corpus" / "fixtures" / "real_v07",
    ],
    "real_papers": [
        ROOT / "data" / "corpus" / "manifests" / "real_v07.yaml",
        ROOT / "data" / "corpus" / "fixtures" / "real_v07",
    ],
    "real_claims": [
        ROOT / "data" / "corpus" / "manifests" / "real_v07.yaml",
        ROOT / "data" / "corpus" / "fixtures" / "real_v07",
    ],
    "real_graph": [
        ROOT / "data" / "corpus" / "normalized" / "real_v07.datasets.jsonl",
        ROOT / "data" / "corpus" / "normalized" / "real_v07.papers.jsonl",
    ],
    "real_embeddings": [ROOT / "data" / "corpus" / "normalized" / "real_v07.records.jsonl"],
}

BENCHMARK_REPORTS = {
    "demo_v02": ROOT / "data" / "eval" / "results" / "demo_v02" / "latest_eval_report.json",
    "adversarial": ROOT / "data" / "eval" / "results" / "adversarial" / "latest_eval_report.json",
    "real_v07": ROOT / "data" / "eval" / "results" / "real_v07" / "latest_eval_report.json",
}


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _line_count(path: Path) -> int | None:
    if not path.exists() or path.is_dir():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _graph_counts(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "nodes": len(payload.get("nodes", {})),
        "edges": len(payload.get("edges", {})),
    }


def _load_benchmark(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_graph_quality_artifact(path: Path) -> dict[str, Any]:
    """Summarize graph QA findings for release reports without changing release gates."""

    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = audit_graph_quality(payload)
    issue_counts = Counter(issue.code for issue in report.issues)
    return {
        "available": True,
        "passed": report.passed,
        "node_count": report.node_count,
        "edge_count": report.edge_count,
        "issue_count": report.issue_count,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _load_source_quality(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_quality = payload.get("source_quality", {})
    if not isinstance(source_quality, dict):
        return {"available": False}
    return {"available": True, **source_quality}


def _source_quality_warnings(source_quality: dict[str, Any]) -> list[str]:
    if not source_quality.get("available"):
        return ["readiness report unavailable; source quality not included"]
    warnings: list[str] = []
    trust_counts = source_quality.get("trust_level_counts", {})
    if trust_counts.get("unknown", 0):
        warnings.append("source quality includes records from unknown sources")
    if trust_counts.get("low", 0):
        warnings.append("source quality includes low-trust fixture/demo records")
    if float(source_quality.get("mean_quality_score", 0.0) or 0.0) < 0.7:
        warnings.append("mean source quality is below 0.70")
    if int(source_quality.get("warning_count", 0) or 0):
        warnings.append("source quality report contains record-level warnings")
    return warnings


def _graph_quality_warnings(graph_quality: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for name, summary in sorted(graph_quality.items()):
        if not summary.get("available"):
            continue
        error_count = int(summary.get("error_count", 0) or 0)
        warning_count = int(summary.get("warning_count", 0) or 0)
        if error_count:
            warnings.append(f"{name} graph QA has {error_count} error(s)")
        if warning_count:
            warnings.append(f"{name} graph QA has {warning_count} warning(s)")
    return warnings


def _iter_existing_inputs(paths: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            inputs.extend(child for child in path.rglob("*") if child.is_file())
        else:
            inputs.append(path)
    return inputs


def _staleness(path: Path, inputs: list[Path]) -> dict[str, Any]:
    if not path.exists():
        return {"stale": None, "newest_input": None, "artifact_mtime": None}
    existing_inputs = _iter_existing_inputs(inputs)
    if not existing_inputs:
        return {
            "stale": False,
            "newest_input": None,
            "artifact_mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
    newest_input = max(existing_inputs, key=lambda item: item.stat().st_mtime)
    artifact_mtime = path.stat().st_mtime
    newest_input_mtime = newest_input.stat().st_mtime
    return {
        "stale": newest_input_mtime > artifact_mtime,
        "newest_input": _display_path(newest_input),
        "newest_input_mtime": datetime.fromtimestamp(newest_input_mtime, UTC).isoformat(),
        "artifact_mtime": datetime.fromtimestamp(artifact_mtime, UTC).isoformat(),
    }


def build_release_summary(
    readiness_report: str | Path | None = READINESS_REPORT,
) -> dict[str, Any]:
    artifact_summary: dict[str, Any] = {}
    missing_artifacts: list[str] = []
    stale_artifacts: list[str] = []
    for name, path in ARTIFACTS.items():
        exists = path.exists()
        if not exists:
            missing_artifacts.append(name)
        staleness = _staleness(path, ARTIFACT_INPUTS.get(name, []))
        if staleness["stale"]:
            stale_artifacts.append(name)
        artifact_summary[name] = {
            "path": _display_path(path),
            "exists": exists,
            "line_count": _line_count(path),
            "sha256": _sha256(path),
            "graph_counts": _graph_counts(path) if "graph" in name else None,
            "staleness": staleness,
            "modified_at": (
                datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                if exists
                else None
            ),
        }

    benchmark_summary: dict[str, Any] = {}
    hard_negative_violations = 0
    missing_benchmarks: list[str] = []
    for suite, path in BENCHMARK_REPORTS.items():
        report = _load_benchmark(path)
        if report is None:
            missing_benchmarks.append(suite)
            benchmark_summary[suite] = {"path": _display_path(path), "exists": False}
            continue
        suite_violations = sum(
            len(query.get("hard_negative_violations", []))
            for query in report.get("queries", [])
        )
        hard_negative_violations += suite_violations
        benchmark_summary[suite] = {
            "path": _display_path(path),
            "exists": True,
            "total_queries": report.get("total_queries"),
            "mean_precision_at_5": report.get("mean_precision_at_5"),
            "mean_label_recall_at_10": report.get("mean_label_recall_at_10"),
            "hard_negative_violations": suite_violations,
            "failed_queries": [
                query.get("query_id")
                for query in report.get("queries", [])
                if query.get("why_failed")
            ],
        }

    known_failures = [
        f"missing artifact: {name}" for name in missing_artifacts
    ] + [
        f"stale artifact: {name}" for name in stale_artifacts
    ] + [
        f"missing benchmark report: {suite}" for suite in missing_benchmarks
    ]
    if hard_negative_violations:
        known_failures.append(
            f"hard-negative violations across release benchmarks: {hard_negative_violations}"
        )

    graph_quality = {
        name: summarize_graph_quality_artifact(path)
        for name, path in ARTIFACTS.items()
        if "graph" in name
    }
    source_quality = _load_source_quality(
        Path(readiness_report) if readiness_report else None
    )
    release_warnings = [
        *_source_quality_warnings(source_quality),
        *_graph_quality_warnings(graph_quality),
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "artifact_versions": {
            "demo_corpus": "demo_v05",
            "real_corpus": "real_v07",
            "graph_version": "v0.5.0",
            "embedding_provider": "hashing",
        },
        "artifacts": artifact_summary,
        "benchmarks": benchmark_summary,
        "graph_quality": graph_quality,
        "source_quality": source_quality,
        "release_warnings": release_warnings,
        "known_failures": known_failures,
        "release_ready": not missing_artifacts
        and not stale_artifacts
        and not missing_benchmarks
        and hard_negative_violations == 0,
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Neural Search Release Summary",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Commit: `{summary.get('commit') or 'unknown'}`",
        f"- Release ready: `{summary['release_ready']}`",
        f"- Real corpus: `{summary['artifact_versions']['real_corpus']}`",
        f"- Embedding provider: `{summary['artifact_versions']['embedding_provider']}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in summary["artifacts"].items():
        counts = artifact.get("graph_counts") or {}
        graph_text = (
            f", nodes={counts.get('nodes')}, edges={counts.get('edges')}"
            if counts
            else ""
        )
        lines.append(
            f"- `{name}`: exists={artifact['exists']}, "
            f"lines={artifact.get('line_count')}, "
            f"stale={artifact.get('staleness', {}).get('stale')}{graph_text}"
        )

    lines.extend(["", "## Benchmarks", ""])
    for suite, benchmark in summary["benchmarks"].items():
        lines.append(
            f"- `{suite}`: exists={benchmark.get('exists')}, "
            f"P@5={benchmark.get('mean_precision_at_5')}, "
            f"label_recall@10={benchmark.get('mean_label_recall_at_10')}, "
            f"hard_negatives={benchmark.get('hard_negative_violations')}"
        )

    lines.extend(["", "## Known Failures", ""])
    if summary["known_failures"]:
        lines.extend(f"- {failure}" for failure in summary["known_failures"])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Release Warnings", ""])
    if summary.get("release_warnings"):
        lines.extend(f"- {warning}" for warning in summary["release_warnings"])
    else:
        lines.append("- None recorded.")

    source_quality = summary.get("source_quality", {})
    graph_quality = summary.get("graph_quality", {})
    lines.extend(["", "## Graph Quality", ""])
    for name, quality in sorted(graph_quality.items()):
        lines.append(
            f"- `{name}`: available={quality.get('available', False)}, "
            f"errors={quality.get('error_count')}, "
            f"warnings={quality.get('warning_count')}"
        )

    lines.extend(
        [
            "",
            "## Source Quality",
            "",
            f"- Available: {source_quality.get('available', False)}",
            f"- Mean quality score: {source_quality.get('mean_quality_score')}",
            f"- Trust levels: {source_quality.get('trust_level_counts', {})}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_release_summary(
    out_dir: str | Path = RELEASE_REPORT_DIR,
    *,
    readiness_report: str | Path | None = READINESS_REPORT,
) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_release_summary(readiness_report=readiness_report)
    json_path = output / "release_summary.json"
    md_path = output / "release_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_summary(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate release-readiness reports.")
    parser.add_argument("--out", default=str(RELEASE_REPORT_DIR))
    parser.add_argument(
        "--readiness-report",
        default=str(READINESS_REPORT),
        help="Optional scientific readiness JSON report for non-failing source warnings.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write reports and do not fail on missing artifacts or benchmark violations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_release_summary(args.out, readiness_report=args.readiness_report)
    summary = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print(json.dumps(paths, indent=2, sort_keys=True))
    if args.summary_only:
        return 0
    return 0 if summary.get("release_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
