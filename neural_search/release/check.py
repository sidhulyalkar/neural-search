"""Release-readiness summary generation for Neural Search."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RELEASE_REPORT_DIR = ROOT / "data" / "reports" / "release"

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


def build_release_summary() -> dict[str, Any]:
    artifact_summary: dict[str, Any] = {}
    missing_artifacts: list[str] = []
    for name, path in ARTIFACTS.items():
        exists = path.exists()
        if not exists:
            missing_artifacts.append(name)
        artifact_summary[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": exists,
            "line_count": _line_count(path),
            "graph_counts": _graph_counts(path) if "graph" in name else None,
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
            benchmark_summary[suite] = {"path": str(path.relative_to(ROOT)), "exists": False}
            continue
        suite_violations = sum(
            len(query.get("hard_negative_violations", []))
            for query in report.get("queries", [])
        )
        hard_negative_violations += suite_violations
        benchmark_summary[suite] = {
            "path": str(path.relative_to(ROOT)),
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
        f"missing benchmark report: {suite}" for suite in missing_benchmarks
    ]
    if hard_negative_violations:
        known_failures.append(
            f"hard-negative violations across release benchmarks: {hard_negative_violations}"
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "artifact_versions": {
            "demo_corpus": "demo_v05",
            "real_corpus": "real_v07",
            "graph_version": "v0.5.0",
            "embedding_provider": "hashing",
        },
        "artifacts": artifact_summary,
        "benchmarks": benchmark_summary,
        "known_failures": known_failures,
        "release_ready": not missing_artifacts
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
            f"lines={artifact.get('line_count')}{graph_text}"
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
    return "\n".join(lines) + "\n"


def write_release_summary(out_dir: str | Path = RELEASE_REPORT_DIR) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_release_summary()
    json_path = output / "release_summary.json"
    md_path = output / "release_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_summary(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate release-readiness reports.")
    parser.add_argument("--out", default=str(RELEASE_REPORT_DIR))
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write reports and do not fail on missing artifacts or benchmark violations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_release_summary(args.out)
    summary = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print(json.dumps(paths, indent=2, sort_keys=True))
    if args.summary_only:
        return 0
    return 0 if summary.get("release_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
