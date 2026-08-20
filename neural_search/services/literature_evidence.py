"""Application service for literature and finding evidence retrieval."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_search.literature.search import search_findings, search_papers
from neural_search.runtime import artifact_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_SHARDS = (
    PROJECT_ROOT / "data" / "corpus" / "normalized" / "openalex_neuro"
)


class LiteratureEvidenceService:
    """Search scientific evidence without coupling callers to opaque failures."""

    def _artifact_path(self, artifact_id: str) -> Path | None:
        status = artifact_status(artifact_id)
        if not status["usable"]:
            return None
        return Path(status["absolute_path"])

    def source_state(self) -> dict[str, Any]:
        findings = artifact_status("literature_findings")
        links = artifact_status("paper_dataset_links")
        paper_shards_available = (
            DEFAULT_PAPER_SHARDS.is_dir()
            and any(DEFAULT_PAPER_SHARDS.glob("*.jsonl"))
        )
        return {
            "paper_shards": {
                "available": paper_shards_available,
                "path": str(DEFAULT_PAPER_SHARDS),
                "state": "present" if paper_shards_available else "missing",
                "registry_status": "legacy_path_pending_registry",
            },
            "findings": {
                "available": bool(findings["usable"]),
                "artifact_id": "literature_findings",
                "state": findings["state"],
            },
            "paper_dataset_links": {
                "available": bool(links["usable"]),
                "artifact_id": "paper_dataset_links",
                "state": links["state"],
            },
        }

    def findings(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        path = self._artifact_path("literature_findings")
        if path is None:
            return []
        return [
            asdict(result)
            for result in search_findings(
                query,
                findings_path=path,
                filters=filters,
                limit=limit,
            )
        ]

    def papers(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        links_path = self._artifact_path("paper_dataset_links")
        if not DEFAULT_PAPER_SHARDS.is_dir():
            return []
        return [
            asdict(result)
            for result in search_papers(
                query,
                shard_dir=DEFAULT_PAPER_SHARDS,
                links_path=links_path,
                filters=filters,
                limit=limit,
            )
        ]

    def search(
        self,
        query: str,
        *,
        result_types: tuple[str, ...] = ("papers", "findings"),
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        requested = {item.strip().lower() for item in result_types}
        source_state = self.source_state()
        papers = (
            self.papers(query, filters=filters, limit=limit)
            if "papers" in requested
            else []
        )
        findings = (
            self.findings(query, filters=filters, limit=limit)
            if "findings" in requested
            else []
        )
        warnings: list[str] = []
        if "papers" in requested and not source_state["paper_shards"]["available"]:
            warnings.append(
                "Paper search infrastructure is unavailable on this host; zero paper "
                "results must not be interpreted as evidence that no relevant papers exist."
            )
        if "findings" in requested and not source_state["findings"]["available"]:
            warnings.append(
                "Extracted literature findings are unavailable on this host; zero finding "
                "results reflect missing infrastructure rather than an exhaustive negative."
            )
        return {
            "query": query,
            "papers": papers,
            "findings": findings,
            "total_papers": len(papers),
            "total_findings": len(findings),
            "source_state": source_state,
            "warnings": warnings,
        }
