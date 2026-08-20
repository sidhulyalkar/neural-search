"""Application service for literature and finding evidence retrieval."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_search.literature.search import search_findings, search_papers
from neural_search.runtime import artifact_status

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_SHARDS = PROJECT_ROOT / "data" / "corpus" / "normalized" / "openalex_neuro"


class LiteratureEvidenceService:
    """Search scientific evidence without coupling callers to filesystem paths."""

    def _artifact_path(self, artifact_id: str) -> Path | None:
        status = artifact_status(artifact_id)
        if not status["usable"]:
            return None
        return Path(status["absolute_path"])

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
        if not DEFAULT_PAPER_SHARDS.exists():
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
        papers = self.papers(query, filters=filters, limit=limit) if "papers" in requested else []
        findings = (
            self.findings(query, filters=filters, limit=limit)
            if "findings" in requested
            else []
        )
        return {
            "query": query,
            "papers": papers,
            "findings": findings,
            "total_papers": len(papers),
            "total_findings": len(findings),
        }
