"""Thin HTTP adapters over reusable discovery application services."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neural_search.services import DatasetSearchService, LiteratureEvidenceService

router = APIRouter(prefix="/api/v2", tags=["discovery-v2"])
_dataset_search = DatasetSearchService()
_literature = LiteratureEvidenceService()


class SearchV2Request(BaseModel):
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    structured_query: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=200)
    retrieval_config: dict[str, Any] | None = None


class LiteratureV2Request(BaseModel):
    query: str
    result_types: list[str] = Field(default_factory=lambda: ["papers", "findings"])
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=50)


@router.post("/search")
async def search_v2(request: SearchV2Request) -> dict[str, Any]:
    """Core search response plus the exact corpus context that served it."""

    try:
        response, runtime_context = _dataset_search.search_with_context(
            request.query,
            filters=request.filters,
            structured_query=request.structured_query,
            limit=request.limit,
            retrieval_config=request.retrieval_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    payload = response.model_dump(mode="json")
    payload["runtime_context"] = runtime_context
    return payload


@router.post("/literature/search")
async def literature_v2(request: LiteratureV2Request) -> dict[str, Any]:
    """Search papers/findings through the reusable literature service."""

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Enter a literature search query.")
    return _literature.search(
        request.query,
        result_types=tuple(request.result_types),
        filters=request.filters,
        limit=request.limit,
    )
