"""FastAPI application for Neural Search."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from neural_search.cards import generate_dataset_card_json
from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.ontology import get_all_tasks, get_ontology, load_ontology
from neural_search.schemas import (
    DatasetCardRead,
    NotebookGenerationResponse,
    OntologyTermRead,
    SearchRequest,
    SearchResponse,
)
from neural_search.search import search_datasets


# In-memory store for demo (replace with DB in production)
_demo_data: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ontology and demo data on startup."""
    global _demo_data
    load_ontology()
    _demo_data = build_demo_seed()
    yield


app = FastAPI(
    title="Neural Search API",
    description="Experiment-aware neural data discovery system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# Search endpoints
class FrontendSearchResult(BaseModel):
    """Search result format expected by frontend."""
    dataset: dict[str, Any]
    score: float
    why_matched: list[str]
    warnings: list[str]
    suggested_next_actions: list[str] = Field(default_factory=list)


class FrontendSearchResponse(BaseModel):
    """Search response format expected by frontend."""
    query: str
    total_count: int
    results: list[FrontendSearchResult]
    search_time_ms: float = 0.0


@app.post("/api/search", response_model=FrontendSearchResponse)
async def search(request: SearchRequest) -> FrontendSearchResponse:
    """
    Search datasets by experimental meaning.

    Uses hybrid search combining:
    - Keyword matching
    - Ontology-based synonym expansion
    - Metadata filtering
    - Analysis readiness weighting
    """
    import time
    start = time.time()

    response = search_datasets(
        query=request.query,
        filters=request.filters,
        datasets=_demo_data,
        limit=request.limit,
    )

    # Transform to frontend format
    frontend_results = []
    for result in response.results:
        # Find the dataset record
        dataset = None
        for record in _demo_data:
            ds = record["dataset"]
            if ds.get("id") == result.dataset_id or ds.get("source_id") == result.dataset_id:
                dataset = ds
                break

        if dataset:
            frontend_results.append(FrontendSearchResult(
                dataset=dataset,
                score=result.score / 100,  # Normalize to 0-1
                why_matched=result.why_matched,
                warnings=result.warnings,
                suggested_next_actions=result.dataset_card_preview.get("suggested_analyses", [])[:3],
            ))

    elapsed = (time.time() - start) * 1000

    return FrontendSearchResponse(
        query=request.query,
        total_count=len(frontend_results),
        results=frontend_results,
        search_time_ms=elapsed,
    )


# Dataset endpoints
class DatasetListResponse(BaseModel):
    total: int
    datasets: list[dict[str, Any]]


@app.get("/api/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> DatasetListResponse:
    """List all indexed datasets."""
    datasets = [record["dataset"] for record in _demo_data]
    return DatasetListResponse(
        total=len(datasets),
        datasets=datasets[offset : offset + limit],
    )


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Get a specific dataset by ID."""
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


@app.get("/api/datasets/{dataset_id}/card", response_model=DatasetCardRead)
async def get_dataset_card(dataset_id: str) -> DatasetCardRead:
    """Get the dataset card for a specific dataset."""
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            card = record.get("card")
            if card is None:
                # Generate card on the fly
                extraction = record.get("extraction")
                if extraction is None:
                    extraction = extract_dataset_labels(
                        title=ds.get("title", ""),
                        description=ds.get("description", ""),
                        file_paths=[],
                        source_metadata=ds,
                        linked_paper_abstracts=[],
                    )
                card = generate_dataset_card_json(
                    ds, extraction, record.get("papers", [])
                )
            return card
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


class NotebookRequest(BaseModel):
    asset_path: str | None = Field(default=None, description="Specific NWB file path")


@app.post("/api/datasets/{dataset_id}/notebook")
async def generate_notebook(
    dataset_id: str,
    request: NotebookRequest | None = None,
) -> NotebookGenerationResponse:
    """Generate a starter Jupyter notebook for a dataset."""
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            # Create a temporary output path
            output_path = Path(f"/tmp/{dataset_id}_starter.ipynb")
            asset = {
                "id": "generated",
                "path": request.asset_path if request else "example.nwb",
            }
            response = generate_nwb_starter_notebook(ds, asset, output_path)
            return response
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


# Ontology endpoints
class OntologyResponse(BaseModel):
    tasks: list[OntologyTermRead]


@app.get("/api/ontology/tasks", response_model=OntologyResponse)
async def get_ontology_tasks() -> OntologyResponse:
    """Get all tasks from the behavioral ontology."""
    tasks = get_all_tasks()
    return OntologyResponse(
        tasks=[
            OntologyTermRead(
                id=t.id,
                label=t.label,
                category=t.category,
                definition=t.definition,
                synonyms=list(t.synonyms),
                common_events=list(t.common_events),
                relevant_modalities=list(t.relevant_modalities),
                relevant_regions=list(t.relevant_regions),
                suggested_analyses=list(t.suggested_analyses),
            )
            for t in tasks
        ]
    )


@app.get("/api/ontology/tasks/{task_id}", response_model=OntologyTermRead)
async def get_ontology_task(task_id: str) -> OntologyTermRead:
    """Get a specific task from the ontology."""
    ontology = get_ontology()
    task = ontology.task_by_id.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return OntologyTermRead(
        id=task.id,
        label=task.label,
        category=task.category,
        definition=task.definition,
        synonyms=list(task.synonyms),
        common_events=list(task.common_events),
        relevant_modalities=list(task.relevant_modalities),
        relevant_regions=list(task.relevant_regions),
        suggested_analyses=list(task.suggested_analyses),
    )


# Ingestion endpoints (stubs for now)
class IngestRequest(BaseModel):
    dataset_ids: list[str] | None = Field(default=None, description="Specific IDs to ingest")
    limit: int = Field(default=10, ge=1, le=100)


class IngestResponse(BaseModel):
    status: str
    datasets_ingested: int
    message: str


@app.post("/api/ingest/dandi", response_model=IngestResponse)
async def ingest_dandi(request: IngestRequest) -> IngestResponse:
    """Ingest datasets from DANDI Archive."""
    # TODO: Implement actual DANDI ingestion
    return IngestResponse(
        status="pending",
        datasets_ingested=0,
        message="DANDI ingestion not yet implemented. Using demo data.",
    )


@app.post("/api/ingest/openneuro", response_model=IngestResponse)
async def ingest_openneuro(request: IngestRequest) -> IngestResponse:
    """Ingest datasets from OpenNeuro."""
    # TODO: Implement actual OpenNeuro ingestion
    return IngestResponse(
        status="pending",
        datasets_ingested=0,
        message="OpenNeuro ingestion not yet implemented. Using demo data.",
    )


@app.post("/api/ingest/openalex", response_model=IngestResponse)
async def ingest_openalex(request: IngestRequest) -> IngestResponse:
    """Link papers from OpenAlex to datasets."""
    # TODO: Implement actual OpenAlex ingestion
    return IngestResponse(
        status="pending",
        datasets_ingested=0,
        message="OpenAlex ingestion not yet implemented. Using demo data.",
    )


# Evaluation endpoints
class EvaluationRequest(BaseModel):
    benchmark_file: str | None = Field(
        default=None,
        description="Path to benchmark queries YAML",
    )


class EvaluationResult(BaseModel):
    query: str
    expected_ids: list[str]
    returned_ids: list[str]
    precision_at_k: float
    recall: float


class EvaluationResponse(BaseModel):
    status: str
    results: list[EvaluationResult]
    mean_precision: float
    mean_recall: float


@app.post("/api/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Run benchmark evaluation queries."""
    # TODO: Implement actual evaluation
    return EvaluationResponse(
        status="pending",
        results=[],
        mean_precision=0.0,
        mean_recall=0.0,
    )
