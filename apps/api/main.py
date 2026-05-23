"""FastAPI application for Neural Search."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from neural_search.cards import generate_dataset_card_json
from neural_search.compare import compare_datasets, generate_comparison_markdown
from neural_search.evaluation.run_benchmark import run_full_benchmark
from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion import services as ingestion_services
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.notebooks.templates import (
    evaluate_template_for_dataset,
    get_notebook_template,
)
from neural_search.ontology import get_ontology, load_ontology
from neural_search.qa import (
    QA_FIELD_DEFAULTS,
    attach_qa_to_card,
    attach_qa_to_dataset,
    get_dataset_qa,
    load_qa_state,
    update_dataset_qa_fields,
    update_dataset_status,
)
from neural_search.recipes import get_recipe
from neural_search.reports.dataset_compilation import compile_dataset_report
from neural_search.schemas import (
    ComparisonResultRead,
    DatasetCardRead,
    DatasetCompareRequest,
    OntologyTermRead,
    SearchRequest,
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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
    matched_terms: list[str] = Field(default_factory=list)
    inferred_concepts: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    missing_metadata_warnings: list[str] = Field(default_factory=list)
    reusable_reason: str | None = None
    suggested_next_actions: list[str] = Field(default_factory=list)
    readiness_score: float | None = None
    linked_papers: list[dict[str, Any]] = Field(default_factory=list)


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
    if not (request.query or "").strip() and not request.structured_query:
        raise HTTPException(
            status_code=400,
            detail=(
                "Enter a free-text experiment description or choose at least one "
                "structured search filter."
            ),
        )
    qa_state = load_qa_state()
    records_with_qa = [
        {**record, "dataset": attach_qa_to_dataset(record["dataset"], qa_state)}
        for record in _demo_data
    ]

    response = search_datasets(
        query=request.query,
        filters=request.filters,
        structured_query=request.structured_query.model_dump()
        if request.structured_query
        else None,
        datasets=records_with_qa,
        limit=request.limit,
    )

    # Transform to frontend format
    frontend_results = []
    for result in response.results:
        # Find the dataset record
        dataset = None
        papers: list[dict[str, Any]] = []
        for record in records_with_qa:
            ds = record["dataset"]
            if ds.get("id") == result.dataset_id or ds.get("source_id") == result.dataset_id:
                dataset = ds
                papers = record.get("papers", [])
                break

        if dataset:
            frontend_results.append(FrontendSearchResult(
                dataset=dataset,
                score=result.score / 100,  # Normalize to 0-1
                why_matched=result.why_matched,
                warnings=result.warnings,
                matched_terms=result.matched_terms,
                inferred_concepts=result.inferred_concepts,
                evidence_snippets=result.evidence_snippets,
                missing_metadata_warnings=result.missing_metadata_warnings,
                reusable_reason=result.reusable_reason,
                suggested_next_actions=result.dataset_card_preview.get("suggested_analyses", [])[:3],
                readiness_score=result.dataset_card_preview.get("analysis_readiness_score"),
                linked_papers=[_frontend_paper(paper) for paper in papers],
            ))

    elapsed = (time.time() - start) * 1000

    return FrontendSearchResponse(
        query=response.query,
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
    qa_status: list[str] | None = Query(default=None),
) -> DatasetListResponse:
    """List all indexed datasets."""
    qa_state = load_qa_state()
    datasets = [attach_qa_to_dataset(record["dataset"], qa_state) for record in _demo_data]
    if qa_status:
        accepted = set(qa_status)
        datasets = [dataset for dataset in datasets if dataset.get("qa_status") in accepted]
    return DatasetListResponse(
        total=len(datasets),
        datasets=datasets[offset : offset + limit],
    )


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Get a specific dataset by ID."""
    qa_state = load_qa_state()
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return attach_qa_to_dataset(ds, qa_state)
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


@app.get("/api/datasets/{dataset_id}/card")
async def get_dataset_card(dataset_id: str) -> dict[str, Any]:
    """Get the dataset card for a specific dataset."""
    qa_state = load_qa_state()
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            dataset = attach_qa_to_dataset(ds, qa_state)
            card = record.get("card")
            if card is None:
                # Generate card on the fly
                extraction = record.get("extraction")
                if extraction is None:
                    extraction = extract_dataset_labels(
                        title=ds.get("title", ""),
                        description=ds.get("description", ""),
                        file_paths=[],
                        source_metadata=dataset,
                        linked_paper_abstracts=[],
                    )
                card = generate_dataset_card_json(
                    dataset, extraction, record.get("papers", [])
                )
            attach_qa_to_card(card, dataset, qa_state)
            return _frontend_card_payload(dataset, card, record)
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


def _card_for_dataset(dataset_id: str) -> DatasetCardRead:
    qa_state = load_qa_state()
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            dataset = attach_qa_to_dataset(ds, qa_state)
            extraction = record.get("extraction")
            if extraction is None:
                extraction = extract_dataset_labels(
                    title=dataset.get("title", ""),
                    description=dataset.get("description", ""),
                    file_paths=[asset.get("path", "") for asset in record.get("assets", [])],
                    source_metadata=dataset,
                    linked_paper_abstracts=[
                        paper.get("abstract", "") for paper in record.get("papers", [])
                    ],
                )
            card = generate_dataset_card_json(
                dataset,
                extraction,
                record.get("papers", []),
            )
            attach_qa_to_card(card, dataset, qa_state)
            return card
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


@app.get("/api/datasets/{dataset_id}/card/export/json")
async def export_dataset_card_json(dataset_id: str) -> JSONResponse:
    """Export a scientific reuse card as JSON."""
    card = _card_for_dataset(dataset_id)
    return JSONResponse(
        content=card.model_dump(mode="json"),
        headers={
            "Content-Disposition": f'attachment; filename="{dataset_id}_reuse_card.json"'
        },
    )


@app.get("/api/datasets/{dataset_id}/card/export/markdown")
async def export_dataset_card_markdown(dataset_id: str) -> Response:
    """Export a scientific reuse card as Markdown."""
    card = _card_for_dataset(dataset_id)
    return Response(
        content=card.card_markdown or "",
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{dataset_id}_reuse_card.md"'
        },
    )


class DatasetQAUpdateRequest(BaseModel):
    qa_status: str | None = None
    task_labels_verified: bool | None = None
    modality_labels_verified: bool | None = None
    behavior_labels_verified: bool | None = None
    brain_regions_verified: bool | None = None
    linked_papers_verified: bool | None = None
    notebook_tested: bool | None = None
    reviewer_notes: str | None = None


@app.patch("/api/datasets/{dataset_id}/qa")
async def update_dataset_qa(
    dataset_id: str,
    request: DatasetQAUpdateRequest,
) -> dict[str, Any]:
    """Update QA fields for a dataset card."""
    if not _dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    updates = {
        key: value
        for key, value in request.model_dump().items()
        if value is not None and (key in QA_FIELD_DEFAULTS or key == "qa_status")
    }
    if not updates:
        return get_dataset_qa({"source_id": dataset_id})
    return update_dataset_qa_fields(dataset_id, updates)


@app.post("/api/datasets/{dataset_id}/qa/status/{qa_status}")
async def mark_dataset_qa_status(dataset_id: str, qa_status: str) -> dict[str, Any]:
    """Mark a dataset reviewed, trusted, rejected, or back into review."""
    if not _dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    try:
        return update_dataset_status(dataset_id, qa_status)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class NotebookRequest(BaseModel):
    asset_path: str | None = Field(default=None, description="Specific NWB file path")
    recipe_id: str | None = Field(default=None, description="Optional analysis recipe ID")
    template_id: str = Field(default="generic_nwb_inspection", description="Notebook template ID")


@app.post("/api/datasets/{dataset_id}/notebook")
async def generate_notebook(
    dataset_id: str,
    request: NotebookRequest | None = None,
) -> FileResponse:
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
            recipes = []
            if request and request.recipe_id:
                recipe = get_recipe(request.recipe_id)
                if recipe is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Recipe {request.recipe_id} not found",
                    )
                recipes.append(recipe)
            template_id = request.template_id if request else "generic_nwb_inspection"
            template = get_notebook_template(template_id)
            if template is None:
                raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            template_status = evaluate_template_for_dataset(template, ds)
            for recipe_id in template.get("recipes", []):
                recipe = get_recipe(recipe_id)
                if recipe and recipe not in recipes:
                    recipes.append(recipe)
            response = generate_nwb_starter_notebook(
                ds,
                asset,
                output_path,
                recipes=recipes,
                notebook_template=template,
                template_warnings=template_status["missing_requirements"],
            )
            if not response.valid:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Notebook was generated but failed validation: "
                        + "; ".join(response.warnings)
                    ),
                )
            return FileResponse(
                path=response.output_path,
                media_type="application/x-ipynb+json",
                filename=f"{dataset_id}_{template_id}.ipynb",
            )
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


def _dataset_exists(dataset_id: str) -> bool:
    return any(
        record["dataset"].get("id") == dataset_id
        or record["dataset"].get("source_id") == dataset_id
        for record in _demo_data
    )


def _find_record(dataset_id: str) -> dict[str, Any] | None:
    """Find a dataset record by ID."""
    for record in _demo_data:
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return record
    return None


# Dataset Comparison endpoints
@app.post("/api/datasets/compare", response_model=ComparisonResultRead)
async def compare_datasets_endpoint(
    request: DatasetCompareRequest,
) -> ComparisonResultRead:
    """
    Compare 2-5 datasets side-by-side.

    Returns detailed comparison including:
    - Source and metadata
    - Task labels, modalities, species, brain regions, behaviors
    - Trial/event availability
    - Linked papers
    - Analysis readiness scores
    - Missing metadata
    - Available notebook templates
    - Suggested analyses
    """
    # Find all requested datasets
    records: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for dataset_id in request.dataset_ids:
        record = _find_record(dataset_id)
        if record is None:
            missing_ids.append(dataset_id)
        else:
            records.append(record)

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Datasets not found: {', '.join(missing_ids)}",
        )

    try:
        result = compare_datasets(records)
        return ComparisonResultRead(
            dataset_ids=result.dataset_ids,
            datasets=[ds.model_dump() for ds in result.datasets],
            field_comparisons=[fc.model_dump() for fc in result.field_comparisons],
            summary=result.summary,
            generated_at=result.generated_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/compare/export/markdown")
async def export_comparison_markdown(
    request: DatasetCompareRequest,
) -> Response:
    """Export dataset comparison as Markdown."""
    # Find all requested datasets
    records: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for dataset_id in request.dataset_ids:
        record = _find_record(dataset_id)
        if record is None:
            missing_ids.append(dataset_id)
        else:
            records.append(record)

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Datasets not found: {', '.join(missing_ids)}",
        )

    try:
        result = compare_datasets(records)
        markdown = generate_comparison_markdown(result)
        filename = f"comparison_{'_'.join(request.dataset_ids[:3])}.md"
        return Response(
            content=markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/compare/export/json")
async def export_comparison_json(
    request: DatasetCompareRequest,
) -> JSONResponse:
    """Export dataset comparison as JSON."""
    # Find all requested datasets
    records: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for dataset_id in request.dataset_ids:
        record = _find_record(dataset_id)
        if record is None:
            missing_ids.append(dataset_id)
        else:
            records.append(record)

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Datasets not found: {', '.join(missing_ids)}",
        )

    try:
        result = compare_datasets(records)
        filename = f"comparison_{'_'.join(request.dataset_ids[:3])}.json"
        return JSONResponse(
            content=result.model_dump(mode="json"),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _frontend_paper(paper: dict[str, Any]) -> dict[str, Any]:
    authors = paper.get("authors_json") or paper.get("authors") or []
    if authors and isinstance(authors[0], dict):
        authors = [
            author.get("name", author.get("display_name", "Unknown author"))
            for author in authors
        ]
    return {
        "id": paper.get("id", paper.get("doi", "paper")),
        "title": paper.get("title", "Untitled paper"),
        "authors": authors,
        "year": paper.get("publication_year"),
        "doi": paper.get("doi"),
        "url": paper.get("url"),
    }


def _frontend_card_payload(
    dataset: dict[str, Any],
    card: DatasetCardRead,
    record: dict[str, Any],
) -> dict[str, Any]:
    standard = next(iter(dataset.get("data_standards", [])), None)
    payload = card.model_dump(mode="json")
    payload.update(
        {
            "dataset_id": dataset.get("source_id", dataset.get("id")),
            "title": dataset.get("title", "Untitled dataset"),
            "summary": card.summary,
            "source": dataset.get("source", "other"),
            "data_standard": standard.lower() if isinstance(standard, str) else standard,
            "species": dataset.get("species", []),
            "modalities": dataset.get("modalities", []),
            "brain_regions": dataset.get("brain_regions", []),
            "tasks": dataset.get("tasks", []),
            "behaviors": dataset.get("behaviors", []),
            "readiness": {
                "score": card.analysis_readiness.score,
                "strengths": card.analysis_readiness.strengths,
                "limitations": card.analysis_readiness.limitations,
                "missing_metadata": card.missing_fields,
                "suggested_analyses": card.suggested_analyses,
            },
            "url": dataset.get("url"),
            "doi": dataset.get("doi"),
            "related_papers": [_frontend_paper(paper) for paper in record.get("papers", [])],
            "assets": record.get("assets", []),
            "missing_metadata": card.missing_fields,
            "provenance": card.provenance,
            "markdown": card.card_markdown,
            "generated_at": payload.get("created_at"),
            "qa_status": card.qa_status,
            "task_labels_verified": card.task_labels_verified,
            "modality_labels_verified": card.modality_labels_verified,
            "behavior_labels_verified": card.behavior_labels_verified,
            "brain_regions_verified": card.brain_regions_verified,
            "linked_papers_verified": card.linked_papers_verified,
            "notebook_tested": card.notebook_tested,
            "reviewer_notes": card.reviewer_notes,
        }
    )
    return payload


# Ontology endpoints
class OntologyResponse(BaseModel):
    tasks: list[OntologyTermRead]
    behavior_labels: list[dict[str, Any]] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    analysis_goals: list[str] = Field(default_factory=list)


@app.get("/api/ontology/tasks", response_model=OntologyResponse)
async def get_ontology_tasks() -> OntologyResponse:
    """Get all tasks from the behavioral ontology."""
    ontology = get_ontology()
    tasks = ontology.tasks
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
        ],
        behavior_labels=[
            {
                "id": behavior.id,
                "label": behavior.label,
                "synonyms": list(behavior.synonyms),
            }
            for behavior in ontology.behavior_labels
        ],
        modalities=ontology.modality_names,
        brain_regions=ontology.region_names,
        analysis_goals=sorted(
            {
                analysis
                for task in ontology.tasks
                for analysis in task.suggested_analyses
            }
        ),
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


# Ingestion endpoints
class IngestRequest(BaseModel):
    query: str = Field(description="Source search query")
    limit: int = Field(default=10, ge=1, le=100)
    save: bool = Field(
        default=False,
        description="Persist normalized records and raw payloads.",
    )
    force: bool = Field(default=False, description="Overwrite existing records when saving.")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query is required")
        return value


class IngestResponse(BaseModel):
    source: str
    query: str
    fetched: int
    normalized: int
    saved: int
    skipped: int
    raw_response_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)


@app.post("/api/ingest/dandi", response_model=IngestResponse)
async def ingest_dandi(request: IngestRequest) -> IngestResponse:
    """Ingest datasets from DANDI Archive."""
    return _ingestion_response(
        ingestion_services.ingest_dandi(
            request.query,
            request.limit,
            save=request.save,
            force=request.force,
        )
    )


@app.post("/api/ingest/openneuro", response_model=IngestResponse)
async def ingest_openneuro(request: IngestRequest) -> IngestResponse:
    """Ingest datasets from OpenNeuro."""
    return _ingestion_response(
        ingestion_services.ingest_openneuro(
            request.query,
            request.limit,
            save=request.save,
            force=request.force,
        )
    )


@app.post("/api/ingest/openalex", response_model=IngestResponse)
async def ingest_openalex(request: IngestRequest) -> IngestResponse:
    """Link papers from OpenAlex to datasets."""
    return _ingestion_response(
        ingestion_services.ingest_openalex(
            request.query,
            request.limit,
            save=request.save,
            force=request.force,
        )
    )


def _ingestion_response(result: ingestion_services.IngestionRunResult) -> IngestResponse:
    return IngestResponse(**result.to_dict())


# Reports endpoints
@app.get("/api/reports/compilation")
async def get_compilation_report() -> dict[str, Any]:
    """Get dataset compilation and corpus QA report statistics."""
    report = compile_dataset_report()
    return {
        "generated_at": report["report_generated_at"],
        "total_datasets": report["summary"]["total_datasets"],
        "qa_review_counts": report["qa_review_counts"],
        "common_missing_metadata": report["common_missing_metadata"],
        "datasets_by_source": report["by_source"],
        "datasets_by_task": report["by_task"],
        "datasets_by_modality": report["by_modality"],
        "datasets_by_species": report["by_species"],
        "datasets_by_brain_region": report["by_brain_region"],
        "datasets_by_data_standard": report["by_data_standard"],
        "top_analysis_ready": [
            {
                "dataset_id": item["source_id"],
                "title": item["title"],
                "score": item["score"],
                "source": item["source"],
            }
            for item in report["top_20_analysis_readiness"][:10]
        ],
        "top_demo_ready": [
            {
                "dataset_id": item["source_id"],
                **item,
            }
            for item in report["top_datasets_ready_for_demo"]
        ],
    }


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


@app.get("/api/evaluation/report")
async def get_evaluation_report() -> dict[str, Any]:
    """Return a benchmark report for the current demo corpus."""
    report = run_full_benchmark(datasets=build_demo_seed())
    return _frontend_evaluation_payload(report)


@app.post("/api/evaluation/run")
async def run_evaluation(request: EvaluationRequest | None = None) -> dict[str, Any]:
    """Run benchmark evaluation queries for the current demo corpus."""
    benchmark_path = Path(request.benchmark_file) if request and request.benchmark_file else None
    if benchmark_path and not benchmark_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark file not found: {benchmark_path}",
        )
    report = run_full_benchmark(benchmark_path=benchmark_path, datasets=build_demo_seed())
    return _frontend_evaluation_payload(report)


def _frontend_evaluation_payload(report: Any) -> dict[str, Any]:
    dataset_titles = {
        record["dataset"].get("source_id", record["dataset"].get("id")): record["dataset"].get(
            "title", "Untitled dataset"
        )
        for record in build_demo_seed()
    }
    query_evaluations = []
    passed_queries = 0
    for query in report.queries:
        expected_tasks = sorted(
            set(query.matched_tasks) | set(query.missing_expected_tasks)
        )
        expected_modalities = sorted(
            set(query.matched_modalities) | set(query.missing_expected_modalities)
        )
        passed = query.precision_at_5 >= 0.4 and query.label_recall_at_10 >= 0.5
        if passed:
            passed_queries += 1
        query_evaluations.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "expected_tasks": expected_tasks,
                "expected_modalities": expected_modalities,
                "found_tasks": query.matched_tasks,
                "found_modalities": query.matched_modalities,
                "precision_at_5": query.precision_at_5,
                "label_recall": query.label_recall_at_10,
                "passed": passed,
                "warnings": query.warnings,
                "top_results": [
                    {
                        "dataset_id": result["dataset_id"],
                        "title": dataset_titles.get(
                            result["dataset_id"], result["dataset_id"]
                        ),
                        "score": round(float(result.get("score", 0)) / 100, 3),
                    }
                    for result in query.top_results
                ],
            }
        )
    return {
        "timestamp": report.generated_at,
        "total_queries": report.total_queries,
        "passed_queries": passed_queries,
        "queries_with_results": report.queries_with_results,
        "avg_precision_at_5": report.mean_precision_at_5,
        "avg_label_recall_at_10": report.mean_label_recall_at_10,
        "avg_task_match_rate": report.mean_task_match_rate,
        "avg_modality_match_rate": report.mean_modality_match_rate,
        "avg_behavior_match_rate": report.mean_behavior_match_rate,
        "summary_warnings": report.summary_warnings,
        "recommendations": report.recommendations,
        "query_evaluations": query_evaluations,
    }
