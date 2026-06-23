"""FastAPI application for Neural Search."""

import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from apps.api.claims_router import router as claims_router
from apps.api.graph_router import router as graph_router
from apps.api.spectral_router import router as spectral_router
from neural_search.cards import generate_dataset_card_json
from neural_search.compare import compare_datasets, generate_comparison_markdown
from neural_search.corpus.brain_region_index import build_brain_region_index
from neural_search.evaluation.run_benchmark import run_full_benchmark
from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion import services as ingestion_services
from neural_search.ingestion.demo_seed import build_combined_corpus, build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.notebooks.templates import (
    evaluate_template_for_dataset,
    get_notebook_template,
)
from neural_search.ontology import (
    get_brain_regions,
    get_ontology,
    get_recording_scales,
    load_ontology,
)
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
from neural_search.reports.dataset_compilation import (
    compile_dataset_report,
    compute_corpus_completeness,
)
from neural_search.reports.scientific_readiness import build_scientific_readiness_report
from neural_search.schemas import (
    ComparisonResultRead,
    DatasetCardRead,
    DatasetCompareRequest,
    OntologyTermRead,
    SearchRequest,
)
from neural_search.search import search_datasets

# When NEURAL_SEARCH_DEMO_MODE=1, serve only the 26-record demo fixture
# (useful for CI and quick local demos). Default: full combined corpus.
_DEMO_MODE = os.getenv("NEURAL_SEARCH_DEMO_MODE", "").lower() in ("1", "true", "yes")
FRONTEND_ARTIFACT_DIR = Path("artifacts/frontend")
LITERATURE_SHARD_DIR = Path("data/corpus/normalized/openalex_neuro")
LITERATURE_FINDINGS_PATH = Path("artifacts/literature/findings_v1.jsonl")
LITERATURE_LINKS_PATH = Path("artifacts/literature/paper_dataset_links.jsonl")
NEURO_JUDGE_WATERMARK = (
    "PRELIMINARY NEURO-JUDGE EVALUATION — RAG-GROUNDED LLM LABELS, "
    "NOT PURE HUMAN GOLD"
)

# In-memory store for demo (replace with DB in production)
_demo_data: list[dict[str, Any]] = []


def _load_corpus() -> list[dict[str, Any]]:
    return build_demo_seed() if _DEMO_MODE else build_combined_corpus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ontology and corpus on startup."""
    global _demo_data
    load_ontology()
    _demo_data = _load_corpus()
    yield


def _ensure_demo_data() -> list[dict[str, Any]]:
    """Load corpus lazily for tests and local API usage."""
    global _demo_data
    if not _demo_data:
        load_ontology()
        _demo_data = _load_corpus()
    return _demo_data


app = FastAPI(
    title="Neural Search API",
    description="Experiment-aware neural data discovery system",
    version="0.1.0",
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

app.include_router(graph_router)
app.include_router(claims_router)
app.include_router(spectral_router)


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
    rank: int | None = None
    retrieval_method: str = "hybrid_search"
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    neuro_judge: dict[str, Any] | None = None
    evidence_packet: dict[str, Any] | None = None
    prior_feedback: list[dict[str, Any]] = Field(default_factory=list)
    memory_graph_evidence: dict[str, Any] | None = None


class FrontendSearchResponse(BaseModel):
    """Search response format expected by frontend."""
    query: str
    total_count: int
    results: list[FrontendSearchResult]
    search_time_ms: float | None = None


class LiteratureSearchRequest(BaseModel):
    """Search request for paper and finding literature results."""
    query: str
    result_types: list[str] = Field(default_factory=lambda: ["papers", "findings"])
    limit: int = Field(default=10, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)


class LiteratureSearchResponse(BaseModel):
    """Paper/finding search response."""
    query: str
    papers: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    total_papers: int = 0
    total_findings: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _normalized_query(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _dataset_lookup_keys(dataset: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    source = str(dataset.get("source", "")).lower()
    for value in (dataset.get("id"), dataset.get("source_id")):
        if value:
            text = str(value).lower()
            keys.add(text)
            if source and not text.startswith(f"{source}:"):
                keys.add(f"{source}:{text}")
    return keys


def _record_lookup_keys(record: dict[str, Any]) -> set[str]:
    dataset = {
        "id": record.get("dataset_id"),
        "source_id": record.get("dataset_id"),
        "source": record.get("source_archive"),
    }
    return _dataset_lookup_keys(dataset)


def _neuro_judge_paths() -> tuple[Path, Path]:
    consensus = Path("artifacts/field_state/neuro_qrels_consensus.jsonl")
    if not consensus.exists():
        consensus = Path("artifacts/field_state/neuro_qrels_consensus_mock.jsonl")
    judgments = Path("artifacts/field_state/neuro_qrels_judgments.jsonl")
    if not judgments.exists():
        judgments = Path("artifacts/field_state/neuro_qrels_judgments_mock.jsonl")
    return consensus, judgments


def _load_neuro_judge_index() -> dict[tuple[str, str], dict[str, Any]]:
    packets = _read_jsonl(Path("artifacts/field_state/neuro_judge_evidence_packets.jsonl"))
    packet_by_qid_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for packet in packets:
        dataset_keys = _record_lookup_keys(packet)
        for dataset_key in dataset_keys:
            qid = str(packet.get("query_id", ""))
            if qid:
                packet_by_qid_dataset[(qid, dataset_key)] = packet

    consensus_path, judgments_path = _neuro_judge_paths()
    judgments = _read_jsonl(consensus_path) or _read_jsonl(judgments_path)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for judgment in judgments:
        dataset_keys = _record_lookup_keys(judgment)
        qid = str(judgment.get("query_id", ""))
        for dataset_key in dataset_keys:
            packet = packet_by_qid_dataset.get((qid, dataset_key))
            qtext = _normalized_query(packet.get("query_text") if packet else "")
            if not qtext:
                continue
            snapshot = dict(judgment)
            if "judge_model" not in snapshot and snapshot.get("judge_models"):
                snapshot["judge_model"] = snapshot["judge_models"][0]
            snapshot.setdefault("watermark", NEURO_JUDGE_WATERMARK)
            index[(qtext, dataset_key)] = {
                "judgment": snapshot,
                "packet": packet,
            }
    return index


def _neuro_judge_for_result(
    query: str,
    dataset: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    qtext = _normalized_query(query)
    for dataset_key in _dataset_lookup_keys(dataset):
        match = index.get((qtext, dataset_key))
        if match:
            return match
    return None


def _load_feedback_for_pair(query: str, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    query_norm = _normalized_query(query)
    dataset_keys = _dataset_lookup_keys(dataset)
    feedback = []
    for row in _read_jsonl(FRONTEND_ARTIFACT_DIR / "retrieval_feedback.jsonl"):
        if _normalized_query(row.get("query_text")) != query_norm:
            continue
        row_dataset = str(row.get("dataset_id", "")).lower()
        if row_dataset in dataset_keys:
            feedback.append(row)
    return feedback[-5:]


def _compact_evidence_packet(packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not packet:
        return None
    linked = packet.get("linked_papers") or []
    compact_linked = []
    for paper in linked[:2]:
        abstract = paper.get("abstract", "")
        compact_linked.append({
            **paper,
            "abstract_snippet": abstract[:500] if isinstance(abstract, str) else "",
        })
    return {
        **packet,
        "linked_papers": compact_linked,
        "raw_json": packet,
    }


def _try_exact_id_lookup(query: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first record whose source_id matches the query exactly.

    Recognised patterns:
        DANDI:000026   DANDI/000026   dandi 26   026          → dandi source_id
        ds003505       DS003505       openneuro/ds003505       → openneuro source_id
    """
    q = query.strip()

    # DANDI: optional prefix + up to 7 digits
    dandi_m = re.fullmatch(r'(?:dandi[:/\s]*)?0*(\d{1,7})', q, re.IGNORECASE)
    if dandi_m:
        num = dandi_m.group(1)
        for record in records:
            ds = record.get("dataset", {})
            if ds.get("source") == "dandi":
                sid = re.sub(r'^\D+', '', ds.get("source_id", "")).lstrip('0') or '0'
                if sid == (num.lstrip('0') or '0'):
                    return record

    # OpenNeuro: optional prefix + ds + 6 digits
    on_m = re.fullmatch(r'(?:openneuro[:/\s]*)?(ds\d{6})', q, re.IGNORECASE)
    if on_m:
        sid = on_m.group(1).lower()
        for record in records:
            ds = record.get("dataset", {})
            if ds.get("source") == "openneuro" and ds.get("source_id", "").lower() == sid:
                return record

    return None


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
    demo_data = _ensure_demo_data()
    records_with_qa = [
        {**record, "dataset": attach_qa_to_dataset(record["dataset"], qa_state)}
        for record in demo_data
    ]

    exact_record = _try_exact_id_lookup(request.query or "", records_with_qa)

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
    neuro_index = _load_neuro_judge_index()
    for rank, result in enumerate(response.results, start=1):
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
            neuro_match = _neuro_judge_for_result(request.query, dataset, neuro_index)
            neuro_judge = neuro_match["judgment"] if neuro_match else None
            evidence_packet = _compact_evidence_packet(
                neuro_match["packet"] if neuro_match else None
            )
            warnings = list(result.warnings)
            if neuro_judge and neuro_judge.get("hard_negative_detected"):
                warnings.append("neuro-judge hard-negative warning")
            frontend_results.append(FrontendSearchResult(
                dataset=dataset,
                score=result.score / 100,  # Normalize to 0-1
                why_matched=result.why_matched,
                warnings=warnings,
                matched_terms=result.matched_terms,
                inferred_concepts=result.inferred_concepts,
                evidence_snippets=result.evidence_snippets,
                missing_metadata_warnings=result.missing_metadata_warnings,
                reusable_reason=result.reusable_reason,
                suggested_next_actions=result.dataset_card_preview.get("suggested_analyses", [])[:3],
                readiness_score=result.dataset_card_preview.get("analysis_readiness_score"),
                linked_papers=[_frontend_paper(paper) for paper in papers],
                rank=rank,
                score_breakdown=result.score_breakdown,
                neuro_judge=neuro_judge,
                evidence_packet=evidence_packet,
                prior_feedback=_load_feedback_for_pair(request.query, dataset),
                memory_graph_evidence=result.memory_graph_evidence,
            ))

    if exact_record:
        exact_ds = exact_record.get("dataset", {})
        exact_id = exact_ds.get("id") or exact_ds.get("source_id")
        already_first = bool(
            frontend_results
            and (
                frontend_results[0].dataset.get("id") == exact_id
                or frontend_results[0].dataset.get("source_id") == exact_id
            )
        )
        if not already_first:
            without_dup = [
                r for r in frontend_results
                if r.dataset.get("id") != exact_id and r.dataset.get("source_id") != exact_id
            ]
            exact_papers = exact_record.get("papers", [])
            pin = FrontendSearchResult(
                dataset=exact_ds,
                score=1.0,
                why_matched=["exact ID match"],
                warnings=[],
                matched_terms=[],
                inferred_concepts=[],
                evidence_snippets=[],
                missing_metadata_warnings=[],
                reusable_reason=None,
                suggested_next_actions=[],
                readiness_score=None,
                linked_papers=[_frontend_paper(p) for p in exact_papers],
                rank=1,
                score_breakdown={"exact_id_match": 1.0, "final_score": 1.0},
                prior_feedback=_load_feedback_for_pair(request.query, exact_ds),
            )
            frontend_results = [pin] + [
                item.model_copy(update={"rank": idx})
                for idx, item in enumerate(without_dup, start=2)
            ]

    elapsed = (time.time() - start) * 1000

    return FrontendSearchResponse(
        query=response.query,
        total_count=len(frontend_results),
        results=frontend_results,
        search_time_ms=elapsed,
    )


@app.post("/api/literature/search", response_model=LiteratureSearchResponse)
async def search_literature(request: LiteratureSearchRequest) -> LiteratureSearchResponse:
    """Search literature papers and extracted findings."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Enter a literature search query.")

    from neural_search.literature.search import search_findings, search_papers

    requested = {item.strip().lower() for item in request.result_types}
    papers = (
        search_papers(
            request.query,
            shard_dir=LITERATURE_SHARD_DIR,
            links_path=LITERATURE_LINKS_PATH,
            filters=request.filters,
            limit=request.limit,
        )
        if "papers" in requested
        else []
    )
    findings = (
        search_findings(
            request.query,
            findings_path=LITERATURE_FINDINGS_PATH,
            filters=request.filters,
            limit=request.limit,
        )
        if "findings" in requested
        else []
    )

    return LiteratureSearchResponse(
        query=request.query,
        papers=[asdict(paper) for paper in papers],
        findings=[asdict(finding) for finding in findings],
        total_papers=len(papers),
        total_findings=len(findings),
    )


# Frontend feedback endpoints
class SearchSessionRequest(BaseModel):
    query_text: str = ""
    query_id: str | None = None
    retrieval_method: str = "hybrid_search"
    filters: dict[str, Any] = Field(default_factory=dict)
    structured_query: dict[str, Any] | None = None


class SearchSessionResponse(BaseModel):
    session_id: str
    timestamp: str
    provenance: str = "user_feedback_downstream_signal"


class FeedbackEventRequest(BaseModel):
    session_id: str | None = None
    query_id: str | None = None
    query_text: str = ""
    retrieval_method: str = "hybrid_search"
    rank: int | None = None
    dataset_id: str
    dataset_title: str
    usefulness: str
    would_use_for_analysis: str | None = None
    clicked: bool = False
    opened_evidence: bool = False
    saved: bool = False
    exported: bool = False
    reason_tags: list[str] = Field(default_factory=list)
    free_text_note: str = ""
    judge_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("usefulness")
    @classmethod
    def valid_usefulness(cls, value: str) -> str:
        allowed = {"useful", "partially_useful", "not_useful", "unsure"}
        if value not in allowed:
            raise ValueError(f"usefulness must be one of {sorted(allowed)}")
        return value

    @field_validator("would_use_for_analysis")
    @classmethod
    def valid_would_use(cls, value: str | None) -> str | None:
        allowed = {None, "yes", "maybe", "no"}
        if value not in allowed:
            raise ValueError("would_use_for_analysis must be yes, maybe, no, or null")
        return value


class SavedDatasetRequest(BaseModel):
    session_id: str | None = None
    query_id: str | None = None
    query_text: str = ""
    dataset_id: str
    dataset_title: str
    rank: int | None = None
    retrieval_method: str = "hybrid_search"
    exported: bool = False
    judge_snapshot: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/frontend/search-sessions", response_model=SearchSessionResponse)
async def create_search_session(request: SearchSessionRequest) -> SearchSessionResponse:
    """Create a lightweight frontend search session artifact."""
    timestamp = datetime.now(UTC).isoformat()
    session = {
        "session_id": f"session_{uuid4().hex}",
        "timestamp": timestamp,
        "query_id": request.query_id,
        "query_text": request.query_text,
        "retrieval_method": request.retrieval_method,
        "filters": request.filters,
        "structured_query": request.structured_query,
        "provenance": "user_feedback_downstream_signal",
    }
    _append_jsonl(FRONTEND_ARTIFACT_DIR / "search_sessions.jsonl", session)
    return SearchSessionResponse(
        session_id=session["session_id"],
        timestamp=timestamp,
    )


@app.post("/api/frontend/feedback")
async def log_feedback_event(request: FeedbackEventRequest) -> dict[str, Any]:
    """Log downstream retrieval feedback, not gold relevance labels."""
    event = request.model_dump(mode="json")
    event.update(
        {
            "feedback_id": f"feedback_{uuid4().hex}",
            "timestamp": datetime.now(UTC).isoformat(),
            "provenance": "user_feedback_downstream_signal",
        }
    )
    _append_jsonl(FRONTEND_ARTIFACT_DIR / "retrieval_feedback.jsonl", event)
    return event


@app.post("/api/frontend/saved-datasets")
async def save_frontend_dataset(request: SavedDatasetRequest) -> dict[str, Any]:
    """Persist a frontend save/export event for downstream success analysis."""
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "saved_dataset_id": f"saved_{uuid4().hex}",
            "timestamp": datetime.now(UTC).isoformat(),
            "saved": True,
            "provenance": "user_feedback_downstream_signal",
        }
    )
    _append_jsonl(FRONTEND_ARTIFACT_DIR / "saved_datasets.jsonl", payload)
    return payload


@app.get("/api/frontend/feedback/{dataset_id}")
async def get_frontend_feedback(
    dataset_id: str,
    query_text: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return prior downstream feedback for a dataset, optionally scoped by query."""
    dataset_key = dataset_id.lower()
    query_norm = _normalized_query(query_text)
    rows = []
    for row in _read_jsonl(FRONTEND_ARTIFACT_DIR / "retrieval_feedback.jsonl"):
        if str(row.get("dataset_id", "")).lower() != dataset_key:
            continue
        if query_norm and _normalized_query(row.get("query_text")) != query_norm:
            continue
        rows.append(row)
    return {"dataset_id": dataset_id, "feedback": rows}


@app.get("/api/frontend/feedback-summary")
async def get_frontend_feedback_summary() -> dict[str, Any]:
    """Summarize local downstream retrieval feedback artifacts."""
    feedback = _read_jsonl(FRONTEND_ARTIFACT_DIR / "retrieval_feedback.jsonl")
    sessions = _read_jsonl(FRONTEND_ARTIFACT_DIR / "search_sessions.jsonl")
    saved = _read_jsonl(FRONTEND_ARTIFACT_DIR / "saved_datasets.jsonl")
    usefulness: dict[str, int] = {}
    for row in feedback:
        key = str(row.get("usefulness", "unknown"))
        usefulness[key] = usefulness.get(key, 0) + 1
    return {
        "sessions": len(sessions),
        "feedback_events": len(feedback),
        "saved_datasets": len(saved),
        "usefulness_distribution": usefulness,
        "provenance": "user_feedback_downstream_signal",
    }


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
    datasets = [
        attach_qa_to_dataset(record["dataset"], qa_state)
        for record in _ensure_demo_data()
    ]
    if qa_status:
        accepted = set(qa_status)
        datasets = [dataset for dataset in datasets if dataset.get("qa_status") in accepted]
    return DatasetListResponse(
        total=len(datasets),
        datasets=datasets[offset : offset + limit],
    )


@app.get("/api/corpus/readiness")
async def get_corpus_readiness() -> dict[str, Any]:
    """Return the v0.8 scientific readiness audit for agent planning."""

    return build_scientific_readiness_report()


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """Get a specific dataset by ID."""
    qa_state = load_qa_state()
    for record in _ensure_demo_data():
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            return attach_qa_to_dataset(ds, qa_state)
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


@app.get("/api/datasets/{dataset_id}/card")
async def get_dataset_card(dataset_id: str) -> dict[str, Any]:
    """Get the dataset card for a specific dataset."""
    qa_state = load_qa_state()
    for record in _ensure_demo_data():
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
    for record in _ensure_demo_data():
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
    for record in _ensure_demo_data():
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
        for record in _ensure_demo_data()
    )


def _find_record(dataset_id: str) -> dict[str, Any] | None:
    """Find a dataset record by ID."""
    for record in _ensure_demo_data():
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
            "recording_scales": dataset.get("recording_scales", []),
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
    recording_scales: list[dict[str, Any]] = Field(default_factory=list)
    brain_regions: list[str] = Field(default_factory=list)
    brain_region_index: list[dict[str, Any]] = Field(default_factory=list)
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
        recording_scales=[
            scale.model_dump(mode="json") for scale in get_recording_scales()
        ],
        brain_regions=sorted(region.id for region in get_brain_regions()),
        brain_region_index=[
            entry.model_dump(mode="json")
            for entry in sorted(
                build_brain_region_index().values(),
                key=lambda item: item.id,
            )
        ],
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


# Demo report endpoint
@app.get("/api/demo/report")
async def get_demo_report() -> dict[str, Any]:
    """Return the pre-computed killer demo report from reports/killer_demo.json."""
    import json
    report_path = Path("reports/killer_demo.json")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Demo report not found. Run: python scripts/run_killer_demo.py",
        )
    return json.loads(report_path.read_text())


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


@app.get("/api/reports/corpus-completeness")
async def get_corpus_completeness() -> dict[str, Any]:
    """Field fill rates per source, showing metadata coverage gaps."""
    return compute_corpus_completeness()


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


# ── Coverage gap API ──────────────────────────────────────────────────────────
# NOTE: No app-wide auth exists; this API is intended to run on localhost only
# behind an authenticating reverse proxy. Do not expose it directly to the
# public internet. See deployment docs for proxy configuration.

_COVERAGE_DB_PATH = Path("data/coverage/ledger.duckdb")

_ALLOWED_COVERAGE_DIMS = frozenset({
    "brain_regions",
    "modalities",
    "species",
    "tasks",
    "recording_scales",
})

_SPECIES_ID_RE = re.compile(r"^[a-zA-Z0-9_:\-]{1,64}$")


def _validate_dim(name: str, value: str) -> str:
    if value not in _ALLOWED_COVERAGE_DIMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {name!r}: must be one of {sorted(_ALLOWED_COVERAGE_DIMS)}",
        )
    return value


def _validate_species(value: str | None) -> str | None:
    if value is None:
        return None
    if not _SPECIES_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail="Invalid species filter: use alphanumeric/underscore/hyphen/colon IDs only",
        )
    return value


def _coverage_store() -> Any:
    from neural_search.coverage.duckdb_store import CoverageStore
    if not _COVERAGE_DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Coverage ledger not built. Run: python scripts/coverage/build_duckdb_ledger.py",
        )
    return CoverageStore(_COVERAGE_DB_PATH)


@app.get("/api/coverage/summary")
async def get_coverage_summary() -> dict[str, Any]:
    """Overall coverage statistics from the DuckDB ledger."""
    with _coverage_store() as store:
        return store.coverage_summary()


@app.get("/api/coverage/source-rates")
async def get_coverage_source_rates() -> list[dict[str, Any]]:
    """Per-source coverage percentages for brain_regions, modalities, species, tasks."""
    with _coverage_store() as store:
        rows = store.source_coverage_rates().fetchall()
    return [
        {
            "source": r[0],
            "n_total": r[1],
            "regions_covered": r[2],
            "regions_pct": r[3],
            "modalities_covered": r[4],
            "modalities_pct": r[5],
            "species_covered": r[6],
            "species_pct": r[7],
            "tasks_covered": r[8],
            "tasks_pct": r[9],
        }
        for r in rows
    ]


@app.get("/api/coverage/uncovered-regions")
async def get_uncovered_regions() -> list[dict[str, Any]]:
    """Ontology regions with zero corpus datasets."""
    with _coverage_store() as store:
        rows = store.uncovered_regions().fetchall()
    return [
        {
            "id": r[0],
            "label": r[1],
            "uberon_id": r[2],
            "allen_ccf_mouse_id": r[3],
            "parents": json.loads(r[4] or "[]"),
        }
        for r in rows
    ]


@app.get("/api/coverage/gap-matrix")
async def get_coverage_gap_matrix(
    row_dim: str = "brain_regions",
    col_dim: str = "modalities",
    species: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Top region × modality (or other dimension) dataset counts."""
    row_dim = _validate_dim("row_dim", row_dim)
    col_dim = _validate_dim("col_dim", col_dim)
    species = _validate_species(species)
    with _coverage_store() as store:
        rows = store.gap_matrix(
            row_dim, col_dim, species_filter=species
        ).fetchmany(limit)
    row_key = row_dim.rstrip("s")
    col_key = col_dim.rstrip("s")
    return [{"row": r[0], "col": r[1], "n_datasets": r[2], "row_dim": row_key, "col_dim": col_key} for r in rows]


@app.get("/api/coverage/dark-pairs")
async def get_dark_pairs(
    dim_a: str = "brain_regions",
    dim_b: str = "modalities",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Highest-opportunity zero-coverage dimension pairs."""
    dim_a = _validate_dim("dim_a", dim_a)
    dim_b = _validate_dim("dim_b", dim_b)
    with _coverage_store() as store:
        rows = store.dark_pairs(dim_a, dim_b, top_n=limit).fetchall()
    return [
        {
            "a_value": r[0],
            "b_value": r[1],
            "n_observed": r[2],
            "a_marginal": r[3],
            "b_marginal": r[4],
            "opportunity_score": r[5],
            "dim_a": dim_a,
            "dim_b": dim_b,
        }
        for r in rows
    ]


@app.get("/api/datasets/{dataset_id}/affordances")
async def get_dataset_affordances(dataset_id: str) -> dict[str, Any]:
    """Affordance validation results for a dataset (all 15 analysis types)."""
    from neural_search.affordances.registry import (
        AFFORDANCE_REGISTRY,
        detect_features_from_metadata,
        validate_all_affordances,
    )

    qa_state = load_qa_state()
    dataset_dict: dict[str, Any] | None = None
    for record in _ensure_demo_data():
        ds = record["dataset"]
        if ds.get("id") == dataset_id or ds.get("source_id") == dataset_id:
            dataset_dict = attach_qa_to_dataset(ds, qa_state)
            break

    if dataset_dict is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    features = detect_features_from_metadata(dataset_dict)
    results = validate_all_affordances(features)

    affordances = []
    for r in results:
        req = AFFORDANCE_REGISTRY.get(r.affordance_id)
        affordances.append({
            "affordance_id": r.affordance_id,
            "label": req.label if req else r.affordance_id,
            "support_level": r.support_level,
            "confidence": r.confidence,
            "found_required_features": r.found_required_features,
            "missing_required_features": r.missing_required_features,
            "found_optional_features": r.found_optional_features,
        })

    return {
        "dataset_id": dataset_id,
        "affordances": affordances,
        "detection_method": features.detection_method,
    }


_graph_cache: dict[str, Any] = {}


@app.get("/api/datasets/{dataset_id}/similar")
async def get_similar_datasets(dataset_id: str, limit: int = 6) -> dict[str, Any]:
    """Datasets related via cross-dataset knowledge graph edges."""
    import json

    from neural_search.graph.query import find_similar_datasets
    from neural_search.graph.schema import KnowledgeGraph

    graph_path = Path("data/graph/neural_search_graph.real_corpus.json")
    if not graph_path.exists():
        return {"dataset_id": dataset_id, "similar": [], "source": "graph_unavailable"}

    if "graph" not in _graph_cache:
        with graph_path.open(encoding="utf-8") as f:
            _graph_cache["graph"] = KnowledgeGraph.model_validate(json.load(f))

    results = find_similar_datasets(_graph_cache["graph"], dataset_id, limit=limit)
    return {"dataset_id": dataset_id, "similar": results, "source": "knowledge_graph"}


@app.get("/api/coverage/region-counts")
async def get_region_counts() -> list[dict[str, Any]]:
    """All brain regions with dataset counts for the Brain Atlas heatmap."""
    with _coverage_store() as store:
        return store.region_dataset_counts()


@app.get("/api/coverage/region/{region_id}/datasets")
async def get_region_datasets(
    region_id: str,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Datasets tagged with a specific brain region."""
    with _coverage_store() as store:
        datasets = store.datasets_for_region(region_id, limit=limit, offset=offset)
    return {"region_id": region_id, "datasets": datasets, "count": len(datasets)}
