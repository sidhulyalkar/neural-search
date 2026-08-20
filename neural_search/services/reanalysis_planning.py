"""Application service composing retrieval, literature evidence, and reanalysis logic."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from neural_search.graph.reanalysis_bridge_builder import load_dataset_method_evidence
from neural_search.reanalysis import ReanalysisPlan, build_reanalysis_plan
from neural_search.runtime import artifact_status
from neural_search.services.corpus_access import (
    CorpusAccessService,
    dataset_identity,
    dataset_lookup_keys,
)
from neural_search.services.dataset_search import DatasetSearchService
from neural_search.services.literature_evidence import LiteratureEvidenceService


@lru_cache(maxsize=4)
def _load_method_evidence_cached(
    links_path: str,
    links_mtime_ns: int,
    ner_path: str,
    ner_mtime_ns: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    del links_mtime_ns, ner_mtime_ns  # cache keys only
    return load_dataset_method_evidence(links_path, ner_path)


def _dataset(record: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = record.get("dataset")
    return nested if isinstance(nested, Mapping) else record


def _values(record: Mapping[str, Any], field_name: str) -> list[str]:
    dataset = _dataset(record)
    raw = dataset.get(field_name) or record.get(field_name) or []
    if isinstance(raw, str):
        return [raw]
    result: list[str] = []
    if isinstance(raw, list | tuple | set):
        for item in raw:
            if isinstance(item, Mapping):
                value = item.get("label") or item.get("id") or item.get("name")
                if value:
                    result.append(str(value))
            elif item is not None:
                result.append(str(item))
    return result


class ReanalysisPlanningService:
    """Build researcher-facing reanalysis plans from currently available evidence."""

    def __init__(
        self,
        *,
        corpus_service: CorpusAccessService | None = None,
        search_service: DatasetSearchService | None = None,
        literature_service: LiteratureEvidenceService | None = None,
    ) -> None:
        self.corpus_service = corpus_service or CorpusAccessService()
        self.search_service = search_service or DatasetSearchService(
            corpus_service=self.corpus_service
        )
        self.literature_service = literature_service or LiteratureEvidenceService()

    def corpus(self) -> tuple[list[dict[str, Any]], str]:
        """Use the same profile-aware corpus policy as normal dataset search."""

        return self.corpus_service.load()

    def _find_record(
        self,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        wanted = dataset_id.casefold()
        for record in records:
            if wanted in dataset_lookup_keys(record):
                return record
        raise ValueError(f"Dataset not found in active corpus: {dataset_id}")

    def _method_evidence(self) -> dict[str, dict[str, dict[str, Any]]]:
        links_status = artifact_status("paper_dataset_links")
        ner_status = artifact_status("ner_method_graph")
        if not links_status["usable"] or not ner_status["usable"]:
            return {}
        links = Path(links_status["absolute_path"])
        ner = Path(ner_status["absolute_path"])
        return _load_method_evidence_cached(
            str(links),
            links.stat().st_mtime_ns,
            str(ner),
            ner.stat().st_mtime_ns,
        )

    def _precedents(
        self,
        target: Mapping[str, Any],
        records: list[dict[str, Any]],
        method_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        dataset = _dataset(target)
        query_terms = [str(dataset.get("title") or "")]
        for field_name in ("modalities", "tasks", "species", "brain_regions"):
            query_terms.extend(_values(target, field_name)[:3])
        query = " ".join(term for term in query_terms if term).strip()
        if not query:
            return {}

        response = self.search_service.search(query, datasets=records, limit=15)
        target_keys = dataset_lookup_keys(target)
        precedents: dict[str, list[dict[str, Any]]] = {}
        for result in response.results:
            related: dict[str, Any] | None = None
            result_id = str(result.dataset_id).casefold()
            for record in records:
                if result_id in dataset_lookup_keys(record):
                    related = record
                    break
            if related is None or target_keys & dataset_lookup_keys(related):
                continue
            related_id = dataset_identity(related)
            evidence = method_evidence.get(related_id) or {}
            if not evidence:
                continue
            raw_score = float(getattr(result, "score", 0.0) or 0.0)
            similarity_confidence = min(
                1.0,
                raw_score / 100.0 if raw_score > 1 else raw_score,
            )
            for method_id, method_payload in evidence.items():
                precedents.setdefault(method_id, []).append(
                    {
                        "dataset_id": related_id,
                        "title": str(_dataset(related).get("title") or related_id),
                        "confidence": round(similarity_confidence, 4),
                        "paper_openalex_id": method_payload.get("paper_openalex_id"),
                        "summary": (
                            "A retrieval-neighbor dataset has paper-method evidence for "
                            "this method. Similarity is a discovery signal, not proof of "
                            "experimental equivalence."
                        ),
                    }
                )
        for values in precedents.values():
            values.sort(
                key=lambda item: float(item.get("confidence") or 0),
                reverse=True,
            )
        return precedents

    def plan(self, dataset_id: str, *, limit: int = 12) -> dict[str, Any]:
        records, corpus_source = self.corpus()
        target = self._find_record(dataset_id, records)
        all_method_evidence = self._method_evidence()
        target_evidence = all_method_evidence.get(dataset_identity(target), {})
        precedents = self._precedents(target, records, all_method_evidence)

        preliminary = build_reanalysis_plan(
            target,
            existing_method_evidence=target_evidence,
            precedent_by_method=precedents,
            limit=limit,
        )
        literature_by_method: dict[str, list[dict[str, Any]]] = {}
        context_terms = " ".join(
            _values(target, "tasks")[:2] + _values(target, "brain_regions")[:2]
        )
        for candidate in preliminary.candidates[:6]:
            query = f"{candidate.method_label} {context_terms}".strip()
            literature_by_method[candidate.method_id] = (
                self.literature_service.findings(query, limit=3)
            )

        final_plan: ReanalysisPlan = build_reanalysis_plan(
            target,
            existing_method_evidence=target_evidence,
            precedent_by_method=precedents,
            literature_by_method=literature_by_method,
            limit=limit,
        )
        payload = final_plan.to_dict()
        payload["corpus_source"] = corpus_source
        payload["execution_profile"] = self.corpus_service.profile
        payload["evidence_capabilities"] = {
            "paper_method_evidence": bool(all_method_evidence),
            "literature_findings": bool(
                artifact_status("literature_findings")["usable"]
            ),
            "related_dataset_precedents": bool(precedents),
        }
        payload["literature_source_state"] = self.literature_service.source_state()
        return payload
