"""Populate `dataset_old_dataset_new_method_candidate` edges from the corpus.

For each raw corpus record, detects which `neural_search.awareness.taxonomy`
data forms it matches (using the same modality/task/species/brain_region
fields `scripts/build_real_corpus_graph.py` already reads), maps each
matched data form's `analysis_families` to techniques via the methodology
registry (`data/methods/method_registry.yaml`), and emits a candidate edge
per (dataset, technique) pair.

These edges are a heuristic signal, not a verified claim: "this dataset's
profile matches a data form/analysis family that this technique supports."
They do NOT claim the dataset hasn't already been analyzed this way — that
would require a paper-to-method join this module does not perform (see
`has_linked_papers` in edge properties for a weak, non-authoritative hint).
Every edge is therefore marked `requires_human_review=True`.

Node ID conventions are matched literally to what each upstream builder
already emits, not re-derived:

- Dataset node ids match `scripts/build_real_corpus_graph.py::build_graph`'s
  inline construction (`f"node:dataset:{source}:{source_id}"`, with the same
  dataset_id fallback when source_id is empty) — duplicated here rather than
  imported, since that script is a standalone entrypoint with its own
  sys.path setup, not a normal package module.
- Technique node ids match `methods_builder._method_node_id`
  (`f"method:{method_id}"`), as established by
  `neural_search.graph.method_registry_builder`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from neural_search.awareness.taxonomy import DATA_FORMS, detect_data_forms
from neural_search.graph.method_registry_builder import load_method_registry
from neural_search.graph.schema import GraphEvidence, KnowledgeGraphEdge
from neural_search.kg.schemas.evidence_tier import EvidenceTier

log = logging.getLogger(__name__)

BUILDER_NAME = "reanalysis_candidates"
BUILDER_VERSION = "v1.0.0"

CORPUS_FIELDS = ("modalities", "tasks", "species", "brain_regions")


def extract_str_list(items: Any) -> list[str]:
    """Match scripts/build_real_corpus_graph.py::extract_str_list exactly."""

    if not items:
        return []
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = str(item.get("label") or item.get("id") or item.get("name") or "").strip()
        else:
            value = str(item).strip()
        if value:
            result.append(value)
    return result


def dataset_node_id(record: dict[str, Any]) -> str:
    """Match scripts/build_real_corpus_graph.py's dataset node id construction exactly."""

    source = str(record.get("source", "unknown"))
    source_id = str(record.get("source_id") or "")
    if not source_id:
        dataset_id = str(record.get("dataset_id", ""))
        parts = dataset_id.split(":")
        source_id = parts[-1] if len(parts) >= 3 else dataset_id
    return f"node:dataset:{source}:{source_id}"


def _method_node_id(method_id: str) -> str:
    """Match neural_search.ingestion.methods_builder._method_node_id exactly."""

    return f"method:{method_id}"


def _record_terms(record: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for field in CORPUS_FIELDS:
        terms.extend(extract_str_list(record.get(field) or []))
    return terms


def build_reanalysis_candidate_edges(
    corpus_records: Iterable[dict[str, Any]],
) -> list[KnowledgeGraphEdge]:
    registry = load_method_registry()
    links_by_family = {link.analysis_family: link for link in registry.links}

    edges: list[KnowledgeGraphEdge] = []
    seen_edge_ids: set[str] = set()
    record_count = 0
    for record in corpus_records:
        record_count += 1
        terms = _record_terms(record)
        if not terms:
            continue
        data_form_ids = detect_data_forms(terms)
        if not data_form_ids:
            continue

        ds_node_id = dataset_node_id(record)
        has_linked_papers = bool(record.get("linked_papers"))

        for data_form_id in data_form_ids:
            data_form = DATA_FORMS[data_form_id]
            for analysis_family in data_form.analysis_families:
                link = links_by_family.get(analysis_family)
                if link is None:
                    continue  # honest gap: not yet covered by method_registry.yaml
                for method_id in link.taxonomy_method_ids:
                    edge_id = (
                        f"edge:dataset:{ds_node_id}:new_method_candidate:"
                        f"{data_form_id}:{analysis_family}:{method_id}"
                    )
                    if edge_id in seen_edge_ids:
                        continue
                    seen_edge_ids.add(edge_id)
                    evidence = GraphEvidence(
                        evidence_id=f"evidence:{BUILDER_NAME}:{ds_node_id}:{method_id}",
                        source_type=BUILDER_NAME,
                        source_id=ds_node_id,
                        evidence_text=link.rationale.strip(),
                        confidence=link.confidence,
                        extractor_name=BUILDER_NAME,
                        extractor_version=BUILDER_VERSION,
                    )
                    edges.append(
                        KnowledgeGraphEdge(
                            edge_id=edge_id,
                            source_node_id=ds_node_id,
                            target_node_id=_method_node_id(method_id),
                            edge_type="dataset_old_dataset_new_method_candidate",
                            confidence=link.confidence,
                            evidence=[evidence],
                            properties={
                                "data_form": data_form_id,
                                "analysis_family": analysis_family,
                                "rationale": link.rationale.strip(),
                                "requires_human_review": True,
                                "has_linked_papers": has_linked_papers,
                                "evidence_tier": EvidenceTier.HEURISTIC_CANDIDATE.value,
                            },
                        )
                    )
    log.info(
        "Reanalysis candidates: %d edges from %d corpus records",
        len(edges),
        record_count,
    )
    return edges
