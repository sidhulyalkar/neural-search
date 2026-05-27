"""Semantic paper-dataset linking for knowledge graph enrichment.

This module provides functionality to link papers and datasets based on:
1. Shared concept similarity (tasks, modalities, regions, behaviors)
2. Text-based semantic matching
3. Citation and DOI-based explicit links
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from neural_search.graph.query import get_neighbors
from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    make_edge_id,
    make_node_id,
)


@dataclass
class PaperDatasetLink:
    """A potential link between a paper and dataset."""

    paper_node_id: str
    dataset_node_id: str
    paper_title: str
    dataset_title: str
    similarity_score: float
    link_type: str  # "explicit", "concept_match", "semantic"
    shared_tasks: list[str] = field(default_factory=list)
    shared_modalities: list[str] = field(default_factory=list)
    shared_regions: list[str] = field(default_factory=list)
    shared_behaviors: list[str] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.5


@dataclass
class LinkingConfig:
    """Configuration for paper-dataset linking."""

    # Similarity thresholds
    min_concept_similarity: float = 0.3
    min_shared_concepts: int = 2

    # Weights for concept types
    task_weight: float = 0.30
    modality_weight: float = 0.25
    region_weight: float = 0.20
    behavior_weight: float = 0.15
    species_weight: float = 0.10

    # Edge types to create
    edge_type_explicit: str = "paper_uses_dataset"
    edge_type_semantic: str = "paper_related_to_dataset"

    # Maximum links per paper/dataset
    max_links_per_node: int = 5


DEFAULT_LINKING_CONFIG = LinkingConfig()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _get_node_concepts(
    graph: KnowledgeGraph,
    node_id: str,
    node_type: str,
) -> dict[str, set[str]]:
    """Extract connected concepts for a node."""

    concepts: dict[str, set[str]] = {
        "task": set(),
        "modality": set(),
        "brain_region": set(),
        "behavioral_event": set(),
        "species": set(),
    }

    # Map edge types to concept categories based on node type
    if node_type == "dataset":
        edge_type_mapping = {
            "dataset_has_task": "task",
            "dataset_has_modality": "modality",
            "dataset_records_region": "brain_region",
            "dataset_has_behavioral_event": "behavioral_event",
            "dataset_has_species": "species",
        }
    else:  # paper
        edge_type_mapping = {
            "paper_studies_task": "task",
            "paper_uses_modality": "modality",
            "paper_mentions_region": "brain_region",
            # Papers don't typically have explicit behavior edges
        }

    for edge_type, category in edge_type_mapping.items():
        neighbors = get_neighbors(
            graph,
            node_id,
            edge_types=[edge_type],
            direction="out",
        )
        concepts[category].update(n.label for n in neighbors)

    return concepts


def compute_paper_dataset_similarity(
    graph: KnowledgeGraph,
    paper_node_id: str,
    dataset_node_id: str,
    config: LinkingConfig | None = None,
) -> PaperDatasetLink:
    """Compute similarity between a paper and dataset based on shared concepts.

    Args:
        graph: Knowledge graph
        paper_node_id: Paper node ID
        dataset_node_id: Dataset node ID
        config: Linking configuration

    Returns:
        PaperDatasetLink with similarity score and shared concepts
    """
    config = config or DEFAULT_LINKING_CONFIG

    paper_node = graph.nodes.get(paper_node_id)
    dataset_node = graph.nodes.get(dataset_node_id)

    if not paper_node or not dataset_node:
        return PaperDatasetLink(
            paper_node_id=paper_node_id,
            dataset_node_id=dataset_node_id,
            paper_title="",
            dataset_title="",
            similarity_score=0.0,
            link_type="none",
        )

    paper_concepts = _get_node_concepts(graph, paper_node_id, "paper")
    dataset_concepts = _get_node_concepts(graph, dataset_node_id, "dataset")

    # Compute shared concepts
    shared = {
        category: paper_concepts[category] & dataset_concepts[category]
        for category in paper_concepts
    }

    # Compute weighted similarity
    weights = {
        "task": config.task_weight,
        "modality": config.modality_weight,
        "brain_region": config.region_weight,
        "behavioral_event": config.behavior_weight,
        "species": config.species_weight,
    }

    total_score = 0.0
    weight_sum = 0.0

    for category, weight in weights.items():
        paper_set = paper_concepts[category]
        dataset_set = dataset_concepts[category]
        union = paper_set | dataset_set

        if union:
            jaccard = len(shared[category]) / len(union)
            total_score += jaccard * weight
            weight_sum += weight

    similarity = total_score / weight_sum if weight_sum > 0 else 0.0

    # Determine link type and evidence
    total_shared = sum(len(s) for s in shared.values())

    if total_shared >= config.min_shared_concepts:
        link_type = "concept_match"
    else:
        link_type = "weak"

    evidence_parts = []
    if shared["task"]:
        evidence_parts.append(f"tasks: {', '.join(sorted(shared['task']))}")
    if shared["modality"]:
        evidence_parts.append(f"modalities: {', '.join(sorted(shared['modality']))}")
    if shared["brain_region"]:
        evidence_parts.append(f"regions: {', '.join(sorted(shared['brain_region']))}")

    evidence = f"Shared {'; '.join(evidence_parts)}" if evidence_parts else "No strong concept overlap"

    return PaperDatasetLink(
        paper_node_id=paper_node_id,
        dataset_node_id=dataset_node_id,
        paper_title=paper_node.label,
        dataset_title=dataset_node.label,
        similarity_score=round(similarity, 4),
        link_type=link_type,
        shared_tasks=sorted(shared["task"]),
        shared_modalities=sorted(shared["modality"]),
        shared_regions=sorted(shared["brain_region"]),
        shared_behaviors=sorted(shared["behavioral_event"]),
        evidence=evidence,
        confidence=min(similarity + 0.2, 1.0) if total_shared >= 2 else similarity,
    )


def find_related_papers_for_dataset(
    graph: KnowledgeGraph,
    dataset_node_id: str,
    min_similarity: float = 0.3,
    top_k: int = 10,
    config: LinkingConfig | None = None,
) -> list[PaperDatasetLink]:
    """Find papers related to a dataset based on shared concepts.

    Args:
        graph: Knowledge graph
        dataset_node_id: Dataset node ID to find papers for
        min_similarity: Minimum similarity threshold
        top_k: Maximum number of results
        config: Linking configuration

    Returns:
        List of PaperDatasetLink sorted by similarity
    """
    config = config or DEFAULT_LINKING_CONFIG
    links: list[PaperDatasetLink] = []

    for node in graph.nodes.values():
        if node.node_type != "paper":
            continue

        link = compute_paper_dataset_similarity(
            graph, node.node_id, dataset_node_id, config
        )

        if link.similarity_score >= min_similarity:
            links.append(link)

    links.sort(key=lambda x: x.similarity_score, reverse=True)
    return links[:top_k]


def find_related_datasets_for_paper(
    graph: KnowledgeGraph,
    paper_node_id: str,
    min_similarity: float = 0.3,
    top_k: int = 10,
    config: LinkingConfig | None = None,
) -> list[PaperDatasetLink]:
    """Find datasets related to a paper based on shared concepts.

    Args:
        graph: Knowledge graph
        paper_node_id: Paper node ID to find datasets for
        min_similarity: Minimum similarity threshold
        top_k: Maximum number of results
        config: Linking configuration

    Returns:
        List of PaperDatasetLink sorted by similarity
    """
    config = config or DEFAULT_LINKING_CONFIG
    links: list[PaperDatasetLink] = []

    for node in graph.nodes.values():
        if node.node_type != "dataset":
            continue

        link = compute_paper_dataset_similarity(
            graph, paper_node_id, node.node_id, config
        )

        if link.similarity_score >= min_similarity:
            links.append(link)

    links.sort(key=lambda x: x.similarity_score, reverse=True)
    return links[:top_k]


def build_paper_dataset_linking_edges(
    graph: KnowledgeGraph,
    config: LinkingConfig | None = None,
) -> list[KnowledgeGraphEdge]:
    """Build semantic linking edges between papers and datasets.

    Args:
        graph: Knowledge graph
        config: Linking configuration

    Returns:
        List of new linking edges
    """
    config = config or DEFAULT_LINKING_CONFIG
    edges: list[KnowledgeGraphEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    paper_nodes = [n for n in graph.nodes.values() if n.node_type == "paper"]

    for paper in paper_nodes:
        links = find_related_datasets_for_paper(
            graph,
            paper.node_id,
            min_similarity=config.min_concept_similarity,
            top_k=config.max_links_per_node,
            config=config,
        )

        for link in links:
            pair = (link.paper_node_id, link.dataset_node_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Check if we have enough shared concepts
            total_shared = (
                len(link.shared_tasks)
                + len(link.shared_modalities)
                + len(link.shared_regions)
                + len(link.shared_behaviors)
            )

            if total_shared < config.min_shared_concepts:
                continue

            edge_type = config.edge_type_semantic
            edge = KnowledgeGraphEdge(
                edge_id=make_edge_id(
                    link.paper_node_id,
                    edge_type,
                    link.dataset_node_id,
                ),
                source_node_id=link.paper_node_id,
                target_node_id=link.dataset_node_id,
                edge_type=edge_type,
                directed=True,
                confidence=link.confidence,
                evidence=[
                    GraphEvidence(
                        evidence_id=f"evidence:semantic_linking:{link.paper_node_id}:{link.dataset_node_id}",
                        source_type="semantic_linking",
                        source_id=link.paper_node_id,
                        source_field="shared_concepts",
                        evidence_text=link.evidence,
                        confidence=link.confidence,
                        extractor_name="neural_search.graph.paper_linking",
                        extractor_version="v1.0.0",
                    )
                ],
                properties={
                    "link_type": link.link_type,
                    "similarity_score": link.similarity_score,
                    "shared_tasks": link.shared_tasks,
                    "shared_modalities": link.shared_modalities,
                    "shared_regions": link.shared_regions,
                    "shared_behaviors": link.shared_behaviors,
                },
                created_at=_now(),
            )
            edges.append(edge)

    return edges


def add_paper_dataset_links_to_graph(
    graph: KnowledgeGraph,
    config: LinkingConfig | None = None,
) -> int:
    """Add semantic paper-dataset linking edges to graph in place.

    Args:
        graph: Knowledge graph to modify
        config: Linking configuration

    Returns:
        Number of edges added
    """
    new_edges = build_paper_dataset_linking_edges(graph, config)
    added = 0

    for edge in new_edges:
        if edge.edge_id not in graph.edges:
            graph.edges[edge.edge_id] = edge
            added += 1

    return added


@dataclass
class LinkingReport:
    """Report on paper-dataset linking status."""

    total_papers: int
    total_datasets: int
    explicit_links: int
    semantic_links: int
    papers_with_links: int
    datasets_with_links: int
    avg_links_per_paper: float
    avg_links_per_dataset: float
    top_linked_datasets: list[tuple[str, int]]
    top_linked_papers: list[tuple[str, int]]


def generate_linking_report(graph: KnowledgeGraph) -> LinkingReport:
    """Generate a report on paper-dataset linking coverage.

    Args:
        graph: Knowledge graph

    Returns:
        LinkingReport with coverage statistics
    """
    paper_nodes = [n for n in graph.nodes.values() if n.node_type == "paper"]
    dataset_nodes = [n for n in graph.nodes.values() if n.node_type == "dataset"]

    explicit_edge_types = {"paper_uses_dataset", "paper_mentions_dataset"}
    semantic_edge_types = {"paper_related_to_dataset"}

    paper_link_counts: dict[str, int] = {p.node_id: 0 for p in paper_nodes}
    dataset_link_counts: dict[str, int] = {d.node_id: 0 for d in dataset_nodes}

    explicit_links = 0
    semantic_links = 0

    for edge in graph.edges.values():
        if edge.edge_type in explicit_edge_types:
            explicit_links += 1
            if edge.source_node_id in paper_link_counts:
                paper_link_counts[edge.source_node_id] += 1
            if edge.target_node_id in dataset_link_counts:
                dataset_link_counts[edge.target_node_id] += 1

        elif edge.edge_type in semantic_edge_types:
            semantic_links += 1
            if edge.source_node_id in paper_link_counts:
                paper_link_counts[edge.source_node_id] += 1
            if edge.target_node_id in dataset_link_counts:
                dataset_link_counts[edge.target_node_id] += 1

    papers_with_links = sum(1 for c in paper_link_counts.values() if c > 0)
    datasets_with_links = sum(1 for c in dataset_link_counts.values() if c > 0)

    total_paper_links = sum(paper_link_counts.values())
    total_dataset_links = sum(dataset_link_counts.values())

    # Get node labels for top linked nodes
    top_datasets = sorted(dataset_link_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_papers = sorted(paper_link_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    top_linked_datasets = [
        (graph.nodes[node_id].label if node_id in graph.nodes else node_id, count)
        for node_id, count in top_datasets
        if count > 0
    ]
    top_linked_papers = [
        (graph.nodes[node_id].label if node_id in graph.nodes else node_id, count)
        for node_id, count in top_papers
        if count > 0
    ]

    return LinkingReport(
        total_papers=len(paper_nodes),
        total_datasets=len(dataset_nodes),
        explicit_links=explicit_links,
        semantic_links=semantic_links,
        papers_with_links=papers_with_links,
        datasets_with_links=datasets_with_links,
        avg_links_per_paper=total_paper_links / len(paper_nodes) if paper_nodes else 0,
        avg_links_per_dataset=total_dataset_links / len(dataset_nodes) if dataset_nodes else 0,
        top_linked_datasets=top_linked_datasets,
        top_linked_papers=top_linked_papers,
    )


def find_doi_based_links(
    graph: KnowledgeGraph,
    datasets: list[dict[str, Any]],
) -> list[KnowledgeGraphEdge]:
    """Find explicit links between papers and datasets based on DOI/citation.

    This function creates edges when:
    1. A dataset record has a linked_publications field with DOIs
    2. A paper mentions a specific dataset ID
    3. Author overlap suggests connection

    Args:
        graph: Knowledge graph
        datasets: Raw dataset records with metadata

    Returns:
        List of explicit linking edges
    """
    edges: list[KnowledgeGraphEdge] = []
    seen: set[str] = set()

    # Build a paper index by DOI
    paper_by_doi: dict[str, str] = {}  # DOI -> node_id
    for node in graph.nodes.values():
        if node.node_type == "paper":
            props = node.properties or {}
            doi = props.get("doi")
            if doi:
                paper_by_doi[doi.lower()] = node.node_id

    # For each dataset, check linked publications
    for ds in datasets:
        ds_id = ds.get("id") or ds.get("source_id", "")
        if not ds_id:
            continue

        ds_node_id = make_node_id("dataset", ds_id)
        if ds_node_id not in graph.nodes:
            continue

        linked_pubs = ds.get("linked_publications", [])
        if not linked_pubs:
            linked_pubs = ds.get("publications", [])

        for pub in linked_pubs:
            doi = pub.get("doi", "") if isinstance(pub, dict) else str(pub)
            if not doi:
                continue

            doi_clean = doi.lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            paper_node_id = paper_by_doi.get(doi_clean)

            if not paper_node_id:
                continue

            edge_id = make_edge_id(paper_node_id, "paper_uses_dataset", ds_node_id)
            if edge_id in seen or edge_id in graph.edges:
                continue
            seen.add(edge_id)

            edge = KnowledgeGraphEdge(
                edge_id=edge_id,
                source_node_id=paper_node_id,
                target_node_id=ds_node_id,
                edge_type="paper_uses_dataset",
                directed=True,
                confidence=0.95,  # High confidence for DOI-based links
                evidence=[
                    GraphEvidence(
                        evidence_id=f"evidence:doi_link:{paper_node_id}:{ds_node_id}",
                        source_type="doi_linking",
                        source_id=ds_node_id,
                        source_field="linked_publications",
                        evidence_text=f"Explicit DOI link: {doi_clean}",
                        confidence=0.95,
                        extractor_name="neural_search.graph.paper_linking",
                        extractor_version="v1.1.0",
                    )
                ],
                properties={
                    "link_type": "explicit_doi",
                    "doi": doi_clean,
                },
                created_at=_now(),
            )
            edges.append(edge)

    return edges


def find_author_based_links(
    graph: KnowledgeGraph,
    min_author_overlap: int = 2,
) -> list[KnowledgeGraphEdge]:
    """Find paper-dataset links based on author overlap.

    Creates links when papers and datasets share multiple authors,
    suggesting the paper may have used or produced the dataset.

    Args:
        graph: Knowledge graph
        min_author_overlap: Minimum number of shared authors

    Returns:
        List of author-based linking edges
    """
    edges: list[KnowledgeGraphEdge] = []
    seen: set[str] = set()

    # Build author sets for papers and datasets
    paper_authors: dict[str, set[str]] = {}
    dataset_authors: dict[str, set[str]] = {}

    for node in graph.nodes.values():
        props = node.properties or {}
        authors = props.get("authors", [])
        if not authors:
            authors = props.get("contributors", [])

        author_names = set()
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name", "")
            else:
                name = str(author)
            if name:
                author_names.add(name.lower())

        if node.node_type == "paper" and author_names:
            paper_authors[node.node_id] = author_names
        elif node.node_type == "dataset" and author_names:
            dataset_authors[node.node_id] = author_names

    # Find overlaps
    for paper_id, paper_auth in paper_authors.items():
        for ds_id, ds_auth in dataset_authors.items():
            overlap = paper_auth & ds_auth
            if len(overlap) >= min_author_overlap:
                edge_id = make_edge_id(paper_id, "paper_related_to_dataset", ds_id)
                if edge_id in seen or edge_id in graph.edges:
                    continue
                seen.add(edge_id)

                confidence = min(0.5 + len(overlap) * 0.1, 0.85)

                edge = KnowledgeGraphEdge(
                    edge_id=edge_id,
                    source_node_id=paper_id,
                    target_node_id=ds_id,
                    edge_type="paper_related_to_dataset",
                    directed=True,
                    confidence=confidence,
                    evidence=[
                        GraphEvidence(
                            evidence_id=f"evidence:author_overlap:{paper_id}:{ds_id}",
                            source_type="author_linking",
                            source_id=paper_id,
                            source_field="authors",
                            evidence_text=f"Shared authors: {', '.join(sorted(overlap)[:3])}",
                            confidence=confidence,
                            extractor_name="neural_search.graph.paper_linking",
                            extractor_version="v1.1.0",
                        )
                    ],
                    properties={
                        "link_type": "author_overlap",
                        "shared_author_count": len(overlap),
                        "shared_authors": sorted(overlap)[:5],
                    },
                    created_at=_now(),
                )
                edges.append(edge)

    return edges
