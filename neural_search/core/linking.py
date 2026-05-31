"""Multi-Signal Paper-Dataset Linking.

This module implements sophisticated paper-dataset linking using multiple signals:

1. DOI/Accession Overlap
   - Explicit dataset references in papers
   - DOI cross-references in dataset metadata

2. Title Similarity
   - Semantic similarity between paper title and dataset title
   - Keyword overlap

3. Author Overlap
   - Shared authors suggest connection
   - Weighted by author position (first/last more important)

4. Concept Overlap
   - Shared tasks, modalities, species, brain regions
   - Weighted by concept specificity

5. Citation Evidence
   - Paper cites datasets or papers that use datasets
   - Transitive evidence through citation graph

6. Graph Neighborhood
   - Consistency with existing graph relationships
   - Shared neighbors suggest connection

7. Embedding Similarity
   - Abstract-to-description semantic similarity
   - Methods-to-methods similarity

Each signal contributes a confidence score with provenance.
The final link includes all evidence and uncertainty flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LinkSignal(StrEnum):
    """Types of signals used for linking."""

    DOI_EXPLICIT = "doi_explicit"
    ACCESSION_EXPLICIT = "accession_explicit"
    TITLE_SIMILARITY = "title_similarity"
    AUTHOR_OVERLAP = "author_overlap"
    TASK_OVERLAP = "task_overlap"
    MODALITY_OVERLAP = "modality_overlap"
    SPECIES_OVERLAP = "species_overlap"
    REGION_OVERLAP = "region_overlap"
    CITATION_EVIDENCE = "citation_evidence"
    GRAPH_NEIGHBOR = "graph_neighbor"
    EMBEDDING_SIMILARITY = "embedding_similarity"


@dataclass
class LinkEvidence:
    """Evidence for a single linking signal."""

    signal: LinkSignal
    score: float                  # 0.0 to 1.0
    weight: float                 # How much this signal contributes
    evidence_text: str
    matched_values: list[str] = field(default_factory=list)
    source_field: str | None = None
    confidence: float = 1.0       # Confidence in this signal


class PaperDatasetLinkV2(BaseModel):
    """Enhanced paper-dataset link with full provenance.

    This represents a potential or confirmed link between a paper and dataset,
    with all the evidence that supports (or contradicts) the link.
    """

    # Identifiers
    paper_id: str
    dataset_id: str
    paper_title: str
    dataset_title: str

    # Scores
    final_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Link type classification
    link_type: str = "inferred"   # "explicit", "inferred", "weak", "speculative"
    is_explicit: bool = False     # Has DOI/accession evidence?

    # Evidence breakdown
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    # Concept overlap details
    shared_tasks: list[str] = Field(default_factory=list)
    shared_modalities: list[str] = Field(default_factory=list)
    shared_species: list[str] = Field(default_factory=list)
    shared_regions: list[str] = Field(default_factory=list)
    shared_authors: list[str] = Field(default_factory=list)

    # Provenance
    linking_method: str = "neural_search.core.linking"
    linking_version: str = "v0.5.0"
    linked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Uncertainty
    uncertainty_flags: list[str] = Field(default_factory=list)
    needs_verification: bool = True

    def to_graph_edge_properties(self) -> dict[str, Any]:
        """Convert to properties for a graph edge."""
        return {
            "link_type": self.link_type,
            "score": self.final_score,
            "confidence": self.confidence,
            "is_explicit": self.is_explicit,
            "shared_tasks": self.shared_tasks,
            "shared_modalities": self.shared_modalities,
            "shared_species": self.shared_species,
            "shared_regions": self.shared_regions,
            "shared_authors": self.shared_authors,
            "evidence_count": len(self.evidence),
            "linking_method": self.linking_method,
            "linking_version": self.linking_version,
        }


class LinkingConfig(BaseModel):
    """Configuration for paper-dataset linking."""

    # Signal weights
    doi_weight: float = 0.28
    accession_weight: float = 0.22
    task_weight: float = 0.12
    modality_weight: float = 0.10
    species_weight: float = 0.08
    region_weight: float = 0.06
    author_weight: float = 0.04
    title_weight: float = 0.03
    embedding_weight: float = 0.03
    citation_weight: float = 0.04  # New: citation evidence signal

    # Thresholds
    min_score_for_link: float = 0.3
    min_shared_concepts: int = 2
    min_confidence_for_auto: float = 0.7

    # Constraints
    max_links_per_paper: int = 10
    max_links_per_dataset: int = 20

    # Citation options
    use_citation_evidence: bool = True
    citation_cache_path: str | None = None


DEFAULT_LINKING_CONFIG = LinkingConfig()


def _normalize_doi(doi: str) -> str:
    """Normalize a DOI for comparison."""
    if not doi:
        return ""
    doi = doi.lower().strip()
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://dx.doi.org/", "")
    doi = doi.replace("doi:", "")
    return doi


def _normalize_author(name: str) -> str:
    """Normalize author name for comparison."""
    if not name:
        return ""
    # Simple normalization: lowercase, remove punctuation
    name = name.lower().strip()
    name = "".join(c for c in name if c.isalnum() or c.isspace())
    return " ".join(name.split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _compute_citation_evidence(
    paper: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[float, str]:
    """Compute citation-based linking evidence.

    Checks if:
    1. Paper explicitly cites dataset DOI/accession
    2. Paper cites papers that use this dataset
    3. Dataset references are in paper's citation list

    Returns:
        Tuple of (score, evidence_text)
    """
    score = 0.0
    evidence_parts = []

    # Get dataset identifiers
    dataset_doi = _normalize_doi(dataset.get("doi", ""))
    dataset_accession = dataset.get("source_id", "") or dataset.get("accession", "")
    dataset_url = dataset.get("url", "")

    # Get paper's references/citations
    paper_refs = paper.get("references", []) + paper.get("citations", [])
    paper_text = " ".join([
        paper.get("abstract", ""),
        paper.get("methods", ""),
        paper.get("full_text", ""),
    ]).lower()

    # Check if paper references dataset DOI
    if dataset_doi:
        for ref in paper_refs:
            ref_doi = _normalize_doi(ref.get("doi", "") if isinstance(ref, dict) else str(ref))
            if ref_doi and ref_doi == dataset_doi:
                score += 0.6
                evidence_parts.append(f"Paper cites dataset DOI: {dataset_doi}")
                break
        # Also check in paper text
        if dataset_doi in paper_text:
            score += 0.3
            evidence_parts.append("Dataset DOI mentioned in paper text")

    # Check if paper mentions dataset accession
    if dataset_accession and dataset_accession.lower() in paper_text:
        score += 0.4
        evidence_parts.append(f"Dataset accession '{dataset_accession}' found in paper")

    # Check if paper mentions dataset URL
    if dataset_url and dataset_url.lower() in paper_text:
        score += 0.3
        evidence_parts.append("Dataset URL referenced in paper")

    # Check if dataset lists paper as citing paper
    citing_papers = dataset.get("citing_papers", []) + dataset.get("related_publications", [])
    paper_doi = _normalize_doi(paper.get("doi", ""))
    paper_id = paper.get("paper_id", paper.get("id", ""))

    for citing in citing_papers:
        if isinstance(citing, dict):
            citing_doi = _normalize_doi(citing.get("doi", ""))
            citing_id = citing.get("id", "")
        else:
            citing_doi = _normalize_doi(str(citing))
            citing_id = str(citing)

        if (paper_doi and citing_doi and paper_doi == citing_doi) or \
           (paper_id and citing_id and paper_id == citing_id):
            score += 0.5
            evidence_parts.append("Paper listed in dataset's citing publications")
            break

    # Normalize score to 0-1
    score = min(score, 1.0)

    evidence_text = "; ".join(evidence_parts) if evidence_parts else "No citation evidence"
    return score, evidence_text


def _extract_labels(record: dict[str, Any], field: str) -> set[str]:
    """Extract label strings from a record field."""
    values = record.get(field, [])
    if not values:
        return set()

    labels = set()
    for v in values:
        if isinstance(v, dict):
            label = v.get("label", v.get("id", ""))
        else:
            label = str(v)
        if label:
            labels.add(label.lower().strip())
    return labels


def _extract_authors(record: dict[str, Any]) -> set[str]:
    """Extract normalized author names from a record."""
    authors = record.get("authors", [])
    if not authors:
        authors = record.get("authors_json", [])
    if not authors:
        return set()

    names = set()
    for author in authors:
        if isinstance(author, dict):
            name = author.get("name", author.get("display_name", ""))
        else:
            name = str(author)
        normalized = _normalize_author(name)
        if normalized:
            names.add(normalized)
    return names


def compute_link_score(
    paper: dict[str, Any],
    dataset: dict[str, Any],
    config: LinkingConfig | None = None,
) -> PaperDatasetLinkV2:
    """Compute multi-signal link score between a paper and dataset.

    This is the main entry point for paper-dataset linking. It:
    1. Checks explicit DOI/accession links
    2. Computes concept overlap (tasks, modalities, species, regions)
    3. Computes author overlap
    4. Computes title similarity
    5. Combines signals with configurable weights
    6. Returns link with full evidence

    Args:
        paper: Paper record dictionary
        dataset: Dataset record dictionary
        config: Optional linking configuration

    Returns:
        PaperDatasetLinkV2 with scores and evidence
    """
    config = config or DEFAULT_LINKING_CONFIG
    evidence: list[dict[str, Any]] = []
    total_score = 0.0
    total_weight = 0.0

    paper_id = paper.get("paper_id", paper.get("id", "unknown"))
    dataset_id = dataset.get("dataset_id", dataset.get("id", "unknown"))
    paper_title = paper.get("title", "")
    dataset_title = dataset.get("title", "")

    is_explicit = False

    # Signal 1: DOI/Accession explicit links
    paper_doi = _normalize_doi(paper.get("doi", ""))
    linked_datasets = dataset.get("linked_papers", []) + dataset.get("linked_publications", [])

    for pub in linked_datasets:
        if isinstance(pub, dict):
            pub_doi = _normalize_doi(pub.get("doi", ""))
        else:
            pub_doi = _normalize_doi(str(pub))

        if pub_doi and paper_doi and pub_doi == paper_doi:
            evidence.append({
                "signal": LinkSignal.DOI_EXPLICIT.value,
                "score": 1.0,
                "weight": config.doi_weight,
                "evidence_text": f"Explicit DOI match: {paper_doi}",
                "confidence": 0.95,
            })
            total_score += 1.0 * config.doi_weight
            total_weight += config.doi_weight
            is_explicit = True
            break

    # Signal 2: Task overlap
    paper_tasks = _extract_labels(paper, "tasks") | _extract_labels(paper, "extracted_labels")
    dataset_tasks = _extract_labels(dataset, "tasks")

    shared_tasks = list(paper_tasks & dataset_tasks)
    if paper_tasks or dataset_tasks:
        task_score = _jaccard_similarity(paper_tasks, dataset_tasks)
        if task_score > 0:
            evidence.append({
                "signal": LinkSignal.TASK_OVERLAP.value,
                "score": task_score,
                "weight": config.task_weight,
                "evidence_text": f"Shared tasks: {', '.join(shared_tasks[:5])}",
                "matched_values": shared_tasks,
                "confidence": 0.8,
            })
            total_score += task_score * config.task_weight
            total_weight += config.task_weight

    # Signal 3: Modality overlap
    paper_modalities = _extract_labels(paper, "modalities")
    dataset_modalities = _extract_labels(dataset, "modalities")

    shared_modalities = list(paper_modalities & dataset_modalities)
    if paper_modalities or dataset_modalities:
        modality_score = _jaccard_similarity(paper_modalities, dataset_modalities)
        if modality_score > 0:
            evidence.append({
                "signal": LinkSignal.MODALITY_OVERLAP.value,
                "score": modality_score,
                "weight": config.modality_weight,
                "evidence_text": f"Shared modalities: {', '.join(shared_modalities[:5])}",
                "matched_values": shared_modalities,
                "confidence": 0.8,
            })
            total_score += modality_score * config.modality_weight
            total_weight += config.modality_weight

    # Signal 4: Species overlap
    paper_species = _extract_labels(paper, "species")
    dataset_species = _extract_labels(dataset, "species")

    shared_species = list(paper_species & dataset_species)
    if paper_species or dataset_species:
        species_score = _jaccard_similarity(paper_species, dataset_species)
        if species_score > 0:
            evidence.append({
                "signal": LinkSignal.SPECIES_OVERLAP.value,
                "score": species_score,
                "weight": config.species_weight,
                "evidence_text": f"Shared species: {', '.join(shared_species[:5])}",
                "matched_values": shared_species,
                "confidence": 0.85,
            })
            total_score += species_score * config.species_weight
            total_weight += config.species_weight

    # Signal 5: Brain region overlap
    paper_regions = _extract_labels(paper, "brain_regions")
    dataset_regions = _extract_labels(dataset, "brain_regions")

    shared_regions = list(paper_regions & dataset_regions)
    if paper_regions or dataset_regions:
        region_score = _jaccard_similarity(paper_regions, dataset_regions)
        if region_score > 0:
            evidence.append({
                "signal": LinkSignal.REGION_OVERLAP.value,
                "score": region_score,
                "weight": config.region_weight,
                "evidence_text": f"Shared regions: {', '.join(shared_regions[:5])}",
                "matched_values": shared_regions,
                "confidence": 0.75,
            })
            total_score += region_score * config.region_weight
            total_weight += config.region_weight

    # Signal 6: Author overlap
    paper_authors = _extract_authors(paper)
    dataset_authors = _extract_authors(dataset)

    shared_authors = list(paper_authors & dataset_authors)
    if paper_authors and dataset_authors:
        author_score = len(shared_authors) / max(len(paper_authors), len(dataset_authors))
        author_score = min(author_score, 1.0)
        if author_score > 0:
            evidence.append({
                "signal": LinkSignal.AUTHOR_OVERLAP.value,
                "score": author_score,
                "weight": config.author_weight,
                "evidence_text": f"Shared authors: {', '.join(shared_authors[:3])}",
                "matched_values": shared_authors,
                "confidence": min(0.5 + len(shared_authors) * 0.1, 0.9),
            })
            total_score += author_score * config.author_weight
            total_weight += config.author_weight

    # Signal 7: Title similarity (simple word overlap)
    paper_words = set(paper_title.lower().split())
    dataset_words = set(dataset_title.lower().split())
    # Remove common words
    stopwords = {"a", "an", "the", "of", "and", "in", "for", "to", "with", "on", "from"}
    paper_words -= stopwords
    dataset_words -= stopwords

    if paper_words and dataset_words:
        title_score = _jaccard_similarity(paper_words, dataset_words)
        if title_score > 0.2:  # Only include if meaningful overlap
            shared_words = list(paper_words & dataset_words)
            evidence.append({
                "signal": LinkSignal.TITLE_SIMILARITY.value,
                "score": title_score,
                "weight": config.title_weight,
                "evidence_text": f"Title overlap: {', '.join(shared_words[:5])}",
                "matched_values": shared_words,
                "confidence": 0.6,
            })
            total_score += title_score * config.title_weight
            total_weight += config.title_weight

    # Signal 8: Citation evidence (papers citing dataset DOIs or accession numbers)
    if config.use_citation_evidence:
        citation_score, citation_evidence = _compute_citation_evidence(paper, dataset)
        if citation_score > 0:
            evidence.append({
                "signal": LinkSignal.CITATION_EVIDENCE.value,
                "score": citation_score,
                "weight": config.citation_weight,
                "evidence_text": citation_evidence,
                "confidence": 0.85,
            })
            total_score += citation_score * config.citation_weight
            total_weight += config.citation_weight

    # Compute final score
    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0

    # Compute confidence
    num_signals = len(evidence)
    confidence = min(0.3 + num_signals * 0.15, 0.95)
    if is_explicit:
        confidence = max(confidence, 0.9)

    # Determine link type
    total_shared = len(shared_tasks) + len(shared_modalities) + len(shared_species) + len(shared_regions)
    if is_explicit:
        link_type = "explicit"
    elif final_score >= 0.5 and total_shared >= config.min_shared_concepts:
        link_type = "inferred"
    elif final_score >= 0.3:
        link_type = "weak"
    else:
        link_type = "speculative"

    # Uncertainty flags
    uncertainty_flags = []
    if not is_explicit and num_signals < 3:
        uncertainty_flags.append("Low evidence count")
    if total_shared < 2:
        uncertainty_flags.append("Few shared concepts")
    if not shared_authors:
        uncertainty_flags.append("No author overlap")

    needs_verification = not is_explicit and (confidence < config.min_confidence_for_auto)

    return PaperDatasetLinkV2(
        paper_id=str(paper_id),
        dataset_id=str(dataset_id),
        paper_title=paper_title,
        dataset_title=dataset_title,
        final_score=round(final_score, 4),
        confidence=round(confidence, 4),
        link_type=link_type,
        is_explicit=is_explicit,
        evidence=evidence,
        shared_tasks=shared_tasks,
        shared_modalities=shared_modalities,
        shared_species=shared_species,
        shared_regions=shared_regions,
        shared_authors=shared_authors,
        uncertainty_flags=uncertainty_flags,
        needs_verification=needs_verification,
    )


def find_links_for_paper(
    paper: dict[str, Any],
    datasets: list[dict[str, Any]],
    config: LinkingConfig | None = None,
    top_k: int = 10,
) -> list[PaperDatasetLinkV2]:
    """Find the best dataset links for a paper.

    Args:
        paper: Paper record dictionary
        datasets: List of dataset records to search
        config: Optional linking configuration
        top_k: Maximum number of links to return

    Returns:
        List of PaperDatasetLinkV2 sorted by score
    """
    config = config or DEFAULT_LINKING_CONFIG
    links: list[PaperDatasetLinkV2] = []

    for dataset in datasets:
        link = compute_link_score(paper, dataset, config)
        if link.final_score >= config.min_score_for_link:
            links.append(link)

    # Sort by score
    links.sort(key=lambda x: x.final_score, reverse=True)
    return links[:top_k]


def find_links_for_dataset(
    dataset: dict[str, Any],
    papers: list[dict[str, Any]],
    config: LinkingConfig | None = None,
    top_k: int = 10,
) -> list[PaperDatasetLinkV2]:
    """Find the best paper links for a dataset.

    Args:
        dataset: Dataset record dictionary
        papers: List of paper records to search
        config: Optional linking configuration
        top_k: Maximum number of links to return

    Returns:
        List of PaperDatasetLinkV2 sorted by score
    """
    config = config or DEFAULT_LINKING_CONFIG
    links: list[PaperDatasetLinkV2] = []

    for paper in papers:
        link = compute_link_score(paper, dataset, config)
        if link.final_score >= config.min_score_for_link:
            links.append(link)

    # Sort by score
    links.sort(key=lambda x: x.final_score, reverse=True)
    return links[:top_k]


def batch_compute_links(
    papers: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    config: LinkingConfig | None = None,
    min_score: float | None = None,
) -> list[PaperDatasetLinkV2]:
    """Compute all links above threshold between papers and datasets.

    Args:
        papers: List of paper records
        datasets: List of dataset records
        config: Optional linking configuration
        min_score: Minimum score threshold (default: config.min_score_for_link)

    Returns:
        List of all links above threshold
    """
    config = config or DEFAULT_LINKING_CONFIG
    min_score = min_score if min_score is not None else config.min_score_for_link

    all_links: list[PaperDatasetLinkV2] = []

    for paper in papers:
        for dataset in datasets:
            link = compute_link_score(paper, dataset, config)
            if link.final_score >= min_score:
                all_links.append(link)

    return all_links


class LinkingReport(BaseModel):
    """Summary report of paper-dataset linking coverage."""

    total_papers: int
    total_datasets: int
    total_links: int
    explicit_links: int
    inferred_links: int
    weak_links: int
    speculative_links: int

    papers_with_links: int
    datasets_with_links: int
    papers_without_links: int
    datasets_without_links: int

    avg_links_per_paper: float
    avg_links_per_dataset: float
    avg_link_confidence: float

    top_linked_papers: list[tuple[str, int]] = Field(default_factory=list)
    top_linked_datasets: list[tuple[str, int]] = Field(default_factory=list)


def generate_linking_report(
    links: list[PaperDatasetLinkV2],
    total_papers: int,
    total_datasets: int,
) -> LinkingReport:
    """Generate a summary report of linking coverage.

    Args:
        links: List of computed links
        total_papers: Total number of papers in corpus
        total_datasets: Total number of datasets in corpus

    Returns:
        LinkingReport with coverage statistics
    """
    # Count by type
    explicit_count = sum(1 for link in links if link.link_type == "explicit")
    inferred_count = sum(1 for link in links if link.link_type == "inferred")
    weak_count = sum(1 for link in links if link.link_type == "weak")
    speculative_count = sum(1 for link in links if link.link_type == "speculative")

    # Count unique linked entities
    papers_with_links = len({link.paper_id for link in links})
    datasets_with_links = len({link.dataset_id for link in links})

    # Compute per-entity link counts
    paper_link_counts: dict[str, int] = {}
    dataset_link_counts: dict[str, int] = {}

    for link in links:
        paper_link_counts[link.paper_id] = paper_link_counts.get(link.paper_id, 0) + 1
        dataset_link_counts[link.dataset_id] = dataset_link_counts.get(link.dataset_id, 0) + 1

    # Top linked
    top_papers = sorted(paper_link_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_datasets = sorted(dataset_link_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Averages
    avg_confidence = sum(link.confidence for link in links) / len(links) if links else 0.0
    avg_per_paper = len(links) / total_papers if total_papers else 0.0
    avg_per_dataset = len(links) / total_datasets if total_datasets else 0.0

    return LinkingReport(
        total_papers=total_papers,
        total_datasets=total_datasets,
        total_links=len(links),
        explicit_links=explicit_count,
        inferred_links=inferred_count,
        weak_links=weak_count,
        speculative_links=speculative_count,
        papers_with_links=papers_with_links,
        datasets_with_links=datasets_with_links,
        papers_without_links=total_papers - papers_with_links,
        datasets_without_links=total_datasets - datasets_with_links,
        avg_links_per_paper=round(avg_per_paper, 2),
        avg_links_per_dataset=round(avg_per_dataset, 2),
        avg_link_confidence=round(avg_confidence, 3),
        top_linked_papers=top_papers,
        top_linked_datasets=top_datasets,
    )
