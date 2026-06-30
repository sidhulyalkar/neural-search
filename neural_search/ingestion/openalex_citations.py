"""Fetch and store citation edges from OpenAlex referenced_works."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

OPENALEX_API = "https://api.openalex.org"
POLITE_MAILTO = "neuralsearch@example.com"
SLEEP_BETWEEN_REQUESTS = 0.1
LOG_EVERY = 10

logger = logging.getLogger(__name__)


@dataclass
class CitationEdge:
    citing_paper_id: str   # OpenAlex work ID e.g. "W2741809807"
    cited_paper_id: str    # OpenAlex work ID
    citing_year: int | None


def _strip_prefix(work_url: str) -> str:
    """Strip https://openalex.org/ prefix if present."""
    return work_url.replace("https://openalex.org/", "").strip()


def fetch_references_for_work(work_id: str) -> list[str]:
    """
    Fetch the referenced_works list for one OpenAlex work.

    Returns list of referenced work IDs (short form, W-prefixed).
    Handles 404 and network errors gracefully (returns []).
    """
    url = f"{OPENALEX_API}/works/{work_id}"
    params = {"select": "referenced_works,publication_year", "mailto": POLITE_MAILTO}
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url, params=params)
        if response.status_code == 404:
            logger.debug("Work %s not found in OpenAlex", work_id)
            return []
        response.raise_for_status()
        data = response.json()
        raw_refs: list[str] = data.get("referenced_works") or []
        return [_strip_prefix(ref) for ref in raw_refs if ref]
    except Exception as exc:
        logger.warning("Failed to fetch references for %s: %s", work_id, exc)
        return []


def build_citation_edges(
    corpus_papers: list[dict],
    max_workers: int = 5,
) -> list[CitationEdge]:
    """
    Fetch citation edges for all corpus papers with an openalex_id.

    Only includes edges where BOTH citing and cited paper are in the corpus
    (inner-corpus citation graph). Uses a thread pool for concurrency.
    """
    # Build index of corpus openalex IDs → publication_year
    corpus_index: dict[str, int | None] = {}
    for paper in corpus_papers:
        oa_id = paper.get("openalex_id")
        if oa_id:
            short_id = _strip_prefix(str(oa_id))
            corpus_index[short_id] = paper.get("publication_year")

    if not corpus_index:
        logger.warning("No corpus papers with openalex_id found")
        return []

    logger.info("Building citation edges for %d corpus papers", len(corpus_index))

    def _fetch_one(work_id: str) -> tuple[str, list[str]]:
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return work_id, fetch_references_for_work(work_id)

    edges: list[CitationEdge] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, wid): wid for wid in corpus_index}
        for future in as_completed(futures):
            work_id, refs = future.result()
            completed += 1
            if completed % LOG_EVERY == 0:
                logger.info("Progress: %d/%d papers processed", completed, len(corpus_index))
            citing_year = corpus_index.get(work_id)
            for ref_id in refs:
                if ref_id in corpus_index:
                    edges.append(CitationEdge(
                        citing_paper_id=work_id,
                        cited_paper_id=ref_id,
                        citing_year=citing_year,
                    ))

    logger.info("Built %d inner-corpus citation edges", len(edges))
    return edges


def save_citation_edges(edges: list[CitationEdge], output_path: Path) -> None:
    """Save citation edges as JSONL to artifacts/citations/citation_edges.jsonl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(asdict(edge)) + "\n")
    logger.info("Saved %d citation edges to %s", len(edges), output_path)


if __name__ == "__main__":
    import glob

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).parent.parent.parent
    corpus_dir = repo_root / "data" / "corpus" / "normalized"
    output_path = repo_root / "artifacts" / "citations" / "citation_edges.jsonl"

    # Load all normalized paper records
    corpus_papers: list[dict] = []
    for json_path in glob.glob(str(corpus_dir / "**" / "*.json"), recursive=True):
        try:
            with open(json_path, encoding="utf-8") as f:
                record = json.load(f)
            if isinstance(record, list):
                corpus_papers.extend(record)
            elif isinstance(record, dict):
                corpus_papers.append(record)
        except Exception as exc:
            logger.warning("Could not load %s: %s", json_path, exc)

    logger.info("Loaded %d paper records from corpus", len(corpus_papers))
    edges = build_citation_edges(corpus_papers)
    save_citation_edges(edges, output_path)
    print(f"Saved {len(edges)} citation edges to {output_path}")
