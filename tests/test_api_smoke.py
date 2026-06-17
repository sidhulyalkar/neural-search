import asyncio
import json

from apps.api import main as api_main
from neural_search.schemas import DatasetCompareRequest, SearchRequest


def test_core_api_smoke_flow():
    async def run_flow():
        health = await api_main.healthz()
        assert health == {"status": "ok"}

        search = await api_main.search(
            SearchRequest(query="go/no-go calcium imaging", limit=2)
        )
        assert search.total_count >= 1

        dataset_id = search.results[0].dataset["source_id"]

        dataset = await api_main.get_dataset(dataset_id)
        assert dataset["source_id"] == dataset_id

        card = await api_main.get_dataset_card(dataset_id)
        assert card["dataset_id"] == dataset_id

        datasets = await api_main.list_datasets(limit=2, offset=0, qa_status=None)
        listed_ids = [item["source_id"] for item in datasets.datasets]

        comparison = await api_main.compare_datasets_endpoint(
            DatasetCompareRequest(dataset_ids=listed_ids[:2])
        )
        assert len(comparison.dataset_ids) == 2

        report = await api_main.get_compilation_report()
        assert report["total_datasets"] >= 1

    asyncio.run(run_flow())


def test_literature_search_endpoint(tmp_path, monkeypatch):
    shard_dir = tmp_path / "openalex"
    shard_dir.mkdir()
    (shard_dir / "tier1_batch_0000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper:openalex:W1",
                "title": "Hippocampal theta supports memory",
                "abstract": "Theta rhythms in CA1 support spatial memory.",
                "citation_count": 120,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    findings_path = tmp_path / "findings.jsonl"
    findings_path.write_text(
        json.dumps(
            {
                "finding_id": "paper:openalex:W1:f0",
                "finding_text": "CA1 theta increases during spatial memory retrieval.",
                "result_direction": "increase",
                "regions": ["CA1"],
                "species": ["mouse"],
                "modalities": ["ephys"],
                "paper_id": "paper:openalex:W1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LITERATURE_SHARD_DIR", shard_dir)
    monkeypatch.setattr(api_main, "LITERATURE_FINDINGS_PATH", findings_path)
    links_path = tmp_path / "paper_dataset_links.jsonl"
    links_path.write_text(
        json.dumps(
            {
                "dataset_record_id": "dandi:000001",
                "paper_openalex_id": "W1",
                "match_method": "title_fuzzy_local",
                "confidence": 0.95,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LITERATURE_LINKS_PATH", links_path)

    async def run_flow():
        response = await api_main.search_literature(
            api_main.LiteratureSearchRequest(query="CA1 theta memory", limit=3)
        )
        assert response.total_papers == 1
        assert response.total_findings == 1
        assert response.papers[0]["result_type"] == "paper"
        assert response.papers[0]["linked_datasets"] == ["dandi:000001"]
        assert response.findings[0]["result_type"] == "finding"

    asyncio.run(run_flow())
