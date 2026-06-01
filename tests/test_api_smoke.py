import asyncio

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
