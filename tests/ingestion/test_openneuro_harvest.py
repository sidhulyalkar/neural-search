from __future__ import annotations

import httpx
import respx

from neural_search.ingestion.openneuro import fetch_all_openneuro

OPENNEURO_GQL = "https://openneuro.org/crn/graphql"

STUB_NODE = {"id": "ds000001", "name": "Visual EEG Experiment", "created": "2021-01-01", "public": True,
             "latestSnapshot": {"tag": "1.0.0", "created": "2021-01-01", "size": 10000, "readme": None,
                                "summary": {"subjects": 20, "tasks": ["visual"], "modalities": ["eeg"]}}}

def _gql_response(nodes: list, has_next: bool, cursor: str | None) -> dict:
    return {
        "data": {
            "datasets": {
                "edges": [{"cursor": f"cur{i}", "node": n} for i, n in enumerate(nodes)],
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
            }
        }
    }


@respx.mock
def test_fetch_all_openneuro_single_page() -> None:
    respx.post(OPENNEURO_GQL).mock(
        return_value=httpx.Response(200, json=_gql_response([STUB_NODE], False, None))
    )
    records = fetch_all_openneuro(page_size=100)
    assert len(records) == 1
    assert records[0]["source"] == "openneuro"
    assert records[0]["source_id"] == "ds000001"


@respx.mock
def test_fetch_all_openneuro_two_pages() -> None:
    node2 = dict(STUB_NODE, id="ds000002")
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = __import__("json").loads(request.content)
        after = (body.get("variables") or {}).get("after")
        if after is None:
            return httpx.Response(200, json=_gql_response([STUB_NODE], True, "cursor_abc"))
        return httpx.Response(200, json=_gql_response([node2], False, None))

    respx.post(OPENNEURO_GQL).mock(side_effect=handler)
    records = fetch_all_openneuro(page_size=1)
    assert len(records) == 2
    assert {r["source_id"] for r in records} == {"ds000001", "ds000002"}
    assert call_count == 2


@respx.mock
def test_fetch_all_openneuro_respects_max_records() -> None:
    nodes = [dict(STUB_NODE, id=f"ds{i:06d}") for i in range(5)]
    respx.post(OPENNEURO_GQL).mock(
        return_value=httpx.Response(200, json=_gql_response(nodes, False, None))
    )
    records = fetch_all_openneuro(page_size=100, max_records=3)
    assert len(records) == 3


@respx.mock
def test_fetch_all_openneuro_skips_null_edge_node() -> None:
    payload = _gql_response([STUB_NODE], False, None)
    payload["data"]["datasets"]["edges"].insert(0, None)
    payload["data"]["datasets"]["edges"].insert(1, {"cursor": "broken", "node": None})
    payload["errors"] = [{"message": "Not Found"}]
    respx.post(OPENNEURO_GQL).mock(return_value=httpx.Response(200, json=payload))

    records = fetch_all_openneuro(page_size=100)

    assert len(records) == 1
    assert records[0]["source_id"] == "ds000001"


def test_openneuro_registry_adapter(monkeypatch) -> None:
    import neural_search.ingestion.openneuro as openneuro
    from neural_search.ingestion.registry import run_adapter

    monkeypatch.setattr(
        openneuro,
        "fetch_all_openneuro",
        lambda max_records=None: [{"source": "openneuro", "source_id": "ds000001"}],
    )

    records = run_adapter("openneuro", limit=1)

    assert records == [{"source": "openneuro", "source_id": "ds000001"}]
