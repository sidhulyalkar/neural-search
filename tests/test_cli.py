from neural_search import cli


def test_cli_ingest_dispatches_to_service(monkeypatch, capsys):
    calls = {}

    class Result:
        def to_dict(self):
            return {
                "source": "dandi",
                "query": "go no-go",
                "fetched": 1,
                "normalized": 1,
                "saved": 0,
                "skipped": 0,
                "raw_response_paths": [],
                "warnings": [],
                "dataset_ids": ["000001"],
                "paper_ids": [],
            }

    def fake_ingest_source(source, query, limit, *, save, force, database_url):
        calls.update(
            {
                "source": source,
                "query": query,
                "limit": limit,
                "save": save,
                "force": force,
                "database_url": database_url,
            }
        )
        return Result()

    monkeypatch.setattr(cli, "ingest_source", fake_ingest_source)

    exit_code = cli.main(
        [
            "ingest",
            "dandi",
            "--query",
            "go no-go",
            "--limit",
            "1",
            "--save",
            "--database-url",
            "sqlite:///tmp/demo.db",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "source": "dandi",
        "query": "go no-go",
        "limit": 1,
        "save": True,
        "force": False,
        "database_url": "sqlite:///tmp/demo.db",
    }
    assert '"dataset_ids": [' in capsys.readouterr().out


def test_cli_search_outputs_json(capsys):
    exit_code = cli.main(["search", "go/nogo calcium imaging", "--limit", "1"])

    assert exit_code == 0
    assert '"query": "go/nogo calcium imaging"' in capsys.readouterr().out
