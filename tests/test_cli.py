import json

import pytest

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


def test_cli_doctor_outputs_environment_json(capsys):
    exit_code = cli.main(["doctor"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert payload["python_supported"] is True
    assert all(payload["core_dependencies"].values())
    assert payload["source_checkout"] is True
    assert payload["source_assets"]["apps/web/package.json"] is True
    assert payload["profile"]["profile"]["name"] == "demo"
    assert payload["profile"]["ready"] is True


def test_cli_profile_list_and_check(capsys):
    assert cli.main(["profile", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in listed["profiles"]} == {
        "demo",
        "researcher",
        "corpus-builder",
        "evaluator",
        "full-stack",
    }

    assert cli.main(["profile", "check", "demo"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["profile"]["name"] == "demo"
    assert checked["required_artifacts_ready"] is True
    assert "capabilities" in checked


def test_cli_profile_manifest_writes_file(tmp_path, capsys):
    output = tmp_path / "demo-manifest.json"

    exit_code = cli.main(
        ["profile", "manifest", "demo", "--output", str(output)]
    )

    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload["schema_version"] == 2
    assert file_payload["profile"] == "demo"
    assert file_payload["profile_ready"] is True
    assert file_payload["artifacts"]
    assert "artifact_lock" in file_payload


def test_cli_artifact_status_is_machine_readable(capsys):
    assert cli.main(["artifacts", "status", "demo_datasets"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"][0]["id"] == "demo_datasets"
    assert payload["artifacts"][0]["kind"] == "committed_fixture"
    assert payload["artifacts"][0]["exists"] is True
    assert "lineage_state" in payload["artifacts"][0]


def test_cli_artifact_release_index_is_portable(capsys):
    assert cli.main(["artifacts", "releases"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"bundles": []}


def test_cli_empty_artifact_lock_verifies(capsys, tmp_path):
    lock = tmp_path / "missing-lock.json"
    assert cli.main(["artifacts", "verify", "--lock", str(lock)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["artifacts"] == []


def test_cli_adoption_report_is_machine_readable(capsys):
    assert cli.main(
        [
            "adoption",
            "report",
            "--events",
            "data/eval/adoption_events_demo.jsonl",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions"] == 3
    assert payload["interpretation"]["gold_relevance_claim"] is False


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.startswith("neural-search ")
