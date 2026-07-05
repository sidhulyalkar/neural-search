"""Tests for the agent context-brief generator."""

from __future__ import annotations

import json

import neural_search.agents.context_brief as context_brief
from neural_search.agents.ledger import LedgerEntry, append_ledger_entry


def _write_manifest(path, **overrides):
    payload = {
        "generated_at": "2026-07-04T00:00:00+00:00",
        "corpus": {"row_count": 100},
        "knowledge_graph": {"total_nodes": 10, "total_edges": 20},
        "qrels": {
            "gold": {"rows": 0},
            "silver": {"rows": 5},
            "canonical_llm_silver": {"rows": 50},
        },
        "paper_dataset_links": {"combined_datasets_with_real_link": 30, "total_rows": 100},
        "reanalysis_edges": {"dataset_reanalysis_bridge_dataset": 7},
        "ablation_benchmark": {
            "rungs": {
                "hybrid_graph": {"status": "ok", "metrics": {"ndcg@10": 0.85}},
                "bm25": {"status": "skipped", "metrics": {}},
            }
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_registry(path, playbook_dir):
    (playbook_dir / "exists.md").write_text("# stub", encoding="utf-8")
    registry = {
        "version": 1,
        "agents": [
            {"name": "agent-with-playbook", "playbook": "playbooks/exists.md", "trigger": "manual", "gate": "none"},
            {"name": "agent-without-playbook", "playbook": "playbooks/missing.md", "trigger": "schedule", "gate": "soft_review"},
        ],
    }
    path.write_text(json.dumps(registry), encoding="utf-8")


def test_build_context_brief_reads_manifest_and_registry(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    agents_dir = tmp_path / "artifacts_agents"
    playbook_dir = agents_dir / "playbooks"
    playbook_dir.mkdir(parents=True)
    registry_path = agents_dir / "registry.yaml"
    ledger_path = agents_dir / "ledger.jsonl"

    _write_manifest(manifest_path)
    _write_registry(registry_path, playbook_dir)

    monkeypatch.setattr(context_brief, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(context_brief, "AGENTS_DIR", agents_dir)
    monkeypatch.setattr(context_brief, "load_registry", lambda: json.loads(registry_path.read_text()))
    monkeypatch.setattr(
        context_brief,
        "last_run_for_agent",
        lambda name, ledger_path=ledger_path: _last_run(name, ledger_path),
    )

    append_ledger_entry(
        LedgerEntry(
            agent="agent-with-playbook",
            outcome="ok",
            started_at="2026-07-04T00:00:00+00:00",
            finished_at="2026-07-04T00:01:00+00:00",
            findings=["routine finding", "UNCLEAR whether X is reachable"],
        ),
        ledger_path=ledger_path,
    )

    brief = context_brief.build_context_brief()
    assert brief["corpus_rows"] == 100
    assert brief["graph_nodes"] == 10
    assert brief["ndcg_baseline"] == {"hybrid_graph": 0.85}
    assert brief["paper_link_combined_real_matches"] == 30

    agents_by_name = {row["name"]: row for row in brief["agents"]}
    assert agents_by_name["agent-with-playbook"]["has_playbook"] is True
    assert agents_by_name["agent-without-playbook"]["has_playbook"] is False
    assert agents_by_name["agent-with-playbook"]["last_run"]["outcome"] == "ok"
    assert agents_by_name["agent-without-playbook"]["last_run"] is None

    assert len(brief["open_questions"]) == 1
    assert "UNCLEAR" in brief["open_questions"][0]["finding"]

    markdown = context_brief.render_context_brief_markdown(brief)
    assert "agent-with-playbook" in markdown
    assert "**MISSING**" in markdown  # agent-without-playbook's row


def _last_run(name, ledger_path):
    from neural_search.agents.ledger import (
        last_run_for_agent as real_last_run_for_agent,
    )

    return real_last_run_for_agent(name, ledger_path=ledger_path)


def test_build_context_brief_handles_missing_manifest(tmp_path, monkeypatch):
    agents_dir = tmp_path / "artifacts_agents"
    agents_dir.mkdir()
    monkeypatch.setattr(context_brief, "MANIFEST_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(context_brief, "AGENTS_DIR", agents_dir)
    monkeypatch.setattr(context_brief, "load_registry", lambda: {"version": 1, "agents": []})

    brief = context_brief.build_context_brief()
    assert brief["manifest_available"] is False
    markdown = context_brief.render_context_brief_markdown(brief)
    assert "Manifest unavailable" in markdown
