"""Tests for finding_extractor.py (TDD: written before implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_search.literature.finding_extractor import (
    FindingRecord,
    _ensure_list,
    _repair_json,
    build_prompt,
    extract_batch,
    extract_batch_ollama,
    extract_findings_from_corpus,
    load_config,
    parse_findings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 512,
    "temperature": 0.0,
    "system_prompt": "You are a neuroscience assistant.",
    "user_template": "Title: {title}\nAbstract: {abstract}",
}

SAMPLE_PAPERS = [
    {
        "paper_id": "W2741809807",
        "paper_doi": "10.1038/nature12373",
        "title": "Hippocampal theta oscillations in spatial navigation",
        "abstract": (
            "We show that theta oscillations in the hippocampus increase "
            "during active spatial navigation in mice."
        ),
    },
    {
        "paper_id": "W1234567890",
        "paper_doi": None,
        "title": "No abstract paper",
        "abstract": "",
    },
]

VALID_FINDING_JSON = json.dumps([
    {
        "finding_text": "Theta oscillations increase during spatial navigation.",
        "result_direction": "increase",
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "modalities": ["electrophysiology"],
        "tasks": ["spatial navigation"],
        "cell_types": [],
        "molecules": [],
        "confidence": 0.9,
    }
])


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "model: claude-haiku-4-5-20251001\nmax_tokens: 512\ntemperature: 0.0\n"
        )
        result = load_config(cfg_file)
        assert result["model"] == "claude-haiku-4-5-20251001"
        assert result["max_tokens"] == 512

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_config(tmp_path / "does_not_exist.yaml")
        assert result == {}


# ---------------------------------------------------------------------------
# TestBuildPrompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_title_and_abstract(self) -> None:
        sys_prompt, user_msg = build_prompt(
            title="My Title",
            abstract="My abstract text.",
            config=SAMPLE_CONFIG,
        )
        assert "My Title" in user_msg
        assert "My abstract text." in user_msg

    def test_uses_config_system_prompt(self) -> None:
        sys_prompt, _ = build_prompt("T", "A", SAMPLE_CONFIG)
        assert sys_prompt == "You are a neuroscience assistant."

    def test_formats_user_template(self) -> None:
        _, user_msg = build_prompt(
            title="Hippocampal Theta",
            abstract="Abstract here.",
            config=SAMPLE_CONFIG,
        )
        assert user_msg == "Title: Hippocampal Theta\nAbstract: Abstract here."


# ---------------------------------------------------------------------------
# TestParseFindings
# ---------------------------------------------------------------------------

class TestParseFindings:
    def test_valid_json_array(self) -> None:
        findings = parse_findings("W2741809807", "10.1038/x", VALID_FINDING_JSON, "claude-haiku")
        assert len(findings) == 1
        f = findings[0]
        assert isinstance(f, FindingRecord)
        assert f.paper_id == "W2741809807"
        assert f.finding_text == "Theta oscillations increase during spatial navigation."
        assert f.result_direction == "increase"
        assert "hippocampus" in f.regions

    def test_empty_array(self) -> None:
        findings = parse_findings("W001", None, "[]", "claude-haiku")
        assert findings == []

    def test_malformed_json_returns_empty(self) -> None:
        findings = parse_findings("W001", None, "not valid json at all", "claude-haiku")
        assert findings == []

    def test_malformed_json_no_exception_raised(self) -> None:
        # Must not raise even on garbage
        try:
            parse_findings("W001", None, "{broken: true", "model")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"parse_findings raised unexpectedly: {exc}")

    def test_finding_ids_are_unique(self) -> None:
        two_findings = json.dumps([
            {
                "finding_text": "Finding one.",
                "result_direction": "increase",
                "regions": [],
                "species": [],
                "modalities": [],
                "tasks": [],
                "cell_types": [],
                "molecules": [],
                "confidence": 0.9,
            },
            {
                "finding_text": "Finding two.",
                "result_direction": "decrease",
                "regions": [],
                "species": [],
                "modalities": [],
                "tasks": [],
                "cell_types": [],
                "molecules": [],
                "confidence": 0.8,
            },
        ])
        findings = parse_findings("W999", None, two_findings, "haiku")
        ids = [f.finding_id for f in findings]
        assert ids == ["W999:f0", "W999:f1"]
        assert len(set(ids)) == 2

    def test_confidence_clamped_to_01(self) -> None:
        over = json.dumps([{
            "finding_text": "Over confident.",
            "result_direction": "other",
            "regions": [],
            "species": [],
            "modalities": [],
            "tasks": [],
            "cell_types": [],
            "molecules": [],
            "confidence": 5.0,
        }])
        under = json.dumps([{
            "finding_text": "Under confident.",
            "result_direction": "other",
            "regions": [],
            "species": [],
            "modalities": [],
            "tasks": [],
            "cell_types": [],
            "molecules": [],
            "confidence": -0.5,
        }])
        f_over = parse_findings("W1", None, over, "m")[0]
        f_under = parse_findings("W2", None, under, "m")[0]
        assert f_over.confidence == 1.0
        assert f_under.confidence == 0.0

    def test_invalid_result_direction_set_to_other(self) -> None:
        bad = json.dumps([{
            "finding_text": "Something weird.",
            "result_direction": "exploded",
            "regions": [],
            "species": [],
            "modalities": [],
            "tasks": [],
            "cell_types": [],
            "molecules": [],
            "confidence": 0.8,
        }])
        findings = parse_findings("W3", None, bad, "m")
        assert findings[0].result_direction == "other"

    def test_extracted_at_is_set(self) -> None:
        findings = parse_findings("W4", None, VALID_FINDING_JSON, "m")
        assert findings[0].extracted_at  # non-empty string
        # Verify it looks like an ISO timestamp
        assert "T" in findings[0].extracted_at or "-" in findings[0].extracted_at

    def test_extraction_model_stored(self) -> None:
        findings = parse_findings("W5", None, VALID_FINDING_JSON, "claude-haiku-test")
        assert findings[0].extraction_model == "claude-haiku-test"

    def test_paper_doi_stored(self) -> None:
        findings = parse_findings("W6", "10.1234/test", VALID_FINDING_JSON, "m")
        assert findings[0].paper_doi == "10.1234/test"

    def test_none_doi_stored(self) -> None:
        findings = parse_findings("W7", None, VALID_FINDING_JSON, "m")
        assert findings[0].paper_doi is None

    def test_list_valued_optional_fields_do_not_crash(self) -> None:
        # LLM sometimes returns lists for scalar optional fields; must not raise TypeError
        list_fields = json.dumps([{
            "brain_region": "hippocampus",
            "result_direction": "increase",
            "finding_summary": "Test finding",
            "confidence": 0.8,
            "species": "rat",
            "timescale": ["acute", "chronic"],
            "claim_type": ["mechanistic", "correlational"],
            "evidence_strength": ["strong", "moderate"],
        }])
        findings = parse_findings("W99", None, list_fields, "m")
        assert len(findings) == 1
        # First element of each list should be used (if valid) or fall back to default
        assert findings[0].timescale in (None, "acute", "unknown")


# ---------------------------------------------------------------------------
# TestExtractBatch
# ---------------------------------------------------------------------------

class TestExtractBatch:
    def test_no_api_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        results = extract_batch(SAMPLE_PAPERS, SAMPLE_CONFIG, api_key=None)
        assert results == []

    def test_skips_papers_without_abstract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=VALID_FINDING_JSON)]

        mock_messages = MagicMock()
        mock_messages.create.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages = mock_messages

        papers_with_one_no_abstract = [
            {
                "paper_id": "W100",
                "paper_doi": None,
                "title": "Paper with abstract",
                "abstract": "Real abstract content here.",
            },
            {
                "paper_id": "W200",
                "paper_doi": None,
                "title": "Paper without abstract",
                "abstract": "",  # no abstract — should be skipped
            },
        ]

        with patch("neural_search.literature.finding_extractor.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            results = extract_batch(
                papers_with_one_no_abstract, SAMPLE_CONFIG, api_key="test-key"
            )

        # Only the paper WITH an abstract should produce findings
        assert mock_messages.create.call_count == 1
        paper_ids = {r.paper_id for r in results}
        assert "W200" not in paper_ids

    def test_api_error_skips_paper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        papers = [
            {
                "paper_id": "W300",
                "paper_doi": None,
                "title": "Good paper",
                "abstract": "Meaningful neuroscience abstract.",
            },
            {
                "paper_id": "W400",
                "paper_doi": None,
                "title": "Error paper",
                "abstract": "Another abstract that causes an API error.",
            },
        ]

        call_count = 0

        def mock_create(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.content = [MagicMock(text=VALID_FINDING_JSON)]
                return resp
            raise RuntimeError("Simulated API error")

        mock_messages = MagicMock()
        mock_messages.create.side_effect = mock_create

        mock_client = MagicMock()
        mock_client.messages = mock_messages

        with patch("neural_search.literature.finding_extractor.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            results = extract_batch(papers, SAMPLE_CONFIG, api_key="test-key")

        # W300 should yield findings; W400 should be skipped (no raise)
        paper_ids = {r.paper_id for r in results}
        assert "W300" in paper_ids
        assert "W400" not in paper_ids


# ---------------------------------------------------------------------------
# TestExtractFindingsFromCorpus
# ---------------------------------------------------------------------------

def _make_shard(tmp_path: Path, name: str, papers: list[dict]) -> Path:
    shard = tmp_path / name
    shard.write_text("\n".join(json.dumps(p) for p in papers))
    return shard


def _mock_extract_batch(
    papers: list[dict],
    config: dict,
    *,
    api_key: str | None = None,
) -> list[FindingRecord]:
    """Stub that returns one FindingRecord per paper (for testing corpus iteration)."""
    records = []
    for paper in papers:
        if not paper.get("abstract"):
            continue
        records.append(
            FindingRecord(
                paper_id=paper["paper_id"],
                paper_doi=paper.get("paper_doi"),
                finding_id=f"{paper['paper_id']}:f0",
                finding_text="Stub finding.",
                result_direction="other",
                regions=[],
                species=[],
                modalities=[],
                tasks=[],
                cell_types=[],
                molecules=[],
                confidence=0.9,
                extraction_model="stub",
                extracted_at="2026-01-01T00:00:00",
            )
        )
    return records


class TestExtractFindingsFromCorpus:
    def test_writes_findings_to_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shard = _make_shard(
            tmp_path,
            "shard_00.jsonl",
            [
                {
                    "paper_id": "W001",
                    "paper_doi": None,
                    "title": "T1",
                    "abstract": "Abstract one.",
                },
                {
                    "paper_id": "W002",
                    "paper_doi": None,
                    "title": "T2",
                    "abstract": "Abstract two.",
                },
            ],
        )
        out_path = tmp_path / "findings.jsonl"

        with patch(
            "neural_search.literature.finding_extractor.extract_batch",
            side_effect=_mock_extract_batch,
        ):
            extract_findings_from_corpus([shard], SAMPLE_CONFIG, out_path)

        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["paper_id"] == "W001"
        assert first["finding_id"] == "W001:f0"

    def test_max_papers_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = [
            {
                "paper_id": f"W{i:04d}",
                "paper_doi": None,
                "title": f"Title {i}",
                "abstract": f"Abstract {i}.",
            }
            for i in range(20)
        ]
        shard = _make_shard(tmp_path, "big_shard.jsonl", papers)
        out_path = tmp_path / "findings.jsonl"

        with patch(
            "neural_search.literature.finding_extractor.extract_batch",
            side_effect=_mock_extract_batch,
        ):
            count = extract_findings_from_corpus(
                [shard], SAMPLE_CONFIG, out_path, max_papers=5
            )

        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 5
        assert count == 5

    def test_returns_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = [
            {
                "paper_id": f"W{i:04d}",
                "paper_doi": None,
                "title": f"Title {i}",
                "abstract": f"Abstract {i}.",
            }
            for i in range(3)
        ]
        shard = _make_shard(tmp_path, "count_shard.jsonl", papers)
        out_path = tmp_path / "findings.jsonl"

        with patch(
            "neural_search.literature.finding_extractor.extract_batch",
            side_effect=_mock_extract_batch,
        ):
            count = extract_findings_from_corpus([shard], SAMPLE_CONFIG, out_path)

        assert count == 3

    def test_empty_shard_list_returns_zero(self, tmp_path: Path) -> None:
        out_path = tmp_path / "findings.jsonl"
        count = extract_findings_from_corpus([], SAMPLE_CONFIG, out_path)
        assert count == 0

    def test_checkpoint_skips_processed_papers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = [
            {"paper_id": "W010", "paper_doi": None, "title": "T1", "abstract": "A1."},
            {"paper_id": "W011", "paper_doi": None, "title": "T2", "abstract": "A2."},
        ]
        shard = _make_shard(tmp_path, "shard.jsonl", papers)
        checkpoint = tmp_path / "checkpoint.json"
        # Pre-populate checkpoint with W010 already processed
        checkpoint.write_text(json.dumps(["W010"]))
        out_path = tmp_path / "findings.jsonl"

        with patch(
            "neural_search.literature.finding_extractor.extract_batch",
            side_effect=_mock_extract_batch,
        ):
            count = extract_findings_from_corpus(
                [shard], SAMPLE_CONFIG, out_path, checkpoint_path=checkpoint
            )

        # Only W011 should be processed
        assert count == 1
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["paper_id"] == "W011"


# ---------------------------------------------------------------------------
# TestRepairJson
# ---------------------------------------------------------------------------


class TestRepairJson:
    def test_plain_json_unchanged(self) -> None:
        text = '[{"finding_text": "Test."}]'
        assert _repair_json(text) == text

    def test_strips_json_code_fence(self) -> None:
        wrapped = '```json\n[{"finding_text": "Test."}]\n```'
        result = _repair_json(wrapped)
        assert result == '[{"finding_text": "Test."}]'

    def test_strips_plain_code_fence(self) -> None:
        wrapped = '```\n[{"finding_text": "Test."}]\n```'
        result = _repair_json(wrapped)
        assert result == '[{"finding_text": "Test."}]'

    def test_extracts_array_from_prose(self) -> None:
        text = 'Here are the findings:\n[{"finding_text": "Test."}]\nDone.'
        result = _repair_json(text)
        assert result.startswith("[")
        assert result.endswith("]")

    def test_empty_array_returned_as_is(self) -> None:
        assert _repair_json("[]") == "[]"

    def test_fenced_empty_array(self) -> None:
        result = _repair_json("```json\n[]\n```")
        assert result == "[]"

    def test_parse_findings_handles_fenced_output(self) -> None:
        fenced = '```json\n' + VALID_FINDING_JSON + '\n```'
        findings = parse_findings("W_fence", None, fenced, "local-model")
        assert len(findings) == 1
        assert findings[0].paper_id == "W_fence"


# ---------------------------------------------------------------------------
# TestExtractBatchOllama
# ---------------------------------------------------------------------------


class TestExtractBatchOllama:
    def _make_ollama_response(self, content: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": content}}
        return mock_resp

    def test_returns_findings_on_valid_response(self) -> None:
        papers = [
            {
                "paper_id": "W_olla_001",
                "paper_doi": "10.1/test",
                "title": "Hippocampal Theta",
                "abstract": "Theta oscillations increase during spatial navigation.",
            }
        ]
        mock_resp = self._make_ollama_response(VALID_FINDING_JSON)

        with patch("requests.post", return_value=mock_resp):
            findings = extract_batch_ollama(
                papers, SAMPLE_CONFIG, model="qwen2.5:7b-instruct-q4_K_M"
            )

        assert len(findings) == 1
        assert findings[0].paper_id == "W_olla_001"
        assert findings[0].extraction_model == "qwen2.5:7b-instruct-q4_K_M"

    def test_skips_papers_without_abstract(self) -> None:
        papers = [
            {"paper_id": "W_olla_002", "paper_doi": None, "title": "T", "abstract": ""},
        ]
        with patch("requests.post") as mock_post:
            findings = extract_batch_ollama(
                papers, SAMPLE_CONFIG, model="qwen2.5:7b-instruct-q4_K_M"
            )
        mock_post.assert_not_called()
        assert findings == []

    def test_handles_fenced_json_response(self) -> None:
        fenced = "```json\n" + VALID_FINDING_JSON + "\n```"
        papers = [
            {
                "paper_id": "W_olla_003",
                "paper_doi": None,
                "title": "T",
                "abstract": "Some abstract.",
            }
        ]
        mock_resp = self._make_ollama_response(fenced)

        with patch("requests.post", return_value=mock_resp):
            findings = extract_batch_ollama(
                papers, SAMPLE_CONFIG, model="qwen2.5:7b-instruct-q4_K_M"
            )

        assert len(findings) == 1

    def test_api_error_skips_paper(self) -> None:
        papers = [
            {
                "paper_id": "W_olla_004",
                "paper_doi": None,
                "title": "T",
                "abstract": "Some abstract.",
            }
        ]
        with patch("requests.post", side_effect=OSError("connection refused")):
            findings = extract_batch_ollama(
                papers, SAMPLE_CONFIG, model="qwen2.5:7b-instruct-q4_K_M"
            )

        assert findings == []

    def test_species_string_coerced_to_list(self) -> None:
        """Regression: LLM returning "rat" must not become ['r','a','t']."""
        response = json.dumps([{
            "finding_text": "Test finding.",
            "result_direction": "increase",
            "regions": ["hippocampus"],
            "species": "rat",   # bare string, not a list
            "modalities": [], "tasks": [], "cell_types": [], "molecules": [],
            "confidence": 0.9,
        }])
        mock_resp = self._make_ollama_response(response)
        papers = [{"paper_id": "W_str", "paper_doi": None, "title": "T", "abstract": "A."}]
        with patch("requests.post", return_value=mock_resp):
            findings = extract_batch_ollama(papers, SAMPLE_CONFIG, model="m")
        assert findings[0].species == ["rat"]

    def test_uses_correct_ollama_endpoint(self) -> None:
        papers = [
            {
                "paper_id": "W_olla_005",
                "paper_doi": None,
                "title": "T",
                "abstract": "Some abstract.",
            }
        ]
        mock_resp = self._make_ollama_response("[]")

        with patch("requests.post", return_value=mock_resp) as mock_post:
            extract_batch_ollama(
                papers,
                SAMPLE_CONFIG,
                model="qwen2.5:7b-instruct-q4_K_M",
                base_url="http://localhost:11434",
            )

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# TestEnsureList — regression for bare-string species/regions from v2 LLM
# ---------------------------------------------------------------------------


class TestEnsureList:
    def test_string_becomes_single_item_list(self) -> None:
        assert _ensure_list("rat") == ["rat"]

    def test_list_passthrough(self) -> None:
        assert _ensure_list(["rat", "mouse"]) == ["rat", "mouse"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _ensure_list("") == []

    def test_none_returns_empty_list(self) -> None:
        assert _ensure_list(None) == []

    def test_list_with_none_items_filtered(self) -> None:
        assert _ensure_list(["rat", None, "mouse"]) == ["rat", "mouse"]

    def test_whitespace_only_string_returns_empty(self) -> None:
        assert _ensure_list("   ") == []

    def test_string_is_stripped(self) -> None:
        assert _ensure_list("  human  ") == ["human"]

    def test_parse_findings_species_as_string(self) -> None:
        """End-to-end: parse_findings must not character-split species strings."""
        response = json.dumps([{
            "finding_text": "Neurons in hippocampus fire during navigation.",
            "result_direction": "increase",
            "regions": ["hippocampus"],
            "species": "mouse",   # v2 LLM often returns this as a bare string
            "modalities": ["electrophysiology"],
            "tasks": [],
            "cell_types": [],
            "molecules": [],
            "confidence": 0.9,
        }])
        findings = parse_findings("W_regress", None, response, "qwen2.5")
        assert findings[0].species == ["mouse"]

    def test_parse_findings_regions_as_string(self) -> None:
        response = json.dumps([{
            "finding_text": "Test.",
            "result_direction": "increase",
            "regions": "hippocampus",  # bare string
            "species": ["rat"],
            "modalities": [],
            "tasks": [],
            "cell_types": [],
            "molecules": [],
            "confidence": 0.8,
        }])
        findings = parse_findings("W_regress2", None, response, "qwen2.5")
        assert findings[0].regions == ["hippocampus"]
