"""Tests for NLP task enrichment logic."""
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK_ONTOLOGY = Path("data/ontology/behavioral_task_ontology.yaml")
CONFIDENCE = 0.60


def _build_alias_map(task_path: Path = TASK_ONTOLOGY) -> dict[str, str]:
    with task_path.open() as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, str] = {}
    for task in data.get("tasks", []):
        task_id = task["id"]
        candidates = [task.get("label", ""), task_id.replace("_", " ")]
        for key in ("aliases", "synonyms"):
            candidates.extend(task.get(key, []))
        for alias in candidates:
            if not alias:
                continue
            normalized = _normalize_text(str(alias)).strip()
            if len(normalized) >= 4:
                alias_map[normalized] = task_id
    return alias_map


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _match_tasks(text: str, alias_map: dict[str, str]) -> list[tuple[str, float]]:
    normalized = _normalize_text(text)
    found: dict[str, int] = {}
    for alias, task_id in alias_map.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            found[task_id] = max(found.get(task_id, 0), len(alias))
    return [(task_id, CONFIDENCE) for task_id in found]


class TestTaskEnrichmentLogic:
    def setup_method(self):
        self.alias_map = _build_alias_map()

    def test_alias_map_has_entries(self):
        assert len(self.alias_map) > 100

    def test_decision_making_matched(self):
        matches = _match_tasks("Mice performed a decision-making task", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "decision_making" in task_ids

    def test_visual_stimulation_matched(self):
        matches = _match_tasks("Natural image presentation and grating stimulation", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "visual_stimulation" in task_ids

    def test_sleep_matched(self):
        matches = _match_tasks("Recordings during NREM and REM sleep stages", self.alias_map)
        task_ids = [t for t, _ in matches]
        # sleep_recording or spontaneous_activity or similar — any task with sleep content
        assert any("sleep" in t or "spontaneous" in t for t in task_ids), f"Got: {task_ids}"

    def test_passive_viewing_matched(self):
        matches = _match_tasks("Subjects engaged in passive viewing of natural scenes", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "passive_viewing" in task_ids

    def test_no_match_on_empty_text(self):
        assert _match_tasks("", self.alias_map) == []

    def test_no_match_on_unrelated_text(self):
        matches = _match_tasks("xyz abc 123 foobar baz", self.alias_map)
        assert len(matches) == 0

    def test_confidence_is_correct(self):
        matches = _match_tasks("passive viewing task", self.alias_map)
        assert len(matches) > 0
        for _, conf in matches:
            assert conf == CONFIDENCE

    def test_change_detection_matched(self):
        matches = _match_tasks("Oddball paradigm with mismatch negativity", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "change_detection" in task_ids
