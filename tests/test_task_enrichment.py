"""Tests for NLP task enrichment logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from enrich_task_coverage import CONFIDENCE, _build_alias_map, _match_tasks


class TestTaskEnrichmentLogic:
    def setup_method(self) -> None:
        self.alias_map = _build_alias_map()

    def test_alias_map_has_entries(self) -> None:
        assert len(self.alias_map) > 100

    def test_decision_making_matched(self) -> None:
        matches = _match_tasks("Mice performed a decision-making task", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "decision_making" in task_ids

    def test_visual_stimulation_matched(self) -> None:
        matches = _match_tasks("Natural image presentation and grating stimulation", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "visual_stimulation" in task_ids

    def test_sleep_matched(self) -> None:
        matches = _match_tasks("Recordings during NREM and REM sleep stages", self.alias_map)
        task_ids = [t for t, _ in matches]
        # any task covering sleep content is acceptable
        assert any("sleep" in t or "spontaneous" in t for t in task_ids), f"Got: {task_ids}"

    def test_passive_viewing_matched(self) -> None:
        matches = _match_tasks("Subjects engaged in passive viewing of natural scenes", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "passive_viewing" in task_ids

    def test_no_match_on_empty_text(self) -> None:
        assert _match_tasks("", self.alias_map) == []

    def test_no_match_on_unrelated_text(self) -> None:
        matches = _match_tasks("xyz abc 123 foobar baz", self.alias_map)
        assert len(matches) == 0

    def test_confidence_is_correct(self) -> None:
        matches = _match_tasks("passive viewing task", self.alias_map)
        assert len(matches) > 0
        for _, conf in matches:
            assert conf == CONFIDENCE

    def test_change_detection_matched(self) -> None:
        matches = _match_tasks("Oddball paradigm with mismatch negativity", self.alias_map)
        task_ids = [t for t, _ in matches]
        assert "change_detection" in task_ids
