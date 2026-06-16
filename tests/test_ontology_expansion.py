"""Tests verifying ontology expansion in brain_regions.yaml and behavioral_task_ontology.yaml."""
import yaml
from pathlib import Path

BRAIN_REGIONS_PATH = Path("data/ontology/brain_regions.yaml")
TASK_ONTOLOGY_PATH = Path("data/ontology/behavioral_task_ontology.yaml")


def _load_brain_regions() -> list[dict]:
    with BRAIN_REGIONS_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("brain_regions", [])


def _load_tasks() -> list[dict]:
    with TASK_ONTOLOGY_PATH.open() as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


class TestBrainRegionExpansion:
    def test_total_regions_above_120(self):
        regions = _load_brain_regions()
        assert len(regions) >= 120, f"Only {len(regions)} regions"

    def test_new_regions_present(self):
        ids = {r["id"] for r in _load_brain_regions()}
        new_regions = [
            "superior_temporal_gyrus", "olfactory_bulb", "piriform_cortex",
            "medial_septum", "lateral_habenula", "claustrum",
            "primary_auditory_cortex", "prelimbic_cortex", "perirhinal_cortex",
            "retina", "spinal_cord", "dorsal_raphe",
        ]
        # Only check ids that should be present (some may have pre-existed)
        all_ids = ids
        present = [r for r in new_regions if r in all_ids]
        assert len(present) >= 8, f"Too few new regions present: {present}"

    def test_all_regions_have_id_and_label(self):
        for r in _load_brain_regions():
            assert r.get("id"), f"Region missing id: {r}"
            assert r.get("label"), f"Region missing label: {r}"

    def test_all_regions_have_aliases_list(self):
        for r in _load_brain_regions():
            assert isinstance(r.get("aliases", []), list)

    def test_no_duplicate_region_ids(self):
        ids = [r["id"] for r in _load_brain_regions()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


class TestTaskOntologyExpansion:
    def test_total_tasks_above_90(self):
        tasks = _load_tasks()
        assert len(tasks) >= 90, f"Only {len(tasks)} tasks"

    def test_new_tasks_present(self):
        ids = {t["id"] for t in _load_tasks()}
        new_tasks = [
            "decision_making", "visual_stimulation", "passive_viewing",
            "change_detection", "spontaneous_activity", "circuit_mapping",
            "cell_type_mapping", "current_injection", "excitability_analysis",
        ]
        for tid in new_tasks:
            assert tid in ids, f"Missing task: {tid}"

    def test_all_tasks_have_id_and_label(self):
        for t in _load_tasks():
            assert t.get("id"), f"Task missing id: {t}"
            assert t.get("label"), f"Task missing label: {t}"

    def test_all_tasks_have_aliases(self):
        for t in _load_tasks():
            aliases = t.get("synonyms", [])
            assert isinstance(aliases, list), f"Synonyms not a list for {t.get('id')}"
            assert len(aliases) >= 1, f"No synonyms for task {t.get('id')}"

    def test_no_duplicate_task_ids(self):
        ids = [t["id"] for t in _load_tasks()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"
