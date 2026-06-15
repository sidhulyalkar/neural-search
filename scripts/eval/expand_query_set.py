"""Expand benchmark query set from 30 to 100 queries using DuckDB coverage data.

Generates queries covering:
- High-coverage region × modality combinations
- Underrepresented combinations (dark pairs — test recall on sparse coverage)
- Cross-species comparison queries
- Analysis-goal queries (decoding, connectivity, prediction)
- Known-item lookup queries (specific landmark datasets)
- Adversarial / hard-negative queries

Usage
-----
    python scripts/eval/expand_query_set.py
    python scripts/eval/expand_query_set.py --out data/eval/benchmark_queries_v2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "coverage" / "ledger.duckdb"
BASE_QUERIES = ROOT / "data" / "eval" / "benchmark_queries.yaml"
OUT_PATH = ROOT / "data" / "eval" / "benchmark_queries_v2.yaml"


# ── Template-driven query generation ─────────────────────────────────────────

_REGION_MODALITY = [
    # high-coverage region × modality pairs that exist in the corpus
    ("visual_cortex", "calcium_imaging",
     "Find calcium imaging datasets in visual cortex with stimulus responses.",
     {"expected_regions_any": ["visual_cortex", "v1"], "expected_modalities_any": ["calcium_imaging", "two_photon"]}),
    ("hippocampus", "neuropixels",
     "Find Neuropixels recordings in hippocampus with sharp-wave ripples or theta oscillations.",
     {"expected_regions_any": ["hippocampus", "ca1", "ca3"], "expected_modalities_any": ["neuropixels", "extracellular_ephys"]}),
    ("prefrontal_cortex", "fmri",
     "Find fMRI datasets of prefrontal cortex during working memory or cognitive control.",
     {"expected_regions_any": ["prefrontal_cortex", "dlpfc", "mpfc"], "expected_modalities_any": ["fmri"], "expected_tasks": ["working_memory", "cognitive_control"]}),
    ("striatum", "calcium_imaging",
     "Find calcium imaging of striatum during reward learning or dopamine release.",
     {"expected_regions_any": ["striatum", "nucleus_accumbens"], "expected_modalities_any": ["calcium_imaging", "fiber_photometry"], "expected_behaviors": ["reward", "dopamine"]}),
    ("amygdala", "extracellular_ephys",
     "Find amygdala electrophysiology datasets with fear conditioning or anxiety behavior.",
     {"expected_regions_any": ["amygdala", "basolateral_amygdala"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"], "expected_tasks": ["fear_conditioning"]}),
    ("cerebellum", "calcium_imaging",
     "Find cerebellar calcium imaging datasets during motor learning or adaptation.",
     {"expected_regions_any": ["cerebellum", "cerebellar_cortex"], "expected_modalities_any": ["calcium_imaging", "two_photon"], "expected_tasks": ["motor_learning"]}),
    ("thalamus", "extracellular_ephys",
     "Find thalamic recordings during sensory processing or sleep.",
     {"expected_regions_any": ["thalamus", "lgn", "lateral_geniculate_nucleus"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("brainstem", "extracellular_ephys",
     "Find brainstem electrophysiology datasets with locus coeruleus or dopaminergic recordings.",
     {"expected_regions_any": ["brainstem", "locus_coeruleus", "vta"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("somatosensory_cortex", "neuropixels",
     "Find somatosensory cortex Neuropixels recordings with whisker or tactile stimulation.",
     {"expected_regions_any": ["somatosensory_cortex", "barrel_cortex"], "expected_modalities_any": ["neuropixels", "extracellular_ephys"]}),
    ("entorhinal_cortex", "extracellular_ephys",
     "Find entorhinal cortex recordings with grid cells or head direction cells.",
     {"expected_regions_any": ["entorhinal_cortex", "medial_entorhinal_cortex"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("motor_cortex", "calcium_imaging",
     "Find motor cortex calcium imaging datasets during movement or motor planning.",
     {"expected_regions_any": ["motor_cortex", "primary_motor_cortex", "m1"], "expected_modalities_any": ["calcium_imaging", "two_photon"], "expected_tasks": ["reaching", "motor_learning"]}),
    ("parietal_cortex", "fmri",
     "Find parietal cortex fMRI datasets during visuospatial attention or navigation.",
     {"expected_regions_any": ["parietal_cortex", "posterior_parietal_cortex"], "expected_modalities_any": ["fmri"], "expected_tasks": ["spatial_attention", "spatial_navigation"]}),
]

_CROSS_SPECIES = [
    ("Find mouse calcium imaging datasets during spatial navigation in open field or linear track.",
     {"expected_species": ["mouse", "mus_musculus"], "expected_modalities_any": ["calcium_imaging", "two_photon"], "expected_tasks": ["spatial_navigation", "open_field"]}),
    ("Find rat hippocampus electrophysiology during sleep with replay events.",
     {"expected_species": ["rat", "rattus_norvegicus"], "expected_regions_any": ["hippocampus", "ca1"], "expected_modalities_any": ["extracellular_ephys"], "expected_tasks": ["sleep_wake"]}),
    ("Find macaque prefrontal cortex recordings during working memory tasks.",
     {"expected_species": ["macaque", "macaca_mulatta"], "expected_regions_any": ["prefrontal_cortex", "dlpfc"], "expected_tasks": ["working_memory", "delayed_match_to_sample"]}),
    ("Find human EEG datasets during visual perception or P300 responses.",
     {"expected_species": ["human", "homo_sapiens"], "expected_modalities_any": ["eeg"], "expected_tasks": ["visual_perception", "p300"]}),
    ("Find zebrafish whole-brain calcium imaging datasets during sensory stimulation.",
     {"expected_species": ["zebrafish", "danio_rerio"], "expected_modalities_any": ["calcium_imaging", "light_sheet"]}),
    ("Find primate visual cortex recordings comparing V1 and V4 responses.",
     {"expected_species": ["macaque", "rhesus_macaque"], "expected_regions_any": ["visual_cortex", "v1", "v4"], "expected_modalities_any": ["extracellular_ephys"]}),
]

_ANALYSIS_GOALS = [
    ("Find datasets with paired neural recordings and behavior suitable for population dynamics analysis.",
     {"expected_analysis_any": ["population_dynamics", "dimensionality_reduction", "latent_variable"], "expected_behaviors": ["choice", "movement"]}),
    ("Find datasets suitable for fitting drift-diffusion models of decision making.",
     {"expected_tasks": ["decision_making", "two_alternative_forced_choice"], "expected_behaviors": ["reaction_time", "choice", "confidence"], "expected_analysis_any": ["drift_diffusion", "evidence_accumulation"]}),
    ("Find large-scale connectomics datasets for synaptic connectivity analysis.",
     {"expected_modalities_any": ["connectomics", "electron_microscopy"], "expected_analysis_any": ["connectome", "synaptic_connectivity"]}),
    ("Find datasets with single-trial neural activity for cross-session decoding.",
     {"expected_analysis_any": ["cross_session_decoding", "neural_manifold", "trial_averaging"]}),
    ("Find optogenetics datasets with causal manipulation of specific cell types.",
     {"expected_modalities_any": ["optogenetics"], "expected_analysis_any": ["causal_manipulation", "circuit_dissection"]}),
    ("Find datasets with multi-region simultaneous recordings for functional connectivity.",
     {"expected_analysis_any": ["functional_connectivity", "coherence", "cross_correlation"]}),
    ("Find resting-state fMRI datasets for default mode network analysis.",
     {"expected_modalities_any": ["fmri"], "expected_tasks": ["resting_state"], "expected_analysis_any": ["default_mode_network", "resting_state_fmri", "ica"]}),
    ("Find datasets with reward and punishment outcomes for reinforcement learning modeling.",
     {"expected_behaviors": ["reward", "punishment", "outcome"], "expected_analysis_any": ["reinforcement_learning", "value_estimation", "q_learning"]}),
    ("Find neural data with concurrent behavioral video for pose estimation integration.",
     {"expected_modalities_any": ["behavior_video", "pose_tracking"], "expected_analysis_any": ["pose_estimation", "behavior_classification"]}),
    ("Find EEG datasets with independent component analysis ready preprocessing.",
     {"expected_modalities_any": ["eeg"], "expected_analysis_any": ["ica", "artifact_removal", "source_separation"]}),
]

_KNOWN_ITEM = [
    ("Find the Steinmetz 2019 Neuropixels dataset across multiple brain regions during visual decision making.",
     {"expected_modalities_any": ["neuropixels"], "expected_tasks": ["visual_decision_making"], "expected_regions_any": ["visual_cortex", "striatum", "hippocampus"], "notes": "Known-item: Steinmetz 2019 (DANDI:000026)"}),
    ("Find the Allen Brain Observatory calcium imaging dataset with visual stimuli.",
     {"expected_sources": ["allen"], "expected_modalities_any": ["calcium_imaging"], "expected_regions_any": ["visual_cortex", "v1"], "notes": "Known-item: Allen Brain Observatory"}),
    ("Find IBL repeated site Neuropixels datasets with trial-by-trial decision data.",
     {"expected_sources": ["ibl"], "expected_modalities_any": ["neuropixels"], "expected_tasks": ["decision_making"], "notes": "Known-item: IBL repeated site"}),
    ("Find the Human Connectome Project resting-state fMRI dataset.",
     {"expected_modalities_any": ["fmri"], "expected_tasks": ["resting_state"], "expected_species": ["human"], "notes": "Known-item: HCP resting state"}),
    ("Find Buzsaki lab hippocampus recording datasets with sharp-wave ripples.",
     {"expected_sources": ["buzsaki"], "expected_regions_any": ["hippocampus", "ca1"], "expected_modalities_any": ["extracellular_ephys"], "notes": "Known-item: Buzsaki lab"}),
    ("Find NHP reaching datasets from the BrainGate consortium.",
     {"expected_tasks": ["reaching", "bci_control"], "expected_species": ["macaque", "human"], "expected_modalities_any": ["extracellular_ephys", "ecog"], "notes": "Known-item: BrainGate"}),
    ("Find mouse visual cortex datasets with running speed correlations.",
     {"expected_regions_any": ["visual_cortex", "v1"], "expected_behaviors": ["running_speed", "locomotion"], "expected_modalities_any": ["calcium_imaging", "neuropixels"]}),
]

_ADVERSARIAL = [
    ("Find datasets with hippocampal recordings but NOT in sleeping or resting animals.",
     {"expected_regions_any": ["hippocampus", "ca1"], "hard_negative_tasks": ["sleep_wake", "resting_state"], "expected_tasks": ["spatial_navigation", "decision_making"]}),
    ("Find fMRI datasets NOT from NeuroVault whole-brain contrast maps.",
     {"expected_modalities_any": ["fmri"], "hard_negative_sources": ["neurovault"], "expected_analysis_any": ["roi_analysis", "task_fmri"]}),
    ("Find calcium imaging datasets in cortex but NOT using widefield imaging.",
     {"expected_modalities_any": ["calcium_imaging", "two_photon"], "hard_negative_modalities": ["widefield_imaging", "mesoscope"], "expected_regions_any": ["visual_cortex", "motor_cortex"]}),
    ("Find Neuropixels datasets from rats NOT from mice.",
     {"expected_species": ["rat", "rattus_norvegicus"], "hard_negative_species": ["mouse", "mus_musculus"], "expected_modalities_any": ["neuropixels", "extracellular_ephys"]}),
    ("Find motor cortex datasets with clear trial structure but NOT BCI or prosthetics.",
     {"expected_regions_any": ["motor_cortex", "m1", "primary_motor_cortex"], "hard_negative_tasks": ["bci_control", "prosthetics"], "expected_tasks": ["reaching", "motor_learning"]}),
    ("Find decision-making datasets with dopamine signals but NOT using fMRI.",
     {"expected_behaviors": ["reward", "dopamine"], "hard_negative_modalities": ["fmri"], "expected_modalities_any": ["fiber_photometry", "calcium_imaging", "extracellular_ephys"]}),
    ("Find primate electrophysiology datasets NOT from Allen Institute.",
     {"expected_species": ["macaque", "non_human_primate"], "hard_negative_sources": ["allen"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("Find EEG cognitive datasets NOT involving motor imagery.",
     {"expected_modalities_any": ["eeg"], "hard_negative_tasks": ["motor_imagery"], "expected_tasks": ["working_memory", "attention", "cognitive_control"]}),
]

_DATA_QUALITY = [
    ("Find NWB datasets with raw electrophysiology voltage traces and trial annotations.",
     {"expected_data_standards": ["nwb"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("Find BIDS-compliant fMRI datasets with event files and behavioral data.",
     {"expected_data_standards": ["bids"], "expected_modalities_any": ["fmri"], "expected_behaviors": ["choice", "response"]}),
    ("Find datasets with both raw neural data and preprocessed spike trains.",
     {"expected_analysis_any": ["spike_sorting", "unit_activity"], "expected_modalities_any": ["extracellular_ephys", "neuropixels"]}),
    ("Find open-access single-unit recordings with behavioral annotations.",
     {"expected_analysis_any": ["unit_activity", "spike_sorting"], "expected_behaviors": ["choice", "reward", "movement"]}),
    ("Find datasets with electrode coordinates registered to a standard atlas.",
     {"expected_analysis_any": ["atlas_registration", "electrode_localization"]}),
]


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    return data.get("benchmark_queries", [])


def generate_queries(existing: list[dict]) -> list[dict]:
    next_id = max(int(q["id"].lstrip("q")) for q in existing) + 1 if existing else 1
    queries = list(existing)

    def add(text: str, fields: dict) -> None:
        nonlocal next_id
        entry: dict = {"id": f"q{next_id:03d}", "query": text}
        entry.update(fields)
        queries.append(entry)
        next_id += 1

    # Region × modality queries (q031-q042)
    for _region, _modality, text, fields in _REGION_MODALITY:
        add(text, fields)

    # Cross-species queries (q043-q048)
    for text, fields in _CROSS_SPECIES:
        add(text, fields)

    # Analysis goal queries (q049-q058)
    for text, fields in _ANALYSIS_GOALS:
        add(text, fields)

    # Known-item queries (q059-q065)
    for text, fields in _KNOWN_ITEM:
        add(text, fields)

    # Adversarial queries (q066-q073)
    for text, fields in _ADVERSARIAL:
        add(text, fields)

    # Data quality queries (q074-q078)
    for text, fields in _DATA_QUALITY:
        add(text, fields)

    # Fill to 100 with additional task/behavior combinations
    _FILL = [
        ("Find auditory cortex recordings with frequency tuning curves.",
         {"expected_regions_any": ["auditory_cortex", "a1"], "expected_tasks": ["auditory_processing", "frequency_discrimination"]}),
        ("Find prefrontal-hippocampal co-recording datasets during memory consolidation.",
         {"expected_regions_any": ["prefrontal_cortex", "hippocampus"], "expected_tasks": ["working_memory", "memory_consolidation"]}),
        ("Find datasets with chronic multi-day recordings from the same electrodes.",
         {"expected_analysis_any": ["chronic_recording", "longitudinal"]}),
        ("Find large-scale fMRI datasets with whole-brain coverage and >100 subjects.",
         {"expected_modalities_any": ["fmri"], "expected_analysis_any": ["whole_brain", "group_analysis"]}),
        ("Find mouse barrel cortex recordings during active whisking.",
         {"expected_regions_any": ["barrel_cortex", "somatosensory_cortex"], "expected_modalities_any": ["extracellular_ephys", "calcium_imaging"], "expected_behaviors": ["whisking", "tactile"]}),
        ("Find VTA dopamine neuron recordings during reward prediction errors.",
         {"expected_regions_any": ["vta", "ventral_tegmental_area"], "expected_behaviors": ["reward", "reward_prediction_error", "dopamine"]}),
        ("Find anterior cingulate cortex datasets with conflict monitoring.",
         {"expected_regions_any": ["acc", "anterior_cingulate_cortex"], "expected_tasks": ["conflict_monitoring", "stroop", "flanker"]}),
        ("Find human MEG datasets during language comprehension or semantic processing.",
         {"expected_species": ["human"], "expected_modalities_any": ["meg"], "expected_tasks": ["language", "semantic_processing"]}),
        ("Find spinal cord recording datasets during locomotion.",
         {"expected_regions_any": ["spinal_cord"], "expected_tasks": ["locomotion", "walking"], "expected_modalities_any": ["extracellular_ephys"]}),
        ("Find datasets combining electrophysiology and optogenetics for circuit dissection.",
         {"expected_modalities_any": ["extracellular_ephys", "optogenetics"], "expected_analysis_any": ["circuit_dissection", "causal_manipulation"]}),
        ("Find multi-photon imaging datasets with subcellular resolution dendrite recordings.",
         {"expected_modalities_any": ["two_photon", "calcium_imaging"], "expected_analysis_any": ["dendritic_computation", "subcellular"]}),
        ("Find primate area MT/V5 motion processing datasets.",
         {"expected_regions_any": ["mst", "visual_cortex"], "expected_species": ["macaque"], "expected_tasks": ["motion_discrimination"]}),
        ("Find datasets with simultaneous LFP and single-unit activity during theta rhythms.",
         {"expected_modalities_any": ["lfp", "extracellular_ephys"], "expected_regions_any": ["hippocampus", "prefrontal_cortex"]}),
        ("Find social neuroscience datasets with two-player economic games.",
         {"expected_tasks": ["social_interaction", "ultimatum_game", "prisoner_dilemma"], "expected_modalities_any": ["fmri", "eeg"]}),
        ("Find pain or nociception datasets with somatosensory recordings.",
         {"expected_tasks": ["pain", "nociception"], "expected_regions_any": ["somatosensory_cortex", "thalamus"]}),
        ("Find circadian rhythm or homeostatic sleep datasets with long-duration recordings.",
         {"expected_tasks": ["sleep_wake", "circadian"], "expected_modalities_any": ["eeg", "extracellular_ephys"]}),
        ("Find olfactory bulb or piriform cortex datasets during odor discrimination.",
         {"expected_regions_any": ["olfactory_bulb", "piriform_cortex"], "expected_tasks": ["odor_discrimination"]}),
        ("Find cerebellum purkinje cell recordings during eyeblink conditioning.",
         {"expected_regions_any": ["cerebellum", "cerebellar_cortex"], "expected_tasks": ["eyeblink_conditioning"], "expected_modalities_any": ["extracellular_ephys"]}),
        ("Find retinal ganglion cell recordings with natural image or movie stimuli.",
         {"expected_regions_any": ["retina"], "expected_tasks": ["visual_stimulation", "naturalistic_vision"]}),
        ("Find attention datasets with frontal-parietal network recordings.",
         {"expected_tasks": ["attention", "spatial_attention"], "expected_regions_any": ["prefrontal_cortex", "parietal_cortex"]}),
        ("Find habit vs goal-directed decision-making datasets with dorsomedial and dorsolateral striatum.",
         {"expected_regions_any": ["striatum", "dorsomedial_striatum", "dorsolateral_striatum"], "expected_tasks": ["habit_learning", "goal_directed"]}),
        ("Find mouse visual cortex datasets with orientation tuning under anesthesia.",
         {"expected_species": ["mouse"], "expected_regions_any": ["visual_cortex", "v1"], "expected_tasks": ["visual_stimulation", "orientation_tuning"]}),
    ]

    for text, fields in _FILL:
        if len(queries) >= 100:
            break
        add(text, fields)

    return queries[:100]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_QUERIES)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    existing = load_existing(args.base)
    print(f"Loaded {len(existing)} existing queries from {args.base}")

    queries = generate_queries(existing)
    print(f"Generated {len(queries)} total queries")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.dump({"benchmark_queries": queries}, default_flow_style=False, allow_unicode=True))
    print(f"Written to {args.out}")

    intent_counts: dict[str, int] = {}
    for q in queries:
        if "expected_species" in q:
            intent_counts["species-specific"] = intent_counts.get("species-specific", 0) + 1
        if "hard_negative_modalities" in q or "hard_negative_tasks" in q or "hard_negative_sources" in q:
            intent_counts["adversarial"] = intent_counts.get("adversarial", 0) + 1
        if "notes" in q and "Known-item" in str(q.get("notes", "")):
            intent_counts["known-item"] = intent_counts.get("known-item", 0) + 1
    print("\nQuery type breakdown:")
    for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
