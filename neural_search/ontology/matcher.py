"""Fuzzy ontology matching for tasks, behaviors, modalities, and regions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from neural_search.ontology.loader import get_ontology
from neural_search.ontology.models import LabelMatch, Ontology

if TYPE_CHECKING:
    from neural_search.graph.schema import KnowledgeGraph

ALIASES: dict[str, list[str]] = {
    # Recording modalities
    "bci": ["bci", "brain computer interface", "brain-computer interface", "brain machine interface", "bmi"],
    "calcium_imaging": [
        "calcium imaging", "2 photon", "two photon", "2p", "gcamp",
        "optical imaging", "two-photon", "miniscope",
    ],
    "extracellular_ephys": [
        "extracellular electrophysiology", "ephys", "spikes",
        "electrophysiology", "single unit", "multi unit", "spike recordings",
        "extracellular recording", "neural recording", "tetrode", "silicon probe",
    ],
    "neuropixels": ["neuropixels", "neuropixel", "neuropixels probe"],
    "fiber_photometry": ["fiber photometry", "photometry", "fiber optic", "gcamp photometry", "gcamp fiber"],
    "eeg": ["eeg", "electroencephalography", "scalp eeg", "scalp recording", "scalp electrodes", "electroencephalogram"],
    "ecog": ["ecog", "electrocorticography", "cortical surface", "subdural electrode", "cortical recording", "subdural grid"],
    "ieeg": ["ieeg", "intracranial eeg", "depth electrodes", "stereo eeg", "seeg", "intracranial recording", "depth recording", "intracranial electrode"],
    "meg": ["meg", "magnetoencephalography"],
    "fmri": ["fmri", "functional mri", "bold", "functional magnetic resonance", "fmri bold", "functional imaging"],
    "behavior_video": ["behavior video", "video tracking", "video", "behavioral video"],
    "pose_tracking": ["pose tracking", "kinematics", "dlc", "deeplabcut", "sleap", "markerless tracking", "motion capture"],
    "utah_array": ["utah array", "utah electrode", "microelectrode array", "multielectrode array", "mea", "blackrock array"],
    "emg": ["emg", "electromyography", "muscle recording", "muscle activity"],
    "lfp": ["lfp", "local field potential", "field potential"],
    "pupil_tracking": ["pupil tracking", "pupillometry", "eye tracking", "pupil size", "pupil diameter"],
    "running_wheel": ["running wheel", "wheel encoder", "treadmill", "locomotion"],
    "polysomnography": ["polysomnography", "psg", "sleep recording", "sleep eeg"],
    "widefield_imaging": ["widefield imaging", "widefield", "mesoscale imaging", "wide field"],
    "force_sensor": ["force sensor", "force tracking", "grip force", "force transducer"],
    # Brain regions - core
    "motor_cortex": ["motor cortex", "m1", "primary motor cortex", "motor control", "primary motor"],
    "visual_cortex": ["visual cortex", "v1", "v2", "v4", "primary visual cortex", "visual areas", "striate cortex"],
    "somatosensory_cortex": ["somatosensory cortex", "s1", "primary somatosensory", "barrel cortex"],
    "parietal_cortex": ["parietal cortex", "ppc", "posterior parietal", "posterior parietal cortex"],
    "OFC": ["ofc", "orbitofrontal", "orbitofrontal cortex"],
    "mPFC": ["mpfc", "medial prefrontal", "medial prefrontal cortex", "prelimbic", "infralimbic"],
    "ACC": ["acc", "anterior cingulate", "anterior cingulate cortex", "cingulate cortex"],
    "hippocampus": ["hippocampus", "hpc", "ca1", "ca3", "dentate gyrus", "hipp", "hippocampal", "subiculum"],
    "striatum": ["striatum", "dorsal striatum", "ventral striatum", "caudate", "putamen", "nucleus accumbens", "nac", "basal ganglia"],
    "VTA": ["vta", "ventral tegmental area", "midbrain", "dopamine midbrain"],
    "SNc": ["snc", "substantia nigra", "substantia nigra pars compacta", "nigra"],
    "auditory_cortex": ["auditory cortex", "a1", "primary auditory cortex", "auditory areas", "temporal cortex"],
    "prefrontal_cortex": ["prefrontal cortex", "pfc", "frontal cortex", "frontal lobe", "dlpfc", "lateral prefrontal"],
    "temporal_lobe": ["temporal lobe", "temporal cortex", "mtl", "medial temporal lobe"],
    "amygdala": ["amygdala", "amygdalar", "basolateral amygdala", "bla", "central amygdala", "cea"],
    "entorhinal_cortex": ["entorhinal cortex", "ec", "medial entorhinal", "lateral entorhinal"],
    # Brain regions - clinical/speech
    "STG": ["stg", "superior temporal gyrus", "superior temporal", "wernicke"],
    "IFG": ["ifg", "inferior frontal gyrus", "inferior frontal", "broca"],
    "PMC": ["pmc", "premotor cortex", "premotor", "supplementary motor", "sma"],
    "insular_cortex": ["insular cortex", "insula", "insular"],
    "thalamus": ["thalamus", "thalamic", "lateral geniculate", "lgn", "pulvinar"],
    "cerebellum": ["cerebellum", "cerebellar", "purkinje"],
    "brainstem": ["brainstem", "brain stem", "pons", "medulla"],
    "spinal_cord": ["spinal cord", "spinal", "cervical spinal", "lumbar spinal"],
    # Species
    "mouse": ["mouse", "mice", "mus musculus", "c57bl6", "c57bl/6"],
    "rat": ["rat", "rats", "rattus", "sprague dawley", "long evans", "wistar"],
    "human": ["human", "humans", "patient", "patients", "participant", "participants", "subject", "subjects"],
    "macaque": ["macaque", "monkey", "rhesus", "macaca mulatta", "non-human primate", "nonhuman primate", "nhp"],
    "drosophila": ["drosophila", "fruit fly", "fly", "drosophila melanogaster"],
    "zebrafish": ["zebrafish", "danio rerio", "zebrafish larvae"],
    "marmoset": ["marmoset", "callithrix", "common marmoset"],
}


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[/_-]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _ngrams(tokens: list[str], min_size: int, max_size: int) -> Iterable[str]:
    max_size = min(max_size, len(tokens))
    for size in range(max_size, min_size - 1, -1):
        for start in range(0, len(tokens) - size + 1):
            yield " ".join(tokens[start : start + size])


def _fuzzy_confidence(normalized_text: str, phrase: str) -> tuple[float, str] | None:
    normalized_phrase = normalize_text(phrase)
    if len(normalized_phrase) < 4:
        return None
    tokens = normalized_text.split()
    phrase_size = len(normalized_phrase.split())
    best_score = 0.0
    best_evidence = ""
    for candidate in _ngrams(tokens, max(1, phrase_size - 1), phrase_size + 1):
        score = SequenceMatcher(None, candidate, normalized_phrase).ratio()
        if score > best_score:
            best_score = score
            best_evidence = candidate
    if best_score >= 0.84:
        return round(best_score * 0.82, 3), best_evidence
    return None


def _best_phrase_match(
    text: str,
    item_id: str,
    label: str,
    category: str,
    phrases: list[tuple[str, float, str]],
) -> LabelMatch | None:
    normalized = normalize_text(text)
    best: LabelMatch | None = None
    for phrase, confidence, match_type in phrases:
        if _contains_phrase(normalized, phrase):
            candidate: LabelMatch | None = LabelMatch(
                id=item_id,
                label=label,
                confidence=confidence,
                evidence=phrase,
                category=category,
                match_type=match_type,
            )
            best = _choose_better_match(best, candidate)
    if best is not None:
        return best
    if len(normalized.split()) > 32:
        return None

    for phrase, _, _ in phrases:
        fuzzy = _fuzzy_confidence(normalized, phrase)
        if fuzzy:
            fuzzy_confidence, evidence = fuzzy
            candidate = LabelMatch(
                id=item_id,
                label=label,
                confidence=fuzzy_confidence,
                evidence=evidence,
                category=category,
                match_type="fuzzy",
            )
            best = _choose_better_match(best, candidate)
    return best


def _choose_better_match(
    best: LabelMatch | None, candidate: LabelMatch
) -> LabelMatch:
    if best is None or candidate.confidence > best.confidence:
        return candidate
    if abs(candidate.confidence - best.confidence) <= 0.05 and len(
        candidate.evidence
    ) > len(best.evidence):
        return candidate
    return best


def _aliases_for(value: str) -> list[str]:
    aliases = ALIASES.get(value, [])
    normalized_value = normalize_text(value)
    values = {value, normalized_value, normalized_value.replace(" ", "_"), *aliases}
    return [item for item in values if item]


def match_tasks(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for task in ontology.tasks:
        phrases = [(task.label, 0.98, "label"), (task.id, 0.95, "id")]
        phrases.extend((synonym, 0.94, "synonym") for synonym in task.synonyms)
        match = _best_phrase_match(text, task.id, task.label, task.category, phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_behavior_labels(
    text: str, ontology: Ontology | None = None
) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for behavior in ontology.behavior_labels:
        phrases = [(behavior.label, 0.98, "label"), (behavior.id, 0.95, "id")]
        phrases.extend((synonym, 0.92, "synonym") for synonym in behavior.synonyms)
        match = _best_phrase_match(text, behavior.id, behavior.label, "behavior", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_modalities(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    modality_names = sorted({*ontology.modality_names, "bci"})
    for modality in modality_names:
        phrases = [(alias, 0.94 if alias != modality else 0.96, "modality") for alias in _aliases_for(modality)]
        match = _best_phrase_match(text, modality, modality, "modality", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_brain_regions(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for region in ontology.region_names:
        phrases = [(alias, 0.94 if alias != region else 0.96, "region") for alias in _aliases_for(region)]
        match = _best_phrase_match(text, region, region, "brain_region", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_affordances(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    """Match analysis affordances against query text."""
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    for affordance in ontology.analysis_affordances:
        # Build phrase list from affordance properties
        phrases: list[tuple[str, float, str]] = [
            (affordance.label, 0.98, "label"),
            (affordance.id, 0.95, "id"),
        ]
        # Add underscore-to-space variants
        phrases.append((affordance.id.replace("_", " "), 0.94, "id_variant"))
        # Add required signals as lower confidence matches
        for signal in affordance.required_signals:
            phrases.append((signal, 0.85, "required_signal"))
            phrases.append((signal.replace("_", " "), 0.84, "required_signal"))
        # Add typical outputs as hints
        for output in affordance.typical_outputs:
            phrases.append((output, 0.82, "typical_output"))
            phrases.append((output.replace("_", " "), 0.81, "typical_output"))

        match = _best_phrase_match(text, affordance.id, affordance.label, "analysis_affordance", phrases)
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def expand_query_terms(query: str) -> dict[str, list[str]]:
    ontology = get_ontology()
    tasks = match_tasks(query, ontology)
    behaviors = match_behavior_labels(query, ontology)
    affordances = match_affordances(query, ontology)
    terms = {normalize_text(query)}
    suggested_analyses: set[str] = set()

    for match in tasks:
        task = ontology.task_by_id.get(match.id)
        if not task:
            continue
        terms.update(normalize_text(value) for value in [task.label, task.id, *task.synonyms])
        terms.update(normalize_text(value) for value in task.common_events)
        suggested_analyses.update(task.suggested_analyses)

    for match in behaviors:
        behavior = ontology.behavior_by_id.get(match.id)
        if not behavior:
            continue
        terms.update(normalize_text(value) for value in [behavior.label, behavior.id, *behavior.synonyms])

    # Expand affordance terms
    for match in affordances:
        affordance = ontology.affordance_by_id.get(match.id)
        if not affordance:
            continue
        terms.update(normalize_text(value) for value in [affordance.label, affordance.id])
        terms.update(normalize_text(value) for value in affordance.required_signals)
        suggested_analyses.add(affordance.id)

    return {
        "terms": sorted(value for value in terms if value),
        "task_ids": sorted({match.id for match in tasks}),
        "behavior_ids": sorted({match.id for match in behaviors}),
        "affordance_ids": sorted({match.id for match in affordances}),
        "suggested_analyses": sorted(suggested_analyses),
    }


def expand_query_terms_with_graph(
    query: str,
    graph: KnowledgeGraph | None = None,
    max_hops: int = 1,
) -> dict[str, list[str]]:
    """Expand query terms using both ontology and graph relationships.

    This function enhances the standard ontology expansion by:
    - Finding behavioral events related to matched tasks via graph edges
    - Finding required modalities for matched affordances via graph edges
    - Including graph-derived synonyms and related concepts

    Args:
        query: The search query text
        graph: Optional knowledge graph for relationship traversal
        max_hops: Maximum edge traversal depth (default 1)

    Returns:
        Dictionary with expanded terms, IDs, and graph-derived expansions
    """
    # Start with standard ontology expansion
    base_expansion = expand_query_terms(query)
    terms = set(base_expansion["terms"])
    graph_expansions: dict[str, list[str]] = {
        "behavioral_events": [],
        "related_modalities": [],
        "related_regions": [],
    }

    if graph is None:
        return {
            **base_expansion,
            "graph_expansions": graph_expansions,
        }

    # For each matched task, find related behavioral events via graph
    for task_id in base_expansion["task_ids"]:
        task_node_id = f"node:task:{task_id}"
        if task_node_id in graph.nodes:
            # Find behavioral events linked to this task
            for edge in graph.edges.values():
                if edge.source_node_id == task_node_id and edge.edge_type == "task_has_behavioral_event":
                    target_node = graph.nodes.get(edge.target_node_id)
                    if target_node:
                        event_label = normalize_text(target_node.label)
                        terms.add(event_label)
                        graph_expansions["behavioral_events"].append(target_node.label)

    # For each matched affordance, find required modalities via graph
    for affordance_id in base_expansion["affordance_ids"]:
        affordance_node_id = f"node:analysis_affordance:{affordance_id}"
        if affordance_node_id in graph.nodes:
            for edge in graph.edges.values():
                if edge.source_node_id == affordance_node_id:
                    if edge.edge_type == "analysis_requires_modality":
                        target_node = graph.nodes.get(edge.target_node_id)
                        if target_node:
                            modality_label = normalize_text(target_node.label)
                            terms.add(modality_label)
                            graph_expansions["related_modalities"].append(target_node.label)
                    elif edge.edge_type == "analysis_requires_behavioral_event":
                        target_node = graph.nodes.get(edge.target_node_id)
                        if target_node:
                            event_label = normalize_text(target_node.label)
                            terms.add(event_label)
                            graph_expansions["behavioral_events"].append(target_node.label)

    # Second hop expansion if max_hops > 1
    if max_hops > 1:
        # Find regions related to matched tasks
        for task_id in base_expansion["task_ids"]:
            task_node_id = f"node:task:{task_id}"
            if task_node_id in graph.nodes:
                for edge in graph.edges.values():
                    if edge.target_node_id == task_node_id and edge.edge_type == "region_related_to_task":
                        source_node = graph.nodes.get(edge.source_node_id)
                        if source_node:
                            region_label = normalize_text(source_node.label)
                            terms.add(region_label)
                            graph_expansions["related_regions"].append(source_node.label)

    return {
        "terms": sorted(value for value in terms if value),
        "task_ids": base_expansion["task_ids"],
        "behavior_ids": base_expansion["behavior_ids"],
        "affordance_ids": base_expansion["affordance_ids"],
        "suggested_analyses": base_expansion["suggested_analyses"],
        "graph_expansions": {
            "behavioral_events": sorted(set(graph_expansions["behavioral_events"])),
            "related_modalities": sorted(set(graph_expansions["related_modalities"])),
            "related_regions": sorted(set(graph_expansions["related_regions"])),
        },
    }


def match_all(text: str, ontology: Ontology | None = None) -> dict[str, list[LabelMatch]]:
    ontology = ontology or get_ontology()
    return {
        "tasks": match_tasks(text, ontology),
        "behaviors": match_behavior_labels(text, ontology),
        "regions": match_brain_regions(text, ontology),
        "modalities": match_modalities(text, ontology),
        "affordances": match_affordances(text, ontology),
    }


class OntologyMatcher:
    """Object API for ontology matching."""

    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or get_ontology()

    def match_tasks(self, text: str) -> list[LabelMatch]:
        return match_tasks(text, self.ontology)

    def match_behavior_labels(self, text: str) -> list[LabelMatch]:
        return match_behavior_labels(text, self.ontology)

    def match_modalities(self, text: str) -> list[LabelMatch]:
        return match_modalities(text, self.ontology)

    def match_brain_regions(self, text: str) -> list[LabelMatch]:
        return match_brain_regions(text, self.ontology)

    def match_affordances(self, text: str) -> list[LabelMatch]:
        return match_affordances(text, self.ontology)

    def match_all(self, text: str) -> dict[str, list[LabelMatch]]:
        return match_all(text, self.ontology)

    def expand_query_terms(self, query: str) -> dict[str, list[str]]:
        return expand_query_terms(query)

    def expand_query_terms_with_graph(
        self,
        query: str,
        graph: KnowledgeGraph | None = None,
        max_hops: int = 1,
    ) -> dict[str, list[str]]:
        return expand_query_terms_with_graph(query, graph, max_hops)
