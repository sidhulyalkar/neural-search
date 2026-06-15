"""Fuzzy ontology matching for tasks, behaviors, modalities, and regions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from neural_search.ontology.loader import (
    get_brain_regions,
    get_ontology,
    get_recording_scales,
)
from neural_search.ontology.models import (
    BrainRegion,
    LabelMatch,
    Ontology,
    RecordingScale,
)

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
    "motor_cortex": ["motor cortex", "motor control"],
    "visual_cortex": ["visual cortex", "visual areas", "visual cortical areas"],
    "somatosensory_cortex": ["somatosensory cortex"],
    "parietal_cortex": ["parietal cortex", "ppc", "posterior parietal", "posterior parietal cortex"],
    "OFC": ["ofc", "orbitofrontal", "orbitofrontal cortex"],
    "mPFC": ["mpfc", "medial prefrontal", "medial prefrontal cortex", "prelimbic", "infralimbic"],
    "ACC": ["acc", "anterior cingulate", "anterior cingulate cortex", "cingulate cortex"],
    "hippocampus": ["hippocampus", "hpc", "hipp", "hippocampal"],
    "striatum": ["striatum", "striatal", "basal ganglia"],
    "VTA": ["vta", "ventral tegmental area", "midbrain", "dopamine midbrain"],
    "SNc": ["snc", "substantia nigra", "substantia nigra pars compacta", "nigra"],
    "auditory_cortex": ["auditory cortex", "a1", "primary auditory cortex", "auditory areas", "temporal cortex"],
    "prefrontal_cortex": ["prefrontal cortex", "pfc", "frontal cortex", "frontal lobe"],
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
    # Behaviors - reward and value
    "reward_prediction": ["reward prediction", "reward prediction error", "rpe", "prediction error", "reward expectation"],
    "reward_omission": ["reward omission", "omission", "no reward", "reward absence", "unexpected omission"],
    "dopamine": ["dopamine", "da", "dopaminergic", "dopamine signal", "dopamine release"],
    # Behaviors - motor/kinematic
    "position": ["position", "cursor position", "hand position", "limb position", "spatial position"],
    "velocity": ["velocity", "movement velocity", "hand velocity", "speed", "movement speed"],
    "cursor_movement": ["cursor movement", "cursor control", "cursor trajectory"],
    # Behaviors - choice/decision
    "choice": ["choice", "decision", "action selection", "behavioral choice"],
    "trial_outcome": ["trial outcome", "outcome", "success", "failure", "correct", "error"],
    "learning": ["learning", "acquisition", "rule learning", "task learning"],
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
    *,
    allow_fuzzy: bool = True,
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
    if not allow_fuzzy or len(normalized.split()) > 32:
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


SPECIES_CONTEXT_ALIASES: dict[str, list[str]] = {
    species: ALIASES[species]
    for species in (
        "mouse",
        "rat",
        "human",
        "macaque",
        "marmoset",
        "zebrafish",
        "drosophila",
    )
}


def _detect_species_context(text: str) -> set[str]:
    normalized = normalize_text(text)
    detected: set[str] = set()
    for species, aliases in SPECIES_CONTEXT_ALIASES.items():
        if any(_contains_phrase(normalized, alias) for alias in [species, *aliases]):
            detected.add(species)
    if "macaque" in detected or "marmoset" in detected:
        detected.add("non_human_primate")
    return detected


def _species_alias_phrases(region: BrainRegion, detected_species: set[str]) -> list[tuple[str, float, str]]:
    phrases: list[tuple[str, float, str]] = []
    for species in sorted(detected_species):
        for alias in region.species_aliases.get(species, []):
            phrases.append((alias, 0.95, f"species_alias:{species}"))
    return phrases


def _species_scope_adjusted_match(
    match: LabelMatch | None,
    region: BrainRegion,
    detected_species: set[str],
) -> LabelMatch | None:
    if match is None or not detected_species:
        return match
    scope = {normalize_text(value).replace(" ", "_") for value in region.species_scope}
    if not scope or {"cross_species", "generic_mammal"} & scope:
        return match
    if detected_species & scope:
        return match
    return match.model_copy(
        update={
            "confidence": max(round(match.confidence - 0.18, 3), 0.1),
            "match_type": f"{match.match_type}:species_mismatch",
        },
    )


def _brain_region_entries(ontology: Ontology) -> dict[str, BrainRegion]:
    entries = {region.id: region for region in get_brain_regions()}
    generic_region_names = {
        "brain",
        "cortex",
        "cerebral_cortex",
        "frontal_cortex",
    }
    yaml_aliases = {
        normalize_text(alias)
        for region in entries.values()
        for alias in [region.id, region.label, *region.aliases]
    }
    for region_name in ontology.region_names:
        if region_name in generic_region_names:
            continue
        if normalize_text(region_name) in yaml_aliases:
            continue
        entries.setdefault(
            region_name,
            BrainRegion(
                id=region_name,
                label=region_name,
                aliases=ALIASES.get(region_name, []),
            ),
        )
    for region_name, aliases in ALIASES.items():
        if region_name in entries or any(region_name == item for item in ontology.region_names):
            entries.setdefault(
                region_name,
                BrainRegion(id=region_name, label=region_name, aliases=aliases),
            )
    return entries


def _recording_scale_entries() -> dict[str, RecordingScale]:
    return {scale.id: scale for scale in get_recording_scales()}


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


def match_recording_scales(text: str) -> list[LabelMatch]:
    """Match neural sampling/recording scale terms such as LFP or single-unit."""

    matches: list[LabelMatch] = []
    for scale_id, scale in _recording_scale_entries().items():
        phrases = [
            (scale.label, 0.98, "label"),
            (scale_id, 0.96, "id"),
            (scale.sampling_unit, 0.90, "sampling_unit"),
            (scale.signal_form, 0.86, "signal_form"),
        ]
        phrases.extend((alias, 0.94, "scale_alias") for alias in scale.aliases)
        phrases.extend((data_type, 0.90, "data_type") for data_type in scale.data_types)
        match = _best_phrase_match(
            text,
            scale_id,
            scale.label,
            "recording_scale",
            phrases,
        )
        if match:
            matches.append(match)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def match_brain_regions(text: str, ontology: Ontology | None = None) -> list[LabelMatch]:
    ontology = ontology or get_ontology()
    matches: list[LabelMatch] = []
    entries = _brain_region_entries(ontology)
    detected_species = _detect_species_context(text)
    for region_id, region in entries.items():
        phrases = [
            (region.label, 0.97, "label"),
            (region_id, 0.96, "id"),
        ]
        phrases.extend((alias, 0.94, "region_alias") for alias in region.aliases)
        phrases.extend(_species_alias_phrases(region, detected_species))
        for alias in _aliases_for(region_id):
            phrases.append((alias, 0.92 if alias != region_id else 0.96, "legacy_alias"))
        match = _best_phrase_match(
            text,
            region_id,
            region.label,
            "brain_region",
            phrases,
            allow_fuzzy=False,
        )
        match = _species_scope_adjusted_match(match, region, detected_species)
        if match:
            matches.append(match)

    by_id = {match.id: match for match in matches}
    for match in list(matches):
        region = entries.get(match.id)
        if region is None:
            continue
        for parent_id in region.parents:
            if parent_id in by_id:
                continue
            parent = entries.get(parent_id)
            by_id[parent_id] = LabelMatch(
                id=parent_id,
                label=parent.label if parent else parent_id,
                confidence=max(match.confidence - 0.08, 0.1),
                evidence=f"parent:{match.id}",
                category="brain_region",
                match_type="parent",
            )
    return sorted(by_id.values(), key=lambda item: item.confidence, reverse=True)


def brain_region_children_by_parent(
    ontology: Ontology | None = None,
) -> dict[str, list[str]]:
    """Return child region IDs keyed by parent region ID."""

    ontology = ontology or get_ontology()
    entries = _brain_region_entries(ontology)
    children: dict[str, set[str]] = {region_id: set(region.children) for region_id, region in entries.items()}
    for region in entries.values():
        for parent_id in region.parents:
            children.setdefault(parent_id, set()).add(region.id)
    return {parent: sorted(values) for parent, values in sorted(children.items()) if values}


def expand_brain_region_ids(
    region_ids: Iterable[str],
    *,
    include_descendants: bool = True,
    ontology: Ontology | None = None,
) -> list[str]:
    """Expand brain-region IDs through the region hierarchy.

    Parent matches are already emitted by ``match_brain_regions``. This helper
    handles the opposite direction for broad queries such as "hippocampus",
    where callers may want CA1, CA2, CA3, dentate gyrus, and other children.
    """

    ontology = ontology or get_ontology()
    entries = _brain_region_entries(ontology)
    requested = [region_id for region_id in region_ids if region_id in entries]
    expanded: set[str] = set(requested)
    if not include_descendants:
        return sorted(expanded)

    children = brain_region_children_by_parent(ontology)
    stack = list(requested)
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child not in expanded:
                expanded.add(child)
                stack.append(child)
    return sorted(expanded)


def expand_brain_region_query(
    query: str,
    *,
    exact: bool = False,
    ontology: Ontology | None = None,
) -> dict[str, list[str] | bool]:
    """Match brain regions and optionally expand broad regions to descendants."""

    ontology = ontology or get_ontology()
    matches = match_brain_regions(query, ontology)
    matched_ids = [match.id for match in matches]
    return {
        "matched_region_ids": matched_ids,
        "expanded_region_ids": expand_brain_region_ids(
            matched_ids,
            include_descendants=not exact,
            ontology=ontology,
        ),
        "exact": exact,
    }


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
        "recording_scales": match_recording_scales(text),
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

    def match_recording_scales(self, text: str) -> list[LabelMatch]:
        return match_recording_scales(text)

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
