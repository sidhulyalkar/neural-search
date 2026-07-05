"""Build a concept authority KG layer from Scholarpedia curated entries.

License: All records carry CC BY-NC-SA 3.0; do not use for commercial exports.
Source: https://www.scholarpedia.org — metadata only, no full article text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)

BUILDER_NAME = "neural_search.ingestion.scholarpedia_builder"
BUILDER_VERSION = "v1.0.0"

# Curated Scholarpedia concept registry — metadata only, no article text.
SCHOLARPEDIA_CONCEPTS: dict[str, dict[str, Any]] = {
    # neural_coding
    "predictive_coding": {
        "title": "Predictive coding",
        "url": "https://www.scholarpedia.org/article/Predictive_coding",
        "aliases": ["predictive coding", "predictive processing", "prediction error", "free energy principle"],
        "related": ["sparse_coding", "bayesian_brain", "population_coding"],
        "domain": "neural_coding",
        "summary": "A framework in which the brain constantly generates predictions about sensory input and updates internal models based on prediction errors.",
        "license": "CC BY-NC-SA 3.0",
    },
    "sparse_coding": {
        "title": "Sparse coding",
        "url": "https://www.scholarpedia.org/article/Sparse_coding",
        "aliases": ["sparse coding", "sparse representation", "sparsity"],
        "related": ["predictive_coding", "independent_component_analysis", "population_coding"],
        "domain": "neural_coding",
        "summary": "A representational strategy where only a small fraction of neurons are active at any given time, enabling efficient coding of natural stimuli.",
        "license": "CC BY-NC-SA 3.0",
    },
    "population_coding": {
        "title": "Population coding",
        "url": "https://www.scholarpedia.org/article/Population_coding",
        "aliases": ["population coding", "population vector", "neural ensemble coding"],
        "related": ["predictive_coding", "sparse_coding", "rate_coding", "neural_manifold"],
        "domain": "neural_coding",
        "summary": "Information is encoded in the collective activity pattern of many neurons rather than single-unit firing rates.",
        "license": "CC BY-NC-SA 3.0",
    },
    "temporal_coding": {
        "title": "Temporal coding",
        "url": "https://www.scholarpedia.org/article/Temporal_coding",
        "aliases": ["temporal coding", "spike timing", "precise timing", "latency coding"],
        "related": ["rate_coding", "population_coding", "spike_timing_dependent_plasticity"],
        "domain": "neural_coding",
        "summary": "Information is encoded in the precise timing of individual spikes or spike patterns relative to a reference signal.",
        "license": "CC BY-NC-SA 3.0",
    },
    "rate_coding": {
        "title": "Rate coding",
        "url": "https://www.scholarpedia.org/article/Rate_coding",
        "aliases": ["rate coding", "firing rate", "mean firing rate", "spike rate"],
        "related": ["temporal_coding", "population_coding"],
        "domain": "neural_coding",
        "summary": "The simplest neural code in which information is carried by the mean firing rate of a neuron averaged over a time window.",
        "license": "CC BY-NC-SA 3.0",
    },
    "neural_manifold": {
        "title": "Neural manifold",
        "url": "https://www.scholarpedia.org/article/Neural_manifold",
        "aliases": ["neural manifold", "population manifold", "low-dimensional manifold", "latent manifold"],
        "related": ["population_coding", "dimensionality_reduction", "attractor_dynamics"],
        "domain": "neural_coding",
        "summary": "The low-dimensional geometric structure embedded in high-dimensional neural population activity that captures the relevant computational variables.",
        "license": "CC BY-NC-SA 3.0",
    },
    # dynamics
    "attractor_dynamics": {
        "title": "Attractor dynamics",
        "url": "https://www.scholarpedia.org/article/Attractor_network",
        "aliases": ["attractor dynamics", "attractor network", "attractor states", "basin of attraction"],
        "related": ["hopfield_networks", "neural_manifold", "chaos_in_neural_systems", "phase_plane_analysis"],
        "domain": "dynamics",
        "summary": "Neural circuits whose activity converges to stable fixed points, limit cycles, or other geometric structures that represent persistent memory or decision states.",
        "license": "CC BY-NC-SA 3.0",
    },
    "neural_oscillations": {
        "title": "Neural oscillations",
        "url": "https://www.scholarpedia.org/article/Neural_oscillations",
        "aliases": ["neural oscillations", "brain oscillations", "brain rhythms", "neural rhythms"],
        "related": ["gamma_oscillations", "theta_oscillations", "sleep_oscillations", "cross_frequency_coupling"],
        "domain": "dynamics",
        "summary": "Rhythmic patterns of neural activity across multiple frequency bands that coordinate information processing across brain regions.",
        "license": "CC BY-NC-SA 3.0",
    },
    "criticality": {
        "title": "Criticality in neural systems",
        "url": "https://www.scholarpedia.org/article/Criticality",
        "aliases": ["criticality", "critical brain", "neural criticality", "self-organized criticality", "SOC"],
        "related": ["chaos_in_neural_systems", "neural_oscillations", "bifurcations"],
        "domain": "dynamics",
        "summary": "The hypothesis that the brain operates near a phase transition between ordered and disordered dynamics, maximizing dynamic range and information transmission.",
        "license": "CC BY-NC-SA 3.0",
    },
    "chaos_in_neural_systems": {
        "title": "Chaos in neural systems",
        "url": "https://www.scholarpedia.org/article/Chaos_in_neural_systems",
        "aliases": ["chaos", "chaotic dynamics", "neural chaos", "deterministic chaos"],
        "related": ["attractor_dynamics", "criticality", "bifurcations", "reservoir_computing"],
        "domain": "dynamics",
        "summary": "Deterministic yet unpredictable dynamics arising from nonlinear interactions in neural circuits, potentially contributing to flexible computation.",
        "license": "CC BY-NC-SA 3.0",
    },
    "bifurcations": {
        "title": "Bifurcations",
        "url": "https://www.scholarpedia.org/article/Bifurcations",
        "aliases": ["bifurcation", "bifurcation theory", "saddle-node bifurcation", "Hopf bifurcation"],
        "related": ["attractor_dynamics", "chaos_in_neural_systems", "phase_plane_analysis"],
        "domain": "dynamics",
        "summary": "Qualitative changes in the behavior of a dynamical system as a parameter crosses a critical threshold, governing transitions between neural states.",
        "license": "CC BY-NC-SA 3.0",
    },
    "phase_plane_analysis": {
        "title": "Phase plane analysis",
        "url": "https://www.scholarpedia.org/article/Phase_plane",
        "aliases": ["phase plane", "phase portrait", "phase space analysis", "nullcline"],
        "related": ["attractor_dynamics", "bifurcations", "chaos_in_neural_systems"],
        "domain": "dynamics",
        "summary": "Geometric analysis of two-variable dynamical systems by plotting trajectories in the plane of the two variables to reveal fixed points and limit cycles.",
        "license": "CC BY-NC-SA 3.0",
    },
    # plasticity
    "hebbian_learning": {
        "title": "Hebbian learning",
        "url": "https://www.scholarpedia.org/article/Hebbian_learning",
        "aliases": ["Hebbian learning", "Hebb rule", "Hebb's rule", "cells that fire together wire together"],
        "related": ["spike_timing_dependent_plasticity", "synaptic_plasticity", "long_term_potentiation", "hopfield_networks"],
        "domain": "plasticity",
        "summary": "A foundational synaptic learning rule stating that connections between simultaneously active neurons are strengthened over time.",
        "license": "CC BY-NC-SA 3.0",
    },
    "spike_timing_dependent_plasticity": {
        "title": "Spike-timing dependent plasticity",
        "url": "https://www.scholarpedia.org/article/Spike-timing_dependent_plasticity",
        "aliases": ["STDP", "spike timing dependent plasticity", "spike-timing dependent plasticity", "timing-dependent plasticity"],
        "related": ["hebbian_learning", "synaptic_plasticity", "temporal_coding", "long_term_potentiation"],
        "domain": "plasticity",
        "summary": "A biologically grounded form of synaptic plasticity where the relative timing of pre- and postsynaptic spikes determines whether synapses are potentiated or depressed.",
        "license": "CC BY-NC-SA 3.0",
    },
    "synaptic_plasticity": {
        "title": "Synaptic plasticity",
        "url": "https://www.scholarpedia.org/article/Synaptic_plasticity",
        "aliases": ["synaptic plasticity", "synapse plasticity", "synaptic modification"],
        "related": ["hebbian_learning", "spike_timing_dependent_plasticity", "long_term_potentiation", "homeostatic_plasticity"],
        "domain": "plasticity",
        "summary": "The activity-dependent modification of synaptic strength, forming the cellular basis for learning and memory in neural circuits.",
        "license": "CC BY-NC-SA 3.0",
    },
    "long_term_potentiation": {
        "title": "Long-term potentiation",
        "url": "https://www.scholarpedia.org/article/Long-term_potentiation",
        "aliases": ["LTP", "long-term potentiation", "long term potentiation", "NMDA-dependent LTP"],
        "related": ["synaptic_plasticity", "hebbian_learning", "hippocampus", "spike_timing_dependent_plasticity"],
        "domain": "plasticity",
        "summary": "A long-lasting increase in synaptic strength following repeated stimulation, widely studied as a cellular correlate of memory formation in the hippocampus.",
        "license": "CC BY-NC-SA 3.0",
    },
    "homeostatic_plasticity": {
        "title": "Homeostatic plasticity",
        "url": "https://www.scholarpedia.org/article/Homeostatic_plasticity",
        "aliases": ["homeostatic plasticity", "synaptic scaling", "homeostatic regulation", "intrinsic plasticity"],
        "related": ["synaptic_plasticity", "hebbian_learning", "spike_timing_dependent_plasticity"],
        "domain": "plasticity",
        "summary": "A compensatory form of neural plasticity that stabilizes neuronal firing rates by globally adjusting synaptic strengths or intrinsic excitability.",
        "license": "CC BY-NC-SA 3.0",
    },
    # oscillations
    "gamma_oscillations": {
        "title": "Gamma oscillations",
        "url": "https://www.scholarpedia.org/article/Gamma_oscillations",
        "aliases": ["gamma oscillations", "gamma band", "gamma rhythm", "gamma frequency", "30-80 Hz oscillations"],
        "related": ["neural_oscillations", "cross_frequency_coupling", "theta_oscillations", "visual_cortex"],
        "domain": "oscillations",
        "summary": "High-frequency (30-80 Hz) oscillations in local field potentials linked to active cortical processing, attention, and sensory binding.",
        "license": "CC BY-NC-SA 3.0",
    },
    "theta_oscillations": {
        "title": "Theta oscillations",
        "url": "https://www.scholarpedia.org/article/Theta_oscillations",
        "aliases": ["theta oscillations", "theta rhythm", "theta band", "hippocampal theta", "4-12 Hz oscillations"],
        "related": ["neural_oscillations", "hippocampus", "gamma_oscillations", "cross_frequency_coupling", "sleep_oscillations"],
        "domain": "oscillations",
        "summary": "Prominent 4-12 Hz rhythmic activity in the hippocampus and entorhinal cortex associated with spatial navigation, memory encoding, and REM sleep.",
        "license": "CC BY-NC-SA 3.0",
    },
    "sleep_oscillations": {
        "title": "Sleep oscillations",
        "url": "https://www.scholarpedia.org/article/Sleep_oscillations",
        "aliases": ["sleep oscillations", "slow oscillations", "sleep spindles", "K-complex", "slow waves"],
        "related": ["neural_oscillations", "theta_oscillations", "hippocampus", "traveling_waves"],
        "domain": "oscillations",
        "summary": "Characteristic EEG rhythms during NREM and REM sleep—including slow oscillations, spindles, and sharp-wave ripples—that coordinate memory consolidation.",
        "license": "CC BY-NC-SA 3.0",
    },
    "cross_frequency_coupling": {
        "title": "Cross-frequency coupling",
        "url": "https://www.scholarpedia.org/article/Cross-frequency_coupling",
        "aliases": ["cross-frequency coupling", "CFC", "phase-amplitude coupling", "PAC", "theta-gamma coupling"],
        "related": ["neural_oscillations", "gamma_oscillations", "theta_oscillations", "hippocampus"],
        "domain": "oscillations",
        "summary": "Interactions between oscillations at different frequencies, particularly the modulation of gamma amplitude by theta phase, thought to coordinate cortical computation.",
        "license": "CC BY-NC-SA 3.0",
    },
    "traveling_waves": {
        "title": "Traveling waves",
        "url": "https://www.scholarpedia.org/article/Traveling_waves",
        "aliases": ["traveling waves", "cortical traveling waves", "propagating waves", "wave propagation"],
        "related": ["neural_oscillations", "sleep_oscillations", "cross_frequency_coupling"],
        "domain": "oscillations",
        "summary": "Spatiotemporally organized patterns of neural activity that propagate across cortical areas, organizing rhythmic processing across large spatial scales.",
        "license": "CC BY-NC-SA 3.0",
    },
    # methods
    "dimensionality_reduction": {
        "title": "Dimensionality reduction",
        "url": "https://www.scholarpedia.org/article/Dimensionality_reduction",
        "aliases": ["dimensionality reduction", "PCA", "principal component analysis", "manifold learning", "UMAP", "t-SNE"],
        "related": ["neural_manifold", "independent_component_analysis", "population_coding"],
        "domain": "methods",
        "summary": "Computational techniques for projecting high-dimensional neural data onto lower-dimensional representations that preserve relevant structure.",
        "license": "CC BY-NC-SA 3.0",
    },
    "independent_component_analysis": {
        "title": "Independent component analysis",
        "url": "https://www.scholarpedia.org/article/Independent_component_analysis",
        "aliases": ["ICA", "independent component analysis", "blind source separation", "FastICA"],
        "related": ["dimensionality_reduction", "sparse_coding", "information_theory"],
        "domain": "methods",
        "summary": "A signal processing method that decomposes multivariate neural signals into maximally statistically independent components, widely used in EEG and fMRI analysis.",
        "license": "CC BY-NC-SA 3.0",
    },
    "kalman_filter": {
        "title": "Kalman filter",
        "url": "https://www.scholarpedia.org/article/Kalman_filter",
        "aliases": ["Kalman filter", "Kalman filtering", "linear-quadratic estimation", "optimal state estimation"],
        "related": ["bayesian_brain", "information_theory", "reinforcement_learning"],
        "domain": "methods",
        "summary": "An optimal recursive estimator for linear dynamical systems that combines predictions and noisy observations, used in neural decoding and brain-computer interfaces.",
        "license": "CC BY-NC-SA 3.0",
    },
    "bayesian_brain": {
        "title": "Bayesian brain",
        "url": "https://www.scholarpedia.org/article/Bayesian_brain",
        "aliases": ["Bayesian brain", "Bayesian inference", "probabilistic inference", "Bayesian model", "Bayesian perception"],
        "related": ["predictive_coding", "kalman_filter", "information_theory"],
        "domain": "methods",
        "summary": "The view that the brain implements approximate Bayesian inference to integrate prior knowledge with sensory evidence to form perceptions and guide behavior.",
        "license": "CC BY-NC-SA 3.0",
    },
    "information_theory": {
        "title": "Information theory",
        "url": "https://www.scholarpedia.org/article/Information_theory",
        "aliases": ["information theory", "Shannon entropy", "mutual information", "channel capacity", "neural information theory"],
        "related": ["bayesian_brain", "independent_component_analysis", "sparse_coding", "population_coding"],
        "domain": "methods",
        "summary": "Mathematical framework for quantifying information transmission and entropy in neural systems, used to measure how much neural activity carries about stimuli or behavior.",
        "license": "CC BY-NC-SA 3.0",
    },
    # circuits
    "hippocampus": {
        "title": "Hippocampus",
        "url": "https://www.scholarpedia.org/article/Hippocampus",
        "aliases": ["hippocampus", "hippocampal", "CA1", "CA3", "dentate gyrus", "HPC"],
        "related": ["theta_oscillations", "long_term_potentiation", "cross_frequency_coupling", "sleep_oscillations"],
        "domain": "circuits",
        "summary": "A medial temporal lobe structure essential for episodic memory, spatial navigation, and pattern separation, organized into CA1, CA3, and dentate gyrus subfields.",
        "license": "CC BY-NC-SA 3.0",
    },
    "cerebellum": {
        "title": "Cerebellum",
        "url": "https://www.scholarpedia.org/article/Cerebellum",
        "aliases": ["cerebellum", "cerebellar", "purkinje cell", "granule cell", "cerebellum cortex"],
        "related": ["basal_ganglia", "reinforcement_learning", "synaptic_plasticity"],
        "domain": "circuits",
        "summary": "A hindbrain structure critical for motor coordination, error-based learning, and timing, organized via Purkinje cell inhibitory microcircuits.",
        "license": "CC BY-NC-SA 3.0",
    },
    "basal_ganglia": {
        "title": "Basal ganglia",
        "url": "https://www.scholarpedia.org/article/Basal_ganglia",
        "aliases": ["basal ganglia", "striatum", "caudate nucleus", "putamen", "globus pallidus", "substantia nigra"],
        "related": ["reinforcement_learning", "parkinson_disease", "cerebellum", "prefrontal_cortex"],
        "domain": "circuits",
        "summary": "A subcortical system of nuclei involved in action selection, habit formation, and reward-based learning through dopaminergic signaling.",
        "license": "CC BY-NC-SA 3.0",
    },
    "prefrontal_cortex": {
        "title": "Prefrontal cortex",
        "url": "https://www.scholarpedia.org/article/Prefrontal_cortex",
        "aliases": ["prefrontal cortex", "PFC", "dorsolateral prefrontal cortex", "DLPFC", "orbitofrontal cortex", "OFC"],
        "related": ["basal_ganglia", "attractor_dynamics", "recurrent_neural_networks"],
        "domain": "circuits",
        "summary": "The anterior frontal cortex supporting working memory, cognitive control, planning, and decision-making through recurrent excitatory activity.",
        "license": "CC BY-NC-SA 3.0",
    },
    "visual_cortex": {
        "title": "Visual cortex",
        "url": "https://www.scholarpedia.org/article/Visual_cortex",
        "aliases": ["visual cortex", "V1", "primary visual cortex", "striate cortex", "V2", "MT", "extrastriate cortex"],
        "related": ["gamma_oscillations", "sparse_coding", "predictive_coding", "population_coding"],
        "domain": "circuits",
        "summary": "Hierarchically organized cortical areas processing visual input from primary (V1) through higher areas, implementing feature detection and scene understanding.",
        "license": "CC BY-NC-SA 3.0",
    },
    # computation
    "reinforcement_learning": {
        "title": "Reinforcement learning",
        "url": "https://www.scholarpedia.org/article/Reinforcement_learning",
        "aliases": ["reinforcement learning", "RL", "temporal difference learning", "TD learning", "Q-learning"],
        "related": ["basal_ganglia", "boltzmann_machine", "recurrent_neural_networks", "bayesian_brain"],
        "domain": "computation",
        "summary": "A framework in which an agent learns to maximize cumulative reward through trial-and-error interactions with an environment via value functions and policy gradients.",
        "license": "CC BY-NC-SA 3.0",
    },
    "recurrent_neural_networks": {
        "title": "Recurrent neural networks",
        "url": "https://www.scholarpedia.org/article/Recurrent_neural_networks",
        "aliases": ["RNN", "recurrent neural networks", "recurrent network", "LSTM", "gated recurrent unit"],
        "related": ["reservoir_computing", "attractor_dynamics", "hopfield_networks", "prefrontal_cortex"],
        "domain": "computation",
        "summary": "Neural network architectures with feedback connections that maintain temporal context, used to model sequential computation and persistent neural activity.",
        "license": "CC BY-NC-SA 3.0",
    },
    "reservoir_computing": {
        "title": "Reservoir computing",
        "url": "https://www.scholarpedia.org/article/Reservoir_computing",
        "aliases": ["reservoir computing", "echo state network", "ESN", "liquid state machine", "LSM"],
        "related": ["recurrent_neural_networks", "chaos_in_neural_systems", "attractor_dynamics"],
        "domain": "computation",
        "summary": "A paradigm using a fixed, randomly connected recurrent network as a dynamical reservoir from which a trained readout extracts temporal features.",
        "license": "CC BY-NC-SA 3.0",
    },
    "hopfield_networks": {
        "title": "Hopfield networks",
        "url": "https://www.scholarpedia.org/article/Hopfield_network",
        "aliases": ["Hopfield network", "Hopfield model", "associative memory network", "content-addressable memory"],
        "related": ["attractor_dynamics", "hebbian_learning", "recurrent_neural_networks", "boltzmann_machine"],
        "domain": "computation",
        "summary": "Recurrent networks with symmetric weights that converge to stored memory patterns, implementing energy-based associative memory via Hebbian learning.",
        "license": "CC BY-NC-SA 3.0",
    },
    "boltzmann_machine": {
        "title": "Boltzmann machine",
        "url": "https://www.scholarpedia.org/article/Boltzmann_machine",
        "aliases": ["Boltzmann machine", "restricted Boltzmann machine", "RBM", "energy-based model", "stochastic neural network"],
        "related": ["hopfield_networks", "reinforcement_learning", "recurrent_neural_networks"],
        "domain": "computation",
        "summary": "A stochastic generative model based on an energy function over binary units that learns data distributions via contrastive divergence.",
        "license": "CC BY-NC-SA 3.0",
    },
    # clinical
    "parkinson_disease": {
        "title": "Parkinson's disease",
        "url": "https://www.scholarpedia.org/article/Parkinson_disease",
        "aliases": ["Parkinson's disease", "Parkinson disease", "PD", "parkinsonism", "dopamine deficiency"],
        "related": ["basal_ganglia", "neural_oscillations"],
        "domain": "clinical",
        "summary": "A progressive neurodegenerative disorder characterized by loss of dopaminergic neurons in the substantia nigra, causing motor symptoms and basal ganglia circuit disruption.",
        "license": "CC BY-NC-SA 3.0",
    },
    "epilepsy": {
        "title": "Epilepsy",
        "url": "https://www.scholarpedia.org/article/Epilepsy",
        "aliases": ["epilepsy", "seizure", "seizure disorder", "epileptic", "focal epilepsy"],
        "related": ["neural_oscillations", "criticality", "hippocampus", "gamma_oscillations"],
        "domain": "clinical",
        "summary": "A neurological disorder defined by recurrent, unprovoked seizures arising from abnormal hypersynchronous neural discharge.",
        "license": "CC BY-NC-SA 3.0",
    },
    "schizophrenia": {
        "title": "Schizophrenia",
        "url": "https://www.scholarpedia.org/article/Schizophrenia",
        "aliases": ["schizophrenia", "psychosis", "schizophrenic disorder", "positive symptoms", "negative symptoms"],
        "related": ["prefrontal_cortex", "gamma_oscillations", "recurrent_neural_networks"],
        "domain": "clinical",
        "summary": "A complex psychiatric disorder involving disruptions to working memory, attention, and perception, linked to prefrontal-cortex and dopamine system dysfunction.",
        "license": "CC BY-NC-SA 3.0",
    },
    "depression": {
        "title": "Depression",
        "url": "https://www.scholarpedia.org/article/Depression",
        "aliases": ["depression", "major depressive disorder", "MDD", "major depression", "depressive disorder"],
        "related": ["prefrontal_cortex", "hippocampus", "reinforcement_learning", "synaptic_plasticity"],
        "domain": "clinical",
        "summary": "A prevalent mood disorder characterized by persistent low affect and anhedonia, associated with altered reward circuitry, hippocampal neuroplasticity, and cortical dysregulation.",
        "license": "CC BY-NC-SA 3.0",
    },
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _concept_evidence(slug: str, field: str, text: str, confidence: float) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"evidence:scholarpedia:{slug}:{field}:{text[:64]}",
        source_type="scholarpedia_registry",
        source_id=f"scholarpedia:{slug}",
        source_field=field,
        evidence_text=text,
        confidence=confidence,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )


def _concept_node(slug: str, entry: dict[str, Any]) -> KnowledgeGraphNode:
    evidence = _concept_evidence(slug, "title", entry["title"], 1.0)
    return KnowledgeGraphNode(
        node_id=make_node_id("concept", slug),
        node_type="concept",
        label=entry["title"],
        aliases=list(entry["aliases"]),
        source_ids=[f"scholarpedia:{slug}"],
        properties={
            "url": entry["url"],
            "domain": entry["domain"],
            "summary": entry["summary"],
            "license": entry["license"],
            "source": "scholarpedia",
        },
        evidence=[evidence],
        confidence=1.0,
        created_at=_now(),
    )


def _domain_node(domain: str) -> KnowledgeGraphNode:
    evidence = GraphEvidence(
        evidence_id=f"evidence:scholarpedia:domain:{domain}",
        source_type="scholarpedia_registry",
        source_id=f"scholarpedia:domain:{domain}",
        source_field="domain",
        evidence_text=domain,
        confidence=1.0,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )
    return KnowledgeGraphNode(
        node_id=make_node_id("concept", f"domain_{domain}"),
        node_type="concept",
        label=domain.replace("_", " ").title(),
        aliases=[domain, domain.replace("_", " ")],
        source_ids=[f"scholarpedia:domain:{domain}"],
        properties={"source": "scholarpedia", "license": "CC BY-NC-SA 3.0", "is_domain_node": True},
        evidence=[evidence],
        confidence=1.0,
        created_at=_now(),
    )


def _make_edge(
    source_id: str,
    edge_type: str,
    target_id: str,
    evidence: GraphEvidence,
    confidence: float,
    properties: dict[str, Any] | None = None,
) -> KnowledgeGraphEdge:
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source_id, edge_type, target_id),
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type=edge_type,
        directed=True,
        confidence=confidence,
        evidence=[evidence],
        properties=properties or {},
        created_at=_now(),
    )


def build_scholarpedia_kg() -> KnowledgeGraph:
    """Build a concept authority KG from the Scholarpedia registry."""
    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    for slug, entry in SCHOLARPEDIA_CONCEPTS.items():
        node = _concept_node(slug, entry)
        nodes[node.node_id] = node

    domains_seen: set[str] = set()
    for entry in SCHOLARPEDIA_CONCEPTS.values():
        domain = entry["domain"]
        if domain not in domains_seen:
            domain_node = _domain_node(domain)
            nodes[domain_node.node_id] = domain_node
            domains_seen.add(domain)

    for slug, entry in SCHOLARPEDIA_CONCEPTS.items():
        concept_node_id = make_node_id("concept", slug)
        domain_node_id = make_node_id("concept", f"domain_{entry['domain']}")
        ev = _concept_evidence(slug, "domain", entry["domain"], 1.0)
        edge = _make_edge(
            concept_node_id, "concept_in_domain", domain_node_id, ev, 1.0,
            {"domain": entry["domain"], "source": "scholarpedia", "license": entry["license"]},
        )
        edges[edge.edge_id] = edge

    for slug, entry in SCHOLARPEDIA_CONCEPTS.items():
        concept_node_id = make_node_id("concept", slug)
        for related_slug in entry.get("related", []):
            if related_slug not in SCHOLARPEDIA_CONCEPTS:
                continue
            related_node_id = make_node_id("concept", related_slug)
            ev = _concept_evidence(slug, "related", related_slug, 0.8)
            edge = _make_edge(
                concept_node_id, "concept_related_to_concept", related_node_id, ev, 0.8,
                {"source": "scholarpedia", "license": entry["license"], "requires_human_review": True},
            )
            edges[edge.edge_id] = edge

    for slug, entry in SCHOLARPEDIA_CONCEPTS.items():
        concept_node_id = make_node_id("concept", slug)
        for alias in entry.get("aliases", []):
            alias_slug = alias.lower().replace(" ", "_").replace("-", "_")
            if alias_slug == slug:
                continue
            alias_node_id = make_node_id("concept", f"alias_{alias_slug}")
            if alias_node_id not in nodes:
                alias_ev = _concept_evidence(slug, "alias", alias, 0.9)
                alias_node = KnowledgeGraphNode(
                    node_id=alias_node_id,
                    node_type="concept",
                    label=alias,
                    aliases=[alias],
                    source_ids=[f"scholarpedia:{slug}"],
                    properties={
                        "source": "scholarpedia",
                        "license": entry["license"],
                        "is_alias_node": True,
                        "canonical_slug": slug,
                    },
                    evidence=[alias_ev],
                    confidence=0.9,
                    created_at=_now(),
                )
                nodes[alias_node.node_id] = alias_node
            ev = _concept_evidence(slug, "alias", alias, 0.9)
            edge = _make_edge(
                concept_node_id, "concept_has_alias", alias_node_id, ev, 0.9,
                {"source": "scholarpedia", "license": entry["license"]},
            )
            edges[edge.edge_id] = edge

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "builder": BUILDER_NAME,
            "builder_version": BUILDER_VERSION,
            "concept_count": len(SCHOLARPEDIA_CONCEPTS),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source": "scholarpedia",
            "license": "CC BY-NC-SA 3.0",
        },
    )
