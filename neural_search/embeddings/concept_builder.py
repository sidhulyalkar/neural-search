"""Build dense concept embeddings from ontology definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neural_search.embeddings.concept_embeddings import (
    ConceptEmbedding,
    write_concept_embeddings,
)
from neural_search.embeddings.hashing import HashingEmbeddingProvider


class ConceptEmbeddingBuilder:
    """Build dense concept embeddings from ontology definitions."""

    def __init__(
        self,
        text_model: str = "hashing",
        target_dim: int = 128,
        minor_dim: int = 64,
    ):
        """Initialize builder.

        Args:
            text_model: Embedding model ("hashing" or sentence-transformer model name)
            target_dim: Dimension for major concept types (task, modality)
            minor_dim: Dimension for minor concept types (behavior, region)
        """
        self.text_model = text_model
        self.target_dim = target_dim
        self.minor_dim = minor_dim
        self.model_version = f"{text_model}_{target_dim}d"

        # Initialize embedding provider
        if text_model == "hashing":
            self._hash_provider_major = HashingEmbeddingProvider(
                dimensions=target_dim, normalize_embeddings=True
            )
            self._hash_provider_minor = HashingEmbeddingProvider(
                dimensions=minor_dim, normalize_embeddings=True
            )
            self._use_hashing = True
        else:
            self._use_hashing = False
            try:
                from neural_search.embeddings.sentence_transformers import (
                    SentenceTransformerProvider,
                )
                self._st_provider = SentenceTransformerProvider(
                    model_name=text_model
                )
            except (ImportError, RuntimeError):
                # Fall back to hashing
                self._hash_provider_major = HashingEmbeddingProvider(
                    dimensions=target_dim, normalize_embeddings=True
                )
                self._hash_provider_minor = HashingEmbeddingProvider(
                    dimensions=minor_dim, normalize_embeddings=True
                )
                self._use_hashing = True

    def _embed_concept_text(
        self,
        text: str,
        concept_type: str,
    ) -> list[float]:
        """Embed concept text.

        Args:
            text: Text to embed (label + definition + aliases)
            concept_type: Type determines dimension

        Returns:
            Embedding vector
        """
        is_major = concept_type in ("task", "modality", "analysis")
        target_dim = self.target_dim if is_major else self.minor_dim

        if self._use_hashing:
            provider = self._hash_provider_major if is_major else self._hash_provider_minor
            return provider.embed_text(text)
        else:
            # Use sentence transformer
            emb = self._st_provider.encode(text)
            # Reduce dimension if needed
            if len(emb) > target_dim:
                # Simple truncation (PCA would be better for production)
                emb = emb[:target_dim]
            elif len(emb) < target_dim:
                # Pad with zeros
                emb = np.pad(emb, (0, target_dim - len(emb)))
            return emb.tolist()

    def build_task_embeddings(
        self,
        ontology_path: str | Path,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for all tasks from ontology.

        Args:
            ontology_path: Path to behavioral_task_ontology.yaml

        Returns:
            List of task concept embeddings
        """
        path = Path(ontology_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        embeddings: list[ConceptEmbedding] = []
        tasks = data.get("tasks", [])

        for task in tasks:
            task_id = task.get("id", "")
            if not task_id:
                continue

            label = task.get("label", task_id.replace("_", " ").title())
            definition = task.get("definition", "")
            aliases = task.get("aliases", [])
            category = task.get("category", "")
            behavioral_events = task.get("behavioral_events", [])

            # Build rich text for embedding
            text_parts = [
                label,
                definition,
                f"Category: {category}" if category else "",
                f"Aliases: {', '.join(aliases)}" if aliases else "",
                f"Events: {', '.join(behavioral_events)}" if behavioral_events else "",
            ]
            text = " ".join(part for part in text_parts if part)

            embedding = self._embed_concept_text(text, "task")

            embeddings.append(
                ConceptEmbedding(
                    concept_id=f"task:{task_id}",
                    concept_type="task",
                    label=label,
                    embedding=embedding,
                    model_version=self.model_version,
                    aliases=aliases,
                    parent_concepts=[f"category:{category}"] if category else [],
                    child_concepts=[f"event:{e}" for e in behavioral_events],
                    definition=definition,
                )
            )

        return embeddings

    def build_behavior_embeddings(
        self,
        ontology_path: str | Path,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for behavioral events.

        Args:
            ontology_path: Path to behavioral_task_ontology.yaml

        Returns:
            List of behavior concept embeddings
        """
        path = Path(ontology_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        embeddings: list[ConceptEmbedding] = []
        behaviors = data.get("behaviors", [])

        for behavior in behaviors:
            behavior_id = behavior.get("id", "")
            if not behavior_id:
                continue

            label = behavior.get("label", behavior_id.replace("_", " ").title())
            definition = behavior.get("definition", "")
            aliases = behavior.get("aliases", [])
            category = behavior.get("category", "")
            signal_types = behavior.get("signal_types", [])

            text_parts = [
                label,
                definition,
                f"Category: {category}" if category else "",
                f"Aliases: {', '.join(aliases)}" if aliases else "",
                f"Signal types: {', '.join(signal_types)}" if signal_types else "",
            ]
            text = " ".join(part for part in text_parts if part)

            embedding = self._embed_concept_text(text, "behavior")

            embeddings.append(
                ConceptEmbedding(
                    concept_id=f"behavior:{behavior_id}",
                    concept_type="behavior",
                    label=label,
                    embedding=embedding,
                    model_version=self.model_version,
                    aliases=aliases,
                    parent_concepts=[f"category:{category}"] if category else [],
                    definition=definition,
                )
            )

        return embeddings

    def build_modality_embeddings(
        self,
        modalities: list[dict[str, Any]] | None = None,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for recording modalities.

        Args:
            modalities: Optional list of modality definitions

        Returns:
            List of modality concept embeddings
        """
        if modalities is None:
            # Default modalities
            modalities = [
                {
                    "id": "neuropixels",
                    "label": "Neuropixels",
                    "definition": "High-density silicon probe for extracellular electrophysiology recording",
                    "aliases": ["neuropixel", "np probe"],
                    "category": "electrophysiology",
                },
                {
                    "id": "calcium_imaging",
                    "label": "Calcium Imaging",
                    "definition": "Optical imaging of neural activity using calcium indicators",
                    "aliases": ["two photon", "2p", "gcamp", "optical imaging"],
                    "category": "optical",
                },
                {
                    "id": "extracellular_ephys",
                    "label": "Extracellular Electrophysiology",
                    "definition": "Recording electrical activity outside neurons",
                    "aliases": ["ephys", "spikes", "electrophysiology", "electrodes"],
                    "category": "electrophysiology",
                },
                {
                    "id": "eeg",
                    "label": "EEG",
                    "definition": "Electroencephalography recording brain electrical activity from scalp",
                    "aliases": ["electroencephalography", "scalp eeg"],
                    "category": "electrophysiology",
                },
                {
                    "id": "ecog",
                    "label": "ECoG",
                    "definition": "Electrocorticography recording from cortical surface",
                    "aliases": ["electrocorticography", "cortical surface recording"],
                    "category": "electrophysiology",
                },
                {
                    "id": "fmri",
                    "label": "fMRI",
                    "definition": "Functional magnetic resonance imaging measuring blood oxygenation",
                    "aliases": ["functional mri", "bold", "fmri"],
                    "category": "imaging",
                },
                {
                    "id": "behavior_video",
                    "label": "Behavior Video",
                    "definition": "Video recording of animal behavior",
                    "aliases": ["video", "behavioral video", "camera recording"],
                    "category": "behavioral",
                },
                {
                    "id": "pose_tracking",
                    "label": "Pose Tracking",
                    "definition": "Tracking body position and kinematics from video",
                    "aliases": ["kinematics", "deeplabcut", "dlc", "body tracking"],
                    "category": "behavioral",
                },
                {
                    "id": "fiber_photometry",
                    "label": "Fiber Photometry",
                    "definition": "Optical recording through fiber optic implant",
                    "aliases": ["photometry", "fiber recording"],
                    "category": "optical",
                },
                {
                    "id": "ieeg",
                    "label": "iEEG",
                    "definition": "Intracranial EEG with depth electrodes",
                    "aliases": ["intracranial eeg", "depth electrode", "seeg", "stereo eeg"],
                    "category": "electrophysiology",
                },
            ]

        embeddings: list[ConceptEmbedding] = []

        for mod in modalities:
            mod_id = mod.get("id", "")
            if not mod_id:
                continue

            label = mod.get("label", mod_id.replace("_", " ").title())
            definition = mod.get("definition", "")
            aliases = mod.get("aliases", [])
            category = mod.get("category", "")

            text_parts = [
                label,
                definition,
                f"Category: {category}" if category else "",
                f"Aliases: {', '.join(aliases)}" if aliases else "",
            ]
            text = " ".join(part for part in text_parts if part)

            embedding = self._embed_concept_text(text, "modality")

            embeddings.append(
                ConceptEmbedding(
                    concept_id=f"modality:{mod_id}",
                    concept_type="modality",
                    label=label,
                    embedding=embedding,
                    model_version=self.model_version,
                    aliases=aliases,
                    parent_concepts=[f"category:{category}"] if category else [],
                    definition=definition,
                )
            )

        return embeddings

    def build_analysis_embeddings(
        self,
        analysis_path: str | Path | None = None,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for analysis methods.

        Args:
            analysis_path: Path to analysis_methods.yaml

        Returns:
            List of analysis concept embeddings
        """
        analyses: list[dict[str, Any]] = []

        if analysis_path is not None:
            path = Path(analysis_path)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                analyses = data.get("analysis_methods", [])

        if not analyses:
            # Default analysis methods
            analyses = [
                {
                    "id": "population_decoding",
                    "label": "Population Decoding",
                    "definition": "Decode behavioral or stimulus variables from neural population activity",
                    "aliases": ["neural decoding", "classification", "svm decoding", "decoder"],
                    "category": "decoding",
                },
                {
                    "id": "spike_rate_psth",
                    "label": "PSTH Analysis",
                    "definition": "Peri-stimulus time histogram of spike rates",
                    "aliases": ["psth", "firing rate", "spike rate analysis", "trial average"],
                    "category": "descriptive",
                },
                {
                    "id": "dimensionality_reduction",
                    "label": "Dimensionality Reduction",
                    "definition": "Reduce neural data to lower-dimensional representation",
                    "aliases": ["pca", "umap", "tsne", "latent factors"],
                    "category": "latent",
                },
                {
                    "id": "glm_encoding",
                    "label": "GLM Encoding",
                    "definition": "Generalized linear model for neural encoding",
                    "aliases": ["encoding model", "glm", "stimulus encoding"],
                    "category": "encoding",
                },
                {
                    "id": "reinforcement_learning_model",
                    "label": "RL Modeling",
                    "definition": "Fit reinforcement learning models to behavioral data",
                    "aliases": ["rl model", "q learning", "td learning", "value learning"],
                    "category": "computational",
                },
                {
                    "id": "spectral_analysis",
                    "label": "Spectral Analysis",
                    "definition": "Frequency domain analysis of neural signals",
                    "aliases": ["power spectrum", "lfp analysis", "oscillations", "frequency"],
                    "category": "signal_processing",
                },
                {
                    "id": "connectivity_analysis",
                    "label": "Connectivity Analysis",
                    "definition": "Measure functional or effective connectivity between regions",
                    "aliases": ["functional connectivity", "granger causality", "correlation"],
                    "category": "network",
                },
                {
                    "id": "trial_classification",
                    "label": "Trial Classification",
                    "definition": "Classify trials based on neural or behavioral features",
                    "aliases": ["trial type classification", "outcome prediction"],
                    "category": "decoding",
                },
            ]

        embeddings: list[ConceptEmbedding] = []

        for analysis in analyses:
            analysis_id = analysis.get("id", "")
            if not analysis_id:
                continue

            label = analysis.get("label", analysis_id.replace("_", " ").title())
            definition = analysis.get("definition", "")
            aliases = analysis.get("aliases", [])
            category = analysis.get("category", "")
            required_signals = analysis.get("required_signals", [])

            text_parts = [
                label,
                definition,
                f"Category: {category}" if category else "",
                f"Aliases: {', '.join(aliases)}" if aliases else "",
                f"Requires: {', '.join(required_signals)}" if required_signals else "",
            ]
            text = " ".join(part for part in text_parts if part)

            embedding = self._embed_concept_text(text, "analysis")

            embeddings.append(
                ConceptEmbedding(
                    concept_id=f"analysis:{analysis_id}",
                    concept_type="analysis",
                    label=label,
                    embedding=embedding,
                    model_version=self.model_version,
                    aliases=aliases,
                    parent_concepts=[f"category:{category}"] if category else [],
                    definition=definition,
                )
            )

        return embeddings

    def build_region_embeddings(
        self,
        regions: list[dict[str, Any]] | None = None,
    ) -> list[ConceptEmbedding]:
        """Build embeddings for brain regions.

        Args:
            regions: Optional list of region definitions

        Returns:
            List of region concept embeddings
        """
        if regions is None:
            # Default brain regions
            regions = [
                {"id": "prefrontal_cortex", "label": "Prefrontal Cortex", "aliases": ["pfc", "mpfc", "dlpfc"]},
                {"id": "hippocampus", "label": "Hippocampus", "aliases": ["hpc", "ca1", "ca3", "dentate gyrus"]},
                {"id": "striatum", "label": "Striatum", "aliases": ["dorsal striatum", "ventral striatum", "caudate", "putamen"]},
                {"id": "visual_cortex", "label": "Visual Cortex", "aliases": ["v1", "v2", "v4", "primary visual"]},
                {"id": "motor_cortex", "label": "Motor Cortex", "aliases": ["m1", "primary motor", "motor area"]},
                {"id": "amygdala", "label": "Amygdala", "aliases": ["bla", "basolateral amygdala"]},
                {"id": "thalamus", "label": "Thalamus", "aliases": ["thalamic nuclei", "relay nucleus"]},
                {"id": "cerebellum", "label": "Cerebellum", "aliases": ["cerebellar cortex", "purkinje"]},
                {"id": "basal_ganglia", "label": "Basal Ganglia", "aliases": ["bg", "substantia nigra", "gpe", "gpi"]},
                {"id": "orbitofrontal_cortex", "label": "Orbitofrontal Cortex", "aliases": ["ofc"]},
            ]

        embeddings: list[ConceptEmbedding] = []

        for region in regions:
            region_id = region.get("id", "")
            if not region_id:
                continue

            label = region.get("label", region_id.replace("_", " ").title())
            aliases = region.get("aliases", [])
            definition = region.get("definition", f"Brain region: {label}")

            text_parts = [
                label,
                definition,
                f"Aliases: {', '.join(aliases)}" if aliases else "",
            ]
            text = " ".join(part for part in text_parts if part)

            embedding = self._embed_concept_text(text, "region")

            embeddings.append(
                ConceptEmbedding(
                    concept_id=f"region:{region_id}",
                    concept_type="region",
                    label=label,
                    embedding=embedding,
                    model_version=self.model_version,
                    aliases=aliases,
                    definition=definition,
                )
            )

        return embeddings

    def build_all_embeddings(
        self,
        ontology_path: str | Path,
        analysis_path: str | Path | None = None,
    ) -> list[ConceptEmbedding]:
        """Build all concept embeddings from ontology files.

        Args:
            ontology_path: Path to behavioral_task_ontology.yaml
            analysis_path: Optional path to analysis_methods.yaml

        Returns:
            Combined list of all concept embeddings
        """
        all_embeddings: list[ConceptEmbedding] = []

        # Build task embeddings
        all_embeddings.extend(self.build_task_embeddings(ontology_path))

        # Build behavior embeddings
        all_embeddings.extend(self.build_behavior_embeddings(ontology_path))

        # Build modality embeddings
        all_embeddings.extend(self.build_modality_embeddings())

        # Build analysis embeddings
        all_embeddings.extend(self.build_analysis_embeddings(analysis_path))

        # Build region embeddings
        all_embeddings.extend(self.build_region_embeddings())

        return all_embeddings


def build_concept_embeddings_from_ontology(
    ontology_path: str | Path,
    analysis_path: str | Path | None = None,
    output_path: str | Path | None = None,
    text_model: str = "hashing",
    target_dim: int = 128,
) -> list[ConceptEmbedding]:
    """Build and optionally save concept embeddings.

    Args:
        ontology_path: Path to behavioral_task_ontology.yaml
        analysis_path: Optional path to analysis_methods.yaml
        output_path: Optional path to save embeddings
        text_model: Embedding model to use
        target_dim: Target embedding dimension

    Returns:
        List of concept embeddings
    """
    builder = ConceptEmbeddingBuilder(
        text_model=text_model,
        target_dim=target_dim,
    )

    embeddings = builder.build_all_embeddings(ontology_path, analysis_path)

    if output_path is not None:
        write_concept_embeddings(embeddings, output_path)

    return embeddings
