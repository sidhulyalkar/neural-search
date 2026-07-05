"""Build the NeuroSynth topic-activation KG layer.

Creates:
  - topic_activates_region edges (forward inference: topic -> regions activated)
  - region_implicated_in_topic edges (reverse inference: region -> topics)

Reads raw NeuroSynth v7 files directly — no NiMARE Dataset pickle required:
  data/neurosynth/neurosynth/
    data-neurosynth_version-7_coordinates.tsv.gz
    data-neurosynth_version-7_metadata.tsv.gz
    data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz
    data-neurosynth_version-7_vocab-terms_vocabulary.txt

Download files first: python scripts/ingestion/download_neurosynth.py
(The conversion step is no longer required — just the file downloads.)

Requires: pip install pandas numpy scipy
"""

from __future__ import annotations

import logging
from pathlib import Path

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "neurosynth"
# NiMARE nests downloaded files one level deeper
_NESTED_DIR = DATA_DIR / "neurosynth"

# Minimum tfidf weight to consider a term "present" in a study
TFIDF_THRESHOLD = 0.001
# Minimum studies with a term before we emit region edges
MIN_STUDY_COUNT = 5
# Minimum activations in a region (out of all activations for that term)
MIN_REGION_ACTIVATIONS = 3
# Minimum frequency fraction of all activations
MIN_ACTIVATION_FREQ = 0.02

TERM_TO_TOPIC: dict[str, str] = {
    # Motor
    "motor": "motor_learning",
    "motor cortex": "motor_learning",
    "movement": "motor_learning",
    "finger tapping": "motor_learning",
    "motor execution": "motor_learning",
    "motor imagery": "motor_learning",
    # Memory
    "memory": "episodic_memory",
    "episodic memory": "episodic_memory",
    "working memory": "working_memory",
    "long-term memory": "episodic_memory",
    "semantic memory": "episodic_memory",
    "spatial memory": "spatial_navigation",
    "encoding": "episodic_memory",
    "retrieval": "episodic_memory",
    # Attention
    "attention": "attention_and_salience",
    "visual attention": "attention_and_salience",
    "selective attention": "attention_and_salience",
    "sustained attention": "attention_and_salience",
    # Reward / Decision
    "reward": "reward_learning",
    "decision making": "decision_making",
    "decision": "decision_making",
    "reinforcement": "reward_learning",
    "value": "reward_learning",
    "risk": "decision_making",
    # Emotion
    "emotion": "emotional_processing",
    "emotional": "emotional_processing",
    "fear": "fear_and_anxiety",
    "anxiety": "fear_and_anxiety",
    "pain": "emotional_processing",
    # Social
    "social": "social_behavior",
    "mentalizing": "social_behavior",
    "theory of mind": "social_behavior",
    "empathy": "social_behavior",
    # Language
    "language": "cognitive_control",
    "speech": "cognitive_control",
    "semantic": "cognitive_control",
    # Executive
    "cognitive control": "cognitive_control",
    "inhibition": "cognitive_control",
    "executive": "cognitive_control",
    "conflict": "cognitive_control",
    "task switching": "cognitive_control",
    # Spatial
    "navigation": "spatial_navigation",
    "spatial": "spatial_navigation",
    "place cells": "spatial_navigation",
    # Default mode
    "default mode": "episodic_memory",
    "mind wandering": "episodic_memory",
}

MNI_STRUCTURE_TO_REGION: dict[str, str] = {
    "hippocampus": "hippocampus",
    "amygdala": "amygdala_basolateral",
    "prefrontal": "medial_prefrontal_cortex",
    "anterior cingulate": "anterior_cingulate_cortex",
    "striatum": "dorsal_striatum",
    "putamen": "dorsal_striatum",
    "caudate": "dorsal_striatum",
    "nucleus accumbens": "nucleus_accumbens",
    "thalamus": "mediodorsal_thalamus",
    "insula": "insular_cortex",
    "motor cortex": "primary_motor_cortex",
    "cerebellum": "cerebellar_cortex",
    "parahippocampal": "entorhinal_cortex",
    "orbitofrontal": "orbitofrontal_cortex",
    "posterior cingulate": "posterior_cingulate_cortex",
    "parietal": "posterior_parietal_cortex",
    "temporal": "temporal_cortex",
    "occipital": "visual_cortex",
    "substantia nigra": "substantia_nigra_compacta",
    "ventral tegmental": "vta",
    "habenula": "lateral_habenula",
    "periaqueductal": "periaqueductal_gray",
    "locus coeruleus": "locus_coeruleus",
}

_MNI_REGIONS = [
    # (x_range, y_range, z_range, label)
    ((-35, 35), (5, 55), (10, 50), "prefrontal cortex"),
    ((-20, 20), (15, 50), (5, 35), "anterior cingulate cortex"),
    ((-40, 40), (-5, 20), (10, 50), "motor cortex"),
    ((-30, 30), (-55, -10), (-15, 10), "thalamus"),
    ((-30, 30), (-10, 20), (-15, 5), "striatum"),
    ((-15, 15), (5, 25), (-10, 5), "nucleus accumbens"),
    ((-35, 35), (-30, 5), (-30, -5), "hippocampus"),
    ((-25, 25), (-5, 15), (-25, -10), "amygdala"),
    ((-55, 55), (-30, 15), (-5, 15), "insula"),
    ((-30, 30), (-55, -25), (-40, -20), "cerebellum"),
    ((-20, 20), (-30, -5), (-35, -15), "substantia nigra"),
    ((-20, 20), (-18, 0), (-25, -15), "ventral tegmental area"),
    ((-30, 30), (-70, -40), (-10, 15), "posterior cingulate cortex"),
    ((-50, 50), (-80, -55), (-5, 25), "occipital cortex"),
    ((-60, 60), (-50, -10), (-10, 25), "temporal cortex"),
    ((-60, 60), (-80, -50), (20, 65), "parietal cortex"),
    ((-45, 45), (-15, 10), (-25, -5), "orbitofrontal cortex"),
    ((-30, 30), (-30, -10), (-10, 5), "parahippocampal gyrus"),
]


def _mni_to_region_label(x: float, y: float, z: float) -> str | None:
    for (x_min, x_max), (y_min, y_max), (z_min, z_max), label in _MNI_REGIONS:
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
            return label
    return None


def _term_to_topic_id(term: str) -> str:
    lower = term.lower().strip()
    if lower in TERM_TO_TOPIC:
        return f"topic:{TERM_TO_TOPIC[lower]}"
    safe = lower.replace(" ", "_").replace("-", "_")
    return f"concept:{safe}"


def _region_label_to_node_id(region_label: str) -> str:
    lower = region_label.lower().strip()
    for key, region_id in MNI_STRUCTURE_TO_REGION.items():
        if key in lower:
            return f"ontology_region:{region_id}"
    safe = lower.replace(" ", "_").replace("-", "_")
    return f"ontology_region:{safe}"


def _find_neurosynth_files() -> dict[str, Path] | None:
    """Locate the raw TSV/NPZ files downloaded by fetch_neurosynth."""
    search_dirs = [_NESTED_DIR, DATA_DIR]
    for d in search_dirs:
        if not d.exists():
            continue
        coords = list(d.glob("*_coordinates.tsv.gz"))
        meta = list(d.glob("*_metadata.tsv.gz"))
        features = list(d.glob("*vocab-terms*_features.npz"))
        vocab = list(d.glob("*vocab-terms*_vocabulary.txt"))
        if coords and meta and features and vocab:
            return {
                "coordinates": coords[0],
                "metadata": meta[0],
                "features": features[0],
                "vocabulary": vocab[0],
            }
    return None


def build_neurosynth_kg() -> KnowledgeGraph:
    """Build KG directly from raw NeuroSynth TSV/NPZ files."""
    try:
        import numpy as np
        import pandas as pd
        import scipy.sparse
    except ImportError:
        log.error("pandas / numpy / scipy not installed. Run: pip install pandas numpy scipy")
        return KnowledgeGraph(nodes=[], edges=[])

    files = _find_neurosynth_files()
    if files is None:
        log.warning(
            "NeuroSynth raw files not found under %s. "
            "Run: python scripts/ingestion/download_neurosynth.py",
            DATA_DIR,
        )
        return KnowledgeGraph(nodes=[], edges=[])

    log.info("Loading NeuroSynth coordinates from %s…", files["coordinates"].name)
    coords_df = pd.read_csv(files["coordinates"], sep="\t", compression="gzip", low_memory=False)
    # Column is 'id' (integer PMID-like study key), not 'study_id'
    log.info("  %d coordinate rows, %d studies", len(coords_df), coords_df["id"].nunique())

    log.info("Loading NeuroSynth metadata from %s…", files["metadata"].name)
    meta_df = pd.read_csv(files["metadata"], sep="\t", compression="gzip", low_memory=False)
    log.info("  %d studies in metadata", len(meta_df))

    log.info("Loading vocabulary from %s…", files["vocabulary"].name)
    vocabulary = files["vocabulary"].read_text(encoding="utf-8").splitlines()
    vocabulary = [t.strip() for t in vocabulary if t.strip()]
    log.info("  %d terms in vocabulary", len(vocabulary))

    log.info("Loading tfidf feature matrix from %s…", files["features"].name)
    feat = scipy.sparse.load_npz(str(files["features"]))
    log.info("  Feature matrix: %s (studies x terms)", feat.shape)

    # Row order in metadata == row order in feature matrix (NeuroSynth convention).
    # The study key column is 'id' (integer).
    study_ids = meta_df["id"].astype(str).tolist()

    # Build reverse index: study_id -> row index in feature matrix
    study_to_row: dict[str, int] = {sid: i for i, sid in enumerate(study_ids)}

    # Pre-compute: for each coordinate, which MNI region?
    log.info("Mapping %d coordinates to regions…", len(coords_df))
    coords_df = coords_df.copy()
    coords_df["region"] = coords_df.apply(
        lambda r: _mni_to_region_label(
            float(r["x"] if pd.notna(r["x"]) else 0),
            float(r["y"] if pd.notna(r["y"]) else 0),
            float(r["z"] if pd.notna(r["z"]) else 0),
        ),
        axis=1,
    )
    # Drop unmapped coordinates
    coords_df = coords_df[coords_df["region"].notna()]
    log.info("  %d coordinates mapped to regions", len(coords_df))

    # Group coordinates by study id -> {region: count}
    study_region_counts: dict[str, dict[str, int]] = {}
    for sid, grp in coords_df.groupby("id"):
        study_region_counts[str(sid)] = grp["region"].value_counts().to_dict()

    # Now iterate over vocabulary terms and build edges
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_topic_nodes: set[str] = set()

    feat_csc = feat.tocsc()  # column-major for fast term column access

    log.info("Building topic->region edges for %d terms…", len(vocabulary))
    for term_idx, term in enumerate(vocabulary):
        if term_idx % 100 == 0:
            log.info("  Processing term %d / %d…", term_idx, len(vocabulary))

        # Get studies where this term has weight > threshold
        col = feat_csc.getcol(term_idx)
        row_indices = col.nonzero()[0]
        weights = col.data

        # Filter by tfidf threshold
        high_indices = [r for r, w in zip(row_indices, weights) if w >= TFIDF_THRESHOLD]
        if len(high_indices) < MIN_STUDY_COUNT:
            continue

        high_study_ids = {str(study_ids[r]) for r in high_indices if r < len(study_ids)}

        # Aggregate region counts across all studies with this term
        region_counts: dict[str, int] = {}
        for sid in high_study_ids:
            for region, cnt in study_region_counts.get(sid, {}).items():
                region_counts[region] = region_counts.get(region, 0) + cnt

        if not region_counts:
            continue

        total = sum(region_counts.values())
        topic_node_id = _term_to_topic_id(term)

        # Create topic/concept node if new
        if topic_node_id not in seen_topic_nodes:
            seen_topic_nodes.add(topic_node_id)
            ntype = "topic" if topic_node_id.startswith("topic:") else "concept"
            nodes.append(
                GraphNode(
                    node_id=topic_node_id,
                    node_type=ntype,
                    label=term.replace("_", " ").title(),
                    properties={
                        "neurosynth_term": term,
                        "source": "neurosynth_v7",
                        "n_studies": len(high_study_ids),
                    },
                )
            )

        for region_label, count in region_counts.items():
            if count < MIN_REGION_ACTIVATIONS:
                continue
            freq = count / max(total, 1)
            if freq < MIN_ACTIVATION_FREQ:
                continue

            region_node_id = _region_label_to_node_id(region_label)
            safe_region = region_label.replace(" ", "_")
            safe_term = term.replace(" ", "_")

            edges.append(
                GraphEdge(
                    edge_id=f"edge:ns:{safe_term}:activates:{safe_region}",
                    source_node_id=topic_node_id,
                    target_node_id=region_node_id,
                    edge_type="topic_activates_region",
                    confidence=min(freq * 5.0, 1.0),
                    properties={
                        "neurosynth_term": term,
                        "n_studies": len(high_study_ids),
                        "n_activations": count,
                        "activation_frequency": round(freq, 4),
                        "source": "neurosynth_v7",
                        "inference_type": "forward",
                    },
                )
            )
            edges.append(
                GraphEdge(
                    edge_id=f"edge:ns:{safe_region}:implicated:{safe_term}",
                    source_node_id=region_node_id,
                    target_node_id=topic_node_id,
                    edge_type="region_implicated_in_topic",
                    confidence=min(freq * 4.0, 1.0),
                    properties={
                        "neurosynth_term": term,
                        "activation_frequency": round(freq, 4),
                        "source": "neurosynth_v7",
                    },
                )
            )

    fwd = sum(1 for e in edges if e.edge_type == "topic_activates_region")
    rev = sum(1 for e in edges if e.edge_type == "region_implicated_in_topic")
    log.info("NeuroSynth KG: %d nodes, %d edges (forward=%d, reverse=%d)", len(nodes), len(edges), fwd, rev)
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_neurosynth_kg()
    print(f"NeuroSynth KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    fwd = [e for e in kg.edges.values() if e.edge_type == "topic_activates_region"]
    if fwd:
        print("Sample forward edges:")
        for e in list(fwd)[:5]:
            term = e.properties.get("neurosynth_term", "?")
            region = e.target_node_id.split(":")[-1]
            print(f"  [{term}] -> {region}  (freq={e.properties.get('activation_frequency', '?')}, n={e.properties.get('n_studies', '?')})")
