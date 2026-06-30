"""Literature NER → KG builder.

Scans the existing papers in the KG and uses scispaCy NER to extract:
  - Brain region mentions → paper_mentions_region edges
  - Disorder mentions → paper_involves_disorder edges
  - Method mentions → paper_uses_method edges

Data sources:
  - Existing paper nodes from neural_search.ingestion.paper_builder (or equivalent)
  - papers metadata from data/papers/ or via API

Requires: pip install scispacy
          pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from neural_search.graph.schema import GraphNode, GraphEdge, KnowledgeGraph

log = logging.getLogger(__name__)

# ── Region vocabulary (for entity linking) ───────────────────────────────────
# Maps surface forms (lowercased) → our ontology_region IDs
REGION_SURFACE_TO_ID: dict[str, str] = {
    # Hippocampus
    "hippocampus": "hippocampus",
    "hippocampal": "hippocampus",
    "ca1": "hippocampus",
    "ca3": "hippocampus",
    "dentate gyrus": "hippocampus",
    "dg": "hippocampus",
    # PFC
    "prefrontal cortex": "medial_prefrontal_cortex",
    "prefrontal": "medial_prefrontal_cortex",
    "pfc": "medial_prefrontal_cortex",
    "mPFC": "medial_prefrontal_cortex",
    "medial prefrontal": "medial_prefrontal_cortex",
    "prelimbic": "medial_prefrontal_cortex",
    "infralimbic": "medial_prefrontal_cortex",
    "dlpfc": "dorsolateral_prefrontal_cortex",
    "dorsolateral prefrontal": "dorsolateral_prefrontal_cortex",
    # Cingulate
    "anterior cingulate": "anterior_cingulate_cortex",
    "acc": "anterior_cingulate_cortex",
    "cingulate cortex": "anterior_cingulate_cortex",
    "posterior cingulate": "posterior_cingulate_cortex",
    "pcc": "posterior_cingulate_cortex",
    # Amygdala
    "amygdala": "amygdala_basolateral",
    "amygdaloid": "amygdala_basolateral",
    "bla": "amygdala_basolateral",
    "basolateral amygdala": "amygdala_basolateral",
    "central amygdala": "central_amygdala",
    "cea": "central_amygdala",
    # Striatum
    "striatum": "dorsal_striatum",
    "striatal": "dorsal_striatum",
    "caudate": "dorsal_striatum",
    "putamen": "dorsal_striatum",
    "caudate putamen": "dorsal_striatum",
    "nucleus accumbens": "nucleus_accumbens",
    "nac": "nucleus_accumbens",
    "accumbens": "nucleus_accumbens",
    "ventral striatum": "nucleus_accumbens",
    # Thalamus
    "thalamus": "mediodorsal_thalamus",
    "thalamic": "mediodorsal_thalamus",
    "mediodorsal thalamus": "mediodorsal_thalamus",
    "md thalamus": "mediodorsal_thalamus",
    # Dopamine
    "vta": "vta",
    "ventral tegmental": "vta",
    "ventral tegmental area": "vta",
    "substantia nigra": "substantia_nigra_compacta",
    "snc": "substantia_nigra_compacta",
    # Subthalamic
    "subthalamic nucleus": "subthalamic_nucleus",
    "stn": "subthalamic_nucleus",
    # Cerebellum
    "cerebellum": "cerebellar_cortex",
    "cerebellar": "cerebellar_cortex",
    # OFC
    "orbitofrontal cortex": "orbitofrontal_cortex",
    "ofc": "orbitofrontal_cortex",
    "orbitofrontal": "orbitofrontal_cortex",
    # Cortex
    "motor cortex": "primary_motor_cortex",
    "m1": "primary_motor_cortex",
    "somatosensory cortex": "somatosensory_cortex",
    "s1": "somatosensory_cortex",
    "visual cortex": "visual_cortex",
    "v1": "visual_cortex",
    "auditory cortex": "auditory_cortex",
    "insula": "insular_cortex",
    "insular cortex": "insular_cortex",
    "entorhinal cortex": "entorhinal_cortex",
    "entorhinal": "entorhinal_cortex",
    # Habenula
    "habenula": "lateral_habenula",
    "lateral habenula": "lateral_habenula",
    "lhb": "lateral_habenula",
    # Raphe / LC
    "dorsal raphe": "dorsal_raphe",
    "raphe": "dorsal_raphe",
    "locus coeruleus": "locus_coeruleus",
    "lc": "locus_coeruleus",
    # PAG
    "periaqueductal gray": "periaqueductal_gray",
    "pag": "periaqueductal_gray",
    # Hypothalamus
    "hypothalamus": "hypothalamus",
    "lateral hypothalamus": "lateral_hypothalamus",
    # Olfactory
    "olfactory bulb": "olfactory_bulb",
    "ob": "olfactory_bulb",
    "piriform": "piriform_cortex",
    "piriform cortex": "piriform_cortex",
}

# Disorder vocabulary
DISORDER_SURFACE_TO_ID: dict[str, str] = {
    "schizophrenia": "schizophrenia",
    "schizophrenic": "schizophrenia",
    "depression": "major_depressive_disorder",
    "depressive disorder": "major_depressive_disorder",
    "mdd": "major_depressive_disorder",
    "bipolar": "bipolar_disorder",
    "bipolar disorder": "bipolar_disorder",
    "anxiety": "anxiety_disorder",
    "anxiety disorder": "anxiety_disorder",
    "ptsd": "ptsd",
    "post-traumatic stress": "ptsd",
    "ocd": "ocd",
    "obsessive-compulsive": "ocd",
    "adhd": "adhd",
    "attention deficit": "adhd",
    "autism": "autism_spectrum_disorder",
    "asd": "autism_spectrum_disorder",
    "autistic": "autism_spectrum_disorder",
    "parkinson": "parkinsons_disease",
    "parkinson's": "parkinsons_disease",
    "parkinsonian": "parkinsons_disease",
    "alzheimer": "alzheimers_disease",
    "alzheimer's": "alzheimers_disease",
    "dementia": "alzheimers_disease",
    "epilepsy": "epilepsy",
    "epileptic": "epilepsy",
    "seizure": "epilepsy",
    "huntington": "huntingtons_disease",
    "addiction": "substance_use_disorder",
    "substance use": "substance_use_disorder",
    "stroke": "stroke",
    "tbi": "tbi",
    "traumatic brain injury": "tbi",
    "chronic pain": "chronic_pain",
    "neuropathic pain": "chronic_pain",
    "tourette": "tourette_syndrome",
    "als": "als",
    "amyotrophic lateral sclerosis": "als",
    "multiple sclerosis": "multiple_sclerosis",
    "ms": "multiple_sclerosis",
    "fragile x": "fragile_x",
    "rett syndrome": "rett_syndrome",
}

# Analysis method vocabulary
METHOD_SURFACE_TO_ID: dict[str, str] = {
    "fft": "fft",
    "fast fourier transform": "fft",
    "power spectral density": "fft",
    "pac": "pac",
    "phase-amplitude coupling": "pac",
    "phase amplitude coupling": "pac",
    "granger causality": "granger_causality",
    "granger": "granger_causality",
    "coherence": "coherence",
    "spike sorting": "spike_sorting",
    "ica": "ica",
    "independent component analysis": "ica",
    "pca": "pca",
    "principal component analysis": "pca",
    "fooof": "fooof",
    "specparam": "fooof",
    "dcm": "dcm_spectral",
    "dynamic causal modelling": "dcm_spectral",
    "dynamic causal modeling": "dcm_spectral",
    "calcium imaging": "calcium_imaging_analysis",
    "glm": "glm",
    "general linear model": "glm",
    "rsa": "rsa",
    "representational similarity": "rsa",
    "decoding": "decoding",
    "svm": "decoding",
    "support vector machine": "decoding",
    "dimensionality reduction": "dimensionality_reduction",
    "umap": "dimensionality_reduction",
    "t-sne": "dimensionality_reduction",
    "tsne": "dimensionality_reduction",
    "lfads": "lfads",
    "cross-frequency coupling": "pac",
    "theta-gamma coupling": "pac",
    "fiber photometry": "fiber_photometry",
    "optogenetics": "optogenetics",
    "two-photon": "calcium_imaging_analysis",
    "2-photon": "calcium_imaging_analysis",
    "electrocorticography": "ecog_analysis",
    "ecog": "ecog_analysis",
    "seeg": "ecog_analysis",
    "mri": "structural_mri_analysis",
    "fmri": "fmri_analysis",
    "dti": "dti_tractography",
    "eeg": "eeg_analysis",
    "meg": "meg_source_localization",
}


def _load_nlp_model():
    """Load scispaCy model. Returns None if not installed."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_lg")
            return nlp
        except OSError:
            log.warning(
                "scispaCy model not found. Install with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )
            return None
    except ImportError:
        log.warning("spacy not installed. Run: pip install scispacy")
        return None


def _regex_entity_search(text: str) -> dict[str, list[str]]:
    """Fallback entity extraction using vocabulary regex when scispaCy unavailable."""
    text_lower = text.lower()
    results: dict[str, list[str]] = {
        "regions": [],
        "disorders": [],
        "methods": [],
    }
    for surface, region_id in REGION_SURFACE_TO_ID.items():
        if re.search(r"\b" + re.escape(surface) + r"\b", text_lower):
            if region_id not in results["regions"]:
                results["regions"].append(region_id)
    for surface, disorder_id in DISORDER_SURFACE_TO_ID.items():
        if re.search(r"\b" + re.escape(surface) + r"\b", text_lower):
            if disorder_id not in results["disorders"]:
                results["disorders"].append(disorder_id)
    for surface, method_id in METHOD_SURFACE_TO_ID.items():
        if re.search(r"\b" + re.escape(surface) + r"\b", text_lower):
            if method_id not in results["methods"]:
                results["methods"].append(method_id)
    return results


def _link_entities_to_ids(ents: list[str]) -> dict[str, list[str]]:
    """Map raw entity text (from NER) to our vocabulary IDs."""
    results: dict[str, list[str]] = {"regions": [], "disorders": [], "methods": []}
    for ent_text in ents:
        lower = ent_text.lower().strip()
        for surface, region_id in REGION_SURFACE_TO_ID.items():
            if surface in lower and region_id not in results["regions"]:
                results["regions"].append(region_id)
        for surface, disorder_id in DISORDER_SURFACE_TO_ID.items():
            if surface in lower and disorder_id not in results["disorders"]:
                results["disorders"].append(disorder_id)
        for surface, method_id in METHOD_SURFACE_TO_ID.items():
            if surface in lower and method_id not in results["methods"]:
                results["methods"].append(method_id)
    return results


def _find_papers() -> list[dict[str, Any]]:
    """Locate papers from the corpus normalized directory and OpenAlex batches."""
    papers: list[dict[str, Any]] = []
    data_root = Path(__file__).parent.parent.parent / "data"

    # Primary: OpenAlex neuroscience batches (26 files, ~thousands of papers each)
    openalex_dir = data_root / "corpus" / "normalized" / "openalex_neuro"
    if openalex_dir.exists():
        for fp in sorted(openalex_dir.glob("*.jsonl")):
            try:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        papers.append(json.loads(line))
            except Exception as exc:
                log.debug("Skipping %s: %s", fp.name, exc)
        log.info("NER: loaded %d papers from openalex_neuro batches.", len(papers))

    # Secondary: combined_corpus JSONL (curated papers)
    corpus_dir = data_root / "corpus" / "normalized" / "combined_corpus.jsonl"
    if corpus_dir.exists():
        for fp in corpus_dir.glob("*.papers.jsonl"):
            try:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        papers.append(json.loads(line))
            except Exception as exc:
                log.debug("Skipping %s: %s", fp.name, exc)

    # Fallback: legacy data/papers/**/*.json
    if not papers:
        for pattern in ["papers/**/*.json", "literature/**/*.json"]:
            for fp in data_root.glob(pattern):
                try:
                    obj = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(obj, list):
                        papers.extend(obj)
                    elif isinstance(obj, dict) and "title" in obj:
                        papers.append(obj)
                except Exception:
                    pass

    if not papers:
        log.warning(
            "No paper files found. NER builder will produce an empty KG."
        )
    else:
        log.info("NER: %d total papers to process.", len(papers))

    return papers


def build_ner_nodes(papers: list[dict[str, Any]]) -> list[GraphNode]:  # noqa: ARG001
    """Create vocabulary nodes for every method, disorder, and region referenced by edges.

    Paper nodes are assumed to exist in other KG layers (e.g. literature or citation
    builder), so we don't duplicate them here.  But the *target* nodes — methods,
    disorders, and ontology regions — are finite vocabularies that need to be present
    in the graph for edges to resolve; create them here.
    """
    nodes: list[GraphNode] = []

    # Ontology-region nodes
    seen_regions: set[str] = set()
    for region_id in set(REGION_SURFACE_TO_ID.values()):
        if region_id in seen_regions:
            continue
        seen_regions.add(region_id)
        nodes.append(
            GraphNode(
                node_id=f"ontology_region:{region_id}",
                label=region_id.replace("_", " ").title(),
                node_type="ontology_region",
                properties={"source": "ner_vocabulary"},
            )
        )

    # Disorder nodes
    seen_disorders: set[str] = set()
    for disorder_id in set(DISORDER_SURFACE_TO_ID.values()):
        if disorder_id in seen_disorders:
            continue
        seen_disorders.add(disorder_id)
        nodes.append(
            GraphNode(
                node_id=f"disorder:{disorder_id}",
                label=disorder_id.replace("_", " ").title(),
                node_type="disorder",
                properties={"source": "ner_vocabulary"},
            )
        )

    # Method nodes
    seen_methods: set[str] = set()
    for method_id in set(METHOD_SURFACE_TO_ID.values()):
        if method_id in seen_methods:
            continue
        seen_methods.add(method_id)
        nodes.append(
            GraphNode(
                node_id=f"method:{method_id}",
                label=method_id.replace("_", " ").upper()
                if len(method_id) <= 5  # short IDs are acronyms
                else method_id.replace("_", " ").title(),
                node_type="method",
                properties={"source": "ner_vocabulary"},
            )
        )

    log.info(
        "NER nodes: %d ontology_region, %d disorder, %d method",
        len(seen_regions),
        len(seen_disorders),
        len(seen_methods),
    )
    return nodes


def build_ner_edges(papers: list[dict[str, Any]], use_spacy: bool = True) -> list[GraphEdge]:
    nlp = _load_nlp_model() if use_spacy else None
    edges: list[GraphEdge] = []
    processed = 0

    for paper in papers:
        # Support paper:openalex:W... format, bare W IDs, and legacy pmid
        raw_id = (
            paper.get("paper_id")
            or paper.get("pmid")
            or paper.get("id")
            or paper.get("paperId")
            or paper.get("source_id")
        )
        if not raw_id:
            continue
        # Normalise to our standard paper node ID
        if str(raw_id).startswith("paper:"):
            paper_node_id = raw_id
        elif str(raw_id).startswith("W"):
            paper_node_id = f"paper:openalex:{raw_id}"
        else:
            paper_node_id = f"paper:{raw_id}"

        title = paper.get("title", "")
        abstract = paper.get("abstract", paper.get("description", ""))
        text = f"{title}. {abstract}"

        if not text.strip():
            continue

        if nlp is not None:
            doc = nlp(text)
            raw_ents = [ent.text for ent in doc.ents]
            found = _link_entities_to_ids(raw_ents)
        else:
            found = _regex_entity_search(text)

        # Derive a short stable ID for use in edge IDs (no colons or slashes)
        edge_paper_key = paper_node_id.replace(":", "_").replace("/", "_")
        model_name = "en_core_sci_lg" if nlp else "regex_vocabulary"

        # paper_mentions_region edges
        for region_id in found["regions"]:
            region_node_id = f"ontology_region:{region_id}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:ner:{edge_paper_key}:mentions_region:{region_id}",
                    source_node_id=paper_node_id,
                    target_node_id=region_node_id,
                    edge_type="paper_mentions_region",
                    confidence=0.70,
                    properties={"source": "ner_extraction", "model": model_name},
                )
            )

        # paper_involves_disorder edges
        for disorder_id in found["disorders"]:
            disorder_node_id = f"disorder:{disorder_id}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:ner:{edge_paper_key}:involves_disorder:{disorder_id}",
                    source_node_id=paper_node_id,
                    target_node_id=disorder_node_id,
                    edge_type="paper_involves_disorder",
                    confidence=0.75,
                    properties={"source": "ner_extraction", "model": model_name},
                )
            )

        # paper_uses_method edges
        for method_id in found["methods"]:
            method_node_id = f"method:{method_id}"
            edges.append(
                GraphEdge(
                    edge_id=f"edge:ner:{edge_paper_key}:uses_method:{method_id}",
                    source_node_id=paper_node_id,
                    target_node_id=method_node_id,
                    edge_type="paper_uses_method",
                    confidence=0.72,
                    properties={"source": "ner_extraction", "model": model_name},
                )
            )

        processed += 1
        if processed % 100 == 0:
            log.info("NER: processed %d / %d papers…", processed, len(papers))

    log.info(
        "NER: %d edges from %d papers (region=%d, disorder=%d, method=%d)",
        len(edges),
        processed,
        sum(1 for e in edges if e.edge_type == "paper_mentions_region"),
        sum(1 for e in edges if e.edge_type == "paper_involves_disorder"),
        sum(1 for e in edges if e.edge_type == "paper_uses_method"),
    )
    return edges


NER_ARTIFACT_PATH = Path(__file__).parent.parent.parent / "artifacts" / "ner" / "ner_kg.jsonl"


def build_ner_kg(use_spacy: bool = False, max_papers: int = 0) -> KnowledgeGraph:
    papers = _find_papers()
    if max_papers and len(papers) > max_papers:
        log.info("Capping to first %d papers (--max-papers).", max_papers)
        papers = papers[:max_papers]
    nodes = build_ner_nodes(papers)
    edges = build_ner_edges(papers, use_spacy=use_spacy)
    return KnowledgeGraph(nodes=nodes, edges=edges)


def load_cached_ner_kg() -> KnowledgeGraph | None:
    """Load NER KG from artifact cache if it exists."""
    if not NER_ARTIFACT_PATH.exists():
        return None
    from neural_search.graph.schema import read_graph_jsonl
    log.info("Loading NER KG from cache: %s", NER_ARTIFACT_PATH)
    return read_graph_jsonl(NER_ARTIFACT_PATH)


def save_ner_kg(kg: KnowledgeGraph) -> None:
    """Persist NER KG to artifact cache."""
    from neural_search.graph.schema import write_graph_jsonl
    NER_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_graph_jsonl(kg, NER_ARTIFACT_PATH)
    size_mb = NER_ARTIFACT_PATH.stat().st_size / (1024 ** 2)
    log.info("NER KG saved -> %s  (%.1f MB)", NER_ARTIFACT_PATH, size_mb)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy", action="store_true", help="Use scispaCy NER (slower, higher precision)")
    parser.add_argument("--max-papers", type=int, default=0, help="Process at most N papers (0=all)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save output to artifacts/ner/ner_kg.jsonl (default: on)")
    parser.add_argument("--no-save", action="store_false", dest="save")
    args = parser.parse_args()

    kg = build_ner_kg(use_spacy=args.spacy, max_papers=args.max_papers)
    print(f"NER KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    if kg.edges:
        for e in list(kg.edges.values())[:8]:
            print(f"  [{e.edge_type}] {e.source_node_id} -> {e.target_node_id}")

    if args.save:
        save_ner_kg(kg)
        print(f"Saved -> {NER_ARTIFACT_PATH}")
