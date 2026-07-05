"""Build the methods/math KG layer from data/methods/methods_taxonomy.yaml."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.schema import GraphEdge, GraphNode, KnowledgeGraph

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "methods"


def _load_taxonomy() -> dict[str, Any]:
    path = DATA_DIR / "methods_taxonomy.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _method_node_id(method_id: str) -> str:
    return f"method:{method_id}"


def _concept_node_id(concept_id: str) -> str:
    return f"concept:{concept_id}"


def build_methods_nodes(taxonomy: dict[str, Any]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    seen_quantities: set[str] = set()
    seen_topics: set[str] = set()

    for category in taxonomy.get("categories", []):
        cat_id = category["id"]
        cat_label = category["label"]

        # Category node as a concept
        cat_node_id = f"concept:method_category:{cat_id}"
        nodes.append(
            GraphNode(
                node_id=cat_node_id,
                node_type="concept",
                label=cat_label,
                properties={
                    "description": category.get("description", ""),
                    "category_type": "method_category",
                },
            )
        )

        for item in category.get("methods", category.get("concepts", [])):
            item_id = item["id"]
            node_id = _method_node_id(item_id)
            props: dict[str, Any] = {
                "category": cat_id,
                "category_label": cat_label,
                "aliases": item.get("aliases", []),
                "topics": item.get("topics", []),
                "key_papers": item.get("key_papers", []),
            }

            if "formula" in item:
                props["formula"] = item["formula"]
            if "computes" in item:
                props["computes"] = item["computes"]
            if "assumptions" in item:
                props["assumptions"] = item["assumptions"]
            if "limitations" in item:
                props["limitations"] = item["limitations"]
            if "mathematical_basis" in item:
                props["mathematical_basis"] = item["mathematical_basis"]
            if "principle" in item:
                props["principle"] = item["principle"]
            if "neural_relevance" in item:
                props["neural_relevance"] = item["neural_relevance"]
            if "advantages_over_fft" in item:
                props["advantages"] = item["advantages_over_fft"]
            if "advantages_over_granger" in item:
                props["advantages"] = item["advantages_over_granger"]
            if "software" in item:
                props["software"] = item["software"]
            if "circuit" in item:
                props["circuit"] = item["circuit"]
            if "use_cases" in item:
                props["use_cases"] = item["use_cases"]
            if "neuroscience_use" in item:
                props["neuroscience_use"] = item["neuroscience_use"]
            if "clinical" in item:
                props["clinical"] = item["clinical"]
            if "species_note" in item:
                props["species_note"] = item["species_note"]
            if "why_better_than_mean_power" in item:
                props["why_it_matters"] = item["why_better_than_mean_power"]
            if "relevance" in item:
                props["relevance"] = item["relevance"]

            nodes.append(
                GraphNode(
                    node_id=node_id,
                    node_type="method",
                    label=item.get("label", item_id),
                    properties=props,
                )
            )

            # Target nodes for method_computes/method_used_for_topic edges
            # (build_methods_edges emits these edges; without the target
            # nodes existing here too, they dangle when this builder is
            # merged into a graph that doesn't separately create topic/
            # concept:quantity nodes, e.g. scripts/build_real_corpus_graph.py).
            for quantity in item.get("computes", []):
                quantity_node_id = f"concept:quantity:{quantity}"
                if quantity_node_id not in seen_quantities:
                    seen_quantities.add(quantity_node_id)
                    nodes.append(
                        GraphNode(
                            node_id=quantity_node_id,
                            node_type="concept",
                            label=quantity.replace("_", " ").title(),
                            properties={"category_type": "computed_quantity"},
                        )
                    )

            for topic_id in item.get("topics", []):
                topic_node_id = f"topic:{topic_id}"
                if topic_node_id not in seen_topics:
                    seen_topics.add(topic_node_id)
                    nodes.append(
                        GraphNode(
                            node_id=topic_node_id,
                            node_type="topic",
                            label=topic_id.replace("_", " ").title(),
                            properties={"source": "methods_taxonomy_reference"},
                        )
                    )

            # Target nodes for method_assumes edges (same dangling-edge
            # reasoning as computes/topics above).
            if isinstance(item.get("assumptions"), dict):
                for assump_key in item["assumptions"]:
                    assump_node_id = f"concept:assumption:{item_id}:{assump_key}"
                    nodes.append(
                        GraphNode(
                            node_id=assump_node_id,
                            node_type="concept",
                            label=assump_key.replace("_", " ").title(),
                            properties={"category_type": "method_assumption", "method": item_id},
                        )
                    )

    return nodes


def known_method_ids(taxonomy: dict[str, Any]) -> set[str]:
    """All method/concept ids declared anywhere in the taxonomy."""

    return {
        item["id"]
        for category in taxonomy.get("categories", [])
        for item in category.get("methods", category.get("concepts", []))
    }


def build_methods_edges(taxonomy: dict[str, Any]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    known_ids = known_method_ids(taxonomy)

    for category in taxonomy.get("categories", []):
        cat_id = category["id"]
        cat_node_id = f"concept:method_category:{cat_id}"

        for item in category.get("methods", category.get("concepts", [])):
            item_id = item["id"]
            node_id = _method_node_id(item_id)

            # belongs_to_category
            edges.append(
                GraphEdge(
                    edge_id=f"edge:method:{item_id}:belongs_to:{cat_id}",
                    source_node_id=node_id,
                    target_node_id=cat_node_id,
                    edge_type="concept_related_to_concept",
                    properties={"relation": "belongs_to_category"},
                )
            )

            # related_methods — skip references to ids not defined anywhere
            # in the taxonomy (honest gap, e.g. "ccm"/"kalman_filter" are
            # mentioned as related but have no entry of their own yet) rather
            # than emitting a dangling edge to a node that doesn't exist.
            for related_id in item.get("related_methods", []):
                if related_id not in known_ids:
                    log.debug(
                        "Skipping method_related_to_method %s -> %s: %s has no taxonomy entry",
                        item_id,
                        related_id,
                        related_id,
                    )
                    continue
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:method:{item_id}:related:{related_id}",
                        source_node_id=node_id,
                        target_node_id=_method_node_id(related_id),
                        edge_type="method_related_to_method",
                        directed=False,
                    )
                )

            # topic edges
            for topic_id in item.get("topics", []):
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:method:{item_id}:used_for:{topic_id}",
                        source_node_id=node_id,
                        target_node_id=f"topic:{topic_id}",
                        edge_type="method_used_for_topic",
                    )
                )

            # computes edges (to a concept node representing the computed quantity)
            for quantity in item.get("computes", []):
                qty_node_id = f"concept:quantity:{quantity}"
                edges.append(
                    GraphEdge(
                        edge_id=f"edge:method:{item_id}:computes:{quantity}",
                        source_node_id=node_id,
                        target_node_id=qty_node_id,
                        edge_type="method_computes",
                    )
                )

            # assumption edges
            if isinstance(item.get("assumptions"), dict):
                for assump_key, assump_text in item["assumptions"].items():
                    assump_node_id = f"concept:assumption:{item_id}:{assump_key}"
                    edges.append(
                        GraphEdge(
                            edge_id=f"edge:method:{item_id}:assumes:{assump_key}",
                            source_node_id=node_id,
                            target_node_id=assump_node_id,
                            edge_type="method_assumes",
                            properties={"assumption_key": assump_key, "description": assump_text},
                        )
                    )

    return edges


def build_methods_kg() -> KnowledgeGraph:
    taxonomy = _load_taxonomy()
    nodes = build_methods_nodes(taxonomy)
    edges = build_methods_edges(taxonomy)
    log.info("Methods KG: %d nodes, %d edges", len(nodes), len(edges))
    return KnowledgeGraph(nodes=nodes, edges=edges)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = build_methods_kg()
    print(f"Methods KG built: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    method_nodes = [n for n in kg.nodes if n.node_type == "method"]
    print(f"  Method nodes: {len(method_nodes)}")
    for n in method_nodes[:5]:
        print(f"  - {n.label} ({n.node_id})")
