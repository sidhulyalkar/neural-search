"""Adapters that project software provenance records into the Neural Search KG."""

from __future__ import annotations

from neural_search.graph.schema import (
    SUPPORTED_EDGE_TYPES,
    SUPPORTED_NODE_TYPES,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)
from neural_search.software.schema import (
    AuditFinding,
    AuditHypothesis,
    MaintainerDecision,
    SoftwareComponent,
    SoftwarePackage,
    SoftwareRelease,
    VerificationRun,
)

SOFTWARE_NODE_TYPES = {
    "software_package",
    "software_release",
    "software_component",
    "audit_hypothesis",
    "verification_run",
    "software_audit_finding",
    "maintainer_decision",
}

SOFTWARE_EDGE_TYPES = {
    "software_package_has_release",
    "software_package_has_component",
    "software_release_contains_component",
    "paper_uses_software",
    "paper_uses_software_component",
    "software_component_implements_method",
    "audit_hypothesis_concerns_component",
    "software_audit_finding_supported_by_verification",
    "software_audit_finding_affects_release",
    "paper_exposed_to_software_finding",
    "dataset_reanalysis_candidate_for_software_finding",
    "maintainer_decision_adjudicates_finding",
}


def register_software_graph_schema() -> None:
    """Register the additive software vocabulary with the shared KG schema sets."""

    SUPPORTED_NODE_TYPES.update(SOFTWARE_NODE_TYPES)
    SUPPORTED_EDGE_TYPES.update(SOFTWARE_EDGE_TYPES)


register_software_graph_schema()


def package_node(record: SoftwarePackage) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("software_package", record.package_id),
        node_type="software_package",
        label=record.name,
        source_ids=[record.repository_url],
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
    )


def release_node(record: SoftwareRelease) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("software_release", record.release_id),
        node_type="software_release",
        label=f"{record.package_id} {record.version}",
        source_ids=[record.commit_sha] if record.commit_sha else [],
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
    )


def component_node(record: SoftwareComponent) -> KnowledgeGraphNode:
    label = record.path if not record.symbol else f"{record.path}:{record.symbol}"
    return KnowledgeGraphNode(
        node_id=make_node_id("software_component", record.component_id),
        node_type="software_component",
        label=label,
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
    )


def hypothesis_node(record: AuditHypothesis) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("audit_hypothesis", record.hypothesis_id),
        node_type="audit_hypothesis",
        label=record.summary,
        source_ids=record.source_run_ids,
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
        confidence=float(record.metadata.get("confidence", 0.5)),
    )


def verification_node(record: VerificationRun) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("verification_run", record.verification_id),
        node_type="verification_run",
        label=f"{record.level.value}: {record.verification_id}",
        source_ids=record.input_artifact_ids + record.output_artifact_ids,
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
        confidence=1.0 if record.passed else 0.5,
    )


def finding_node(record: AuditFinding) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("software_audit_finding", record.finding_id),
        node_type="software_audit_finding",
        label=record.summary,
        source_ids=record.verification_ids,
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
    )


def maintainer_decision_node(record: MaintainerDecision) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("maintainer_decision", record.decision_id),
        node_type="maintainer_decision",
        label=record.disposition,
        source_ids=[record.issue_or_pr_url] if record.issue_or_pr_url else [],
        properties=record.model_dump(mode="json", exclude={"metadata"}) | record.metadata,
    )


def relation(
    source: KnowledgeGraphNode,
    edge_type: str,
    target: KnowledgeGraphNode,
    **properties: object,
) -> KnowledgeGraphEdge:
    """Create a stable provenance relationship between software graph nodes."""

    if edge_type not in SOFTWARE_EDGE_TYPES:
        raise ValueError(f"unsupported software graph edge type: {edge_type}")
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source.node_id, edge_type, target.node_id),
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=edge_type,
        properties=dict(properties),
    )
