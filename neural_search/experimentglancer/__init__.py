"""ExperimentGlancer bridge: compiles Neural Search results into portable,
shareable multimodal visualization scenes.

Neural Search remains the intelligence/retrieval layer. ExperimentGlancer
owns rendering. This package only produces the versioned scene contract
(``ExperimentGlancerSceneV1``) that an independent viewer can consume.
"""

from neural_search.experimentglancer.anchors import select_scene_anchors
from neural_search.experimentglancer.layer_planner import (
    LayerPlanResult,
    plan_layers_for_result,
)
from neural_search.experimentglancer.scene_builder import build_scene
from neural_search.experimentglancer.schemas import (
    CoordinateSpace,
    DatasetIntrospectionV1,
    DefaultWindow,
    ExperimentGlancerSceneV1,
    LayerAlignment,
    LayerDataRef,
    LayerDisplay,
    LayerProvenance,
    QueryContext,
    SceneAnchor,
    SceneDatasetRef,
    SceneLayer,
    SceneLayout,
    SceneProvenance,
    SceneSource,
)
from neural_search.experimentglancer.source_resolvers import (
    resolve_dandi_nwb,
    resolve_dataset_introspection,
    resolve_metadata_only,
    resolve_openneuro_bids_local,
)

__all__ = [
    "CoordinateSpace",
    "DatasetIntrospectionV1",
    "DefaultWindow",
    "ExperimentGlancerSceneV1",
    "LayerAlignment",
    "LayerDataRef",
    "LayerDisplay",
    "LayerPlanResult",
    "LayerProvenance",
    "QueryContext",
    "SceneAnchor",
    "SceneDatasetRef",
    "SceneLayer",
    "SceneLayout",
    "SceneProvenance",
    "SceneSource",
    "build_scene",
    "plan_layers_for_result",
    "resolve_dandi_nwb",
    "resolve_dataset_introspection",
    "resolve_metadata_only",
    "resolve_openneuro_bids_local",
    "select_scene_anchors",
]
