import type { SearchResultItem } from '../types'

// Affordance registry ids (neural_search/affordances/registry.py) that imply
// a scene should surface model output layers when matched.
const MODEL_AFFORDANCE_IDS = new Set([
  'choice_decoding',
  'q_learning',
  'latent_dynamics_modeling',
  'motor_decoding',
  'bci_decoding',
  'speech_decoding',
  'behavioral_state_decoding',
])

const VIDEO_KEYWORDS = ['video', 'pose', 'behavior_video', 'deeplabcut', 'kinematics']
const MODEL_KEYWORDS = ['model', 'decod', 'latent', 'embedding']

function collectAffordanceIds(result: SearchResultItem): string[] {
  const ids = new Set<string>()
  for (const id of result.memory_graph_evidence?.affordance_matches ?? []) {
    ids.add(id)
  }
  for (const match of result.evidence_packet?.affordance_matches ?? []) {
    if (match.matched && match.affordance) ids.add(match.affordance)
  }
  return Array.from(ids)
}

function collectRequestedLayers(
  result: SearchResultItem,
  queryText: string,
  affordanceIds: string[],
): string[] {
  const layers = new Set<string>()
  const haystack = [
    queryText,
    ...(result.matched_terms ?? []),
    ...(result.evidence_packet?.expected_analysis_affordances ?? []),
  ]
    .join(' ')
    .toLowerCase()

  if (VIDEO_KEYWORDS.some((keyword) => haystack.includes(keyword))) {
    layers.add('video.frames')
  }

  const modelRequested =
    affordanceIds.some((id) => MODEL_AFFORDANCE_IDS.has(id)) ||
    MODEL_KEYWORDS.some((keyword) => haystack.includes(keyword))
  if (modelRequested) {
    layers.add('model.predictions')
    layers.add('model.latent_state')
  }

  return Array.from(layers)
}

export interface SceneRequestContext {
  requested_layers: string[]
  affordance_ids: string[]
}

/** Derive the scene-generation hints (requested layers, matched affordances)
 * that the search result already carries as evidence, so the scene the user
 * opens reflects what retrieval actually matched rather than just the
 * dataset id and raw query string. */
export function buildSceneRequestContext(
  result: SearchResultItem,
  queryText: string,
): SceneRequestContext {
  const affordanceIds = collectAffordanceIds(result)
  return {
    requested_layers: collectRequestedLayers(result, queryText, affordanceIds),
    affordance_ids: affordanceIds,
  }
}
