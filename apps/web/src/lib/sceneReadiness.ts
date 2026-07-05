import type { SearchResultItem } from '../types'

export type Readiness = 'file-verified' | 'inferred' | 'missing' | 'can generate'

export interface ReadinessItem {
  label: string
  readiness: Readiness
}

const CALCIUM_KEYWORDS = ['calcium', 'two-photon', 'two_photon', 'ophys']
const EPHYS_KEYWORDS = ['electrophysiology', 'ephys', 'neuropixels', 'extracellular']
const VIDEO_KEYWORDS = ['video', 'behavior_video']
const POSE_KEYWORDS = ['pose', 'deeplabcut', 'kinematics']

function hasAny(haystack: string[], keywords: string[]): boolean {
  return haystack.some((token) => keywords.some((keyword) => token.includes(keyword)))
}

/** Cheap, client-side approximation of what `plan_layers_for_result` would
 * report -- a preview only, so it's allowed to be wrong in the optimistic
 * direction (upgraded to the real evidence tier once a scene is generated).
 * This exists so a result card can show "what would I see" before the user
 * pays for a scene-generation round trip. */
export function deriveReadinessItems(result: SearchResultItem): ReadinessItem[] {
  const { dataset, evidence_packet } = result
  const modalities = (dataset.modalities || []).map((value) => value.toLowerCase())
  const behaviors = (dataset.behaviors || []).map((value) => value.toLowerCase())
  const tokens = [...modalities, ...behaviors, ...(result.matched_terms || []).map((t) => t.toLowerCase())]

  const hasNeural = hasAny(tokens, CALCIUM_KEYWORDS) || hasAny(tokens, EPHYS_KEYWORDS)
  const hasTrials = behaviors.length > 0 || dataset.tasks.length > 0
  const hasVideo = hasAny(tokens, VIDEO_KEYWORDS) || hasAny(tokens, POSE_KEYWORDS)
  const hasRawData = evidence_packet?.has_raw_data
  const affordanceRequestsModel = (evidence_packet?.expected_analysis_affordances || []).some((id) =>
    ['choice_decoding', 'q_learning', 'latent_dynamics_modeling', 'motor_decoding'].includes(id),
  )

  return [
    {
      label: 'Trials',
      readiness: hasTrials ? 'inferred' : 'missing',
    },
    {
      label: 'Neural',
      readiness: hasRawData === true && hasNeural ? 'file-verified' : hasNeural ? 'inferred' : 'missing',
    },
    {
      label: 'Video',
      readiness: hasVideo ? 'inferred' : 'missing',
    },
    {
      label: 'Model',
      readiness: affordanceRequestsModel ? 'can generate' : 'missing',
    },
  ]
}
