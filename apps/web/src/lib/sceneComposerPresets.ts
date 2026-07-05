import type { AnchorHint } from '../api/experimentglancer'

export interface OpeningModeOption {
  id: string
  label: string
  description: string
}

export const OPENING_MODES: OpeningModeOption[] = [
  { id: 'overview', label: 'Overview', description: 'Let dataset structure decide the anchor (default, fastest).' },
  { id: 'event_aligned', label: 'Event-aligned', description: 'Center on a specific behavioral event you pick below.' },
  { id: 'trial_failures', label: 'Trial failures', description: 'Anchor on trial outcome, for reviewing failure cases.' },
  { id: 'model_errors', label: 'Model errors', description: 'Pull in model prediction/latent layers, anchored on trial outcome.' },
  { id: 'behavior_neural', label: 'Behavior–neural alignment', description: 'Anchor for pose/neural correlation, pull in pose layer.' },
]

export interface EventTypeOption {
  id: string
  label: string
}

export const EVENT_TYPE_OPTIONS: EventTypeOption[] = [
  { id: 'lick_onset', label: 'Lick onset' },
  { id: 'reward_omission', label: 'Reward omission' },
  { id: 'stimulus_onset', label: 'Stimulus onset' },
  { id: 'trial_outcome', label: 'Trial outcome' },
  { id: 'movement_onset', label: 'Movement onset' },
]

export interface SceneComposerOptions {
  openingMode: string
  eventType: string
  includeProbableLayers: boolean
  includeModelOverlays: boolean
  deepIntrospection: boolean
}

export const DEFAULT_COMPOSER_OPTIONS: SceneComposerOptions = {
  openingMode: 'overview',
  eventType: 'lick_onset',
  includeProbableLayers: true,
  includeModelOverlays: false,
  deepIntrospection: false,
}

export interface ComposerPreset {
  anchorHint?: AnchorHint
  extraAffordanceIds: string[]
  extraRequestedLayers: string[]
}

/** Maps a composer's opening mode (+ event type, for event-aligned) to the
 * anchor_hint/affordance_ids/requested_layers the backend scene builder
 * already understands -- see neural_search/experimentglancer/anchors.py and
 * layer_planner.py. No backend changes needed; this only picks from what
 * the contract already supports. */
export function applyOpeningModePreset(options: SceneComposerOptions): ComposerPreset {
  switch (options.openingMode) {
    case 'event_aligned':
      return {
        anchorHint: { kind: 'event', event_type: options.eventType },
        extraAffordanceIds: ['event_aligned_psth'],
        extraRequestedLayers: [],
      }
    case 'trial_failures':
      return {
        anchorHint: { kind: 'trial', event_type: 'trial_outcome' },
        extraAffordanceIds: ['choice_decoding'],
        extraRequestedLayers: [],
      }
    case 'model_errors':
      return {
        anchorHint: { kind: 'trial', event_type: 'trial_outcome' },
        extraAffordanceIds: ['choice_decoding', 'q_learning'],
        extraRequestedLayers: ['model.predictions', 'model.latent_state'],
      }
    case 'behavior_neural':
      return {
        anchorHint: { kind: 'event', event_type: 'movement_onset' },
        extraAffordanceIds: ['pose_neural_correlation'],
        extraRequestedLayers: ['behavior.pose'],
      }
    case 'overview':
    default:
      return { extraAffordanceIds: [], extraRequestedLayers: [] }
  }
}
