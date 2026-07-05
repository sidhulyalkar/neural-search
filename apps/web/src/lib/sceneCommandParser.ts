import type { ExperimentGlancerAnchor, ExperimentGlancerLayer } from '../api/experimentglancer'

// A small, honest command grammar -- not real NLP. Every recognized phrase
// maps to one deterministic action; anything else returns a helpful miss
// message rather than silently doing nothing.
export type SceneCommandAction =
  | { type: 'set_layer_visibility'; layerIds: string[]; visible: boolean; label: string }
  | { type: 'jump_to_anchor'; anchor: ExperimentGlancerAnchor }
  | { type: 'offset_from_anchor'; anchor: ExperimentGlancerAnchor; seconds: number }
  | { type: 'reset_view' }

export interface SceneCommandResult {
  ok: boolean
  message: string
  action?: SceneCommandAction
}

const TRACK_NAMES = ['timeline', 'behavior', 'neural', 'model', 'metadata']

const ANCHOR_KEYWORDS: Array<{ keywords: string[]; matches: (anchor: ExperimentGlancerAnchor) => boolean }> = [
  {
    keywords: ['lick', 'lick onset'],
    matches: (anchor) => anchor.event_type === 'lick_onset',
  },
  {
    keywords: ['reward omission', 'omission'],
    matches: (anchor) => anchor.event_type === 'reward_omission',
  },
  {
    keywords: ['stimulus', 'stimulus onset'],
    matches: (anchor) => anchor.event_type === 'stimulus_onset',
  },
  {
    keywords: ['trial outcome', 'failure', 'failed trial', 'model error'],
    matches: (anchor) => anchor.event_type === 'trial_outcome',
  },
  {
    keywords: ['movement', 'movement onset'],
    matches: (anchor) => anchor.event_type === 'movement_onset',
  },
  {
    keywords: ['first trial', 'trial'],
    matches: (anchor) => anchor.kind === 'trial',
  },
  {
    keywords: ['overview'],
    matches: (anchor) => anchor.kind === 'dataset_overview',
  },
]

function findAnchor(phrase: string, anchors: ExperimentGlancerAnchor[]): ExperimentGlancerAnchor | null {
  const lower = phrase.toLowerCase()
  for (const { keywords, matches } of ANCHOR_KEYWORDS) {
    if (keywords.some((keyword) => lower.includes(keyword))) {
      const found = anchors.find(matches)
      if (found) return found
    }
  }
  // Fall back to a literal match against an anchor's own label.
  return anchors.find((anchor) => lower.includes(anchor.label.toLowerCase())) ?? null
}

interface ParseContext {
  anchors: ExperimentGlancerAnchor[]
  layers: ExperimentGlancerLayer[]
}

export function parseSceneCommand(rawInput: string, { anchors, layers }: ParseContext): SceneCommandResult {
  const input = rawInput.trim().toLowerCase()
  if (!input) return { ok: false, message: 'Type a command, e.g. "hide metadata" or "jump to reward omission".' }

  if (input === 'reset view' || input === 'reset') {
    return { ok: true, message: 'View reset.', action: { type: 'reset_view' } }
  }

  if (input.includes('show only file-verified') || input.includes('show only file verified')) {
    const layerIds = layers.filter((layer) => layer.status !== 'available').map((layer) => layer.layer_id)
    return {
      ok: true,
      message: 'Hid every layer that isn’t file-verified.',
      action: { type: 'set_layer_visibility', layerIds, visible: false, label: 'file-verified only' },
    }
  }

  if (input.includes('show all layers') || input.includes('show all') || input === 'show inferred layers' || input === 'show inferred') {
    return {
      ok: true,
      message: 'Showing all layers.',
      action: { type: 'set_layer_visibility', layerIds: layers.map((l) => l.layer_id), visible: true, label: 'all layers' },
    }
  }

  if (input.includes('hide inferred')) {
    const layerIds = layers.filter((layer) => layer.status === 'probable').map((layer) => layer.layer_id)
    return {
      ok: true,
      message: 'Hid inferred (metadata-only) layers.',
      action: { type: 'set_layer_visibility', layerIds, visible: false, label: 'inferred layers hidden' },
    }
  }

  const trackMatch = TRACK_NAMES.find((track) => input.includes(track))
  if (trackMatch && (input.startsWith('hide') || input.startsWith('show'))) {
    const visible = input.startsWith('show')
    const layerIds = layers.filter((layer) => layer.display.track === trackMatch).map((layer) => layer.layer_id)
    if (layerIds.length === 0) {
      return { ok: false, message: `No layers on the "${trackMatch}" track in this scene.` }
    }
    return {
      ok: true,
      message: `${visible ? 'Showed' : 'Hid'} the ${trackMatch} track.`,
      action: { type: 'set_layer_visibility', layerIds, visible, label: `${trackMatch} track` },
    }
  }

  const offsetMatch = input.match(/center\s+(-?\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\s+(before|after)\s+(.+)/)
  if (offsetMatch) {
    const [, secondsRaw, direction, eventPhrase] = offsetMatch
    const anchor = findAnchor(eventPhrase, anchors)
    if (!anchor) {
      return { ok: false, message: `Couldn't find an anchor matching "${eventPhrase.trim()}" in this scene.` }
    }
    const seconds = Number(secondsRaw) * (direction === 'before' ? -1 : 1)
    return {
      ok: true,
      message: `Centered ${Math.abs(seconds)}s ${direction} ${anchor.label}.`,
      action: { type: 'offset_from_anchor', anchor, seconds },
    }
  }

  const jumpMatch = input.match(/^(?:jump to|go to|center on|center)\s+(.+)/)
  if (jumpMatch) {
    const anchor = findAnchor(jumpMatch[1], anchors)
    if (!anchor) {
      return { ok: false, message: `Couldn't find an anchor matching "${jumpMatch[1].trim()}" in this scene.` }
    }
    return { ok: true, message: `Jumped to ${anchor.label}.`, action: { type: 'jump_to_anchor', anchor } }
  }

  return {
    ok: false,
    message: 'Not recognized. Try: "hide metadata", "show only file-verified layers", "jump to reward omission", "center 2s before lick".',
  }
}
