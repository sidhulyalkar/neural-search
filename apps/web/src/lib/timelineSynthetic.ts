// Deterministic pseudo-random positions for a layer's timeline row when no
// file-validated event times exist yet. These are schematic density hints
// ONLY -- callers must label them as illustrative, never as real data -- so
// a track never renders as an empty, uninformative strip while a scene's
// evidence is still metadata-inferred.
function seededRandom(seed: number): () => number {
  let state = seed % 2147483647
  if (state <= 0) state += 2147483646
  return () => {
    state = (state * 16807) % 2147483647
    return (state - 1) / 2147483646
  }
}

function hashString(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0
  }
  return Math.abs(hash) || 1
}

const DENSITY_BY_KIND: Record<string, number> = {
  'neural.spikes': 28,
  'neural.calcium': 10,
  'neural.lfp': 0,
  'behavior.licks': 14,
  'behavior.pupil': 0,
  'behavior.wheel': 0,
  'timeline.events': 6,
  'timeline.trials': 5,
  'stimulus.identity': 4,
  'reward.delivery': 3,
  'video.frames': 0,
  'behavior.pose': 0,
  'model.predictions': 4,
  'model.latent_state': 0,
}

/** Positions in [0, 1] along the visible window for a layer's schematic
 * marks. Continuous layers (LFP, pupil, wheel, pose, video) return an empty
 * array -- they're drawn as a band, not discrete ticks. */
export function syntheticMarkPositions(layerId: string, kind: string): number[] {
  const count = DENSITY_BY_KIND[kind] ?? 0
  if (count === 0) return []
  const rand = seededRandom(hashString(`${kind}:${layerId}`))
  return Array.from({ length: count }, () => rand()).sort((a, b) => a - b)
}

export const CONTINUOUS_LAYER_KINDS = new Set([
  'neural.lfp',
  'behavior.pupil',
  'behavior.wheel',
  'behavior.pose',
  'video.frames',
])
