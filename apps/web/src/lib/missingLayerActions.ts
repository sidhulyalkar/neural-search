// Maps a free-form scene warning / missing-requirement string to a concrete
// next action, so the workbench reads as "here's what to do" instead of a
// wall of unexplained warning text. Order matters -- first match wins.
const ACTION_RULES: Array<{ keywords: string[]; action: string }> = [
  { keywords: ['pose', 'deeplabcut', 'kinematic'], action: 'Requires DeepLabCut (or equivalent) pose-tracking output.' },
  { keywords: ['video'], action: 'Requires video asset discovery for this dataset.' },
  { keywords: ['model', 'decod', 'latent'], action: 'Requires a decoder / model artifact to be generated.' },
  { keywords: ['spike', 'units', 'ephys', 'neuropixel'], action: 'Requires NWB units-table validation.' },
  { keywords: ['event', 'timestamp', 'trial'], action: 'Requires file-level inspection to validate timestamps.' },
]

export function describeMissingLayerAction(warning: string): string | null {
  const lower = warning.toLowerCase()
  const rule = ACTION_RULES.find((candidate) => candidate.keywords.some((keyword) => lower.includes(keyword)))
  return rule?.action ?? null
}
