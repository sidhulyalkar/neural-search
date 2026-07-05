import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getScene, type ExperimentGlancerScene } from '../../api/experimentglancer'
import { GlassPanel } from './GlassPanel'
import { SceneDiffPanel } from './SceneDiffPanel'
import { SynchronizedTimeline } from './SynchronizedTimeline'
import { SpinnerIcon } from '../Icons'

interface SceneComparePanelProps {
  primaryScene: ExperimentGlancerScene
  sharedCursorTime: number
  onSharedCursorTimeChange: (time: number) => void
}

/** Extracts a scene_id from either a raw id or a full pasted scene URL,
 * so users can compare by copy-pasting the shareable link they already
 * have open in another tab. */
function extractSceneId(input: string): string {
  const trimmed = input.trim()
  try {
    const url = new URL(trimmed, window.location.origin)
    const fromQuery = url.searchParams.get('scene_id')
    if (fromQuery) return fromQuery
  } catch {
    // Not a URL -- fall through and treat it as a raw scene id.
  }
  return trimmed
}

/** Compare two ExperimentGlancer scenes side by side with one shared
 * cursor -- the second timeline is fetched on demand from a pasted scene id
 * or shareable URL, since scenes now persist durably (see
 * neural_search/experimentglancer/persistence.py) and can be looked up long
 * after they were generated. */
export function SceneComparePanel({ primaryScene, sharedCursorTime, onSharedCursorTimeChange }: SceneComparePanelProps) {
  const [input, setInput] = useState('')
  const [compareSceneId, setCompareSceneId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['experimentglancer-scene', compareSceneId],
    queryFn: () => getScene(compareSceneId as string),
    enabled: !!compareSceneId,
  })

  return (
    <div className="space-y-4">
      <GlassPanel tone="violet" className="p-4">
        <p className="text-xs uppercase tracking-widest text-neural-500 mb-2">Compare with another scene</p>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Paste a scene id or shareable scene URL"
            className="flex-1 rounded-lg border border-white/10 bg-neural-950 px-2.5 py-1.5 text-xs text-neural-200 placeholder:text-neural-700 focus:outline-none focus:border-accent-violet/40"
          />
          <button
            type="button"
            onClick={() => setCompareSceneId(extractSceneId(input))}
            disabled={!input.trim()}
            className="text-xs px-3 py-1.5 rounded-lg border border-accent-violet/40 text-accent-violet hover:bg-accent-violet/10 disabled:opacity-40"
          >
            Load
          </button>
        </div>
        {isLoading && (
          <div className="flex items-center gap-2 mt-3 text-neural-500 text-xs">
            <SpinnerIcon className="w-3.5 h-3.5" />
            Loading scene…
          </div>
        )}
        {error && (
          <p className="mt-3 text-xs text-red-400">
            {error instanceof Error ? error.message : 'Could not load that scene.'}
          </p>
        )}
      </GlassPanel>

      {data && (
        <>
          <SceneDiffPanel sceneA={primaryScene} sceneB={data.scene} />
          <SynchronizedTimeline
            scene={data.scene}
            cursorTime={sharedCursorTime}
            onCursorTimeChange={onSharedCursorTimeChange}
            hideCommandBar
          />
        </>
      )}
    </div>
  )
}
