import { useState, type FormEvent } from 'react'
import type { ExperimentGlancerAnchor, ExperimentGlancerLayer } from '../../api/experimentglancer'
import { parseSceneCommand, type SceneCommandAction } from '../../lib/sceneCommandParser'

interface SceneCommandBarProps {
  anchors: ExperimentGlancerAnchor[]
  layers: ExperimentGlancerLayer[]
  onAction: (action: SceneCommandAction) => void
}

const EXAMPLES = ['hide metadata', 'show only file-verified layers', 'jump to reward omission', 'center 2s before lick']

/** A small, honest command bar -- not real NLP, a fixed grammar of scene
 * actions (layer visibility, anchor jumps, relative offsets). Every
 * recognized phrase maps to a deterministic action; unrecognized input gets
 * a helpful miss message instead of silently doing nothing. */
export function SceneCommandBar({ anchors, layers, onAction }: SceneCommandBarProps) {
  const [value, setValue] = useState('')
  const [feedback, setFeedback] = useState<{ ok: boolean; message: string } | null>(null)

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    const result = parseSceneCommand(value, { anchors, layers })
    setFeedback({ ok: result.ok, message: result.message })
    if (result.ok && result.action) {
      onAction(result.action)
      setValue('')
    }
  }

  return (
    <div className="border-t border-white/5 pt-3 mt-3">
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <span className="text-neural-600 font-mono text-xs">›</span>
        <input
          type="text"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={`e.g. "${EXAMPLES[0]}"`}
          className="flex-1 bg-transparent text-xs text-neural-200 placeholder:text-neural-700 focus:outline-none"
        />
        <button
          type="submit"
          className="text-xs text-neural-500 hover:text-accent-cyan transition-colors disabled:opacity-30"
          disabled={!value.trim()}
        >
          Run
        </button>
      </form>
      {feedback && (
        <p className={`mt-1.5 text-[11px] ${feedback.ok ? 'text-accent-emerald/80' : 'text-amber-400/80'}`}>
          {feedback.message}
        </p>
      )}
      {!feedback && (
        <p className="mt-1.5 text-[11px] text-neural-700">Try: {EXAMPLES.slice(1).join(' · ')}</p>
      )}
    </div>
  )
}
