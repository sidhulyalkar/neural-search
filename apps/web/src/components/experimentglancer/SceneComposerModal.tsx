import {
  DEFAULT_COMPOSER_OPTIONS,
  EVENT_TYPE_OPTIONS,
  OPENING_MODES,
  type SceneComposerOptions,
} from '../../lib/sceneComposerPresets'
import { useState } from 'react'
import { XIcon } from '../Icons'

interface SceneComposerModalProps {
  datasetTitle: string
  isGenerating: boolean
  onClose: () => void
  onGenerate: (options: SceneComposerOptions) => void
}

/** Lets the user choose what kind of scene to open before paying for a
 * generation round trip -- different scientists approach the same dataset
 * with different mental models (overview vs. event-aligned vs. reviewing
 * failures), and the backend contract already supports all of these via
 * anchor_hint/affordance_ids/requested_layers (see sceneComposerPresets.ts).
 * No new backend endpoint required. */
export function SceneComposerModal({ datasetTitle, isGenerating, onClose, onGenerate }: SceneComposerModalProps) {
  const [options, setOptions] = useState<SceneComposerOptions>(DEFAULT_COMPOSER_OPTIONS)

  const update = <K extends keyof SceneComposerOptions>(key: K, value: SceneComposerOptions[K]) => {
    setOptions((current) => ({ ...current, [key]: value }))
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-neural-950/80 backdrop-blur-sm p-4">
      <div className="absolute inset-0" onClick={onClose} />
      <div className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-neural-900 shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
          <div>
            <p className="text-xs uppercase tracking-widest text-neural-500">Compose scene</p>
            <h2 className="text-sm text-neural-200 mt-0.5 truncate max-w-sm">{datasetTitle}</h2>
          </div>
          <button type="button" onClick={onClose} className="text-neural-500 hover:text-neural-200">
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
          <div>
            <p className="text-xs uppercase tracking-wide text-neural-600 mb-2">Opening mode</p>
            <div className="space-y-1.5">
              {OPENING_MODES.map((mode) => (
                <label
                  key={mode.id}
                  className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                    options.openingMode === mode.id
                      ? 'border-accent-cyan/50 bg-accent-cyan/5'
                      : 'border-white/5 hover:border-white/15'
                  }`}
                >
                  <input
                    type="radio"
                    name="opening-mode"
                    className="mt-1"
                    checked={options.openingMode === mode.id}
                    onChange={() => update('openingMode', mode.id)}
                  />
                  <span>
                    <span className="block text-sm text-neural-200">{mode.label}</span>
                    <span className="block text-xs text-neural-500">{mode.description}</span>
                  </span>
                </label>
              ))}
            </div>
          </div>

          {options.openingMode === 'event_aligned' && (
            <div>
              <p className="text-xs uppercase tracking-wide text-neural-600 mb-2">Event type</p>
              <select
                value={options.eventType}
                onChange={(event) => update('eventType', event.target.value)}
                className="w-full rounded-lg border border-white/10 bg-neural-950 px-3 py-2 text-sm text-neural-200"
              >
                {EVENT_TYPE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-wide text-neural-600">Layers</p>
            <label className="flex items-center gap-2 text-sm text-neural-300">
              <input
                type="checkbox"
                checked={options.includeProbableLayers}
                onChange={(event) => update('includeProbableLayers', event.target.checked)}
              />
              Include inferred (probable) layers
            </label>
            <label className="flex items-center gap-2 text-sm text-neural-300">
              <input
                type="checkbox"
                checked={options.includeModelOverlays}
                onChange={(event) => update('includeModelOverlays', event.target.checked)}
              />
              Include model overlays
            </label>
            <label className="flex items-center gap-2 text-sm text-neural-300">
              <input
                type="checkbox"
                checked={options.deepIntrospection}
                onChange={(event) => update('deepIntrospection', event.target.checked)}
              />
              Deep file introspection (slower, more file-verified evidence)
            </label>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 px-5 py-4 border-t border-white/10">
          <button type="button" onClick={onClose} className="text-xs text-neural-500 hover:text-neural-200">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onGenerate(options)}
            disabled={isGenerating}
            className="text-xs px-3 py-1.5 rounded-lg border border-accent-cyan/50 text-accent-cyan hover:bg-accent-cyan/10 disabled:opacity-40"
          >
            {isGenerating ? 'Generating…' : 'Generate scene'}
          </button>
        </div>
      </div>
    </div>
  )
}
