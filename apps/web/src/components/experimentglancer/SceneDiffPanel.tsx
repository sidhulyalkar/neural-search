import type { ExperimentGlancerScene } from '../../api/experimentglancer'
import { diffScenes } from '../../lib/sceneDiff'
import { GlassPanel } from './GlassPanel'

interface SceneDiffPanelProps {
  sceneA: ExperimentGlancerScene
  sceneB: ExperimentGlancerScene
}

export function SceneDiffPanel({ sceneA, sceneB }: SceneDiffPanelProps) {
  const diff = diffScenes(sceneA, sceneB)
  const labelA = sceneA.dataset.title ?? sceneA.dataset.dataset_id
  const labelB = sceneB.dataset.title ?? sceneB.dataset.dataset_id

  return (
    <GlassPanel tone="violet" className="p-4 space-y-3 text-xs">
      <p className="text-xs uppercase tracking-widest text-neural-500">Scene diff</p>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-neural-500">Anchor</p>
          <p className="text-neural-300">{diff.anchorLabelA}</p>
          <p className="text-neural-500 mt-1">Evidence tier</p>
          <p className="text-neural-300">{diff.evidenceTierA}</p>
        </div>
        <div>
          <p className="text-neural-500">Anchor</p>
          <p className="text-neural-300">{diff.anchorLabelB}</p>
          <p className="text-neural-500 mt-1">Evidence tier</p>
          <p className="text-neural-300">{diff.evidenceTierB}</p>
        </div>
      </div>

      {diff.modalities.onlyInA.length + diff.modalities.onlyInB.length > 0 && (
        <div>
          <p className="text-neural-500 mb-1">Modality difference</p>
          {diff.modalities.onlyInA.length > 0 && (
            <p className="text-neural-400">
              Only in <span className="text-neural-200">{labelA}</span>: {diff.modalities.onlyInA.join(', ')}
            </p>
          )}
          {diff.modalities.onlyInB.length > 0 && (
            <p className="text-neural-400">
              Only in <span className="text-neural-200">{labelB}</span>: {diff.modalities.onlyInB.join(', ')}
            </p>
          )}
        </div>
      )}

      {diff.layersOnlyInA.length > 0 && (
        <div>
          <p className="text-neural-500 mb-1">
            Layers only in <span className="text-neural-200">{labelA}</span>
          </p>
          <p className="text-neural-400">{diff.layersOnlyInA.map((layer) => layer.label).join(', ')}</p>
        </div>
      )}

      {diff.layersOnlyInB.length > 0 && (
        <div>
          <p className="text-neural-500 mb-1">
            Layers only in <span className="text-neural-200">{labelB}</span>
          </p>
          <p className="text-neural-400">{diff.layersOnlyInB.map((layer) => layer.label).join(', ')}</p>
        </div>
      )}

      {diff.sharedLayersWithDifferentStatus.length > 0 && (
        <div>
          <p className="text-neural-500 mb-1">Shared layers with different evidence</p>
          <ul className="space-y-0.5 text-neural-400">
            {diff.sharedLayersWithDifferentStatus.map((entry) => (
              <li key={entry.kind}>
                {entry.label}: <span className="text-neural-300">{entry.statusA}</span> vs.{' '}
                <span className="text-neural-300">{entry.statusB}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {diff.layersOnlyInA.length === 0 &&
        diff.layersOnlyInB.length === 0 &&
        diff.sharedLayersWithDifferentStatus.length === 0 && (
          <p className="text-neural-600">Both scenes offer the same layers at the same evidence tier.</p>
        )}
    </GlassPanel>
  )
}
