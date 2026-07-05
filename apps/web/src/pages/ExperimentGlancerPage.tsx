import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { createSceneFromSearchResult, getScene, type ExperimentGlancerAnchor } from '../api/experimentglancer'
import { AnchorSpotlight } from '../components/experimentglancer/AnchorSpotlight'
import { GlassPanel } from '../components/experimentglancer/GlassPanel'
import { LayerConfidenceLegend } from '../components/experimentglancer/LayerConfidenceLegend'
import { SceneBackdrop } from '../components/experimentglancer/SceneBackdrop'
import { SceneBookmarkPanel } from '../components/experimentglancer/SceneBookmarkPanel'
import { SceneComparePanel } from '../components/experimentglancer/SceneComparePanel'
import { SceneComposerModal } from '../components/experimentglancer/SceneComposerModal'
import { SceneHeroHeader } from '../components/experimentglancer/SceneHeroHeader'
import { SceneRationalePanel } from '../components/experimentglancer/SceneRationalePanel'
import { SynchronizedTimeline } from '../components/experimentglancer/SynchronizedTimeline'
import { WarningsDrawer } from '../components/experimentglancer/WarningsDrawer'
import { SpinnerIcon } from '../components/Icons'
import { applyOpeningModePreset } from '../lib/sceneComposerPresets'

export function ExperimentGlancerPage() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const sceneId = searchParams.get('scene_id') ?? ''
  const [rawOpen, setRawOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [composerOpen, setComposerOpen] = useState(false)
  const [compareOpen, setCompareOpen] = useState(false)
  const [cursorTime, setCursorTime] = useState(0)
  const [activeAnchor, setActiveAnchor] = useState<ExperimentGlancerAnchor | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['experimentglancer-scene', sceneId],
    queryFn: () => getScene(sceneId),
    enabled: !!sceneId,
  })

  const recomposeMutation = useMutation({
    mutationFn: (options: Parameters<typeof applyOpeningModePreset>[0]) => {
      if (!data) throw new Error('Scene not loaded yet.')
      const preset = applyOpeningModePreset(options)
      const modelOverlayLayers = options.includeModelOverlays ? ['model.predictions', 'model.latent_state'] : []
      return createSceneFromSearchResult({
        query: data.scene.source.query || '',
        dataset_id: data.scene.dataset.dataset_id,
        retrieval_method: data.scene.source.retrieval_method ?? undefined,
        score: data.scene.source.score ?? undefined,
        score_breakdown: data.scene.source.score_breakdown,
        requested_layers: Array.from(new Set([...preset.extraRequestedLayers, ...modelOverlayLayers])),
        affordance_ids: preset.extraAffordanceIds,
        anchor_hint: preset.anchorHint,
        include_probable_layers: options.includeProbableLayers,
        deep_introspection: options.deepIntrospection,
      })
    },
    onSuccess: (response) => {
      setComposerOpen(false)
      navigate(response.scene_url)
    },
  })

  const handleCopy = () => {
    navigator.clipboard?.writeText(window.location.href).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  if (!sceneId) {
    return (
      <div className="relative min-h-[60vh] flex items-center justify-center">
        <SceneBackdrop />
        <p className="text-neural-500">No scene_id provided.</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="relative min-h-[60vh] flex items-center justify-center gap-3 text-neural-400">
        <SceneBackdrop />
        <SpinnerIcon className="w-5 h-5 text-accent-cyan" />
        <span>Compiling scene…</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="relative min-h-[60vh] flex flex-col items-center justify-center gap-3 text-center px-6">
        <SceneBackdrop />
        <p className="text-neural-300">This scene could not be found.</p>
        <p className="text-sm text-neural-600">
          {error instanceof Error ? error.message : 'This scene id does not exist.'}
        </p>
        <Link to="/" className="text-accent-cyan text-sm hover:text-white transition-colors">
          ← Back to search
        </Link>
      </div>
    )
  }

  const { scene } = data
  const primaryAnchor = scene.anchors[0]

  return (
    <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-5">
      <SceneBackdrop />

      <SceneHeroHeader dataset={scene.dataset} source={scene.source} queryContext={scene.query_context} />

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={() => setComposerOpen(true)}
          className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
        >
          Recompose…
        </button>
        <button
          type="button"
          onClick={() => setCompareOpen((value) => !value)}
          className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
        >
          {compareOpen ? 'Close compare' : 'Compare with another scene…'}
        </button>
        {recomposeMutation.error && (
          <span className="text-xs text-red-400">
            {recomposeMutation.error instanceof Error ? recomposeMutation.error.message : 'Recompose failed.'}
          </span>
        )}
      </div>

      {primaryAnchor && (
        <AnchorSpotlight anchor={primaryAnchor} coordinateSpace={scene.coordinate_space} />
      )}

      <GlassPanel tone="neutral" className="p-4">
        <p className="text-xs uppercase tracking-widest text-neural-500 mb-2">Why this scene</p>
        <SceneRationalePanel scene={scene} />
      </GlassPanel>

      <SynchronizedTimeline
        scene={scene}
        cursorTime={cursorTime}
        onCursorTimeChange={setCursorTime}
        onActiveAnchorChange={setActiveAnchor}
      />

      {compareOpen && <SceneComparePanel primaryScene={scene} sharedCursorTime={cursorTime} onSharedCursorTimeChange={setCursorTime} />}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <GlassPanel tone="neutral" className="p-4">
          <LayerConfidenceLegend />
        </GlassPanel>

        <SceneBookmarkPanel
          sceneId={scene.scene_id}
          sceneUrl={window.location.href}
          datasetId={scene.dataset.dataset_id}
          datasetTitle={scene.dataset.title ?? scene.dataset.dataset_id}
          cursorTime={cursorTime}
          activeAnchorLabel={activeAnchor?.label ?? null}
          onJump={setCursorTime}
        />
      </div>

      <WarningsDrawer warnings={scene.warnings} missingRequirements={scene.provenance.missing_requirements} />

      <GlassPanel tone="neutral" className="p-4">
        <div className="flex items-center justify-between gap-3 mb-2">
          <p className="text-xs uppercase tracking-widest text-neural-500">Shareable URL</p>
          <button
            type="button"
            onClick={handleCopy}
            className="text-xs text-neural-400 hover:text-accent-cyan transition-colors"
          >
            {copied ? 'Copied!' : 'Copy link'}
          </button>
        </div>
        <p className="text-xs font-mono text-neural-500 break-all">{window.location.href}</p>

        <div className="mt-4 pt-4 border-t border-white/10">
          <button
            type="button"
            onClick={() => setRawOpen((value) => !value)}
            className="text-xs text-neural-500 hover:text-neural-300 transition-colors"
          >
            {rawOpen ? 'Hide raw scene JSON' : 'View raw scene JSON'}
          </button>
          {rawOpen && (
            <pre className="mt-3 max-h-96 overflow-auto rounded-lg border border-white/10 bg-black/30 p-3 text-xs text-neural-400">
              {JSON.stringify(scene, null, 2)}
            </pre>
          )}
        </div>
      </GlassPanel>

      {composerOpen && (
        <SceneComposerModal
          datasetTitle={scene.dataset.title ?? scene.dataset.dataset_id}
          isGenerating={recomposeMutation.isPending}
          onClose={() => setComposerOpen(false)}
          onGenerate={(options) => recomposeMutation.mutate(options)}
        />
      )}
    </div>
  )
}
