import type { ExperimentGlancerLayer, ExperimentGlancerScene, LayerStatus } from '../api/experimentglancer'

export interface SharedLayerDiff {
  kind: string
  label: string
  statusA: LayerStatus
  statusB: LayerStatus
}

export interface SetDiff {
  onlyInA: string[]
  onlyInB: string[]
  shared: string[]
}

export interface SceneDiffResult {
  layersOnlyInA: ExperimentGlancerLayer[]
  layersOnlyInB: ExperimentGlancerLayer[]
  sharedLayersWithDifferentStatus: SharedLayerDiff[]
  modalities: SetDiff
  evidenceTierA: string
  evidenceTierB: string
  anchorLabelA: string
  anchorLabelB: string
}

function setDiff(a: string[], b: string[]): SetDiff {
  const setA = new Set(a)
  const setB = new Set(b)
  return {
    onlyInA: a.filter((value) => !setB.has(value)),
    onlyInB: b.filter((value) => !setA.has(value)),
    shared: a.filter((value) => setB.has(value)),
  }
}

/** Pure comparison of two scene contracts -- layer availability, evidence
 * tier, anchor, and modality coverage. Used by Compare Mode to answer "what
 * does A have that B doesn't, and vice versa" without re-deriving it by eye
 * from two side-by-side JSON blobs. */
export function diffScenes(a: ExperimentGlancerScene, b: ExperimentGlancerScene): SceneDiffResult {
  const layersByKindA = new Map(a.layers.map((layer) => [layer.kind, layer]))
  const layersByKindB = new Map(b.layers.map((layer) => [layer.kind, layer]))

  const layersOnlyInA = a.layers.filter((layer) => !layersByKindB.has(layer.kind))
  const layersOnlyInB = b.layers.filter((layer) => !layersByKindA.has(layer.kind))

  const sharedLayersWithDifferentStatus: SharedLayerDiff[] = []
  for (const [kind, layerA] of layersByKindA) {
    const layerB = layersByKindB.get(kind)
    if (layerB && layerB.status !== layerA.status) {
      sharedLayersWithDifferentStatus.push({ kind, label: layerA.label, statusA: layerA.status, statusB: layerB.status })
    }
  }

  return {
    layersOnlyInA,
    layersOnlyInB,
    sharedLayersWithDifferentStatus,
    modalities: setDiff(a.dataset.modalities, b.dataset.modalities),
    evidenceTierA: a.provenance.evidence_tier,
    evidenceTierB: b.provenance.evidence_tier,
    anchorLabelA: a.anchors[0]?.label ?? 'none',
    anchorLabelB: b.anchors[0]?.label ?? 'none',
  }
}
