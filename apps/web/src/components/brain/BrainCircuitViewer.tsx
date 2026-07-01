/**
 * BrainCircuitViewer
 *
 * Loads brain.glb (placed at public/brain.glb) and renders it as a
 * highly-transparent shell, revealing circuit nodes and dashed directional
 * connections floating inside.  Clicking any node opens an inline detail
 * panel linking out to papers, disorders, and the KG.
 */
import { useEffect, useRef, useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import * as THREE from 'three'
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

// ── Oscillation band metadata ─────────────────────────────────────────────────

export const BAND_COLOR: Record<string, string> = {
  gamma:   '#f97316',
  beta:    '#10b981',
  alpha:   '#06b6d4',
  theta:   '#3b82f6',
  delta:   '#6366f1',
  spindle: '#a78bfa',
  ripple:  '#ec4899',
}

const BAND_SPEED: Record<string, number> = {
  gamma: 4.0, beta: 2.5, alpha: 1.6, theta: 1.0,
  delta: 0.4, spindle: 0.8, ripple: 5.0,
}

// ── Anatomical node positions (MNI-space, in Three.js units) ─────────────────
// Calibrated relative to the loaded GLB's bounding box at runtime.

const REGION_MNI: Record<string, [number, number, number]> = {
  hippocampus_l:       [-26, -18, -10],
  hippocampus_r:       [ 26, -18, -10],
  pfc_l:               [-28,  30,  24],
  pfc_r:               [ 28,  30,  24],
  dlpfc_l:             [-36,  24,  30],
  dlpfc_r:             [ 36,  24,  30],
  striatum_l:          [-14,   6,   8],
  striatum_r:          [ 14,   6,   8],
  amygdala_l:          [-22,  -4, -16],
  amygdala_r:          [ 22,  -4, -16],
  anterior_cingulate:  [  0,  20,  32],
  posterior_cingulate: [  0, -18,  30],
  thalamus_l:          [ -8, -12,   6],
  thalamus_r:          [  8, -12,   6],
  vta:                 [  0, -20, -10],
  nucleus_accumbens_l: [-10,   8,  -6],
  nucleus_accumbens_r: [ 10,   8,  -6],
  motor_cortex_l:      [-36,  -2,  52],
  motor_cortex_r:      [ 36,  -2,  52],
  insula_l:            [-40,   6,   6],
  insula_r:            [ 40,   6,   6],
  locus_coeruleus:     [  0, -34, -24],
  subthalamic_l:       [-12, -14,  -2],
  subthalamic_r:       [ 12, -14,  -2],
  entorhinal_l:        [-26, -22, -24],
  entorhinal_r:        [ 26, -22, -24],
  angular_gyrus_l:     [-46, -46,  32],
  angular_gyrus_r:     [ 46, -46,  32],
  ofc_l:               [-18,  42, -14],
  ofc_r:               [ 18,  42, -14],
  hypothalamus:        [  0,  -8, -10],
  parahippocampal_l:   [-22, -30, -16],
  parahippocampal_r:   [ 22, -30, -16],
}

// Converts raw MNI coords to Three.js Vector3, optionally scaled
function mniToVec3(key: string, scale = 1): THREE.Vector3 {
  const p = REGION_MNI[key] ?? [0, 0, 0]
  return new THREE.Vector3(p[0] * scale, p[1] * scale, p[2] * scale)
}

// ── Circuit→disorder mapping (for the click panel) ────────────────────────────

const CIRCUIT_DISORDERS: Record<string, string[]> = {
  hippocampal_circuit:   ['alzheimers_disease', 'ptsd', 'major_depressive_disorder', 'schizophrenia'],
  corticostriatal_beta:  ['parkinsons_disease', 'ocd', 'schizophrenia', 'adhd'],
  default_mode:          ['major_depressive_disorder', 'schizophrenia', 'autism_spectrum_disorder', 'alzheimers_disease'],
  fear_circuit:          ['ptsd', 'panic_disorder', 'generalized_anxiety', 'phobia'],
  basal_ganglia_loop:    ['parkinsons_disease', 'huntingtons_disease', 'ocd', 'tourette_syndrome'],
  prefrontal_cortex:     ['schizophrenia', 'adhd', 'major_depressive_disorder', 'bipolar_disorder'],
  reward_addiction:      ['substance_use_disorder', 'gambling_disorder', 'major_depressive_disorder'],
  dopamine_mesolimbic:   ['schizophrenia', 'bipolar_disorder', 'substance_use_disorder'],
  lc_norepinephrine:     ['adhd', 'ptsd', 'major_depressive_disorder', 'panic_disorder'],
  stress_hpa:            ['ptsd', 'major_depressive_disorder', 'generalized_anxiety', 'cushing_syndrome'],
  thalamocortical_sensory: ['epilepsy', 'insomnia', 'schizophrenia', 'fibromyalgia'],
}

// ── Circuit definitions ───────────────────────────────────────────────────────

export const CIRCUIT_DEFS: Record<string, {
  label: string
  color: string
  regions: string[]
  connections: [string, string][]
  oscillations: { band: string; hz?: string; note?: string }[]
  description: string
  keyMechanism: string
}> = {
  hippocampal_circuit: {
    label: 'Hippocampal Memory Circuit',
    color: '#06b6d4',
    regions: ['hippocampus_l', 'hippocampus_r', 'entorhinal_l', 'entorhinal_r', 'thalamus_l', 'pfc_l'],
    connections: [
      ['entorhinal_l', 'hippocampus_l'],
      ['hippocampus_l', 'hippocampus_r'],
      ['hippocampus_l', 'thalamus_l'],
      ['thalamus_l', 'pfc_l'],
      ['hippocampus_l', 'entorhinal_l'],
      ['entorhinal_l', 'parahippocampal_l'],
    ],
    oscillations: [
      { band: 'theta', hz: '4–8 Hz', note: 'encoding rhythm' },
      { band: 'gamma', hz: '30–80 Hz', note: 'retrieval bursts' },
      { band: 'ripple', hz: '100–180 Hz', note: 'consolidation' },
    ],
    description: 'Bidirectional entorhinal–hippocampal–PFC loop for episodic memory encoding, consolidation, and retrieval.',
    keyMechanism: 'Theta–gamma coupling indexes memory load; sharp-wave ripples replay sequences during offline consolidation.',
  },
  corticostriatal_beta: {
    label: 'Cortico-Striatal β Loop',
    color: '#f59e0b',
    regions: ['pfc_l', 'pfc_r', 'striatum_l', 'striatum_r', 'nucleus_accumbens_l', 'vta', 'thalamus_l'],
    connections: [
      ['pfc_l', 'striatum_l'],
      ['striatum_l', 'nucleus_accumbens_l'],
      ['vta', 'nucleus_accumbens_l'],
      ['vta', 'pfc_l'],
      ['striatum_l', 'thalamus_l'],
      ['thalamus_l', 'pfc_l'],
    ],
    oscillations: [
      { band: 'beta', hz: '13–30 Hz', note: 'cortico-striatal sync' },
    ],
    description: 'Cortico-basal ganglia-thalamic loop mediating reward learning and action selection via dopaminergic gating.',
    keyMechanism: 'Beta power indexes the status quo; suppression of beta gates initiation. Elevated beta → akinesia in Parkinson\'s.',
  },
  default_mode: {
    label: 'Default Mode Network',
    color: '#8b5cf6',
    regions: ['pfc_l', 'pfc_r', 'posterior_cingulate', 'angular_gyrus_l', 'angular_gyrus_r', 'hippocampus_l'],
    connections: [
      ['pfc_l', 'posterior_cingulate'],
      ['posterior_cingulate', 'angular_gyrus_l'],
      ['angular_gyrus_l', 'hippocampus_l'],
      ['hippocampus_l', 'posterior_cingulate'],
      ['pfc_r', 'posterior_cingulate'],
    ],
    oscillations: [
      { band: 'alpha', hz: '8–12 Hz', note: 'idling' },
      { band: 'theta', hz: '4–8 Hz', note: 'internal narrative' },
    ],
    description: 'Task-negative resting-state network active during mind-wandering, self-referential thought, and prospection.',
    keyMechanism: 'Anti-correlated with task-positive networks; alpha synchrony suppresses irrelevant processing.',
  },
  fear_circuit: {
    label: 'Fear / Threat Circuit',
    color: '#ef4444',
    regions: ['amygdala_l', 'amygdala_r', 'hippocampus_l', 'anterior_cingulate', 'insula_l', 'pfc_l', 'hypothalamus'],
    connections: [
      ['amygdala_l', 'hippocampus_l'],
      ['amygdala_l', 'anterior_cingulate'],
      ['amygdala_l', 'hypothalamus'],
      ['anterior_cingulate', 'pfc_l'],
      ['anterior_cingulate', 'insula_l'],
    ],
    oscillations: [
      { band: 'gamma', hz: '30–80 Hz', note: 'threat detection' },
      { band: 'theta', hz: '4–8 Hz', note: 'fear memory trace' },
    ],
    description: 'Amygdala-centred network detecting threats and coordinating autonomic, hormonal, and cognitive fear responses.',
    keyMechanism: 'BLA→CeA drives defensive behaviour; vPFC→BLA extinction pathway is disrupted in PTSD.',
  },
  basal_ganglia_loop: {
    label: 'Basal Ganglia Loop',
    color: '#10b981',
    regions: ['striatum_l', 'striatum_r', 'subthalamic_l', 'subthalamic_r', 'thalamus_l', 'thalamus_r', 'motor_cortex_l'],
    connections: [
      ['motor_cortex_l', 'striatum_l'],
      ['striatum_l', 'subthalamic_l'],
      ['subthalamic_l', 'thalamus_l'],
      ['thalamus_l', 'motor_cortex_l'],
      ['striatum_r', 'thalamus_r'],
    ],
    oscillations: [
      { band: 'beta', hz: '13–30 Hz', note: 'pathological in PD' },
    ],
    description: 'Direct/indirect pathway through basal ganglia gating voluntary action. Pathological beta synchrony underlies parkinsonian akinesia.',
    keyMechanism: 'Direct path (D1) → movement facilitation; indirect path (D2) → suppression. STN hyperdirect path provides rapid braking.',
  },
  prefrontal_cortex: {
    label: 'Prefrontal Executive Network',
    color: '#0ea5e9',
    regions: ['dlpfc_l', 'dlpfc_r', 'anterior_cingulate', 'thalamus_l', 'insula_l'],
    connections: [
      ['dlpfc_l', 'anterior_cingulate'],
      ['anterior_cingulate', 'thalamus_l'],
      ['thalamus_l', 'dlpfc_l'],
      ['anterior_cingulate', 'insula_l'],
      ['dlpfc_r', 'anterior_cingulate'],
    ],
    oscillations: [
      { band: 'gamma', hz: '40 Hz', note: 'WM maintenance' },
      { band: 'theta', hz: '6–10 Hz', note: 'cognitive control' },
    ],
    description: 'DLPFC-ACC-thalamus network underpinning working memory, error monitoring, cognitive flexibility, and executive control.',
    keyMechanism: 'Persistent gamma activity maintains representations in WM. Theta phase organises multiple items across cycles.',
  },
  reward_addiction: {
    label: 'Reward / Addiction Circuit',
    color: '#ea580c',
    regions: ['nucleus_accumbens_l', 'nucleus_accumbens_r', 'vta', 'pfc_l', 'ofc_l', 'amygdala_l'],
    connections: [
      ['vta', 'nucleus_accumbens_l'],
      ['vta', 'nucleus_accumbens_r'],
      ['pfc_l', 'nucleus_accumbens_l'],
      ['nucleus_accumbens_l', 'ofc_l'],
      ['ofc_l', 'amygdala_l'],
    ],
    oscillations: [
      { band: 'delta', hz: '1–4 Hz', note: 'outcome coding' },
      { band: 'gamma', hz: '30–60 Hz', note: 'cue salience' },
    ],
    description: 'Mesolimbic network encoding reward prediction errors and cue salience. Dysregulation drives craving and compulsive seeking.',
    keyMechanism: 'VTA dopamine encodes RPE. Repeated drug exposure hijacks LTP at VTA→NAc synapses, blunting natural reward.',
  },
  dopamine_mesolimbic: {
    label: 'Mesolimbic Dopamine',
    color: '#f97316',
    regions: ['vta', 'nucleus_accumbens_l', 'nucleus_accumbens_r', 'pfc_l', 'amygdala_l', 'hippocampus_l'],
    connections: [
      ['vta', 'nucleus_accumbens_l'],
      ['vta', 'nucleus_accumbens_r'],
      ['vta', 'pfc_l'],
      ['vta', 'amygdala_l'],
      ['vta', 'hippocampus_l'],
    ],
    oscillations: [{ band: 'delta', note: 'RPE burst' }],
    description: 'VTA dopaminergic projections broadcasting reward prediction errors to NAc, PFC, amygdala, and hippocampus.',
    keyMechanism: 'Phasic VTA bursts signal better-than-expected rewards; pauses signal worse-than-expected. Sustained tonic DA sets salience threshold.',
  },
  lc_norepinephrine: {
    label: 'Locus Coeruleus–NE System',
    color: '#c084fc',
    regions: ['locus_coeruleus', 'pfc_l', 'pfc_r', 'hippocampus_l', 'amygdala_l', 'thalamus_l'],
    connections: [
      ['locus_coeruleus', 'pfc_l'],
      ['locus_coeruleus', 'pfc_r'],
      ['locus_coeruleus', 'hippocampus_l'],
      ['locus_coeruleus', 'amygdala_l'],
      ['locus_coeruleus', 'thalamus_l'],
    ],
    oscillations: [{ band: 'gamma', note: 'arousal gating' }],
    description: 'LC broadcasts NE to virtually all forebrain regions, regulating arousal, attention, gain, and stress reactivity.',
    keyMechanism: 'Inverted-U: optimal NE maximises PFC gain. Excess NE shifts to subcortical hypervigilance; deficiency → inattention.',
  },
  stress_hpa: {
    label: 'HPA Stress Axis',
    color: '#b45309',
    regions: ['hypothalamus', 'amygdala_l', 'amygdala_r', 'hippocampus_l', 'pfc_l', 'anterior_cingulate'],
    connections: [
      ['amygdala_l', 'hypothalamus'],
      ['amygdala_r', 'hypothalamus'],
      ['hippocampus_l', 'hypothalamus'],
      ['pfc_l', 'amygdala_l'],
      ['anterior_cingulate', 'amygdala_l'],
    ],
    oscillations: [{ band: 'theta', note: 'stress theta' }],
    description: 'Hypothalamic-pituitary-adrenal axis regulating cortisol in response to stress. Chronic stress remodels hippocampal–amygdala connectivity.',
    keyMechanism: 'Hippocampus provides negative feedback to suppress CRH. Chronic stress reduces hippocampal volume → impaired feedback → hypercortisolaemia.',
  },
  thalamocortical_sensory: {
    label: 'Thalamocortical Sensory Loop',
    color: '#38bdf8',
    regions: ['thalamus_l', 'thalamus_r', 'motor_cortex_l', 'motor_cortex_r', 'insula_l', 'anterior_cingulate'],
    connections: [
      ['thalamus_l', 'motor_cortex_l'],
      ['thalamus_r', 'motor_cortex_r'],
      ['thalamus_l', 'insula_l'],
      ['thalamus_l', 'anterior_cingulate'],
      ['motor_cortex_l', 'thalamus_l'],
    ],
    oscillations: [
      { band: 'spindle', hz: '12–15 Hz', note: 'sleep spindles' },
      { band: 'alpha', hz: '8–12 Hz', note: 'sensory gating' },
    ],
    description: 'Thalamo-cortical relay generating sleep spindles during NREM and providing sensory gating via alpha synchrony.',
    keyMechanism: 'Thalamic reticular nucleus inhibits relay cells in rhythmic bursts → spindles. Disruption → hypersensitivity, hyperekplexia.',
  },
}

// ── Dashed directional connection ─────────────────────────────────────────────

function buildDashedArrow(
  pa: THREE.Vector3,
  pb: THREE.Vector3,
  color: THREE.Color,
  opacity: number,
  isActive: boolean,
): THREE.Group {
  const group = new THREE.Group()

  // Arc path through interior
  const mid = pa.clone().lerp(pb, 0.5).multiplyScalar(0.46)
  const curve = new THREE.CatmullRomCurve3([pa, mid, pb], false, 'catmullrom', 0.5)
  const points = curve.getPoints(90)

  // Dashed line — 1px crisp (WebGL linewidth limitation)
  const lineGeo = new THREE.BufferGeometry().setFromPoints(points)
  const lineMat = new THREE.LineDashedMaterial({
    color,
    dashSize: isActive ? 6 : 3,
    gapSize:  isActive ? 3 : 2,
    transparent: true,
    opacity: isActive ? opacity : opacity * 0.6,
  })
  const line = new THREE.Line(lineGeo, lineMat)
  line.computeLineDistances()
  group.add(line)

  // Arrowhead cone at endpoint (active only)
  if (isActive) {
    const tangent = curve.getTangent(0.97).normalize()
    const endpoint = curve.getPoint(1.0)
    const coneH = 7, coneR = 2.2
    const coneGeo = new THREE.ConeGeometry(coneR, coneH, 8)
    const coneMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
    const cone = new THREE.Mesh(coneGeo, coneMat)
    cone.position.copy(endpoint).addScaledVector(tangent, -coneH * 0.5)
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent)
    group.add(cone)
  }

  return group
}

// ── Component props ───────────────────────────────────────────────────────────

export interface BrainCircuitViewerProps {
  /** Which circuits to highlight in 3D (others dim). */
  activeCircuits?: string[]
  /** Which circuit's detail panel to show — pass null to clear. Controlled externally. */
  selectedCircuit?: string | null
  extraOscillations?: { band: string; region?: string; note?: string }[]
  height?: number
  onCircuitSelect?: (id: string) => void
}

// ── Main component ────────────────────────────────────────────────────────────

export function BrainCircuitViewer({
  activeCircuits = [],
  selectedCircuit = null,
  extraOscillations = [],
  height = 520,
  onCircuitSelect,
}: BrainCircuitViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [hoverLabel, setHoverLabel] = useState<{ text: string; x: number; y: number } | null>(null)

  const activeRef = useRef(activeCircuits)
  activeRef.current = activeCircuits

  // ── Three.js scene setup ──────────────────────────────────────────────────

  const buildScene = useCallback(() => {
    const mount = mountRef.current
    if (!mount) return

    const W = mount.clientWidth || 800
    const H = mount.clientHeight || height

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(W, H)
    renderer.setClearColor(0x000000, 0)
    renderer.sortObjects = true
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(40, W / H, 0.5, 2000)
    camera.position.set(40, 30, 230)
    camera.lookAt(0, 8, 0)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.minDistance = 70
    controls.maxDistance = 600
    controls.target.set(0, 8, 0)

    // ── Lighting ─────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x8899bb, 1.1))
    const key = new THREE.DirectionalLight(0xfff8ee, 1.3)
    key.position.set(80, 130, 150)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xccddff, 0.4)
    fill.position.set(-100, -40, 60)
    scene.add(fill)
    const rim = new THREE.DirectionalLight(0x8899cc, 0.3)
    rim.position.set(0, 0, -200)
    scene.add(rim)

    // ── Circuit overlay groups ────────────────────────────────────────────────
    const circuitGroup = new THREE.Group()
    circuitGroup.renderOrder = 2
    scene.add(circuitGroup)

    const ringGroup = new THREE.Group()
    ringGroup.renderOrder = 3
    scene.add(ringGroup)

    // Node meshes for raycasting
    const nodeMeshes: { mesh: THREE.Mesh; circuitId: string; regionKey: string }[] = []

    // ── Build circuit overlays ────────────────────────────────────────────────
    // coordScale: after GLB loads we recalibrate; starts at 1
    let coordScale = 1.0

    const rebuildCircuits = () => {
      while (circuitGroup.children.length) circuitGroup.remove(circuitGroup.children[0])
      while (ringGroup.children.length) ringGroup.remove(ringGroup.children[0])
      nodeMeshes.length = 0

      const actives = activeRef.current
      const hasFilter = actives.length > 0

      Object.entries(CIRCUIT_DEFS).forEach(([id, def]) => {
        const isActive = !hasFilter || actives.includes(id)
        const color = new THREE.Color(def.color)
        const lineOpacity = isActive ? 0.92 : 0.05
        const nodeOpacity = isActive ? 1.0  : 0.06

        // ── Connections (dashed + arrow) ──────────────────────────────────────
        def.connections.forEach(([a, b]) => {
          const pa = mniToVec3(a, coordScale)
          const pb = mniToVec3(b, coordScale)
          const conn = buildDashedArrow(pa, pb, color, lineOpacity, isActive)
          circuitGroup.add(conn)
        })

        // ── Region nodes ──────────────────────────────────────────────────────
        const seen = new Set<string>()
        def.regions.forEach(key => {
          if (seen.has(key)) return
          seen.add(key)
          const p = mniToVec3(key, coordScale)

          // Core node — solid, glowing
          const r = isActive ? 4.8 : 1.6
          const nodeMat = new THREE.MeshPhongMaterial({
            color,
            emissive: color,
            emissiveIntensity: isActive ? 0.7 : 0.03,
            transparent: true,
            opacity: nodeOpacity,
            shininess: 80,
            depthWrite: true,
          })
          const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 18), nodeMat)
          mesh.position.copy(p)
          mesh.renderOrder = 4
          circuitGroup.add(mesh)

          // Outer halo for glow bleed (active only)
          if (isActive) {
            const haloMat = new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: 0.14,
              depthWrite: false,
            })
            const halo = new THREE.Mesh(new THREE.SphereGeometry(r * 2.1, 12, 12), haloMat)
            halo.position.copy(p)
            halo.renderOrder = 3
            circuitGroup.add(halo)

            nodeMeshes.push({ mesh, circuitId: id, regionKey: key })
          }
        })

        // ── Oscillation rings (active circuits only) ──────────────────────────
        if (isActive) {
          def.oscillations.forEach(({ band }, bi) => {
            const p = mniToVec3(def.regions[0], coordScale)
            const ringColor = new THREE.Color(BAND_COLOR[band] ?? '#ffffff')
            const ring = new THREE.Mesh(
              new THREE.RingGeometry(7, 11, 36),
              new THREE.MeshBasicMaterial({
                color: ringColor, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false,
              }),
            )
            ring.position.copy(p)
            ring.renderOrder = 5
            ring.userData = { band, phase: bi * 0.8 + Math.random() * 0.4 }
            ringGroup.add(ring)
          })
        }
      })

      // Extra oscillations from disorder context
      extraOscillations.forEach(({ band, region }, idx) => {
        const key = region
          ? (Object.keys(REGION_MNI).find(k => k.startsWith(region.slice(0, 6))) ?? 'hippocampus_l')
          : 'hippocampus_l'
        const p = mniToVec3(key, coordScale)
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(9, 14, 36),
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(BAND_COLOR[band] ?? '#ffffff'),
            transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false,
          }),
        )
        ring.position.copy(p)
        ring.renderOrder = 5
        ring.userData = { band, phase: idx * 0.5 }
        ringGroup.add(ring)
      })
    }

    rebuildCircuits()

    // ── Load GLTF brain ───────────────────────────────────────────────────────
    const loader = new GLTFLoader()
    loader.load(
      '/brain.glb',
      (gltf: any) => {
        const model: THREE.Group = gltf.scene

        // Auto-scale: fit within ±75 units on the largest axis
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z)
        const targetSize = 150
        const scale = targetSize / maxDim
        model.scale.setScalar(scale)

        // Center the model
        box.setFromObject(model)
        const center = box.getCenter(new THREE.Vector3())
        model.position.sub(center)
        // Shift up slightly so the brain sits in view
        model.position.y += 4

        // Apply transparent shell material — show real topology, hide surface colour
        model.traverse((child: THREE.Object3D) => {
          if ((child as THREE.Mesh).isMesh) {
            const m = child as THREE.Mesh
            m.renderOrder = 1
            // Preserve original geometry (real folds) but override material
            m.material = new THREE.MeshPhongMaterial({
              color: new THREE.Color(0xe8d0bc),      // warm tissue pinkish-beige
              specular: new THREE.Color(0x554433),
              shininess: 16,
              transparent: true,
              opacity: 0.09,                          // highly transparent
              depthWrite: false,                      // let interior through
              side: THREE.FrontSide,
            })

            // Add edge lines at low opacity for sulci hint
            const edgeGeo = new THREE.EdgesGeometry(m.geometry, 22)
            const edgeMat = new THREE.LineBasicMaterial({
              color: 0xc8a898,
              transparent: true,
              opacity: 0.07,
            })
            const edges = new THREE.LineSegments(edgeGeo, edgeMat)
            edges.renderOrder = 1
            // Edges inherit parent transform
            m.add(edges)
          }
        })

        scene.add(model)

        // Calibrate MNI coord scale to the loaded model dimensions
        const loadedBox = new THREE.Box3().setFromObject(model)
        const loadedSize = loadedBox.getSize(new THREE.Vector3())
        // MNI brain is ~170×140×120 mm; match to loaded x-extent
        coordScale = loadedSize.x / 170
        rebuildCircuits()

        setLoadState('ready')
      },
      undefined,
      (err: any) => {
        console.warn('brain.glb load failed:', err)
        setLoadState('error')
      },
    )

    // ── Raycasting ────────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2(-9999, -9999)

    const onMouseMove = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(nodeMeshes.map(n => n.mesh))
      if (hits.length) {
        const info = nodeMeshes.find(n => n.mesh === hits[0].object)!
        const def = CIRCUIT_DEFS[info.circuitId]
        const sp = hits[0].object.position.clone().project(camera)
        const rect2 = renderer.domElement.getBoundingClientRect()
        setHoverLabel({
          text: `${def.label} · ${info.regionKey.replace(/_/g, ' ')}`,
          x: ((sp.x + 1) / 2) * rect2.width + rect2.left,
          y: ((-sp.y + 1) / 2) * rect2.height + rect2.top,
        })
        renderer.domElement.style.cursor = 'pointer'
      } else {
        setHoverLabel(null)
        renderer.domElement.style.cursor = 'default'
      }
    }

    const onClick = () => {
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(nodeMeshes.map(n => n.mesh))
      if (hits.length) {
        const info = nodeMeshes.find(n => n.mesh === hits[0].object)!
        onCircuitSelect?.(info.circuitId)
      }
    }

    renderer.domElement.addEventListener('mousemove', onMouseMove)
    renderer.domElement.addEventListener('click', onClick)

    // ── Animation ─────────────────────────────────────────────────────────────
    let rafId: number
    const timer = new THREE.Timer()

    const animate = () => {
      rafId = requestAnimationFrame(animate)
      timer.update()
      const t = timer.getElapsed()
      controls.update()

      ringGroup.children.forEach(child => {
        const ring = child as THREE.Mesh
        const mat = ring.material as THREE.MeshBasicMaterial
        const { band, phase } = ring.userData as { band: string; phase: number }
        const speed = BAND_SPEED[band] ?? 1
        const pulse = Math.sin((t * speed + phase) % (Math.PI * 2))
        const s = 0.5 + pulse * 0.6
        ring.scale.set(s, s, s)
        mat.opacity = Math.max(0, pulse) * 0.72
        ring.quaternion.copy(camera.quaternion)
      })

      renderer.render(scene, camera)
    }
    animate()

    // ── Resize ────────────────────────────────────────────────────────────────
    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('mousemove', onMouseMove)
      renderer.domElement.removeEventListener('click', onClick)
      controls.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [extraOscillations, onCircuitSelect, height])

  useEffect(() => {
    const cleanup = buildScene()
    return cleanup ?? undefined
  }, [buildScene])

  const selectedDef = selectedCircuit ? CIRCUIT_DEFS[selectedCircuit] : null
  const disorders = selectedCircuit ? (CIRCUIT_DISORDERS[selectedCircuit] ?? []) : []

  return (
    <div className="relative w-full flex flex-col">
      {/* 3D canvas area */}
      <div className="relative w-full" style={{ height }}>
        <div ref={mountRef} className="w-full h-full" />

        {/* Loading / error overlay */}
        {loadState === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-neural-700 border-t-accent-cyan rounded-full animate-spin" />
              <span className="text-[10px] font-mono text-neural-600">loading brain model…</span>
            </div>
          </div>
        )}

        {/* Hover tooltip */}
        {hoverLabel && (
          <div
            className="fixed z-50 pointer-events-none px-2.5 py-1 rounded bg-neural-950/90 border border-neural-700 text-xs font-mono text-neural-200 whitespace-nowrap shadow-lg backdrop-blur-sm"
            style={{ left: hoverLabel.x + 14, top: hoverLabel.y - 10 }}
          >
            {hoverLabel.text}
          </div>
        )}

        {/* Band legend */}
        <div className="absolute bottom-3 right-3 flex flex-col gap-1 bg-neural-950/70 rounded-lg px-2 py-1.5 backdrop-blur-sm">
          {Object.entries(BAND_COLOR).map(([band, color]) => (
            <div key={band} className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: color, boxShadow: `0 0 4px ${color}` }} />
              <span className="text-[9px] text-neural-500 font-mono">{band}</span>
            </div>
          ))}
        </div>

        {/* Controls hint */}
        <div className="absolute top-3 left-3 text-[9px] text-neural-700 font-mono space-y-0.5">
          <div>drag to rotate</div>
          <div>scroll to zoom</div>
          <div>click node for detail</div>
        </div>

        {/* Selected circuit label inside viewer */}
        {selectedDef && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2">
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg border backdrop-blur-sm text-xs font-mono"
              style={{
                borderColor: `${selectedDef.color}50`,
                backgroundColor: `${selectedDef.color}14`,
                color: selectedDef.color,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ background: selectedDef.color, boxShadow: `0 0 6px ${selectedDef.color}` }} />
              {selectedDef.label}
              <button
                type="button"
                className="ml-1 text-neural-600 hover:text-neural-300 transition-colors"
                onClick={() => onCircuitSelect?.('')}
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Detail panel (shown below canvas when a circuit is selected) ──────── */}
      {selectedDef && (
        <div
          className="border-t"
          style={{ borderColor: `${selectedDef.color}28` }}
        >
          <div className="px-5 py-5 grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-6">
            {/* Left: circuit info */}
            <div className="space-y-4">
              {/* Title */}
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: selectedDef.color, boxShadow: `0 0 8px ${selectedDef.color}` }}
                  />
                  <h3 className="text-sm font-mono font-semibold" style={{ color: selectedDef.color }}>
                    {selectedDef.label}
                  </h3>
                </div>
                <p className="text-xs text-neural-400 leading-relaxed">{selectedDef.description}</p>
              </div>

              {/* Key mechanism */}
              <div className="rounded-lg border border-neural-800 bg-neural-900/50 px-3 py-2.5">
                <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-1">
                  Key mechanism
                </span>
                <p className="text-[11px] text-neural-300 leading-relaxed">{selectedDef.keyMechanism}</p>
              </div>

              {/* Regions */}
              <div>
                <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-2">
                  Anatomical nodes ({selectedDef.regions.length})
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {selectedDef.regions.map(r => (
                    <span
                      key={r}
                      className="text-[9px] font-mono px-1.5 py-0.5 rounded border"
                      style={{
                        borderColor: `${selectedDef.color}35`,
                        backgroundColor: `${selectedDef.color}0e`,
                        color: `${selectedDef.color}cc`,
                      }}
                    >
                      {r.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>

              {/* Oscillation signatures */}
              <div>
                <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-2">
                  Neural signatures
                </span>
                <div className="space-y-1.5">
                  {selectedDef.oscillations.map((osc, i) => (
                    <div key={i} className="flex items-center gap-2.5">
                      <span
                        className="text-[10px] font-mono px-2 py-0.5 rounded flex-shrink-0"
                        style={{
                          background: `${BAND_COLOR[osc.band] ?? '#888'}22`,
                          color: BAND_COLOR[osc.band] ?? '#888',
                          border: `1px solid ${BAND_COLOR[osc.band] ?? '#888'}40`,
                        }}
                      >
                        {osc.band}{osc.hz ? ` · ${osc.hz}` : ''}
                      </span>
                      {osc.note && (
                        <span className="text-[10px] text-neural-500">{osc.note}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: links + related disorders */}
            <div className="space-y-4">
              {/* Explore links */}
              <div>
                <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-2.5">
                  Explore in KG
                </span>
                <div className="space-y-2">
                  <Link
                    to={`/search?q=${encodeURIComponent(selectedDef.label)}`}
                    className="flex items-center justify-between px-3 py-2 rounded-lg border border-neural-800 bg-neural-900/40 hover:border-accent-cyan/40 hover:bg-accent-cyan/5 transition-colors group text-xs font-mono"
                  >
                    <span className="text-neural-400 group-hover:text-neural-200 transition-colors">
                      Search papers
                    </span>
                    <span className="text-neural-700 group-hover:text-accent-cyan transition-colors">→</span>
                  </Link>
                  <Link
                    to={`/graph?region=${selectedDef.regions[0]}`}
                    className="flex items-center justify-between px-3 py-2 rounded-lg border border-neural-800 bg-neural-900/40 hover:border-accent-cyan/40 hover:bg-accent-cyan/5 transition-colors group text-xs font-mono"
                  >
                    <span className="text-neural-400 group-hover:text-neural-200 transition-colors">
                      Knowledge Graph
                    </span>
                    <span className="text-neural-700 group-hover:text-accent-cyan transition-colors">→</span>
                  </Link>
                  <Link
                    to="/disorders"
                    className="flex items-center justify-between px-3 py-2 rounded-lg border border-neural-800 bg-neural-900/40 hover:border-accent-cyan/40 hover:bg-accent-cyan/5 transition-colors group text-xs font-mono"
                  >
                    <span className="text-neural-400 group-hover:text-neural-200 transition-colors">
                      Disorder map
                    </span>
                    <span className="text-neural-700 group-hover:text-accent-cyan transition-colors">→</span>
                  </Link>
                </div>
              </div>

              {/* Related disorders */}
              {disorders.length > 0 && (
                <div>
                  <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-2.5">
                    Associated disorders
                  </span>
                  <div className="space-y-1">
                    {disorders.map(d => (
                      <div
                        key={d}
                        className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-neural-800 bg-neural-900/30 text-[10px] font-mono text-neural-500"
                      >
                        <span
                          className="w-1 h-1 rounded-full flex-shrink-0"
                          style={{ background: selectedDef.color }}
                        />
                        {d.replace(/_/g, ' ')}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Oscillation quick-ref */}
              <div className="rounded-lg border border-neural-800 bg-neural-900/30 px-3 py-2.5">
                <span className="text-[9px] font-mono uppercase tracking-wider text-neural-600 block mb-1.5">
                  Search oscillation literature
                </span>
                <div className="flex flex-wrap gap-1">
                  {selectedDef.oscillations.map((osc, i) => (
                    <Link
                      key={i}
                      to={`/search?q=${encodeURIComponent(`${selectedDef.label} ${osc.band}`)}`}
                      className="text-[9px] font-mono px-1.5 py-0.5 rounded transition-colors hover:opacity-80"
                      style={{
                        background: `${BAND_COLOR[osc.band] ?? '#888'}20`,
                        color: BAND_COLOR[osc.band] ?? '#888',
                        border: `1px solid ${BAND_COLOR[osc.band] ?? '#888'}35`,
                      }}
                    >
                      {osc.band}{osc.hz ? ` ${osc.hz}` : ''}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
