import { useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────

type LobeId =
  | 'frontal_lobe'
  | 'parietal_lobe'
  | 'temporal_lobe'
  | 'occipital_lobe'
  | 'cerebellum'
  | 'brainstem'

type LobeDefinition = {
  id: LobeId
  label: string
  path: string
  labelX: number
  labelY: number
  regionIds: string[]
}

export type BrainSchematicProps = {
  countMap: Map<string, { n: number; label: string }>
  selectedLobe: string | null
  onLobeClick: (lobeId: string, regionIds: string[]) => void
}

// ── Region mapping ────────────────────────────────────────────────────────────

const LOBE_DEFINITIONS: LobeDefinition[] = [
  {
    id: 'frontal_lobe',
    label: 'Frontal',
    // Occupies top-left portion: PFC, motor cortex, premotor areas
    path: 'M 30,180 Q 20,140 30,100 Q 40,65 70,45 Q 100,28 130,22 Q 155,18 170,25 Q 185,32 185,55 Q 182,80 175,105 Q 165,130 150,148 Q 135,165 115,172 Q 90,180 70,182 Z',
    labelX: 100,
    labelY: 105,
    regionIds: [
      'prefrontal_cortex', 'dlpfc', 'ofc', 'vlpfc', 'anterior_cingulate_cortex',
      'motor_cortex', 'm1', 'premotor_cortex', 'supplementary_motor_area',
      'orbitofrontal_cortex', 'medial_prefrontal_cortex',
    ],
  },
  {
    id: 'parietal_lobe',
    label: 'Parietal',
    // Top-middle: somatosensory, posterior parietal
    path: 'M 175,105 Q 182,80 185,55 Q 215,45 240,50 Q 265,58 270,80 Q 272,105 262,128 Q 250,150 232,162 Q 212,170 190,168 Q 165,162 150,148 Q 165,130 175,105 Z',
    labelX: 215,
    labelY: 108,
    regionIds: [
      'somatosensory_cortex', 's1', 's2', 'barrel_cortex',
      'parietal_cortex', 'posterior_parietal_cortex', 'inferior_parietal_lobule',
    ],
  },
  {
    id: 'temporal_lobe',
    label: 'Temporal',
    // Bottom-middle: auditory cortex, hippocampus, inferior temporal
    path: 'M 115,172 Q 135,165 150,148 Q 190,168 232,162 Q 240,195 228,220 Q 215,242 195,252 Q 175,260 155,255 Q 130,248 115,235 Q 98,220 95,200 Q 95,185 115,172 Z',
    labelX: 170,
    labelY: 210,
    regionIds: [
      'temporal_cortex', 'auditory_cortex', 'hippocampus', 'inferior_temporal_cortex',
      'ca1', 'ca3', 'dentate_gyrus', 'entorhinal_cortex', 'subiculum',
      'parahippocampal_cortex', 'medial_geniculate',
    ],
  },
  {
    id: 'occipital_lobe',
    label: 'Occipital',
    // Right portion: visual cortex V1-V4
    path: 'M 262,128 Q 272,105 270,80 Q 295,75 312,90 Q 330,108 330,135 Q 328,160 312,175 Q 295,188 275,188 Q 255,186 240,175 Q 232,162 250,150 Q 262,128 262,128 Z',
    labelX: 288,
    labelY: 133,
    regionIds: [
      'visual_cortex', 'v1', 'v2', 'v4', 'area_mt', 'lateral_geniculate',
    ],
  },
  {
    id: 'cerebellum',
    label: 'Cerebellum',
    // Bottom-right
    path: 'M 275,188 Q 295,188 312,175 Q 330,160 335,180 Q 340,205 330,225 Q 318,245 298,250 Q 278,255 262,245 Q 248,235 245,218 Q 242,200 250,188 Q 262,186 275,188 Z',
    labelX: 292,
    labelY: 220,
    regionIds: [
      'cerebellum', 'cerebellar_cortex', 'deep_cerebellar_nuclei',
    ],
  },
  {
    id: 'brainstem',
    label: 'Brainstem',
    // Far right, thin vertical strip
    path: 'M 330,225 Q 335,205 335,180 Q 348,175 360,182 Q 372,190 372,210 Q 372,235 360,248 Q 350,258 338,255 Q 326,250 325,238 Q 326,232 330,225 Z',
    labelX: 350,
    labelY: 218,
    regionIds: [
      'midbrain', 'superior_colliculus', 'inferior_colliculus',
      'substantia_nigra', 'ventral_tegmental_area', 'periaqueductal_gray',
      'pons', 'locus_coeruleus', 'raphe_nucleus', 'medulla',
    ],
  },
]

// ── Color helpers ─────────────────────────────────────────────────────────────

function coverageColor(n: number): string {
  if (n === 0) return '#1e293b'   // neural-800 approximate
  if (n < 5)   return '#450a0a'   // red-950
  if (n < 30)  return '#431407'   // orange-950
  if (n < 100) return '#422006'   // yellow-950 dark
  if (n < 400) return '#022c22'   // emerald-950
  return '#064e3b'                 // emerald-900
}

function strokeColor(n: number): string {
  if (n === 0) return '#334155'
  if (n < 5)   return '#991b1b'
  if (n < 30)  return '#c2410c'
  if (n < 100) return '#a16207'
  if (n < 400) return '#047857'
  return '#059669'
}

function lobeDatasets(
  lobe: LobeDefinition,
  countMap: Map<string, { n: number; label: string }>
): number {
  return lobe.regionIds.reduce((sum, rid) => sum + (countMap.get(rid)?.n ?? 0), 0)
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

type TooltipState = {
  lobeId: LobeId
  label: string
  total: number
  x: number
  y: number
} | null

// ── Component ─────────────────────────────────────────────────────────────────

export function BrainSchematic({ countMap, selectedLobe, onLobeClick }: BrainSchematicProps) {
  const [tooltip, setTooltip] = useState<TooltipState>(null)

  function handleMouseEnter(lobe: LobeDefinition, e: React.MouseEvent<SVGPathElement>) {
    const total = lobeDatasets(lobe, countMap)
    const rect = (e.currentTarget.closest('svg') as SVGSVGElement).getBoundingClientRect()
    setTooltip({
      lobeId: lobe.id,
      label: lobe.label,
      total,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top - 36,
    })
  }

  function handleMouseLeave() {
    setTooltip(null)
  }

  function handleMouseMove(e: React.MouseEvent<SVGPathElement>) {
    if (!tooltip) return
    const rect = (e.currentTarget.closest('svg') as SVGSVGElement).getBoundingClientRect()
    setTooltip((prev) =>
      prev ? { ...prev, x: e.clientX - rect.left, y: e.clientY - rect.top - 36 } : null
    )
  }

  return (
    <div className="relative bg-neural-950 border border-neural-800 rounded-xl overflow-hidden">
      <div className="px-3 py-2 border-b border-neural-800">
        <span className="text-xs font-mono text-neural-500 uppercase tracking-wider">
          Lateral View — Anatomical Regions
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox="0 0 400 278"
          className="w-full"
          style={{ maxHeight: '260px' }}
          aria-label="Brain lateral view schematic"
        >
          {/* Dark background */}
          <rect width="400" height="278" fill="#050d1a" />

          {/* Brain outline — subtle outer silhouette */}
          <path
            d="M 30,180 Q 20,140 30,100 Q 40,65 70,45 Q 100,28 130,22 Q 155,18 185,25 Q 215,32 250,38 Q 285,44 312,65 Q 340,88 345,120 Q 350,150 340,180 Q 332,210 315,232 Q 296,255 268,262 Q 240,268 200,265 Q 160,262 130,258 Q 100,252 75,238 Q 48,222 30,195 Z"
            fill="none"
            stroke="#1e2d42"
            strokeWidth="1"
          />

          {/* Lobe regions */}
          {LOBE_DEFINITIONS.map((lobe) => {
            const total = lobeDatasets(lobe, countMap)
            const isSelected = selectedLobe === lobe.id
            const isHovered = tooltip?.lobeId === lobe.id
            const fill = coverageColor(total)
            const stroke = isSelected ? '#22d3ee' : strokeColor(total)
            const strokeWidth = isSelected ? 2 : 1
            const opacity = isHovered && !isSelected ? 0.9 : 0.75

            return (
              <g key={lobe.id}>
                <path
                  d={lobe.path}
                  fill={fill}
                  fillOpacity={opacity}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  strokeLinejoin="round"
                  className={`cursor-pointer transition-all duration-150 ${isSelected ? 'lobe-selected' : ''}`}
                  onClick={() => onLobeClick(lobe.id, lobe.regionIds)}
                  onMouseEnter={(e) => handleMouseEnter(lobe, e)}
                  onMouseLeave={handleMouseLeave}
                  onMouseMove={handleMouseMove}
                  role="button"
                  aria-label={`${lobe.label}: ${total} datasets`}
                />
                {/* Lobe label */}
                <text
                  x={lobe.labelX}
                  y={lobe.labelY}
                  textAnchor="middle"
                  className="pointer-events-none"
                  style={{
                    fontSize: lobe.id === 'brainstem' ? '7px' : '8px',
                    fontFamily: 'ui-monospace, monospace',
                    fill: isSelected ? '#22d3ee' : '#94a3b8',
                    fontWeight: isSelected ? '700' : '400',
                    letterSpacing: '0.02em',
                  }}
                >
                  {lobe.label}
                </text>
              </g>
            )
          })}

          {/* Tooltip — rendered inside SVG for correct positioning */}
          {tooltip && (
            <g transform={`translate(${Math.min(tooltip.x, 320)}, ${Math.max(tooltip.y, 8)})`}>
              <rect
                x={-4}
                y={-14}
                width={120}
                height={30}
                rx={3}
                fill="#0f172a"
                stroke="#334155"
                strokeWidth={1}
              />
              <text
                x={56}
                y={-2}
                textAnchor="middle"
                style={{ fontSize: '9px', fontFamily: 'ui-monospace, monospace', fill: '#e2e8f0' }}
              >
                {tooltip.label}
              </text>
              <text
                x={56}
                y={10}
                textAnchor="middle"
                style={{ fontSize: '8px', fontFamily: 'ui-monospace, monospace', fill: '#64748b' }}
              >
                {tooltip.total.toLocaleString()} datasets
              </text>
            </g>
          )}
        </svg>

        {/* Selected lobe pulse animation */}
        <style>{`
          @keyframes lobe-pulse {
            0%, 100% { stroke-opacity: 1; }
            50% { stroke-opacity: 0.5; }
          }
          .lobe-selected {
            animation: lobe-pulse 2s ease-in-out infinite;
          }
        `}</style>
      </div>

      {/* Coverage legend strip */}
      <div className="px-3 py-2 border-t border-neural-800 flex items-center gap-2 flex-wrap">
        {[
          { label: '0', color: '#1e293b', border: '#334155' },
          { label: '1–4', color: '#450a0a', border: '#991b1b' },
          { label: '5–29', color: '#431407', border: '#c2410c' },
          { label: '30–99', color: '#422006', border: '#a16207' },
          { label: '100+', color: '#022c22', border: '#047857' },
        ].map((e) => (
          <div key={e.label} className="flex items-center gap-1">
            <div
              className="w-2.5 h-2.5 rounded-sm border"
              style={{ backgroundColor: e.color, borderColor: e.border }}
            />
            <span className="text-xs text-neural-600 font-mono">{e.label}</span>
          </div>
        ))}
        <span className="text-xs text-neural-700 ml-1">datasets</span>
      </div>
    </div>
  )
}
