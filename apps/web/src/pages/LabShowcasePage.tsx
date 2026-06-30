import { useState } from 'react'
import { Link } from 'react-router-dom'
import { NEATLABS, SID_PAPERS } from '../data/neatlabs'

// ── Topic color map (subset matching NEATLab's KG topics) ─────────────────────

const TOPIC_COLORS: Record<string, string> = {
  spectral_dynamics: '#0891b2',
  reward_learning: '#f59e0b',
  neural_synchrony: '#6366f1',
  attention_and_salience: '#7c3aed',
  working_memory: '#0d9488',
  cognitive_control: '#0ea5e9',
  executive_function: '#2563eb',
  decision_making: '#d97706',
  fear_and_anxiety: '#ef4444',
  sleep_and_oscillations: '#4338ca',
  development_and_plasticity: '#15803d',
  neuromodulation: '#f97316',
  emotional_processing: '#db2777',
  social_behavior: '#8b5cf6',
}

const RELEVANCE_BORDER: Record<string, string> = {
  primary: '#22d3ee',
  high: '#8b5cf6',
  medium: '#4b5563',
}

const CIRCUIT_COLORS: Record<string, string> = {
  spectral_dynamics: '#0891b2',
  basal_ganglia_loop: '#f59e0b',
  executive_control: '#0ea5e9',
  default_mode: '#8b5cf6',
  reward_addiction: '#ea580c',
  sleep_oscillations: '#4338ca',
  stress_hpa: '#b45309',
  attention_salience: '#7c3aed',
  neuromodulatory: '#f97316',
}

// ── Small UI atoms ─────────────────────────────────────────────────────────────

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full border"
      style={{ borderColor: `${color}60`, backgroundColor: `${color}18`, color }}
    >
      {label}
    </span>
  )
}

function FundingChip({ name }: { name: string }) {
  return (
    <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-neural-700 bg-neural-900 text-neural-500">
      {name}
    </span>
  )
}

function SpeciesPill({ species }: { species: string }) {
  if (species === 'both') return (
    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-neural-700 bg-neural-800 text-neural-400">
      🐀 + 🧠
    </span>
  )
  if (species === 'rodent') return (
    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-amber-800/40 bg-amber-900/20 text-amber-500">
      🐀 rodent
    </span>
  )
  return (
    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-cyan-800/40 bg-cyan-900/20 text-cyan-400">
      🧠 human
    </span>
  )
}

function SectionHeading({ label, sub }: { label: string; sub?: string }) {
  return (
    <div className="mb-6">
      <span className="block text-[9px] uppercase tracking-widest font-mono text-neural-600 mb-1">
        {sub ?? 'NEATLabs × Neural KG'}
      </span>
      <h2 className="text-xl font-mono text-neural-100 tracking-tight">{label}</h2>
    </div>
  )
}

// ── Translational Pipeline ─────────────────────────────────────────────────────

function TranslationalPipeline() {
  const steps = NEATLABS.translational_rationale.steps
  return (
    <div className="rounded-2xl border border-neural-800 bg-neural-900/40 p-6">
      <div className="grid grid-cols-1 md:grid-cols-[1fr_40px_1fr_40px_1fr] gap-0 items-stretch">
        {steps.map((step, idx) => (
          <>
            <div
              key={step.platform}
              className="rounded-xl border p-4 flex flex-col gap-2"
              style={{
                borderColor:
                  step.species === 'rodent' ? '#92400e80' :
                  step.species === 'both' ? '#1e40af80' :
                  '#0e766880',
                backgroundColor:
                  step.species === 'rodent' ? '#78350f18' :
                  step.species === 'both' ? '#1d4ed818' :
                  '#13474218',
              }}
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">
                  {step.species === 'rodent' ? '🐀' : step.species === 'both' ? '🔬' : '🧠'}
                </span>
                <span
                  className="text-xs font-mono font-semibold"
                  style={{
                    color:
                      step.species === 'rodent' ? '#fbbf24' :
                      step.species === 'both' ? '#93c5fd' :
                      '#34d399',
                  }}
                >
                  {step.platform}
                </span>
              </div>
              <p className="text-[11px] text-neural-200 font-medium leading-snug">{step.what}</p>
              <p className="text-[10px] text-neural-500 leading-snug mt-auto">{step.why}</p>
            </div>
            {idx < steps.length - 1 && (
              <div key={`arrow-${idx}`} className="flex items-center justify-center text-neural-700 text-xl select-none">
                →
              </div>
            )}
          </>
        ))}
      </div>
      <p className="text-[11px] text-neural-500 italic mt-4 leading-relaxed border-t border-neural-800 pt-4">
        {NEATLABS.translational_rationale.insight}
      </p>
    </div>
  )
}

// ── Topic Fingerprint ──────────────────────────────────────────────────────────

function TopicFingerprint() {
  const [hovered, setHovered] = useState<string | null>(null)

  const grouped = {
    primary: NEATLABS.kg_topics.filter((t) => t.relevance === 'primary'),
    high: NEATLABS.kg_topics.filter((t) => t.relevance === 'high'),
    medium: NEATLABS.kg_topics.filter((t) => t.relevance === 'medium'),
  }

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([rel, topics]) => (
        <div key={rel} className="flex items-start gap-3">
          <span className="text-[9px] font-mono uppercase tracking-wider text-neural-700 w-16 pt-1 flex-shrink-0">
            {rel}
          </span>
          <div className="flex flex-wrap gap-1.5">
            {topics.map((t) => {
              const color = TOPIC_COLORS[t.id] ?? '#6b7280'
              const border = RELEVANCE_BORDER[rel as keyof typeof RELEVANCE_BORDER]
              return (
                <div key={t.id} className="relative">
                  <span
                    className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full border cursor-default transition-all"
                    style={{
                      borderColor: hovered === t.id ? color : `${border}50`,
                      backgroundColor: hovered === t.id ? `${color}25` : `${color}12`,
                      color: hovered === t.id ? color : `${color}cc`,
                    }}
                    onMouseEnter={() => setHovered(t.id)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    {t.id.replace(/_/g, ' ')}
                  </span>
                  {hovered === t.id && (
                    <div className="absolute z-10 top-full left-0 mt-1.5 w-56 bg-neural-900 border border-neural-700 rounded-lg px-3 py-2 shadow-xl pointer-events-none">
                      <p className="text-[10px] text-neural-300 leading-snug">{t.note}</p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Circuit overlap panel ──────────────────────────────────────────────────────

function CircuitOverlap() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
      {NEATLABS.kg_circuits.map((c) => {
        const color = CIRCUIT_COLORS[c.id] ?? '#6b7280'
        const isCore = c.relevance === 'primary'
        return (
          <div
            key={c.id}
            className="rounded-xl border p-3 flex flex-col gap-1.5"
            style={{
              borderColor: isCore ? `${color}60` : `${color}30`,
              backgroundColor: isCore ? `${color}12` : `${color}06`,
            }}
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
              <span className="text-[11px] font-mono text-neural-200 leading-tight">
                {c.id.replace(/_/g, ' ')}
              </span>
              {isCore && (
                <span className="ml-auto text-[9px] font-mono px-1 py-0 rounded" style={{ color, backgroundColor: `${color}20` }}>
                  core
                </span>
              )}
            </div>
            <p className="text-[9px] text-neural-600 leading-snug">{c.note}</p>
          </div>
        )
      })}
    </div>
  )
}

// ── Paper card ─────────────────────────────────────────────────────────────────

type Paper = typeof NEATLABS.highlight_papers[number]

function PaperCard({ paper }: { paper: Paper }) {
  const isContributed = 'highlight' in paper && paper.highlight === 'contributed'

  return (
    <div
      className={`rounded-xl border p-4 flex flex-col gap-2.5 transition-colors ${
        isContributed
          ? 'border-accent-cyan/40 bg-accent-cyan/5'
          : 'border-neural-800 bg-neural-900/40 hover:border-neural-700'
      }`}
    >
      {/* Year + species + venue */}
      <div className="flex items-start gap-2 flex-wrap">
        <span className="text-[10px] font-mono text-neural-600 flex-shrink-0">{paper.year}</span>
        <SpeciesPill species={paper.species} />
        {'citations' in paper && (
          <span className="text-[9px] font-mono text-neural-700 flex-shrink-0 ml-auto">
            {paper.citations} citations
          </span>
        )}
        {isContributed && (
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-accent-cyan/40 bg-accent-cyan/10 text-accent-cyan ml-auto">
            contributed
          </span>
        )}
      </div>

      {/* Title */}
      <p className="text-sm font-mono text-neural-100 leading-snug">{paper.title}</p>

      {/* Venue */}
      {'venue' in paper && paper.venue && (
        <p className="text-[10px] text-neural-600 italic">{paper.venue}</p>
      )}

      {/* PI */}
      {'pi' in paper && paper.pi && (
        <p className="text-[10px] text-neural-600">
          {paper.pi === 'both' ? 'Mishra + Ramanathan' : `PI: ${paper.pi}`}
        </p>
      )}

      {/* Topic + region tags */}
      <div className="flex flex-wrap gap-1 mt-0.5">
        {paper.topics.map((t) => (
          <Badge key={t} label={t.replace(/_/g, ' ')} color={TOPIC_COLORS[t] ?? '#6b7280'} />
        ))}
      </div>

      {/* Significance */}
      <p className="text-[10px] text-neural-500 leading-relaxed border-t border-neural-800 pt-2 mt-0.5">
        {paper.significance}
      </p>
    </div>
  )
}

// ── Technology cards ───────────────────────────────────────────────────────────

function TechGrid() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      {NEATLABS.technologies.map((tech) => {
        const speciesColor =
          tech.species === 'rodent' ? '#f59e0b' :
          tech.species === 'human' ? '#22d3ee' :
          '#8b5cf6'
        return (
          <div
            key={tech.id}
            className="rounded-xl border border-neural-800 bg-neural-900/40 p-4 flex flex-col gap-2"
          >
            <span className="text-2xl">{tech.icon}</span>
            <span className="text-xs font-mono font-semibold text-neural-200">{tech.label}</span>
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: speciesColor }} />
              <span className="text-[9px] font-mono" style={{ color: speciesColor }}>
                {tech.species === 'both' ? 'cross-species' : tech.species}
              </span>
            </div>
            <p className="text-[9px] text-neural-600 leading-snug">{tech.description}</p>
          </div>
        )
      })}
    </div>
  )
}

// ── KG Value Propositions ──────────────────────────────────────────────────────

function ValuePropositions() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {NEATLABS.kg_value_propositions.map((vp) => (
        <div
          key={vp.title}
          className="rounded-xl border border-neural-800 bg-neural-900/40 p-4 flex flex-col gap-2 hover:border-neural-700 transition-colors"
        >
          <span className="text-xl">{vp.icon}</span>
          <p className="text-sm font-mono text-neural-200 font-medium">{vp.title}</p>
          <p className="text-[10px] text-neural-500 leading-relaxed">{vp.description}</p>
        </div>
      ))}
    </div>
  )
}

// ── Director Card ──────────────────────────────────────────────────────────────

function DirectorCard({ director }: { director: typeof NEATLABS.directors[number] }) {
  return (
    <a
      href={director.scholar}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-xl border border-neural-800 bg-neural-900/40 p-4 hover:border-accent-cyan/40 transition-colors group"
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-neural-800 border border-neural-700 flex items-center justify-center text-sm flex-shrink-0">
          {director.name.split(' ')[1][0]}
        </div>
        <div className="min-w-0">
          <p className="text-xs font-mono text-neural-200 group-hover:text-white transition-colors">{director.name}</p>
          <p className="text-[9px] text-neural-600 mt-0.5">{director.role}</p>
          <p className="text-[10px] text-neural-500 mt-1.5 leading-snug">{director.focus}</p>
        </div>
        <span className="text-neural-700 text-xs ml-auto group-hover:text-accent-cyan transition-colors flex-shrink-0">↗</span>
      </div>
    </a>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────────

export function LabShowcasePage() {
  const [paperFilter, setPaperFilter] = useState<'all' | 'rodent' | 'human' | 'both'>('all')

  const visiblePapers = NEATLABS.highlight_papers.filter(
    (p) => paperFilter === 'all' || p.species === paperFilter || (paperFilter === 'both' && p.species === 'both')
  )

  return (
    <div className="max-w-5xl mx-auto px-6 lg:px-8 py-10 space-y-16">

      {/* ── Hero ── */}
      <section>
        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-900 to-emerald-900 border border-neural-700 flex items-center justify-center text-xl flex-shrink-0">
            🔬
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <a
                href={NEATLABS.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs font-mono text-neural-500 hover:text-accent-cyan transition-colors"
              >
                {NEATLABS.institution} ↗
              </a>
            </div>
            <h1 className="text-2xl md:text-3xl font-mono font-bold text-neural-100 tracking-tight leading-tight">
              {NEATLABS.name}
            </h1>
            <p className="text-base font-mono text-accent-cyan mt-1">{NEATLABS.acronym}</p>
          </div>
        </div>

        <p className="text-sm text-neural-400 leading-relaxed max-w-3xl mb-6">{NEATLABS.mission}</p>

        {/* Translational tagline */}
        <div className="inline-flex items-center gap-3 rounded-xl border border-neural-700 bg-neural-900/60 px-4 py-3 mb-6">
          <span className="text-sm">🐀</span>
          <span className="text-xs font-mono text-neural-500">Animal Models</span>
          <span className="text-neural-700 text-sm">→</span>
          <span className="text-xs font-mono text-neural-400">Standardized Paradigms</span>
          <span className="text-neural-700 text-sm">→</span>
          <span className="text-xs font-mono text-neural-400">Human Brain Models</span>
          <span className="text-sm">🧠</span>
        </div>

        {/* Co-directors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
          {NEATLABS.directors.map((d) => <DirectorCard key={d.name} director={d} />)}
        </div>

        {/* Funding + Clinical Trials */}
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-[9px] font-mono text-neural-700 uppercase tracking-wider">Funding:</span>
          {NEATLABS.funding.map((f) => <FundingChip key={f} name={f} />)}
        </div>
        <div className="flex flex-wrap gap-2 items-center mt-2">
          <span className="text-[9px] font-mono text-neural-700 uppercase tracking-wider">Active Trials:</span>
          {NEATLABS.clinical_trials.map((t) => (
            <a
              key={t.nct}
              href={`https://clinicaltrials.gov/study/${t.nct}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] font-mono px-2 py-0.5 rounded border border-neural-700 bg-neural-900 text-neural-500 hover:text-accent-cyan hover:border-accent-cyan/50 transition-colors"
            >
              {t.nct} — {t.title}
            </a>
          ))}
        </div>
      </section>

      {/* ── Translational Pipeline ── */}
      <section>
        <SectionHeading
          label="Animal → Human Translation Pipeline"
          sub="Core Research Strategy"
        />
        <TranslationalPipeline />
      </section>

      {/* ── Research Topic Fingerprint ── */}
      <section>
        <SectionHeading
          label="Research Topic Fingerprint"
          sub="KG Coverage — 14 of 26 topics"
        />
        <p className="text-xs text-neural-600 mb-4 leading-snug">
          Topics from the neuroscience KG that NEATLabs publications directly address.
          Hover any topic badge to see the specific finding context. Primary topics are the lab's strongest signal in the corpus.
        </p>
        <TopicFingerprint />
        <div className="flex items-center gap-4 mt-4">
          {(['primary', 'high', 'medium'] as const).map((rel) => (
            <div key={rel} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: RELEVANCE_BORDER[rel] }} />
              <span className="text-[9px] font-mono text-neural-600">{rel} relevance</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Circuit Overlap ── */}
      <section>
        <SectionHeading
          label="Functional Circuit Overlap"
          sub="Brain Atlas × NEATLabs Research"
        />
        <p className="text-xs text-neural-600 mb-4 leading-snug">
          Which of the 20 functional circuits in the Brain Atlas map onto NEATLabs' published work.
          Core circuits are where the lab has published primary mechanistic data.
        </p>
        <CircuitOverlap />
        <div className="flex gap-3 mt-4">
          <Link
            to="/atlas?tab=circuits"
            className="text-xs font-mono text-accent-cyan hover:underline"
          >
            Explore all circuits in Brain Atlas →
          </Link>
        </div>
      </section>

      {/* ── Landmark Papers ── */}
      <section>
        <SectionHeading
          label="Landmark Publications"
          sub="Selected from Google Scholar"
        />
        <p className="text-xs text-neural-600 mb-4 leading-snug">
          Key papers from Dr. Mishra and Dr. Ramanathan's Google Scholar profiles, annotated with KG topics,
          target regions, and translational significance. Papers marked <span className="text-accent-cyan font-mono">contributed</span> are
          direct lab output you were part of.
        </p>

        {/* Species filter */}
        <div className="flex gap-2 mb-5">
          {(['all', 'rodent', 'human', 'both'] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setPaperFilter(f)}
              className={`text-[10px] font-mono px-2.5 py-1 rounded-lg border transition-colors ${
                paperFilter === f
                  ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10'
                  : 'border-neural-800 text-neural-600 hover:border-neural-700 hover:text-neural-400'
              }`}
            >
              {f === 'all' ? 'All' : f === 'rodent' ? '🐀 Rodent' : f === 'human' ? '🧠 Human' : '🔬 Cross-species'}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {visiblePapers.map((paper) => (
            <PaperCard key={paper.title} paper={paper as Paper} />
          ))}
        </div>
      </section>

      {/* ── Your Research ── */}
      <section>
        <SectionHeading
          label="Your Contributions"
          sub="Sidharth Hulyalkar — Google Scholar"
        />
        <p className="text-xs text-neural-600 mb-4 leading-snug max-w-2xl">
          Your 6 publications from NEATLabs form a coherent research thread: open-source hardware (operant box, chronic probes)
          enabling multi-site rodent LFP → network mapping (DMN suppression, impulsivity circuits)
          → oscillatory reward coding (cortico-striatal beta/gamma as a certainty signal).
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          {SID_PAPERS.map((paper) => (
            <div
              key={paper.title}
              className="rounded-xl border border-accent-cyan/30 bg-accent-cyan/5 p-4 flex flex-col gap-2"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] font-mono text-neural-600">{paper.year}</span>
                <SpeciesPill species={paper.species} />
                <span className="text-[9px] font-mono text-neural-700 ml-auto">{paper.citations} citations</span>
              </div>
              <p className="text-sm font-mono text-neural-100 leading-snug">{paper.title}</p>
              <p className="text-[10px] text-neural-600 italic">{paper.venue}</p>
              <div className="flex flex-wrap gap-1">
                {paper.topics.map((t) => (
                  <Badge key={t} label={t.replace(/_/g, ' ')} color={TOPIC_COLORS[t] ?? '#6b7280'} />
                ))}
              </div>
              <p className="text-[10px] text-neural-500 leading-relaxed border-t border-neural-800 pt-2">
                {paper.significance}
              </p>
            </div>
          ))}
        </div>
        <div className="rounded-xl border border-neural-800 bg-neural-900/40 p-4">
          <p className="text-[11px] text-neural-400 leading-relaxed">
            <span className="text-accent-cyan font-mono">Research arc:</span>{' '}
            The cortico-striatal beta oscillation is your primary signal of interest — a mesoscale biomarker
            sitting at the intersection of dopaminergic reward circuits, oscillatory dynamics, and translatable
            cognitive paradigms. Your work grounds the KG's most data-sparse gap: cross-species LFP
            validation of reward circuit oscillations.
          </p>
        </div>
      </section>

      {/* ── Platform Technologies ── */}
      <section>
        <SectionHeading
          label="Platforms & Technologies"
          sub="Lab-Built Tools"
        />
        <p className="text-xs text-neural-600 mb-4 leading-snug">
          NEATLabs builds its own platforms to enable direct cross-species comparison.
          BrainE and BrainER are designed to run identical paradigms in humans and rodents.
        </p>
        <TechGrid />
      </section>

      {/* ── KG Value Propositions ── */}
      <section>
        <SectionHeading
          label="How This KG Amplifies NEATLabs"
          sub="Neuroscience Knowledge Graph Use Cases"
        />
        <p className="text-xs text-neural-600 mb-6 leading-snug max-w-2xl">
          The translational mission of NEATLabs is exactly what this knowledge graph is built to support.
          A shared ontology across species, circuit-level coverage maps, and citation ancestry tools create
          infrastructure for the kind of cross-species meta-analysis NEATLabs pioneered.
        </p>
        <ValuePropositions />

        <div className="mt-6 rounded-xl border border-neural-800 bg-neural-900/40 p-5">
          <p className="text-xs font-mono text-accent-cyan mb-2">The Translational Hypothesis</p>
          <p className="text-sm text-neural-300 leading-relaxed">
            By standardizing cognitive paradigms and neural metrics across species — using platforms like BrainER and BrainE —
            NEATLabs creates the ground truth needed to build better computational models of the human brain.
            Rodent data reveals <em className="text-neural-200">mechanisms</em>; human data reveals <em className="text-neural-200">variability</em>.
            Together, they define the design space for any model that must generalize across individuals and translate to clinical use.
          </p>
          <div className="flex gap-3 mt-4 flex-wrap">
            <Link to="/graph" className="text-xs font-mono text-accent-cyan hover:underline">
              Explore KG Topics →
            </Link>
            <Link to="/atlas" className="text-xs font-mono text-accent-emerald hover:underline">
              Open Brain Atlas →
            </Link>
            <Link to="/coverage" className="text-xs font-mono text-accent-violet hover:underline">
              View Coverage Gaps →
            </Link>
          </div>
        </div>
      </section>

    </div>
  )
}
