import type { ReactNode } from 'react'

export type GlowTone = 'cyan' | 'violet' | 'emerald' | 'amber' | 'neutral'

const GLOW_BORDER: Record<GlowTone, string> = {
  cyan: 'border-accent-cyan/25 shadow-[0_0_50px_-18px_rgba(34,211,238,0.45)]',
  violet: 'border-accent-violet/25 shadow-[0_0_50px_-18px_rgba(139,92,246,0.45)]',
  emerald: 'border-accent-emerald/25 shadow-[0_0_50px_-18px_rgba(16,185,129,0.45)]',
  amber: 'border-amber-500/25 shadow-[0_0_50px_-18px_rgba(245,158,11,0.45)]',
  neutral: 'border-white/10 shadow-[0_20px_50px_-25px_rgba(0,0,0,0.7)]',
}

interface GlassPanelProps {
  children: ReactNode
  tone?: GlowTone
  className?: string
}

export function GlassPanel({ children, tone = 'neutral', className = '' }: GlassPanelProps) {
  return (
    <div
      className={`relative rounded-2xl border backdrop-blur-xl bg-white/[0.04] ${GLOW_BORDER[tone]} ${className}`}
    >
      {children}
    </div>
  )
}
