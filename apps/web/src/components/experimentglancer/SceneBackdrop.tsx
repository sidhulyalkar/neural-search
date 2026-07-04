import { useReducedMotion } from '../../hooks/useReducedMotion'

const GRID_STYLE = {
  backgroundImage:
    'linear-gradient(to right, rgba(255,255,255,0.05) 1px, transparent 1px), ' +
    'linear-gradient(to bottom, rgba(255,255,255,0.05) 1px, transparent 1px)',
  backgroundSize: '48px 48px',
}

/** A precise-instrument backdrop instead of a mood board: a faint measurement
 * grid plus one slow scan sweep, so the strongest visual metaphor in the app
 * is "oscilloscope", not "gradient blob". Color stays reserved for layer
 * state and evidence tiers, not ambience. */
export function SceneBackdrop() {
  const reduceMotion = useReducedMotion()

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-neural-950">
      <div className="absolute inset-0" style={GRID_STYLE} />
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent-cyan/30 to-transparent" />
      {!reduceMotion && (
        <div className="absolute inset-y-0 w-px bg-gradient-to-b from-transparent via-accent-cyan/20 to-transparent animate-scan-sweep" />
      )}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(255,255,255,0.04),transparent_65%)]" />
    </div>
  )
}
