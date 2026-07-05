interface LayerKindIconProps {
  kind: string
  className?: string
}

const SHARED_SVG_PROPS = {
  fill: 'none',
  stroke: 'currentColor',
  viewBox: '0 0 24 24',
  xmlns: 'http://www.w3.org/2000/svg',
} as const

export function LayerKindIcon({ kind, className = 'w-4 h-4' }: LayerKindIconProps) {
  switch (kind) {
    case 'neural.spikes':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2 12h3l2-7 3 14 3-10 2 5h6" />
        </svg>
      )
    case 'neural.calcium':
    case 'neural.lfp':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0" />
        </svg>
      )
    case 'neural.population_heatmap':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <rect x="3" y="3" width="7" height="7" rx="1" strokeWidth={2} />
          <rect x="14" y="3" width="7" height="7" rx="1" strokeWidth={2} />
          <rect x="3" y="14" width="7" height="7" rx="1" strokeWidth={2} />
          <rect x="14" y="14" width="7" height="7" rx="1" strokeWidth={2} />
        </svg>
      )
    case 'behavior.pupil':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z" />
          <circle cx="12" cy="12" r="3" strokeWidth={2} />
        </svg>
      )
    case 'behavior.licks':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3s6 6.5 6 11a6 6 0 01-12 0c0-4.5 6-11 6-11z" />
        </svg>
      )
    case 'behavior.wheel':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <circle cx="12" cy="12" r="8" strokeWidth={2} />
          <path strokeLinecap="round" strokeWidth={2} d="M12 4v4M12 16v4M4 12h4M16 12h4" />
        </svg>
      )
    case 'behavior.pose':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <circle cx="12" cy="5" r="2" strokeWidth={2} />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 7v6m0 0l-4 6m4-6l4 6m-4-9l-5 3m5-3l5 3" />
        </svg>
      )
    case 'video.frames':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <rect x="3" y="5" width="14" height="14" rx="2" strokeWidth={2} />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9l4-3v12l-4-3" />
        </svg>
      )
    case 'stimulus.identity':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <circle cx="12" cy="12" r="9" strokeWidth={2} />
          <circle cx="12" cy="12" r="4" strokeWidth={2} />
        </svg>
      )
    case 'reward.delivery':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V8z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8h18v3H3zM12 8v13" />
        </svg>
      )
    case 'timeline.trials':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <rect x="3" y="10" width="4" height="7" rx="1" strokeWidth={2} />
          <rect x="10" y="6" width="4" height="11" rx="1" strokeWidth={2} />
          <rect x="17" y="13" width="4" height="4" rx="1" strokeWidth={2} />
        </svg>
      )
    case 'timeline.events':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 12h16M8 6l-4 6 4 6M16 6l4 6-4 6" />
        </svg>
      )
    case 'model.predictions':
    case 'model.latent_state':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <circle cx="12" cy="12" r="3" strokeWidth={2} />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v3m0 12v3m9-9h-3M6 12H3" />
        </svg>
      )
    case 'provenance.evidence':
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3l7 4v5c0 4.4-3 8.4-7 9.5C8 20.4 5 16.4 5 12V7l7-4z" />
        </svg>
      )
    case 'metadata.labels':
    default:
      return (
        <svg className={className} {...SHARED_SVG_PROPS}>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4h8l8 8-8 8-8-8V4z" />
          <circle cx="8" cy="8" r="1.2" fill="currentColor" strokeWidth={0} />
        </svg>
      )
  }
}
