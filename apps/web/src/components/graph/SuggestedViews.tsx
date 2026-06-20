import { useQuery } from '@tanstack/react-query'
import { fetchSuggestedViews } from '../../api/graph'
import type { SuggestedView } from '../../types/graph'

interface SuggestedViewsProps {
  activeSlug: string | null
  companionSlugs: string[]
  onViewSelect: (slug: string) => void
}

export function SuggestedViews({ activeSlug, companionSlugs, onViewSelect }: SuggestedViewsProps) {
  const { data: views = [] } = useQuery<SuggestedView[]>({
    queryKey: ['suggested-views'],
    queryFn: fetchSuggestedViews,
    staleTime: Infinity,
  })

  return (
    <div className="space-y-3">
      <span className="block text-xs uppercase tracking-widest text-neural-600">Views</span>

      <div className="space-y-1">
        {views.map((view) => (
          <button
            key={view.slug}
            type="button"
            onClick={() => onViewSelect(view.slug)}
            className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
              activeSlug === view.slug
                ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20'
                : 'text-neural-500 hover:text-neural-200 hover:bg-neural-900/50'
            }`}
            title={view.description}
          >
            {view.label}
          </button>
        ))}
      </div>

      {companionSlugs.length > 0 && (
        <div>
          <span className="block text-xs text-neural-700 mb-1">Also explore</span>
          <div className="flex flex-wrap gap-1">
            {companionSlugs.map((slug) => {
              const view = views.find((v) => v.slug === slug)
              if (!view) return null
              return (
                <button
                  key={slug}
                  type="button"
                  onClick={() => onViewSelect(slug)}
                  className="text-xs text-neural-600 hover:text-accent-cyan border border-neural-800 hover:border-accent-cyan/30 rounded px-2 py-0.5 transition-colors"
                >
                  {view.label}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
