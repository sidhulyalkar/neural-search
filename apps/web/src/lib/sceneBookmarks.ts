export type BookmarkTag =
  | 'none'
  | 'interesting_moment'
  | 'artifact'
  | 'strategy_switch'
  | 'model_error'
  | 'replay'
  | 'uncertainty'

export const BOOKMARK_TAGS: Array<{ id: BookmarkTag; label: string }> = [
  { id: 'none', label: 'No tag' },
  { id: 'interesting_moment', label: 'Interesting moment' },
  { id: 'artifact', label: 'Artifact' },
  { id: 'strategy_switch', label: 'Strategy switch' },
  { id: 'model_error', label: 'Model error' },
  { id: 'replay', label: 'Replay' },
  { id: 'uncertainty', label: 'Uncertainty' },
]

export interface SceneBookmark {
  id: string
  sceneId: string
  sceneUrl: string
  datasetId: string
  datasetTitle: string
  time: number
  anchorLabel: string | null
  note: string
  tag: BookmarkTag
  createdAt: string
}

export const SCENE_BOOKMARKS_STORAGE_KEY = 'experimentglancer.bookmarks.v1'

export function bookmarksForScene(bookmarks: SceneBookmark[], sceneId: string): SceneBookmark[] {
  return bookmarks
    .filter((bookmark) => bookmark.sceneId === sceneId)
    .sort((a, b) => a.time - b.time)
}

export function bookmarksToMarkdown(bookmarks: SceneBookmark[]): string {
  if (bookmarks.length === 0) return '# Scene bookmarks\n\n_No bookmarks yet._\n'
  const lines = ['# Scene bookmarks', '']
  for (const bookmark of bookmarks) {
    lines.push(`## ${bookmark.datasetTitle} — t = ${bookmark.time.toFixed(2)}s`)
    if (bookmark.anchorLabel) lines.push(`Anchor: ${bookmark.anchorLabel}`)
    if (bookmark.tag !== 'none') lines.push(`Tag: ${bookmark.tag.replace(/_/g, ' ')}`)
    if (bookmark.note) lines.push('', bookmark.note)
    lines.push('', `[Open scene](${bookmark.sceneUrl})`, '')
  }
  return lines.join('\n')
}
