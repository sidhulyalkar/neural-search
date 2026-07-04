import { useState } from 'react'
import { useSceneBookmarks } from '../../hooks/useSceneBookmarks'
import { BOOKMARK_TAGS, bookmarksForScene, bookmarksToMarkdown, type BookmarkTag } from '../../lib/sceneBookmarks'
import { GlassPanel } from './GlassPanel'

interface SceneBookmarkPanelProps {
  sceneId: string
  sceneUrl: string
  datasetId: string
  datasetTitle: string
  cursorTime: number
  activeAnchorLabel: string | null
  onJump: (time: number) => void
}

/** Scientists should be able to save a timestamp, a note, and a question --
 * localStorage-backed since a scene id is already the durable, shareable
 * handle (see persistence.py on the backend); no server-side bookmark
 * storage needed for a personal annotation list. */
export function SceneBookmarkPanel({
  sceneId,
  sceneUrl,
  datasetId,
  datasetTitle,
  cursorTime,
  activeAnchorLabel,
  onJump,
}: SceneBookmarkPanelProps) {
  const { bookmarks, addBookmark, removeBookmark } = useSceneBookmarks()
  const [note, setNote] = useState('')
  const [tag, setTag] = useState<BookmarkTag>('interesting_moment')

  const sceneBookmarks = bookmarksForScene(bookmarks, sceneId)

  const handleAdd = () => {
    addBookmark({
      sceneId,
      sceneUrl,
      datasetId,
      datasetTitle,
      time: cursorTime,
      anchorLabel: activeAnchorLabel,
      note,
      tag,
    })
    setNote('')
  }

  const handleExport = () => {
    const blob = new Blob([bookmarksToMarkdown(sceneBookmarks)], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${sceneId}_bookmarks.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <GlassPanel tone="amber" className="p-4">
      <p className="text-xs uppercase tracking-widest text-neural-500 mb-3">Bookmarks & notes</p>

      <div className="space-y-2 mb-4">
        <p className="text-xs text-neural-500 font-mono">
          At t = {cursorTime.toFixed(2)}s{activeAnchorLabel ? ` (${activeAnchorLabel})` : ''}
        </p>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="e.g. Trial 128: possible strategy switch before reward omission."
          rows={2}
          className="w-full rounded-lg border border-white/10 bg-neural-950 px-2.5 py-1.5 text-xs text-neural-200 placeholder:text-neural-700 focus:outline-none focus:border-amber-500/40"
        />
        <div className="flex items-center gap-2">
          <select
            value={tag}
            onChange={(event) => setTag(event.target.value as BookmarkTag)}
            className="rounded-lg border border-white/10 bg-neural-950 px-2 py-1 text-xs text-neural-300"
          >
            {BOOKMARK_TAGS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleAdd}
            className="text-xs px-2.5 py-1 rounded-lg border border-amber-500/40 text-amber-300 hover:bg-amber-500/10"
          >
            Add bookmark
          </button>
        </div>
      </div>

      {sceneBookmarks.length === 0 ? (
        <p className="text-xs text-neural-600">No bookmarks yet for this scene.</p>
      ) : (
        <div className="space-y-1.5">
          {sceneBookmarks.map((bookmark) => (
            <div key={bookmark.id} className="rounded-lg border border-white/5 px-2.5 py-1.5">
              <div className="flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => onJump(bookmark.time)}
                  className="text-xs font-mono text-amber-300 hover:text-amber-200"
                >
                  t = {bookmark.time.toFixed(2)}s
                </button>
                <div className="flex items-center gap-2">
                  {bookmark.tag !== 'none' && (
                    <span className="text-[10px] uppercase tracking-wide text-neural-500">
                      {bookmark.tag.replace(/_/g, ' ')}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => removeBookmark(bookmark.id)}
                    className="text-neural-600 hover:text-red-400 text-xs"
                    title="Delete bookmark"
                  >
                    ×
                  </button>
                </div>
              </div>
              {bookmark.note && <p className="text-xs text-neural-400 mt-1">{bookmark.note}</p>}
            </div>
          ))}
          <button
            type="button"
            onClick={handleExport}
            className="text-xs text-neural-500 hover:text-neural-300 mt-2"
          >
            Export notes as Markdown
          </button>
        </div>
      )}
    </GlassPanel>
  )
}
