import { SCENE_BOOKMARKS_STORAGE_KEY, type BookmarkTag, type SceneBookmark } from '../lib/sceneBookmarks'
import { useLocalStorageState } from './useLocalStorageState'

interface AddBookmarkInput {
  sceneId: string
  sceneUrl: string
  datasetId: string
  datasetTitle: string
  time: number
  anchorLabel: string | null
  note: string
  tag: BookmarkTag
}

export function useSceneBookmarks() {
  const [bookmarks, setBookmarks] = useLocalStorageState<SceneBookmark[]>(SCENE_BOOKMARKS_STORAGE_KEY, [])

  const addBookmark = (input: AddBookmarkInput): SceneBookmark => {
    const bookmark: SceneBookmark = {
      id: `bm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      createdAt: new Date().toISOString(),
      ...input,
    }
    setBookmarks((current) => [...current, bookmark])
    return bookmark
  }

  const removeBookmark = (id: string) => {
    setBookmarks((current) => current.filter((bookmark) => bookmark.id !== id))
  }

  const updateBookmark = (id: string, updates: Partial<Pick<SceneBookmark, 'note' | 'tag'>>) => {
    setBookmarks((current) =>
      current.map((bookmark) => (bookmark.id === id ? { ...bookmark, ...updates } : bookmark)),
    )
  }

  return { bookmarks, addBookmark, removeBookmark, updateBookmark }
}
