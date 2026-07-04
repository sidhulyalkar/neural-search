import { useState } from 'react'

/** Same shape as `useState`, but persisted to localStorage under `key`.
 * Falls back silently to in-memory-only state if localStorage is
 * unavailable (private browsing, quota exceeded) -- persistence is a nice-
 * to-have here, not a correctness requirement. */
export function useLocalStorageState<T>(
  key: string,
  defaultValue: T,
): [T, (next: T | ((previous: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key)
      return raw ? (JSON.parse(raw) as T) : defaultValue
    } catch {
      return defaultValue
    }
  })

  const setAndPersist = (next: T | ((previous: T) => T)) => {
    setValue((previous) => {
      const resolved = typeof next === 'function' ? (next as (previous: T) => T)(previous) : next
      try {
        window.localStorage.setItem(key, JSON.stringify(resolved))
      } catch {
        // Ignore write failures (quota, private mode) -- in-memory state still updates.
      }
      return resolved
    })
  }

  return [value, setAndPersist]
}
