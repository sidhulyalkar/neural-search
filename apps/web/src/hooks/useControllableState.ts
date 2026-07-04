import { useState } from 'react'

/** Standard controlled/uncontrolled hybrid: behaves like plain `useState`
 * when `controlledValue` is `undefined` (single-scene timeline), but defers
 * to the parent when it's supplied (compare mode, where two timelines must
 * share one cursor). */
export function useControllableState<T>(
  controlledValue: T | undefined,
  onChange: ((value: T) => void) | undefined,
  defaultValue: T,
): [T, (value: T) => void] {
  const [internal, setInternal] = useState(defaultValue)
  const value = controlledValue !== undefined ? controlledValue : internal
  const setValue = (next: T) => {
    if (controlledValue === undefined) setInternal(next)
    onChange?.(next)
  }
  return [value, setValue]
}
