export interface Toast {
  id: number
  kind: 'error' | 'warning' | 'info'
  text: string
}

let nextId = 0
export const toasts = $state<Toast[]>([])

export function toast(text: string, kind: Toast['kind'] = 'error', ms = 12000) {
  const id = nextId++
  toasts.push({ id, kind, text })
  if (ms) setTimeout(() => dismiss(id), ms)
}

export function dismiss(id: number) {
  const i = toasts.findIndex((t) => t.id === id)
  if (i >= 0) toasts.splice(i, 1)
}
