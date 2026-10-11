// UI preferences, kept in localStorage. Only settings: never audio or transcripts.
import type { Overrides } from './api'

const KEY = 'whisper-diarize:prefs'

interface Prefs {
  /** Options that differ from the server defaults. */
  overrides: Overrides
  autoCopy: boolean
  /** Saved voices unticked as "not in the recording". */
  absent: string[]
  /** Only the ticked voices are in the recording. */
  onlyThese: boolean
}

function load(): Prefs {
  const defaults: Prefs = { overrides: {}, autoCopy: true, absent: [], onlyThese: false }
  try {
    const stored = JSON.parse(localStorage.getItem(KEY) ?? '{}')
    // Earlier shapes: {onlyTheseSpeakers, speakers} and {presence, nobodyElse}.
    stored.onlyThese ??= Boolean(stored.onlyTheseSpeakers || stored.nobodyElse)
    if (stored.presence && !stored.absent) {
      stored.absent = Object.keys(stored.presence).filter((n) => stored.presence[n] === 'away')
    }
    for (const old of ['onlyTheseSpeakers', 'speakers', 'presence', 'nobodyElse']) delete stored[old]
    return { ...defaults, ...stored }
  } catch {
    return defaults
  }
}

export const prefs = $state(load())

$effect.root(() => {
  $effect(() => localStorage.setItem(KEY, JSON.stringify(prefs)))
})
