// UI preferences, kept in localStorage. Only settings: never audio or transcripts.
import type { Overrides } from './api'

const KEY = 'whisper-diarize:prefs'

interface Prefs {
  /** Options that differ from the server defaults. */
  overrides: Overrides
  autoCopy: boolean
  /** "Only these speakers are in the recording", and which saved voices those are. */
  onlyTheseSpeakers: boolean
  speakers: string[]
}

function load(): Prefs {
  const defaults: Prefs = {
    overrides: {},
    autoCopy: true,
    onlyTheseSpeakers: false,
    speakers: [],
  }
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') }
  } catch {
    return defaults
  }
}

export const prefs = $state(load())

$effect.root(() => {
  $effect(() => localStorage.setItem(KEY, JSON.stringify(prefs)))
})
