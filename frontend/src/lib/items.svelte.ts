import type { JobView, LiveLine, Sentence } from './api'
import { paragraphs, toText } from './transcript'

export type Phase =
  | 'uploading'
  | 'processing'
  | 'recording'
  | 'stopping'
  | 'done'
  | 'failed'
  | 'cancelled'

let nextKey = 0

/** Which item's text is currently in the clipboard (only one can be). */
const clipboard = $state<{ owner: Item | null }>({ owner: null })

/** One file or recording on the page, from upload/recording to its transcript. */
export class Item {
  readonly key = nextKey++
  phase = $state<Phase>('uploading')
  uploaded = $state(0)
  job = $state<JobView | null>(null)
  lines = $state<LiveLine[]>([])
  partial = $state('')
  notice = $state<string | null>(null)
  sentences = $state<Sentence[] | null>(null)
  /** Speaker name -> voiceprint, for saving speakers to the voice bank. */
  voiceprints = $state<Record<string, number[]>>({})
  error = $state<string | null>(null)
  /** The browser refused to copy without a click (some do after async work). */
  copyBlocked = $state(false)
  /** Recordings: which microphone, and its loudest 100 ms (to tell silence from no speech). */
  mic = $state('')
  loudest = 0
  /** Server-side job still to delete once we have its result (or are cancelled). */
  jobId: string | null = null
  readonly abort = new AbortController()

  constructor(
    readonly title: string,
    readonly kind: 'file' | 'recording',
  ) {}

  get active() {
    return ['uploading', 'processing', 'recording', 'stopping'].includes(this.phase)
  }

  /** What's on screen: the result, or the live lines until it arrives. */
  get shown(): Sentence[] {
    return this.sentences ?? this.lines.map((l) => ({ ...l, speaker: null }))
  }

  /** Show `speaker` as `name` everywhere in this transcript. */
  rename(speaker: string, name: string) {
    if (!this.sentences) return
    this.sentences = this.sentences.map((s) => (s.speaker === speaker ? { ...s, speaker: name } : s))
    const print = this.voiceprints[speaker]
    delete this.voiceprints[speaker]
    if (print) this.voiceprints[name] = print
  }

  get text() {
    return toText(paragraphs(this.shown))
  }

  get copied() {
    return clipboard.owner === this
  }

  async copy() {
    try {
      await navigator.clipboard.writeText(this.text)
      clipboard.owner = this
      this.copyBlocked = false
    } catch {
      this.copyBlocked = true
    }
  }
}
