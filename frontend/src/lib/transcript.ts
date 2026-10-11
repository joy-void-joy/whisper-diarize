import type { Sentence } from './api'

export interface Paragraph {
  start: number
  end: number
  speaker: string | null
  text: string
  /** Positions of its sentences in the transcript. */
  indices: number[]
}

/** Same as Transcript.paragraphs(): consecutive sentences by the same speaker, merged; without
 * speaker labels, a new paragraph starts at every pause of PARAGRAPH_PAUSE_S or more. */
const PARAGRAPH_PAUSE_S = 2

export function paragraphs(sentences: Sentence[]): Paragraph[] {
  const merged: Paragraph[] = []
  sentences.forEach((s, i) => {
    const last = merged.at(-1)
    const pause = last && last.speaker === null && s.start - last.end >= PARAGRAPH_PAUSE_S
    if (last && last.speaker === s.speaker && !pause) {
      last.end = s.end
      last.text += ` ${s.text}`
      last.indices.push(i)
    } else {
      merged.push({ ...s, indices: [i] })
    }
  })
  return merged
}

export function toText(paras: Paragraph[]): string {
  return paras.map((p) => (p.speaker ? `${p.speaker}: ${p.text}` : p.text)).join('\n\n')
}

export function toSrt(sentences: Sentence[]): string {
  return sentences
    .map((s, i) => {
      const text = s.speaker ? `${s.speaker}: ${s.text}` : s.text
      return `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${text}\n`
    })
    .join('\n')
}

function srtTime(seconds: number): string {
  const ms = Math.round(seconds * 1000)
  const pad = (n: number, width = 2) => String(n).padStart(width, '0')
  return `${pad(Math.floor(ms / 3_600_000))}:${pad(Math.floor(ms / 60_000) % 60)}:${pad(
    Math.floor(ms / 1000) % 60,
  )},${pad(ms % 1000, 3)}`
}

export function clock(seconds: number): string {
  const s = Math.floor(seconds)
  const mmss = `${Math.floor(s / 60) % 60}:${String(s % 60).padStart(2, '0')}`
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mmss.padStart(5, '0')}` : mmss
}

export function download(filename: string, text: string, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }))
  const a = Object.assign(document.createElement('a'), { href: url, download: filename })
  a.click()
  URL.revokeObjectURL(url)
}

const PALETTE = ['#7aa2f7', '#e0af68', '#9ece6a', '#f7768e', '#bb9af7', '#2ac3de', '#ff9e64', '#c0caf5']

/** A color per speaker, in order of first appearance. */
export function speakerColors(sentences: Sentence[]): Map<string, string> {
  const colors = new Map<string, string>()
  for (const { speaker } of sentences) {
    if (speaker && !colors.has(speaker)) colors.set(speaker, PALETTE[colors.size % PALETTE.length])
  }
  return colors
}
