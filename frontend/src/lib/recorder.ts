/** Below this RMS (0-1), a microphone is effectively silent: muted, or the wrong device. */
export const SILENT = 0.002

/** Microphone capture as 16 kHz PCM16, buffered until `take()` is called. */
export class Recorder {
  private context?: AudioContext
  private stream?: MediaStream
  private chunks: Int16Array[] = []
  /** Loudest 100 ms so far (RMS, 0-1). */
  loudest = 0

  /** `onPeaks` gets the peak level (0-1) of every 25 ms, for drawing a waveform. */
  constructor(private onPeaks: (peaks: number[]) => void = () => {}) {}

  /** The device's name, as the browser reports it. */
  get label(): string {
    return this.stream?.getAudioTracks()[0]?.label || 'the microphone'
  }

  /** Call straight from a click handler: the audio context is created before any await, while
   * the click still counts as a user gesture. Otherwise some browsers keep it suspended. */
  async start() {
    const context = new AudioContext()
    this.context = context
    try {
      // Raw-ish audio: browser noise suppression/echo cancellation tend to hurt diarization.
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      })
      await context.resume()
      if (context.state !== 'running') {
        throw new Error("The browser didn't start audio capture. Click Record again.")
      }
      await context.audioWorklet.addModule(new URL('pcm-recorder.js', document.baseURI))
      const source = context.createMediaStreamSource(this.stream)
      const node = new AudioWorkletNode(context, 'pcm-recorder')
      node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        const chunk = new Int16Array(e.data)
        this.chunks.push(chunk)
        this.loudest = Math.max(this.loudest, rms(chunk))
        this.onPeaks(peaks(chunk, 4))
      }
      // Some browsers only run nodes that lead to the destination: route it there, muted.
      const mute = context.createGain()
      mute.gain.value = 0
      source.connect(node).connect(mute).connect(context.destination)
    } catch (e) {
      await this.stop()
      throw friendly(e)
    }
  }

  /** Everything recorded since the last call, as little-endian PCM16 bytes. */
  take(): Uint8Array {
    const total = this.chunks.reduce((n, c) => n + c.length, 0)
    const out = new Int16Array(total)
    let offset = 0
    for (const c of this.chunks) {
      out.set(c, offset)
      offset += c.length
    }
    this.chunks = []
    return new Uint8Array(out.buffer)
  }

  async stop() {
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.context?.close().catch(() => {})
    this.context = this.stream = undefined
  }
}

function friendly(e: unknown): Error {
  const name = e instanceof DOMException ? e.name : ''
  if (name === 'NotAllowedError') {
    return new Error('Microphone access is blocked: allow it from the icon in the address bar.')
  }
  if (name === 'NotFoundError') return new Error('No microphone found.')
  if (name === 'NotReadableError') {
    return new Error('The microphone is busy or unavailable (used by another app?).')
  }
  return e instanceof Error ? e : new Error(String(e))
}

/** 16 kHz mono PCM16 bytes as a .wav file. */
export function wav(pcm: Uint8Array): Blob {
  const header = new DataView(new ArrayBuffer(44))
  const ascii = (offset: number, text: string) =>
    [...text].forEach((c, i) => header.setUint8(offset + i, c.charCodeAt(0)))
  ascii(0, 'RIFF')
  header.setUint32(4, 36 + pcm.length, true)
  ascii(8, 'WAVEfmt ')
  header.setUint32(16, 16, true) // fmt chunk size
  header.setUint16(20, 1, true) // PCM
  header.setUint16(22, 1, true) // mono
  header.setUint32(24, 16000, true) // sample rate
  header.setUint32(28, 32000, true) // byte rate
  header.setUint16(32, 2, true) // block align
  header.setUint16(34, 16, true) // bits per sample
  ascii(36, 'data')
  header.setUint32(40, pcm.length, true)
  return new Blob([header, pcm as BlobPart], { type: 'audio/wav' })
}

function rms(chunk: Int16Array): number {
  let sum = 0
  for (const s of chunk) sum += s * s
  return Math.sqrt(sum / chunk.length) / 0x8000
}

function peaks(chunk: Int16Array, parts: number): number[] {
  const size = Math.ceil(chunk.length / parts)
  const out: number[] = []
  for (let start = 0; start < chunk.length; start += size) {
    let peak = 0
    for (let i = start; i < Math.min(start + size, chunk.length); i++) {
      peak = Math.max(peak, Math.abs(chunk[i]))
    }
    out.push(peak / 0x8000)
  }
  return out
}
