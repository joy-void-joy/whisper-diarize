// Thin client for the FastAPI backend (see src/whisper_diarize/server.py).
// URLs are relative so the app also works when served under a sub-path.

export type Overrides = Record<string, unknown>

export interface Sentence {
  start: number
  end: number
  speaker: string | null
  text: string
}

export interface Transcript {
  language: string
  duration: number
  sentences: Sentence[]
  /** Speaker name -> voiceprint, to save a speaker to the voice bank. */
  voiceprints: Record<string, number[]>
}

export interface Voice {
  name: string
  samples: number
}

export type JobState = 'uploading' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface JobView {
  id: string
  name: string
  state: JobState
  position: number | null
  stage: string
  progress: number | null
  messages: string[]
  error: string | null
  duration: number
  result: Transcript | null
}

export interface LiveLine {
  start: number
  end: number
  text: string
}

export interface LiveView {
  lines: LiveLine[]
  partial: string
  notice: string | null
  job_id?: string | null
}

export interface SchemaProperty {
  type?: string
  anyOf?: SchemaProperty[]
  enum?: string[]
  items?: SchemaProperty
  title?: string
  description?: string
  default?: unknown
  group?: string
  choices?: string[]
  suggestions?: string[]
  minimum?: number
}

export interface Config {
  schema: { properties: Record<string, SchemaProperty> }
  cuda: boolean
}

function errorMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  // FastAPI validation errors: [{loc: [...], msg: "..."}]
  if (Array.isArray(detail)) {
    return detail.map((d) => `${d.loc?.slice(-1)[0] ?? ''}: ${d.msg}`).join('; ')
  }
  return fallback
}

async function call<T>(method: string, path: string, body?: object | BodyInit): Promise<T> {
  const init: RequestInit = { method }
  if (body instanceof Blob || body instanceof ArrayBuffer || ArrayBuffer.isView(body)) {
    init.body = body as BodyInit
  } else if (body !== undefined) {
    init.body = JSON.stringify(body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  const res = await fetch(`api/${path}`, init)
  if (!res.ok) {
    const json = await res.json().catch(() => null)
    throw new Error(errorMessage(json, `${res.status} ${res.statusText}`))
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  config: () => call<Config>('GET', 'config'),

  createJob: (options: Overrides, name: string) =>
    call<{ id: string }>('POST', 'jobs', { options, name }),
  job: (id: string) => call<JobView>('GET', `jobs/${id}`),
  deleteJob: (id: string) => call<void>('DELETE', `jobs/${id}`),

  // XHR rather than fetch, for upload progress.
  uploadAudio: (
    id: string,
    file: Blob,
    onProgress: (fraction: number) => void,
    signal?: AbortSignal,
  ) =>
    new Promise<JobView>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      signal?.addEventListener('abort', () => xhr.abort())
      xhr.onabort = () => reject(new Error('Upload cancelled'))
      xhr.open('PUT', `api/jobs/${id}/audio`)
      xhr.responseType = 'json'
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
      xhr.onload = () =>
        xhr.status < 300
          ? resolve(xhr.response)
          : reject(new Error(errorMessage(xhr.response, `${xhr.status} ${xhr.statusText}`)))
      xhr.onerror = () => reject(new Error('Upload failed (network error)'))
      xhr.send(file)
    }),

  voices: () => call<Voice[]>('GET', 'voices'),
  enrollVoice: (name: string, audio: Blob) =>
    call<Voice & { speech_s: number }>('POST', `voices/enroll?name=${encodeURIComponent(name)}`, audio),
  saveVoice: (name: string, embedding: number[]) =>
    call<Voice>('POST', 'voices', { name, embedding }),
  deleteVoice: (name: string) => call<void>('DELETE', `voices/${encodeURIComponent(name)}`),

  startLive: (options: Overrides) => call<{ id: string }>('POST', 'live', { options }),
  liveAudio: (id: string, pcm: Uint8Array, since: number) =>
    call<LiveView>('POST', `live/${id}/audio?since=${since}`, pcm),
  stopLive: (id: string, since: number, diarize: boolean) =>
    call<LiveView>('POST', `live/${id}/stop?since=${since}&diarize=${diarize}`),
  discardLive: (id: string) => call<void>('DELETE', `live/${id}`),
}

/** Best-effort cleanup when the tab goes away (keepalive lets the request outlive the page). */
export function deleteOnExit(path: string) {
  fetch(`api/${path}`, { method: 'DELETE', keepalive: true }).catch(() => {})
}
