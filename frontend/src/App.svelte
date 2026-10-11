<script lang="ts">
  import { api, deleteOnExit, type Config, type LiveView, type Overrides, type Voice } from './lib/api'
  import ItemCard from './lib/ItemCard.svelte'
  import { Item } from './lib/items.svelte'
  import OptionsForm from './lib/OptionsForm.svelte'
  import { prefs } from './lib/prefs.svelte'
  import { Recorder, SILENT } from './lib/recorder'
  import Toasts from './lib/Toasts.svelte'
  import { toast } from './lib/toasts.svelte'
  import { clock } from './lib/transcript'
  import VoicesPanel from './lib/VoicesPanel.svelte'
  import Waveform from './lib/Waveform.svelte'

  let config = $state<Config | null>(null)
  let error = $state<string | null>(null)
  let items = $state<Item[]>([])
  let voices = $state<Voice[]>([])

  // The current recording (one at a time: there's one microphone).
  let recording = $state<Item | null>(null)
  let stopRequested = $state(false)
  let liveId: string | null = null
  let recorder: Recorder | null = null
  let wave = $state<number[]>([])
  let startedAt = $state(0)
  let now = $state(Date.now())

  // Uploads go one after the other; the server queues the transcriptions.
  let uploads: Promise<void> = Promise.resolve()
  let dragging = $state(false)
  let fileInput: HTMLInputElement

  api.config().then(
    (c) => (config = c),
    (e) => (error = `Can't reach the server: ${message(e)}`),
  )
  refreshVoices()
  // Label speakers is on unless switched off; recordings then get a full pass when stopped.
  const diarize = () => prefs.overrides.diarize !== false

  function refreshVoices() {
    api.voices().then((v) => {
      voices = v
      prefs.speakers = prefs.speakers.filter((name) => v.some((voice) => voice.name === name))
    }, () => {})
  }

  /** Options for a new job or recording: the settings form, plus "only these speakers". */
  function jobOptions(): Overrides {
    const speakers = prefs.onlyTheseSpeakers ? prefs.speakers : []
    return speakers.length ? { ...prefs.overrides, speakers } : prefs.overrides
  }

  $effect(() => {
    const tick = setInterval(() => (now = Date.now()), 500)
    // Don't leave audio or transcripts behind on the server if the tab is closed mid-way.
    const leave = () => {
      for (const item of items) if (item.jobId) deleteOnExit(`jobs/${item.jobId}`)
      if (liveId) deleteOnExit(`live/${liveId}`)
    }
    addEventListener('pagehide', leave)
    return () => {
      clearInterval(tick)
      removeEventListener('pagehide', leave)
    }
  })

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
  const message = (e: unknown) => (e instanceof Error ? e.message : String(e))

  function add(item: Item): Item {
    items.unshift(item)
    return items[0]
  }

  /** No speech: say so in a toast rather than leaving an empty card. */
  function dropEmpty(item: Item) {
    const silent = item.kind === 'recording' && item.loudest < SILENT
    toast(
      silent
        ? `Nothing recorded: “${item.mic}” only sent silence. Wrong microphone, or muted?`
        : `No speech found in ${item.kind === 'recording' ? 'the recording' : item.title}.`,
    )
    items = items.filter((i) => i !== item)
  }

  function fail(item: Item, e: unknown) {
    if (item.abort.signal.aborted) return
    item.phase = 'failed'
    item.error = message(e)
  }

  function transcribeFiles(files: Iterable<File>) {
    for (const file of files) {
      const item = add(new Item(file.name, 'file'))
      uploads = uploads.then(() => upload(item, file))
    }
  }

  async function upload(item: Item, file: File) {
    if (item.abort.signal.aborted) return
    try {
      const { id } = await api.createJob(jobOptions(), file.name)
      item.jobId = id
      item.job = await api.uploadAudio(id, file, (f) => (item.uploaded = f), item.abort.signal)
      follow(item, id) // not awaited: the next file uploads while this one is processed
    } catch (e) {
      fail(item, e)
    }
  }

  /** Poll a job until it finishes, show its result, then delete it from the server. */
  async function follow(item: Item, id: string) {
    item.phase = 'processing'
    item.jobId = id
    try {
      let view = await api.job(id)
      while (['uploading', 'queued', 'running'].includes(view.state)) {
        item.job = view
        await sleep(view.state === 'running' ? 1000 : 3000)
        if (item.abort.signal.aborted) return
        view = await api.job(id)
      }
      item.job = view
      item.jobId = null
      api.deleteJob(id).catch(() => {})
      if (view.result) {
        // An empty full pass shouldn't erase lines the live pass did catch.
        if (view.result.sentences.length || !item.lines.length) {
          item.sentences = view.result.sentences
          item.voiceprints = view.result.voiceprints ?? {}
        }
        item.phase = 'done'
        if (!item.shown.length) return dropEmpty(item)
        if (prefs.autoCopy) await item.copy()
      } else {
        item.phase = 'failed'
        item.error = view.error ?? `Job ${view.state}`
      }
    } catch (e) {
      fail(item, e)
    }
  }

  function cancel(item: Item) {
    item.abort.abort()
    item.phase = 'cancelled'
    if (item.jobId) api.deleteJob(item.jobId).catch(() => {})
    item.jobId = null
    if (item === recording) endRecording(true)
  }

  function remove(item: Item) {
    if (item.active) cancel(item)
    items = items.filter((i) => i !== item)
  }

  async function startRecording() {
    if (recording) return
    // Microphone first: if it fails, a toast says why and no card is left behind.
    const rec = new Recorder((peaks) => (wave = [...wave.slice(-400), ...peaks]))
    try {
      await rec.start()
    } catch (e) {
      toast(message(e))
      return
    }
    const item = add(new Item(`Recording ${new Date().toLocaleString()}`, 'recording'))
    item.phase = 'recording'
    item.mic = rec.label
    recording = item
    recorder = rec
    wave = []
    stopRequested = false
    startedAt = Date.now()
    try {
      liveId = (await api.startLive(jobOptions())).id
      await liveLoop(item, liveId, rec)
    } catch (e) {
      toast(message(e))
      endRecording(true)
      items = items.filter((i) => i !== item)
    }
  }

  function applyLive(item: Item, view: LiveView) {
    item.lines.push(...view.lines)
    item.partial = view.partial
    item.notice = view.notice
  }

  // The only place that talks to the live session, so its requests never overlap or reorder.
  async function liveLoop(item: Item, id: string, rec: Recorder) {
    let warned = false
    while (!item.abort.signal.aborted) {
      const stopping = stopRequested
      if (stopping) {
        item.phase = 'stopping'
        await rec.stop()
      }
      applyLive(item, await api.liveAudio(id, rec.take(), item.lines.length))
      if (stopping) break
      if (!warned && Date.now() - startedAt > 2500 && rec.loudest < SILENT) {
        warned = true
        toast(`Not hearing anything from “${rec.label}”. Wrong microphone, or muted?`, 'warning')
      }
      await sleep(500)
    }
    if (item.abort.signal.aborted) return
    item.loudest = rec.loudest
    const end = await api.stopLive(id, item.lines.length, diarize())
    endRecording(false)
    applyLive(item, end)
    item.partial = ''
    if (end.job_id) {
      if (prefs.autoCopy && item.lines.length) await item.copy()
      item.notice = 'Adding speaker names…'
      await follow(item, end.job_id)
      item.notice = null
    } else {
      item.phase = 'done'
      if (!item.shown.length) return dropEmpty(item)
      if (prefs.autoCopy) await item.copy()
    }
  }

  function stopRecording() {
    stopRequested = true
    // Copy right away while the click still counts as a user gesture (some browsers need one);
    // the final version is copied again when it arrives.
    if (prefs.autoCopy && recording?.lines.length) recording.copy()
  }

  function endRecording(discard: boolean) {
    if (discard && liveId) api.discardLive(liveId).catch(() => {})
    recorder?.stop()
    liveId = recorder = recording = null
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    dragging = false
    if (e.dataTransfer?.files.length) transcribeFiles(e.dataTransfer.files)
  }
</script>

<svelte:window
  ondragover={(e) => {
    e.preventDefault()
    dragging = true
  }}
  ondragleave={(e) => {
    if (!e.relatedTarget) dragging = false
  }}
  ondrop={onDrop}
/>

<main class:dragging>
  <header>
    <h1>whisper-diarize</h1>
    {#if config && !config.cuda}<span class="warn">no GPU: running on CPU</span>{/if}
  </header>

  <div class="layout">
    <div class="content">
      <section class="actions">
        {#if recording}
          <button class="big stop" onclick={stopRecording} disabled={stopRequested}>
            <span class="dot"></span>
            {stopRequested ? 'Finishing…' : `Stop  ${clock((now - startedAt) / 1000)}`}
          </button>
          <div class="wave">
            <Waveform levels={wave} />
            <small>🎙 {recording.mic}</small>
          </div>
        {:else}
          <button class="big record" onclick={startRecording}>🎙 Record</button>
        {/if}
        <button class="big" onclick={() => fileInput.click()}>📄 Transcribe files…</button>
        <input
          bind:this={fileInput}
          type="file"
          accept="audio/*,video/*"
          multiple
          hidden
          onchange={(e) => {
            transcribeFiles([...(e.currentTarget.files ?? [])])
            e.currentTarget.value = ''
          }}
        />
        {#if !recording}<span class="hint">or drop files anywhere</span>{/if}
      </section>

      {#if config}
        <section class="card">
          <OptionsForm properties={config.schema.properties} bind:overrides={prefs.overrides}>
            {#snippet extra()}
              <label class="check">
                <input type="checkbox" bind:checked={prefs.autoCopy} /> Copy results automatically
              </label>
            {/snippet}
          </OptionsForm>
        </section>
      {/if}

      {#if error}<p class="error">{error}</p>{/if}

      {#each items as item (item.key)}
        <ItemCard
          {item}
          voiceNames={voices.map((v) => v.name)}
          oncancel={() => cancel(item)}
          onremove={() => remove(item)}
          onvoicesaved={refreshVoices}
        />
      {/each}
    </div>

    <aside>
      <VoicesPanel {voices} onchange={refreshVoices} />
    </aside>
  </div>
</main>

<Toasts />

<style>
  main {
    max-width: 78rem;
    margin: 0 auto;
    padding: 1.5rem 1rem 4rem;
    min-height: 100vh;
    box-sizing: border-box;
  }
  main.dragging {
    outline: 3px dashed var(--accent);
    outline-offset: -0.75rem;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
  }
  h1 {
    font-size: 1.4rem;
    margin: 0 0 1rem;
  }
  .warn {
    color: var(--warn);
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .big {
    font-size: 1.1rem;
    padding: 0.8rem 1.4rem;
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
  }
  .record {
    background: var(--accent);
    color: var(--bg);
    border-color: var(--accent);
  }
  .stop {
    background: var(--danger);
    color: white;
    border-color: var(--danger);
    font-variant-numeric: tabular-nums;
  }
  .dot {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 50%;
    background: white;
    animation: pulse 1s infinite alternate;
  }
  @keyframes pulse {
    to {
      opacity: 0.3;
    }
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 18rem;
    gap: 1.5rem;
    align-items: start;
  }
  aside {
    position: sticky;
    top: 1rem;
  }
  @media (max-width: 52rem) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    aside {
      position: static;
    }
  }
  .wave {
    flex: 1 1 12rem;
    min-width: 8rem;
  }
  .wave small {
    color: var(--muted);
  }
  .check {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    padding-bottom: 0.4rem;
    font-size: 0.85rem;
  }
  .hint {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .error {
    color: var(--danger);
  }
</style>
