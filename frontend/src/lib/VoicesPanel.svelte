<script lang="ts">
  // The voice bank: saved voices name matching speakers in every transcript. New voices come
  // from a short recording or an uploaded clip (or from a transcript, see ItemCard).
  import { api, type Voice } from './api'
  import { prefs } from './prefs.svelte'
  import { Recorder, SILENT, wav } from './recorder'
  import { toast } from './toasts.svelte'
  import { clock } from './transcript'
  import Waveform from './Waveform.svelte'

  let { voices, onchange }: { voices: Voice[]; onchange: () => void } = $props()

  let name = $state('')
  let recorder = $state<Recorder | null>(null)
  let wave = $state<number[]>([])
  let startedAt = $state(0)
  let now = $state(0)
  let busy = $state(false)
  let status = $state<string | null>(null)
  let confirmDelete = $state<string | null>(null)
  let fileInput: HTMLInputElement

  const message = (e: unknown) => (e instanceof Error ? e.message : String(e))
  const ready = $derived(name.trim().length > 0 && !busy)

  $effect(() => {
    if (!recorder) return
    let warned = false
    const tick = setInterval(() => {
      now = Date.now()
      if (!warned && recorder && now - startedAt > 2500 && recorder.loudest < SILENT) {
        warned = true
        toast(`Not hearing anything from “${recorder.label}”. Wrong microphone, or muted?`, 'warning')
      }
    }, 250)
    return () => clearInterval(tick)
  })

  async function startRecording() {
    status = null
    const rec = new Recorder((peaks) => (wave = [...wave.slice(-300), ...peaks]))
    try {
      await rec.start()
    } catch (e) {
      toast(message(e))
      return
    }
    wave = []
    recorder = rec
    startedAt = now = Date.now()
  }

  async function stopRecording() {
    const rec = recorder
    recorder = null
    if (!rec) return
    await rec.stop()
    if (rec.loudest < SILENT) {
      toast(`Nothing recorded: “${rec.label}” only sent silence. Wrong microphone, or muted?`)
      return
    }
    await enroll(wav(rec.take()))
  }

  async function enroll(audio: Blob) {
    const who = name.trim()
    busy = true
    status = `Learning ${who}'s voice…`
    try {
      const voice = await api.enrollVoice(who, audio)
      const extra = voice.samples > 1 ? `, now ${voice.samples} samples` : ''
      status = `Saved ${voice.name} (${voice.speech_s} s of speech${extra})`
      name = ''
      onchange()
    } catch (e) {
      toast(message(e))
      status = null
    } finally {
      busy = false
    }
  }

  async function remove(voice: Voice) {
    if (confirmDelete !== voice.name) {
      confirmDelete = voice.name
      return
    }
    confirmDelete = null
    try {
      await api.deleteVoice(voice.name)
    } catch (e) {
      toast(message(e))
    }
    onchange()
  }

  function tick(voice: string, on: boolean) {
    prefs.speakers = on ? [...prefs.speakers, voice] : prefs.speakers.filter((n) => n !== voice)
  }
</script>

<section class="card">
  <h3>Voices</h3>
  <p class="hint">Speakers who sound like a saved voice get its name.</p>

  {#if voices.length}
    <ul>
      {#each voices as voice (voice.name)}
        <li class:off={prefs.onlyTheseSpeakers && !prefs.speakers.includes(voice.name)}>
          <label>
            {#if prefs.onlyTheseSpeakers}
              <input
                type="checkbox"
                checked={prefs.speakers.includes(voice.name)}
                onchange={(e) => tick(voice.name, e.currentTarget.checked)}
              />
            {/if}
            {voice.name}
            {#if voice.samples > 1}<small title="samples">×{voice.samples}</small>{/if}
          </label>
          <button
            class="remove"
            title="Delete this voice"
            onclick={() => remove(voice)}
            onblur={() => (confirmDelete = null)}
          >
            {confirmDelete === voice.name ? 'delete?' : '✕'}
          </button>
        </li>
      {/each}
    </ul>
    <label class="only">
      <input type="checkbox" bind:checked={prefs.onlyTheseSpeakers} />
      <span>
        Only these speakers are in the recording
        {#if prefs.onlyTheseSpeakers}
          <small>
            {prefs.speakers.length
              ? `Split into exactly ${prefs.speakers.length}.`
              : 'Tick who is in it.'}
          </small>
        {/if}
      </span>
    </label>
  {/if}

  <h4>Add a voice</h4>
  <input placeholder="Their name" bind:value={name} disabled={!!recorder || busy} />
  {#if recorder}
    <div class="recording">
      <button class="stop" onclick={stopRecording}>■ Stop {clock((now - startedAt) / 1000)}</button>
      <Waveform levels={wave} height={32} />
      <small>🎙 {recorder.label}. Say a few sentences, about 15 seconds.</small>
    </div>
  {:else}
    <div class="buttons">
      <button onclick={startRecording} disabled={!ready}>🎙 Record</button>
      <button onclick={() => fileInput.click()} disabled={!ready}>Upload…</button>
    </div>
  {/if}
  <input
    bind:this={fileInput}
    type="file"
    accept="audio/*,video/*"
    hidden
    onchange={(e) => {
      const file = e.currentTarget.files?.[0]
      e.currentTarget.value = ''
      if (file) enroll(file)
    }}
  />
  {#if status}<p class="status">{status}</p>{/if}
</section>

<style>
  .card {
    margin-top: 0;
  }
  h3 {
    margin: 0;
    font-size: 1rem;
  }
  h4 {
    margin: 1rem 0 0.4rem;
    font-size: 0.9rem;
  }
  .hint,
  small {
    color: var(--muted);
    font-size: 0.8rem;
  }
  .hint {
    margin: 0.2rem 0 0;
  }
  ul {
    list-style: none;
    padding: 0;
    margin: 0.7rem 0 0;
  }
  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0;
    border-bottom: 1px solid var(--border);
  }
  li.off {
    opacity: 0.5;
  }
  li label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .remove {
    border: none;
    background: none;
    color: var(--muted);
    padding: 0 0.2rem;
    font-size: 0.8rem;
  }
  .only {
    display: flex;
    gap: 0.4rem;
    align-items: flex-start;
    margin-top: 0.6rem;
    font-size: 0.85rem;
  }
  .only small {
    display: block;
  }
  input:not([type='checkbox']) {
    width: 100%;
    box-sizing: border-box;
  }
  .buttons {
    display: flex;
    gap: 0.4rem;
    margin-top: 0.4rem;
  }
  .buttons button {
    flex: 1;
  }
  .recording {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.4rem;
  }
  .stop {
    background: var(--danger);
    border-color: var(--danger);
    color: white;
    font-variant-numeric: tabular-nums;
  }
  .status {
    color: var(--muted);
    margin: 0.5rem 0 0;
    font-size: 0.85rem;
  }
</style>
