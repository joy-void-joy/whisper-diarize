<script lang="ts">
  import { api } from './api'
  import type { Item } from './items.svelte'
  import { clock, download, paragraphs, speakerColors, toSrt } from './transcript'

  let {
    item,
    voiceNames,
    oncancel,
    onremove,
    onvoicesaved,
  }: {
    item: Item
    voiceNames: string[]
    oncancel: () => void
    onremove: () => void
    onvoicesaved: () => void
  } = $props()

  const paras = $derived(paragraphs(item.shown))
  const colors = $derived(speakerColors(item.shown))

  // Naming a speaker: saves their voiceprint to the voice bank and renames them here.
  let naming = $state<string | null>(null)
  let draft = $state('')
  let namingError = $state<string | null>(null)

  function startNaming(speaker: string) {
    naming = speaker
    draft = /^Speaker \d+$/.test(speaker) ? '' : speaker
    namingError = null
  }

  async function saveName() {
    const speaker = naming
    const name = draft.trim()
    if (!speaker || !name) return
    try {
      await api.saveVoice(name, item.voiceprints[speaker])
    } catch (e) {
      namingError = e instanceof Error ? e.message : String(e)
      return
    }
    item.rename(speaker, name)
    naming = null
    onvoicesaved()
  }
  const stem = $derived(item.title.replace(/\.[^.]+$/, '') || 'transcript')

  const status = $derived.by(() => {
    const job = item.job
    switch (item.phase) {
      case 'uploading':
        return `Uploading… ${Math.round(item.uploaded * 100)}%`
      case 'recording':
        return 'Recording'
      case 'stopping':
        return 'Finishing…'
      case 'processing':
        if (!job || job.state === 'queued') {
          const ahead = (job?.position ?? 1) - 1
          return ahead > 0 ? `Queued (${ahead} ahead)` : 'Queued'
        }
        return `${job.stage || 'starting'}${job.progress != null ? ` ${Math.round(job.progress * 100)}%` : '…'}`
      case 'cancelled':
        return 'Cancelled'
      default:
        return ''
    }
  })
  const progress = $derived(
    item.phase === 'uploading'
      ? item.uploaded
      : item.phase === 'processing' && item.job?.state === 'running'
        ? (item.job.progress ?? undefined)
        : null,
  )
</script>

<section class="card">
  <div class="bar">
    <h2 title={item.title}>{item.title}</h2>
    {#if status}<span class="status">{status}</span>{/if}
    {#if paras.length}
      <button onclick={() => item.copy()} class:attention={item.copyBlocked}>
        {item.copied ? 'Copied ✓' : item.copyBlocked ? 'Click to copy' : 'Copy'}
      </button>
    {/if}
    {#if !item.active && paras.length}
      <button onclick={() => download(`${stem}.txt`, item.text)}>.txt</button>
      <button onclick={() => download(`${stem}.srt`, toSrt(item.shown))}>.srt</button>
    {/if}
    {#if item.active && item.phase !== 'recording'}
      <button onclick={oncancel}>Cancel</button>
    {:else if !item.active}
      <button class="close" onclick={onremove} title="Remove from the page">✕</button>
    {/if}
  </div>

  {#if naming}
    <form class="naming" onsubmit={(e) => (e.preventDefault(), saveName())}>
      <span>Who is <b style="color: {colors.get(naming)}">{naming}</b>?</span>
      <!-- svelte-ignore a11y_autofocus -->
      <input bind:value={draft} list="voice-names-{item.key}" placeholder="Name" autofocus />
      <datalist id="voice-names-{item.key}">
        {#each voiceNames as n}<option value={n}></option>{/each}
      </datalist>
      <button type="submit" disabled={!draft.trim()}>Save voice</button>
      <button type="button" onclick={() => (naming = null)}>Cancel</button>
      <small>
        {voiceNames.includes(draft.trim())
          ? `Adds this as another sample of ${draft.trim()}'s voice.`
          : 'Saved to the voice bank, so they get named automatically next time.'}
      </small>
      {#if namingError}<p class="error">{namingError}</p>{/if}
    </form>
  {/if}

  {#if progress !== null}<progress value={progress}></progress>{/if}
  {#if item.phase === 'processing' && item.job?.messages.length}
    <small>{item.job.messages.at(-1)}</small>
  {/if}
  {#if item.notice}<small>{item.notice}</small>{/if}
  {#if item.error}<p class="error">{item.error}</p>{/if}

  {#each paras as p}
    <p>
      <span class="time">{clock(p.start)}</span>
      {#if p.speaker && item.voiceprints[p.speaker] && !item.active}
        <button
          class="speaker"
          style="color: {colors.get(p.speaker)}"
          title="Name this speaker"
          onclick={() => startNaming(p.speaker ?? '')}>{p.speaker}</button
        >
      {:else if p.speaker}
        <b style="color: {colors.get(p.speaker)}">{p.speaker}</b>
      {/if}
      {p.text}
    </p>
  {/each}
  {#if item.partial}<p class="partial">{item.partial}</p>{/if}
</section>

<style>
  .bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  h2 {
    font-size: 1rem;
    margin: 0;
    margin-right: auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .status {
    color: var(--muted);
    font-size: 0.85rem;
    white-space: nowrap;
  }
  progress {
    width: 100%;
    margin-top: 0.5rem;
  }
  small {
    display: block;
    color: var(--muted);
    margin-top: 0.3rem;
  }
  p {
    line-height: 1.55;
    margin: 0.6rem 0;
  }
  .error {
    color: var(--danger);
    white-space: pre-wrap;
  }
  .time {
    color: var(--muted);
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
    margin-right: 0.4rem;
  }
  .partial {
    color: var(--muted);
    font-style: italic;
  }
  .attention {
    border-color: var(--warn);
    color: var(--warn);
  }
  .close {
    border: none;
    background: none;
    color: var(--muted);
  }
  .speaker {
    border: none;
    background: none;
    padding: 0;
    font-weight: bold;
    text-decoration: underline dotted;
    text-underline-offset: 0.2em;
  }
  .naming {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.6rem;
    padding: 0.6rem;
    border: 1px solid var(--border);
    border-radius: 0.4rem;
  }
  .naming small {
    flex-basis: 100%;
    margin: 0;
  }
</style>
