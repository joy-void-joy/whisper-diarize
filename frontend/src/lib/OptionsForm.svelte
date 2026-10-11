<script lang="ts">
  // Renders whisper-diarize's Options model from its JSON schema: "basic" fields up front,
  // "advanced" ones one click away, the rest under "All settings". Only values that differ
  // from the default are kept.
  import type { Snippet } from 'svelte'
  import type { Overrides, SchemaProperty } from './api'

  let {
    properties,
    overrides = $bindable(),
    extra,
  }: {
    properties: Record<string, SchemaProperty>
    overrides: Overrides
    /** Rendered at the end of the basic row. */
    extra?: Snippet
  } = $props()

  type Kind = 'bool' | 'enum' | 'choice' | 'integer' | 'number' | 'list' | 'text'

  interface Field {
    name: string
    title: string
    description: string
    group: string
    kind: Kind
    nullable: boolean
    options: string[]
    suggestions: string[]
    defaultValue: unknown
  }

  function describe(name: string, prop: SchemaProperty): Field {
    const variants = prop.anyOf ?? [prop]
    const typed = variants.find((v) => v.type !== 'null') ?? {}
    const kind: Kind = prop.choices
      ? 'choice'
      : typed.enum
        ? 'enum'
        : typed.type === 'boolean'
          ? 'bool'
          : typed.type === 'integer'
            ? 'integer'
            : typed.type === 'number'
              ? 'number'
              : typed.type === 'array'
                ? 'list'
                : 'text'
    return {
      name,
      title: prop.title ?? name,
      description: prop.description ?? '',
      group: prop.group ?? 'whisper',
      kind,
      nullable: variants.some((v) => v.type === 'null'),
      options: prop.choices ?? typed.enum ?? [],
      suggestions: prop.suggestions ?? [],
      defaultValue: prop.default,
    }
  }

  const fields = $derived(Object.entries(properties).map(([name, p]) => describe(name, p)))
  const groups = $derived({
    basic: fields.filter((f) => f.group === 'basic'),
    advanced: fields.filter((f) => f.group === 'advanced'),
    diarization: fields.filter((f) => f.group === 'diarization'),
    whisper: fields.filter((f) => f.group === 'whisper'),
  })
  const changed = (...groupNames: string[]) =>
    Object.keys(overrides).filter((k) => groupNames.includes(properties[k]?.group ?? '')).length
  const changedAdvanced = $derived(changed('advanced', 'diarization', 'whisper'))
  const changedAll = $derived(changed('diarization', 'whisper'))

  const languageNames = new Intl.DisplayNames(['en'], { type: 'language' })
  function label(field: Field, option: string): string {
    if (field.name !== 'language') return option
    try {
      return `${languageNames.of(option)} (${option})`
    } catch {
      return option
    }
  }

  function value(field: Field): unknown {
    return field.name in overrides ? overrides[field.name] : field.defaultValue
  }

  function set(field: Field, v: unknown) {
    if (JSON.stringify(v) === JSON.stringify(field.defaultValue)) delete overrides[field.name]
    else overrides[field.name] = v
  }

  function parse(field: Field, raw: string): unknown {
    const text = raw.trim()
    if (text === '') return field.nullable ? null : field.defaultValue
    switch (field.kind) {
      case 'integer':
        return Number.isNaN(parseInt(text)) ? field.defaultValue : parseInt(text)
      case 'number':
        return Number.isNaN(Number(text)) ? field.defaultValue : Number(text)
      case 'list':
        return text
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean)
          .map(Number)
          .filter((n) => !Number.isNaN(n))
      default:
        return raw
    }
  }

  function shown(v: unknown): string {
    if (v === null || v === undefined) return ''
    return Array.isArray(v) ? v.join(', ') : String(v)
  }
</script>

{#snippet input(field: Field)}
  <label class="field" class:wide={field.name === 'initial_prompt'} title={field.description}>
    {#if field.kind === 'bool'}
      <span class="check">
        <input
          type="checkbox"
          checked={value(field) as boolean}
          onchange={(e) => set(field, e.currentTarget.checked)}
        />
        {field.title}
      </span>
    {:else}
      <span class="name">{field.title}</span>
      {#if field.kind === 'enum' || field.kind === 'choice'}
        <select
          value={value(field) ?? ''}
          onchange={(e) => set(field, e.currentTarget.value || null)}
        >
          {#if field.nullable}<option value="">Auto-detect</option>{/if}
          {#each field.options as option}
            <option value={option}>{label(field, option)}</option>
          {/each}
        </select>
      {:else if field.name === 'initial_prompt'}
        <textarea
          rows="2"
          placeholder={field.description}
          value={shown(value(field))}
          onchange={(e) => set(field, parse(field, e.currentTarget.value))}
        ></textarea>
      {:else}
        <input
          type={field.kind === 'integer' || field.kind === 'number' ? 'number' : 'text'}
          step={field.kind === 'number' ? 'any' : '1'}
          list={field.suggestions.length ? `${field.name}-suggestions` : undefined}
          placeholder={field.nullable && field.kind !== 'text' ? 'auto' : ''}
          value={shown(value(field))}
          onchange={(e) => set(field, parse(field, e.currentTarget.value))}
        />
        {#if field.suggestions.length}
          <datalist id="{field.name}-suggestions">
            {#each field.suggestions as s}<option value={s}></option>{/each}
          </datalist>
        {/if}
      {/if}
    {/if}
  </label>
{/snippet}

<div class="grid">
  {#each groups.basic as field (field.name)}{@render input(field)}{/each}
  {@render extra?.()}
</div>

<details>
  <summary>
    Advanced
    {#if changedAdvanced}<span class="badge">{changedAdvanced} changed</span>{/if}
  </summary>
  <div class="grid">
    {#each groups.advanced as field (field.name)}{@render input(field)}{/each}
  </div>

  <details class="all">
    <summary>
      All settings
      {#if changedAll}<span class="badge">{changedAll} changed</span>{/if}
    </summary>
    <p class="hint">Everything the models accept. Hover a setting for its description.</p>
    <h4>Speakers</h4>
    <div class="grid">
      {#each groups.diarization as field (field.name)}{@render input(field)}{/each}
    </div>
    <h4>Whisper</h4>
    <div class="grid">
      {#each groups.whisper as field (field.name)}{@render input(field)}{/each}
    </div>
  </details>
  <button class="link" onclick={() => (overrides = {})}>Reset all settings to defaults</button>
</details>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(12rem, 1fr));
    gap: 0.6rem 1rem;
    align-items: end;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.85rem;
  }
  .field.wide {
    grid-column: 1 / -1;
  }
  .name {
    color: var(--muted);
  }
  .check {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    padding-bottom: 0.4rem;
  }
  details {
    margin-top: 0.8rem;
  }
  details > .grid {
    margin-top: 0.6rem;
  }
  .all {
    margin-left: 0.2rem;
  }
  summary {
    cursor: pointer;
    color: var(--muted);
  }
  h4 {
    margin: 1rem 0 0.4rem;
    font-weight: 600;
  }
  .hint {
    color: var(--muted);
    font-size: 0.8rem;
    margin: 0.4rem 0 0;
  }
  .badge {
    background: var(--accent);
    color: var(--bg);
    border-radius: 1rem;
    padding: 0 0.5rem;
    font-size: 0.75rem;
    margin-left: 0.4rem;
  }
  .link {
    margin-top: 1rem;
  }
</style>
