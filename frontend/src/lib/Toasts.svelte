<script lang="ts">
  import { dismiss, toasts } from './toasts.svelte'
</script>

<div class="toasts" aria-live="polite">
  {#each toasts as t (t.id)}
    <button class="toast {t.kind}" onclick={() => dismiss(t.id)} title="Dismiss">{t.text}</button>
  {/each}
</div>

<style>
  .toasts {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: min(26rem, calc(100vw - 2rem));
    z-index: 10;
  }
  .toast {
    text-align: left;
    padding: 0.7rem 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 16px rgb(0 0 0 / 0.3);
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
  }
  .error {
    border-left-color: var(--danger);
  }
  .warning {
    border-left-color: var(--warn);
  }
</style>
