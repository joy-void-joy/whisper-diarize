<script lang="ts">
  // Scrolling bars of the latest input peaks, newest on the right: shows the mic is hearing you.
  let { levels, height = 40 }: { levels: number[]; height?: number } = $props()
  let canvas: HTMLCanvasElement

  $effect(() => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = devicePixelRatio || 1
    const width = canvas.clientWidth
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.fillStyle = getComputedStyle(canvas).color
    const step = 4 // 3 px bar + 1 px gap
    const shown = levels.slice(-Math.floor(width / step))
    shown.forEach((peak, i) => {
      // sqrt: speech peaks sit around 0.05-0.5, so a linear scale would look flat.
      const bar = Math.max(1, Math.min(1, Math.sqrt(peak) * 1.3) * height)
      ctx.fillRect(width - (shown.length - i) * step, (height - bar) / 2, step - 1, bar)
    })
  })
</script>

<canvas bind:this={canvas} style:height="{height}px"></canvas>

<style>
  canvas {
    display: block;
    width: 100%;
    color: var(--accent);
  }
</style>
