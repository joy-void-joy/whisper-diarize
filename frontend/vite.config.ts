import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [svelte()],
  // Relative asset URLs, so the build also works when served under a sub-path.
  base: './',
  server: {
    proxy: { '/api': 'http://127.0.0.1:5801' },
  },
})
