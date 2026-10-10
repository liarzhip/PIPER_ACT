import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

// Keep Host and Origin unchanged: Flask validates same-origin device commands.
const proxy = { '/api': { target: 'http://127.0.0.1:5088', changeOrigin: false } };
export default defineConfig({
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, strictPort: true, proxy },
  build: {
    rollupOptions: {
      input: {
        console: fileURLToPath(new URL('./index.html', import.meta.url)),
        review: fileURLToPath(new URL('./review.html', import.meta.url)),
      },
    },
  },
});