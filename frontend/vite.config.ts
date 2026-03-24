import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('@splinetool')) return 'spline-runtime'
          if (id.includes('framer-motion') || id.includes('motion-dom')) return 'framer'
          if (id.includes('react-dom') || id.includes('react-router')) return 'react-vendor'
        },
      },
    },
  },
})
