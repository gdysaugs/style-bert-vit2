import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Match the EXPOSE port in Dockerfile and ports in docker-compose.yml
    // host: '0.0.0.0' // Already set via CMD flag in Dockerfile
  }
}) 