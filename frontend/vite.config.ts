import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The backend runs separately (uvicorn on port 8000); the dev server forwards /api to it,
// so the browser sees a single origin and no CORS setup is needed.
const backend = process.env.NEUROITBURD_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
  preview: {
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
});
