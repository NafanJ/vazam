import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two-page build, matching the routes api.py serves:
//   index.html        → GET /
//   characters.html   → GET /characters.html
// Paths are relative to this config's directory (Vite's project root). Output
// lands in ../static/dist; api.py serves those files and mounts /assets from
// there. base:"/" keeps absolute asset URLs (both pages live at the web root).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: "index.html",
        characters: "characters.html",
      },
    },
  },
});
