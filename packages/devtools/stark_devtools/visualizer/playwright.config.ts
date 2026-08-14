import { defineConfig } from "@playwright/test";

// Smoke test against the built bundle served by `vite preview`. No backend, so the
// app falls back to the bundled sample.json (the /api/* fetches fail and are caught).
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: { baseURL: "http://localhost:4173", trace: "off" },
  webServer: {
    command: "npm run preview",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
