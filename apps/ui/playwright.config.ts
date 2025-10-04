import { defineConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const projectRoot = path.resolve(__dirname, '..', '..');
const tmpDir = path.resolve(projectRoot, 'tmp', 'playwright');
const uploadsDir = path.join(tmpDir, 'uploads');
const qdrantDir = path.join(tmpDir, 'qdrant');
const dbPath = path.join(tmpDir, 'app.db');

for (const dir of [uploadsDir, qdrantDir]) {
  fs.mkdirSync(dir, { recursive: true });
}
fs.mkdirSync(path.dirname(dbPath), { recursive: true });

const sqliteUrl = `sqlite:///${dbPath.replace(/\\/g, '/')}`;

export default defineConfig({
  testDir: path.resolve(__dirname, 'e2e'),
  timeout: 120_000,
  expect: {
    timeout: 30_000,
  },
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:3100',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    locale: 'fa-IR',
  },
  webServer: [
    {
      command: 'poetry run uvicorn apps.api.main:app --host 127.0.0.1 --port 8000',
      cwd: projectRoot,
      env: {
        ...process.env,
        AUTH_REQUIRED: 'true',
        UPLOAD_DIR: uploadsDir,
        DATABASE_URL: sqliteUrl,
        QDRANT_MODE: 'local',
        QDRANT_LOCATION: qdrantDir,
        EMBED_MODEL: 'mock',
        LLM_BACKEND: 'mock',
        LLM_MODEL: 'mock',
      },
      port: 8000,
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'npm run dev -- --port 3100',
      cwd: __dirname,
      env: {
        ...process.env,
        PORT: '3100',
        NEXT_TELEMETRY_DISABLED: '1',
        KETABMIND_API_ORIGIN: 'http://127.0.0.1:8000',
      },
      port: 3100,
      timeout: 120_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
