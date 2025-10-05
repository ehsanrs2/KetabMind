const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let backendProcess;

function startBackend() {
  if (backendProcess) {
    return backendProcess;
  }

  const pythonExecutable = process.env.PYTHON || process.env.PYTHON_EXE || 'python';
  const backendEntry = path.resolve(__dirname, '..', '..', 'backend', 'main.py');

  backendProcess = spawn(pythonExecutable, [backendEntry], {
    cwd: path.resolve(__dirname, '..', '..'),
    env: process.env,
    stdio: 'inherit',
  });

  backendProcess.on('exit', (code, signal) => {
    if (code && code !== 0) {
      console.error(`FastAPI backend exited with code ${code}`);
    }
    if (signal) {
      console.warn(`FastAPI backend terminated via signal ${signal}`);
    }
    backendProcess = undefined;
  });

  backendProcess.on('error', (error) => {
    console.error('Unable to launch FastAPI backend:', error);
  });

  return backendProcess;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1024,
    height: 768,
    webPreferences: {
      contextIsolation: true,
    },
  });

  const appUrl = process.env.FRONTEND_URL || 'http://localhost:3000';
  void win.loadURL(appUrl);
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
});
