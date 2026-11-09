// frontend/electron/main.cjs
// Electron Main process for BreatheFlow AI desktop runner with Presage SmartSpectra SDK

'use strict';

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const dotenv = require('dotenv');

const fs = require('fs');

// Robust root .env resolution
const possibleEnvPaths = [
  path.resolve(__dirname, '../../.env'),
  path.resolve(__dirname, '../.env'),
  path.resolve(process.cwd(), '.env'),
  path.resolve(process.cwd(), '../.env'),
];

for (const p of possibleEnvPaths) {
  if (fs.existsSync(p)) {
    dotenv.config({ path: p });
    break;
  }
}

const { bindSmartSpectraIpc } = require('@smartspectra/node-sdk/main');

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 940,
    minWidth: 1024,
    minHeight: 768,
    title: 'BreatheFlow AI - Powered by Presage SmartSpectra & Gnani Conversational Voice',
    backgroundColor: '#0a0d14',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // Required for SmartSpectra preload bridge MessageChannel
    },
  });

  // Explicitly grant camera and microphone media permissions for local dev
  mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') {
      return callback(true);
    }
    callback(true);
  });

  // Intercept the MessagePort on attach so configure options automatically inherit
  // developer SMARTSPECTRA_API_KEY loaded from root .env in the main process,
  // keeping credentials completely out of the renderer, localStorage, and IPC responses.
  ipcMain.on('smartspectra:attach', (event) => {
    if (event.ports && event.ports[0]) {
      const port = event.ports[0];
      const origOn = port.on.bind(port);
      port.on = function (name, listener) {
        if (name === 'message') {
          const wrapped = (msgEvent) => {
            if (msgEvent && msgEvent.data && msgEvent.data.kind === 'configure') {
              if (!msgEvent.data.options) msgEvent.data.options = {};
              msgEvent.data.options.apiKey = process.env.SMARTSPECTRA_API_KEY || '';
            }
            return listener(msgEvent);
          };
          return origOn(name, wrapped);
        }
        return origOn(name, listener);
      };
    }
  });

  // Bind Presage SmartSpectra IPC bridge to mainWindow
  bindSmartSpectraIpc(mainWindow, {
    logger: (level, msg) => {
      console.log(`[SmartSpectra:${level}]`, msg);
    },
  });

  // Safe IPC handler: reports developer configuration status without returning or logging the key
  ipcMain.handle('smartspectra:check-config', () => {
    const key = process.env.SMARTSPECTRA_API_KEY;
    return {
      configured: Boolean(key && key.trim().length > 0),
    };
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

  if (process.env.NODE_ENV === 'production' && !process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  } else {
    mainWindow.loadURL(devServerUrl).catch((err) => {
      console.error('Failed to load dev server, falling back to dist/index.html if present:', err.message);
      mainWindow.loadFile(path.join(__dirname, '../dist/index.html')).catch(() => {});
    });
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
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
