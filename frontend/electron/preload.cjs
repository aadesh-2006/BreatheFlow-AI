// frontend/electron/preload.cjs
// Official Presage SmartSpectra preload bridge + BreatheFlow secure IPC helpers

'use strict';

const { contextBridge, ipcRenderer } = require('electron');

const ATTACH_CHANNEL = 'smartspectra:attach';

let port = null;
let messageListener = null;

const DIAGNOSTICS = process.env.SMARTSPECTRA_DIAGNOSTICS === '1';

const onPortMessage = (event) => {
  if (DIAGNOSTICS) {
    const kind = event.data && event.data.kind;
    console.log(`[smartspectra/preload] <- ${kind || '?'}`);
  }
  if (messageListener) messageListener(event.data);
};

// Expose official Presage SmartSpectra bridge to the renderer main world
contextBridge.exposeInMainWorld('__smartspectraBridge', {
  diagnostics: DIAGNOSTICS,

  attach() {
    if (port) return; // idempotent
    const channel = new MessageChannel();
    port = channel.port1;
    port.onmessage = onPortMessage;
    ipcRenderer.postMessage(ATTACH_CHANNEL, null, [channel.port2]);
    port.start();
  },

  postMessage(message) {
    if (!port) throw new Error('@smartspectra/node-sdk/preload: postMessage before attach()');
    port.postMessage(message);
  },

  onMessage(callback) {
    messageListener = typeof callback === 'function' ? callback : null;
  },

  close() {
    if (!port) return;
    try {
      port.close();
    } finally {
      port = null;
      messageListener = null;
    }
  },
});

// Expose safe BreatheFlow configuration check without leaking any secret
contextBridge.exposeInMainWorld('electronAPI', {
  isElectron: true,
  checkSmartSpectraConfig: () => ipcRenderer.invoke('smartspectra:check-config'),
});
