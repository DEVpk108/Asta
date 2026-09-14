'use strict'

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('asta', {
  platform: process.platform,
  minimize: () => ipcRenderer.send('win:minimize'),
  toggleMaximize: () => ipcRenderer.send('win:toggle-maximize'),
  close: () => ipcRenderer.send('win:close'),
  toggleFullscreen: () => ipcRenderer.send('win:toggle-fullscreen'),
  onCommand: (fn) => {
    ipcRenderer.on('asta:command', (_event, cmd) => {
      try { fn(cmd) } catch (e) { /* ignore */ }
    })
  },
  onHudState: (fn) => {
    ipcRenderer.on('asta:hud-state', (_event, state) => {
      try { fn(state) } catch (e) { /* ignore */ }
    })
  }
})
