'use strict'

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('asta', {
  platform: process.platform,
  minimize: () => ipcRenderer.send('win:minimize'),
  toggleMaximize: () => ipcRenderer.send('win:toggle-maximize'),
  close: () => ipcRenderer.send('win:close'),
  toggleFullscreen: () => ipcRenderer.send('win:toggle-fullscreen'),
  sendTextMessage: (text) => ipcRenderer.send('hud:text-message', text),
  selectChatSession: (sessionId) => ipcRenderer.send('hud:select-chat', sessionId),
  deleteChatSession: (sessionId) => ipcRenderer.send('hud:delete-chat', sessionId),
  startNewChat: () => ipcRenderer.send('hud:new-chat'),
  setConversationMode: (enabled) => ipcRenderer.send('hud:conversation-mode', !!enabled),
  onCommand: (fn) => {
    ipcRenderer.on('asta:command', (_event, cmd) => {
      try { fn(cmd) } catch (e) { /* ignore */ }
    })
  },
  onHudLifecycle: (fn) => {
    ipcRenderer.on('asta:hud-lifecycle', (_event, lifecycle) => {
      try { fn(lifecycle) } catch (e) { /* ignore */ }
    })
  },
  onHudState: (fn) => {
    ipcRenderer.on('asta:hud-state', (_event, state) => {
      try { fn(state) } catch (e) { /* ignore */ }
    })
  },
  onHudAudio: (fn) => {
    ipcRenderer.on('asta:hud-audio', (_event, audio) => {
      try { fn(audio) } catch (e) { /* ignore */ }
    })
  },
  onHudChatHistory: (fn) => {
    ipcRenderer.on('asta:hud-chat-history', (_event, history) => {
      try { fn(history) } catch (e) { /* ignore */ }
    })
  },
  onHudChatSessions: (fn) => {
    ipcRenderer.on('asta:hud-chat-sessions', (_event, sessions) => {
      try { fn(sessions) } catch (e) { /* ignore */ }
    })
  },
  onHudConversationMode: (fn) => {
    ipcRenderer.on('asta:hud-conversation-mode', (_event, enabled) => {
      try { fn(!!enabled) } catch (e) { /* ignore */ }
    })
  },
  onHudChat: (fn) => {
    ipcRenderer.on('asta:hud-chat', (_event, message) => {
      try { fn(message) } catch (e) { /* ignore */ }
    })
  }
})
