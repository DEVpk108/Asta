'use strict'

const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron')
const net = require('net')
const path = require('path')

app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required')

const isMac = process.platform === 'darwin'
const HUD_HOST = process.env.ASTA_HUD_HOST || '127.0.0.1'
const HUD_PORT = Number(process.env.ASTA_HUD_PORT || 18765)
let win = null
let hudSocket = null
let hudReconnectTimer = null
let hudClosing = false
let shutdownSent = false
let hudBuffer = ''
let rendererReady = false
let runtimeReady = false
let latestHudLifecycle = null
let latestHudState = null
let latestHudAudio = null
let latestHudChatHistory = null

function send (cmd) {
  if (win && !win.isDestroyed()) win.webContents.send('asta:command', cmd)
}

function sendHudLifecycle (lifecycle) {
  latestHudLifecycle = lifecycle
  if (!rendererReady || !win || win.isDestroyed()) return
  win.webContents.send('asta:hud-lifecycle', lifecycle)
  if (lifecycle && lifecycle.status === 'ready') runtimeReady = true
  else if (lifecycle && lifecycle.status === 'starting') runtimeReady = false
}

function sendHudState (state) {
  latestHudState = state
  if (!rendererReady || !win || win.isDestroyed()) return
  win.webContents.send('asta:hud-state', state)
}

function sendHudAudio (audio) {
  latestHudAudio = audio
  if (!rendererReady || !win || win.isDestroyed()) return
  win.webContents.send('asta:hud-audio', audio)
}

function sendHudChatHistory (history) {
  latestHudChatHistory = history
  if (!rendererReady || !win || win.isDestroyed()) return
  win.webContents.send('asta:hud-chat-history', history)
}

function sendHudChat (message) {
  if (!rendererReady || !win || win.isDestroyed()) return
  win.webContents.send('asta:hud-chat', message)
}

function flushRendererTelemetry () {
  if (!rendererReady || !win || win.isDestroyed()) return
  if (latestHudLifecycle) win.webContents.send('asta:hud-lifecycle', latestHudLifecycle)
  if (latestHudState) win.webContents.send('asta:hud-state', latestHudState)
  if (latestHudAudio) win.webContents.send('asta:hud-audio', latestHudAudio)
  if (latestHudChatHistory) win.webContents.send('asta:hud-chat-history', latestHudChatHistory)
}

function writeHudMessage (message) {
  if (!hudSocket || hudSocket.destroyed || !hudSocket.writable) {
    console.warn('[HUD] Cannot send command: A.S.T.A. transport is unavailable')
    return false
  }
  try {
    hudSocket.write(JSON.stringify(message) + '\n')
    return true
  } catch (error) {
    console.warn('[HUD] Command send failed:', error.message)
    return false
  }
}

function sendTextToAsta (text) {
  const value = String(text || '').trim()
  if (!value) return false
  return writeHudMessage({ type: 'hud.input', version: 1, input: { text: value } })
}

function selectChatSession (sessionId) {
  const value = String(sessionId || '').trim()
  if (!value) return false
  return writeHudMessage({ type: 'hud.chat_select', version: 1, session_id: value })
}

function startNewChat () {
  return writeHudMessage({ type: 'hud.chat_new', version: 1 })
}

function sendShutdownToAsta () {
  if (shutdownSent) return false
  shutdownSent = true
  return writeHudMessage({ type: 'hud.shutdown', version: 1 })
}

function handleHudMessage (message) {
  if (!message) return
  if (message.type === 'hud.lifecycle') {
    sendHudLifecycle(message.lifecycle || {})
    console.log(`[HUD] A.S.T.A. lifecycle: ${(message.lifecycle || {}).status || 'unknown'}`)
    return
  }
  if (message.type === 'hud.state') {
    sendHudState(message.state || {})
    console.log(`[HUD] A.S.T.A. state: ${(message.state || {}).mode || 'unknown'}`)
    return
  }
  if (message.type === 'hud.audio') {
    sendHudAudio(message.audio || {})
    return
  }
  if (message.type === 'hud.chat_history') {
    sendHudChatHistory({
      session_id: message.session_id || null,
      sessions: Array.isArray(message.sessions) ? message.sessions : [],
      messages: Array.isArray(message.messages) ? message.messages : [],
    })
    return
  }
  if (message.type === 'hud.chat') sendHudChat(message.chat || {})
}

function scheduleHudReconnect () {
  if (hudClosing || hudReconnectTimer) return
  hudReconnectTimer = setTimeout(() => {
    hudReconnectTimer = null
    connectToAstaHud()
  }, 1000)
}

function disconnectHudSocket () {
  const socket = hudSocket
  hudSocket = null
  if (!socket) return
  try { socket.destroy() } catch (e) { /* ignore */ }
}

function connectToAstaHud () {
  if (hudClosing || (hudSocket && !hudSocket.destroyed)) return
  const socket = new net.Socket()
  hudSocket = socket
  hudBuffer = ''
  socket.setEncoding('utf8')
  socket.on('connect', () => console.log(`[HUD] Connected to A.S.T.A. at ${HUD_HOST}:${HUD_PORT}`))
  socket.on('data', (chunk) => {
    hudBuffer += chunk
    const lines = hudBuffer.split('\n')
    hudBuffer = lines.pop() || ''
    for (const line of lines) {
      if (!line.trim()) continue
      try { handleHudMessage(JSON.parse(line)) } catch (e) { console.warn('[HUD] Invalid transport message:', e.message) }
    }
  })
  socket.on('error', (error) => console.warn(`[HUD] A.S.T.A. transport unavailable: ${error.message}`))
  socket.on('close', () => {
    if (hudSocket === socket) hudSocket = null
    if (!hudClosing) scheduleHudReconnect()
  })
  socket.connect(HUD_PORT, HUD_HOST)
}

function createWindow () {
  win = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 760,
    minHeight: 560,
    backgroundColor: '#000000',
    show: false,
    frame: isMac,
    titleBarStyle: isMac ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false
    }
  })

  rendererReady = false
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))
  win.webContents.once('did-finish-load', () => {
    rendererReady = true
    console.log('[HUD] Renderer ready; flushing latest runtime telemetry')
    flushRendererTelemetry()
  })
  win.once('ready-to-show', () => { if (!win.isDestroyed()) win.show() })
  win.on('closed', () => { rendererReady = false; win = null })
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' } })
}

function buildMenu () {
  const template = []
  if (isMac) template.push({ role: 'appMenu' })
  template.push({
    label: 'Core',
    submenu: [
      { label: 'Idle', accelerator: 'CmdOrCtrl+1', click: () => send('state:idle') },
      { label: 'Listening', accelerator: 'CmdOrCtrl+2', click: () => send('state:listening') },
      { label: 'Thinking', accelerator: 'CmdOrCtrl+3', click: () => send('state:thinking') },
      { label: 'Speaking', accelerator: 'CmdOrCtrl+4', click: () => send('state:speaking') },
      { type: 'separator' },
      { label: 'Re-assemble', accelerator: 'CmdOrCtrl+R', click: () => send('reassemble') },
      { label: 'Mute ambience', accelerator: 'CmdOrCtrl+M', click: () => send('mute') },
      { type: 'separator' },
      isMac ? { role: 'close' } : { role: 'quit' }
    ]
  })
  template.push({
    label: 'View',
    submenu: [
      { role: 'togglefullscreen' },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' },
      { role: 'zoomIn' },
      { role: 'zoomOut' }
    ]
  })
  template.push({ role: 'windowMenu' })
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

const gotLock = app.requestSingleInstanceLock()
if (!gotLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) win.restore()
      win.show()
      win.focus()
    }
  })
  app.whenReady().then(() => {
    buildMenu()
    createWindow()
    connectToAstaHud()
    app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
  })
  app.on('before-quit', (event) => {
    if (hudClosing) return
    hudClosing = true
    event.preventDefault()
    if (hudReconnectTimer) clearTimeout(hudReconnectTimer)
    hudReconnectTimer = null
    sendShutdownToAsta()
    setTimeout(() => { disconnectHudSocket(); app.exit(0) }, 150)
  })
  app.on('window-all-closed', () => { if (!isMac) app.quit() })
}

ipcMain.on('win:minimize', () => { if (win) win.minimize() })
ipcMain.on('win:toggle-maximize', () => {
  if (!win) return
  if (win.isMaximized()) win.unmaximize()
  else win.maximize()
})
ipcMain.on('win:close', () => { if (win) win.close() })
ipcMain.on('win:toggle-fullscreen', () => { if (win) win.setFullScreen(!win.isFullScreen()) })
ipcMain.on('hud:text-message', (_event, text) => { sendTextToAsta(text) })
ipcMain.on('hud:select-chat', (_event, sessionId) => { selectChatSession(sessionId) })
ipcMain.on('hud:new-chat', () => { startNewChat() })
