'use strict'

const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron')
const path = require('path')

/* the HUD starts its ambient drone on boot, so allow audio without a gesture */
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required')

const isMac = process.platform === 'darwin'
let win = null

function send (cmd) {
  if (win && !win.isDestroyed()) win.webContents.send('asta:command', cmd)
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

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))
  win.once('ready-to-show', () => win.show())
  win.on('closed', () => { win = null })

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
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
      win.focus()
    }
  })

  app.whenReady().then(() => {
    buildMenu()
    createWindow()
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
  })

  app.on('window-all-closed', () => {
    if (!isMac) app.quit()
  })
}

ipcMain.on('win:minimize', () => { if (win) win.minimize() })
ipcMain.on('win:toggle-maximize', () => {
  if (!win) return
  if (win.isMaximized()) win.unmaximize()
  else win.maximize()
})
ipcMain.on('win:close', () => { if (win) win.close() })
ipcMain.on('win:toggle-fullscreen', () => { if (win) win.setFullScreen(!win.isFullScreen()) })
