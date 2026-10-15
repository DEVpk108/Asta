'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const path = require('node:path')
const root = path.join(__dirname, '..', '..')
function source (name) { return fs.readFileSync(path.join(root, name), 'utf8') }

class Element {
  constructor () { this.children = []; this.dataset = {}; this.style = {}; this.classList = { toggle () {}, contains () { return false } }; this.textContent = '' }
  appendChild (child) { this.children.push(child); child.parentNode = this; return child }
  append (...children) { children.forEach(child => this.appendChild(child)) }
  removeChild (child) { this.children.splice(this.children.indexOf(child), 1); child.parentNode = null }
  addEventListener () {}
  setAttribute () {}
  focus () {}
  querySelector () { return null }
}
function renderer () {
  const nodes = new Map(['chatForm', 'chatInput', 'chatMessages', 'chatPanel', 'chatRecentList', 'chatRecentEmpty'].map(id => [id, new Element()]))
  const callbacks = {}, frames = [], acknowledgements = [], views = []
  const bridge = {
    onCaptureResponse (callback) { callbacks.capture = callback },
    onHudChat (callback) { callbacks.chat = callback },
    onHudChatHistory (callback) { callbacks.history = callback },
    acknowledgeResponseRender (id) { acknowledgements.push(id) }
  }
  const document = { visibilityState: 'visible', body: new Element(),
    getElementById (id) { return nodes.get(id) || null }, createElement () { return new Element() }, createTextNode (text) { const node = new Element(); node.textContent = text; return node } }
  const window = { asta: bridge, innerWidth: 1200, addEventListener () {}, requestAnimationFrame (fn) { frames.push(fn) }, AstaViews: { show (view) { views.push(view) } } }
  vm.runInNewContext(source('hud/renderer/chat_ui.js'), { window, document, console, navigator: {} })
  function frame () { const fns = frames.splice(0); fns.forEach(fn => fn()) }
  const id = 'a'.repeat(32)
  function message (phase, text, responseId = id) { callbacks.capture({ phase, text, response_id: responseId }) }
  return { callbacks, document, nodes, frames, frame, message, acknowledgements, views, id }
}

test('full filtered response replaces only ticketed fragments and ack waits two frames', () => {
  const r = renderer()
  r.callbacks.chat({ role: 'assistant', text: 'Older answer' })
  const old = r.nodes.get('chatMessages').children[0]
  r.message('begin')
  r.callbacks.chat({ role: 'assistant', text: 'Partial answer.' })
  r.callbacks.chat({ role: 'user', text: 'User text' })
  r.message('ready', 'Complete **answer**.')
  assert.equal(r.nodes.get('chatMessages').children.length, 3)
  assert.equal(r.nodes.get('chatMessages').children[0], old)
  assert.deepEqual(r.views, ['chat'])
  assert.equal(r.nodes.get('chatMessages').children[2].style.animation, 'none')
  assert.equal(r.nodes.get('chatMessages').children[2].style.opacity, '1')
  assert.deepEqual(r.acknowledgements, [])
  r.frame(); assert.deepEqual(r.acknowledgements, [])
  r.frame(); assert.deepEqual(r.acknowledgements, [r.id])
  r.message('ready', 'Duplicate'); r.frame(); r.frame()
  assert.equal(r.acknowledgements.length, 1)
})

test('non-streaming model still places full response into the DOM', () => {
  const r = renderer(); r.message('begin'); r.message('ready', 'Non-streaming answer')
  assert.equal(r.nodes.get('chatMessages').children.length, 1)
  r.frame(); r.frame(); assert.deepEqual(r.acknowledgements, [r.id])
})

for (const interrupt of ['cancel', 'new_response', 'history', 'hidden']) {
  test(`${interrupt} suppresses queued acknowledgement`, () => {
    const r = renderer(); r.message('begin'); r.message('ready', 'Answer'); r.frame()
    if (interrupt === 'cancel') r.message('cancel')
    if (interrupt === 'new_response') r.message('begin', '', 'b'.repeat(32))
    if (interrupt === 'history') r.callbacks.history({ messages: [], sessions: [] })
    if (interrupt === 'hidden') r.document.visibilityState = 'hidden'
    r.frame(); assert.deepEqual(r.acknowledgements, [])
  })
}

test('unknown token, malformed token and ready without begin are ignored', () => {
  const r = renderer(); r.message('ready', 'Unknown')
  r.message('begin'); r.message('ready', 'Wrong', 'b'.repeat(32)); r.message('ready', 'Bad', '../bad')
  r.frame(); r.frame(); assert.deepEqual(r.acknowledgements, [])
  assert.equal(r.nodes.get('chatMessages').children.length, 0)
})

test('preload carries render ticket and returns response-specific acknowledgement', () => {
  const callbacks = {}, sent = []; let exposed
  const ipcRenderer = { on (channel, fn) { callbacks[channel] = fn }, send (...message) { sent.push(message) } }
  const electron = { contextBridge: { exposeInMainWorld (name, api) { exposed = api } }, ipcRenderer }
  vm.runInNewContext(source('hud/preload.js'), { require: () => electron, process })
  let received
  exposed.onCaptureResponse(message => { received = message })
  const ticket = { phase: 'ready', response_id: 'a'.repeat(32) }
  callbacks['asta:capture-response']({}, ticket)
  assert.equal(received, ticket)
  exposed.acknowledgeResponseRender(ticket.response_id)
  assert.deepEqual(sent, [['hud:rendered', ticket.response_id]])
})

test('Electron forwards ephemeral ticket and accepts ack only from current window', () => {
  const handlers = {}, sent = [], writes = []
  const webContents = { send (...message) { sent.push(message) } }
  const electron = { app: { commandLine: { appendSwitch () {} }, whenReady () { return { then () {} } }, on () {}, requestSingleInstanceLock () { return true } },
    ipcMain: { on (name, callback) { handlers[name] = callback } }, Menu: {}, shell: {}, BrowserWindow: {} }
  const sandbox = { require: name => name === 'electron' ? electron : require(name), process, console, __dirname: path.join(root, 'hud'), setTimeout, clearTimeout }
  vm.createContext(sandbox)
  vm.runInContext(source('hud/main.js'), sandbox)
  sandbox.testWindow = { webContents, isDestroyed () { return false } }
  sandbox.testSocket = { destroyed: false, writable: true, write (value) { writes.push(JSON.parse(value)) } }
  vm.runInContext('win = testWindow; hudSocket = testSocket; rendererReady = true', sandbox)
  const ticket = { type: 'hud.capture_response', phase: 'ready', response_id: 'a'.repeat(32), text: 'Answer' }
  sandbox.handleHudMessage(ticket)
  assert.deepEqual(sent, [['asta:capture-response', ticket]])
  handlers['hud:rendered']({ sender: {} }, ticket.response_id)
  handlers['hud:rendered']({ sender: webContents }, 'bad')
  assert.equal(writes.length, 0)
  handlers['hud:rendered']({ sender: webContents }, ticket.response_id)
  assert.deepEqual(writes, [{ type: 'hud.rendered', version: 1, response_id: ticket.response_id }])
  sent.length = 0
  sandbox.flushRendererTelemetry()
  assert.equal(sent.length, 0, 'Capture tickets must not enter telemetry replay cache')
})
