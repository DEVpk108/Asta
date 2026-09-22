'use strict'

;(function () {
  var $ = function (id) { return document.getElementById(id) }

  var glCanvas = $('gl')
  var fbCanvas = $('fallback')
  var readoutLabel = $('readoutLabel')
  var readoutPct = $('readoutPct')
  var statusMode = $('statusMode')
  var statusIntensity = $('statusIntensity')
  var waveform = $('waveform')
  var engineLabel = $('engineLabel')
  var engineLed = $('engineLed')
  var particleCountEl = $('particleCount')
  var fpsEl = $('fps')
  var uptimeEl = $('uptime')
  var reassembleBtn = $('reassembleBtn')
  var dockButtons = document.querySelectorAll('.dock button[data-state]')

  /* ---------- desktop shell bridge ---------- */
  var bridge = window.asta || null
  if (bridge && bridge.platform === 'darwin') document.body.classList.add('is-mac')

  if (!bridge) {
    var right = document.querySelector('.tb-right')
    if (right) right.style.display = 'none'
  } else {
    var wcMin = $('wcMin'), wcMax = $('wcMax'), wcClose = $('wcClose')
    if (wcMin) wcMin.addEventListener('click', function () { bridge.minimize() })
    if (wcMax) wcMax.addEventListener('click', function () { bridge.toggleMaximize() })
    if (wcClose) wcClose.addEventListener('click', function () { bridge.close() })
    var drag = document.querySelector('.tb-drag')
    if (drag) drag.addEventListener('dblclick', function () { bridge.toggleMaximize() })
    if (bridge.onCommand) {
      bridge.onCommand(function (cmd) {
        if (cmd === 'reassemble') reassemble()
        else if (cmd === 'mute') toggleMute()
        else if (cmd && cmd.indexOf('state:') === 0) {
          forceComplete()
          setVisualProfile(cmd.slice(6))
        }
      })
    }
  }

  /* ---------- visual profiles ---------- */
  var STATES = {
    idle: { mode: 'IDLE', intensity: 'LOW', amp: 0.17, colorMix: 0.30, wave: 0.16 },
    listening: { mode: 'LISTENING', intensity: 'MEDIUM', amp: 0.52, colorMix: 0.58, wave: 0.42 },
    thinking: { mode: 'THINKING', intensity: 'HIGH', amp: 0.74, colorMix: 0.82, wave: 0.62 },
    speaking: { mode: 'SPEAKING', intensity: 'HIGH', amp: 0.52, colorMix: 1.00, wave: 0.32 }
  }

  var MODE_LABELS = {
    idle: 'IDLE',
    listening: 'LISTENING',
    thinking: 'THINKING',
    speaking: 'SPEAKING',
    executing: 'EXECUTING',
    approval: 'APPROVAL',
    error: 'ERROR'
  }

  var INTENSITY_LABELS = {
    low: 'LOW',
    medium: 'MEDIUM',
    high: 'HIGH'
  }

  var VISUAL_PROFILE = {
    idle: 'idle',
    listening: 'listening',
    thinking: 'thinking',
    speaking: 'speaking',
    executing: 'thinking',
    approval: 'listening',
    error: 'thinking'
  }

  var VALID_MODES = {
    idle: true,
    listening: true,
    thinking: true,
    speaking: true,
    executing: true,
    approval: true,
    error: true
  }

  var VALID_INTENSITIES = {
    low: true,
    medium: true,
    high: true
  }

  var current = 'idle'
  var visualProfile = 'idle'
  var canonicalState = {
    mode: 'idle',
    intensity: 'low',
    status: 'IDLE',
    progress: null,
    activity: null
  }
  var live = { amp: 0.05, colorMix: 0, wave: 0.1 }
  var audioTarget = 0
  var audioLevel = 0
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  var ASSEMBLE_MS = reduced ? 900 : 5200
  var assembleStart = performance.now()
  var assembled = false
  var startupReady = false
  var progress = 0

  /* ---------- engine boot ---------- */
  var engine = null
  var engineError = ''
  try {
    if (window.AstaEngine && window.AstaEngine.create) engine = window.AstaEngine.create(glCanvas)
    else engineError = 'engine script missing'
    if (engine && !engine.ok) engineError = 'engine returned no ok flag'
  } catch (e) {
    engine = null
    engineError = (e && e.message) ? e.message : String(e)
    if (window.console) console.error('[ASTA] WebGL engine failed:', e)
  }

  if (engine && engine.ok) {
    engineLabel.textContent = 'ENGINE WEBGL2 · BLOOM ' + (engine.hdr ? 'HDR' : 'ON')
    particleCountEl.textContent = engine.count.toLocaleString()
  } else {
    glCanvas.style.display = 'none'
    fbCanvas.style.display = 'block'
    engine = window.AstaFallback2D ? window.AstaFallback2D.create(fbCanvas) : null
    engineLabel.textContent = 'ENGINE CANVAS2D · FALLBACK'
      + (engineError ? ' · ' + engineError.slice(0, 70) : '')
    engineLed.classList.add('warn')
    particleCountEl.textContent = '—'
  }

  /* ---------- waveform bars ---------- */
  var bars = []
  for (var b = 0; b < 38; b++) {
    var bar = document.createElement('i')
    waveform.appendChild(bar)
    bars.push(bar)
  }

  function speakReady () {
    try {
      if (!window.speechSynthesis) return
      var u = new SpeechSynthesisUtterance('Ready')
      u.rate = 0.85
      u.pitch = 0.55
      u.volume = 0.9
      window.speechSynthesis.speak(u)
    } catch (e) { /* ignore */ }
  }

  function beginRuntime () {
    if (startupReady) return
    startupReady = true
    assembled = false
    progress = 0
    assembleStart = performance.now()
  }

  window.AstaHUD = window.AstaHUD || {}
  window.AstaHUD.beginRuntime = beginRuntime
  window.AstaHUD.applyCanonicalState = applyCanonicalState
  window.AstaHUD.setAudioLevel = setAudioLevel

  /* ---------- state handling ---------- */
  function updateCanonicalLabels () {
    var mode = canonicalState.mode
    var intensity = canonicalState.intensity
    statusMode.textContent = MODE_LABELS[mode] || 'IDLE'
    statusIntensity.textContent = INTENSITY_LABELS[intensity] || 'LOW'

    if (assembled) {
      readoutLabel.textContent = canonicalState.status || MODE_LABELS[mode] || 'STATUS:'
      readoutPct.textContent = MODE_LABELS[mode] || 'IDLE'
      readoutPct.className = 'pct status-big' + (mode === 'speaking' ? ' pulse' : '')
    }
  }

  function setVisualProfile (name) {
    if (!STATES[name]) return
    visualProfile = name
    current = name

    for (var i = 0; i < dockButtons.length; i++) {
      var isActive = dockButtons[i].getAttribute('data-state') === name
      if (isActive) dockButtons[i].classList.add('active')
      else dockButtons[i].classList.remove('active')
    }

    if (assembled) updateCanonicalLabels()
  }

  function setState (name) {
    setVisualProfile(name)
    canonicalState = {
      mode: name,
      intensity: STATES[name] ? String(STATES[name].intensity).toLowerCase() : 'low',
      status: STATES[name] ? STATES[name].mode : 'IDLE',
      progress: null,
      activity: 'manual'
    }
  }

  function applyCanonicalState (state) {
    state = state || {}

    var mode = String(state.mode || 'idle').toLowerCase()
    var intensity = String(state.intensity || 'low').toLowerCase()
    if (!VALID_MODES[mode]) mode = 'idle'
    if (!VALID_INTENSITIES[intensity]) intensity = 'low'

    canonicalState = {
      mode: mode,
      intensity: intensity,
      status: state.status || MODE_LABELS[mode],
      progress: state.progress == null ? null : Number(state.progress),
      activity: state.activity || null
    }

    /* Before runtime readiness, state is buffered without cancelling the
       synchronized startup assembly. Once started, backend state is applied
       normally and remains authoritative. */
    if (!startupReady) {
      updateCanonicalLabels()
      return
    }

    if (!assembled) {
      assembled = true
      progress = 1
    }

    setVisualProfile(VISUAL_PROFILE[mode] || 'idle')
    updateCanonicalLabels()
  }

  function setAudioLevel (level) {
    var value = Number(level)
    if (!isFinite(value)) value = 0
    audioTarget = Math.max(0, Math.min(1, value))
  }

  function renderBigStatus () {
    readoutLabel.textContent = canonicalState.status || MODE_LABELS[canonicalState.mode] || 'STATUS:'
    readoutPct.textContent = MODE_LABELS[canonicalState.mode] || 'IDLE'
    readoutPct.className = 'pct status-big' + (canonicalState.mode === 'speaking' ? ' pulse' : '')
  }

  function completeAssembly () {
    assembled = true
    progress = 1
    if (!canonicalState || canonicalState.mode === 'idle') {
      canonicalState = {
        mode: 'idle',
        intensity: 'low',
        status: 'IDLE',
        progress: 1,
        activity: null
      }
      setVisualProfile('idle')
    }
    renderBigStatus()
    speakReady()
  }

  function forceComplete () {
    if (!startupReady) beginRuntime()
    if (!assembled) completeAssembly()
  }

  function reassemble () {
    if (!startupReady) beginRuntime()
    assembled = false
    progress = 0
    assembleStart = performance.now()
    canonicalState = {
      mode: 'idle',
      intensity: 'low',
      status: 'IDLE',
      progress: null,
      activity: 'assembly'
    }
    audioTarget = 0
    readoutPct.className = 'pct'
    readoutLabel.textContent = 'ASSEMBLING...'
    readoutPct.textContent = '0%'
    setVisualProfile('idle')
  }
  if (reassembleBtn) reassembleBtn.addEventListener('click', reassemble)

  for (var d = 0; d < dockButtons.length; d++) {
    dockButtons[d].addEventListener('click', function () {
      forceComplete()
      setState(this.getAttribute('data-state'))
      updateCanonicalLabels()
    })
  }

  function toggleFullscreen () {
    if (bridge && bridge.toggleFullscreen) {
      bridge.toggleFullscreen()
      return
    }
    if (!document.fullscreenElement) {
      if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen()
    } else if (document.exitFullscreen) {
      document.exitFullscreen()
    }
  }

  window.addEventListener('keydown', function (e) {
    var target = e.target
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return

    if (e.metaKey || e.ctrlKey || e.altKey) return
    var k = (e.key || '').toLowerCase()
    var map = { '1': 'idle', '2': 'listening', '3': 'thinking', '4': 'speaking' }
    if (map[k]) {
      e.preventDefault()
      forceComplete()
      setState(map[k])
      updateCanonicalLabels()
    } else if (k === 'r') {
      e.preventDefault()
      reassemble()
    } else if (k === 'f') {
      e.preventDefault()
      toggleFullscreen()
    }
  })

  /* ---------- render loop ---------- */
  var rot = 0
  var swayT = 0
  var last = performance.now()
  var startedAt = performance.now()
  var frames = 0
  var fpsTimer = 0
  var uptimeTimer = 0

  function pad (n) { return (n < 10 ? '0' : '') + n }

  function frame (now) {
    var dt = Math.min(0.05, (now - last) / 1000)
    last = now

    if (startupReady && !assembled) {
      progress = Math.min(1, (now - assembleStart) / ASSEMBLE_MS)
      if (progress >= 1) completeAssembly()
    }

    /* Smooth real speech telemetry before it touches the particle engine. */
    var audioK = Math.min(1, dt * 16.0)
    audioLevel += (audioTarget - audioLevel) * audioK

    var target = STATES[visualProfile]
    var liveAudio = canonicalState.mode === 'speaking' ? audioLevel : 0
    var wantAmp = assembled
      ? Math.min(1.30, target.amp + liveAudio * 0.62)
      : (0.18 + 0.42 * progress)
    var wantMix = assembled
      ? Math.min(1.20, target.colorMix + liveAudio * 0.18)
      : (0.15 + 0.55 * progress)
    var wantWave = assembled
      ? Math.min(1.45, target.wave + liveAudio * 1.05)
      : (0.12 + 0.30 * progress)

    var k = Math.min(1, dt * 3.0)
    live.amp += (wantAmp - live.amp) * k
    live.colorMix += (wantMix - live.colorMix) * k
    live.wave += (wantWave - live.wave) * k

    /* gentle sway instead of a full spin, so the face stays toward the viewer */
    swayT += dt * (0.17 + 0.10 * live.amp + liveAudio * 0.12)
    rot = Math.sin(swayT) * (0.14 + liveAudio * 0.04)

    if (engine && engine.render) {
      engine.render({
        time: now / 1000,
        progress: progress,
        amp: live.amp,
        colorMix: live.colorMix,
        rot: rot,
        bloom: 1.05 + 0.55 * live.amp + 0.65 * liveAudio
      })
    }

    if (!assembled) readoutPct.textContent = Math.round(progress * 100) + '%'

    for (var i = 0; i < bars.length; i++) {
      var center = 1 - Math.abs(i / (bars.length - 1) - 0.5) * 2
      var wob = 0.35 + 0.65 * Math.abs(Math.sin(now / 1000 * (2.1 + i * 0.17) + i))
      var signal = live.wave * (0.35 + liveAudio * 1.8)
      bars[i].style.height = (3 + 25 * signal * wob * (0.35 + 0.65 * center)).toFixed(1) + 'px'
    }

    frames++
    fpsTimer += dt * 1000
    if (fpsTimer >= 500) {
      fpsEl.textContent = Math.round(frames / (fpsTimer / 1000))
      frames = 0
      fpsTimer = 0
    }

    uptimeTimer += dt * 1000
    if (uptimeTimer >= 1000) {
      uptimeTimer = 0
      var secs = Math.floor((now - startedAt) / 1000)
      uptimeEl.textContent = pad(Math.floor(secs / 60)) + ':' + pad(secs % 60)
    }

    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)

  window.addEventListener('resize', function () {
    if (engine && engine.resize) engine.resize()
  })

  setVisualProfile('idle')
  updateCanonicalLabels()
})()