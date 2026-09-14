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
  var muteBtn = $('muteBtn')
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
          setState(cmd.slice(6))
        }
      })
    }
  }

  /* ---------- states ---------- */
  var STATES = {
    idle: { mode: 'IDLE', intensity: 'LOW', amp: 0.17, colorMix: 0.30, wave: 0.16 },
    listening: { mode: 'LISTENING', intensity: 'MEDIUM', amp: 0.52, colorMix: 0.58, wave: 0.55 },
    thinking: { mode: 'THINKING', intensity: 'HIGH', amp: 0.74, colorMix: 0.82, wave: 0.72 },
    speaking: { mode: 'SPEAKING', intensity: 'HIGH', amp: 1.00, colorMix: 1.00, wave: 1.00 }
  }
  var current = 'idle'
  var live = { amp: 0.05, colorMix: 0, wave: 0.1 }
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  var ASSEMBLE_MS = reduced ? 900 : 5200
  var assembleStart = performance.now()
  var assembled = false
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
    engineLabel.textContent = 'ENGINE WEBGL2 \u00b7 BLOOM ' + (engine.hdr ? 'HDR' : 'ON')
    particleCountEl.textContent = engine.count.toLocaleString()
  } else {
    glCanvas.style.display = 'none'
    fbCanvas.style.display = 'block'
    engine = window.AstaFallback2D ? window.AstaFallback2D.create(fbCanvas) : null
    engineLabel.textContent = 'ENGINE CANVAS2D \u00b7 FALLBACK'
      + (engineError ? ' \u00b7 ' + engineError.slice(0, 70) : '')
    engineLed.classList.add('warn')
    particleCountEl.textContent = '\u2014'
  }

  /* ---------- waveform bars ---------- */
  var bars = []
  for (var b = 0; b < 38; b++) {
    var bar = document.createElement('i')
    waveform.appendChild(bar)
    bars.push(bar)
  }

  /* ---------- audio ---------- */
  var audio = { ctx: null, master: null, nodes: [], muted: false, level: 0.16 }

  function startDrone () {
    if (audio.muted) return
    try {
      if (!audio.ctx) {
        var AC = window.AudioContext || window.webkitAudioContext
        if (!AC) return
        audio.ctx = new AC()
      }
      if (audio.ctx.state === 'suspended') audio.ctx.resume()
      if (audio.nodes.length) return

      var ctx = audio.ctx
      var now = ctx.currentTime

      var master = ctx.createGain()
      master.gain.setValueAtTime(0.0001, now)
      master.gain.exponentialRampToValueAtTime(audio.level, now + 1.4)
      master.connect(ctx.destination)

      var lp = ctx.createBiquadFilter()
      lp.type = 'lowpass'
      lp.frequency.value = 380
      lp.Q.value = 0.7
      lp.connect(master)

      var lfo = ctx.createOscillator()
      var lfoGain = ctx.createGain()
      lfo.frequency.value = 0.13
      lfoGain.gain.value = 140
      lfo.connect(lfoGain)
      lfoGain.connect(lp.frequency)
      lfo.start()

      var nodes = [master, lp, lfo, lfoGain]
      var specs = [[55, 'sine', 0.4], [82.5, 'sine', 0.4], [110.55, 'triangle', 0.18]]
      for (var i = 0; i < specs.length; i++) {
        var osc = ctx.createOscillator()
        var g = ctx.createGain()
        osc.type = specs[i][1]
        osc.frequency.value = specs[i][0]
        g.gain.value = specs[i][2]
        osc.connect(g)
        g.connect(lp)
        osc.start()
        nodes.push(osc, g)
      }

      var len = Math.floor(ctx.sampleRate * 2)
      var buf = ctx.createBuffer(1, len, ctx.sampleRate)
      var chan = buf.getChannelData(0)
      for (var n = 0; n < len; n++) chan[n] = (Math.random() * 2 - 1) * 0.6
      var noise = ctx.createBufferSource()
      noise.buffer = buf
      noise.loop = true
      var bp = ctx.createBiquadFilter()
      bp.type = 'bandpass'
      bp.frequency.value = 600
      bp.Q.value = 0.5
      var ng = ctx.createGain()
      ng.gain.value = 0.15
      noise.connect(bp)
      bp.connect(ng)
      ng.connect(master)
      noise.start()
      nodes.push(noise, bp, ng)

      audio.master = master
      audio.nodes = nodes
    } catch (e) { /* audio unavailable */ }
  }

  function stopDrone (immediate) {
    if (!audio.ctx || !audio.nodes.length) return
    try {
      if (audio.master) {
        var now = audio.ctx.currentTime
        audio.master.gain.cancelScheduledValues(now)
        audio.master.gain.setValueAtTime(Math.max(0.0001, audio.master.gain.value), now)
        audio.master.gain.exponentialRampToValueAtTime(0.0001, now + (immediate ? 0.05 : 0.7))
      }
    } catch (e) { /* ignore */ }
    var nodes = audio.nodes
    audio.nodes = []
    audio.master = null
    setTimeout(function () {
      for (var i = 0; i < nodes.length; i++) {
        try { if (nodes[i].stop) nodes[i].stop() } catch (e) { /* ignore */ }
        try { nodes[i].disconnect() } catch (e) { /* ignore */ }
      }
    }, immediate ? 80 : 900)
  }

  function toggleMute () {
    audio.muted = !audio.muted
    if (audio.muted) {
      stopDrone(false)
      muteBtn.textContent = 'AMBIENCE OFF'
    } else {
      muteBtn.textContent = 'AMBIENCE ON'
      startDrone()
    }
  }
  if (muteBtn) muteBtn.addEventListener('click', toggleMute)

  function speakReady () {
    if (audio.muted) return
    try {
      if (!window.speechSynthesis) return
      var u = new SpeechSynthesisUtterance('Ready')
      u.rate = 0.85
      u.pitch = 0.55
      u.volume = 0.9
      window.speechSynthesis.speak(u)
    } catch (e) { /* ignore */ }
  }

  startDrone()
  var unlock = function () {
    if (audio.ctx && audio.ctx.state === 'suspended') audio.ctx.resume()
    if (!audio.muted && !audio.nodes.length) startDrone()
    window.removeEventListener('pointerdown', unlock)
    window.removeEventListener('keydown', unlock)
  }
  window.addEventListener('pointerdown', unlock)
  window.addEventListener('keydown', unlock)

  /* ---------- state handling ---------- */
  function setState (name) {
    if (!STATES[name]) return
    current = name
    statusMode.textContent = STATES[name].mode
    statusIntensity.textContent = STATES[name].intensity
    for (var i = 0; i < dockButtons.length; i++) {
      var isActive = dockButtons[i].getAttribute('data-state') === name
      if (isActive) dockButtons[i].classList.add('active')
      else dockButtons[i].classList.remove('active')
    }
    if (assembled) renderBigStatus()
  }

  function renderBigStatus () {
    readoutLabel.textContent = 'STATUS:'
    readoutPct.textContent = STATES[current].mode
    readoutPct.className = 'pct status-big' + (current === 'speaking' ? ' pulse' : '')
  }

  function completeAssembly () {
    assembled = true
    progress = 1
    setState('speaking')
    renderBigStatus()
    speakReady()
  }

  function forceComplete () {
    if (!assembled) completeAssembly()
  }

  function reassemble () {
    assembled = false
    progress = 0
    assembleStart = performance.now()
    readoutPct.className = 'pct'
    readoutLabel.textContent = 'ASSEMBLING...'
    readoutPct.textContent = '0%'
    setState('idle')
  }
  if (reassembleBtn) reassembleBtn.addEventListener('click', reassemble)

  for (var d = 0; d < dockButtons.length; d++) {
    dockButtons[d].addEventListener('click', function () {
      forceComplete()
      setState(this.getAttribute('data-state'))
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
    if (e.metaKey || e.ctrlKey || e.altKey) return
    var k = (e.key || '').toLowerCase()
    var map = { '1': 'idle', '2': 'listening', '3': 'thinking', '4': 'speaking' }
    if (map[k]) {
      e.preventDefault()
      forceComplete()
      setState(map[k])
    } else if (k === 'r') {
      e.preventDefault()
      reassemble()
    } else if (k === 'f') {
      e.preventDefault()
      toggleFullscreen()
    } else if (k === 'm') {
      e.preventDefault()
      toggleMute()
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

    if (!assembled) {
      progress = Math.min(1, (now - assembleStart) / ASSEMBLE_MS)
      if (progress >= 1) completeAssembly()
    }

    var target = STATES[current]
    var wantAmp = assembled ? target.amp : (0.18 + 0.42 * progress)
    var wantMix = assembled ? target.colorMix : (0.15 + 0.55 * progress)
    var wantWave = assembled ? target.wave : (0.12 + 0.30 * progress)
    var k = Math.min(1, dt * 3.0)
    live.amp += (wantAmp - live.amp) * k
    live.colorMix += (wantMix - live.colorMix) * k
    live.wave += (wantWave - live.wave) * k

    /* gentle sway instead of a full spin, so the face stays toward the viewer */
    swayT += dt * (0.17 + 0.10 * live.amp)
    rot = Math.sin(swayT) * 0.17

    if (engine && engine.render) {
      engine.render({
        time: now / 1000,
        progress: progress,
        amp: live.amp,
        colorMix: live.colorMix,
        rot: rot,
        bloom: 1.15 + 0.45 * live.amp
      })
    }

    if (!assembled) readoutPct.textContent = Math.round(progress * 100) + '%'

    for (var i = 0; i < bars.length; i++) {
      var center = 1 - Math.abs(i / (bars.length - 1) - 0.5) * 2
      var wob = 0.35 + 0.65 * Math.abs(Math.sin(now / 1000 * (2.1 + i * 0.17) + i))
      bars[i].style.height = (3 + 25 * live.wave * wob * (0.35 + 0.65 * center)).toFixed(1) + 'px'
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

  setState('idle')
})()
