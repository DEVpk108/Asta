/* ASTA Core avatar: a 3D-projected particle orb drawn with Canvas 2D.
   Each runtime mode has its own colour and motion; speech audio drives the
   speaking pulse. Rendering is driven by reactive_avatar.js, which owns
   scheduling, frame budgets, reduced motion and pause. */
'use strict'

;(function (global) {
  var TAU = Math.PI * 2

  // a = front colour, b = back/edge colour, glow = halo, core = warm centre strength.
  var MODES = {
    idle:      { r: 0.300, rot: 0.16, noise: 0.030, listen: 0, swirl: 0,   speak: 0, beat: 0,   core: 0.22, bright: 0.80, a: [150, 205, 255], b: [60, 110, 220],  glow: [70, 130, 230] },
    listening: { r: 0.318, rot: 0.22, noise: 0.024, listen: 1, swirl: 0,   speak: 0, beat: 0,   core: 0.30, bright: 1.00, a: [170, 245, 255], b: [40, 170, 210],  glow: [50, 190, 220] },
    thinking:  { r: 0.290, rot: 0.45, noise: 0.045, listen: 0, swirl: 1,   speak: 0, beat: 0,   core: 0.95, bright: 0.92, a: [205, 175, 255], b: [110, 80, 220],  glow: [130, 90, 230] },
    speaking:  { r: 0.310, rot: 0.24, noise: 0.022, listen: 0, swirl: 0,   speak: 1, beat: 0,   core: 0.55, bright: 1.00, a: [255, 240, 215], b: [90, 160, 240],  glow: [90, 170, 240] },
    executing: { r: 0.296, rot: 0.55, noise: 0.035, listen: 0, swirl: 0.6, speak: 0, beat: 0,   core: 0.60, bright: 0.95, a: [175, 245, 215], b: [40, 150, 130],  glow: [60, 180, 150] },
    approval:  { r: 0.305, rot: 0.12, noise: 0.025, listen: 0, swirl: 0,   speak: 0, beat: 1,   core: 0.70, bright: 0.95, a: [255, 215, 160], b: [210, 120, 50],  glow: [222, 146, 85] },
    error:     { r: 0.292, rot: 0.06, noise: 0.060, listen: 0, swirl: 0,   speak: 0, beat: 0,   core: 0.15, bright: 0.70, a: [255, 175, 165], b: [190, 70, 70],   glow: [233, 115, 102] },
    offline:   { r: 0.290, rot: 0.00, noise: 0.020, listen: 0, swirl: 0,   speak: 0, beat: 0,   core: 0.05, bright: 0.40, a: [165, 175, 190], b: [70, 80, 95],    glow: [90, 100, 120] }
  }
  var KEYS = Object.keys(MODES.idle)

  function clamp (v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  function lerp (a, b, k) { return a + (b - a) * k }
  function copy (p) {
    var out = {}
    KEYS.forEach(function (k) { out[k] = Array.isArray(p[k]) ? p[k].slice() : p[k] })
    return out
  }
  function rgba (c, a) { return 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + a + ')' }

  function geometry (n) {
    var pts = [], dust = [], golden = Math.PI * (3 - Math.sqrt(5)), seed = 7
    function rnd () { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
    for (var i = 0; i < n; i++) {
      var y = 1 - (i / (n - 1)) * 2, rr = Math.sqrt(1 - y * y), th = golden * i
      pts.push({ x: Math.cos(th) * rr, y: y, z: Math.sin(th) * rr, s: 0.7 + rnd() * 1.1, b: 0.55 + rnd() * 0.45, ph: rnd() * TAU })
    }
    var dustCount = Math.round(n / 17)
    for (var j = 0; j < dustCount; j++) dust.push({ a: rnd() * TAU, r: 1.16 + rnd() * 0.22, y: (rnd() - 0.5) * 0.5, sp: 0.2 + rnd() * 0.5, s: 0.5 + rnd() * 0.9 })
    return { pts: pts, dust: dust }
  }

  function create (canvas, options) {
    options = options || {}
    var ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas 2D not available')
    var count = clamp(Math.round(Number(options.maxParticles) || 2400), 400, 4000)
    var geo = geometry(count)
    var BANDS = 10
    var px = new Float32Array(count), py = new Float32Array(count), pr = new Float32Array(count), pb = new Uint8Array(count)
    var dpr = 1, W = 1, H = 1
    var cur = copy(MODES.offline), last = null
    var angle = 0, swirlAngle = 0, lastTime = 0

    function resize () {
      var nextDpr = Math.min(global.devicePixelRatio || 1, Number(options.maxDpr) || 2)
      var nextW = Math.max(1, canvas.clientWidth || 1)
      var nextH = Math.max(1, canvas.clientHeight || 1)
      nextDpr *= Math.min(1, (Number(options.maxDimension) || 2048) / Math.max(nextW * nextDpr, nextH * nextDpr))
      if (W === nextW && H === nextH && dpr === nextDpr && canvas.width > 1) return
      dpr = nextDpr; W = nextW; H = nextH
      canvas.width = Math.max(1, Math.round(W * dpr))
      canvas.height = Math.max(1, Math.round(H * dpr))
    }

    function advance (s) {
      var target = MODES[s.mode] || MODES.idle
      var t = Number(s.time) || 0
      var dt = clamp(t - lastTime, 0, 0.15)
      lastTime = t
      if (s.still || !last) {
        cur = copy(target)
      } else {
        var k = 1 - Math.exp(-dt * 4.5)
        KEYS.forEach(function (key) {
          if (Array.isArray(target[key])) cur[key] = cur[key].map(function (v, i) { return lerp(v, target[key][i], k) })
          else cur[key] = lerp(cur[key], target[key], k)
        })
      }
      angle += cur.rot * dt
      swirlAngle += cur.swirl * dt * 1.4
      return t
    }

    function render (s) {
      s = s || {}
      resize()
      var t = advance(s)
      last = s
      var p = cur
      var audio = clamp(Number(s.audio) || 0, 0, 1)
      var S = Math.min(W, H), cx = W / 2, cy = H / 2
      var beat = p.beat * (0.5 + 0.5 * Math.sin(t * 2.2))
      var small = S < 120
      var R = S * p.r * (small ? 1.35 : 1) * (1 + 0.012 * Math.sin(t * 1.25) + 0.03 * beat + 0.05 * p.speak * audio)

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.globalCompositeOperation = 'source-over'
      ctx.clearRect(0, 0, W, H)

      var g = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, Math.max(W, H) * 0.55)
      g.addColorStop(0, rgba(p.glow, (0.20 * p.bright).toFixed(3)))
      g.addColorStop(0.5, rgba(p.glow, (0.06 * p.bright).toFixed(3)))
      g.addColorStop(1, rgba(p.glow, 0))
      ctx.fillStyle = g
      ctx.fillRect(0, 0, W, H)
      ctx.globalCompositeOperation = 'lighter'

      var cp = p.core * (0.82 + 0.18 * Math.sin(t * 3.1)) + p.speak * audio * 0.3 + beat * 0.25
      g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.78)
      g.addColorStop(0, 'rgba(255,214,150,' + (0.55 * cp).toFixed(3) + ')')
      g.addColorStop(0.45, 'rgba(255,170,110,' + (0.18 * cp).toFixed(3) + ')')
      g.addColorStop(1, 'rgba(255,170,110,0)')
      ctx.fillStyle = g
      ctx.fillRect(0, 0, W, H)

      var i, q
      if (!small) {
        for (i = 0; i < geo.dust.length; i++) {
          q = geo.dust[i]
          var da = q.a + t * q.sp * (0.25 + p.swirl * 1.2)
          var dx = Math.cos(da) * q.r, dz = Math.sin(da) * q.r, dy = q.y + 0.08 * Math.sin(da * 2)
          var dDepth = (dz / q.r + 1) / 2, dp = 1 / (1 - dz * 0.18)
          ctx.fillStyle = rgba(p.a, ((0.05 + 0.25 * dDepth) * p.bright).toFixed(3))
          ctx.beginPath()
          ctx.arc(cx + dx * R * dp, cy + (dy * Math.cos(0.35) - dz * 0.3) * R * dp, q.s * S / 420, 0, TAU)
          ctx.fill()
        }
      }

      // Project every particle, then draw them in a few depth bands: one
      // fillStyle and one fill() per band instead of one per particle.
      var tilt = 0.38, ct = Math.cos(tilt), st = Math.sin(tilt), unit = S / 300
      var stride = small ? 3 : 1
      var minR = small ? 0.6 : 0.35
      var n = 0
      for (i = 0; i < geo.pts.length; i += stride) {
        q = geo.pts[i]
        var a = angle + p.swirl * swirlAngle * Math.sin(q.y * Math.PI * 2.2) * 1.6
        var ca = Math.cos(a), sa = Math.sin(a)
        var X = q.x * ca + q.z * sa, Z = -q.x * sa + q.z * ca
        var Y2 = q.y * ct - Z * st, Z2 = q.y * st + Z * ct
        var d = 1 + p.noise * Math.sin(q.x * 4 + t * 1.1 + q.ph * 0.3) * Math.sin(q.y * 5 - t * 0.8)
        if (p.listen > 0.01) {
          var dist = Math.acos(clamp(Z2, -1, 1))
          d += p.listen * 0.06 * Math.max(0, Math.sin(dist * 9 - t * 6.5)) * (1 - dist / Math.PI)
        }
        if (p.speak > 0.01) d += p.speak * audio * (0.07 + 0.06 * Math.max(0, Math.sin(Y2 * 7 - t * 9)))
        var depth = (Z2 + 1) / 2, persp = 1 / (1 - Z2 * d * 0.28)
        px[n] = cx + X * d * R * persp
        py[n] = cy + Y2 * d * R * persp
        pr[n] = Math.max(q.s * unit * (0.55 + 0.85 * depth) * (0.75 + 0.25 * q.b) * (small ? 1.3 : 1) * 0.6, minR)
        pb[n] = Math.min(BANDS - 1, (depth * BANDS) | 0)
        n++
      }
      var boost = small ? 1.5 : 1
      for (var band = 0; band < BANDS; band++) {
        var bd = (band + 0.5) / BANDS, bz = bd * 2 - 1
        var rim = Math.pow(1 - Math.abs(bz), 3) * 0.35
        var al = Math.min(1, (0.10 + 0.95 * bd * bd + rim) * p.bright * 0.8 * boost)
        ctx.fillStyle = rgba([lerp(p.b[0], p.a[0], bd), lerp(p.b[1], p.a[1], bd), lerp(p.b[2], p.a[2], bd)], al.toFixed(3))
        ctx.beginPath()
        for (var j = 0; j < n; j++) {
          if (pb[j] !== band) continue
          var r = pr[j]
          if (r < 0.9) ctx.rect(px[j] - r, py[j] - r, r * 2, r * 2)
          else { ctx.moveTo(px[j] + r, py[j]); ctx.arc(px[j], py[j], r, 0, TAU) }
        }
        ctx.fill()
      }
      ctx.globalCompositeOperation = 'source-over'
    }

    function dispose () {
      geo = { pts: [], dust: [] }
      canvas.width = 1; canvas.height = 1
    }

    resize()
    return { ok: true, count: count, resize: resize, render: render, dispose: dispose }
  }

  global.AstaCoreAvatar = { create: create, MODES: Object.keys(MODES) }
})(typeof window !== 'undefined' ? window : globalThis)
