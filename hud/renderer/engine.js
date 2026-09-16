/* ASTA - particle humanoid engine (geometry).
   Proportions modelled on the reference photo: human skull with jaw taper
   and ears, short thick neck, trapezius sloping ~30deg into deltoid caps,
   hanging upper arms, narrow ribcage, bright silhouette rim over a sparse
   dark interior, amber throat cords and sternum trident. */
'use strict'

;(function (global) {
  var TOP_Y = 6.02
  var BOTTOM_Y = 0.85
  var ORB_Y = -0.85
  var FACE_Y = 4.58
  var EAR_Y = 4.56
  var CORE_Y = 1.15
  var HEAD_HW = 0.76

  /* cross sections: y, half-width (x), half-depth (z), z-centre offset.
     head 6.02..3.92 (w 1.52 : h 2.10 = human 0.72 ratio, depth 0.9 of height)
     neck 3.92..3.18 (short, flares into the trapezius)
     ribcage 3.18..0.85 (max width 2.88 = 1.9 head widths) */
  var SECTIONS = [
    { y: 6.02, rx: 0.11, rz: 0.13, zc: -0.02 },
    { y: 5.92, rx: 0.27, rz: 0.33, zc: -0.03 },
    { y: 5.78, rx: 0.43, rz: 0.53, zc: -0.05 },
    { y: 5.62, rx: 0.55, rz: 0.67, zc: -0.07 },
    { y: 5.45, rx: 0.64, rz: 0.77, zc: -0.08 },
    { y: 5.28, rx: 0.70, rz: 0.84, zc: -0.09 },
    { y: 5.10, rx: 0.74, rz: 0.89, zc: -0.09 },
    { y: 4.92, rx: 0.76, rz: 0.92, zc: -0.09 },
    { y: 4.75, rx: 0.755, rz: 0.94, zc: -0.08 },
    { y: 4.58, rx: 0.74, rz: 0.93, zc: -0.06 },
    { y: 4.42, rx: 0.70, rz: 0.90, zc: -0.04 },
    { y: 4.26, rx: 0.64, rz: 0.85, zc: -0.01 },
    { y: 4.12, rx: 0.55, rz: 0.78, zc: 0.01 },
    { y: 4.00, rx: 0.44, rz: 0.66, zc: 0.02 },
    { y: 3.92, rx: 0.34, rz: 0.53, zc: 0.02 },
    { y: 3.80, rx: 0.44, rz: 0.46, zc: 0.03 },
    { y: 3.60, rx: 0.43, rz: 0.45, zc: 0.03 },
    { y: 3.40, rx: 0.49, rz: 0.51, zc: 0.03 },
    { y: 3.18, rx: 0.60, rz: 0.58, zc: 0.02 },
    { y: 3.05, rx: 0.74, rz: 0.62, zc: 0.01 },
    { y: 2.90, rx: 0.92, rz: 0.67, zc: 0.00 },
    { y: 2.70, rx: 1.10, rz: 0.72, zc: 0.00 },
    { y: 2.50, rx: 1.24, rz: 0.77, zc: 0.00 },
    { y: 2.30, rx: 1.33, rz: 0.82, zc: 0.00 },
    { y: 2.10, rx: 1.38, rz: 0.86, zc: 0.00 },
    { y: 1.90, rx: 1.41, rz: 0.89, zc: 0.00 },
    { y: 1.65, rx: 1.43, rz: 0.91, zc: 0.00 },
    { y: 1.40, rx: 1.44, rz: 0.92, zc: 0.00 },
    { y: 1.15, rx: 1.43, rz: 0.92, zc: 0.00 },
    { y: 0.95, rx: 1.41, rz: 0.91, zc: 0.00 },
    { y: 0.85, rx: 1.40, rz: 0.90, zc: 0.00 }
  ]

  function sectionAt (y) {
    var first = SECTIONS[0]
    var last = SECTIONS[SECTIONS.length - 1]
    if (y >= first.y) return { rx: first.rx, rz: first.rz, zc: first.zc }
    if (y <= last.y) return { rx: last.rx, rz: last.rz, zc: last.zc }
    for (var i = 0; i < SECTIONS.length - 1; i++) {
      var a = SECTIONS[i]
      var b = SECTIONS[i + 1]
      if (y <= a.y && y >= b.y) {
        var t = (a.y - y) / (a.y - b.y)
        return {
          rx: a.rx + (b.rx - a.rx) * t,
          rz: a.rz + (b.rz - a.rz) * t,
          zc: a.zc + (b.zc - a.zc) * t
        }
      }
    }
    return { rx: 0, rz: 0, zc: 0 }
  }

  var CYAN = [0.05, 0.78, 1.00]
  var CYAN_RIM = [0.62, 0.97, 1.00]
  var CYAN_DEEP = [0.03, 0.30, 0.88]
  var AMBER = [1.00, 0.52, 0.02]
  var AMBER_HOT = [1.00, 0.84, 0.46]

  function mix3 (a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
  }

  /* trapezius ridge: neck base out to the acromion, dropping 1.0 over 1.7 (~30deg) */
  var TRAP_PTS = [[0.30, 3.30, 0.02], [0.70, 3.24, 0.04], [1.15, 3.10, 0.05], [1.55, 2.88, 0.05], [1.85, 2.60, 0.04], [2.00, 2.30, 0.02]]
  var TRAP_R = [0.66, 0.64, 0.60, 0.56, 0.52, 0.48]
  /* upper arm hanging from the deltoid: outer edge 2.47 = 3.25 head widths across */
  var ARM_PTS = [[2.02, 1.92, 0.02], [2.07, 1.45, 0.02], [2.06, 1.00, 0.01], [2.01, 0.55, 0.00]]
  var ARM_R = [0.42, 0.40, 0.38, 0.36]

  /* amber: trachea cords, sternum trunk, clavicle branches, inner cords */
  var NEURAL_PATHS = [
    { m: false, pts: [[0, 3.98, 0.15], [0, 3.62, 0.19], [0, 3.32, 0.21], [0, 3.14, 0.19], [0, 2.80, 0.14], [0, 2.40, 0.10], [0, 1.95, 0.06], [0, 1.50, 0.04], [0, 1.18, 0.02]] },
    { m: true, pts: [[0.07, 3.95, 0.17], [0.11, 3.60, 0.20], [0.13, 3.30, 0.21], [0.12, 3.16, 0.19]] },
    { m: true, pts: [[0.05, 3.14, 0.20], [0.42, 2.96, 0.20], [0.76, 2.74, 0.18], [1.02, 2.48, 0.14], [1.14, 2.18, 0.10]] },
    { m: true, pts: [[0.04, 2.92, 0.18], [0.24, 2.40, 0.16], [0.33, 1.90, 0.12], [0.36, 1.48, 0.08]] }
  ]

  function makePath (pts) {
    var seg = []
    var total = 0
    for (var i = 0; i < pts.length - 1; i++) {
      var dx = pts[i + 1][0] - pts[i][0]
      var dy = pts[i + 1][1] - pts[i][1]
      var dz = pts[i + 1][2] - pts[i][2]
      var l = Math.sqrt(dx * dx + dy * dy + dz * dz)
      seg.push(l)
      total += l
    }
    return { pts: pts, seg: seg, total: total }
  }

  function samplePath (path, t) {
    var d = t * path.total
    var i = 0
    while (i < path.seg.length - 1 && d > path.seg[i]) {
      d -= path.seg[i]
      i++
    }
    var u = path.seg[i] > 0 ? d / path.seg[i] : 0
    if (u > 1) u = 1
    var a = path.pts[i]
    var b = path.pts[i + 1]
    return {
      p: [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u],
      t: [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
      i: i,
      u: u
    }
  }

  function lerpArr (arr, i, u) {
    var a = arr[Math.min(i, arr.length - 1)]
    var b = arr[Math.min(i + 1, arr.length - 1)]
    return a + (b - a) * u
  }

  function buildParticles () {
    var source = []
    var target = []
    var color = []
    var seed = []
    var delay = []
    var kind = []
    var ring = []
    var scale = []
    var flow = []
    var count = 0

    function push (s, t, c, k, rIndex, d, sc, fl) {
      source.push(s[0], s[1], s[2])
      target.push(t[0], t[1], t[2])
      color.push(c[0], c[1], c[2])
      seed.push(Math.random())
      delay.push(d)
      kind.push(k)
      ring.push(rIndex)
      scale.push(sc)
      flow.push(fl || 0)
      count++
    }

    function dly (y) {
      var f = (y - BOTTOM_Y) / (TOP_Y - BOTTOM_Y)
      if (f < 0) f = 0
      if (f > 1) f = 1
      return 0.04 + 0.5 * f + Math.random() * 0.08
    }

    function orbSource (spread) {
      var a = Math.random() * Math.PI * 2
      var p = Math.acos(2 * Math.random() - 1)
      var r = Math.pow(Math.random(), 0.55) * spread
      return [Math.sin(p) * Math.cos(a) * r, ORB_Y + Math.cos(p) * r * 0.55, Math.sin(p) * Math.sin(a) * r]
    }

    /* bias samples toward the silhouette edge (theta near 0 or PI) */
    function rimAngle () {
      if (Math.random() < 0.54) {
        var base = Math.random() < 0.5 ? 0 : Math.PI
        return base + (Math.random() + Math.random() - 1) * 0.36
      }
      return Math.random() * Math.PI * 2
    }

    /* bright crisp edge over a dim sparse interior - the reference's key cue */
    function rimLook (rim) {
      if (rim > 0.84) return { c: mix3(CYAN, CYAN_RIM, 0.40 + Math.random() * 0.45), s: 1.28 + Math.random() * 0.32 }
      if (rim > 0.50) return { c: mix3(CYAN, CYAN_RIM, Math.random() * 0.18), s: 0.92 + Math.random() * 0.22 }
      return { c: mix3(CYAN_DEEP, CYAN, 0.20 + Math.random() * 0.35), s: 0.60 + Math.random() * 0.22 }
    }

    /* dissolve into dust above the UI dock instead of a hard cut */
    function fade (y) {
      if (y >= 1.5) return 1
      var f = (y - 0.72) / 0.78
      if (f < 0) f = 0
      if (f > 1) f = 1
      return f * f
    }

    function shell (n, yLo, yHi) {
      for (var i = 0; i < n; i++) {
        var y = yLo + Math.random() * (yHi - yLo)
        var s = sectionAt(y)
        if (s.rx < 0.05) continue
        var th = rimAngle()
        var cosA = Math.cos(th)
        var sinA = Math.sin(th)
        var rim = Math.abs(cosA)
        if (rim < 0.46 && Math.random() < 0.38) continue
        var j = 0.962 + Math.random() * 0.072
        var x = cosA * s.rx * j
        var z = sinA * s.rz * j + s.zc
        var look = rimLook(rim)
        var c = look.c
        var sc = look.s
        var k = 0
        if (y > 4.02 && y < 5.16 && sinA > 0.22) {
          var fy = (y - FACE_Y) / 0.54
          var fx = x / (HEAD_HW * 0.66)
          var dd = Math.sqrt(fx * fx * 0.92 + fy * fy)
          if (dd < 1.0) {
            var w = (1.0 - dd) * Math.min(1, (sinA - 0.22) / 0.4)
            if (w > 0.06) {
              k = 1
              c = mix3(AMBER, AMBER_HOT, Math.min(1, w * 1.5))
              sc = 0.78 + w * 0.8
            }
          }
        }
        var fd = fade(y)
        if (Math.random() > fd) continue
        sc *= 0.55 + 0.45 * fd
        push(orbSource(1.5), [x, y, z], c, k, Math.floor((TOP_Y - y) * 7), dly(y), sc, 0)
      }
    }

    /* drop: shift the section so the path acts as the upper surface.
       half: keep only the upper half, so the bright rim traces the
       shoulder line and the mass hangs down into the ribcage. */
    function tube (n, pts, radii, depth, side, drop, half) {
      var path = makePath(pts)
      for (var i = 0; i < n; i++) {
        var sp = samplePath(path, Math.random())
        var r = lerpArr(radii, sp.i, sp.u)
        var tl = Math.sqrt(sp.t[0] * sp.t[0] + sp.t[1] * sp.t[1]) || 1
        var ux = -sp.t[1] / tl
        var uy = sp.t[0] / tl
        var th
        if (half) {
          th = Math.random() < 0.58 ? (Math.random() + Math.random() - 1) * 0.36 : (Math.random() - 0.5) * Math.PI
        } else {
          th = rimAngle()
        }
        var cosA = Math.cos(th)
        var sinA = Math.sin(th)
        var rim = Math.abs(cosA)
        if (rim < 0.46 && Math.random() < 0.34) continue
        var j = 0.94 + Math.random() * 0.10
        var dp = drop || 0
        var x = sp.p[0] - ux * r * dp + ux * r * cosA * j
        var y = sp.p[1] - uy * r * dp + uy * r * cosA * j
        var z = sp.p[2] + sinA * r * depth * j
        var look = rimLook(rim)
        var fd = fade(y)
        if (Math.random() > fd) continue
        push(orbSource(1.5), [side * x, y, z], look.c, 0, Math.floor((TOP_Y - y) * 7), dly(y), look.s * (0.55 + 0.45 * fd), 0)
      }
    }

    function blob (n, cx, cy, cz, rx, ry, rz, k, ca, cb, scBase, biasRim) {
      for (var i = 0; i < n; i++) {
        var a = Math.random() * Math.PI * 2
        var p = Math.acos(2 * Math.random() - 1)
        var rr = biasRim ? (0.82 + Math.random() * 0.18) : Math.pow(Math.random(), 0.42)
        var sx = Math.sin(p) * Math.cos(a)
        var sy = Math.cos(p)
        var sz = Math.sin(p) * Math.sin(a)
        var x = cx + sx * rx * rr
        var y = cy + sy * ry * rr
        var z = cz + sz * rz * rr
        if (biasRim) {
          var look = rimLook(Math.abs(sx))
          push(orbSource(1.4), [x, y, z], look.c, k, Math.floor((TOP_Y - y) * 7), dly(y), look.s * scBase, 0)
        } else {
          push(orbSource(1.4), [x, y, z], mix3(cb, ca, 1 - Math.min(1, rr)), k, Math.floor((TOP_Y - y) * 7), dly(y), scBase * (0.7 + Math.random() * 0.55), 0)
        }
      }
    }

    function circularVisualizer (n) {
      var cx = 0
      var cy = 3.50
      var radius = 1.82
      for (var i = 0; i < n; i++) {
        var a = Math.random() * Math.PI * 2
        var mode = Math.random()
        var rr
        if (mode < 0.38) {
          rr = radius * (0.94 + Math.random() * 0.08)
        } else if (mode < 0.76) {
          rr = radius * (0.34 + Math.random() * 0.52)
        } else {
          var spoke = Math.floor(Math.random() * 32) / 32
          a = spoke * Math.PI * 2 + (Math.random() - 0.5) * 0.035
          rr = radius * Math.pow(Math.random(), 0.72)
        }
        var x = cx + Math.cos(a) * rr
        var y = cy + Math.sin(a) * rr
        var z = (Math.random() - 0.5) * (0.12 + 0.12 * (rr / radius))
        var edge = Math.min(1, Math.abs(rr / radius - 1) * 7 + 0.18)
        var c = mix3(CYAN, CYAN_RIM, edge * 0.82)
        var sc = 0.62 + edge * 0.66 + Math.random() * 0.24
        push(orbSource(1.5), [x, y, z], c, 0, Math.floor(rr * 12), dly(y), sc, a)
      }
    }

    function ear (n, side) {
      for (var i = 0; i < n; i++) {
        var a = -0.55 + Math.random() * 3.9
        var rr = 0.74 + Math.random() * 0.26
        var y = EAR_Y + Math.sin(a) * 0.23 * rr
        var z = -0.14 + Math.cos(a) * 0.16 * rr
        var s = sectionAt(y)
        var x = s.rx * 0.94 + 0.10 + Math.random() * 0.06
        push(orbSource(1.3), [side * x, y, z], mix3(CYAN, CYAN_RIM, 0.5 + Math.random() * 0.4), 0, Math.floor((TOP_Y - y) * 7), dly(y), 1.15 + Math.random() * 0.35, 0)
      }
    }

    function neural (n) {
      var built = []
      for (var i = 0; i < NEURAL_PATHS.length; i++) {
        var def = NEURAL_PATHS[i]
        built.push({ path: makePath(def.pts), side: 1, idx: i })
        if (def.m) built.push({ path: makePath(def.pts), side: -1, idx: i })
      }
      for (var q = 0; q < n; q++) {
        var b = built[Math.floor(Math.random() * built.length)]
        var tN = Math.random()
        var sp = samplePath(b.path, tN)
        var spread = 0.03 + 0.035 * Math.random()
        var x = b.side * (sp.p[0] + (Math.random() - 0.5) * spread)
        var y = sp.p[1] + (Math.random() - 0.5) * spread
        var z = sp.p[2] + (Math.random() - 0.5) * spread
        if (y < 1.45) {
          var f = (y - 1.05) / 0.4
          if (f < 0) f = 0
          if (Math.random() > f) continue
        }
        push(orbSource(1.2), [x, y, z], mix3(AMBER, AMBER_HOT, 0.25 + Math.random() * 0.6), 2, 30 + tN * 18, dly(y) * 0.8, 0.85 + Math.random() * 0.45, tN + b.idx * 0.37)
      }
    }

    function dust (n) {
      for (var i = 0; i < n; i++) {
        var y = 1.0 + Math.random() * 4.8
        var a = Math.random() * Math.PI * 2
        var rr = 1.1 + Math.pow(Math.random(), 0.7) * 2.3
        push(orbSource(2.0), [Math.cos(a) * rr, y, Math.sin(a) * rr * 0.6], mix3(CYAN_DEEP, CYAN, Math.random()), 4, 10 + Math.random() * 30, Math.random() * 0.5, 0.5 + Math.random() * 0.4, 0)
      }
    }

    circularVisualizer(22000)

    blob(2400, 0, 3.50, 0.12, 0.38, 0.38, 0.20, 1, AMBER_HOT, AMBER, 1.0, false)
    blob(560, 0, 3.50, 0.18, 0.14, 0.14, 0.10, 3, AMBER_HOT, AMBER, 1.05, false)
    dust(1200)

    return {
      count: count,
      source: new Float32Array(source),
      target: new Float32Array(target),
      color: new Float32Array(color),
      seed: new Float32Array(seed),
      delay: new Float32Array(delay),
      kind: new Float32Array(kind),
      ring: new Float32Array(ring),
      scale: new Float32Array(scale),
      flow: new Float32Array(flow)
    }
  }
  /* ------------------------------- shaders ------------------------------- */
  var VS_POINTS = [
    '#version 300 es',
    'precision highp float;',
    'layout(location=0) in vec3 aSource;',
    'layout(location=1) in vec3 aTarget;',
    'layout(location=2) in vec3 aColor;',
    'layout(location=3) in float aSeed;',
    'layout(location=4) in float aDelay;',
    'layout(location=5) in float aKind;',
    'layout(location=6) in float aRing;',
    'layout(location=7) in float aScale;',
    'layout(location=8) in float aFlow;',
    'uniform mat4 uProj; uniform mat4 uView;',
    'uniform float uTime; uniform float uProgress; uniform float uAmp;',
    'uniform float uColorMix; uniform float uRot; uniform float uWorldSize; uniform float uProjScale;',
    'out vec3 vColor; out float vAlpha;',
    'float hash(float n){ return fract(sin(n*127.1+0.371)*43758.5453); }',
    'void main(){',
    '  float t = clamp((uProgress - aDelay)/max(0.08, 1.0 - aDelay), 0.0, 1.0);',
    '  float eased = t*t*(3.0-2.0*t);',
    '  vec3 pos = mix(aSource, aTarget, eased);',
    '  float swirl = (1.0-eased)*0.9;',
    '  float a = uTime*1.6 + aSeed*6.2831 + aRing*0.2;',
    '  pos.x += sin(a)*swirl*0.35;',
    '  pos.y += cos(a*0.8)*swirl*0.22;',
    '  pos.z += sin(a*1.3)*swirl*0.30;',
    '  float alphaMul = 1.0; float energy = 1.0;',
    '  if (aKind < 1.5) {',
    /* per-particle drift + twinkle: this is what gives the loose dust look */
    '    float ripple = sin(uTime*1.05*(0.55+uAmp) + aRing*1.25 + aSeed*6.2831)*(0.012+0.055*uAmp)*eased;',
    '    vec2 radial = normalize(vec2(pos.x, pos.z) + vec2(0.0001));',
    '    pos.xz += radial*ripple;',
    '    pos.y += sin(uTime*0.75 + aSeed*41.3)*0.016*eased;',
    '    energy = 0.84 + 0.30*sin(uTime*2.2 + aSeed*24.7);',
    '  } else if (aKind < 2.5) {',
    '    float pulse = 0.5 + 0.5*sin(aFlow*9.0 - uTime*(2.2+1.8*uAmp));',
    '    energy = 0.55 + 1.25*pulse*pulse;',
    '  } else if (aKind > 3.5) {',
    '    float rise = mod(uTime*(0.22+0.35*uAmp) + aSeed*5.0, 2.4);',
    '    pos.y += rise*eased;',
    '    alphaMul = (1.0 - rise/2.4)*0.85;',
    '    energy = 0.80 + 0.40*sin(uTime*3.1 + aSeed*31.7);',
    '  } else {',
    '    energy = 1.0 + 0.35*sin(uTime*3.0 + aSeed*6.2831);',
    '  }',
    '  float cs = cos(uRot), sn = sin(uRot);',
    '  vec3 rp = vec3(pos.x*cs + pos.z*sn, pos.y, -pos.x*sn + pos.z*cs);',
    '  vec4 viewPos = uView*vec4(rp, 1.0);',
    '  gl_Position = uProj*viewPos;',
    '  float boost = 1.0;',
    '  if (aKind > 3.5) boost = 0.85;',
    '  else if (aKind > 2.5) boost = 3.0;',
    '  else if (aKind > 1.5) boost = 1.45;',
    '  else if (aKind > 0.5) boost = 1.15;',
    '  gl_PointSize = clamp(uWorldSize*boost*aScale*uProjScale/max(0.001, -viewPos.z), 1.0, 40.0);',
    '  vec3 c = mix(aColor, aColor*vec3(1.18,1.06,1.0), uColorMix*0.4);',
    '  vColor = c*energy*(0.78 + 0.50*uAmp);',
    '  vAlpha = eased*(0.40 + 0.60*hash(aSeed))*alphaMul*clamp(aScale*0.9, 0.45, 1.4);',
    '}'
  ].join('\n')

  var FS_POINTS = [
    '#version 300 es',
    'precision highp float;',
    'in vec3 vColor; in float vAlpha;',
    'out vec4 fragColor;',
    'void main(){',
    '  float r = length(gl_PointCoord - 0.5)*2.0;',
    '  if (r > 1.0) discard;',
    '  float f = pow(1.0 - r, 2.2);',
    '  float core = pow(1.0 - r, 8.0);',
    '  float alpha = f*vAlpha;',
    '  if (alpha < 0.004) discard;',
    '  fragColor = vec4(vColor*(f + core*1.6)*1.25, alpha);',
    '}'
  ].join('\n')

  var VS_QUAD = [
    '#version 300 es',
    'precision highp float;',
    'layout(location=0) in vec2 aPos;',
    'out vec2 vUv;',
    'void main(){ vUv = aPos*0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }'
  ].join('\n')

  var FS_BRIGHT = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D uTex; uniform float uThreshold; uniform float uKnee; uniform float uIntensity;',
    'in vec2 vUv; out vec4 fragColor;',
    'void main(){',
    '  vec3 c = texture(uTex, vUv).rgb;',
    '  float l = max(c.r, max(c.g, c.b));',
    '  float soft = clamp((l - uThreshold + uKnee)/(2.0*uKnee), 0.0, 1.0);',
    '  float w = max(l - uThreshold, l*soft*0.5)/max(l, 0.0001);',
    '  fragColor = vec4(c*w*uIntensity, 1.0);',
    '}'
  ].join('\n')

  var FS_BLUR = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D uTex; uniform vec2 uDir;',
    'in vec2 vUv; out vec4 fragColor;',
    'void main(){',
    '  vec4 s = texture(uTex, vUv)*0.2270270270;',
    '  s += texture(uTex, vUv + uDir*1.3846153846)*0.3162162162;',
    '  s += texture(uTex, vUv - uDir*1.3846153846)*0.3162162162;',
    '  s += texture(uTex, vUv + uDir*3.2307692308)*0.0702702703;',
    '  s += texture(uTex, vUv - uDir*3.2307692308)*0.0702702703;',
    '  fragColor = s;',
    '}'
  ].join('\n')

  var FS_COMP = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D uScene; uniform sampler2D uB0; uniform sampler2D uB1;',
    'uniform sampler2D uB2; uniform sampler2D uB3; uniform float uBloom;',
    'in vec2 vUv; out vec4 fragColor;',
    'void main(){',
    '  vec3 c = texture(uScene, vUv).rgb;',
    '  vec3 b = texture(uB0, vUv).rgb*0.60 + texture(uB1, vUv).rgb*0.44',
    '         + texture(uB2, vUv).rgb*0.32 + texture(uB3, vUv).rgb*0.24;',
    '  c += b*uBloom;',
    '  c = c/(1.0 + c*0.38);',
    '  c = pow(max(c, 0.0), vec3(0.92));',
    '  float d = length(vUv - 0.5)*1.32;',
    '  c *= 1.0 - smoothstep(0.42, 1.05, d)*0.42;',
    '  fragColor = vec4(c, 1.0);',
    '}'
  ].join('\n')

  function compile (gl, type, src) {
    var s = gl.createShader(type)
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('shader compile: ' + gl.getShaderInfoLog(s))
    }
    return s
  }

  function program (gl, vs, fs) {
    var p = gl.createProgram()
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs))
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs))
    gl.linkProgram(p)
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('program link: ' + gl.getProgramInfoLog(p))
    }
    return p
  }

  function uniforms (gl, p, names) {
    var o = {}
    for (var i = 0; i < names.length; i++) o[names[i]] = gl.getUniformLocation(p, names[i])
    return o
  }

  function setPerspective (m, fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2)
    var nf = 1 / (near - far)
    m[0] = f / aspect; m[1] = 0; m[2] = 0; m[3] = 0
    m[4] = 0; m[5] = f; m[6] = 0; m[7] = 0
    m[8] = 0; m[9] = 0; m[10] = (far + near) * nf; m[11] = -1
    m[12] = 0; m[13] = 0; m[14] = 2 * far * near * nf; m[15] = 0
  }

  /* -------------------------------- engine -------------------------------- */
  function create (canvas) {
    var gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, powerPreference: 'high-performance'
    })
    if (!gl) throw new Error('WebGL2 not available')

    var hdr = !!gl.getExtension('EXT_color_buffer_half_float')
    var texType = hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE
    var texFmt = hdr ? gl.RGBA16F : gl.RGBA8

    var data = buildParticles()
    var pPoints = program(gl, VS_POINTS, FS_POINTS)
    var pBright = program(gl, VS_QUAD, FS_BRIGHT)
    var pBlur = program(gl, VS_QUAD, FS_BLUR)
    var pComp = program(gl, VS_QUAD, FS_COMP)

    var uP = uniforms(gl, pPoints, ['uProj', 'uView', 'uTime', 'uProgress', 'uAmp', 'uColorMix', 'uRot', 'uWorldSize', 'uProjScale'])
    var uBr = uniforms(gl, pBright, ['uTex', 'uThreshold', 'uKnee', 'uIntensity'])
    var uBl = uniforms(gl, pBlur, ['uTex', 'uDir'])
    var uC = uniforms(gl, pComp, ['uScene', 'uB0', 'uB1', 'uB2', 'uB3', 'uBloom'])

    function attrib (index, arr, size) {
      var b = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, b)
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW)
      gl.enableVertexAttribArray(index)
      gl.vertexAttribPointer(index, size, gl.FLOAT, false, 0, 0)
    }

    var vao = gl.createVertexArray()
    gl.bindVertexArray(vao)
    attrib(0, data.source, 3)
    attrib(1, data.target, 3)
    attrib(2, data.color, 3)
    attrib(3, data.seed, 1)
    attrib(4, data.delay, 1)
    attrib(5, data.kind, 1)
    attrib(6, data.ring, 1)
    attrib(7, data.scale, 1)
    attrib(8, data.flow, 1)
    gl.bindVertexArray(null)

    var qvao = gl.createVertexArray()
    gl.bindVertexArray(qvao)
    var qb = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, qb)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.bindVertexArray(null)

    var scene = null, bright = null, levels = []

    function makeTarget (w, h) {
      var tex = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texImage2D(gl.TEXTURE_2D, 0, texFmt, w, h, 0, gl.RGBA, texType, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      var fb = gl.createFramebuffer()
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      return { tex: tex, fb: fb, w: w, h: h }
    }

    function destroyTargets () {
      var all = []
      if (scene) all.push(scene)
      if (bright) all.push(bright)
      for (var i = 0; i < levels.length; i++) { all.push(levels[i].a, levels[i].b) }
      for (var k = 0; k < all.length; k++) {
        gl.deleteTexture(all[k].tex)
        gl.deleteFramebuffer(all[k].fb)
      }
      scene = null; bright = null; levels = []
    }

    function buildTargets (w, h) {
      scene = makeTarget(w, h)
      bright = makeTarget(Math.max(2, w >> 1), Math.max(2, h >> 1))
      levels = []
      for (var i = 0; i < 4; i++) {
        var lw = Math.max(2, w >> (i + 1))
        var lh = Math.max(2, h >> (i + 1))
        levels.push({ a: makeTarget(lw, lh), b: makeTarget(lw, lh), w: lw, h: lh })
      }
    }

    function resize () {
      var dpr = Math.min(global.devicePixelRatio || 1, 2)
      var w = Math.max(2, Math.round((canvas.clientWidth || 1) * dpr))
      var h = Math.max(2, Math.round((canvas.clientHeight || 1) * dpr))
      if (scene && w === scene.w && h === scene.h) return
      canvas.width = w
      canvas.height = h
      destroyTargets()
      buildTargets(w, h)
    }

    function pass (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null)
      gl.viewport(0, 0, target ? target.w : canvas.width, target ? target.h : canvas.height)
      gl.bindVertexArray(qvao)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      gl.bindVertexArray(null)
    }

    var projM = new Float32Array(16)
    var viewM = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    var FOV = 42 * Math.PI / 180

    function render (o) {
      o = o || {}
      resize()

      var aspect = scene.w / scene.h
      var tanH = Math.tan(FOV / 2)
      /* keep the full shoulder span in frame on any window shape */
      var halfH = Math.max(2.95, 3.10 / aspect)
      if (halfH > 4.2) halfH = 4.2
      var camZ = halfH / tanH
      var camY = 6.45 - halfH

      setPerspective(projM, FOV, aspect, 0.1, 60)
      viewM[13] = -camY
      viewM[14] = -camZ

      gl.bindFramebuffer(gl.FRAMEBUFFER, scene.fb)
      gl.viewport(0, 0, scene.w, scene.h)
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE)

      gl.useProgram(pPoints)
      gl.bindVertexArray(vao)
      gl.uniformMatrix4fv(uP.uProj, false, projM)
      gl.uniformMatrix4fv(uP.uView, false, viewM)
      gl.uniform1f(uP.uTime, o.time || 0)
      gl.uniform1f(uP.uProgress, o.progress == null ? 1 : o.progress)
      gl.uniform1f(uP.uAmp, o.amp || 0)
      gl.uniform1f(uP.uColorMix, o.colorMix || 0)
      gl.uniform1f(uP.uRot, o.rot || 0)
      gl.uniform1f(uP.uWorldSize, 0.0185)
      gl.uniform1f(uP.uProjScale, scene.h / (2 * tanH))
      gl.drawArrays(gl.POINTS, 0, data.count)
      gl.bindVertexArray(null)
      gl.disable(gl.BLEND)

      gl.useProgram(pBright)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, scene.tex)
      gl.uniform1i(uBr.uTex, 0)
      gl.uniform1f(uBr.uThreshold, 0.16)
      gl.uniform1f(uBr.uKnee, 0.7)
      gl.uniform1f(uBr.uIntensity, 1.0)
      pass(bright)

      gl.useProgram(pBlur)
      gl.uniform1i(uBl.uTex, 0)
      var srcTex = bright.tex
      for (var i = 0; i < levels.length; i++) {
        var lv = levels[i]
        var spread = 1.0 + i * 0.45
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, srcTex)
        gl.uniform2f(uBl.uDir, spread / lv.w, 0)
        pass(lv.b)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, lv.b.tex)
        gl.uniform2f(uBl.uDir, 0, spread / lv.h)
        pass(lv.a)
        srcTex = lv.a.tex
      }

      gl.useProgram(pComp)
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scene.tex)
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, levels[0].a.tex)
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, levels[1].a.tex)
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, levels[2].a.tex)
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, levels[3].a.tex)
      gl.uniform1i(uC.uScene, 0)
      gl.uniform1i(uC.uB0, 1)
      gl.uniform1i(uC.uB1, 2)
      gl.uniform1i(uC.uB2, 3)
      gl.uniform1i(uC.uB3, 4)
      gl.uniform1f(uC.uBloom, o.bloom == null ? 1.2 : o.bloom)
      pass(null)
    }

    resize()

    return {
      ok: true,
      render: render,
      resize: resize,
      count: data.count,
      hdr: hdr,
      gl: gl
    }
  }

  global.AstaEngine = {
    create: create,
    sectionAt: sectionAt,
    SECTIONS: SECTIONS,
    TOP_Y: TOP_Y,
    BOTTOM_Y: BOTTOM_Y,
    ORB_Y: ORB_Y,
    FACE_Y: FACE_Y
  }
})(typeof window !== 'undefined' ? window : this)
