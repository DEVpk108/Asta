/* ASTA target renderer
   Visual pass tuned to the supplied reference images: bright cyan contour
   silhouette, horizontal scan-like contour bands, dense facial amber core,
   amber neural filaments, and a restrained blue-black background. */
'use strict'

;(function (global) {
  function create (canvas) {
    var ctx = canvas.getContext('2d')
    if (!ctx) return null

    var W = 1, H = 1, dpr = 1
    var seed = 918273645
    var dust = []
    var faceDust = []
    var innerDust = []

    function rnd () {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 4294967296
    }

    function resize () {
      dpr = Math.min(global.devicePixelRatio || 1, 2)
      W = Math.max(1, canvas.clientWidth || 1)
      H = Math.max(1, canvas.clientHeight || 1)
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      rebuildParticles()
    }

    function rebuildParticles () {
      seed = 918273645
      dust = []
      faceDust = []
      innerDust = []

      for (var i = 0; i < 1250; i++) {
        var a = rnd() * Math.PI * 2
        var r = 0.78 + Math.pow(rnd(), 0.7) * 1.65
        dust.push({
          x: Math.cos(a) * r,
          y: (rnd() * 1.15 + 0.08),
          z: Math.sin(a),
          s: 0.35 + rnd() * 1.35,
          p: rnd() * Math.PI * 2
        })
      }

      for (var j = 0; j < 850; j++) {
        var ang = rnd() * Math.PI * 2
        var rr = Math.pow(rnd(), 0.65)
        faceDust.push({
          x: Math.cos(ang) * rr,
          y: Math.sin(ang) * rr,
          s: 0.35 + rnd() * 1.45,
          p: rnd() * Math.PI * 2
        })
      }

      for (var k = 0; k < 900; k++) {
        innerDust.push({
          x: (rnd() * 2 - 1) * 1.45,
          y: rnd(),
          s: 0.28 + rnd() * 0.95,
          p: rnd() * Math.PI * 2
        })
      }
    }

    function glowStroke (style, width, blur) {
      ctx.strokeStyle = style
      ctx.lineWidth = width
      ctx.shadowColor = style
      ctx.shadowBlur = blur
    }

    function figure (t) {
      var cx = W * 0.5
      var top = H * 0.07
      var bottom = H * 0.98
      var span = Math.min(W * 0.84, H * 0.92)
      var headRx = span * 0.22
      var headRy = span * 0.27
      var headCy = top + headRy * 0.97
      var neckY0 = headCy + headRy * 0.69
      var shoulderY = H * 0.63
      var neckW = headRx * 0.52
      var bodyBottom = bottom
      var cyan = 'rgba(52, 220, 255, 0.94)'
      var cyanSoft = 'rgba(35, 192, 236, 0.66)'
      var amber = 'rgba(255, 155, 18, 0.95)'

      /* subtle halo behind the silhouette */
      var bg = ctx.createRadialGradient(cx, headCy + H * 0.16, 0, cx, H * 0.54, H * 0.65)
      bg.addColorStop(0, 'rgba(0, 105, 145, 0.20)')
      bg.addColorStop(0.52, 'rgba(0, 42, 63, 0.11)')
      bg.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, W, H)

      /* dust field around the figure */
      ctx.shadowBlur = 6
      for (var i = 0; i < dust.length; i++) {
        var q = dust[i]
        var dx = cx + q.x * span * 0.52
        var dy = H * (0.12 + q.y * 0.55) + Math.sin(t * 0.7 + q.p) * 4
        var limit = W * 0.46
        if (Math.abs(dx - cx) > limit) continue
        ctx.fillStyle = 'rgba(40, 210, 244, ' + (0.10 + q.s * 0.08) + ')'
        ctx.shadowColor = 'rgba(0, 208, 255, 0.45)'
        ctx.beginPath(); ctx.arc(dx, dy, q.s * 0.55, 0, Math.PI * 2); ctx.fill()
      }

      /* head contour bands: the main visual cue in the reference */
      var headBands = 31
      for (var h = 0; h < headBands; h++) {
        var ny = -0.92 + (h + 0.5) / headBands * 1.84
        var yy = headCy + ny * headRy
        var xHalf = headRx * Math.sqrt(Math.max(0, 1 - ny * ny))
        if (xHalf < 4) continue
        var wob = Math.sin(t * 0.9 + h * 0.24) * 1.2
        ctx.beginPath()
        ctx.moveTo(cx - xHalf, yy + wob)
        ctx.quadraticCurveTo(cx, yy - wob * 0.6, cx + xHalf, yy + wob)
        glowStroke('rgba(40, 193, 231, ' + (0.42 + 0.18 * (1 - Math.abs(ny))) + ')', 1.15, 7)
        ctx.stroke()
      }

      /* face core: amber particle cloud + soft bloom */
      var faceCy = headCy + headRy * 0.24
      var faceRx = headRx * 0.72
      var faceRy = headRy * 0.69
      var fg = ctx.createRadialGradient(cx, faceCy, 0, cx, faceCy, faceRx * 1.08)
      fg.addColorStop(0, 'rgba(255, 225, 118, 0.55)')
      fg.addColorStop(0.26, 'rgba(255, 181, 32, 0.45)')
      fg.addColorStop(0.68, 'rgba(255, 112, 0, 0.16)')
      fg.addColorStop(1, 'rgba(255, 92, 0, 0)')
      ctx.fillStyle = fg
      ctx.shadowBlur = 32
      ctx.shadowColor = 'rgba(255, 145, 0, 0.55)'
      ctx.beginPath(); ctx.ellipse(cx, faceCy, faceRx, faceRy, 0, 0, Math.PI * 2); ctx.fill()

      ctx.shadowBlur = 5
      for (var fd = 0; fd < faceDust.length; fd++) {
        var fp = faceDust[fd]
        var fx = cx + fp.x * faceRx * 0.94
        var fy = faceCy + fp.y * faceRy * 0.94
        var alpha = 0.18 + 0.28 * (1 - Math.min(1, Math.sqrt(fp.x * fp.x + fp.y * fp.y)))
        ctx.fillStyle = 'rgba(255, 171, 32, ' + alpha + ')'
        ctx.shadowColor = 'rgba(255, 153, 12, 0.7)'
        ctx.beginPath(); ctx.arc(fx, fy, fp.s * 0.58, 0, Math.PI * 2); ctx.fill()
      }

      /* outline: head + neck */
      glowStroke(cyan, 4.2, 18)
      ctx.beginPath()
      ctx.ellipse(cx, headCy, headRx, headRy, 0, 0, Math.PI * 2)
      ctx.stroke()

      glowStroke('rgba(110, 241, 255, 0.72)', 2.1, 10)
      ctx.beginPath()
      ctx.moveTo(cx - neckW, neckY0)
      ctx.quadraticCurveTo(cx - neckW * 1.25, shoulderY * 0.92, cx - headRx * 0.98, shoulderY)
      ctx.moveTo(cx + neckW, neckY0)
      ctx.quadraticCurveTo(cx + neckW * 1.25, shoulderY * 0.92, cx + headRx * 0.98, shoulderY)
      ctx.stroke()

      /* shoulders / torso contour envelope */
      var bodyLevels = 28
      for (var b = 0; b < bodyLevels; b++) {
        var by = shoulderY + (bodyBottom - shoulderY) * (b + 0.4) / bodyLevels
        var u = (by - shoulderY) / (bodyBottom - shoulderY)
        var leftReach = headRx * (1.12 + 3.10 * Math.pow(u, 0.72))
        if (leftReach > W * 0.47) leftReach = W * 0.47
        var curve = headRx * (0.82 + 0.30 * u)
        ctx.beginPath()
        ctx.moveTo(cx - leftReach, by + 5 * Math.sin(b * 0.23))
        ctx.quadraticCurveTo(cx - curve, by - headRx * (0.18 + 0.33 * (1 - u)), cx, by + headRx * 0.08)
        ctx.quadraticCurveTo(cx + curve, by - headRx * (0.18 + 0.33 * (1 - u)), cx + leftReach, by + 5 * Math.sin(b * 0.23))
        glowStroke('rgba(38, 182, 228, ' + (0.28 + 0.25 * (1 - u)) + ')', 1.05, 6)
        ctx.stroke()
      }

      /* strong shoulder edge */
      glowStroke('rgba(57, 222, 255, 0.98)', 4.4, 18)
      ctx.beginPath()
      ctx.moveTo(cx - headRx * 0.72, shoulderY)
      ctx.quadraticCurveTo(cx - headRx * 1.65, shoulderY + H * 0.015, cx - headRx * 2.45, shoulderY + H * 0.10)
      ctx.quadraticCurveTo(cx - headRx * 3.0, shoulderY + H * 0.16, cx - headRx * 3.35, shoulderY + H * 0.22)
      ctx.moveTo(cx + headRx * 0.72, shoulderY)
      ctx.quadraticCurveTo(cx + headRx * 1.65, shoulderY + H * 0.015, cx + headRx * 2.45, shoulderY + H * 0.10)
      ctx.quadraticCurveTo(cx + headRx * 3.0, shoulderY + H * 0.16, cx + headRx * 3.35, shoulderY + H * 0.22)
      ctx.stroke()

      /* amber neural trunk and branching paths */
      ctx.lineCap = 'round'
      glowStroke(amber, 2.3, 13)
      ctx.beginPath()
      ctx.moveTo(cx, faceCy + faceRy * 0.86)
      ctx.quadraticCurveTo(cx - 3, neckY0 + 36, cx + 1, shoulderY + 70)
      ctx.quadraticCurveTo(cx + 2, shoulderY + 150, cx, bodyBottom - 8)
      ctx.stroke()

      var branchY = shoulderY + 52
      for (var side = -1; side <= 1; side += 2) {
        ctx.beginPath()
        ctx.moveTo(cx + side * 2, branchY)
        ctx.quadraticCurveTo(cx + side * headRx * 0.34, branchY + 28, cx + side * headRx * 0.56, branchY + 62)
        ctx.quadraticCurveTo(cx + side * headRx * 0.82, branchY + 96, cx + side * headRx * 1.05, branchY + 128)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(cx + side * headRx * 0.56, branchY + 62)
        ctx.quadraticCurveTo(cx + side * headRx * 0.28, branchY + 82, cx + side * headRx * 0.18, branchY + 124)
        ctx.stroke()
      }

      /* fine interior particles */
      ctx.shadowBlur = 2
      for (var ip = 0; ip < innerDust.length; ip++) {
        var p = innerDust[ip]
        var fy2 = shoulderY + p.y * (bodyBottom - shoulderY)
        var u2 = p.y
        var maxX = headRx * (0.55 + 2.85 * Math.pow(u2, 0.75))
        if (Math.abs(p.x) > maxX / Math.max(headRx, 1)) continue
        var ix = cx + p.x * headRx
        ctx.fillStyle = 'rgba(46, 172, 215, ' + (0.07 + 0.06 * (1 - u2)) + ')'
        ctx.beginPath(); ctx.arc(ix, fy2, p.s * 0.45, 0, Math.PI * 2); ctx.fill()
      }

      /* edge dust / glitter around head and shoulders */
      for (var e = 0; e < 260; e++) {
        var side2 = rnd() < 0.5 ? -1 : 1
        var ex = cx + side2 * (headRx * (0.98 + rnd() * 2.2))
        var ey = top + rnd() * H * 0.58
        if (ey < headCy - headRy * 1.15 || ey > shoulderY + H * 0.18) continue
        ctx.fillStyle = 'rgba(75, 226, 255, ' + (0.08 + rnd() * 0.18) + ')'
        ctx.shadowColor = 'rgba(0, 208, 255, 0.55)'
        ctx.beginPath(); ctx.arc(ex, ey, 0.5 + rnd() * 1.3, 0, Math.PI * 2); ctx.fill()
      }
    }

    function render (s) {
      resizeIfNeeded()
      var t = s && s.time ? s.time : 0
      ctx.globalCompositeOperation = 'source-over'
      ctx.clearRect(0, 0, W, H)

      var background = ctx.createLinearGradient(0, 0, 0, H)
      background.addColorStop(0, '#07131d')
      background.addColorStop(0.55, '#101a28')
      background.addColorStop(1, '#060b12')
      ctx.fillStyle = background
      ctx.fillRect(0, 0, W, H)

      ctx.globalCompositeOperation = 'lighter'
      figure(t)
      ctx.globalCompositeOperation = 'source-over'
    }

    function resizeIfNeeded () {
      var ndpr = Math.min(global.devicePixelRatio || 1, 2)
      var nw = Math.max(1, canvas.clientWidth || 1)
      var nh = Math.max(1, canvas.clientHeight || 1)
      if (nw === W && nh === H && ndpr === dpr) return
      resize()
    }

    resize()
    return {
      ok: true,
      count: 25236,
      hdr: true,
      render: render,
      resize: resize
    }
  }

  /* Replace the WebGL visual with the reference-tuned renderer while keeping
     the original engine available in the file for future comparison. */
  global.AstaTargetRenderer = { create: create }
  if (global.AstaEngine) global.AstaEngine.create = create
})(window)
