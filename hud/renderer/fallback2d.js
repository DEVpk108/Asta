/* Canvas2D fallback renderer — used only when WebGL2 is unavailable.
   Mirrors the humanoid cross-section model from engine.js. */
'use strict'

;(function (global) {
  function create (canvas) {
    var ctx = canvas.getContext('2d')
    if (!ctx) return null

    var E = global.AstaEngine || {}
    var TOP_Y = E.TOP_Y == null ? 6.0 : E.TOP_Y
    var BOTTOM_Y = E.BOTTOM_Y == null ? 0.85 : E.BOTTOM_Y
    var ORB_Y = E.ORB_Y == null ? -0.85 : E.ORB_Y
    var FACE_Y = E.FACE_Y == null ? 4.58 : E.FACE_Y
    var sectionAt = E.sectionAt || function () { return { rx: 0, rz: 0 } }

    /* shoulder girdle + hanging upper arm, mirrored left/right */
    var LIMBS2D = [
      [[0.30, 3.30], [0.70, 3.24], [1.15, 3.10], [1.55, 2.88], [1.85, 2.60], [2.00, 2.30], [2.10, 2.05]],
      [[2.04, 1.92], [2.07, 1.45], [2.06, 1.00], [2.01, 0.55]]
    ]

    var NEURAL = [
      [[0, ORB_Y], [0, 1.30], [0, 2.10], [0, 2.70], [0, 3.56]],
      [[0, 2.55], [-0.28, 2.20], [-0.55, 1.70], [-0.72, 1.15], [-0.80, 0.55]],
      [[0, 2.55], [0.28, 2.20], [0.55, 1.70], [0.72, 1.15], [0.80, 0.55]],
      [[0, 2.36], [-0.26, 1.25], [-0.30, 0.65]],
      [[0, 2.36], [0.26, 1.25], [0.30, 0.65]]
    ]

    var dpr = 1, W = 1, H = 1

    function resize () {
      dpr = Math.min(global.devicePixelRatio || 1, 2)
      W = Math.max(1, canvas.clientWidth)
      H = Math.max(1, canvas.clientHeight)
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()

    function render (s) {
      resize()
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, W, H)
      ctx.globalCompositeOperation = 'lighter'

      var circleCx = W / 2
      var circleCy = H * 0.50
      var circleR = Math.min(W, H) * 0.27
      ctx.lineWidth = 1.2
      ctx.shadowBlur = 14
      ctx.shadowColor = 'rgba(0,229,255,0.7)'
      for (var ringIndex = 0; ringIndex < 9; ringIndex++) {
        var ringRadius = circleR * (0.42 + ringIndex * 0.07)
        ctx.strokeStyle = 'rgba(30,190,255,' + (0.16 + ringIndex * 0.025) + ')'
        ctx.beginPath()
        ctx.arc(circleCx, circleCy, ringRadius, 0, Math.PI * 2)
        ctx.stroke()
      }
      for (var spoke = 0; spoke < 32; spoke++) {
        var spokeAngle = spoke / 32 * Math.PI * 2 + s.time * 0.08
        var inner = circleR * 0.34
        var outer = circleR * (0.90 + 0.08 * Math.sin(s.time * 2 + spoke * 1.7))
        ctx.strokeStyle = 'rgba(90,220,255,' + (0.18 + 0.12 * s.progress) + ')'
        ctx.beginPath()
        ctx.moveTo(circleCx + Math.cos(spokeAngle) * inner, circleCy + Math.sin(spokeAngle) * inner)
        ctx.lineTo(circleCx + Math.cos(spokeAngle) * outer, circleCy + Math.sin(spokeAngle) * outer)
        ctx.stroke()
      }
      var coreRadius = circleR * (0.16 + 0.03 * Math.sin(s.time * 2.2))
      ctx.fillStyle = 'rgba(255,145,0,' + (0.45 * s.progress) + ')'
      ctx.shadowColor = 'rgba(255,145,0,0.9)'
      ctx.beginPath()
      ctx.arc(circleCx, circleCy, coreRadius, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
      ctx.globalCompositeOperation = 'source-over'
    }

    return { ok: true, count: 0, resize: resize, render: render }
  }

  global.AstaFallback2D = { create: create }
})(window)
