'use strict'

/*
 * A.S.T.A. HUD runtime bridge.
 *
 * Python owns canonical state and live speech telemetry. The app renderer
 * owns the actual visual state machine and WebGL animation response.
 */
(function () {
  var bridge = window.asta || null
  if (!bridge) return

  if (bridge.onHudState) {
    bridge.onHudState(function (state) {
      try {
        if (window.AstaHUD && window.AstaHUD.applyCanonicalState) {
          window.AstaHUD.applyCanonicalState(state)
        }

        state = state || {}
        document.body.dataset.hudMode = String(state.mode || 'idle').toLowerCase()
        document.body.dataset.hudActivity = state.activity || ''

        if (state.progress !== null && state.progress !== undefined) {
          document.body.dataset.hudProgress = String(state.progress)
        } else {
          delete document.body.dataset.hudProgress
        }
      } catch (error) {
        if (window.console) {
          console.warn('[ASTA HUD] Invalid canonical state:', error)
        }
      }
    })
  }

  if (bridge.onHudAudio) {
    bridge.onHudAudio(function (audio) {
      try {
        var level = audio && audio.level
        if (window.AstaHUD && window.AstaHUD.setAudioLevel) {
          window.AstaHUD.setAudioLevel(level)
        }
      } catch (error) {
        if (window.console) {
          console.warn('[ASTA HUD] Invalid audio telemetry:', error)
        }
      }
    })
  }
})()
