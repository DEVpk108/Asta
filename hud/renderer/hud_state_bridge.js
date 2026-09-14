;'use strict'

/*
 * A.S.T.A. HUD runtime bridge.
 *
 * Python owns canonical state and live speech telemetry. The app renderer
 * owns the actual visual state machine and WebGL animation response.
 */
;(function () {
  var bridge = window.asta || null
  if (!bridge) return

  /* Preserve the HUD's startup assembly animation on first launch. The
     backend's initial state is still authoritative; we simply defer applying
     the first state until the visual assembly has had time to complete. */
  var startupState = null
  var startupTimer = null
  var startupComplete = false
  var STARTUP_ASSEMBLE_MS = 5400

  function applyState (state) {
    state = state || {}

    if (window.AstaHUD && window.AstaHUD.applyCanonicalState) {
      window.AstaHUD.applyCanonicalState(state)
    }

    document.body.dataset.hudMode = String(state.mode || 'idle').toLowerCase()
    document.body.dataset.hudActivity = state.activity || ''

    if (state.progress !== null && state.progress !== undefined) {
      document.body.dataset.hudProgress = String(state.progress)
    } else {
      delete document.body.dataset.hudProgress
    }
  }

  function finishStartup () {
    if (startupComplete) return
    startupComplete = true
    if (startupTimer) {
      clearTimeout(startupTimer)
      startupTimer = null
    }

    if (startupState) {
      var state = startupState
      startupState = null
      applyState(state)
    }
  }

  if (bridge.onHudState) {
    bridge.onHudState(function (state) {
      try {
        state = state || {}
        if (window.console) {
          console.log('[ASTA HUD] Renderer state:', state.mode || 'idle')
        }

        if (!startupComplete) {
          startupState = state
          if (!startupTimer) {
            startupTimer = setTimeout(finishStartup, STARTUP_ASSEMBLE_MS)
          }
        } else {
          applyState(state)
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
