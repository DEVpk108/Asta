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

  /* The renderer's assembly animation starts immediately, but its built-in
     "Ready" speech must not fire until Python reports that A.S.T.A. is
     actually ready to listen. */
  var nativeSpeak = null
  var backendReady = false
  if (window.speechSynthesis && typeof window.speechSynthesis.speak === 'function') {
    nativeSpeak = window.speechSynthesis.speak.bind(window.speechSynthesis)
    window.speechSynthesis.speak = function (utterance) {
      var text = utterance && String(utterance.text || '').trim().toLowerCase()
      if (!backendReady && text === 'ready') return
      nativeSpeak(utterance)
    }
  }

  /* Start the visual runtime as soon as the HUD is loaded. Python/kernel
     readiness remains authoritative for canonical state, but the visual
     assembly should not wait for the AI stack to finish initializing. */
  if (window.AstaHUD && window.AstaHUD.beginRuntime) {
    window.AstaHUD.beginRuntime()
  }

  if (bridge.onHudLifecycle) {
    bridge.onHudLifecycle(function (lifecycle) {
      try {
        var status = String((lifecycle && lifecycle.status) || '').toLowerCase()
        if (window.console) {
          console.log('[ASTA HUD] Runtime lifecycle:', status || 'unknown')
        }

        if (status === 'ready') {
          backendReady = true
          if (nativeSpeak && window.speechSynthesis) {
            window.speechSynthesis.speak = nativeSpeak
          }
        }

        if (status === 'ready' && window.AstaHUD && window.AstaHUD.beginRuntime) {
          window.AstaHUD.beginRuntime()
        }
      } catch (error) {
        if (window.console) {
          console.warn('[ASTA HUD] Invalid runtime lifecycle:', error)
        }
      }
    })
  }

  if (bridge.onHudState) {
    bridge.onHudState(function (state) {
      try {
        state = state || {}
        if (window.console) {
          console.log('[ASTA HUD] Renderer state:', state.mode || 'idle')
        }

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
