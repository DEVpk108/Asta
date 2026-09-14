'use strict'

/*
 * A.S.T.A. HUD state bridge.
 *
 * The Python HUD protocol owns the canonical state. The existing renderer
 * still has four visual animation profiles, so richer states select the
 * nearest visual profile while preserving the real mode in the HUD labels.
 */
(function () {
  var VALID_MODES = {
    idle: true,
    listening: true,
    thinking: true,
    speaking: true,
    executing: true,
    approval: true,
    error: true
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

  function $(id) {
    return document.getElementById(id)
  }

  function selectVisualProfile (profile) {
    var button = document.querySelector('.dock button[data-state="' + profile + '"]')
    if (!button) return
    button.click()
  }

  function renderCanonicalState (state) {
    state = state || {}
    var mode = String(state.mode || 'idle').toLowerCase()
    var intensity = String(state.intensity || 'low').toLowerCase()

    if (!VALID_MODES[mode]) mode = 'idle'
    if (!INTENSITY_LABELS[intensity]) intensity = 'low'

    selectVisualProfile(VISUAL_PROFILE[mode])

    var statusMode = $('statusMode')
    var statusIntensity = $('statusIntensity')
    var readoutLabel = $('readoutLabel')
    var readoutPct = $('readoutPct')

    if (statusMode) statusMode.textContent = MODE_LABELS[mode]
    if (statusIntensity) statusIntensity.textContent = INTENSITY_LABELS[intensity]

    if (readoutLabel) {
      readoutLabel.textContent = state.status || 'STATUS:'
    }

    if (readoutPct) {
      readoutPct.textContent = MODE_LABELS[mode]
      readoutPct.className = 'pct status-big' + (mode === 'speaking' ? ' pulse' : '')
    }

    document.body.dataset.hudMode = mode
    document.body.dataset.hudActivity = state.activity || ''

    if (state.progress !== null && state.progress !== undefined) {
      document.body.dataset.hudProgress = String(state.progress)
    } else {
      delete document.body.dataset.hudProgress
    }
  }

  var bridge = window.asta || null
  if (!bridge || !bridge.onHudState) return

  bridge.onHudState(function (state) {
    try {
      renderCanonicalState(state)
    } catch (error) {
      if (window.console) {
        console.warn('[ASTA HUD] Invalid canonical state:', error)
      }
    }
  })
})()
