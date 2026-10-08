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
  var readyPending = false
  if (window.speechSynthesis && typeof window.speechSynthesis.speak === 'function') {
    nativeSpeak = window.speechSynthesis.speak.bind(window.speechSynthesis)
    window.speechSynthesis.speak = function (utterance) {
      var text = utterance && String(utterance.text || '').trim().toLowerCase()
      if (!backendReady && text === 'ready') {
        readyPending = true
        return
      }
      nativeSpeak(utterance)
    }
  }

  function announceReady () {
    if (!backendReady || !readyPending || !nativeSpeak || !window.speechSynthesis) return
    readyPending = false

    try {
      var utterance = new SpeechSynthesisUtterance('Ready')
      utterance.rate = 0.85
      utterance.pitch = 0.55
      utterance.volume = 0.9
      nativeSpeak(utterance)
    } catch (error) {
      if (window.console) {
        console.warn('[ASTA HUD] Ready announcement failed:', error)
      }
    }
  }

  function renderWorkspaceContext (workspace) {
    var card = document.getElementById('workspaceContext')
    var nameNode = document.getElementById('workspaceProjectName')
    var branchNode = document.getElementById('workspaceBranch')
    var repositoryNode = document.getElementById('workspaceRepository')
    var recentRow = document.getElementById('workspaceRecentRow')
    var recentList = document.getElementById('workspaceRecentFiles')
    if (!card || !nameNode || !branchNode || !repositoryNode || !recentRow || !recentList) return

    workspace = workspace || {}
    var projectName = String(workspace.project_name || '').trim().slice(0, 96)
    var branch = String(workspace.branch || '').trim().slice(0, 72)
    var repository = String(workspace.repository || '').trim().slice(0, 180)
    var files = Array.isArray(workspace.recent_files) ? workspace.recent_files.slice(0, 4) : []

    card.hidden = !projectName
    if (!projectName) return

    nameNode.textContent = projectName
    branchNode.textContent = branch
    branchNode.hidden = !branch
    repositoryNode.textContent = repository
    repositoryNode.hidden = !repository

    while (recentList.firstChild) recentList.removeChild(recentList.firstChild)
    files.forEach(function (file) {
      if (typeof file !== 'string' || !file.trim()) return
      var item = document.createElement('span')
      item.className = 'workspace-file'
      item.textContent = file.trim().slice(0, 160)
      item.title = item.textContent
      recentList.appendChild(item)
    })
    recentRow.hidden = recentList.childNodes.length === 0
  }

  function renderTaskStatus (state) {
    var card = document.getElementById('taskStatus')
    var stageNode = document.getElementById('taskStatusStage')
    var detailNode = document.getElementById('taskStatusDetail')
    if (!card || !stageNode || !detailNode) return

    var mode = String((state && state.mode) || '').toLowerCase()
    var status = String((state && state.status) || '').trim()
    var active = mode === 'executing' || mode === 'approval' ||
      (mode === 'thinking' && /^(planning|planned|replanning|recovery)$/i.test(status))
    card.hidden = !active
    if (!active) return

    var activity = String((state && state.activity) || '').trim()
    var sandboxRun = status.toUpperCase() === 'SANDBOX RUN'
    if (mode === 'approval') {
      stageNode.textContent = 'APPROVAL REQUIRED'
      detailNode.textContent = activity || 'Review the requested action before it runs.'
      card.dataset.stage = 'approval'
      return
    }

    card.dataset.stage = 'active'
    if (sandboxRun) {
      stageNode.textContent = 'RUNNING IN SANDBOX'
      detailNode.textContent = 'Python/pytest · read-only project snapshot · network disabled'
    } else if (mode === 'thinking' && /^(planning|planned)$/i.test(status)) {
      stageNode.textContent = 'PLANNING'
      detailNode.textContent = 'Preparing a bounded sequence of actions and checks.'
    } else if (mode === 'thinking') {
      stageNode.textContent = status ? status.replace(/_/g, ' ').toUpperCase() : 'RECOVERING'
      detailNode.textContent = activity || 'Choosing a safe next step.'
    } else {
      stageNode.textContent = status ? status.replace(/_/g, ' ').toUpperCase() : 'EXECUTING'
      detailNode.textContent = activity || 'Working through the current step.'
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
          announceReady()
        } else if (status === 'starting') {
          backendReady = false
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
        renderTaskStatus(state)

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

  if (bridge.onHudWorkspace) {
    bridge.onHudWorkspace(function (workspace) {
      try {
        renderWorkspaceContext(workspace)
      } catch (error) {
        if (window.console) {
          console.warn('[ASTA HUD] Invalid workspace context:', error)
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
