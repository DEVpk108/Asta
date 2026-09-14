'use strict'

;(function () {
  var bridge = window.asta || null
  var form = document.getElementById('chatForm')
  var input = document.getElementById('chatInput')
  var messages = document.getElementById('chatMessages')
  var empty = document.getElementById('chatEmpty')
  var chatPanel = document.getElementById('chatPanel')
  var fullscreenButton = document.getElementById('chatFullscreen')
  var fullscreenLabel = fullscreenButton && fullscreenButton.querySelector('.chat-expand-label')
  var historyButton = document.getElementById('chatHistory')
  var recentList = document.getElementById('chatRecentList')
  var recentEmpty = document.getElementById('chatRecentEmpty')
  var newChatButton = document.getElementById('chatNew')

  if (!form || !input || !messages || !chatPanel) return

  var currentSessionId = null
  var recentVisible = false

  function removeEmpty () {
    if (empty && empty.parentNode) empty.parentNode.removeChild(empty)
    empty = null
  }

  function progressiveReveal (body, text) {
    var chars = Array.from(String(text || ''))
    if (!chars.length) return
    body.textContent = ''
    var index = 0
    var chunk = chars.length > 600 ? 10 : chars.length > 300 ? 7 : 4
    var timer = window.setInterval(function () {
      var end = Math.min(chars.length, index + chunk)
      body.textContent += chars.slice(index, end).join('')
      messages.scrollTop = messages.scrollHeight
      index = end
      if (index >= chars.length) window.clearInterval(timer)
    }, 9)
  }

  function appendMessage (role, text, animate) {
    var value = String(text || '').trim()
    if (!value) return
    removeEmpty()
    var item = document.createElement('div')
    item.className = 'chat-message ' + (role === 'user' ? 'user' : 'assistant')
    var label = document.createElement('div')
    label.className = 'chat-role'
    label.textContent = role === 'user' ? 'YOU' : 'A.S.T.A.'
    var body = document.createElement('div')
    body.className = 'chat-text'
    item.appendChild(label)
    item.appendChild(body)
    messages.appendChild(item)
    messages.scrollTop = messages.scrollHeight
    if (animate && role === 'assistant' && document.body.classList.contains('text-chat-fullscreen')) progressiveReveal(body, value)
    else body.textContent = value
  }

  function loadHistory (history) {
    var payload = Array.isArray(history) ? { messages: history, sessions: [] } : (history || {})
    currentSessionId = payload.session_id || currentSessionId
    renderRecents(Array.isArray(payload.sessions) ? payload.sessions : [])
    var items = Array.isArray(payload.messages) ? payload.messages : []
    messages.innerHTML = ''
    empty = null
    for (var index = 0; index < items.length; index += 1) {
      var message = items[index]
      if (!message || !message.text) continue
      appendMessage(message.role === 'user' ? 'user' : 'assistant', message.text, false)
    }
    if (!messages.children.length) {
      empty = document.createElement('div')
      empty.className = 'chat-empty'
      empty.id = 'chatEmpty'
      empty.textContent = 'NEW CONVERSATION'
      messages.appendChild(empty)
    }
    messages.scrollTop = messages.scrollHeight
  }

  function renderRecents (sessions) {
    recentList.innerHTML = ''
    recentEmpty.style.display = sessions.length ? 'none' : 'block'
    for (var i = 0; i < sessions.length; i += 1) {
      var session = sessions[i]
      if (!session || !session.id) continue
      var button = document.createElement('button')
      button.type = 'button'
      button.className = 'chat-recent-item' + (session.id === currentSessionId ? ' active' : '')
      button.dataset.sessionId = session.id
      button.title = session.preview || session.title || 'Conversation'
      var title = document.createElement('span')
      title.className = 'chat-recent-title'
      title.textContent = session.title || 'New conversation'
      var meta = document.createElement('span')
      meta.className = 'chat-recent-meta'
      var count = Number(session.message_count || 0)
      meta.textContent = count ? count + (count === 1 ? ' MESSAGE' : ' MESSAGES') : 'NEW'
      button.appendChild(title)
      button.appendChild(meta)
      button.addEventListener('click', function () {
        var id = this.dataset.sessionId
        if (bridge && bridge.selectChatSession && id) bridge.selectChatSession(id)
      })
      recentList.appendChild(button)
    }
  }

  function setRecents (enabled) {
    recentVisible = !!enabled
    chatPanel.classList.toggle('chat-show-history', recentVisible)
    if (historyButton) {
      historyButton.classList.toggle('active', recentVisible)
      historyButton.setAttribute('aria-pressed', recentVisible ? 'true' : 'false')
    }
  }

  function toggleRecents () { setRecents(!recentVisible) }

  function setFullscreen (enabled) {
    chatPanel.classList.toggle('chat-fullscreen', enabled)
    document.body.classList.toggle('text-chat-fullscreen', enabled)
    if (fullscreenButton) {
      fullscreenButton.title = enabled ? 'Exit fullscreen chat' : 'Expand conversation'
      fullscreenButton.setAttribute('aria-label', enabled ? 'Exit fullscreen chat' : 'Expand conversation')
    }
    if (fullscreenLabel) fullscreenLabel.textContent = enabled ? 'EXIT' : 'FULLSCREEN'
    if (enabled) setRecents(true)
    window.requestAnimationFrame(function () {
      window.dispatchEvent(new Event('resize'))
      messages.scrollTop = messages.scrollHeight
      if (enabled) input.focus()
    })
  }

  if (historyButton) historyButton.addEventListener('click', function (event) {
    event.preventDefault()
    event.stopPropagation()
    toggleRecents()
  })

  if (newChatButton) newChatButton.addEventListener('click', function (event) {
    event.preventDefault()
    event.stopPropagation()
    if (bridge && bridge.startNewChat) bridge.startNewChat()
  })

  if (fullscreenButton) fullscreenButton.addEventListener('click', function (event) {
    event.preventDefault()
    event.stopPropagation()
    setFullscreen(!chatPanel.classList.contains('chat-fullscreen'))
  })

  window.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && chatPanel.classList.contains('chat-fullscreen')) {
      event.preventDefault()
      setFullscreen(false)
      input.focus()
    }
  })

  form.addEventListener('submit', function (event) {
    event.preventDefault()
    var text = input.value.trim()
    if (!text) return
    if (!bridge || !bridge.sendTextMessage) return
    input.value = ''
    input.focus()
    bridge.sendTextMessage(text)
  })

  if (bridge && bridge.onHudChatHistory) {
    bridge.onHudChatHistory(function (history) {
      try { loadHistory(history) } catch (error) { console.warn('[ASTA HUD] Invalid chat history:', error) }
    })
  }

  if (bridge && bridge.onHudChatSessions) {
    bridge.onHudChatSessions(function (sessions) {
      try { renderRecents(Array.isArray(sessions) ? sessions : []) } catch (error) { console.warn('[ASTA HUD] Invalid chat sessions:', error) }
    })
  }

  if (bridge && bridge.onHudChat) {
    bridge.onHudChat(function (message) {
      try {
        if (!message || !message.text) return
        appendMessage(message.role === 'user' ? 'user' : 'assistant', message.text, true)
      } catch (error) { console.warn('[ASTA HUD] Invalid chat message:', error) }
    })
  }

  setRecents(false)
})()
