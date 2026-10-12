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
  var conversationButton = document.getElementById('chatConversation')

  if (!form || !input || !messages || !chatPanel) return

  var currentSessionId = null
  var recentVisible = false
  var conversationMode = false

  function removeEmpty () {
    if (empty && empty.parentNode) empty.parentNode.removeChild(empty)
    empty = null
  }

  function inlineText (node, value) {
    // Small, deliberately safe Markdown subset: no model-generated HTML/URLs.
    var pattern = /(\*\*([^*]+)\*\*|`([^`]+)`)/g
    var previous = 0, match
    while ((match = pattern.exec(value))) {
      node.appendChild(document.createTextNode(value.slice(previous, match.index)))
      var element = document.createElement(match[2] ? 'strong' : 'code')
      element.textContent = match[2] || match[3]
      node.appendChild(element); previous = pattern.lastIndex
    }
    node.appendChild(document.createTextNode(value.slice(previous)))
  }
  function renderText (body, text) {
    var parts = String(text).split(/```([^\n]*)\n([\s\S]*?)(?:```|$)/g)
    for (var index = 0; index < parts.length; index += 3) {
      var plain = parts[index]
      if (plain) {
        plain.split(/\n\n+/).forEach(function (paragraph) {
          if (!paragraph.trim()) return
          var heading = paragraph.match(/^#{1,3}\s+(.+)$/)
          var node = document.createElement(heading ? 'h3' : 'p')
          inlineText(node, heading ? heading[1] : paragraph)
          body.appendChild(node)
        })
      }
      if (index + 2 < parts.length) {
        var block = document.createElement('section'); block.className = 'code-block'
        var toolbar = document.createElement('div'); toolbar.className = 'code-toolbar'
        var language = document.createElement('span'); language.textContent = parts[index + 1].trim() || 'Code'
        var button = document.createElement('button'); button.type = 'button'; button.textContent = 'Copy code'
        button.setAttribute('aria-label', 'Copy code block')
        ;(function (content, control) {
          control.addEventListener('click', function () {
            if (!navigator.clipboard || !navigator.clipboard.writeText) { control.textContent = 'Select to copy'; return }
            navigator.clipboard.writeText(content).then(function () { control.textContent = 'Copied' }, function () { control.textContent = 'Select to copy' })
          })
        })(parts[index + 2], button)
        toolbar.append(language, button)
        var pre = document.createElement('pre'); var code = document.createElement('code')
        code.textContent = parts[index + 2]; pre.appendChild(code)
        block.append(toolbar, pre); body.appendChild(block)
      }
    }
  }

  function appendMessage (role, text, animate) {
    var value = String(text || '').trim()
    if (!value) return
    removeEmpty()
    var item = document.createElement('div')
    item.className = 'chat-message ' + (role === 'user' ? 'user' : 'assistant')
    var label = document.createElement('div')
    label.className = 'chat-role'
    label.textContent = role === 'user' ? 'You' : 'ASTA'
    var body = document.createElement('div')
    body.className = 'chat-text'
    item.appendChild(label)
    item.appendChild(body)
    messages.appendChild(item)
    messages.scrollTop = messages.scrollHeight
    if (role === 'assistant') renderText(body, value)
    else body.textContent = value
    messages.scrollTop = messages.scrollHeight
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
      empty.textContent = 'A new conversation. What would you like to work on?'
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

      var item = document.createElement('div')
      item.className = 'chat-recent-item' + (session.id === currentSessionId ? ' active' : '')
      item.dataset.sessionId = session.id
      item.title = session.preview || session.title || 'Conversation'

      var selectButton = document.createElement('button')
      selectButton.type = 'button'
      selectButton.className = 'chat-recent-select'

      var title = document.createElement('span')
      title.className = 'chat-recent-title'
      title.textContent = session.title || 'New conversation'
      var meta = document.createElement('span')
      meta.className = 'chat-recent-meta'
      var count = Number(session.message_count || 0)
      meta.textContent = count ? count + (count === 1 ? ' message' : ' messages') : 'New'
      selectButton.appendChild(title)
      selectButton.appendChild(meta)
      selectButton.addEventListener('click', function () {
        var id = this.closest('.chat-recent-item') && this.closest('.chat-recent-item').dataset.sessionId
        if (bridge && bridge.selectChatSession && id) bridge.selectChatSession(id)
      })

      var deleteButton = document.createElement('button')
      deleteButton.type = 'button'
      deleteButton.className = 'chat-recent-delete'
      deleteButton.textContent = '×'
      deleteButton.title = 'Delete conversation'
      deleteButton.setAttribute('aria-label', 'Delete conversation')
      deleteButton.addEventListener('click', function (event) {
        event.preventDefault()
        event.stopPropagation()
        var id = this.closest('.chat-recent-item') && this.closest('.chat-recent-item').dataset.sessionId
        if (!id || !bridge || !bridge.deleteChatSession) return
        var target = this.closest('.chat-recent-item')
        var titleText = target ? target.querySelector('.chat-recent-title') : null
        var label = titleText && titleText.textContent ? titleText.textContent : 'this conversation'
        if (!window.confirm('Delete "' + label + '"?\n\nThis permanently removes its local chat history.')) return
        bridge.deleteChatSession(id)
      })

      item.appendChild(selectButton)
      item.appendChild(deleteButton)
      recentList.appendChild(item)
    }
  }

  function setConversationMode (enabled) {
    conversationMode = !!enabled
    if (!conversationButton) return
    conversationButton.classList.toggle('active', conversationMode)
    conversationButton.setAttribute('aria-pressed', conversationMode ? 'true' : 'false')
    conversationButton.title = conversationMode ? 'Disable conversation mode' : 'Enable conversation mode'
    conversationButton.setAttribute('aria-label', conversationMode ? 'Disable conversation mode' : 'Enable conversation mode')
  }

  function toggleConversationMode () {
    setConversationMode(!conversationMode)
    if (bridge && bridge.setConversationMode) bridge.setConversationMode(conversationMode)
  }

  function setRecents (enabled) {
    recentVisible = !!enabled
    chatPanel.classList.toggle('chat-show-history', recentVisible)
    if (historyButton) {
      historyButton.classList.toggle('active', recentVisible)
      historyButton.setAttribute('aria-pressed', recentVisible ? 'true' : 'false')
      historyButton.setAttribute('aria-expanded', recentVisible ? 'true' : 'false')
      document.getElementById('chatRecent').setAttribute('aria-hidden', recentVisible ? 'false' : 'true')
    }
  }

  function toggleRecents () { setRecents(!recentVisible) }

  function setFullscreen (enabled) {
    if (window.AstaViews) window.AstaViews.show('chat', false)
    chatPanel.classList.toggle('chat-fullscreen', enabled)
    document.body.classList.toggle('text-chat-fullscreen', enabled)
    if (fullscreenButton) {
      fullscreenButton.title = enabled ? 'Exit fullscreen chat' : 'Expand conversation'
      fullscreenButton.setAttribute('aria-label', enabled ? 'Exit fullscreen chat' : 'Expand conversation')
    }
    if (fullscreenLabel) fullscreenLabel.textContent = enabled ? 'Exit focus' : 'Focus'
    setRecents(enabled ? false : window.innerWidth > 900)
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

  if (conversationButton) conversationButton.addEventListener('click', function (event) {
    event.preventDefault()
    event.stopPropagation()
    toggleConversationMode()
  })

  if (fullscreenButton) fullscreenButton.addEventListener('click', function (event) {
    event.preventDefault()
    event.stopPropagation()
    setFullscreen(!chatPanel.classList.contains('chat-fullscreen'))
  })

  window.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && recentVisible && !chatPanel.classList.contains('chat-fullscreen')) {
      setRecents(false); historyButton.focus(); return
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); input.focus() }
    if (event.key === 'Escape' && window.AstaViews && window.AstaViews.current() === 'chat' && !chatPanel.classList.contains('chat-fullscreen')) {
      event.preventDefault(); window.AstaViews.show('home'); return
    }
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

  if (bridge && bridge.onHudConversationMode) {
    bridge.onHudConversationMode(function (enabled) {
      try { setConversationMode(enabled) } catch (error) { console.warn('[ASTA HUD] Invalid conversation mode:', error) }
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

  window.addEventListener('asta:view-change', function (event) {
    if (!event.detail) return
    setRecents(event.detail.view === 'chat' && window.innerWidth > 900)
    if (event.detail.view === 'home') {
      chatPanel.classList.remove('chat-fullscreen'); document.body.classList.remove('text-chat-fullscreen')
      if (fullscreenLabel) fullscreenLabel.textContent = 'Focus'
    }
  })
  setConversationMode(false)
  setRecents(false)
})()
