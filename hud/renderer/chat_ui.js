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

  if (!form || !input || !messages) return

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
      index = end
      messages.scrollTop = messages.scrollHeight
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

    if (animate && role === 'assistant' && document.body.classList.contains('text-chat-fullscreen')) {
      progressiveReveal(body, value)
    } else {
      body.textContent = value
    }
  }

  function loadHistory (history) {
    var items = Array.isArray(history) ? history : []
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
      empty.textContent = 'TEXT CHANNEL READY'
      messages.appendChild(empty)
    }

    messages.scrollTop = messages.scrollHeight
  }

  function setFullscreen (enabled) {
    if (!chatPanel) return

    chatPanel.classList.toggle('chat-fullscreen', enabled)
    document.body.classList.toggle('text-chat-fullscreen', enabled)

    if (fullscreenButton) {
      fullscreenButton.title = enabled ? 'Exit fullscreen chat' : 'Expand conversation'
      fullscreenButton.setAttribute('aria-label', enabled ? 'Exit fullscreen chat' : 'Expand conversation')
    }

    if (fullscreenLabel) fullscreenLabel.textContent = enabled ? 'EXIT' : 'FULLSCREEN'

    window.requestAnimationFrame(function () {
      window.dispatchEvent(new Event('resize'))
      messages.scrollTop = messages.scrollHeight
      if (enabled) input.focus()
    })
  }

  function toggleFullscreen () {
    var enabled = chatPanel && chatPanel.classList.contains('chat-fullscreen')
    setFullscreen(!enabled)
  }

  if (fullscreenButton) {
    fullscreenButton.addEventListener('click', function (event) {
      event.preventDefault()
      event.stopPropagation()
      toggleFullscreen()
    })
  }

  window.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && chatPanel && chatPanel.classList.contains('chat-fullscreen')) {
      event.preventDefault()
      setFullscreen(false)
      input.focus()
    }
  })

  form.addEventListener('submit', function (event) {
    event.preventDefault()
    var text = input.value.trim()
    if (!text) return

    if (!bridge || !bridge.sendTextMessage) {
      console.warn('[ASTA HUD] Text input bridge unavailable')
      return
    }

    input.value = ''
    input.focus()
    bridge.sendTextMessage(text)
  })

  if (bridge && bridge.onHudChatHistory) {
    bridge.onHudChatHistory(function (history) {
      try {
        loadHistory(history)
      } catch (error) {
        console.warn('[ASTA HUD] Invalid chat history:', error)
      }
    })
  }

  if (bridge && bridge.onHudChat) {
    bridge.onHudChat(function (message) {
      try {
        if (!message || !message.text) return
        appendMessage(message.role === 'user' ? 'user' : 'assistant', message.text, true)
      } catch (error) {
        console.warn('[ASTA HUD] Invalid chat message:', error)
      }
    })
  }
})()
