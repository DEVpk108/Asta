'use strict'

;(function () {
  var bridge = window.asta || null
  var form = document.getElementById('chatForm')
  var input = document.getElementById('chatInput')
  var messages = document.getElementById('chatMessages')
  var empty = document.getElementById('chatEmpty')

  if (!form || !input || !messages) return

  function removeEmpty () {
    if (empty && empty.parentNode) empty.parentNode.removeChild(empty)
    empty = null
  }

  function appendMessage (role, text) {
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
    body.textContent = value

    item.appendChild(label)
    item.appendChild(body)
    messages.appendChild(item)
    messages.scrollTop = messages.scrollHeight
  }

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

  if (bridge && bridge.onHudChat) {
    bridge.onHudChat(function (message) {
      try {
        if (!message || !message.text) return
        appendMessage(message.role === 'user' ? 'user' : 'assistant', message.text)
      } catch (error) {
        console.warn('[ASTA HUD] Invalid chat message:', error)
      }
    })
  }
})()
