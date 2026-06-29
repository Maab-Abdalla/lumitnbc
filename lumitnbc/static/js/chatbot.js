// Chatbot Functions
function toggleChatbot() {
  const chatWindow = document.getElementById('chatbot-window');
  const chatToggle = document.getElementById('chatbot-toggle');
  
  if (chatWindow.style.display === 'flex') {
    chatWindow.style.display = 'none';
    chatToggle.style.display = 'flex';
  } else {
    chatWindow.style.display = 'flex';
    chatToggle.style.display = 'none';
  }
}

// Conversation history for context (kept short server-side too)
var chatHistory = [];

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

function sendMessage(message) {
  const messagesContainer = document.getElementById('chatbot-messages');

  // Add user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chatbot-message user-message';
  userMsg.innerHTML = `
    <div class="message-content">
      <p>${escapeHtml(message)}</p>
    </div>
    <div class="message-avatar">You</div>
  `;
  messagesContainer.appendChild(userMsg);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Typing indicator
  const typing = document.createElement('div');
  typing.className = 'chatbot-message bot-message';
  typing.id = 'chatbot-typing';
  typing.innerHTML = `
    <div class="message-avatar">Li</div>
    <div class="message-content"><p class="chat-typing"><span></span><span></span><span></span></p></div>
  `;
  messagesContainer.appendChild(typing);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  chatHistory.push({ role: 'user', content: message });

  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message, history: chatHistory.slice(-6) })
  })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var t = document.getElementById('chatbot-typing');
      if (t) t.remove();
      var reply = (d && d.reply) ? d.reply : "Sorry, I couldn't respond just now. Please try again.";
      chatHistory.push({ role: 'assistant', content: reply });
      const botMsg = document.createElement('div');
      botMsg.className = 'chatbot-message bot-message';
      botMsg.innerHTML = `
        <div class="message-avatar">Li</div>
        <div class="message-content"><p>${escapeHtml(reply)}</p></div>
      `;
      messagesContainer.appendChild(botMsg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch(function () {
      var t = document.getElementById('chatbot-typing');
      if (t) t.remove();
      const botMsg = document.createElement('div');
      botMsg.className = 'chatbot-message bot-message';
      botMsg.innerHTML = `
        <div class="message-avatar">Li</div>
        <div class="message-content"><p>Sorry, I'm having trouble connecting. Please try again.</p></div>
      `;
      messagesContainer.appendChild(botMsg);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

function sendUserMessage() {
  const input = document.getElementById('chatbot-input-field');
  const message = input.value.trim();
  
  if (message) {
    sendMessage(message);
    input.value = '';
  }
}

function handleKeyPress(event) {
  if (event.key === 'Enter') {
    sendUserMessage();
  }
}


// Note: button actions are wired via inline onclick handlers in
// chatbot_widget.html (toggleChatbot, sendMessage, sendUserMessage,
// handleKeyPress). Do NOT also addEventListener('click', toggleChatbot)
// here, as that would fire the toggle twice per click and cancel itself out.
