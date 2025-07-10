// src/templates/chat-ui.ts
export function createChatUI(model: string, agent: string, port: number): string {
    return `<!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>LlamaFarm Chat - ${model}</title>
      <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
              min-height: 100vh;
              display: flex;
              align-items: center;
              justify-content: center;
          }
          .container {
              background: white;
              border-radius: 20px;
              box-shadow: 0 20px 60px rgba(0,0,0,0.1);
              width: 90%;
              max-width: 800px;
              height: 80vh;
              display: flex;
              flex-direction: column;
              overflow: hidden;
          }
          .header {
              background: #2E7D32;
              color: white;
              padding: 20px;
              text-align: center;
          }
          .header h1 {
              font-size: 24px;
              margin-bottom: 5px;
          }
          .header p {
              opacity: 0.9;
              font-size: 14px;
          }
          .chat-container {
              flex: 1;
              overflow-y: auto;
              padding: 20px;
              background: #f5f5f5;
          }
          .message {
              margin-bottom: 20px;
              display: flex;
              align-items: flex-start;
              gap: 10px;
          }
          .message.user {
              flex-direction: row-reverse;
          }
          .message-avatar {
              width: 40px;
              height: 40px;
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 20px;
              flex-shrink: 0;
          }
          .message.ai .message-avatar {
              background: #E8F5E9;
          }
          .message.user .message-avatar {
              background: #FFF3E0;
          }
          .message-content {
              background: white;
              padding: 15px 20px;
              border-radius: 15px;
              max-width: 70%;
              box-shadow: 0 2px 10px rgba(0,0,0,0.05);
          }
          .message.user .message-content {
              background: #2E7D32;
              color: white;
          }
          .input-container {
              padding: 20px;
              background: white;
              border-top: 1px solid #e0e0e0;
          }
          .input-wrapper {
              display: flex;
              gap: 10px;
          }
          input {
              flex: 1;
              padding: 15px 20px;
              border: 2px solid #e0e0e0;
              border-radius: 25px;
              font-size: 16px;
              outline: none;
              transition: border-color 0.3s;
          }
          input:focus {
              border-color: #2E7D32;
          }
          button {
              background: #2E7D32;
              color: white;
              border: none;
              padding: 15px 30px;
              border-radius: 25px;
              font-size: 16px;
              cursor: pointer;
              transition: background 0.3s;
          }
          button:hover {
              background: #1B5E20;
          }
          .status {
              text-align: center;
              padding: 10px;
              background: #E8F5E9;
              color: #2E7D32;
              font-size: 14px;
          }
          .loading {
              display: inline-block;
              width: 10px;
              height: 10px;
              background: #2E7D32;
              border-radius: 50%;
              margin: 0 2px;
              animation: bounce 1.4s infinite ease-in-out both;
          }
          .loading:nth-child(1) { animation-delay: -0.32s; }
          .loading:nth-child(2) { animation-delay: -0.16s; }
          @keyframes bounce {
              0%, 80%, 100% { transform: scale(0); }
              40% { transform: scale(1); }
          }
      </style>
  </head>
  <body>
      <div class="container">
          <div class="header">
              <h1>🌾 LlamaFarm Chat</h1>
              <p>Model: ${model} | Agent: ${agent}</p>
          </div>
          <div class="status" id="status">
              🟢 Connected to local model
          </div>
          <div class="chat-container" id="chat">
              <div class="message ai">
                  <div class="message-avatar">🦙</div>
                  <div class="message-content">
                      Hello! I'm your local AI assistant running ${model}. How can I help you today?
                  </div>
              </div>
          </div>
          <div class="input-container">
              <div class="input-wrapper">
                  <input type="text" id="input" placeholder="Type your message..." autofocus>
                  <button onclick="sendMessage()">Send</button>
              </div>
          </div>
      </div>
  
      <script>
          const ws = new WebSocket('ws://localhost:${port}/ws');
          const chatContainer = document.getElementById('chat');
          const input = document.getElementById('input');
          const status = document.getElementById('status');
  
          ws.onopen = () => {
              status.innerHTML = '🟢 Connected to local model';
          };
  
          ws.onclose = () => {
              status.innerHTML = '🔴 Disconnected';
          };
  
          ws.onmessage = (event) => {
              const data = JSON.parse(event.data);
              if (data.type === 'response') {
                  addMessage(data.content, 'ai');
              }
          };
  
          function sendMessage() {
              const message = input.value.trim();
              if (!message) return;
  
              addMessage(message, 'user');
              ws.send(JSON.stringify({ type: 'message', content: message }));
              input.value = '';
              
              // Show loading
              addLoading();
          }
  
          function addMessage(content, sender) {
              // Remove loading if exists
              const loading = document.querySelector('.message.loading');
              if (loading) loading.remove();
  
              const messageDiv = document.createElement('div');
              messageDiv.className = \`message \${sender}\`;
              messageDiv.innerHTML = \`
                  <div class="message-avatar">\${sender === 'ai' ? '🦙' : '👤'}</div>
                  <div class="message-content">\${content}</div>
              \`;
              chatContainer.appendChild(messageDiv);
              chatContainer.scrollTop = chatContainer.scrollHeight;
          }
  
          function addLoading() {
              const loadingDiv = document.createElement('div');
              loadingDiv.className = 'message ai loading';
              loadingDiv.innerHTML = \`
                  <div class="message-avatar">🦙</div>
                  <div class="message-content">
                      <span class="loading"></span>
                      <span class="loading"></span>
                      <span class="loading"></span>
                  </div>
              \`;
              chatContainer.appendChild(loadingDiv);
              chatContainer.scrollTop = chatContainer.scrollHeight;
          }
  
          input.addEventListener('keypress', (e) => {
              if (e.key === 'Enter') sendMessage();
          });
      </script>
  </body>
  </html>`;
  }
  