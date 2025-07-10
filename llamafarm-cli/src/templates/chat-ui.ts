// src/templates/chat-ui.ts
export function createChatUI(model: string, agent: string, port: number): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaFarm - ${model}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            background: #2e7d32;
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            box-sizing: border-box;
        }
        .message {
            margin-bottom: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        .message.user {
            justify-content: flex-end;
        }
        .message-content {
            max-width: 70%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message.user .message-content {
            background: #2e7d32;
            color: white;
        }
        .input-container {
            padding: 1rem;
            background: white;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
        }
        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            gap: 0.5rem;
        }
        input {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            font-size: 1rem;
        }
        button {
            padding: 0.75rem 1.5rem;
            background: #2e7d32;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
        }
        button:hover {
            background: #1b5e20;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 LlamaFarm Chat</h1>
        <div class="status">
            Model: ${model} | Agent: ${agent} | Port: ${port}
        </div>
    </div>
    <div class="chat-container" id="chat"></div>
    <div class="input-container">
        <div class="input-wrapper">
            <input type="text" id="input" placeholder="Type your message..." autofocus>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chat');
        const input = document.getElementById('input');
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'assistant');
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                addMessage(data.response);
            } catch (error) {
                addMessage('Error: Could not connect to the agent server.');
            }
        }
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Welcome message
        addMessage('Welcome to LlamaFarm! I\\'m running locally on your device. How can I help you today?');
    </script>
</body>
</html>`;
}
  