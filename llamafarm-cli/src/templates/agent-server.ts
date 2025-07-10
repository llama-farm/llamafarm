
  // src/templates/agent-server.ts
  export function createAgentServer(config: any): string {
    return `const express = require('express');
  const WebSocket = require('ws');
  const { Ollama } = require('ollama');
  
  const app = express();
  const port = process.env.PORT || ${config.port || 3000};
  
  // Initialize Ollama
  const ollama = new Ollama({
    host: 'http://localhost:11434'
  });
  
  // Agent configuration
  const agentConfig = ${JSON.stringify(config, null, 2)};
  
  // WebSocket server
  const wss = new WebSocket.Server({ port: port + 1 });
  
  wss.on('connection', (ws) => {
    console.log('🤝 New client connected');
  
    ws.on('message', async (message) => {
      try {
        const data = JSON.parse(message);
        
        if (data.type === 'message') {
          console.log('📨 Received:', data.content);
          
          // Generate response using Ollama
          const response = await ollama.chat({
            model: agentConfig.model,
            messages: [
              {
                role: 'system',
                content: agentConfig.systemPrompt
              },
              {
                role: 'user',
                content: data.content
              }
            ],
            stream: false
          });
  
          // Send response back
          ws.send(JSON.stringify({
            type: 'response',
            content: response.message.content
          }));
        }
      } catch (error) {
        console.error('❌ Error:', error);
        ws.send(JSON.stringify({
          type: 'error',
          content: 'Sorry, I encountered an error processing your request.'
        }));
      }
    });
  
    ws.on('close', () => {
      console.log('👋 Client disconnected');
    });
  });
  
  // HTTP endpoints
  app.get('/health', (req, res) => {
    res.json({ status: 'healthy', agent: agentConfig.name });
  });
  
  app.get('/config', (req, res) => {
    res.json(agentConfig);
  });
  
  // Start server
  app.listen(port, () => {
    console.log(\`🌾 LlamaFarm Agent Server running on port \${port}\`);
    console.log(\`🔌 WebSocket server on port \${port + 1}\`);
    console.log(\`🤖 Agent: \${agentConfig.name}\`);
    console.log(\`📦 Model: \${agentConfig.model}\`);
  });`;
  }
  
 