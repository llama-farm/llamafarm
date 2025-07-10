
  // src/templates/agent-server.ts
  export function createAgentServer(config: any): string {
    return `const express = require('express');
const WebSocket = require('ws');
const { Ollama } = require('ollama');
const path = require('path');

const app = express();
const port = ${config.port || 8080};

// Initialize Ollama client
const ollama = new Ollama();

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Chat API endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    
    const response = await ollama.generate({
      model: '${config.model}',
      prompt: message,
      system: '${config.systemPrompt || 'You are a helpful AI assistant.'}',
      stream: false
    });
    
    res.json({ response: response.response });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    model: '${config.model}',
    agent: '${config.name}'
  });
});

// Start server
const server = app.listen(port, () => {
  console.log(\`🌾 LlamaFarm agent server running on port \${port}\`);
});

// WebSocket support
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('New WebSocket connection');
  
  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      
      if (data.type === 'message') {
        const response = await ollama.generate({
          model: '${config.model}',
          prompt: data.content,
          stream: false
        });
        
        ws.send(JSON.stringify({
          type: 'response',
          content: response.response
        }));
      }
    } catch (error) {
      console.error('WebSocket error:', error);
      ws.send(JSON.stringify({
        type: 'error',
        content: 'Failed to process message'
      }));
    }
  });
});`;
  }
  
 