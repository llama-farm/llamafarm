// demo/demo.ts - Working demo implementation
import express from 'express';
import { WebSocketServer } from 'ws';
import { ChromaClient } from 'chromadb';
import * as path from 'path';
import * as fs from 'fs-extra';

interface DemoConfig {
  model: string;
  port: number;
  enableRAG: boolean;
}

export class LlamaFarmDemo {
  private app: express.Application;
  private wss: WebSocketServer;
  private chromaClient?: ChromaClient;
  private config: DemoConfig;

  constructor(config: DemoConfig) {
    this.config = config;
    this.app = express();
    this.wss = new WebSocketServer({ port: config.port + 1 });
    
    if (config.enableRAG) {
      this.chromaClient = new ChromaClient({
        path: './demo_chroma_db'
      });
    }
  }

  async initialize() {
    // Set up Express routes
    this.app.use(express.static(path.join(__dirname, 'public')));
    this.app.use(express.json());

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        model: this.config.model,
        features: {
          rag: this.config.enableRAG,
          websocket: true
        }
      });
    });

    // Model info endpoint
    this.app.get('/api/model', (req, res) => {
      res.json({
        name: this.config.model,
        status: 'ready',
        capabilities: [
          'chat',
          'completion',
          this.config.enableRAG ? 'rag' : null
        ].filter(Boolean)
      });
    });

    // Simple chat endpoint (for demonstration)
    this.app.post('/api/chat', async (req, res) => {
      const { message } = req.body;
      
      // Simulate model response
      const response = await this.generateResponse(message);
      
      res.json({
        response,
        model: this.config.model,
        timestamp: new Date().toISOString()
      });
    });

    // WebSocket handling
    this.wss.on('connection', (ws) => {
      console.log('🔌 New WebSocket connection');

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          
          if (message.type === 'chat') {
            // Simulate thinking...
            ws.send(JSON.stringify({
              type: 'status',
              content: 'thinking'
            }));

            const response = await this.generateResponse(message.content);
            
            ws.send(JSON.stringify({
              type: 'response',
              content: response
            }));
          }
        } catch (error) {
          ws.send(JSON.stringify({
            type: 'error',
            content: 'Failed to process message'
          }));
        }
      });

      ws.on('close', () => {
        console.log('👋 WebSocket connection closed');
      });
    });

    // Start the server
    this.app.listen(this.config.port, () => {
      console.log(`🌾 LlamaFarm demo server running on port ${this.config.port}`);
      console.log(`🔌 WebSocket server on port ${this.config.port + 1}`);
      console.log(`📦 Model: ${this.config.model}`);
      console.log(`🌐 Open http://localhost:${this.config.port} in your browser`);
    });
  }

  private async generateResponse(message: string): Promise<string> {
    // In a real implementation, this would call Ollama or another model
    // For demo purposes, we'll return contextual responses
    
    const responses: { [key: string]: string } = {
      'hello': `Hello! I'm your local AI assistant running ${this.config.model}. How can I help you today?`,
      'help': 'I can help you with various tasks like answering questions, writing code, or having a conversation. What would you like to know?',
      'rag': this.config.enableRAG 
        ? 'I have RAG (Retrieval Augmented Generation) enabled, so I can search through documents to provide more accurate answers!'
        : 'RAG is not enabled for this deployment. You can enable it with the --rag flag when planting.',
      'default': `I understand you said: "${message}". This is a demo response from ${this.config.model}. In a full implementation, I would process this with the actual model.`
    };

    // Simple keyword matching for demo
    const lowercaseMessage = message.toLowerCase();
    for (const [keyword, response] of Object.entries(responses)) {
      if (lowercaseMessage.includes(keyword)) {
        return response;
      }
    }

    return responses.default;
  }

  async seedVectorDatabase() {
    if (!this.chromaClient || !this.config.enableRAG) {
      return;
    }

    console.log('🌱 Seeding vector database...');

    try {
      // Create a collection
      const collection = await this.chromaClient.createCollection({
        name: 'demo_documents',
        metadata: { description: 'Demo documents for LlamaFarm' }
      });

      // Add some sample documents
      const documents = [
        {
          id: '1',
          document: 'LlamaFarm is a tool for deploying AI models locally. It packages models, agents, and databases into single binaries.',
          metadata: { source: 'documentation', type: 'overview' }
        },
        {
          id: '2',
          document: 'The plant command creates a deployable package with your chosen model, agent configuration, and optional RAG pipeline.',
          metadata: { source: 'documentation', type: 'command' }
        },
        {
          id: '3',
          document: 'RAG (Retrieval Augmented Generation) allows models to search through documents to provide more accurate, contextual responses.',
          metadata: { source: 'documentation', type: 'feature' }
        }
      ];

      await collection.add({
        ids: documents.map(d => d.id),
        documents: documents.map(d => d.document),
        metadatas: documents.map(d => d.metadata)
      });

      console.log('✅ Vector database seeded with sample documents');
    } catch (error) {
      console.error('Failed to seed vector database:', error);
    }
  }
}

// Demo HTML UI
export function createDemoUI(port: number): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaFarm Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c12 0%, #2e7d32 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            color: #333;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #2e7d32;
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 14px;
        }
        
        .demo-badge {
            display: inline-block;
            background: #ff6b6b;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
            font-weight: bold;
        }
        
        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            width: 100%;
            margin: 20px auto;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }
        
        .info-bar {
            background: #f5f5f5;
            padding: 15px 20px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
            color: #666;
        }
        
        .info-bar span {
            margin-right: 20px;
        }
        
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fafafa;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            text-align: right;
        }
        
        .message-content {
            display: inline-block;
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            background: white;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .message.user .message-content {
            background: #2e7d32;
            color: white;
        }
        
        .message.ai .message-content {
            background: white;
            border: 1px solid #e0e0e0;
        }
        
        .typing-indicator {
            display: none;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
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
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        input:focus {
            border-color: #2e7d32;
        }
        
        button {
            background: #2e7d32;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            font-weight: 500;
        }
        
        button:hover {
            background: #1b5e20;
        }
        
        button:active {
            transform: scale(0.98);
        }
        
        .welcome-message {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .welcome-message h2 {
            color: #2e7d32;
            margin-bottom: 20px;
        }
        
        .demo-hints {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .demo-hints h3 {
            color: #2e7d32;
            margin-bottom: 10px;
        }
        
        .demo-hints ul {
            list-style: none;
            padding-left: 0;
        }
        
        .demo-hints li {
            padding: 5px 0;
            color: #555;
        }
        
        .demo-hints li:before {
            content: "🌱 ";
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 LlamaFarm <span class="demo-badge">DEMO</span></h1>
        <p>Local AI Deployment Made Simple</p>
    </div>
    
    <div class="container">
        <div class="info-bar">
            <span><span class="status-dot"></span>Connected</span>
            <span>Model: Demo</span>
            <span>Port: ${port}</span>
        </div>
        
        <div class="chat-container" id="chat">
            <div class="welcome-message">
                <h2>Welcome to LlamaFarm Demo! 🦙</h2>
                <p>This is a demonstration of how LlamaFarm packages and deploys AI models locally.</p>
                
                <div class="demo-hints">
                    <h3>Try these messages:</h3>
                    <ul>
                        <li>"Hello" - Get a friendly greeting</li>
                        <li>"Help" - Learn what I can do</li>
                        <li>"Tell me about RAG" - Learn about RAG features</li>
                        <li>Or ask anything else!</li>
                    </ul>
                </div>
                
                <p style="margin-top: 20px; font-size: 14px; color: #999;">
                    Note: This is a demo with simulated responses. In a real deployment, 
                    responses would come from your chosen AI model.
                </p>
            </div>
        </div>
        
        <div class="typing-indicator" id="typing">
            <span></span>
            <span></span>
            <span></span>
        </div>
        
        <div class="input-container">
            <div class="input-wrapper">
                <input 
                    type="text" 
                    id="input" 
                    placeholder="Type your message..." 
                    autofocus
                >
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:${port + 1}');
        const chatContainer = document.getElementById('chat');
        const input = document.getElementById('input');
        const typing = document.getElementById('typing');
        let isFirstMessage = true;

        ws.onopen = () => {
            console.log('Connected to LlamaFarm demo');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'status' && data.content === 'thinking') {
                showTyping();
            } else if (data.type === 'response') {
                hideTyping();
                addMessage(data.content, 'ai');
            } else if (data.type === 'error') {
                hideTyping();
                addMessage('Error: ' + data.content, 'ai');
            }
        };

        function sendMessage() {
            const message = input.value.trim();
            if (!message) return;

            if (isFirstMessage) {
                // Clear welcome message on first interaction
                chatContainer.innerHTML = '';
                isFirstMessage = false;
            }

            addMessage(message, 'user');
            
            ws.send(JSON.stringify({
                type: 'chat',
                content: message
            }));

            input.value = '';
            showTyping();
        }

        function addMessage(content, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${sender}\`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showTyping() {
            typing.style.display = 'block';
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function hideTyping() {
            typing.style.display = 'none';
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Add some demo-specific features
        setTimeout(() => {
            console.log('💡 Tip: This is a demo. In production, responses would come from your actual AI model.');
        }, 3000);
    </script>
</body>
</html>`;
}

// Run the demo if called directly
if (require.main === module) {
  const demo = new LlamaFarmDemo({
    model: 'demo-model',
    port: 8080,
    enableRAG: true
  });
  
  demo.initialize();
  demo.seedVectorDatabase();
}