// demo/run-demo.ts
// Simple demo implementation without external dependencies
import * as fs from 'fs-extra';
import * as path from 'path';
import chalk from 'chalk';
import figlet from 'figlet';
import express from 'express';

async function runDemo() {
  console.clear();
  console.log(chalk.green(figlet.textSync('LlamaFarm', { horizontalLayout: 'full' })));
  console.log(chalk.yellow('🌾 Demo Mode - See how LlamaFarm works! 🦙\n'));

  // Create demo public directory
  const publicDir = path.join(__dirname, 'public');
  await fs.ensureDir(publicDir);

  // Create simple demo server
  const app = express();
  const port = 8080;
  
  app.use(express.json());
  app.use(express.static(publicDir));

  // Simple chat endpoint
  app.post('/api/chat', (req, res) => {
    const { message } = req.body;
    setTimeout(() => {
      res.json({
        response: `Demo response to: "${message}". In production, this would come from your locally running AI model!`
      });
    }, 500);
  });

  app.listen(port, () => {
    console.log(chalk.cyan('📋 Demo Configuration:'));
    console.log(chalk.gray('   Model: demo-llama3-8b (simulated)'));
    console.log(chalk.gray('   RAG: Enabled'));
    console.log(chalk.gray('   Vector DB: ChromaDB'));
    console.log(chalk.gray(`   Port: ${port}\n`));
    
    console.log(chalk.green('\n✨ Demo is running!'));
    console.log(chalk.white(`\n🌐 Open http://localhost:${port} in your browser`));
    console.log(chalk.gray('\nℹ️  This demo simulates LlamaFarm functionality.'));
    console.log(chalk.gray('    In production, responses come from real AI models.\n'));
    console.log(chalk.yellow('Press Ctrl+C to stop the demo\n'));
  });

  // Create a simple demo HTML file
  const demoHTML = `<!DOCTYPE html>
<html>
<head>
    <title>LlamaFarm Demo</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { background: #2e7d32; color: white; padding: 20px; text-align: center; margin: -20px -20px 20px; }
        .chat { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
        .input-group { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; }
        button { padding: 10px 20px; background: #2e7d32; color: white; border: none; cursor: pointer; }
        .message { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .user { background: #e8f5e9; text-align: right; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 LlamaFarm Demo</h1>
        <p>This simulates how LlamaFarm works locally</p>
    </div>
    <div class="chat" id="chat"></div>
    <div class="input-group">
        <input type="text" id="input" placeholder="Type a message...">
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        function addMessage(text, isUser) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message' + (isUser ? ' user' : '');
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('input');
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
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Error: Could not connect to demo server', false);
            }
        }
        
        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        addMessage('Welcome to the LlamaFarm demo! Try asking me something.', false);
    </script>
</body>
</html>`;

  await fs.writeFile(path.join(publicDir, 'index.html'), demoHTML);
}

// Run if called directly
if (require.main === module) {
  runDemo().catch(error => {
    console.error(chalk.red('Demo failed to start:'), error);
    process.exit(1);
  });
}

export { runDemo };


