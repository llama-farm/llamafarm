// demo/run-demo.ts
import { LlamaFarmDemo, createDemoUI } from './demo';
import * as fs from 'fs-extra';
import * as path from 'path';
import chalk from 'chalk';
import figlet from 'figlet';

async function runDemo() {
  console.clear();
  console.log(chalk.green(figlet.textSync('LlamaFarm', { horizontalLayout: 'full' })));
  console.log(chalk.yellow('🌾 Demo Mode - See how LlamaFarm works! 🦙\n'));

  // Create demo public directory
  const publicDir = path.join(__dirname, 'public');
  await fs.ensureDir(publicDir);

  // Write demo UI
  const demoUI = createDemoUI(8080);
  await fs.writeFile(path.join(publicDir, 'index.html'), demoUI);

  // Initialize demo
  const demo = new LlamaFarmDemo({
    model: 'demo-llama3-8b',
    port: 8080,
    enableRAG: true
  });

  console.log(chalk.cyan('📋 Demo Configuration:'));
  console.log(chalk.gray('   Model: demo-llama3-8b (simulated)'));
  console.log(chalk.gray('   RAG: Enabled'));
  console.log(chalk.gray('   Vector DB: ChromaDB'));
  console.log(chalk.gray('   Port: 8080\n'));

  await demo.initialize();
  await demo.seedVectorDatabase();

  console.log(chalk.green('\n✨ Demo is running!'));
  console.log(chalk.white('\n🌐 Open http://localhost:8080 in your browser'));
  console.log(chalk.gray('\nℹ️  This demo simulates LlamaFarm functionality.'));
  console.log(chalk.gray('    In production, responses come from real AI models.\n'));
  console.log(chalk.yellow('Press Ctrl+C to stop the demo\n'));
}

// Run if called directly
if (require.main === module) {
  runDemo().catch(error => {
    console.error(chalk.red('Demo failed to start:'), error);
    process.exit(1);
  });
}

export { runDemo };

// scripts/demo.js (for package.json)
#!/usr/bin/env node
const { runDemo } = require('../dist/demo/run-demo');
runDemo();

// src/utils/system-check.ts
import * as os from 'os';
import * as fs from 'fs-extra';
import chalk from 'chalk';

export interface SystemInfo {
  platform: string;
  arch: string;
  totalMemory: number;
  freeMemory: number;
  cpuCores: number;
  nodeVersion: string;
  hasCuda: boolean;
  hasMetal: boolean;
  diskSpace: number;
}

export async function getSystemInfo(): Promise<SystemInfo> {
  const totalMemory = os.totalmem();
  const freeMemory = os.freemem();
  
  // Check for GPU support
  const hasCuda = await checkCudaSupport();
  const hasMetal = process.platform === 'darwin' && process.arch === 'arm64';
  
  // Get disk space (simplified)
  const diskSpace = await getDiskSpace();
  
  return {
    platform: os.platform(),
    arch: os.arch(),
    totalMemory: Math.round(totalMemory / 1024 / 1024 / 1024), // GB
    freeMemory: Math.round(freeMemory / 1024 / 1024 / 1024), // GB
    cpuCores: os.cpus().length,
    nodeVersion: process.version,
    hasCuda,
    hasMetal,
    diskSpace
  };
}

async function checkCudaSupport(): Promise<boolean> {
  // Simplified CUDA check
  if (process.platform === 'win32') {
    return await fs.pathExists('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA');
  } else if (process.platform === 'linux') {
    return await fs.pathExists('/usr/local/cuda');
  }
  return false;
}

async function getDiskSpace(): Promise<number> {
  // This is a simplified implementation
  // In production, use a proper disk space library
  return 100; // GB placeholder
}

export function displaySystemRequirements(): void {
  console.log(chalk.cyan('\n📋 System Requirements:\n'));
  console.log(chalk.gray('   Minimum:'));
  console.log(chalk.gray('   • Node.js 18+'));
  console.log(chalk.gray('   • 4GB RAM'));
  console.log(chalk.gray('   • 10GB free disk space'));
  console.log(chalk.gray('\n   Recommended:'));
  console.log(chalk.gray('   • 8GB+ RAM'));
  console.log(chalk.gray('   • 50GB+ free disk space'));
  console.log(chalk.gray('   • GPU with 4GB+ VRAM (optional)'));
}

// scripts/postinstall.js
const chalk = require('chalk');
const figlet = require('figlet');

console.log(chalk.green(figlet.textSync('LlamaFarm', { horizontalLayout: 'full' })));
console.log(chalk.yellow('\n🌾 Thank you for installing LlamaFarm! 🦙\n'));
console.log(chalk.white('Quick Start:'));
console.log(chalk.gray('  1. Run "llamafarm till" to initialize'));
console.log(chalk.gray('  2. Run "llamafarm plant llama3-8b" to plant your first model'));
console.log(chalk.gray('  3. Run "llamafarm demo" to see a live demo\n'));
console.log(chalk.cyan('📚 Full documentation: https://github.com/llamafarm/llamafarm-cli'));
console.log(chalk.cyan('💬 Join our Discord: https://discord.gg/llamafarm\n'));

// Add to package.json scripts:
{
  "scripts": {
    // ... existing scripts ...
    "demo": "node scripts/demo.js",
    "postinstall": "node scripts/postinstall.js",
    "system-check": "node dist/utils/system-check.js"
  }
}