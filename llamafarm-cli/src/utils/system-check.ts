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

// Main function when run directly
if (require.main === module) {
  (async () => {
    displaySystemRequirements();
    console.log(chalk.cyan('\n📊 Current System Info:\n'));
    const info = await getSystemInfo();
    console.log(chalk.gray(`   Platform: ${info.platform} ${info.arch}`));
    console.log(chalk.gray(`   Node.js: ${info.nodeVersion}`));
    console.log(chalk.gray(`   CPU Cores: ${info.cpuCores}`));
    console.log(chalk.gray(`   Total Memory: ${info.totalMemory}GB`));
    console.log(chalk.gray(`   Free Memory: ${info.freeMemory}GB`));
    console.log(chalk.gray(`   Disk Space: ${info.diskSpace}GB`));
    console.log(chalk.gray(`   CUDA: ${info.hasCuda ? 'Available' : 'Not found'}`));
    console.log(chalk.gray(`   Metal: ${info.hasMetal ? 'Available' : 'Not available'}`));
  })();
}