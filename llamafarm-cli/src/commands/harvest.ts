// src/commands/harvest.ts - Updated for binary deployment
import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs-extra';
import * as path from 'path';
import * as os from 'os';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import fetch from 'node-fetch';

const execAsync = promisify(exec);

interface HarvestOptions {
  output?: string;
  verify?: boolean;
  run?: boolean;
  daemon?: boolean;
  port?: number;
}

export async function harvestCommand(source: string, options: HarvestOptions) {
  console.log(chalk.green(`\n🌾 Harvesting LlamaFarm deployment...\n`));
  
  const spinner = ora();
  
  try {
    // Step 1: Determine source type
    let binaryPath: string;
    
    if (source.startsWith('http://') || source.startsWith('https://')) {
      // Download from URL
      spinner.start(`Downloading from ${source}...`);
      binaryPath = await downloadBinary(source, options.output);
      spinner.succeed('Binary downloaded');
    } else if (source.endsWith('.bin') || source.endsWith('.run') || source.endsWith('.exe')) {
      // Local binary file
      spinner.start('Locating binary...');
      binaryPath = path.resolve(source);
      
      if (!await fs.pathExists(binaryPath)) {
        throw new Error(`Binary not found: ${binaryPath}`);
      }
      spinner.succeed('Binary located');
    } else {
      // Assume it's a model name to fetch from registry
      spinner.start(`Looking up ${source} in model registry...`);
      const registryUrl = await lookupInRegistry(source);
      
      if (registryUrl) {
        binaryPath = await downloadBinary(registryUrl, options.output);
        spinner.succeed('Binary downloaded from registry');
      } else {
        throw new Error(`Cannot find ${source}. Provide a direct path or URL.`);
      }
    }
    
    // Step 2: Verify binary (if requested)
    if (options.verify) {
      spinner.start('Verifying binary integrity...');
      const isValid = await verifyBinary(binaryPath);
      
      if (!isValid) {
        throw new Error('Binary verification failed!');
      }
      spinner.succeed('Binary verified ✓');
    }
    
    // Step 3: Check system compatibility
    spinner.start('Checking system compatibility...');
    const compatibility = await checkCompatibility(binaryPath);
    
    if (!compatibility.compatible) {
      console.log(chalk.yellow(`\n⚠️  Compatibility warning: ${compatibility.message}`));
      const shouldContinue = await promptUser('Continue anyway?');
      if (!shouldContinue) {
        throw new Error('Deployment cancelled');
      }
    } else {
      spinner.succeed('System compatible');
    }
    
    // Step 4: Prepare deployment directory
    const deployDir = options.output || path.join(os.homedir(), '.llamafarm', 'deployments', 
      path.basename(binaryPath, path.extname(binaryPath)));
    
    spinner.start('Preparing deployment directory...');
    await fs.ensureDir(deployDir);
    
    // Copy binary to deployment directory
    const deployedBinary = path.join(deployDir, path.basename(binaryPath));
    await fs.copy(binaryPath, deployedBinary);
    
    // Make executable on Unix-like systems
    if (process.platform !== 'win32') {
      await fs.chmod(deployedBinary, '755');
    }
    
    spinner.succeed('Deployment prepared');
    
    // Step 5: Extract metadata from binary
    spinner.start('Reading deployment metadata...');
    const metadata = await extractMetadata(deployedBinary);
    
    if (metadata) {
      console.log(chalk.cyan('\n📋 Deployment Info:'));
      console.log(chalk.gray(`   Model: ${metadata.model || 'Unknown'}`));
      console.log(chalk.gray(`   Version: ${metadata.version || 'Unknown'}`));
      console.log(chalk.gray(`   Platform: ${metadata.platform || 'Unknown'}`));
      console.log(chalk.gray(`   Features: ${metadata.features?.join(', ') || 'Standard'}`));
    }
    
    spinner.succeed('Metadata extracted');
    
    // Step 6: Create management scripts
    spinner.start('Creating management scripts...');
    await createManagementScripts(deployDir, deployedBinary, metadata, options);
    spinner.succeed('Management scripts created');
    
    // Step 7: Run the binary (if requested)
    if (options.run) {
      console.log(chalk.green('\n🚀 Starting LlamaFarm...\n'));
      
      if (options.daemon) {
        // Run as daemon/background process
        await runAsDaemon(deployedBinary, deployDir, options.port);
        console.log(chalk.green('✅ LlamaFarm is running in the background'));
        console.log(chalk.gray(`   Logs: ${path.join(deployDir, 'llamafarm.log')}`));
        console.log(chalk.gray(`   Stop with: ${path.join(deployDir, 'stop.sh')}`));
      } else {
        // Run in foreground
        await runInForeground(deployedBinary, options.port);
      }
    } else {
      // Just show instructions
      console.log(chalk.green('\n✅ Harvest complete!'));
      console.log(chalk.yellow('\n🚀 To run your deployment:'));
      
      if (process.platform === 'win32') {
        console.log(chalk.white(`   cd ${deployDir}`));
        console.log(chalk.white(`   start.bat`));
      } else {
        console.log(chalk.white(`   cd ${deployDir}`));
        console.log(chalk.white(`   ./start.sh`));
      }
      
      console.log(chalk.gray(`\n   Or run directly: ${deployedBinary}`));
      console.log(chalk.gray(`   Run as daemon: ${deployedBinary} --daemon`));
      
      if (metadata?.port) {
        console.log(chalk.cyan(`\n🌐 Will be available at: http://localhost:${metadata.port}`));
      }
    }
    
  } catch (error) {
    spinner.fail('Harvest failed');
    console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
    process.exit(1);
  }
}

// Helper functions

async function downloadBinary(url: string, outputDir?: string): Promise<string> {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.statusText}`);
  }
  
  // Get filename from URL or headers
  const contentDisposition = response.headers.get('content-disposition');
  let filename = 'llamafarm-deployment.bin';
  
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="(.+)"/);
    if (match) filename = match[1];
  } else {
    filename = path.basename(url) || filename;
  }
  
  const downloadDir = outputDir || path.join(os.tmpdir(), 'llamafarm-downloads');
  await fs.ensureDir(downloadDir);
  
  const downloadPath = path.join(downloadDir, filename);
  const buffer = await response.buffer();
  await fs.writeFile(downloadPath, buffer);
  
  return downloadPath;
}

async function lookupInRegistry(modelName: string): Promise<string | null> {
  // In production, this would query a real registry
  // For now, return null to indicate not found
  console.log(chalk.gray(`   (Registry lookup not implemented yet)`));
  return null;
}

async function verifyBinary(binaryPath: string): Promise<boolean> {
  // Verify binary integrity
  // In production: check signatures, checksums, etc.
  
  try {
    const stats = await fs.stat(binaryPath);
    
    // Basic checks
    if (stats.size < 1000000) { // Less than 1MB is suspicious
      console.log(chalk.yellow('   Binary seems too small'));
      return false;
    }
    
    // Check if it's actually executable
    if (process.platform !== 'win32') {
      const { stdout } = await execAsync(`file "${binaryPath}"`);
      if (!stdout.includes('executable') && !stdout.includes('script')) {
        console.log(chalk.yellow('   File does not appear to be executable'));
        return false;
      }
    }
    
    return true;
  } catch (error) {
    return false;
  }
}

async function checkCompatibility(binaryPath: string): Promise<{compatible: boolean, message?: string}> {
  const filename = path.basename(binaryPath).toLowerCase();
  const platform = process.platform;
  const arch = process.arch;
  
  // Check platform compatibility
  if (platform === 'darwin' && !filename.includes('mac') && !filename.includes('darwin')) {
    return { compatible: false, message: 'This binary appears to be for a different platform' };
  }
  
  if (platform === 'linux' && !filename.includes('linux')) {
    return { compatible: false, message: 'This binary appears to be for a different platform' };
  }
  
  if (platform === 'win32' && !filename.includes('win') && !filename.endsWith('.exe')) {
    return { compatible: false, message: 'This binary appears to be for a different platform' };
  }
  
  // Check architecture for ARM devices
  if (arch === 'arm64' && !filename.includes('arm')) {
    return { compatible: false, message: 'This binary may not be optimized for ARM' };
  }
  
  return { compatible: true };
}

async function extractMetadata(binaryPath: string): Promise<any> {
  // Try to extract metadata from the binary
  // This could be embedded JSON, or extracted from the binary itself
  
  try {
    // Method 1: Check if binary has embedded metadata
    if (process.platform !== 'win32') {
      const { stdout } = await execAsync(`strings "${binaryPath}" | grep -A5 "LLAMAFARM_METADATA"`, {
        maxBuffer: 1024 * 1024
      });
      
      if (stdout) {
        const jsonMatch = stdout.match(/{[^}]+}/);
        if (jsonMatch) {
          return JSON.parse(jsonMatch[0]);
        }
      }
    }
    
    // Method 2: Parse from filename
    const filename = path.basename(binaryPath);
    const match = filename.match(/llamafarm-(.+)-(.+)-v(.+)\./);
    
    if (match) {
      return {
        model: match[1],
        platform: match[2],
        version: match[3],
        port: 8080
      };
    }
    
    // Default metadata
    return {
      model: 'unknown',
      platform: process.platform,
      version: '1.0.0',
      port: 8080
    };
    
  } catch (error) {
    return null;
  }
}

async function createManagementScripts(deployDir: string, binaryPath: string, metadata: any, options: HarvestOptions) {
  const binaryName = path.basename(binaryPath);
  
  if (process.platform === 'win32') {
    // Windows scripts
    const startScript = `@echo off
echo 🌾 Starting LlamaFarm...
cd /d "%~dp0"
start "LlamaFarm" "${binaryName}" %*
`;
    
    const stopScript = `@echo off
echo 🛑 Stopping LlamaFarm...
taskkill /IM "${binaryName}" /F
`;
    
    await fs.writeFile(path.join(deployDir, 'start.bat'), startScript);
    await fs.writeFile(path.join(deployDir, 'stop.bat'), stopScript);
    
  } else {
    // Unix-like scripts
    const startScript = `#!/bin/bash
# Start LlamaFarm

echo "🌾 Starting LlamaFarm..."
cd "$(dirname "$0")"

# Check if already running
if pgrep -f "${binaryName}" > /dev/null; then
    echo "⚠️  LlamaFarm is already running"
    exit 1
fi

# Start with options
./${binaryName} $@
`;
    
    const stopScript = `#!/bin/bash
# Stop LlamaFarm

echo "🛑 Stopping LlamaFarm..."
pkill -f "${binaryName}"
echo "✅ LlamaFarm stopped"
`;
    
    const daemonScript = `#!/bin/bash
# Run LlamaFarm as daemon

echo "🌾 Starting LlamaFarm daemon..."
cd "$(dirname "$0")"

nohup ./${binaryName} > llamafarm.log 2>&1 &
echo $! > llamafarm.pid
echo "✅ LlamaFarm daemon started (PID: $(cat llamafarm.pid))"
`;
    
    await fs.writeFile(path.join(deployDir, 'start.sh'), startScript);
    await fs.writeFile(path.join(deployDir, 'stop.sh'), stopScript);
    await fs.writeFile(path.join(deployDir, 'daemon.sh'), daemonScript);
    
    // Make scripts executable
    await fs.chmod(path.join(deployDir, 'start.sh'), '755');
    await fs.chmod(path.join(deployDir, 'stop.sh'), '755');
    await fs.chmod(path.join(deployDir, 'daemon.sh'), '755');
  }
  
  // Create status script
  const statusContent = metadata ? `
Deployment Status:
- Model: ${metadata.model}
- Version: ${metadata.version}
- Platform: ${metadata.platform}
- Default Port: ${metadata.port || 8080}
` : 'No metadata available';
  
  await fs.writeFile(path.join(deployDir, 'STATUS.txt'), statusContent);
}

async function runAsDaemon(binaryPath: string, deployDir: string, port?: number) {
  const env = { ...process.env };
  if (port) env.LLAMAFARM_PORT = port.toString();
  
  if (process.platform === 'win32') {
    // Windows: Use start command
    exec(`start /B "${binaryPath}"`, { cwd: deployDir, env });
  } else {
    // Unix-like: Use nohup
    const logPath = path.join(deployDir, 'llamafarm.log');
    const pidPath = path.join(deployDir, 'llamafarm.pid');
    
    const child = spawn('nohup', [binaryPath], {
      cwd: deployDir,
      env,
      detached: true,
      stdio: ['ignore', 'ignore', 'ignore']
    });
    
    child.unref();
    
    // Save PID
    if (child.pid) {
      await fs.writeFile(pidPath, child.pid.toString());
    }
  }
}

async function runInForeground(binaryPath: string, port?: number) {
  const env = { ...process.env };
  if (port) env.LLAMAFARM_PORT = port.toString();
  
  const child = spawn(binaryPath, [], {
    env,
    stdio: 'inherit'
  });
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    child.kill('SIGINT');
  });
  
  process.on('SIGTERM', () => {
    child.kill('SIGTERM');
  });
  
  return new Promise((resolve, reject) => {
    child.on('exit', (code) => {
      if (code === 0) {
        resolve(code);
      } else {
        reject(new Error(`Process exited with code ${code}`));
      }
    });
  });
}

async function promptUser(question: string): Promise<boolean> {
  // Simple prompt implementation
  const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  return new Promise((resolve) => {
    readline.question(`${question} (y/n) `, (answer: string) => {
      readline.close();
      resolve(answer.toLowerCase() === 'y');
    });
  });
}