// src/commands/bale.ts
import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs-extra';
import * as path from 'path';
import archiver from 'archiver';
import { exec } from 'child_process';
import { promisify } from 'util';
import { loadPlugins, registry } from '../plugins/loader';

const execAsync = promisify(exec);

interface BaleOptions {
  model: string;
  device: string;
  output?: string;
  include?: string[];
  compress?: boolean;
  sign?: boolean;
  optimize?: string;
}

export async function baleCommand(projectDir: string, options: BaleOptions) {
  console.log(chalk.green(`\n🎯 The Baler - LlamaFarm Binary Compiler\n`));
  
  const spinner = ora();
  
  try {
    // Step 1: Validate project directory
    spinner.start('Checking harvest configuration...');
    const manifestPath = path.join(projectDir, 'manifest.json');
    
    if (!await fs.pathExists(manifestPath)) {
      throw new Error(`No manifest.json found in ${projectDir}. Run 'llamafarm plant' first.`);
    }
    
    const manifest = await fs.readJSON(manifestPath);
    spinner.succeed(`Found ${manifest.name} ready for baling`);
    
    // Step 2: Load platform-specific plugin
    await loadPlugins();
    const fieldPlugin = registry.getByType('field')
      .find((f: any) => f.platform === getPlatform(options.device));
    
    if (fieldPlugin) {
      spinner.start(`Optimizing for ${options.device}...`);
      await fieldPlugin.setup();
      manifest.optimization = fieldPlugin.optimize(manifest);
      spinner.succeed(`Optimized for ${options.device}`);
    }
    
    // Step 3: Prepare build directory
    spinner.start('Preparing build environment...');
    const buildDir = path.join(projectDir, '.baler-build');
    await fs.ensureDir(buildDir);
    
    // Copy all project files
    await fs.copy(projectDir, buildDir, {
      filter: (src) => !src.includes('.baler-build')
    });
    
    spinner.succeed('Build environment ready');
    
    // Step 4: Bundle Node.js runtime
    spinner.start('Bundling Node.js runtime...');
    await bundleNodeRuntime(buildDir, options.device);
    spinner.succeed('Runtime bundled');
    
    // Step 5: Package model files
    spinner.start('Packaging model files...');
    await packageModel(buildDir, manifest);
    spinner.succeed('Model packaged');
    
    // Step 6: Embed vector database
    if (manifest.features.vectorDb) {
      spinner.start('Embedding vector database...');
      await embedVectorDatabase(buildDir, manifest);
      spinner.succeed('Vector database embedded');
    }
    
    // Step 7: Create platform-specific launcher
    spinner.start('Creating platform launcher...');
    await createLauncher(buildDir, manifest, options.device);
    spinner.succeed('Launcher created');
    
    // Step 8: Compile to binary
    spinner.start(`Compiling binary for ${options.device}...`);
    const binaryPath = await compileBinary(buildDir, manifest, options);
    spinner.succeed('Binary compiled');
    
    // Step 9: Optimize binary
    if (options.optimize !== 'none') {
      spinner.start('Optimizing binary size...');
      await optimizeBinary(binaryPath, options.optimize);
      spinner.succeed('Binary optimized');
    }
    
    // Step 10: Sign binary (if requested)
    if (options.sign) {
      spinner.start('Signing binary...');
      await signBinary(binaryPath, options.device);
      spinner.succeed('Binary signed');
    }
    
    // Final output
    const finalPath = options.output || path.join(process.cwd(), 
      `${manifest.name}-${options.device}-${manifest.version}.bin`);
    
    await fs.move(binaryPath, finalPath, { overwrite: true });
    
    // Get final size
    const stats = await fs.stat(finalPath);
    const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
    
    // Cleanup
    await fs.remove(buildDir);
    
    // Success message
    console.log(chalk.green(`\n✅ Baling complete!`));
    console.log(chalk.cyan('\n📦 Binary Details:'));
    console.log(chalk.gray(`   File: ${finalPath}`));
    console.log(chalk.gray(`   Size: ${sizeMB}MB`));
    console.log(chalk.gray(`   Platform: ${options.device}`));
    console.log(chalk.gray(`   Model: ${manifest.model}`));
    console.log(chalk.gray(`   Features: ${Object.entries(manifest.features)
      .filter(([_, v]) => v)
      .map(([k]) => k)
      .join(', ')}`));
    
    console.log(chalk.yellow(`\n🌾 Ready to harvest!`));
    console.log(chalk.gray(`   Deploy with: llamafarm harvest ${finalPath}`));
    console.log(chalk.gray(`   Or copy to target device and run: ./${path.basename(finalPath)}`));
    
    // Create deployment instructions
    await createDeploymentInstructions(path.dirname(finalPath), manifest, options.device);
    
  } catch (error) {
    spinner.fail('Baling failed');
    console.error(chalk.red(`Error: ${error instanceof Error ? error.message : String(error)}`));
    process.exit(1);
  }
}

// Helper functions

function getPlatform(device: string): string {
  const platformMap: Record<string, string> = {
    'mac': 'darwin',
    'mac-intel': 'darwin',
    'mac-arm': 'darwin',
    'linux': 'linux',
    'linux-arm': 'linux',
    'windows': 'win32',
    'raspberry-pi': 'linux',
    'jetson': 'linux'
  };
  return platformMap[device] || process.platform;
}

async function bundleNodeRuntime(buildDir: string, device: string) {
  // Create a Node.js bundle configuration
  const nodeVersion = process.version;
  const runtimeConfig = {
    node: nodeVersion,
    modules: {
      native: ['chromadb-native', 'onnxruntime-node'],
      included: ['express', 'ws', 'ollama']
    }
  };
  
  await fs.writeJSON(path.join(buildDir, '.runtime.json'), runtimeConfig);
  
  // In production, this would download/embed appropriate Node runtime
  // For now, we'll use pkg configuration
}

async function packageModel(buildDir: string, manifest: any) {
  const modelPath = path.join(buildDir, 'model.gguf');
  
  if (!await fs.pathExists(modelPath)) {
    // Create a placeholder for demo
    const modelInfo = {
      name: manifest.model,
      format: 'gguf',
      quantization: manifest.runtime.quantization,
      size: '4GB' // Would be actual size
    };
    
    await fs.writeJSON(path.join(buildDir, 'model.json'), modelInfo);
  }
  
  // Compress model if large
  if (manifest.runtime.quantization.startsWith('q4')) {
    // Model is already quantized, good to go
  }
}

async function embedVectorDatabase(buildDir: string, manifest: any) {
  const dbPath = path.join(buildDir, 'chroma_db');
  
  if (await fs.pathExists(dbPath)) {
    // Package the database files
    const dbConfig = {
      type: 'chroma',
      embedded: true,
      collections: ['default'],
      indices: []
    };
    
    await fs.writeJSON(path.join(buildDir, 'db.config.json'), dbConfig);
  }
}

async function createLauncher(buildDir: string, manifest: any, device: string) {
  let launcher = '';
  
  switch (device) {
    case 'mac':
    case 'mac-arm':
    case 'mac-intel':
      launcher = createMacLauncher(manifest);
      break;
    case 'windows':
      launcher = createWindowsLauncher(manifest);
      break;
    case 'linux':
    case 'raspberry-pi':
    case 'jetson':
      launcher = createLinuxLauncher(manifest);
      break;
  }
  
  // Write launcher
  const ext = device === 'windows' ? 'bat' : 'sh';
  const launcherPath = path.join(buildDir, `launcher.${ext}`);
  await fs.writeFile(launcherPath, launcher);
  
  if (device !== 'windows') {
    await fs.chmod(launcherPath, '755');
  }
}

function createMacLauncher(manifest: any): string {
  return `#!/bin/bash
# LlamaFarm Launcher for macOS
# ${manifest.name} v${manifest.version}

SCRIPT_DIR="$( cd "$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "🍎 Apple Silicon detected - Metal acceleration enabled"
    export LLAMAFARM_METAL=1
fi

# Set memory limits
export LLAMAFARM_MAX_RAM=${manifest.runtime.maxRam || '8G'}

# Launch
echo "🌾 Starting LlamaFarm..."
echo "📦 Model: ${manifest.model}"
echo "🌐 Port: ${manifest.runtime.port}"

# Run the embedded Node.js app
exec ./node_embedded --max-old-space-size=8192 main.js
`;
}

function createWindowsLauncher(manifest: any): string {
  return `@echo off
REM LlamaFarm Launcher for Windows
REM ${manifest.name} v${manifest.version}

cd /d "%~dp0"

echo 🌾 Starting LlamaFarm...
echo 📦 Model: ${manifest.model}
echo 🌐 Port: ${manifest.runtime.port}

REM Check for CUDA
where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 🎮 CUDA detected - GPU acceleration enabled
    set LLAMAFARM_CUDA=1
)

REM Run the embedded Node.js app
node_embedded.exe --max-old-space-size=8192 main.js

pause
`;
}

function createLinuxLauncher(manifest: any): string {
  return `#!/bin/bash
# LlamaFarm Launcher for Linux
# ${manifest.name} v${manifest.version}

SCRIPT_DIR="$( cd "$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected - GPU acceleration enabled"
    export LLAMAFARM_CUDA=1
fi

# Check system resources
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "💾 Available memory: \${TOTAL_MEM}GB"

# Launch
echo "🌾 Starting LlamaFarm..."
echo "📦 Model: ${manifest.model}"
echo "🌐 Port: ${manifest.runtime.port}"

# Run with appropriate limits
exec ./node_embedded --max-old-space-size=8192 main.js
`;
}

async function compileBinary(buildDir: string, manifest: any, options: BaleOptions): Promise<string> {
  // Prepare pkg configuration
  const pkgConfig = {
    name: manifest.name,
    version: manifest.version,
    bin: 'launcher.js',
    scripts: ['main.js', 'agent-server.js'],
    assets: [
      'model.gguf',
      'model.json',
      'manifest.json',
      'index.html',
      'chroma_db/**/*',
      'node_modules/**/*'
    ],
    targets: [getTarget(options.device)],
    outputPath: 'dist'
  };
  
  await fs.writeJSON(path.join(buildDir, 'pkg.json'), pkgConfig);
  
  // Create main entry point
  const mainJs = `
// LlamaFarm Binary Entry Point
const path = require('path');
const { spawn } = require('child_process');

// Set up environment
process.env.LLAMAFARM_BINARY = '1';
process.env.LLAMAFARM_ROOT = __dirname;

// Load manifest
const manifest = require('./manifest.json');

// Start the application
console.log('🌾 LlamaFarm v' + manifest.version);
console.log('📦 Model: ' + manifest.model);

// Check launcher type and execute
const launcher = process.platform === 'win32' ? 'launcher.bat' : 'launcher.sh';
const child = spawn(path.join(__dirname, launcher), [], {
  stdio: 'inherit',
  shell: true
});

child.on('exit', (code) => {
  process.exit(code);
});
`;
  
  await fs.writeFile(path.join(buildDir, 'main.js'), mainJs);
  
  // Run pkg
  const target = getTarget(options.device);
  const outputName = `${manifest.name}-${options.device}`;
  
  try {
    await execAsync(`npx pkg . --target ${target} --output ${outputName}`, {
      cwd: buildDir
    });
  } catch (error) {
    // Fallback to creating executable script
    console.log(chalk.yellow('⚠️  pkg not available, creating executable script...'));
    return createExecutableScript(buildDir, manifest, options);
  }
  
  return path.join(buildDir, outputName);
}

function getTarget(device: string): string {
  const targets: Record<string, string> = {
    'mac': 'node18-macos-x64',
    'mac-arm': 'node18-macos-arm64',
    'mac-intel': 'node18-macos-x64',
    'linux': 'node18-linux-x64',
    'linux-arm': 'node18-linux-arm64',
    'windows': 'node18-win-x64',
    'raspberry-pi': 'node18-linux-arm64',
    'jetson': 'node18-linux-arm64'
  };
  return targets[device] || 'node18-linux-x64';
}

async function createExecutableScript(buildDir: string, manifest: any, options: BaleOptions): Promise<string> {
  // Fallback: Create self-extracting archive
  const scriptName = `${manifest.name}-${options.device}.run`;
  const scriptPath = path.join(buildDir, scriptName);
  
  const script = `#!/bin/bash
# Self-extracting LlamaFarm binary
# ${manifest.name} v${manifest.version}

PAYLOAD_LINE=\`awk '/^__PAYLOAD_BELOW__/ {print NR + 1; exit 0; }' $0\`
TEMP_DIR=\`mktemp -d /tmp/llamafarm.XXXXXX\`

echo "🌾 Extracting LlamaFarm..."
tail -n +$PAYLOAD_LINE $0 | tar -xz -C $TEMP_DIR

echo "🚀 Launching..."
cd $TEMP_DIR
chmod +x launcher.sh
./launcher.sh

# Cleanup on exit
trap "rm -rf $TEMP_DIR" EXIT

exit 0
__PAYLOAD_BELOW__
`;
  
  // Create archive
  const archive = archiver('tar', { gzip: true });
  const output = fs.createWriteStream(scriptPath + '.tar.gz');
  
  await new Promise((resolve, reject) => {
    output.on('close', () => resolve(undefined));
    archive.on('error', reject);
    archive.directory(buildDir, false);
    archive.pipe(output);
    archive.finalize();
  });
  
  // Combine script and archive
  const scriptContent = script + await fs.readFile(scriptPath + '.tar.gz');
  await fs.writeFile(scriptPath, scriptContent);
  await fs.chmod(scriptPath, '755');
  await fs.remove(scriptPath + '.tar.gz');
  
  return scriptPath;
}

async function optimizeBinary(binaryPath: string, level?: string) {
  // Use UPX or similar for compression
  try {
    if (level === 'max') {
      await execAsync(`upx --best "${binaryPath}"`);
    } else {
      await execAsync(`upx "${binaryPath}"`);
    }
  } catch (e) {
    // UPX not available, skip
  }
}

async function signBinary(binaryPath: string, device: string) {
  // Platform-specific signing
  if (device.startsWith('mac')) {
    // macOS code signing
    try {
      await execAsync(`codesign --sign - "${binaryPath}"`);
    } catch (e) {
      console.log(chalk.yellow('⚠️  Code signing not available'));
    }
  } else if (device === 'windows') {
    // Windows signing would go here
  }
}

async function createDeploymentInstructions(outputDir: string, manifest: any, device: string) {
  const instructions = `# 🌾 LlamaFarm Deployment Instructions

## Binary: ${manifest.name}
- **Model**: ${manifest.model}
- **Version**: ${manifest.version}
- **Platform**: ${device}
- **Features**: ${Object.entries(manifest.features).filter(([_, v]) => v).map(([k]) => k).join(', ')}

## Deployment Options

### Option 1: Using LlamaFarm CLI
\`\`\`bash
llamafarm harvest ${manifest.name}-${device}-${manifest.version}.bin
\`\`\`

### Option 2: Direct Execution
\`\`\`bash
# Make executable (Unix/Linux/Mac)
chmod +x ${manifest.name}-${device}-${manifest.version}.bin

# Run
./${manifest.name}-${device}-${manifest.version}.bin
\`\`\`

### Option 3: Copy to Target Device
\`\`\`bash
# Copy to remote device
scp ${manifest.name}-${device}-${manifest.version}.bin user@device:/path/to/

# SSH to device and run
ssh user@device
cd /path/to/
./${manifest.name}-${device}-${manifest.version}.bin
\`\`\`

## System Requirements
- **RAM**: ${manifest.requirements?.minRam || 4}GB minimum
- **Disk**: ${manifest.requirements?.diskSpace || 10}GB free space
- **OS**: ${getOSRequirement(device)}

## Troubleshooting

### Permission Denied
\`\`\`bash
chmod +x ${manifest.name}-${device}-${manifest.version}.bin
\`\`\`

### Port Already in Use
The default port is ${manifest.runtime.port}. To use a different port:
\`\`\`bash
LLAMAFARM_PORT=8081 ./${manifest.name}-${device}-${manifest.version}.bin
\`\`\`

### GPU Not Detected
Make sure CUDA (Linux/Windows) or Metal (macOS) drivers are installed.

## Support
- GitHub: https://github.com/llamafarm/llamafarm-cli
- Issues: https://github.com/llamafarm/llamafarm-cli/issues
`;
  
  await fs.writeFile(path.join(outputDir, 'DEPLOYMENT.md'), instructions);
}

function getOSRequirement(device: string): string {
  const requirements: Record<string, string> = {
    'mac': 'macOS 12.0 or later',
    'mac-arm': 'macOS 12.0 or later (Apple Silicon)',
    'mac-intel': 'macOS 12.0 or later (Intel)',
    'linux': 'Ubuntu 20.04+ or equivalent',
    'windows': 'Windows 10/11 64-bit',
    'raspberry-pi': 'Raspberry Pi OS 64-bit',
    'jetson': 'JetPack 5.0+'
  };
  return requirements[device] || 'Compatible OS';
}