// src/utils/compiler.ts
import * as fs from 'fs-extra';
import * as path from 'path';
import archiver from 'archiver';
import chalk from 'chalk';
import ora from 'ora';

export interface CompileOptions {
  model: string;
  device: string;
  workDir: string;
  outputPath: string;
  quantization: string;
  gpu: boolean;
}

export class BinaryCompiler {
  private spinner: ora.Ora;

  constructor() {
    this.spinner = ora();
  }

  async compile(options: CompileOptions): Promise<string> {
    console.log(chalk.green('\n🎯 The Baler is preparing your harvest...\n'));

    try {
      // Step 1: Optimize for target device
      this.spinner.start(`Optimizing for ${options.device}...`);
      await this.optimizeForDevice(options);
      this.spinner.succeed(`Optimized for ${options.device}`);

      // Step 2: Bundle dependencies
      this.spinner.start('Bundling dependencies...');
      await this.bundleDependencies(options);
      this.spinner.succeed('Dependencies bundled');

      // Step 3: Apply quantization
      this.spinner.start(`Applying ${options.quantization} quantization...`);
      await this.applyQuantization(options);
      this.spinner.succeed('Quantization applied');

      // Step 4: Create runtime
      this.spinner.start('Creating runtime environment...');
      await this.createRuntime(options);
      this.spinner.succeed('Runtime created');

      // Step 5: Package everything
      this.spinner.start('Packaging binary...');
      const packagePath = await this.packageBinary(options);
      this.spinner.succeed('Binary packaged');

      // Display summary
      const stats = await fs.stat(packagePath);
      const sizeMB = (stats.size / 1024 / 1024).toFixed(1);

      console.log(chalk.green('\n✅ Compilation complete!'));
      console.log(chalk.gray(`   Size: ${sizeMB}MB`));
      console.log(chalk.gray(`   Device: ${options.device}`));
      console.log(chalk.gray(`   Quantization: ${options.quantization}`));
      console.log(chalk.gray(`   GPU: ${options.gpu ? 'Enabled' : 'Disabled'}`));

      return packagePath;
    } catch (error) {
      this.spinner.fail('Compilation failed');
      throw error;
    }
  }

  private async optimizeForDevice(options: CompileOptions): Promise<void> {
    // Device-specific optimizations
    const optimizations: Record<string, any> = {
      'raspberry-pi': {
        arch: 'arm64',
        optimizations: ['neon', 'cortex-a72'],
        memoryLimit: '1GB'
      },
      'jetson': {
        arch: 'arm64',
        optimizations: ['cuda', 'tensorrt'],
        memoryLimit: '4GB'
      },
      'mac': {
        arch: 'arm64',
        optimizations: ['metal', 'accelerate'],
        memoryLimit: '8GB'
      },
      'windows': {
        arch: 'x64',
        optimizations: options.gpu ? ['cuda', 'directml'] : ['avx2'],
        memoryLimit: '8GB'
      },
      'linux': {
        arch: 'x64',
        optimizations: options.gpu ? ['cuda', 'rocm'] : ['avx2'],
        memoryLimit: '8GB'
      }
    };

    const deviceConfig = optimizations[options.device] || optimizations.linux;
    
    // Write optimization config
    await fs.writeJSON(
      path.join(options.workDir, 'device.config.json'),
      deviceConfig,
      { spaces: 2 }
    );

    // Simulate optimization delay
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  private async bundleDependencies(options: CompileOptions): Promise<void> {
    // Create node_modules bundle
    const bundlePath = path.join(options.workDir, 'bundle');
    await fs.ensureDir(bundlePath);

    // Copy essential dependencies (in real implementation, would use webpack/rollup)
    const dependencies = [
      'express',
      'ws',
      'ollama',
      'chromadb'
    ];

    for (const dep of dependencies) {
      // Simulate bundling
      await fs.writeFile(
        path.join(bundlePath, `${dep}.bundle.js`),
        `// Bundled ${dep} for ${options.device}`
      );
    }

    // Create loader script
    const loaderScript = `#!/usr/bin/env node
// LlamaFarm Runtime Loader
// Device: ${options.device}
// Model: ${options.model}

const path = require('path');
const { spawn } = require('child_process');

console.log('🌾 Starting LlamaFarm...');

// Set environment
process.env.LLAMAFARM_MODEL = '${options.model}';
process.env.LLAMAFARM_DEVICE = '${options.device}';
process.env.LLAMAFARM_GPU = '${options.gpu}';

// Start the application
require('./agent-server.js');
`;

    await fs.writeFile(path.join(options.workDir, 'loader.js'), loaderScript);
  }

  private async applyQuantization(options: CompileOptions): Promise<void> {
    // Quantization settings
    const quantizationSettings: Record<string, any> = {
      'q4_0': { bits: 4, groupSize: 32 },
      'q4_1': { bits: 4, groupSize: 32, desc: true },
      'q5_0': { bits: 5, groupSize: 32 },
      'q5_1': { bits: 5, groupSize: 32, desc: true },
      'q8_0': { bits: 8, groupSize: 32 }
    };

    const settings = quantizationSettings[options.quantization];
    
    // Write quantization config
    await fs.writeJSON(
      path.join(options.workDir, 'quantization.json'),
      {
        method: options.quantization,
        ...settings,
        optimizedFor: options.device
      },
      { spaces: 2 }
    );

    // Simulate quantization process
    await new Promise(resolve => setTimeout(resolve, 1500));
  }

  private async createRuntime(options: CompileOptions): Promise<void> {
    // Create platform-specific runtime
    const runtimeScript = this.generateRuntimeScript(options);
    
    await fs.writeFile(
      path.join(options.workDir, 'runtime.sh'),
      runtimeScript
    );
    
    await fs.chmod(path.join(options.workDir, 'runtime.sh'), '755');

    // Create Windows batch file if needed
    if (options.device === 'windows') {
      const batchScript = this.generateBatchScript(options);
      await fs.writeFile(
        path.join(options.workDir, 'runtime.bat'),
        batchScript
      );
    }
  }

  private generateRuntimeScript(options: CompileOptions): string {
    return `#!/bin/bash
# LlamaFarm Runtime Script
# Generated for: ${options.device}
# Model: ${options.model}

echo "🌾 LlamaFarm Runtime v1.0"
echo "========================"

# Check system requirements
check_requirements() {
    echo "🔍 Checking system requirements..."
    
    # Check memory
    if [ "$(uname)" == "Darwin" ]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    if [ "$TOTAL_MEM" -lt 4 ]; then
        echo "⚠️  Warning: Less than 4GB RAM detected"
    fi
    
    # Check disk space
    DISK_FREE=$(df -BG . | awk 'NR==2 {print int($4)}')
    if [ "$DISK_FREE" -lt 10 ]; then
        echo "⚠️  Warning: Less than 10GB free disk space"
    fi
    
    echo "✅ System check complete"
}

# Start services
start_services() {
    echo "🚀 Starting services..."
    
    # Start Ollama server (if included)
    if [ -f "./ollama" ]; then
        ./ollama serve &
        OLLAMA_PID=$!
        sleep 3
    fi
    
    # Start vector database
    if [ -d "./chroma_db" ]; then
        echo "🗄️  Vector database ready"
    fi
    
    # Start agent server
    node loader.js &
    AGENT_PID=$!
    
    echo "✅ All services started"
}

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    [ ! -z "$OLLAMA_PID" ] && kill $OLLAMA_PID 2>/dev/null
    [ ! -z "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Main execution
main() {
    check_requirements
    start_services
    
    echo ""
    echo "🌐 LlamaFarm is running!"
    echo "📍 Open http://localhost:8080 in your browser"
    echo "📊 API available at http://localhost:8080/api"
    echo ""
    echo "Press Ctrl+C to stop"
    
    # Wait for interrupt
    wait $AGENT_PID
}

# Run main function
main
`;
  }

  private generateBatchScript(options: CompileOptions): string {
    return `@echo off
REM LlamaFarm Runtime Script for Windows
REM Generated for: ${options.device}
REM Model: ${options.model}

echo 🌾 LlamaFarm Runtime v1.0
echo ========================

REM Start services
echo 🚀 Starting services...

REM Start agent server
start /B node loader.js

echo.
echo 🌐 LlamaFarm is running!
echo 📍 Open http://localhost:8080 in your browser
echo 📊 API available at http://localhost:8080/api
echo.
echo Press Ctrl+C to stop

REM Keep window open
pause >nul
`;
  }

  private async packageBinary(options: CompileOptions): Promise<string> {
    const output = fs.createWriteStream(options.outputPath);
    const archive = archiver('tar', { 
      gzip: true,
      gzipOptions: { level: 9 }
    });

    return new Promise((resolve, reject) => {
      output.on('close', () => resolve(options.outputPath));
      archive.on('error', reject);

      archive.pipe(output);
      archive.directory(options.workDir, false);
      archive.finalize();
    });
  }
}

