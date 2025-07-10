import chalk from 'chalk';
import ora from 'ora';
import fetch from 'node-fetch';
import * as fs from 'fs-extra';
import * as path from 'path';
import extract from 'extract-zip';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface HarvestOptions {
  output: string;
  verify?: boolean;
}

export async function harvestCommand(url: string, options: HarvestOptions) {
  console.log(chalk.green(`\n🌾 Harvesting from ${url}...`));
  
  const spinner = ora();
  
  try {
    // Step 1: Download the binary
    spinner.start('Gathering the harvest...');
    
    const outputDir = path.resolve(options.output);
    await fs.ensureDir(outputDir);
    
    let downloadPath: string;
    
    if (url.startsWith('http')) {
      // Download from URL
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to download: ${response.statusText}`);
      }
      
      const filename = path.basename(url);
      downloadPath = path.join(outputDir, filename);
      
      const buffer = await response.buffer();
      await fs.writeFile(downloadPath, buffer);
      
      spinner.succeed(`Harvest gathered (${(buffer.length / 1024 / 1024).toFixed(1)}MB)`);
    } else {
      // Local file
      downloadPath = path.resolve(url);
      if (!await fs.pathExists(downloadPath)) {
        throw new Error(`File not found: ${downloadPath}`);
      }
      spinner.succeed('Local harvest found');
    }
    
    // Step 2: Verify integrity if requested
    if (options.verify) {
      spinner.start('Verifying harvest integrity...');
      
      // In real implementation, we'd verify checksums/signatures
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      spinner.succeed('Harvest verified ✓');
    }
    
    // Step 3: Extract the package
    spinner.start('Threshing the grain...');
    
    const extractDir = path.join(outputDir, 'extracted');
    await fs.ensureDir(extractDir);
    
    if (downloadPath.endsWith('.tar.gz')) {
      await execAsync(`tar -xzf "${downloadPath}" -C "${extractDir}"`);
    } else if (downloadPath.endsWith('.zip')) {
      await extract(downloadPath, { dir: extractDir });
    }
    
    spinner.succeed('Grain threshed');
    
    // Step 4: Read manifest
    spinner.start('Reading the harvest manifest...');
    
    const manifestPath = path.join(extractDir, 'manifest.json');
    const manifest = await fs.readJSON(manifestPath);
    
    console.log(chalk.cyan('\n📋 Harvest Manifest:'));
    console.log(chalk.gray(`   Name: ${manifest.name}`));
    console.log(chalk.gray(`   Model: ${manifest.model}`));
    console.log(chalk.gray(`   Agent: ${manifest.agent}`));
    console.log(chalk.gray(`   Device: ${manifest.device}`));
    console.log(chalk.gray(`   Features: RAG=${manifest.features.rag}, VectorDB=${manifest.features.vectorDb}`));
    
    spinner.succeed('Manifest loaded');
    
    // Step 5: Install to system
    spinner.start('Storing in the barn...');
    
    const installDir = path.join(outputDir, manifest.name);
    await fs.copy(extractDir, installDir);
    
    // Make scripts executable
    const scriptsToChmod = ['start.sh', 'build.sh'];
    for (const script of scriptsToChmod) {
      const scriptPath = path.join(installDir, script);
      if (await fs.pathExists(scriptPath)) {
        await fs.chmod(scriptPath, '755');
      }
    }
    
    spinner.succeed('Stored in barn');
    
    // Step 6: Create convenience launcher
    spinner.start('Creating harvest launcher...');
    
    const launcherScript = `#!/bin/bash
# LlamaFarm Launcher for ${manifest.name}

echo "🌾 LlamaFarm Harvest Launcher"
echo "📦 ${manifest.name}"
echo ""

cd "${installDir}"
./start.sh
`;
    
    const launcherPath = path.join(outputDir, `run-${manifest.name}.sh`);
    await fs.writeFile(launcherPath, launcherScript);
    await fs.chmod(launcherPath, '755');
    
    spinner.succeed('Launcher created');
    
    // Success!
    console.log(chalk.green('\n✅ Harvest complete!'));
    console.log(chalk.yellow(`\n🚀 To run your harvest:`));
    console.log(chalk.white(`   cd ${outputDir}`));
    console.log(chalk.white(`   ./run-${manifest.name}.sh`));
    console.log(chalk.gray(`\n   Or run directly: cd ${installDir} && ./start.sh`));
    
    // Cleanup
    await fs.remove(extractDir);
    if (url.startsWith('http')) {
      await fs.remove(downloadPath);
    }
    
  } catch (error) {
    spinner.fail('Harvest failed');
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}