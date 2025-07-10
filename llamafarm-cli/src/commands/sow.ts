// src/commands/sow.ts
import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs-extra';
import * as path from 'path';
import { loadConfig } from '../utils/config';

interface SowOptions {
  type: string;
  path?: string;
  chunkSize: string;
  overlap: string;
}

export async function sowCommand(seeds: string, options: SowOptions) {
  console.log(chalk.green(`\n🌰 Sowing ${seeds} seeds...\n`));
  
  const spinner = ora();
  const config = await loadConfig();
  
  try {
    spinner.start('Preparing seed bed...');
    
    const seedsDir = path.join(config.paths.data, 'seeds');
    await fs.ensureDir(seedsDir);
    
    const seedConfig = {
      name: seeds,
      type: options.type,
      source: options.path || seeds,
      processing: {
        chunkSize: parseInt(options.chunkSize),
        chunkOverlap: parseInt(options.overlap)
      },
      created: new Date().toISOString()
    };
    
    spinner.succeed('Seed bed prepared');
    
    // Handle different seed types
    switch (options.type) {
      case 'pdf':
        spinner.start('Planting PDF seeds...');
        console.log(chalk.gray(`\n   📄 Source: ${options.path}`));
        console.log(chalk.gray(`   📏 Chunk size: ${options.chunkSize}`));
        console.log(chalk.gray(`   🔄 Overlap: ${options.overlap}`));
        
        // In real implementation, would process PDFs
        await fs.writeJSON(
          path.join(seedsDir, `${seeds}-pdf.json`),
          { ...seedConfig, documents: ['example.pdf'] }
        );
        
        spinner.succeed('PDF seeds planted');
        break;
        
      case 'csv':
        spinner.start('Planting CSV seeds...');
        console.log(chalk.gray(`\n   📊 Source: ${options.path}`));
        
        // In real implementation, would process CSVs
        await fs.writeJSON(
          path.join(seedsDir, `${seeds}-csv.json`),
          { ...seedConfig, schema: { columns: ['id', 'text', 'metadata'] } }
        );
        
        spinner.succeed('CSV seeds planted');
        break;
        
      case 'web':
        spinner.start('Planting web seeds...');
        console.log(chalk.gray(`\n   🌐 URL: ${seeds}`));
        
        // In real implementation, would crawl websites
        await fs.writeJSON(
          path.join(seedsDir, `${seeds}-web.json`),
          { ...seedConfig, urls: [seeds], crawlDepth: 2 }
        );
        
        spinner.succeed('Web seeds planted');
        break;
        
      case 'api':
        spinner.start('Planting API seeds...');
        console.log(chalk.gray(`\n   🔌 Endpoint: ${seeds}`));
        
        await fs.writeJSON(
          path.join(seedsDir, `${seeds}-api.json`),
          { ...seedConfig, endpoint: seeds, pollInterval: 3600 }
        );
        
        spinner.succeed('API seeds planted');
        break;
    }
    
    console.log(chalk.green('\n✅ Seeds sown successfully!'));
    console.log(chalk.yellow('\n💡 Next steps:'));
    console.log(chalk.gray('   1. Run "llamafarm irrigate <agent>" to water with agent capabilities'));
    console.log(chalk.gray('   2. Run "llamafarm plant <model>" to grow your harvest'));
    
  } catch (error) {
    spinner.fail('Sowing failed');
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}
