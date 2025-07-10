#!/usr/bin/env node

import { program } from 'commander';
import chalk from 'chalk';
import figlet from 'figlet';
const updateNotifier = require('update-notifier');
import { readFileSync } from 'fs';
import { join } from 'path';

// Commands
import { plantCommand } from './commands/plant';
import { harvestCommand } from './commands/harvest';
import { tillCommand } from './commands/till';
import { sowCommand } from './commands/sow';
import { irrigateCommand } from './commands/irrigate';
import { fertilizeCommand } from './commands/fertilize';
import { pruneCommand } from './commands/prune';
import { greenhouseCommand } from './commands/greenhouse';
import { siloCommand } from './commands/silo';
import { barnCommand } from './commands/barn';
import { fieldCommand } from './commands/field';
import { compostCommand } from './commands/compost';
import { almanacCommand } from './commands/almanac';
import { weatherCommand } from './commands/weather';

const pkg = JSON.parse(readFileSync(join(__dirname, '../package.json'), 'utf-8'));

// Check for updates
updateNotifier({ pkg }).notify();

// ASCII art banner
console.log(chalk.green(figlet.textSync('LlamaFarm', { horizontalLayout: 'full' })));
console.log(chalk.yellow('🌾 Plant and harvest AI models locally 🦙\n'));

program
  .name('llamafarm')
  .description('CLI for planting and harvesting AI models, agents, and databases into deployable binaries')
  .version(pkg.version);

// Core commands
program
  .command('plant <model>')
  .description('🌱 Plant a model with agents, databases, and configurations')
  .option('-d, --device <device>', 'target device (raspberry-pi, jetson, mac, windows, linux)', 'mac')
  .option('-a, --agent <agent>', 'agent name or template', 'chat-basic')
  .option('-r, --rag <enabled>', 'enable RAG pipeline', 'disabled')
  .option('-db, --database <type>', 'database type (vector, sqlite, postgres)', 'vector')
  .option('-p, --port <port>', 'port to run on (auto-assigns if not specified)')
  .option('-c, --config <file>', 'load configuration from YAML file')
  .option('--gpu', 'enable GPU acceleration')
  .option('--quantize <level>', 'quantization level (q4_0, q4_1, q5_0, q5_1, q8_0)', 'q4_0')
  .action(plantCommand);

program
  .command('harvest <url>')
  .description('🌾 Harvest a planted binary from URL or local path')
  .option('-o, --output <path>', 'output directory', './harvest')
  .option('--verify', 'verify binary integrity')
  .action(harvestCommand);

// Configuration commands
program
  .command('till')
  .description('🚜 Prepare the soil (initialize configuration)')
  .option('-f, --force', 'force re-initialization')
  .option('--import <file>', 'import configuration from YAML')
  .option('--export <file>', 'export current configuration to YAML')
  .action(tillCommand);

program
  .command('sow <seeds>')
  .description('🌰 Sow data seeds (configure data sources for RAG)')
  .option('-t, --type <type>', 'seed type (pdf, csv, json, api, web)', 'pdf')
  .option('-p, --path <path>', 'path to data source')
  .option('--chunk-size <size>', 'chunk size for processing', '512')
  .option('--overlap <overlap>', 'chunk overlap', '50')
  .action(sowCommand);

// Agent commands
program
  .command('irrigate <agent>')
  .description('💧 Irrigate agents (configure agent pipelines)')
  .option('-f, --framework <framework>', 'agent framework (langchain, autogen, crewai, llamaindex)', 'langchain')
  .option('-t, --template <template>', 'use agent template')
  .option('--tools <tools...>', 'tools to enable for agent')
  .option('--memory <type>', 'memory type (buffer, summary, vector)', 'buffer')
  .action(irrigateCommand);

program
  .command('fertilize')
  .description('🪴 Fertilize the deployment (optimize performance)')
  .option('--target <target>', 'optimization target (speed, memory, accuracy)', 'balanced')
  .option('--profile', 'run profiling before optimization')
  .action(fertilizeCommand);

// Management commands
program
  .command('prune')
  .description('✂️  Prune unused resources')
  .option('--models', 'prune unused models')
  .option('--agents', 'prune unused agents')
  .option('--data', 'prune old data')
  .option('--all', 'prune everything')
  .action(pruneCommand);

program
  .command('greenhouse')
  .description('🏡 Test in the greenhouse (sandbox environment)')
  .option('-m, --model <model>', 'model to test')
  .option('-a, --agent <agent>', 'agent to test')
  .option('--scenario <file>', 'test scenario file')
  .option('--benchmark', 'run performance benchmarks')
  .action(greenhouseCommand);

// Storage commands
program
  .command('silo')
  .description('🏭 Manage the silo (vector database)')
  .option('--init <type>', 'initialize vector DB (chroma, pinecone, weaviate, qdrant)', 'chroma')
  .option('--index <name>', 'create or select index')
  .option('--embed <model>', 'embedding model', 'all-MiniLM-L6-v2')
  .option('--dimension <size>', 'embedding dimension', '384')
  .action(siloCommand);

program
  .command('barn')
  .description('🏚️  Manage the barn (model storage)')
  .option('--list', 'list stored models')
  .option('--add <model>', 'add model to barn')
  .option('--remove <model>', 'remove model from barn')
  .option('--update', 'update all models')
  .action(barnCommand);

// Deployment commands
program
  .command('field')
  .description('🌾 Manage fields (deployment environments)')
  .option('--create <name>', 'create new field')
  .option('--list', 'list all fields')
  .option('--select <name>', 'select active field')
  .option('--config <file>', 'apply field configuration')
  .action(fieldCommand);

// Utility commands
program
  .command('compost')
  .description('♻️  Compost old deployments (cleanup and archive)')
  .option('--days <days>', 'archive deployments older than N days', '30')
  .option('--keep <count>', 'keep last N deployments', '5')
  .action(compostCommand);

program
  .command('almanac')
  .description('📚 View the almanac (documentation and guides)')
  .option('--recipe <name>', 'show specific recipe')
  .option('--list', 'list all recipes')
  .option('--search <term>', 'search recipes')
  .action(almanacCommand);

program
  .command('weather')
  .description('🌤️  Check the weather (system status and health)')
  .option('--detailed', 'show detailed metrics')
  .option('--history <days>', 'show history for N days', '7')
  .action(weatherCommand);

// Hidden/Advanced commands
program
  .command('baler', { hidden: true })
  .description('🎯 Advanced binary compilation options')
  .option('--custom-runtime', 'use custom runtime')
  .option('--strip-debug', 'strip debug symbols')
  .option('--compress <algo>', 'compression algorithm (zstd, lz4, none)', 'zstd')
  .action(() => {
    console.log(chalk.yellow('🎯 The Baler - Advanced compilation mode activated!'));
    console.log(chalk.gray('This feature is experimental. Use with caution.'));
  });

// Parse command line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}