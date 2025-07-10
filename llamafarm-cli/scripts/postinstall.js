#!/usr/bin/env node

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