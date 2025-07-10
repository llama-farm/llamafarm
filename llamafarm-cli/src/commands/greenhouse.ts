import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs-extra';
import * as path from 'path';
import { Ollama } from 'ollama';
import { performance } from 'perf_hooks';

interface GreenhouseOptions {
  model?: string;
  agent?: string;
  scenario?: string;
  benchmark?: boolean;
}

export async function greenhouseCommand(options: GreenhouseOptions) {
  console.log(chalk.green('\n🏡 Welcome to the Greenhouse!\n'));
  console.log(chalk.gray('Testing environment for your AI harvest\n'));
  
  const spinner = ora();
  
  try {
    // Initialize test environment
    spinner.start('Preparing greenhouse environment...');
    
    const greenhouseDir = path.join(process.cwd(), '.llamafarm', 'greenhouse');
    await fs.ensureDir(greenhouseDir);
    
    spinner.succeed('Greenhouse ready');
    
    // Load test scenarios
    if (options.scenario) {
      spinner.start('Loading test scenario...');
      
      const scenarioPath = path.resolve(options.scenario);
      const scenario = await fs.readJSON(scenarioPath);
      
      spinner.succeed(`Loaded scenario: ${scenario.name}`);
      
      console.log(chalk.cyan('\n📋 Test Scenario:'));
      console.log(chalk.gray(`   Name: ${scenario.name}`));
      console.log(chalk.gray(`   Tests: ${scenario.tests.length}`));
      console.log(chalk.gray(`   Model: ${options.model || scenario.model}`));
    }
    
    // Run benchmarks if requested
    if (options.benchmark) {
      console.log(chalk.cyan('\n⚡ Running performance benchmarks...\n'));
      
      const ollama = new Ollama();
      const model = options.model || 'llama3-8b';
      
      // Test 1: Response time
      spinner.start('Testing response time...');
      const startTime = performance.now();
      
      await ollama.generate({
        model: model,
        prompt: 'Hello, how are you?',
        stream: false
      });
      
      const responseTime = performance.now() - startTime;
      spinner.succeed(`Response time: ${responseTime.toFixed(2)}ms`);
      
      // Test 2: Token generation speed
      spinner.start('Testing token generation speed...');
      
      const longPrompt = 'Write a detailed explanation of photosynthesis.';
      const genStart = performance.now();
      
      const response = await ollama.generate({
        model: model,
        prompt: longPrompt,
        stream: false
      });
      
      const genTime = performance.now() - genStart;
      const tokenCount = response.response.split(' ').length;
      const tokensPerSecond = (tokenCount / genTime) * 1000;
      
      spinner.succeed(`Token generation: ${tokensPerSecond.toFixed(1)} tokens/sec`);
      
      // Test 3: Memory usage
      spinner.start('Testing memory usage...');
      
      const memoryUsage = process.memoryUsage();
      const heapUsedMB = (memoryUsage.heapUsed / 1024 / 1024).toFixed(1);
      const rssMB = (memoryUsage.rss / 1024 / 1024).toFixed(1);
      
      spinner.succeed(`Memory: ${heapUsedMB}MB heap, ${rssMB}MB total`);
      
      // Summary
      console.log(chalk.green('\n✅ Benchmark Results:'));
      console.log(chalk.gray(`   Model: ${model}`));
      console.log(chalk.gray(`   Response Time: ${responseTime.toFixed(0)}ms`));
      console.log(chalk.gray(`   Generation Speed: ${tokensPerSecond.toFixed(1)} tokens/sec`));
      console.log(chalk.gray(`   Memory Usage: ${rssMB}MB`));
      
      // Save results
      const results = {
        model,
        timestamp: new Date().toISOString(),
        metrics: {
          responseTime,
          tokensPerSecond,
          memoryUsageMB: parseFloat(rssMB)
        }
      };
      
      await fs.writeJSON(
        path.join(greenhouseDir, `benchmark-${Date.now()}.json`),
        results,
        { spaces: 2 }
      );
    }
    
    // Interactive testing mode
    if (!options.scenario && !options.benchmark) {
      console.log(chalk.yellow('\n🧪 Greenhouse Test Menu:\n'));
      console.log(chalk.gray('   1. Run model benchmark'));
      console.log(chalk.gray('   2. Test agent conversation'));
      console.log(chalk.gray('   3. Test RAG pipeline'));
      console.log(chalk.gray('   4. Test vector search'));
      console.log(chalk.gray('   5. Load test scenario'));
      console.log(chalk.gray('\n   Use --benchmark or --scenario <file> for automated testing'));
    }
    
  } catch (error) {
    spinner.fail('Greenhouse test failed');
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}
