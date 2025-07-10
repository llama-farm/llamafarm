// src/utils/errors.ts
import chalk from 'chalk';

export class LlamaFarmError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'LlamaFarmError';
  }
}

export class ModelNotFoundError extends LlamaFarmError {
  constructor(model: string) {
    super(`Model '${model}' not found. Try 'llamafarm barn --list' to see available models.`, 'MODEL_NOT_FOUND');
  }
}

export class ConfigurationError extends LlamaFarmError {
  constructor(message: string) {
    super(`Configuration error: ${message}`, 'CONFIG_ERROR');
  }
}

export class DeploymentError extends LlamaFarmError {
  constructor(message: string) {
    super(`Deployment failed: ${message}`, 'DEPLOY_ERROR');
  }
}

export function handleError(error: any): void {
  if (error instanceof LlamaFarmError) {
    console.error(chalk.red(`\n❌ ${error.message}`));
    
    // Provide helpful suggestions based on error type
    switch (error.code) {
      case 'MODEL_NOT_FOUND':
        console.log(chalk.yellow('\n💡 Try one of these models:'));
        console.log(chalk.gray('   • llama3-8b'));
        console.log(chalk.gray('   • mixtral-8x7b'));
        console.log(chalk.gray('   • phi-2'));
        break;
      
      case 'CONFIG_ERROR':
        console.log(chalk.yellow('\n💡 Run "llamafarm till" to set up your configuration'));
        break;
      
      case 'DEPLOY_ERROR':
        console.log(chalk.yellow('\n💡 Check the logs above for more details'));
        console.log(chalk.gray('   You can also run "llamafarm weather" to check system status'));
        break;
    }
  } else {
    console.error(chalk.red(`\n❌ Unexpected error: ${error.message}`));
    if (process.env.DEBUG) {
      console.error(error.stack);
    }
  }
  
  process.exit(1);
}
