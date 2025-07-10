import chalk from 'chalk';
import ora from 'ora';
import * as fs from 'fs-extra';
import * as path from 'path';
import * as yaml from 'js-yaml';
import * as inquirer from 'inquirer';
import { getConfigPath, loadConfig, saveConfig } from '../utils/config';

interface TillOptions {
  force?: boolean;
  import?: string;
  export?: string;
}

export async function tillCommand(options: TillOptions) {
  const spinner = ora();
  
  try {
    // Export configuration
    if (options.export) {
      spinner.start('Exporting field configuration...');
      
      const config = await loadConfig();
      const yamlContent = yaml.dump(config, { indent: 2 });
      
      await fs.writeFile(options.export, yamlContent, 'utf-8');
      
      spinner.succeed(`Configuration exported to ${options.export}`);
      
      console.log(chalk.green('\n✅ Field configuration exported!'));
      console.log(chalk.gray(`   View with: cat ${options.export}`));
      return;
    }
    
    // Import configuration
    if (options.import) {
      spinner.start('Importing field configuration...');
      
      const yamlContent = await fs.readFile(options.import, 'utf-8');
      const importedConfig = yaml.load(yamlContent);
      
      await saveConfig(importedConfig);
      
      spinner.succeed('Configuration imported');
      
      console.log(chalk.green('\n✅ Field configuration imported!'));
      return;
    }
    
    // Initialize configuration
    console.log(chalk.green('\n🚜 Preparing the soil...\n'));
    
    const configPath = getConfigPath();
    const configExists = await fs.pathExists(configPath);
    
    if (configExists && !options.force) {
      console.log(chalk.yellow('⚠️  Configuration already exists!'));
      const { proceed } = await inquirer.prompt([{
        type: 'confirm',
        name: 'proceed',
        message: 'Do you want to re-till the field?',
        default: false
      }]);
      
      if (!proceed) {
        console.log(chalk.gray('Keeping existing configuration.'));
        return;
      }
    }
    
    // Interactive configuration
    const answers = await inquirer.prompt([
      {
        type: 'input',
        name: 'farmerName',
        message: '👨‍🌾 What\'s your farmer name?',
        default: process.env.USER || 'farmer'
      },
      {
        type: 'list',
        name: 'defaultDevice',
        message: '🖥️  What\'s your primary planting device?',
        choices: [
          { name: '🍎 Mac (Apple Silicon)', value: 'mac' },
          { name: '🐧 Linux', value: 'linux' },
          { name: '🪟 Windows', value: 'windows' },
          { name: '🥧 Raspberry Pi', value: 'raspberry-pi' },
          { name: '🚀 NVIDIA Jetson', value: 'jetson' }
        ]
      },
      {
        type: 'list',
        name: 'defaultQuantization',
        message: '📊 Default model quantization?',
        choices: [
          { name: 'Q4_0 (Balanced)', value: 'q4_0' },
          { name: 'Q4_1 (Better quality)', value: 'q4_1' },
          { name: 'Q5_0 (Higher quality)', value: 'q5_0' },
          { name: 'Q5_1 (Best for most)', value: 'q5_1' },
          { name: 'Q8_0 (Highest quality)', value: 'q8_0' }
        ],
        default: 'q4_0'
      },
      {
        type: 'confirm',
        name: 'enableGpu',
        message: '🎮 Enable GPU acceleration?',
        default: false
      },
      {
        type: 'list',
        name: 'defaultVectorDb',
        message: '🗄️  Default vector database?',
        choices: [
          { name: '🌈 ChromaDB (Recommended)', value: 'chroma' },
          { name: '🔍 Qdrant', value: 'qdrant' },
          { name: '🌲 Pinecone', value: 'pinecone' },
          { name: '🕸️  Weaviate', value: 'weaviate' }
        ]
      },
      {
        type: 'list',
        name: 'defaultAgent',
        message: '🤖 Default agent framework?',
        choices: [
          { name: '🦜 LangChain', value: 'langchain' },
          { name: '🤝 AutoGen', value: 'autogen' },
          { name: '👥 CrewAI', value: 'crewai' },
          { name: '🦙 LlamaIndex', value: 'llamaindex' }
        ]
      },
      {
        type: 'input',
        name: 'harvestDir',
        message: '📁 Where should we store harvests?',
        default: './harvests'
      },
      {
        type: 'confirm',
        name: 'telemetry',
        message: '📊 Share anonymous usage data to improve LlamaFarm?',
        default: true
      }
    ]);
    
    // Advanced configuration
    const { advanced } = await inquirer.prompt([{
      type: 'confirm',
      name: 'advanced',
      message: '⚙️  Configure advanced settings?',
      default: false
    }]);
    
    let advancedConfig = {};
    if (advanced) {
      advancedConfig = await inquirer.prompt([
        {
          type: 'input',
          name: 'chunkSize',
          message: '📏 Default RAG chunk size?',
          default: '512'
        },
        {
          type: 'input',
          name: 'chunkOverlap',
          message: '🔄 Default chunk overlap?',
          default: '50'
        },
        {
          type: 'input',
          name: 'embeddingModel',
          message: '🧮 Default embedding model?',
          default: 'all-MiniLM-L6-v2'
        },
        {
          type: 'input',
          name: 'contextWindow',
          message: '🪟 Default context window?',
          default: '2048'
        },
        {
          type: 'confirm',
          name: 'autoUpdate',
          message: '🔄 Enable auto-updates?',
          default: true
        }
      ]);
    }
    
    // Create configuration
    const config = {
      version: '1.0.0',
      farmer: {
        name: answers.farmerName,
        created: new Date().toISOString()
      },
      defaults: {
        device: answers.defaultDevice,
        quantization: answers.defaultQuantization,
        gpu: answers.enableGpu,
        vectorDb: answers.defaultVectorDb,
        agentFramework: answers.defaultAgent
      },
      paths: {
        harvest: answers.harvestDir,
        models: path.join(answers.harvestDir, 'models'),
        agents: path.join(answers.harvestDir, 'agents'),
        data: path.join(answers.harvestDir, 'data')
      },
      advanced: {
        chunkSize: parseInt(advancedConfig.chunkSize || '512'),
        chunkOverlap: parseInt(advancedConfig.chunkOverlap || '50'),
        embeddingModel: advancedConfig.embeddingModel || 'all-MiniLM-L6-v2',
        contextWindow: parseInt(advancedConfig.contextWindow || '2048'),
        autoUpdate: advancedConfig.autoUpdate ?? true
      },
      telemetry: answers.telemetry
    };
    
    // Save configuration
    spinner.start('Tilling the field...');
    
    await saveConfig(config);
    
    // Create directories
    for (const dirPath of Object.values(config.paths)) {
      await fs.ensureDir(dirPath as string);
    }
    
    spinner.succeed('Field tilled and ready!');
    
    // Display summary
    console.log(chalk.green('\n✅ Configuration complete!\n'));
    console.log(chalk.cyan('🌾 Your LlamaFarm is configured:'));
    console.log(chalk.gray(`   Farmer: ${config.farmer.name}`));
    console.log(chalk.gray(`   Device: ${config.defaults.device}`));
    console.log(chalk.gray(`   GPU: ${config.defaults.gpu ? 'Enabled' : 'Disabled'}`));
    console.log(chalk.gray(`   Vector DB: ${config.defaults.vectorDb}`));
    console.log(chalk.gray(`   Agent Framework: ${config.defaults.agentFramework}`));
    
    // Create example YAML
    const exampleYaml = `# LlamaFarm Configuration Example
# Save this as farm.yaml and use with: llamafarm plant <model> --config farm.yaml

model:
  name: llama3-8b
  quantization: q4_0
  
agent:
  name: research-assistant
  framework: langchain
  tools:
    - web_search
    - calculator
    - code_interpreter
  memory: vector
  
database:
  type: vector
  provider: chroma
  embedding_model: all-MiniLM-L6-v2
  
rag:
  enabled: true
  chunk_size: 512
  chunk_overlap: 50
  retrieval_k: 4
  
deployment:
  device: ${config.defaults.device}
  port: auto
  gpu: ${config.defaults.gpu}
  
data_sources:
  - type: pdf
    path: ./documents
  - type: web
    urls:
      - https://example.com/docs
`;
    
    const examplePath = path.join(process.cwd(), 'farm-example.yaml');
    await fs.writeFile(examplePath, exampleYaml, 'utf-8');
    
    console.log(chalk.yellow(`\n💡 Example configuration saved to: ${examplePath}`));
    console.log(chalk.gray('   Use it with: llamafarm plant llama3-8b --config farm-example.yaml'));
    
  } catch (error) {
    spinner.fail('Tilling failed');
    console.error(chalk.red(`Error: ${error.message}`));
    process.exit(1);
  }
}