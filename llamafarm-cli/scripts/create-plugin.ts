#!/usr/bin/env ts-node

import * as inquirer from 'inquirer';
import * as fs from 'fs-extra';
import * as path from 'path';
import chalk from 'chalk';

async function createPlugin() {
  console.log(chalk.green('\n🔌 LlamaFarm Plugin Creator\n'));
  
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'type',
      message: 'What type of plugin?',
      choices: [
        { name: '🌾 Field (Platform configuration)', value: 'field' },
        { name: '🛠️  Equipment (Database/Tool/Runtime)', value: 'equipment' },
        { name: '🔧 Pipe (Communication channel)', value: 'pipe' }
      ]
    },
    {
      type: 'input',
      name: 'name',
      message: 'Plugin name (lowercase, hyphens):',
      validate: (input) => /^[a-z0-9-]+$/.test(input) || 'Use lowercase and hyphens only'
    },
    {
      type: 'input',
      name: 'description',
      message: 'Brief description:'
    },
    {
      type: 'input',
      name: 'author',
      message: 'Author name:',
      default: 'Community Contributor'
    }
  ]);
  
  // Additional questions based on type
  if (answers.type === 'field') {
    const fieldAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'platform',
        message: 'Target platform:',
        choices: ['darwin', 'linux', 'win32', 'freebsd', 'android']
      }
    ]);
    Object.assign(answers, fieldAnswers);
  }
  
  if (answers.type === 'equipment') {
    const equipAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'category',
        message: 'Equipment category:',
        choices: ['database', 'rag-pipeline', 'model-runtime', 'tool']
      }
    ]);
    Object.assign(answers, equipAnswers);
  }
  
  if (answers.type === 'pipe') {
    const pipeAnswers = await inquirer.prompt([
      {
        type: 'input',
        name: 'protocol',
        message: 'Protocol name:',
        default: answers.name.replace('-pipe', '')
      }
    ]);
    Object.assign(answers, pipeAnswers);
  }
  
  // Create plugin from template
  await createFromTemplate(answers);
  
  console.log(chalk.green(`\n✅ Plugin created successfully!`));
  console.log(chalk.gray(`\n📁 Location: src/plugins/${answers.type}s/${answers.name}/`));
  console.log(chalk.gray(`\n📝 Next steps:`));
  console.log(chalk.gray(`   1. Implement the required methods`));
  console.log(chalk.gray(`   2. Add tests in test.ts`));
  console.log(chalk.gray(`   3. Add documentation in README.md`));
  console.log(chalk.gray(`   4. Test with: npm test src/plugins/${answers.type}s/${answers.name}`));
}

async function createFromTemplate(options: any) {
  const { type, name, description, author } = options;
  
  // Determine paths
  let templatePath = path.join('templates/plugins', type, 'template.ts');
  let targetDir = path.join('src/plugins', `${type}s`);
  
  if (type === 'equipment') {
    targetDir = path.join(targetDir, options.category);
  }
  
  targetDir = path.join(targetDir, name);
  
  // Create directory
  await fs.ensureDir(targetDir);
  
  // Read and process template
  let template = await fs.readFile(templatePath, 'utf8');
  
  // Replace placeholders
  template = template
    .replace(/\{\{NAME\}\}/g, name.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(''))
    .replace(/\{\{LOWERCASE_NAME\}\}/g, name)
    .replace(/\{\{DESCRIPTION\}\}/g, description)
    .replace(/\{\{AUTHOR\}\}/g, author)
    .replace(/\{\{PLATFORM\}\}/g, options.platform || '')
    .replace(/\{\{CATEGORY\}\}/g, options.category || '')
    .replace(/\{\{PROTOCOL\}\}/g, options.protocol || '');
  
  // Write files
  await fs.writeFile(path.join(targetDir, 'index.ts'), template);
  await fs.writeFile(path.join(targetDir, 'README.md'), `# ${name}\n\n${description}\n`);
  await fs.writeFile(path.join(targetDir, 'test.ts'), `import plugin from './index';\n\n// Add tests here\n`);
}

createPlugin().catch(console.error);
