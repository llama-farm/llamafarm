import chalk from 'chalk';
import { loadPlugins, registry } from '../plugins/loader';

export async function pluginsCommand(options: any) {
  console.log(chalk.green('\n🔌 LlamaFarm Plugins\n'));
  
  await loadPlugins();
  
  const fields = registry.getByType('field');
  const equipment = registry.getByType('equipment');
  const pipes = registry.getByType('pipe');
  
  if (fields.length > 0) {
    console.log(chalk.cyan('\n🌾 Field Plugins:'));
    fields.forEach((f: any) => {
      console.log(chalk.gray(`   • ${f.name} v${f.version} - ${f.description}`));
    });
    console.log();
  }
  
  if (equipment.length > 0) {
    console.log(chalk.cyan('\n🔧 Equipment Plugins:'));
    equipment.forEach((e: any) => {
      console.log(chalk.gray(`   • ${e.name} v${e.version} - ${e.description}`));
    });
    console.log();
  }
  
  if (pipes.length > 0) {
    console.log(chalk.cyan('\n🚰 Pipe Plugins:'));
    pipes.forEach((p: any) => {
      console.log(chalk.gray(`   • ${p.name} v${p.version} - ${p.description}`));
    });
    console.log();
  }
  
  console.log(chalk.yellow('💡 To create a plugin, go to the /plugins directory'));
  console.log(chalk.gray('   See templates and examples there!\n'));
}
