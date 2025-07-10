import chalk from 'chalk';
import ora from 'ora';

export async function plantWithPlugins(model: string, options: any) {
  console.log(chalk.green(`\n🌱 Planting ${model} with plugins...`));
  
  const spinner = ora();
  
  try {
    // Try to load plugins from top level
    const { loadPlugins, registry } = await import('../plugins/loader');
    await loadPlugins();
    
    // Get field plugin for current platform
    const platform = options.device || process.platform;
    const fieldPlugin = registry.getByType('field')
      .find((f: any) => f.platform === platform);
      
    if (fieldPlugin) {
      spinner.start(`Using field: ${fieldPlugin.name}`);
      await fieldPlugin.setup();
      const optimizedConfig = fieldPlugin.optimize(options);
      spinner.succeed(`Optimized for ${platform}`);
    } else {
      spinner.info(`No field plugin found for ${platform}`);
    }
    
    // Get database plugin
    const dbName = options.database || 'demo';
    const dbPlugin = registry.get(`${dbName}-database`);
    
    if (dbPlugin) {
      spinner.start(`Using database: ${dbPlugin.name}`);
      await dbPlugin.install();
      await dbPlugin.configure({ path: `./${model}-db` });
      const testPassed = await dbPlugin.test();
      spinner.succeed(testPassed ? 'Database ready' : 'Database configured (test failed)');
    } else {
      spinner.info(`No database plugin found for ${dbName}`);
    }
    
    // Get pipe plugin
    const pipeName = options.pipe || 'websocket';
    const pipePlugin = registry.get(`${pipeName}-pipe`);
    
    if (pipePlugin) {
      spinner.info(`Will use ${pipePlugin.name} for communication`);
    } else {
      spinner.info(`No pipe plugin found for ${pipeName}`);
    }
    
    console.log(chalk.green('\n✅ Planting complete!'));
    
  } catch (error) {
    spinner.fail('Plugin loading failed');
    console.log(chalk.yellow('\n💡 Make sure plugins are installed in the /plugins directory'));
    console.log(chalk.gray('   You can continue without plugins\n'));
  }
}
