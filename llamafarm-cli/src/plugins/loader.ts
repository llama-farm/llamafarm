/**
 * Plugin Loader - Imports from top-level plugins directory
 * All plugins are now in the /plugins directory at repository root
 */

import * as path from 'path';
import chalk from 'chalk';

// Import registry from top-level plugins
// Adjust the path based on your actual structure
let registry: any;
let FieldPlugin: any;
let EquipmentPlugin: any;
let PipePlugin: any;

try {
  // Try to import from top-level plugins
  const pluginsPath = path.resolve(__dirname, '../../../../plugins/registry');
  const pluginModule = require(pluginsPath);
  registry = pluginModule.registry || pluginModule.default;
  
  // Export types
  exports.FieldPlugin = pluginModule.FieldPlugin;
  exports.EquipmentPlugin = pluginModule.EquipmentPlugin;
  exports.PipePlugin = pluginModule.PipePlugin;
} catch (e) {
  console.warn(chalk.yellow('⚠️  Could not load plugins from top level. Creating mock registry.'));
  
  // Create a mock registry for development
  registry = {
    plugins: new Map(),
    register(plugin: any) {
      this.plugins.set(plugin.name, plugin);
    },
    get(name: string) {
      return this.plugins.get(name);
    },
    getByType(type: string) {
      return Array.from(this.plugins.values()).filter((p: any) => p.type === type);
    },
    async loadAll() {
      console.log(chalk.yellow('🦙 Plugins are in the /plugins directory at repository root'));
    }
  };
}

export { registry };

export async function loadPlugins() {
  console.log(chalk.green('🔌 Loading plugins from top-level directory...'));
  
  if (registry && registry.loadAll) {
    await registry.loadAll();
    
    const fields = registry.getByType('field');
    const equipment = registry.getByType('equipment');
    const pipes = registry.getByType('pipe');
    
    console.log(chalk.gray(`📊 Loaded: ${fields.length} fields, ${equipment.length} equipment, ${pipes.length} pipes`));
  }
}

// Define plugin types locally
export type Plugin = any;
export type FieldPlugin = {
  type: 'field';
  name: string;
  version: string;
  init(): Promise<void>;
};
export type EquipmentPlugin = {
  type: 'equipment';
  name: string;
  version: string;
  init(): Promise<void>;
};
export type PipePlugin = {
  type: 'pipe';
  name: string;
  version: string;
  init(): Promise<void>;
};
