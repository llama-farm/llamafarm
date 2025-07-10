/**
 * LlamaFarm Plugin Registry
 * This is imported by the CLI to load plugins
 */

export interface Plugin {
  name: string;
  type: 'field' | 'equipment' | 'pipe';
  version: string;
  description: string;
  author?: string;
  dependencies?: string[];
  setup: () => Promise<void>;
  teardown?: () => Promise<void>;
}

export interface FieldPlugin extends Plugin {
  type: 'field';
  platform: string;
  constraints: {
    minMemory?: number;
    minDisk?: number;
    requiredTools?: string[];
    supportedModels?: string[];
  };
  optimize: (config: any) => any;
}

export interface EquipmentPlugin extends Plugin {
  type: 'equipment';
  category: 'database' | 'rag-pipeline' | 'model-runtime' | 'tool';
  install: () => Promise<void>;
  configure: (options: any) => Promise<void>;
  test: () => Promise<boolean>;
}

export interface PipePlugin extends Plugin {
  type: 'pipe';
  protocol: string;
  createServer: (options: any) => any;
  createClient: (options: any) => any;
}

class PluginRegistry {
  private plugins: Map<string, Plugin> = new Map();

  register(plugin: Plugin) {
    console.log(`🔌 Registering plugin: ${plugin.name} (${plugin.type})`);
    this.plugins.set(plugin.name, plugin);
  }

  get(name: string): Plugin | undefined {
    return this.plugins.get(name);
  }

  getByType(type: 'field' | 'equipment' | 'pipe'): Plugin[] {
    return Array.from(this.plugins.values()).filter(p => p.type === type);
  }

  async loadAll() {
    console.log('📦 Loading plugins from top-level directory...');
    
    // In production, this would dynamically load all plugins
    // For now, we'll manually import them
    try {
      // Import individual plugins as they're created
      const plugins = [
        await import('./fields/mac'),
        await import('./equipment/databases/demo'),
        await import('./pipes/websocket'),
      ];
      
      plugins.forEach(p => {
        if (p.default) {
          this.register(p.default);
        }
      });
      
      console.log(`✅ Loaded ${this.plugins.size} plugins`);
    } catch (e) {
      console.log('⚠️  Some plugins not found (this is normal for new setup)');
    }
  }
}

export const registry = new PluginRegistry();

// Export for CLI to use
export default registry;
