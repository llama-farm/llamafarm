/**
 * Base classes for LlamaFarm plugins
 */

import { Plugin, FieldPlugin, EquipmentPlugin, PipePlugin } from './registry';

export abstract class BasePlugin implements Plugin {
  abstract name: string;
  abstract type: 'field' | 'equipment' | 'pipe';
  abstract version: string;
  abstract description: string;
  author?: string;
  dependencies?: string[];

  async setup(): Promise<void> {
    console.log(`🌱 Setting up ${this.name}...`);
  }

  async teardown(): Promise<void> {
    console.log(`🧹 Tearing down ${this.name}...`);
  }
}

export abstract class BaseFieldPlugin extends BasePlugin implements FieldPlugin {
  type: 'field' = 'field';
  abstract platform: string;
  abstract constraints: {
    minMemory?: number;
    minDisk?: number;
    requiredTools?: string[];
    supportedModels?: string[];
  };

  abstract optimize(config: any): any;
}

export abstract class BaseEquipmentPlugin extends BasePlugin implements EquipmentPlugin {
  type: 'equipment' = 'equipment';
  abstract category: 'database' | 'rag-pipeline' | 'model-runtime' | 'tool';
  
  abstract install(): Promise<void>;
  abstract configure(options: any): Promise<void>;
  abstract test(): Promise<boolean>;
}

export abstract class BasePipePlugin extends BasePlugin implements PipePlugin {
  type: 'pipe' = 'pipe';
  abstract protocol: string;
  
  abstract createServer(options: any): any;
  abstract createClient(options: any): any;
}
