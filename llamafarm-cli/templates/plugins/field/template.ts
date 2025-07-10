import { BaseFieldPlugin } from '../../base';
import * as os from 'os';

export class {{NAME}}FieldPlugin extends BaseFieldPlugin {
  name = '{{LOWERCASE_NAME}}-field';
  version = '1.0.0';
  platform = '{{PLATFORM}}'; // darwin, linux, win32
  description = '{{DESCRIPTION}}';
  author = '{{AUTHOR}}';

  constraints = {
    minMemory: 8, // GB
    minDisk: 20, // GB
    requiredTools: [], // e.g., ['docker', 'nvidia-smi']
    supportedModels: [] // e.g., ['llama3-8b', 'mixtral-8x7b']
  };

  async setup(): Promise<void> {
    await super.setup();
    
    // Check system requirements
    console.log(`🌾 Setting up ${this.name}...`);
    
    // Add your platform-specific setup here
  }

  optimize(config: any): any {
    // Platform-specific optimizations
    const optimized = { ...config };
    
    // Add your optimizations here
    // e.g., GPU settings, thread count, memory settings
    
    return optimized;
  }
}

export default new {{NAME}}FieldPlugin();
