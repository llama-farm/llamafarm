import { BaseFieldPlugin } from '../../base';
import * as os from 'os';

export class MacFieldPlugin extends BaseFieldPlugin {
  name = 'mac-field';
  version = '1.0.0';
  platform = 'darwin';
  description = 'macOS deployment field with Metal acceleration support';
  author = 'LlamaFarm Team';

  constraints = {
    minMemory: 8,
    minDisk: 20,
    requiredTools: ['xcode-select'],
    supportedModels: ['llama3-8b', 'mixtral-8x7b', 'phi-2']
  };

  async setup(): Promise<void> {
    await super.setup();
    console.log(`✅ macOS ${os.release()} detected`);
    console.log(`🖥️  Architecture: ${os.arch()}`);
  }

  optimize(config: any): any {
    const optimized = { ...config };
    if (os.arch() === 'arm64') {
      optimized.acceleration = 'metal';
    }
    return optimized;
  }
}

export default new MacFieldPlugin();
