// src/utils/config.ts
import * as fs from 'fs-extra';
import * as path from 'path';
import * as os from 'os';

export function getConfigPath(): string {
  return path.join(os.homedir(), '.llamafarm', 'config.json');
}

export async function loadConfig(): Promise<any> {
  const configPath = getConfigPath();
  
  if (await fs.pathExists(configPath)) {
    return await fs.readJSON(configPath);
  }
  
  // Return default config
  return {
    version: '1.0.0',
    farmer: {
      name: 'farmer',
      created: new Date().toISOString()
    },
    defaults: {
      device: 'mac',
      quantization: 'q4_0',
      gpu: false,
      vectorDb: 'chroma',
      agentFramework: 'langchain'
    },
    paths: {
      harvest: './harvests',
      models: './harvests/models',
      agents: './harvests/agents',
      data: './harvests/data'
    }
  };
}

export async function saveConfig(config: any): Promise<void> {
  const configPath = getConfigPath();
  await fs.ensureDir(path.dirname(configPath));
  await fs.writeJSON(configPath, config, { spaces: 2 });
}
