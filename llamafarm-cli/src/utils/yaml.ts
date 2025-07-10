import * as fs from 'fs-extra';
import * as yaml from 'js-yaml';

export async function loadYamlConfig(filepath: string): Promise<any> {
  const content = await fs.readFile(filepath, 'utf-8');
  const parsed = yaml.load(content) as any;
  
  // Flatten the config structure for easier access
  return {
    device: parsed.deployment?.device || 'mac',
    agent: parsed.agent?.name || 'chat-basic',
    rag: parsed.rag?.enabled ? 'enabled' : 'disabled',
    database: parsed.database?.type || 'vector',
    gpu: parsed.deployment?.gpu || false,
    quantize: parsed.model?.quantization || 'q4_0',
    ...parsed
  };
}