import * as yaml from 'js-yaml';
import * as fs from 'fs-extra';

export async function loadYamlConfig(filePath: string): Promise<any> {
  const content = await fs.readFile(filePath, 'utf-8');
  const config = yaml.load(content) as any;
  
  // Transform YAML structure to match command options
  return {
    device: config.deployment?.device,
    agent: config.agent?.name,
    rag: config.rag?.enabled ? 'enabled' : 'disabled',
    database: config.database?.type,
    port: config.deployment?.port,
    gpu: config.deployment?.gpu,
    quantize: config.model?.quantization
  };
}