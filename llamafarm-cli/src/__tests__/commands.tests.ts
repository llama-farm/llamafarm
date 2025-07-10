
// src/__tests__/commands.test.ts
import { validateModelName, validateDevice, validateQuantization } from '../utils/validators';
import { getModelInfo, validateModelRequirements } from '../models/registry';

describe('Model Validation', () => {
  describe('validateModelName', () => {
    it('should accept valid model names', () => {
      expect(validateModelName('llama3-8b')).toBe(true);
      expect(validateModelName('mixtral-8x7b')).toBe(true);
      expect(validateModelName('phi-2')).toBe(true);
    });

    it('should reject invalid model names', () => {
      expect(validateModelName('Llama3-8B')).toBe(false); // uppercase
      expect(validateModelName('llama3_8b')).toBe(false); // underscore
      expect(validateModelName('llama3--8b')).toBe(false); // double dash
    });
  });

  describe('validateDevice', () => {
    it('should accept valid devices', () => {
      expect(validateDevice('mac')).toBe(true);
      expect(validateDevice('raspberry-pi')).toBe(true);
    });

    it('should reject invalid devices', () => {
      expect(validateDevice('iphone')).toBe(false);
      expect(validateDevice('android')).toBe(false);
    });
  });
});

describe('Model Registry', () => {
  it('should retrieve model information', () => {
    const model = getModelInfo('llama3-8b');
    expect(model).toBeDefined();
    expect(model?.name).toBe('Llama 3 8B');
    expect(model?.requirements.minRam).toBe(6);
  });

  it('should validate model requirements', () => {
    const systemInfo = {
      totalMemory: 8,
      diskSpace: 50
    };

    const result = validateModelRequirements('llama3-8b', systemInfo);
    expect(result.valid).toBe(true);
    expect(result.warnings).toHaveLength(0);
  });

  it('should warn about insufficient resources', () => {
    const systemInfo = {
      totalMemory: 4,
      diskSpace: 50
    };

    const result = validateModelRequirements('llama3-8b', systemInfo);
    expect(result.valid).toBe(false);
    expect(result.warnings[0]).toContain('Insufficient RAM');
  });
});

// src/config/defaults.ts
export const DEFAULT_CONFIG = {
  version: '1.0.0',
  defaults: {
    device: 'mac',
    quantization: 'q4_0',
    gpu: false,
    vectorDb: 'chroma',
    agentFramework: 'langchain',
    port: 8080
  },
  paths: {
    harvest: './harvests',
    models: './harvests/models',
    agents: './harvests/agents',
    data: './harvests/data',
    logs: './logs'
  },
  advanced: {
    chunkSize: 512,
    chunkOverlap: 50,
    embeddingModel: 'all-MiniLM-L6-v2',
    contextWindow: 2048,
    autoUpdate: true,
    telemetry: true
  },
  limits: {
    maxConcurrentModels: 2,
    maxMemoryUsage: '80%',
    maxDiskUsage: '90%'
  }
};