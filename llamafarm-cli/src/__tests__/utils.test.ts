// src/__tests__/utils.test.ts

import { getPort } from '../utils/portfinder';
import { loadYamlConfig } from '../utils/yaml';
import * as fs from 'fs-extra';
import * as path from 'path';

describe('Utility Functions', () => {
  describe('getPort', () => {
    it('should return an available port', async () => {
      const port = await getPort();
      expect(port).toBeGreaterThanOrEqual(8080);
      expect(port).toBeLessThan(65536);
    });

    it('should return different ports when called multiple times', async () => {
      const port1 = await getPort();
      const port2 = await getPort(port1 + 1);
      expect(port2).not.toBe(port1);
    });
  });

  describe('loadYamlConfig', () => {
    const testYaml = `
model:
  name: test-model
  quantization: q4_0
agent:
  name: test-agent
deployment:
  device: mac
  gpu: true
`;

    beforeEach(async () => {
      await fs.writeFile('test-config.yaml', testYaml);
    });

    afterEach(async () => {
      await fs.remove('test-config.yaml');
    });

    it('should load and parse YAML configuration', async () => {
      const config = await loadYamlConfig('test-config.yaml');
      expect(config.device).toBe('mac');
      expect(config.agent).toBe('test-agent');
      expect(config.gpu).toBe(true);
    });
  });
});

---
