/**
 * Example: Using LlamaFarm Plugins
 */

import { registry } from '../src/plugins/registry';

async function example() {
  // Load all plugins
  await registry.loadAll();
  
  // Use a field plugin
  const macField = registry.get('mac-field');
  if (macField) {
    await macField.setup();
  }
  
  // Use an equipment plugin
  const demoDb = registry.get('demo-database');
  if (demoDb && demoDb.type === 'equipment') {
    await demoDb.install();
    await demoDb.configure({ dimensions: 384 });
  }
  
  // Use a pipe plugin
  const wsPipe = registry.get('websocket-pipe');
  if (wsPipe && wsPipe.type === 'pipe') {
    const server = wsPipe.createServer({
      port: 8080,
      onMessage: (msg: any) => console.log('Received:', msg)
    });
  }
}

example().catch(console.error);
