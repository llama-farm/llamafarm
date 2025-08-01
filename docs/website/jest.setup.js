// Jest setup file
jest.setTimeout(30000);

// Mock fetch if not available in Node environment
if (!global.fetch) {
  global.fetch = require('node-fetch');
}