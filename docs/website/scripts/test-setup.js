// Test setup configuration for Docusaurus testing
const puppeteer = require('puppeteer');

module.exports = {
  // Test configuration
  testTimeout: 30000,
  
  // Server configuration
  serverUrl: 'http://localhost:3000',
  
  // Browser setup
  async setupBrowser() {
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    return browser;
  },
  
  // Common test utilities
  async waitForServer(url, maxRetries = 30) {
    let retries = 0;
    while (retries < maxRetries) {
      try {
        const response = await fetch(url);
        if (response.ok) {
          return true;
        }
      } catch (error) {
        // Server not ready yet
      }
      retries++;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error('Server failed to start within timeout');
  }
};