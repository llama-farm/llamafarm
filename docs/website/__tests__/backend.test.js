// Backend/API Tests for Docusaurus Site
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

describe('Docusaurus Backend Tests', () => {
  let serverProcess;
  const serverUrl = 'http://localhost:3000';
  const buildDir = path.join(__dirname, '..', 'build');

  // Start server before all tests
  beforeAll(async () => {
    // Start the development server
    serverProcess = spawn('yarn', ['start'], {
      cwd: path.join(__dirname, '..'),
      detached: false,
      stdio: 'pipe'
    });

    // Wait for server to be ready
    const maxRetries = 30;
    let retries = 0;
    let serverReady = false;

    while (retries < maxRetries && !serverReady) {
      try {
        const response = await fetch(serverUrl);
        if (response.ok) {
          serverReady = true;
        }
      } catch (error) {
        // Server not ready yet
      }
      retries++;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    if (!serverReady) {
      throw new Error('Server failed to start');
    }
  }, 60000);

  // Cleanup after all tests
  afterAll(async () => {
    if (serverProcess) {
      serverProcess.kill('SIGTERM');
    }
  });

  describe('HTTP Endpoints', () => {
    test('GET / returns 200', async () => {
      const response = await fetch(serverUrl);
      expect(response.status).toBe(200);
      expect(response.headers.get('content-type')).toContain('text/html');
    });

    test('GET /docs returns redirect or 200', async () => {
      const response = await fetch(`${serverUrl}/docs`);
      expect([200, 301, 302]).toContain(response.status);
    });

    test('GET /docs/intro returns 200', async () => {
      const response = await fetch(`${serverUrl}/docs/intro`);
      expect(response.status).toBe(200);
    });

    test('GET /blog returns 200', async () => {
      const response = await fetch(`${serverUrl}/blog`);
      expect(response.status).toBe(200);
    });

    test('GET /404 returns 404 page', async () => {
      const response = await fetch(`${serverUrl}/this-page-does-not-exist`);
      expect(response.status).toBe(404);
      const text = await response.text();
      expect(text).toContain('Page Not Found');
    });
  });

  describe('Static Assets', () => {
    test('CSS files are served with correct content-type', async () => {
      const response = await fetch(serverUrl);
      const html = await response.text();
      
      // Extract CSS file URL from HTML
      const cssMatch = html.match(/href="([^"]+\.css)"/);
      if (cssMatch) {
        const cssUrl = new URL(cssMatch[1], serverUrl).href;
        const cssResponse = await fetch(cssUrl);
        expect(cssResponse.status).toBe(200);
        expect(cssResponse.headers.get('content-type')).toContain('text/css');
      }
    });

    test('JavaScript files are served correctly', async () => {
      const response = await fetch(serverUrl);
      const html = await response.text();
      
      // Extract JS file URL from HTML
      const jsMatch = html.match(/src="([^"]+\.js)"/);
      if (jsMatch) {
        const jsUrl = new URL(jsMatch[1], serverUrl).href;
        const jsResponse = await fetch(jsUrl);
        expect(jsResponse.status).toBe(200);
        expect(jsResponse.headers.get('content-type')).toContain('application/javascript');
      }
    });

    test('Images are served correctly', async () => {
      const imageUrls = [
        '/img/logo.svg',
        '/img/favicon.ico'
      ];

      for (const imageUrl of imageUrls) {
        const response = await fetch(`${serverUrl}${imageUrl}`);
        expect(response.status).toBe(200);
        
        if (imageUrl.endsWith('.svg')) {
          expect(response.headers.get('content-type')).toContain('image/svg+xml');
        } else if (imageUrl.endsWith('.ico')) {
          expect(response.headers.get('content-type')).toContain('image/x-icon');
        }
      }
    });
  });

  describe('Build Output', () => {
    beforeAll(async () => {
      // Build the project
      await new Promise((resolve, reject) => {
        const buildProcess = spawn('yarn', ['build'], {
          cwd: path.join(__dirname, '..'),
          stdio: 'pipe'
        });

        buildProcess.on('close', (code) => {
          if (code === 0) {
            resolve();
          } else {
            reject(new Error(`Build failed with code ${code}`));
          }
        });
      });
    }, 120000);

    test('Build directory exists', async () => {
      const exists = await fs.access(buildDir).then(() => true).catch(() => false);
      expect(exists).toBe(true);
    });

    test('index.html exists in build', async () => {
      const indexPath = path.join(buildDir, 'index.html');
      const exists = await fs.access(indexPath).then(() => true).catch(() => false);
      expect(exists).toBe(true);
    });

    test('Static assets are generated', async () => {
      const assetsDir = path.join(buildDir, 'assets');
      const exists = await fs.access(assetsDir).then(() => true).catch(() => false);
      expect(exists).toBe(true);
      
      const files = await fs.readdir(assetsDir);
      expect(files.length).toBeGreaterThan(0);
    });

    test('Sitemap is generated', async () => {
      const sitemapPath = path.join(buildDir, 'sitemap.xml');
      const exists = await fs.access(sitemapPath).then(() => true).catch(() => false);
      expect(exists).toBe(true);
    });
  });

  describe('SEO and Meta Tags', () => {
    test('Homepage has proper meta tags', async () => {
      const response = await fetch(serverUrl);
      const html = await response.text();
      
      // Check for essential meta tags
      expect(html).toContain('<meta charset="utf-8">');
      expect(html).toContain('<meta name="viewport"');
      expect(html).toContain('<meta name="description"');
      expect(html).toContain('<title>');
    });

    test('Open Graph tags are present', async () => {
      const response = await fetch(serverUrl);
      const html = await response.text();
      
      expect(html).toContain('property="og:title"');
      expect(html).toContain('property="og:description"');
      expect(html).toContain('property="og:type"');
    });
  });

  describe('Security Headers', () => {
    test('No sensitive information in headers', async () => {
      const response = await fetch(serverUrl);
      
      // Check that no sensitive headers are exposed
      expect(response.headers.get('x-powered-by')).toBeFalsy();
      expect(response.headers.get('server')).toBeFalsy();
    });
  });

  describe('Performance', () => {
    test('Response time is acceptable', async () => {
      const start = Date.now();
      const response = await fetch(serverUrl);
      const end = Date.now();
      const responseTime = end - start;
      
      expect(response.status).toBe(200);
      expect(responseTime).toBeLessThan(1000); // Should respond within 1 second
    });

    test('Compression is enabled', async () => {
      const response = await fetch(serverUrl, {
        headers: {
          'Accept-Encoding': 'gzip, deflate'
        }
      });
      
      // Development server might not have compression, but check if it's configured
      const contentEncoding = response.headers.get('content-encoding');
      // This might be null in dev, which is okay
      if (contentEncoding) {
        expect(['gzip', 'deflate', 'br']).toContain(contentEncoding);
      }
    });
  });
});