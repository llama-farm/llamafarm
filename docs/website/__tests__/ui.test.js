// UI Tests for Docusaurus Site
const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const path = require('path');

describe('Docusaurus UI Tests', () => {
  let browser;
  let page;
  let serverProcess;
  const serverUrl = 'http://localhost:3000';

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

    // Launch browser
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    page = await browser.newPage();
  }, 60000);

  // Cleanup after all tests
  afterAll(async () => {
    if (page) await page.close();
    if (browser) await browser.close();
    if (serverProcess) {
      serverProcess.kill('SIGTERM');
    }
  });

  test('Homepage loads successfully', async () => {
    const response = await page.goto(serverUrl, {
      waitUntil: 'networkidle2'
    });
    
    expect(response.status()).toBe(200);
    
    // Check page title
    const title = await page.title();
    expect(title).toBeTruthy();
    
    // Check for main heading
    const heading = await page.$eval('h1', el => el.textContent);
    expect(heading).toBeTruthy();
  });

  test('Navigation menu is visible and functional', async () => {
    await page.goto(serverUrl);
    
    // Check for navigation bar
    const navbar = await page.$('.navbar');
    expect(navbar).toBeTruthy();
    
    // Check for navigation items
    const navItems = await page.$$('.navbar__item');
    expect(navItems.length).toBeGreaterThan(0);
  });

  test('Documentation page loads correctly', async () => {
    await page.goto(`${serverUrl}/docs/intro`, {
      waitUntil: 'networkidle2'
    });
    
    // Check for documentation content
    const docContent = await page.$('.markdown');
    expect(docContent).toBeTruthy();
    
    // Check for sidebar
    const sidebar = await page.$('.theme-doc-sidebar-container');
    expect(sidebar).toBeTruthy();
  });

  test('Dark mode toggle works', async () => {
    await page.goto(serverUrl);
    
    // Find and click dark mode toggle
    const darkModeToggle = await page.$('[class*="colorModeToggle"]');
    if (darkModeToggle) {
      // Get initial theme
      const initialTheme = await page.evaluate(() => 
        document.documentElement.getAttribute('data-theme')
      );
      
      // Click toggle
      await darkModeToggle.click();
      await page.waitForTimeout(500);
      
      // Check theme changed
      const newTheme = await page.evaluate(() => 
        document.documentElement.getAttribute('data-theme')
      );
      
      expect(newTheme).not.toBe(initialTheme);
    }
  });

  test('Search functionality is available', async () => {
    await page.goto(serverUrl);
    
    // Look for search button or input
    const searchButton = await page.$('[class*="searchButton"]');
    const searchBox = await page.$('[class*="searchBox"]');
    
    expect(searchButton || searchBox).toBeTruthy();
  });

  test('Blog section is accessible', async () => {
    await page.goto(`${serverUrl}/blog`, {
      waitUntil: 'networkidle2'
    });
    
    // Check for blog posts
    const blogPosts = await page.$$('article');
    expect(blogPosts.length).toBeGreaterThan(0);
  });

  test('Mobile responsive design', async () => {
    // Set mobile viewport
    await page.setViewport({ width: 375, height: 667 });
    await page.goto(serverUrl);
    
    // Check for mobile menu button
    const mobileMenuButton = await page.$('[class*="toggle"]');
    expect(mobileMenuButton).toBeTruthy();
    
    // Click mobile menu
    if (mobileMenuButton) {
      await mobileMenuButton.click();
      await page.waitForTimeout(500);
      
      // Check if menu is visible
      const mobileMenu = await page.$('[class*="sidebar"]');
      expect(mobileMenu).toBeTruthy();
    }
  });

  test('Static assets load correctly', async () => {
    await page.goto(serverUrl);
    
    // Check for logo
    const logo = await page.$('img[class*="logo"]');
    if (logo) {
      const logoSrc = await logo.evaluate(el => el.src);
      const logoResponse = await page.goto(logoSrc);
      expect(logoResponse.status()).toBe(200);
    }
    
    // Navigate back
    await page.goto(serverUrl);
  });

  test('Footer is present and contains links', async () => {
    await page.goto(serverUrl);
    
    // Scroll to bottom to ensure footer is loaded
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);
    
    // Check for footer
    const footer = await page.$('footer');
    expect(footer).toBeTruthy();
    
    // Check for footer links
    const footerLinks = await page.$$('footer a');
    expect(footerLinks.length).toBeGreaterThan(0);
  });

  test('Page performance metrics', async () => {
    await page.goto(serverUrl, {
      waitUntil: 'networkidle2'
    });
    
    // Get performance metrics
    const metrics = await page.metrics();
    
    // Basic performance checks
    expect(metrics.TaskDuration).toBeLessThan(3000); // Page should load within 3 seconds
    expect(metrics.JSHeapUsedSize).toBeLessThan(50 * 1024 * 1024); // Less than 50MB heap usage
  });
});