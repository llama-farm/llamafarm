/**
 * Custom Windows signing script for Azure Trusted Signing
 *
 * This script is called by electron-builder to sign Windows executables.
 * It fixes path quoting issues with the Invoke-TrustedSigning PowerShell cmdlet.
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const path = require('path');

const execAsync = promisify(exec);

/**
 * Sign a Windows executable using Azure Trusted Signing
 *
 * @param {Object} configuration - Signing configuration from electron-builder
 * @param {string} configuration.path - Path to the file to sign
 * @param {string} configuration.name - Name of the file
 * @param {string} configuration.hash - Hash algorithm (e.g., 'sha256')
 * @param {string} configuration.isNest - Whether this is a nested signing operation
 */
exports.default = async function (configuration) {
  const { path: filePath, name } = configuration;

  // Get Azure credentials from environment
  const endpoint = process.env.AZURE_SIGNING_ENDPOINT || 'https://eus.codesigning.azure.net/';
  const certificateProfile = process.env.AZURE_SIGNING_CERTIFICATE_PROFILE || 'llamafarm-app';
  const accountName = process.env.AZURE_SIGNING_ACCOUNT_NAME || 'LlamaFarm';

  // Validate required environment variables
  if (!process.env.AZURE_TENANT_ID || !process.env.AZURE_CLIENT_ID || !process.env.AZURE_CLIENT_SECRET) {
    console.error('❌ Azure credentials not found in environment');
    console.error('Required: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET');
    throw new Error('Azure credentials not configured');
  }

  console.log(`\n📝 Signing ${name}...`);
  console.log(`   Path: ${filePath}`);
  console.log(`   Endpoint: ${endpoint}`);
  console.log(`   Profile: ${certificateProfile}`);
  console.log(`   Account: ${accountName}`);

  // Normalize the path (removes extra quotes and converts to Windows format)
  const normalizedPath = path.resolve(filePath).replace(/\//g, '\\');

  // Build the PowerShell command with proper escaping
  // NOTE: We import the module first, then call Invoke-TrustedSigning
  // We don't quote the path in -Files because Invoke-TrustedSigning
  // expects an unquoted path or an array of paths
  const psCommand = `Import-Module TrustedSigning -ErrorAction Stop; Invoke-TrustedSigning -Endpoint '${endpoint}' -CertificateProfileName '${certificateProfile}' -CodeSigningAccountName '${accountName}' -TimestampRfc3161 'http://timestamp.acs.microsoft.com' -TimestampDigest 'SHA256' -FileDigest 'SHA256' -Files '${normalizedPath}'`;

  const command = `pwsh.exe -NoProfile -NonInteractive -Command "${psCommand}"`;

  console.log(`\n🔐 Executing signing command...`);

  try {
    // Execute with retry logic (Azure Trusted Signing can be flaky)
    const maxRetries = 3;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 1) {
          console.log(`\n⚠️  Retry attempt ${attempt}/${maxRetries}...`);
          // Wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 2000));
        }

        const { stdout, stderr } = await execAsync(command, {
          maxBuffer: 10 * 1024 * 1024, // 10MB buffer
          timeout: 5 * 60 * 1000 // 5 minute timeout
        });

        if (stdout) {
          console.log(stdout);
        }
        if (stderr) {
          console.error(stderr);
        }

        console.log(`✅ Successfully signed ${name}`);
        return; // Success!

      } catch (error) {
        lastError = error;
        console.error(`❌ Attempt ${attempt} failed: ${error.message}`);

        if (error.stdout) {
          console.log('STDOUT:', error.stdout);
        }
        if (error.stderr) {
          console.error('STDERR:', error.stderr);
        }

        // Don't retry if it's clearly a configuration error
        if (error.message.includes('not rooted') ||
          error.message.includes('credentials') ||
          error.message.includes('authentication')) {
          throw error;
        }
      }
    }

    // All retries failed
    throw lastError;

  } catch (error) {
    console.error(`\n❌ Failed to sign ${name} after ${maxRetries} attempts`);
    console.error('Error:', error.message);

    if (error.stdout) {
      console.log('STDOUT:', error.stdout);
    }
    if (error.stderr) {
      console.error('STDERR:', error.stderr);
    }

    throw error;
  }
};
