#!/usr/bin/env node
/**
 * sync-required-models.js
 *
 * Extracts required models from config templates and updates required-models.yaml
 * Run during electron build to ensure models are synced with config changes.
 *
 * Usage: node scripts/sync-required-models.js
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// Config file paths (relative to electron-app directory)
const CONFIG_PATHS = [
  '../config/templates/default.yaml',
  '../server/seeds/project_seed/llamafarm.yaml'
];

const OUTPUT_PATH = './required-models.yaml';

// Known model size estimates (in MB)
const SIZE_ESTIMATES = {
  // Language models (GGUF Q4_K_M quantization)
  'unsloth/gemma-3-1b-it-gguf': 700,
  'unsloth/Qwen3-1.7B-GGUF': 1200,
  'unsloth/Qwen3-0.6B-GGUF': 400,
  'unsloth/Qwen3-4B-GGUF': 2800,
  'unsloth/Qwen3-8B-GGUF': 5000,
  // Embedding models
  'nomic-ai/nomic-embed-text-v1.5': 550,
  'sentence-transformers/all-MiniLM-L6-v2': 90,
  'sentence-transformers/all-mpnet-base-v2': 420,
  // Default fallback
  'default': 500
};

/**
 * Extract model ID and quantization from model string
 * e.g., "unsloth/gemma-3-1b-it-gguf:Q4_K_M" -> { id: "unsloth/gemma-3-1b-it-gguf", quantization: "Q4_K_M" }
 */
function parseModelId(modelStr) {
  const [id, quantization] = modelStr.split(':');
  return { id, quantization: quantization || null };
}

/**
 * Get size estimate for a model
 */
function getSizeEstimate(modelId) {
  // Check exact match first
  if (SIZE_ESTIMATES[modelId]) {
    return SIZE_ESTIMATES[modelId];
  }

  // Check if it's a variant of a known model
  for (const [known, size] of Object.entries(SIZE_ESTIMATES)) {
    if (modelId.includes(known) || known.includes(modelId)) {
      return size;
    }
  }

  return SIZE_ESTIMATES.default;
}

/**
 * Get display name from model ID
 */
function getDisplayName(modelId) {
  // Extract the model name from the ID
  const parts = modelId.split('/');
  const name = parts[parts.length - 1];

  // Clean up common suffixes and format nicely
  return name
    .replace(/-gguf$/i, '')
    .replace(/-GGUF$/i, '')
    .replace(/-it$/i, '')
    .replace(/_/g, ' ')
    .replace(/-/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Extract models from a config file
 */
function extractModelsFromConfig(configPath) {
  const models = {
    language: [],
    embedding: []
  };

  try {
    const content = fs.readFileSync(configPath, 'utf8');
    const config = yaml.load(content);

    // Extract language models from runtime.models
    if (config.runtime?.models) {
      const runtimeModels = Array.isArray(config.runtime.models)
        ? config.runtime.models
        : Object.values(config.runtime.models);

      for (const model of runtimeModels) {
        if (model.provider === 'universal' && model.model) {
          const parsed = parseModelId(model.model);
          models.language.push({
            id: parsed.id,
            quantization: parsed.quantization,
            display_name: model.description || getDisplayName(parsed.id),
            source: path.basename(configPath)
          });
        }
      }
    }

    // Extract embedding models from rag.databases[].embedding_strategies[]
    if (config.rag?.databases) {
      for (const db of config.rag.databases) {
        if (db.embedding_strategies) {
          for (const strategy of db.embedding_strategies) {
            if (strategy.type === 'UniversalEmbedder' && strategy.config?.model) {
              const modelId = strategy.config.model;
              models.embedding.push({
                id: modelId,
                display_name: getDisplayName(modelId),
                source: path.basename(configPath)
              });
            }
          }
        }
      }
    }

    console.log(`  Found ${models.language.length} language models, ${models.embedding.length} embedding models`);

  } catch (error) {
    console.error(`  Error reading ${configPath}: ${error.message}`);
  }

  return models;
}

/**
 * Deduplicate models by ID
 */
function deduplicateModels(models) {
  const seen = new Map();

  for (const model of models) {
    const key = model.quantization ? `${model.id}:${model.quantization}` : model.id;
    if (!seen.has(key)) {
      seen.set(key, model);
    }
  }

  return Array.from(seen.values());
}

/**
 * Main sync function
 */
function syncRequiredModels() {
  console.log('Syncing required models from config templates...\n');

  const allLanguageModels = [];
  const allEmbeddingModels = [];

  // Process each config file
  for (const configPath of CONFIG_PATHS) {
    const fullPath = path.resolve(__dirname, '..', configPath);
    console.log(`Processing: ${configPath}`);

    if (!fs.existsSync(fullPath)) {
      console.log(`  Skipping (not found): ${fullPath}`);
      continue;
    }

    const models = extractModelsFromConfig(fullPath);
    allLanguageModels.push(...models.language);
    allEmbeddingModels.push(...models.embedding);
  }

  // Deduplicate
  const uniqueLanguageModels = deduplicateModels(allLanguageModels);
  const uniqueEmbeddingModels = deduplicateModels(allEmbeddingModels);

  console.log(`\nTotal unique models: ${uniqueLanguageModels.length} language, ${uniqueEmbeddingModels.length} embedding`);

  // Build output structure
  const output = {
    version: '1',
    models: []
  };

  // Add language models
  for (const model of uniqueLanguageModels) {
    output.models.push({
      id: model.id,
      ...(model.quantization && { quantization: model.quantization }),
      display_name: model.display_name,
      type: 'language',
      required: true,
      size_estimate_mb: getSizeEstimate(model.id)
    });
  }

  // Add embedding models
  for (const model of uniqueEmbeddingModels) {
    output.models.push({
      id: model.id,
      display_name: model.display_name,
      type: 'embedding',
      required: true,
      size_estimate_mb: getSizeEstimate(model.id)
    });
  }

  // Generate YAML with comments
  const header = `# Required models for LlamaFarm Desktop
# AUTO-GENERATED by scripts/sync-required-models.js
# DO NOT EDIT MANUALLY - changes will be overwritten
#
# Source files:
${CONFIG_PATHS.map(p => `#   - ${p}`).join('\n')}
#
# To update: npm run sync-models
# Generated: ${new Date().toISOString()}

`;

  const yamlContent = yaml.dump(output, {
    indent: 2,
    lineWidth: 120,
    quotingType: '"'
  });

  const outputPath = path.resolve(__dirname, '..', OUTPUT_PATH);
  fs.writeFileSync(outputPath, header + yamlContent);

  console.log(`\nWrote ${output.models.length} models to ${OUTPUT_PATH}`);

  // Print summary
  console.log('\nModels:');
  for (const model of output.models) {
    const quantStr = model.quantization ? `:${model.quantization}` : '';
    console.log(`  - ${model.id}${quantStr} (${model.type}, ~${model.size_estimate_mb}MB)`);
  }

  return output.models.length;
}

// Run if called directly
if (require.main === module) {
  try {
    const count = syncRequiredModels();
    console.log(`\nSync complete: ${count} models`);
    process.exit(0);
  } catch (error) {
    console.error('Error syncing models:', error);
    process.exit(1);
  }
}

module.exports = { syncRequiredModels };
