# Image Vision Example

This example demonstrates LlamaFarm's image processing capabilities using vision models.

## Prerequisites

1. **Install LlamaFarm CLI**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```

2. **Install Ollama**
   - Download from https://ollama.com/download
   - Or use Docker

3. **Pull a vision model**
   ```bash
   ollama pull llava:13b
   # or for a smaller model:
   # ollama pull llava:7b
   ```

## Quick Start

```bash
# Navigate to this example
cd examples/image-vision

# Start LlamaFarm services
lf start

# Test with any image
lf chat yourimage.jpg

# Or with a prompt
lf chat "What's in this image? yourimage.jpg"
```

## Example Usage

### Basic Image Description

```bash
lf chat photo.jpg
```

Output:
```
📸 Sending image: photo.jpg
This image shows...
```

### Specific Analysis

```bash
lf chat "Analyze the lighting and composition: photo.jpg"
lf chat "What objects are visible in: screenshot.png"
lf chat "Describe the colors in: artwork.jpg"
```

### Different Image Formats

```bash
# JPEG
lf chat photo.jpg

# PNG
lf chat screenshot.png

# WebP
lf chat image.webp

# GIF (first frame)
lf chat animation.gif
```

## Configuration

The `llamafarm.yaml` in this directory configures:

- **Provider**: Ollama (local)
- **Model**: llava:13b (vision-capable)
- **Prompt**: Vision assistant system prompt
- **Format**: Unstructured (required for vision)

### Using Different Models

Edit `llamafarm.yaml` to use different models:

```yaml
# Smaller model (faster, less accurate)
model: llava:7b

# Larger model (slower, more accurate)
model: llava:34b

# Alternative model
model: bakllava
```

### Using OpenAI Instead

Replace the runtime section:

```yaml
runtime:
  models:
    - name: vision-cloud
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}
      prompt_format: unstructured
```

Then set your API key:
```bash
export OPENAI_API_KEY=sk-...
lf chat photo.jpg
```

## Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WebP
- BMP
- TIFF

## Use Cases

### Photo Analysis

```bash
lf chat "Is this photo well-lit for a real estate listing? interior.jpg"
```

### Object Detection

```bash
lf chat "List all visible objects: room.jpg"
```

### Text Extraction (OCR)

```bash
lf chat "What text is visible in this image? document.jpg"
```

### Comparison

```bash
lf chat "Describe the differences: before.jpg after.jpg"
```

## Tips

1. **Image Quality**: Higher resolution images provide better results but take longer
2. **Clear Questions**: Be specific about what you want to know
3. **Context**: Provide context in your prompt for better responses
4. **Model Size**: llava:7b is faster, llava:13b is more accurate

## Troubleshooting

### Model Not Found

```bash
ollama list
ollama pull llava:13b
```

### Connection Error

Make sure Ollama is running:
```bash
ollama serve
```

Or check Docker:
```bash
docker ps | grep ollama
```

### Image Not Detected

Use explicit paths:
```bash
lf chat ./photo.jpg
lf chat /full/path/to/image.jpg
```

## Next Steps

- Try different vision models
- Experiment with different prompts
- Use with your own images
- See [IMAGE_SUPPORT.md](../../docs/IMAGE_SUPPORT.md) for full documentation

## Notes

- Images are base64-encoded and sent to the model
- Large images may take longer to process
- Vision models work best with clear, well-lit images
- Some models have better text recognition than others
