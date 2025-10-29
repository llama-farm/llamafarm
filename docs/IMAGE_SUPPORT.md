# Image Support in LlamaFarm

LlamaFarm now supports multimodal input/output with images following the OpenAI Vision API format.

## Features

- **Auto-detection**: CLI automatically detects image files in chat inputs
- **OpenAI-compatible**: Uses standard multimodal message format
- **Multiple providers**: Works with Ollama (llava, bakllava), OpenAI (gpt-4o, gpt-4-vision), and other OpenAI-compatible vision APIs
- **Output handling**: Automatically saves generated images to disk

## Quick Start

### Using Images with `lf chat`

```bash
# Send an image file directly
lf chat photo.jpg

# Send an image with a prompt
lf chat "what's in this image? photo.jpg"

# Use vision model to describe
lf chat "describe: interior.png"
```

### Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WebP
- BMP
- TIFF

## Configuration

### Using Ollama with Vision Models

```yaml
version: v1
name: vision-project
namespace: my-namespace

runtime:
  models:
    - name: vision
      provider: ollama
      model: llava:13b  # or llava:7b, bakllava
      base_url: http://localhost:11434/v1

prompts:
  - role: system
    content: You are a helpful vision assistant that can analyze images.
```

### Using OpenAI

```yaml
version: v1
name: vision-project
namespace: my-namespace

runtime:
  models:
    - name: vision
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}

prompts:
  - role: system
    content: You are a helpful vision assistant.
```

## How It Works

### Input Processing

1. CLI detects if input contains or is an image file
2. Image is read and base64-encoded
3. Multimodal message is constructed following OpenAI format:
   ```json
   {
     "role": "user",
     "content": [
       {"type": "text", "text": "What's in this image?"},
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
     ]
   }
   ```
4. Message is sent to server and passed to runtime provider

### Output Processing

1. Server streams response from vision model
2. CLI checks if response contains base64-encoded images
3. Images are automatically decoded and saved to current directory
4. Filename format: `output_<pid>.<ext>`

## Examples

### Image Description

```bash
# Pull a vision model
ollama pull llava:13b

# Start LlamaFarm
lf start

# Describe an image
lf chat photo.jpg
```

Example output:
```
📸 Sending image: photo.jpg
This image shows a modern kitchen with white cabinets,
stainless steel appliances, and marble countertops.
Natural light enters through large windows...
```

### Image Analysis with Context

```bash
lf chat "Analyze the lighting in this real estate photo: interior.jpg"
```

### Multiple Images (Future)

```bash
# Not yet implemented, but infrastructure supports it
lf chat "Compare these photos: before.jpg after.jpg"
```

## Technical Details

### Message Format

LlamaFarm uses the OpenAI Vision API message format:

```typescript
interface MessageContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: {
    url: string;        // base64 data URL or http(s) URL
    detail?: "auto" | "low" | "high";
  };
}

interface Message {
  role: "system" | "user" | "assistant";
  content: string | MessageContentPart[];
}
```

### Base64 Encoding

Images are automatically encoded to base64 and embedded in data URLs:
```
data:image/jpeg;base64,/9j/4AAQSkZJRg...
```

This works with all OpenAI-compatible vision APIs including:
- Ollama with vision models
- OpenAI gpt-4o / gpt-4-vision
- Custom OpenAI-compatible endpoints

### Provider Compatibility

| Provider | Vision Support | Models |
|----------|----------------|--------|
| Ollama | ✅ Yes | llava, bakllava, llava-phi3 |
| OpenAI | ✅ Yes | gpt-4o, gpt-4-vision, gpt-4-turbo |
| Lemonade | ⚠️  Depends on backend | Check model capabilities |

## Limitations

- Maximum image size: 50MB (configurable in CLI)
- Large images may take longer to encode and transmit
- Image generation output format depends on model capabilities
- RAG does not currently index images (text-only)

## Troubleshooting

### "Image not found"

Make sure the file path is correct and the file exists:
```bash
ls -la photo.jpg
file photo.jpg  # Check file type
```

### "Model doesn't support vision"

Ensure you're using a vision-capable model:
```bash
# Ollama
ollama list  # Check if llava or bakllava is available
ollama pull llava:13b

# OpenAI
# Use gpt-4o or gpt-4-vision, not gpt-3.5 or gpt-4
```

### "Failed to encode image"

Check image file format and size:
```bash
file photo.jpg
ls -lh photo.jpg  # Check size
```

### Images not being detected

Use explicit file references:
```bash
# Clear file path
lf chat ./photo.jpg

# Or with absolute path
lf chat /full/path/to/photo.jpg
```

## Future Enhancements

- [ ] Image generation (Stable Diffusion, DALL-E)
- [ ] Image processing tools (resize, crop, filter)
- [ ] Multi-image comparison
- [ ] Image indexing in RAG
- [ ] Streaming image output
- [ ] Custom image processing pipelines

## See Also

- [OpenAI Vision API Documentation](https://platform.openai.com/docs/guides/vision)
- [Ollama Vision Models](https://ollama.com/library/llava)
- [LlamaFarm Configuration Guide](../docs/website/docs/configuration/index.md)
