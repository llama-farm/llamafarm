# Built-in Voice Samples for Chatterbox TTS

This directory contains voice samples for Chatterbox Turbo voice cloning.

## Download Voices

Run the download script to fetch high-quality voice samples from LibriSpeech:

```bash
cd runtimes/universal/voices
python download_voices.py
```

Or use the `--list` flag to see available voices:

```bash
python download_voices.py --list
```

## Available Voices

| Voice ID | Description | File |
|----------|-------------|------|
| `cb_male_calm` | Male narrator with calm, measured delivery | `male_calm.wav` |
| `cb_female_warm` | Female narrator with warm, expressive tone | `female_warm.wav` |
| `cb_male_energetic` | Male speaker with energetic delivery | `male_energetic.wav` |
| `cb_female_professional` | Female speaker with clear, professional tone | `female_professional.wav` |

## Usage in Config

```yaml
voice:
  tts:
    model: chatterbox-turbo
    voice: cb_male_calm  # Use a built-in voice
```

## Custom Voices

You can also add your own voice samples:

1. Add a WAV file (10-15 seconds of clear speech) to this directory or elsewhere
2. Reference it in your config:

```yaml
voice:
  tts:
    model: chatterbox-turbo
    voice: my_narrator  # Profile name
    voice_profiles:
      my_narrator:
        audio_path: "./voices/my_custom_voice.wav"
        description: "My custom narrator voice"
```

## Requirements

For the download script:
- `soundfile` (pip install soundfile)
- Internet connection (~340MB download from LibriSpeech)
