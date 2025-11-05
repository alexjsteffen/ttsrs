# ttsrs - Text-to-Speech CLI Tool

A Rust-based command-line tool for converting text to speech using OpenAI's TTS API, ElevenLabs API, or a compatible custom endpoint.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Voice Options](#voice-options)
  - [Audio Formats](#audio-formats)
- [Examples](#examples)
- [Environment Variables](#environment-variables)
- [Technical Details](#technical-details)

## Features

- 🎯 Easy-to-use command-line interface
- 🔊 High-quality text-to-speech conversion using OpenAI's API, ElevenLabs API, or custom endpoints
- 🌐 Support for multiple TTS providers (OpenAI and ElevenLabs)
- 📝 Supports large text files through automatic chunking
- 🎨 Multiple voice options and audio formats
- ⚡ Adjustable speaking speed (OpenAI) and voice settings (ElevenLabs)
- 🔄 Interactive mode for selecting voices and formats (when defaults are used)
- 📁 Organized output with automatic file management
- 🚀 Progress indicators during conversion

## Installation

```bash
cargo install ttsrs
```

## Prerequisites

- Rust (latest stable version)
- ffmpeg (for audio file combining)
- API key for the target TTS service
- Internet connection

## Usage

### Command-Line Arguments

```bash
ttsrs [OPTIONS] <INPUT_FILE>
```

| Argument       | Description                                     | Default                         |
| -------------- | ----------------------------------------------- | ------------------------------- |
| `<INPUT_FILE>` | Path to the input text file                     | - (Required, or prompted)       |
| `--provider`   | TTS provider (`openai` or `elevenlabs`)         | `openai`                        |
| `--model`, `-m`  | TTS model to use                                | `tts-1-hd` (OpenAI)             |
| `--voice`, `-v`  | Voice selection (OpenAI only)                   | `alloy` (prompted if default)   |
| `--format`, `-f` | Output audio format                             | `flac` (OpenAI), `mp3_44100_128` (ElevenLabs) |
| `--speed`      | Speaking speed (0.25 - 4.0, OpenAI only)        | `1.0`                           |
| `--apikey`, `-a` | API key for the TTS service                   | - (Required, env var, or prompted) |
| `--endpoint-url`| Custom API endpoint URL (e.g., for local AI)  | Provider-specific default       |
| `--elevenlabs-voice-id` | ElevenLabs voice ID (required for ElevenLabs) | - |
| `--elevenlabs-model` | ElevenLabs model ID                        | `eleven_turbo_v2_5`             |
| `--elevenlabs-stability` | Voice stability (0.0 - 1.0)            | `0.5`                           |
| `--elevenlabs-similarity` | Voice similarity boost (0.0 - 1.0)    | `0.75`                          |

### Voice Options

#### OpenAI Voices

Available voices (supported by OpenAI's TTS API):

- **alloy** - A versatile, well-balanced voice
- **echo** - Clear and professional, ideal for announcements
- **fable** - Warm and engaging, perfect for storytelling
- **onyx** - Deep and authoritative
- **nova** - Young and energetic
- **shimmer** - Soft and soothing
- **ash** - Clear and conversational
- **coral** - Warm and friendly
- **sage** - Calm and measured

#### ElevenLabs Voices

ElevenLabs uses unique voice IDs instead of names. To get available voices:
1. Visit the [ElevenLabs Voice Library](https://elevenlabs.io/voice-library)
2. Or use the ElevenLabs API: `GET https://api.elevenlabs.io/v1/voices`
3. Use the voice ID with `--elevenlabs-voice-id` flag

### Audio Formats

#### OpenAI Formats

Supported output formats (may vary depending on the endpoint):
- `flac` (default) - Lossless audio compression
- `mp3` - Common compressed audio format
- `wav` - Uncompressed audio
- `pcm` - Raw audio data
- `opus` - High-quality compressed audio
- `aac` - Widely supported compressed audio

#### ElevenLabs Formats

Supported output formats:
- `mp3_44100_64`, `mp3_44100_96`, `mp3_44100_128` (default), `mp3_44100_192` - MP3 at different bitrates
- `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100` - PCM at different sample rates
- `ulaw_8000` - 8kHz μ-law encoding (saved as .wav files)

## Examples

### OpenAI Examples

Basic usage (will prompt for API key, voice, format if defaults are used):
```bash
ttsrs input.txt
```

Specifying voice, format, speed, and API key:
```bash
ttsrs --voice nova --format mp3 --speed 1.2 --apikey sk-... input.txt
```

Using environment variable for API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
ttsrs --voice echo --format wav input.txt
```

Using different OpenAI voices:
```bash
ttsrs --voice ash --format mp3 input.txt
ttsrs --voice coral --format opus input.txt
```

Using a custom endpoint URL (e.g., for a local LM Studio instance):
```bash
ttsrs --endpoint-url "http://localhost:1234/v1/audio/speech" --apikey N/A --voice some-local-voice input.txt
```

### ElevenLabs Examples

Basic usage with ElevenLabs:
```bash
ttsrs --provider elevenlabs --elevenlabs-voice-id "21m00Tcm4TlvDq8ikWAM" input.txt
```

Specifying ElevenLabs model and voice settings:
```bash
ttsrs --provider elevenlabs \
  --elevenlabs-voice-id "21m00Tcm4TlvDq8ikWAM" \
  --elevenlabs-model "eleven_turbo_v2_5" \
  --elevenlabs-stability 0.6 \
  --elevenlabs-similarity 0.8 \
  --format mp3_44100_192 \
  --apikey your-elevenlabs-api-key \
  input.txt
```

Using environment variable for ElevenLabs API key:
```bash
export ELEVENLABS_API_KEY='your-api-key-here'
ttsrs --provider elevenlabs --elevenlabs-voice-id "your-voice-id" input.txt
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key. The `--apikey` flag takes precedence if both are set.
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key. The `--apikey` flag takes precedence if both are set.

## Configuration File

ttsrs supports storing API keys in a configuration file for convenience. When you provide an API key via prompt (not through the command line or environment variable), it will be automatically saved to `.ttsrs_config.json` in the current directory for future use.

The configuration file has the following structure:
```json
{
  "openai_api_key": "your-openai-api-key",
  "elevenlabs_api_key": "your-elevenlabs-api-key"
}
```

Priority order for API keys:
1. Command-line flag (`--apikey`)
2. Environment variable (`OPENAI_API_KEY` or `ELEVENLABS_API_KEY`)
3. Configuration file (`.ttsrs_config.json`)
4. Interactive prompt (will save to config file)

## Technical Details

- Text is automatically chunked based on token count (using `tiktoken_rs` with `cl100k_base`) to stay within API limits (approx. 500 tokens per chunk).
- Each chunk is sent separately to the specified API endpoint.
- Audio responses for each chunk are saved as temporary files.
- `ffmpeg` is used to concatenate the temporary audio files into a single output file.
- Temporary files are automatically cleaned up after successful combination.
- Output is saved in a directory named after the input file.
- **OpenAI**: Supports adjustable speaking speed via `--speed` and multiple voices/formats.
- **ElevenLabs**: Supports voice stability and similarity boost settings, with multiple models and formats.
- Both providers can work with custom endpoints via `--endpoint-url`.

## Development

### Code Signing (macOS)

macOS builds are automatically code-signed in GitHub Actions using an Apple Developer certificate. For information on setting up code signing for your fork or development environment, see [CODESIGNING_SETUP.md](CODESIGNING_SETUP.md).

## License

MIT License

## Acknowledgments

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project.
