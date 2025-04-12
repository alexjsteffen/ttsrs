# ttsrs - Text-to-Speech CLI Tool

A Rust-based command-line tool for converting text to speech using OpenAI's TTS API or a compatible custom endpoint.

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
- 🔊 High-quality text-to-speech conversion using OpenAI's API or custom endpoints
- 📝 Supports large text files through automatic chunking
- 🎨 Multiple voice options and audio formats
- ⚡ Adjustable speaking speed
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
| `--model`, `-m`  | TTS model to use                                | `tts-1-hd`                      |
| `--voice`, `-v`  | Voice selection                                 | `alloy` (prompted if default)   |
| `--format`, `-f` | Output audio format                             | `flac` (prompted if default)    |
| `--speed`      | Speaking speed (0.25 - 4.0)                     | `1.0`                           |
| `--apikey`, `-a` | API key for the TTS service                   | - (Required, env var, or prompted) |
| `--endpoint-url`| Custom API endpoint URL (e.g., for local AI)  | `https://api.openai.com/v1/audio/speech` |

### Voice Options

Available voices (may vary depending on the endpoint):

- **alloy** - A versatile, well-balanced voice
- **echo** - Clear and professional, ideal for announcements
- **fable** - Warm and engaging, perfect for storytelling
- **onyx** - Deep and authoritative
- **nova** - Young and energetic
- **shimmer** - Soft and soothing
- **ballad** - New!
- **coral** - New!
- **sage** - New!

### Audio Formats

Supported output formats (may vary depending on the endpoint):
- `flac` (default) - Lossless audio compression
- `mp3` - Common compressed audio format
- `wav` - Uncompressed audio
- `pcm` - Raw audio data
- `opus` - High-quality compressed audio
- `aac` - Widely supported compressed audio

## Examples

Basic usage (will prompt for API key, voice, format if defaults are used):
```bash
ttsrs input.txt
```

Specifying voice, format, speed, and API key:
```bash
ttsrs --voice nova --format mp3 --speed 1.2 --apikey sk-... input.txt
```

Using a custom endpoint URL (e.g., for a local LM Studio instance):
```bash
ttsrs --endpoint-url "http://localhost:1234/v1/audio/speech" --apikey N/A --voice some-local-voice input.txt
```

Using environment variable for API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
ttsrs --voice echo --format wav input.txt
```

## Environment Variables

- `OPENAI_API_KEY`: Your API key. The `--apikey` flag takes precedence if both are set.

## Technical Details

- Text is automatically chunked based on token count (using `tiktoken_rs` with `cl100k_base`) to stay within API limits (approx. 500 tokens per chunk).
- Each chunk is sent separately to the specified API endpoint.
- Audio responses for each chunk are saved as temporary files.
- `ffmpeg` is used to concatenate the temporary audio files into a single output file.
- Temporary files are automatically cleaned up after successful combination.
- Output is saved in a directory named after the input file.
- Supports adjustable speaking speed via `--speed`.
- Supports multiple OpenAI voices and audio formats (or those supported by the custom endpoint).

## License

MIT License

## Acknowledgments

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project.
