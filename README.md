# ttsrs - Text-to-Speech CLI Tool

A Rust-based command-line tool for converting text to speech using OpenAI's TTS API.

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
- 🔊 High-quality text-to-speech conversion using OpenAI's API
- 📝 Supports large text files through automatic chunking
- 🎨 Multiple voice options and audio formats
- ⚡ Adjustable speaking speed
- 🔄 Interactive mode for selecting voices and formats
- 📁 Organized output with automatic file management
- 🚀 Progress indicators during conversion

## Installation

```bash
cargo install ttsrs
```

## Prerequisites

- Rust (latest stable version)
- ffmpeg (for audio file combining)
- OpenAI API key
- Internet connection

## Usage

### Command-Line Arguments

```bash
ttsrs [OPTIONS] <INPUT_FILE>
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model`, `-m` | TTS model to use | `tts-1-hd` |
| `--voice`, `-v` | Voice selection | `alloy` |
| `--format`, `-f` | Output audio format | `flac` |
| `--speed` | Speaking speed (0.25 - 4.0) | `1.0` |
| `--apikey`, `-a` | OpenAI API key | - |

### Voice Options

Available voices with their characteristics:

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

Supported output formats:
- `flac` (default) - Lossless audio compression
- `mp3` - Common compressed audio format
- `wav` - Uncompressed audio
- `pcm` - Raw audio data
- `opus` - New! High-quality compressed audio
- `aac` - New! Widely supported compressed audio

## Examples

Basic usage:
```bash
ttsrs input.txt
```

Specifying voice, format, and speed:
```bash
ttsrs --voice nova --format mp3 --speed 1.2 input.txt
```

Using API key inline:
```bash
ttsrs --apikey sk-... --voice echo --format wav --speed 0.9 input.txt
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key  
  ```bash
  export OPENAI_API_KEY='your-api-key-here'
  ```

## Technical Details

- Text is automatically chunked into segments of 500 tokens or less
- Each chunk is processed separately and then combined
- Temporary files are automatically cleaned up
- Output is saved in a directory named after the input file
- Supports adjustable speaking speed via `--speed`
- Supports new OpenAI voices and audio formats

## License

MIT License

## Acknowledgments

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project.
