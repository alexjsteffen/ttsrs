# ttsrs

A command-line tool for generating spoken audio from text files using OpenAI's text-to-speech (TTS) API.

## Overview

**ttsrs** converts written text into high-quality spoken audio, making it useful for:
- Audiobook creation
- Accessibility tools
- Automated announcements
- Educational content

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project, but with enhanced functionality and a streamlined interface.

## Features

- Interactive wizard for easy configuration
- Multiple voice options with descriptions
- Support for multiple output formats (MP3, FLAC, WAV, PCM)
- Automatic text chunking to handle API limits
- Internal audio processing (no ffmpeg required)
- Progress indication during generation
- Organized output file management

## Usage

### Basic Command

```bash
ttsrs [--model MODEL] [--voice VOICE] [--format FORMAT] [--apikey KEY] [INPUT_FILE]
```

### Arguments

- `--model`: TTS model (default: tts-1-hd)
- `--voice`: Voice selection (default: interactive selection)
- `--format`: Output format (default: interactive selection)
- `--apikey`: OpenAI API key (optional)
- `INPUT_FILE`: Text file path (can be provided via prompt)

### Available Voices

- **Echo**: Clear and bright, ideal for announcements
- **Fable**: Great for storytelling
- **Onyx**: Deep and resonant
- **Nova**: Youthful and energetic
- **Shimmer**: Soft and soothing
- **Alloy**: Versatile and natural-sounding

### Output Formats

- MP3: Compressed audio with good quality
- FLAC: Lossless compression
- WAV: Uncompressed audio
- PCM: Raw audio data

## Setup

Set your OpenAI API key using one of these methods:
1. Environment variable: `OPENAI_API_KEY`
2. Command-line argument: `--apikey`
3. Interactive prompt during execution

## Example

```bash
# With all options specified
ttsrs --apikey YOUR_API_KEY --voice alloy --model tts-1-hd --format mp3 input.txt

# Interactive mode
ttsrs
```

The tool will:
1. Create an output directory named after your input file
2. Split text into API-friendly chunks
3. Generate audio for each chunk
4. Combine chunks internally using native audio processing
5. Save the final file in your chosen format
6. Clean up temporary files automatically
