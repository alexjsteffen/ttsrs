# ttsrs - Text-to-Speech CLI Tool

A Rust-based command-line tool for converting text to speech using OpenAI's TTS API, ElevenLabs API, or a compatible custom endpoint.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [From Source](#from-source)
  - [Pre-built Binaries](#pre-built-binaries)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [TUI Mode (Interactive)](#tui-mode-interactive)
  - [Command-Line Arguments](#command-line-arguments)
  - [Voice Options](#voice-options)
  - [Audio Formats](#audio-formats)
- [Examples](#examples)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Configuration File](#configuration-file)
- [Technical Details](#technical-details)
- [Development](#development)
  - [Building](#building)
  - [Testing](#testing)
  - [Code Signing (macOS)](#code-signing-macos)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- 🎯 Easy-to-use command-line interface
- 🖥️ Optional TUI (Terminal User Interface) mode for interactive configuration
- 🔊 High-quality text-to-speech conversion using OpenAI's API, ElevenLabs API, or custom endpoints
- 🌐 Support for multiple TTS providers (OpenAI and ElevenLabs)
- 📝 Supports large text files through automatic chunking
- ✏️ Built-in Vim-inspired text editor for creating input files on the fly
- 🎨 Multiple voice options and audio formats
- ⚡ Adjustable speaking speed (OpenAI) and voice settings (ElevenLabs)
- 🔄 Interactive mode for selecting voices and formats (when defaults are used)
- 📁 Organized output with automatic file management
- 🚀 Progress indicators during conversion
- 🔑 Persistent API key storage via configuration file

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version) — only needed when building from source
- [ffmpeg](https://ffmpeg.org/download.html) — required at runtime for combining audio chunks
- An API key for the target TTS service ([OpenAI](https://platform.openai.com/api-keys) or [ElevenLabs](https://elevenlabs.io/))
- Internet connection

## Installation

### From Source

```bash
git clone https://github.com/alexjsteffen/ttsrs.git
cd ttsrs
cargo build --release
```

The compiled binary will be at `target/release/ttsrs`. You can copy it to a directory in your `PATH`:

```bash
cp target/release/ttsrs /usr/local/bin/
```

### Pre-built Binaries

Check the [Releases](https://github.com/alexjsteffen/ttsrs/releases) page for pre-built binaries for your platform.

## Quick Start

1. **Set your API key** (choose one method):
   ```bash
   # Via environment variable
   export OPENAI_API_KEY='sk-...'

   # Or pass it directly
   ttsrs --apikey sk-... input.txt

   # Or let ttsrs prompt you (the key will be saved for future use)
   ttsrs input.txt
   ```

2. **Convert a text file to speech:**
   ```bash
   ttsrs input.txt
   ```

3. **Or use the interactive TUI:**
   ```bash
   ttsrs --tui
   ```

## Usage

### TUI Mode (Interactive)

For an interactive terminal user interface, use the `--tui` flag:

```bash
ttsrs --tui
```

This launches a full-screen TUI where you can:
- Select TTS provider (OpenAI or ElevenLabs)
- Enter input file path
- **Create text files** using the built-in Vim-inspired text editor
- Choose voice, model, and format using arrow keys
- Configure all settings interactively
- Submit to generate audio

Navigation:
- **↑↓ / Tab**: Move between fields
- **← →**: Change selection for dropdown fields
- **Enter**: Edit text fields or submit
- **Esc / q**: Quit

#### Built-in Text Editor

The TUI includes an internal text editor (powered by EdTUI) for creating text files directly within the application:

1. Navigate to "Create Text File" and press Enter
2. A help modal will appear on first use explaining the Vim-like keybindings
3. Use Vim keybindings to edit text:
   - Press `i` to enter Insert mode
   - Type your text
   - Press `Esc` to return to Normal mode
   - Press `F2` to save and exit, or `Esc` (in Normal mode) to cancel
4. After editing, enter a filename (without .txt extension) to save the file
5. The file will be saved in the current directory and automatically set as the input file

### Command-Line Arguments

```bash
ttsrs [OPTIONS] <INPUT_FILE>
```

| Argument       | Description                                     | Default                         |
| -------------- | ----------------------------------------------- | ------------------------------- |
| `<INPUT_FILE>` | Path to the input text file                     | - (Required, or prompted)       |
| `--tui`        | Launch TUI mode for interactive configuration   | -                               |
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

### TUI Mode

Launch the interactive TUI for easy configuration:
```bash
ttsrs --tui
```

The TUI provides a user-friendly interface where you can:
- Navigate with arrow keys or Tab
- Edit fields by pressing Enter
- Select options with left/right arrows
- Submit your configuration to generate audio

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

## Configuration

### Environment Variables

| Variable | Description |
| -------- | ----------- |
| `OPENAI_API_KEY` | Your OpenAI API key. The `--apikey` flag takes precedence if both are set. |
| `ELEVENLABS_API_KEY` | Your ElevenLabs API key. The `--apikey` flag takes precedence if both are set. |

### Configuration File

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
- `ffmpeg` is used to concatenate the temporary audio files into a single output file, re-encoding with the appropriate codec to ensure continuous timestamps.
- Temporary files are automatically cleaned up after successful combination.
- Output is saved in a directory named after the input file.
- **OpenAI**: Supports adjustable speaking speed via `--speed` and multiple voices/formats.
- **ElevenLabs**: Supports voice stability and similarity boost settings, with multiple models and formats.
- Both providers can work with custom endpoints via `--endpoint-url`.

## Development

### Building

```bash
cargo build          # Debug build
cargo build --release # Optimized release build
```

### Testing

```bash
cargo test           # Run all tests
cargo clippy         # Run linter
```

### Code Signing (macOS)

macOS builds are automatically code-signed in GitHub Actions using an Apple Developer certificate. For information on setting up code signing for your fork or development environment, see [docs/CODESIGNING_SETUP.md](docs/CODESIGNING_SETUP.md).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

MIT License

## Acknowledgments

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project.
