# ttsrs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub Release](https://img.shields.io/github/v/release/alexjsteffen/ttsrs)](https://github.com/alexjsteffen/ttsrs/releases)

A fast, feature-rich command-line tool for converting text to speech using [OpenAI](https://platform.openai.com/docs/guides/text-to-speech) or [ElevenLabs](https://elevenlabs.io/) APIs. Includes an interactive TUI and a built-in Vim-inspired text editor.

## Features

- Easy-to-use command-line interface with sensible defaults
- Interactive TUI (Terminal User Interface) for guided configuration
- Built-in Vim-inspired text editor for creating input files on the fly
- High-quality TTS via OpenAI, ElevenLabs, or custom endpoints
- Automatic chunking for large text files
- Multiple voice options and audio formats per provider
- Adjustable speaking speed (OpenAI) and voice settings (ElevenLabs)
- Organized output with automatic file management
- Progress indicators during conversion
- Persistent API key storage via configuration file

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (1.70+) — only needed when building from source
- [ffmpeg](https://ffmpeg.org/download.html) — required at runtime for combining audio chunks
- An API key for [OpenAI](https://platform.openai.com/api-keys) or [ElevenLabs](https://elevenlabs.io/)

## Installation

### From Source

```bash
git clone https://github.com/alexjsteffen/ttsrs.git
cd ttsrs
cargo build --release
```

The compiled binary will be at `target/release/ttsrs`. Optionally copy it into your `PATH`:

```bash
cp target/release/ttsrs /usr/local/bin/
```

### Pre-built Binaries

Download the latest binary for your platform from the [Releases](https://github.com/alexjsteffen/ttsrs/releases) page.

## Quick Start

```bash
# 1. Set your API key (choose one method)
export OPENAI_API_KEY='sk-...'        # environment variable
ttsrs --apikey sk-... input.txt       # inline flag
ttsrs input.txt                       # interactive prompt (saves key for reuse)

# 2. Convert a text file to speech
ttsrs input.txt

# 3. Or use the interactive TUI
ttsrs --tui
```

## Usage

### TUI Mode

Launch the interactive terminal UI with `--tui`:

```bash
ttsrs --tui
```

The TUI lets you:

- Select a TTS provider (OpenAI / ElevenLabs)
- Enter or browse for an input file path
- Create text files using the built-in editor
- Choose voice, model, and format with arrow keys
- Configure all settings before submitting

**Navigation:**

| Key | Action |
|-----|--------|
| `↑` / `↓` / `Tab` | Move between fields |
| `←` / `→` | Cycle through options |
| `Enter` | Edit text fields or submit |
| `Esc` / `q` | Quit |

#### Built-in Text Editor

The TUI includes a Vim-inspired editor (powered by [EdTUI](https://github.com/preiter93/edtui)):

1. Navigate to **Create Text File** and press `Enter`
2. Use Vim keybindings (`i` for insert, `Esc` for normal mode, `h/j/k/l` to move)
3. Press `F2` to save and exit, or `Esc` in normal mode to cancel
4. Enter a filename to save — the file is automatically set as the input

### Command-Line Arguments

```
ttsrs [OPTIONS] [INPUT_FILE]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `INPUT_FILE` | Path to the input text file | _(prompted)_ |
| `--tui` | Launch interactive TUI | — |
| `--provider` | TTS provider (`openai` or `elevenlabs`) | `openai` |
| `-m`, `--model` | TTS model | `tts-1-hd` |
| `-v`, `--voice` | Voice name (OpenAI only) | `alloy` |
| `-f`, `--format` | Output audio format | `flac` (OpenAI) |
| `--speed` | Speaking speed, 0.25–4.0 (OpenAI only) | `1.0` |
| `-a`, `--apikey` | API key | _(env var or prompted)_ |
| `--endpoint-url` | Custom API endpoint URL | _(provider default)_ |
| `--elevenlabs-voice-id` | ElevenLabs voice ID | — |
| `--elevenlabs-model` | ElevenLabs model ID | `eleven_turbo_v2_5` |
| `--elevenlabs-stability` | Voice stability, 0.0–1.0 | `0.5` |
| `--elevenlabs-similarity` | Voice similarity boost, 0.0–1.0 | `0.75` |

### Voice Options

#### OpenAI Voices

| Voice | Description |
|-------|-------------|
| **alloy** | Versatile, well-balanced |
| **ash** | Clear and conversational |
| **coral** | Warm and friendly |
| **echo** | Clear and professional |
| **fable** | Warm and engaging, great for storytelling |
| **nova** | Young and energetic |
| **onyx** | Deep and authoritative |
| **sage** | Calm and measured |
| **shimmer** | Soft and soothing |

#### ElevenLabs Voices

ElevenLabs uses unique voice IDs. Find available voices at the [ElevenLabs Voice Library](https://elevenlabs.io/voice-library) or via the API:

```bash
curl https://api.elevenlabs.io/v1/voices -H "xi-api-key: YOUR_KEY"
```

Pass the voice ID with `--elevenlabs-voice-id`.

### Audio Formats

#### OpenAI

`flac` (default) · `mp3` · `wav` · `pcm` · `opus` · `aac`

#### ElevenLabs

`mp3_44100_64` · `mp3_44100_96` · `mp3_44100_128` (default) · `mp3_44100_192` · `pcm_16000` · `pcm_22050` · `pcm_24000` · `pcm_44100` · `ulaw_8000`

## Examples

### OpenAI

```bash
# Basic — prompts for voice and format
ttsrs input.txt

# Explicit settings
ttsrs --voice nova --format mp3 --speed 1.2 --apikey sk-... input.txt

# Custom endpoint (e.g., LM Studio)
ttsrs --endpoint-url "http://localhost:1234/v1/audio/speech" --apikey N/A input.txt
```

### ElevenLabs

```bash
# Basic
ttsrs --provider elevenlabs --elevenlabs-voice-id "21m00Tcm4TlvDq8ikWAM" input.txt

# Full configuration
ttsrs --provider elevenlabs \
  --elevenlabs-voice-id "21m00Tcm4TlvDq8ikWAM" \
  --elevenlabs-model "eleven_turbo_v2_5" \
  --elevenlabs-stability 0.6 \
  --elevenlabs-similarity 0.8 \
  --format mp3_44100_192 \
  input.txt
```

## Configuration

### API Key Resolution Order

1. `--apikey` flag
2. Environment variable (`OPENAI_API_KEY` or `ELEVENLABS_API_KEY`)
3. Configuration file (`.ttsrs_config.json` in the current directory)
4. Interactive prompt (saves the key to the config file for future use)

### Configuration File

When you provide an API key via the interactive prompt, it is saved to `.ttsrs_config.json` in the current directory:

```json
{
  "openai_api_key": "sk-...",
  "elevenlabs_api_key": "..."
}
```

## Technical Details

- Text is automatically chunked based on token count (~500 tokens per chunk via `tiktoken_rs` with `cl100k_base`) to stay within API limits.
- Each chunk is sent as a separate API request; responses are streamed to temporary files.
- `ffmpeg` concatenates the temporary audio files into a single output, re-encoding with the appropriate codec for continuous timestamps.
- Temporary files are cleaned up automatically after a successful combination.
- Output is saved in a directory named after the input file.

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

macOS builds are automatically code-signed in GitHub Actions. See [docs/CODESIGNING_SETUP.md](docs/CODESIGNING_SETUP.md) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Based on the [unofficial-openai-tts-cli](https://github.com/tom-huntington/unofficial-openai-tts-cli) Python project.
