# _ttsrs_ Documentation

**Overview**
--------

The **ttsrs** project provides a command-line tool for generating spoken audio from text files using OpenAI's text-to-speech (TTS) API. It is designed to facilitate the conversion of written text into high-quality spoken audio, making it accessible for various applications such as audiobooks, accessibility tools, and automated announcements. This project is based on a similar Python script but offers enhanced functionality and ease of use through a streamlined command-line interface.

For more details, visit the [original Python project on GitHub](https://github.com/tom-huntington/unofficial-openai-tts-cli).

**Usage**
-----

### Command-Line Arguments

- `--model` (optional): The TTS model to use. Default is `tts-1-hd`.
- `--voice` (optional): The voice to use for TTS. Default is `fable`.
- `--apikey` (optional): Use an OpenAI api key inline.
- `input_file`: The path to the input text file. **It must inserted after the flags**

### Features

* Reads text from an input file.
* Chunks the text into smaller pieces to comply with the API's token limit.
* Generates audio files for each text chunk using OpenAI's TTS API.
* Combines the generated audio chunks into a single output file.
* Displays a spinning animation while generating audio to indicate progress.
* Creates a directory with the same name as the input file to store the output and temporary files.
* Removes temporary files after the final output file is generated.

### Setup

Before using ttsrs, make sure to set the `OPENAI_API_KEY` environment variable with your OpenAI API key.

### Example

The completed markdown string for your shell command (API Key is not functional) would look like this:

```bash
ttsrs --apikey sk-m8xy4xZg7E5VgfRzTg7cY4HlckFJ92eKlH1zpqv5PQKTYUBl --voice alloy --model tts-1
```

Make sure to replace the API key with your actual key before executing the command. Enjoy using ttsrs!
