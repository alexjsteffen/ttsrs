# _ai-tts.rs_ Documentation

- `input_file`: The path to the input text file.
- `--model` (optional): The TTS model to use. Default is `tts-1-hd`.
- `--voice` (optional): The voice to use for TTS. Default is `fable`.

## Features

- Reads text from an input file.
- Chunks the text into smaller pieces to comply with the API's token limit.
- Generates audio files for each text chunk using OpenAI's TTS API.
- Combines the generated audio chunks into a single output file.
- Displays a spinning animation while generating audio to indicate progress.
- Creates a directory with the same name as the input file to store the output and temporary files.
- Removes temporary files after the final output file is generated.
