# _ai-tts.rs_ Documentation

The ai-tts.rs project provides a command-line tool for generating spoken audio from text files using OpenAI's text-to-speech (TTS) API. It takes an input text file specified by the input_file argument. The TTS model can be optionally specified using the --model flag, with the default being "tts-1-hd". Similarly, the voice used for the TTS can be set with the --voice option, defaulting to "fable" if not provided.
The tool offers several key features to make the TTS process smooth and efficient. It starts by reading the text from the provided input file. To comply with the token limit of the OpenAI API, the tool intelligently chunks the input text into smaller pieces. It then generates audio files for each text chunk by making requests to OpenAI's TTS API.
Once all the audio chunks have been generated, the tool combines them into a single output audio file for convenience. During the audio generation process, a spinning animation is displayed in the console to provide visual feedback and indicate progress to the user.
To keep things organized, the tool automatically creates a new directory with the same name as the input text file. This directory is used to store the final output audio file as well as any temporary audio chunks generated during the process. After the final output file has been successfully generated, the tool cleans up by removing the temporary files, leaving only the complete audio file in the output directory.

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
