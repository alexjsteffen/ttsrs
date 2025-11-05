// Import necessary crates and modules
use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use dialoguer::{Input, Select};
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tiktoken_rs::cl100k_base;

// Define the TTS provider
#[derive(Debug, Clone, PartialEq)]
enum TtsProvider {
    OpenAI,
    ElevenLabs,
}

// Configuration structure for storing API keys
#[derive(Debug, Serialize, Deserialize, Default)]
struct Config {
    openai_api_key: Option<String>,
    elevenlabs_api_key: Option<String>,
}

impl Config {
    /// Get the path to the config file in the current directory
    fn config_path() -> PathBuf {
        PathBuf::from(".ttsrs_config.json")
    }

    /// Load the config from the file, or return a default config if it doesn't exist
    fn load() -> Result<Self> {
        let config_path = Self::config_path();
        if config_path.exists() {
            let contents = fs::read_to_string(&config_path)?;
            let config: Config = serde_json::from_str(&contents)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Save the config to the file
    fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(&config_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Get the API key for the specified provider
    fn get_api_key(&self, provider: &TtsProvider) -> Option<String> {
        match provider {
            TtsProvider::OpenAI => self.openai_api_key.clone(),
            TtsProvider::ElevenLabs => self.elevenlabs_api_key.clone(),
        }
    }

    /// Set the API key for the specified provider
    fn set_api_key(&mut self, provider: &TtsProvider, api_key: String) {
        match provider {
            TtsProvider::OpenAI => self.openai_api_key = Some(api_key),
            TtsProvider::ElevenLabs => self.elevenlabs_api_key = Some(api_key),
        }
    }
}

// Define command-line arguments using the clap crate
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input text file name
    #[arg()]
    input_file: Option<String>,

    /// TTS model to use (default: tts-1-hd)
    #[arg(short, long, default_value = "tts-1-hd")]
    model: String,

    /// Voice to use for TTS (default: alloy)
    #[arg(short, long, default_value = "alloy")]
    voice: String,

    /// Output audio format (options: mp3, flac, wav, pcm, opus, aac)
    #[arg(short, long, default_value = "flac")]
    format: String,

    /// Speaking speed (0.25 - 4.0, default 1.0)
    #[arg(long, default_value = "1.0")]
    speed: f32,

    /// OpenAI API key (optional, can also be set via the OPENAI_API_KEY environment variable)
    #[arg(short, long)]
    apikey: Option<String>,

    /// Custom API endpoint URL (optional, defaults to OpenAI)
    #[arg(long)]
    endpoint_url: Option<String>,

    /// TTS provider (openai or elevenlabs, default: openai)
    #[arg(long, default_value = "openai")]
    provider: String,

    /// ElevenLabs voice ID (required when using ElevenLabs provider)
    #[arg(long)]
    elevenlabs_voice_id: Option<String>,

    /// ElevenLabs model ID (default: eleven_turbo_v2_5)
    #[arg(long, default_value = "eleven_turbo_v2_5")]
    elevenlabs_model: String,

    /// ElevenLabs voice stability (0.0 - 1.0, default: 0.5)
    #[arg(long, default_value = "0.5")]
    elevenlabs_stability: f32,

    /// ElevenLabs voice similarity boost (0.0 - 1.0, default: 0.75)
    #[arg(long, default_value = "0.75")]
    elevenlabs_similarity: f32,
}

/// The main function of the program.
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let mut args = Args::parse();

    // Determine the TTS provider
    let provider = match args.provider.to_lowercase().as_str() {
        "openai" => TtsProvider::OpenAI,
        "elevenlabs" => TtsProvider::ElevenLabs,
        _ => {
            anyhow::bail!("Invalid provider '{}'. Must be 'openai' or 'elevenlabs'", args.provider);
        }
    };

    // Load config file
    let mut config = Config::load().unwrap_or_default();

    // Get the API key from command-line, environment variable, config file, or prompt
    let api_key = args.apikey.clone()
        .or_else(|| {
            // Try environment variable
            match provider {
                TtsProvider::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
                TtsProvider::ElevenLabs => std::env::var("ELEVENLABS_API_KEY").ok(),
            }
        })
        .or_else(|| {
            // Try config file
            config.get_api_key(&provider)
        })
        .or_else(|| {
            // Prompt the user for the API key if not provided
            let prompt = match provider {
                TtsProvider::OpenAI => "Enter your OpenAI API Key",
                TtsProvider::ElevenLabs => "Enter your ElevenLabs API Key",
            };
            let input: String = Input::new()
                .with_prompt(prompt)
                .interact_text()
                .ok()?;
            
            // Save the API key to the config file for future use
            config.set_api_key(&provider, input.clone());
            if let Err(e) = config.save() {
                eprintln!("Warning: Failed to save API key to config file (.ttsrs_config.json): {}", e);
            } else {
                println!("API key saved to config file (.ttsrs_config.json) for future use.");
            }
            
            Some(input)
        })
        .context(
            "API key not provided. Set it via the --apikey flag, the appropriate environment variable, or input it when prompted."
        )?;

    // Prompt for input file if not provided
    let input_file = args
        .input_file
        .clone()
        .or_else(|| {
            let input: String = Input::new()
                .with_prompt("Enter the path to the input text file")
                .interact_text()
                .ok()?;
            Some(input)
        })
        .context("Input file not provided.")?;

    args.input_file = Some(input_file);

    // Prompt for voice selection (for OpenAI only, ElevenLabs uses voice_id)
    if provider == TtsProvider::OpenAI {
        let voices = vec![
            "Echo - Clear and professional, ideal for announcements.",
            "Fable - Warm and engaging, perfect for storytelling.",
            "Onyx - Deep and authoritative.",
            "Nova - Young and energetic.",
            "Shimmer - Soft and soothing.",
            "Alloy - Versatile and well-balanced.",
            "Ash - Clear and conversational.",
            "Coral - Warm and friendly.",
            "Sage - Calm and measured.",
        ];
        if args.voice.to_lowercase() == "alloy" {
            // Only prompt if default is used (case-insensitive comparison)
            let selection = Select::new()
                .with_prompt("Select a voice")
                .items(&voices)
                .default(5) // Set default index for Alloy
                .interact()?;
            // Extract only the voice name before the hyphen
            args.voice = voices[selection]
                .split(" - ")
                .next()
                .unwrap_or("alloy")
                .to_lowercase()
                .to_string();
        }
    } else {
        // For ElevenLabs, ensure voice_id is provided
        if args.elevenlabs_voice_id.is_none() {
            let input: String = Input::new()
                .with_prompt("Enter the ElevenLabs voice ID (or run with --elevenlabs-voice-id)")
                .interact_text()?;
            args.elevenlabs_voice_id = Some(input);
        }
    }

    // Prompt for output format
    let formats = match provider {
        TtsProvider::OpenAI => vec!["mp3", "flac", "wav", "pcm", "opus", "aac"],
        TtsProvider::ElevenLabs => vec!["mp3_44100_128", "mp3_44100_192", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"],
    };
    
    // For ElevenLabs, set default format if using the OpenAI default
    if provider == TtsProvider::ElevenLabs && args.format.to_lowercase() == "flac" {
        args.format = "mp3_44100_128".to_string();
    }
    
    if (provider == TtsProvider::OpenAI && args.format.to_lowercase() == "flac") ||
       (provider == TtsProvider::ElevenLabs && args.format == "mp3_44100_128") {
        // Only prompt if default is used
        let default_idx = if provider == TtsProvider::OpenAI { 1 } else { 0 };
        let selection = Select::new()
            .with_prompt("Select an output format")
            .items(&formats)
            .default(default_idx)
            .interact()?;
        args.format = formats[selection].to_string();
    }

    // Initialize HTTP client
    let client = Client::new();

    // Get the input file name and create an output directory
    let input_file_path = Path::new(args.input_file.as_ref().unwrap());
    let input_file_name = input_file_path
        .file_stem()
        .context("Invalid input file")?
        .to_str()
        .context("Invalid input file name")?;
    println!(
        "Now creating a folder called {} for you.",
        green_text(input_file_name)
    );
    let output_dir = Path::new("./").join(input_file_name);
    fs::create_dir_all(&output_dir)?;

    // Read the input file and chunk the text
    let lines = read_text_file(input_file_path)?; //
    let chunks = chunk_text(&lines); //

    // Determine the API endpoint URL
    let api_endpoint = if let Some(custom_url) = args.endpoint_url.as_deref() {
        custom_url.to_string()
    } else {
        match provider {
            TtsProvider::OpenAI => "https://api.openai.com/v1/audio/speech".to_string(),
            TtsProvider::ElevenLabs => {
                let voice_id = args.elevenlabs_voice_id.as_ref()
                    .context("ElevenLabs voice ID is required when using ElevenLabs provider. Use --elevenlabs-voice-id")?;
                format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice_id)
            }
        }
    };

    // Generate audio files for each chunk
    let (timestamp, voice_used) = generate_audio_files(
        &chunks,
        &output_dir,
        &args.model,
        &args.voice,
        &args.format,
        &client,
        &api_key,
        args.speed,
        &api_endpoint, // Pass the determined endpoint URL
        &provider,
        &args.elevenlabs_model,
        args.elevenlabs_stability,
        args.elevenlabs_similarity,
    )
    .await?; //

    // Notify the user about the generated files
    println!(
        "Chunk {} files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        args.format, // Use the selected format in the message
        green_text(input_file_name)
    );

    // Determine the output file extension
    let output_ext = if args.format.starts_with("mp3") {
        "mp3"
    } else if args.format.starts_with("pcm") {
        "pcm"
    } else if args.format.starts_with("ulaw") {
        "wav"
    } else {
        &args.format
    };

    // Combine the audio files into a single output file
    combine_audio_files(&output_dir, output_ext, &timestamp, &voice_used)?; //

    // Remove temporary files
    remove_tmp(&output_dir, output_ext, &timestamp, &voice_used)?; // Pass timestamp and voice to remove correct tmp files

    // Final message
    println!(
        "\nThe File [ {}/output.{} ] is ready for you. \n",
        green_text(input_file_name),
        output_ext
    );

    Ok(())
}

// Formats text in green color for console output
fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

// Reads a text file and returns its contents as a vector of strings
fn read_text_file(file_path: &Path) -> Result<Vec<String>> {
    //
    let content = fs::read_to_string(file_path)?;
    Ok(content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(String::from)
        .collect())
}

// Chunks the input text into smaller pieces, each containing up to 500 tokens
fn chunk_text(lines: &[String]) -> Vec<Vec<String>> {
    //
    // Initialize the tokenizer
    let bpe = cl100k_base().unwrap();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_token_count = 0;
    const MAX_TOKENS_PER_CHUNK: usize = 500; // Use a constant for clarity

    // Iterate over each line of text
    for line in lines {
        // Calculate the number of tokens in the current line
        let line_token_count = bpe.encode_ordinary(line).len();

        // If the line itself is too long, split it (basic split, could be improved)
        if line_token_count > MAX_TOKENS_PER_CHUNK {
            // Handle very long lines if necessary (e.g., split them further)
            // For now, we'll push the current chunk and add the long line as its own chunk
            // (or potentially skip/error if that's preferred)
            if !current_chunk.is_empty() {
                chunks.push(std::mem::take(&mut current_chunk));
            }
            chunks.push(vec![line.clone()]); // Add the long line as a separate chunk
            current_token_count = 0; // Reset token count for the next chunk
            continue; // Move to the next line
        }

        // If adding this line exceeds the token limit, start a new chunk
        if current_token_count + line_token_count > MAX_TOKENS_PER_CHUNK {
            if !current_chunk.is_empty() {
                // Ensure we don't push empty chunks
                chunks.push(std::mem::take(&mut current_chunk));
            }
            current_token_count = 0;
        }

        // Add the line to the current chunk and update the token count
        current_chunk.push(line.clone());
        current_token_count += line_token_count;
    }

    // Add any remaining lines as the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

// Generates audio files for each chunk of text using the specified API endpoint
// Returns (timestamp, voice_lowercase) for identifying the generated files
// Note: clippy::too_many_arguments is allowed here because these parameters represent
// distinct configuration options that are most clearly expressed as separate arguments
#[allow(clippy::too_many_arguments)]
async fn generate_audio_files(
    chunks: &[Vec<String>],
    output_dir: &Path,
    model: &str,
    voice: &str,
    format: &str,
    client: &Client,
    api_key: &str,
    speed: f32,
    api_endpoint: &str, // Accept the API endpoint URL
    provider: &TtsProvider,
    elevenlabs_model: &str,
    elevenlabs_stability: f32,
    elevenlabs_similarity: f32,
) -> Result<(String, String)> {
    //
    // Generate a timestamp for file naming with seconds for better uniqueness
    let date_time_string = Local::now().format("%Y%m%d%H%M%S").to_string();

    // Convert voice name to lowercase for API call and filename
    let voice_lowercase = voice.to_lowercase();

    // Iterate over each chunk
    for (i, chunk) in chunks.iter().enumerate() {
        // Join the lines in the chunk into a single string
        let chunk_string = chunk.join(" ");
        println!("〰️〰️〰️〰️〰️〰️");
        println!(
            "{} {:06} of {}",
            green_text("Processing chunk"), // Changed message slightly
            i + 1,
            chunks.len()
        );
        println!(
            "Input String: {}...",
            &chunk_string[..chunk_string.len().min(60)]
        );

        // Check if the chunk exceeds the character limit (provider-specific limit)
        let max_chars = match provider {
            TtsProvider::OpenAI => 4096,      // OpenAI's documented limit
            TtsProvider::ElevenLabs => 5000,  // ElevenLabs has a 5000 character limit
        };
        if chunk_string.len() > max_chars {
            eprintln!( // Use eprintln for errors
                "Warning: Chunk {:06} exceeds {} characters ({}). Attempting to process, but it might fail.",
                i + 1,
                max_chars,
                chunk_string.len()
            );
            // Optionally, you could truncate here:
            // chunk_string = chunk_string[..max_chars].to_string();
            // Or skip the chunk: continue;
            // Or return an error: anyhow::bail!(...)
        }

        // Show a progress bar while generating audio
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.green} {msg}")?,
        );
        pb.set_message(format!("Sending chunk {} to API...", i + 1)); // More specific message

        // Make the API request to the specified endpoint
        let (request_body, auth_header) = match provider {
            TtsProvider::OpenAI => {
                let body = serde_json::json!({
                    "model": model,
                    "voice": voice_lowercase,
                    "input": chunk_string,
                    "speed": speed,
                    "response_format": format,
                });
                (body, ("Authorization".to_string(), format!("Bearer {}", api_key)))
            }
            TtsProvider::ElevenLabs => {
                let body = serde_json::json!({
                    "text": chunk_string,
                    "model_id": elevenlabs_model,
                    "voice_settings": {
                        "stability": elevenlabs_stability,
                        "similarity_boost": elevenlabs_similarity,
                    }
                });
                (body, ("xi-api-key".to_string(), api_key.to_string()))
            }
        };

        let response = client
            .post(api_endpoint) // Use the passed endpoint URL
            .header(&auth_header.0, &auth_header.1)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        pb.set_message(format!("Waiting for response for chunk {}...", i + 1));

        let response = match response {
            Ok(resp) => resp,
            Err(e) => {
                pb.finish_with_message(format!(
                    "❌ Error sending request for chunk {}: {}",
                    i + 1,
                    e
                ));
                // Decide how to handle: continue to next chunk, or return error?
                // For now, let's continue to allow processing other chunks
                continue;
                // Or return the error:
                // return Err(e.into());
            }
        };

        // Handle API errors
        if !response.status().is_success() {
            let status = response.status();
            let error_text = match response.text().await {
                // Try to get text body for more info
                Ok(text) => text,
                Err(_) => "Could not read error response body".to_string(),
            };

            pb.finish_with_message(format!(
                "❌ API Error for chunk {}: Status Code: {}. Response: {}",
                i + 1,
                status,
                error_text
            ));
            // Decide how to handle: continue or error out?
            // Let's continue processing other chunks for now.
            continue;
            // Or return an error:
            // anyhow::bail!("API error for chunk {}: Status Code: {}. Response: {}", i+1, status, error_text);
        }

        // Save the audio response to a file with voice and timestamp for uniqueness
        // Extract the file extension based on format
        let file_ext = if format.starts_with("mp3") {
            "mp3"
        } else if format.starts_with("pcm") {
            "pcm"
        } else if format.starts_with("ulaw") {
            "wav" // ulaw is typically in wav container
        } else {
            format
        };
        let file_name = format!("tmp_{}_{}_chunk{:06}.{}", date_time_string, voice_lowercase, i + 1, file_ext);
        let file_path = output_dir.join(&file_name);

        // Stream the response and write it to the file
        // Use try_fold for better error handling during streaming
        let file_result = async {
            let mut file = File::create(&file_path)?;
            let mut stream = response.bytes_stream();
            while let Some(item) = stream.next().await {
                let chunk_bytes = item.context("Failed to read chunk from response stream")?;
                file.write_all(&chunk_bytes)
                    .context("Failed to write chunk to file")?;
            }
            Ok::<(), anyhow::Error>(()) // Explicitly type Ok value
        }
        .await;

        match file_result {
            Ok(_) => pb.finish_with_message(format!(
                "✅ Chunk {} audio saved as {}",
                i + 1,
                file_path.display()
            )),
            Err(e) => {
                pb.finish_with_message(format!("❌ Error saving audio for chunk {}: {}", i + 1, e));
                // Clean up partially written file?
                let _ = fs::remove_file(&file_path); // Attempt removal, ignore error if it fails
                                                     // Continue to next chunk or return error? Let's continue.
                continue;
                // Or return the error:
                // return Err(e);
            }
        }
    }

    Ok((date_time_string, voice_lowercase))
}

/// Combines all the generated temporary audio files into a single file using ffmpeg.
fn combine_audio_files(output_dir: &Path, format: &str, timestamp: &str, voice: &str) -> Result<()> {
    //
    // Collect all the temporary files for this specific run (matching timestamp, voice, and format)
    let mut input_files = Vec::new();
    let prefix = format!("tmp_{}_{}_", timestamp, voice);
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && // Ensure it's a file
            path
                .extension()
                .map(|ext| ext.to_str() == Some(format)) // Compare extension safely
                .unwrap_or(false) &&
            path.file_name()
                .and_then(|name| name.to_str()) // Safely get filename as str
                .map(|name_str| name_str.starts_with(&prefix)) // Check for specific prefix with timestamp and voice
                .unwrap_or(false)
        {
            input_files.push(path);
        }
    }

    // Check if there are any files to combine
    if input_files.is_empty() {
        println!(
            "No temporary audio files found to combine for format '{}'. Skipping combination.",
            format
        );
        return Ok(()); // Not an error, just nothing to do
    }

    println!(
        "Combining {} temporary {} files...",
        input_files.len(),
        format
    );

    // Sort the files to ensure they are combined in the correct order
    input_files.sort();

    // Construct the ffmpeg command arguments using a temporary file list for safety with many files
    let list_file_path = output_dir.join("ffmpeg_list.txt");
    {
        // Scope for file handle to ensure it's closed before ffmpeg runs
        let mut list_file = File::create(&list_file_path)?;
        for input_file in &input_files {
            // Use only the filename (not the full path) since ffmpeg_list.txt is in the same directory
            let filename = input_file.file_name()
                .and_then(|name| name.to_str())
                .context("Failed to get filename")?;
            writeln!(
                list_file,
                "file '{}'",
                filename
            )?;
        }
    } // list_file goes out of scope and is closed

    let output_file_path = output_dir.join(format!("output.{}", format));

    // Simpler ffmpeg command using the file list
    let ffmpeg_args = vec![
        "-f",
        "concat",
        "-safe",
        "0", // Needed if paths are relative or contain certain characters
        "-i",
        list_file_path.to_str().unwrap(),
        "-c",
        "copy", // Try to copy codecs directly if possible (faster, lossless)
        "-y",   // Overwrite output files without asking
        output_file_path.to_str().unwrap(),
    ];

    // Execute the ffmpeg command
    println!("Running ffmpeg command..."); // Log before running
    let ffmpeg_output = Command::new("ffmpeg")
        .args(&ffmpeg_args)
        .output() // Use output() to capture stderr
        .context("Failed to execute ffmpeg command. Is ffmpeg installed and in your PATH?")?;

    // Clean up the temporary list file regardless of ffmpeg success
    let _ = fs::remove_file(&list_file_path);

    if !ffmpeg_output.status.success() {
        eprintln!(
            "ffmpeg stderr: {}",
            String::from_utf8_lossy(&ffmpeg_output.stderr)
        );
        anyhow::bail!(
            "ffmpeg command failed with status: {}. Check ffmpeg output above for details.",
            ffmpeg_output.status
        );
    } else {
        println!("ffmpeg command successful.");
    }

    Ok(())
}

/// Removes temporary files from the output directory matching the specified format, timestamp, and voice.
fn remove_tmp(output_dir: &Path, format: &str, timestamp: &str, voice: &str) -> Result<()> {
    //
    let mut removed_count = 0;
    let prefix = format!("tmp_{}_{}_", timestamp, voice);
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name_str| name_str.starts_with(&prefix))
                .unwrap_or(false)
            && path.extension().and_then(|ext| ext.to_str()) == Some(format)
        {
            match fs::remove_file(&path) {
                Ok(_) => removed_count += 1,
                Err(e) => eprintln!(
                    "Warning: Failed to remove temporary file {}: {}",
                    path.display(),
                    e
                ),
            }
        }
    }
    if removed_count > 0 {
        println!("Removed {} temporary {} files.", removed_count, format);
    }
    Ok(())
}
