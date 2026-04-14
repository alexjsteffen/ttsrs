use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use dialoguer::{Input, Select};
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tiktoken_rs::cl100k_base;
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::download::auto_download;

mod editor;
mod tui;

#[derive(Debug, Clone, PartialEq)]
enum TtsProvider {
    OpenAI,
    ElevenLabs,
    Custom,
}

impl fmt::Display for TtsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TtsProvider::OpenAI => write!(f, "openai"),
            TtsProvider::ElevenLabs => write!(f, "elevenlabs"),
            TtsProvider::Custom => write!(f, "custom"),
        }
    }
}

impl FromStr for TtsProvider {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(TtsProvider::OpenAI),
            "elevenlabs" => Ok(TtsProvider::ElevenLabs),
            "custom" => Ok(TtsProvider::Custom),
            other => anyhow::bail!(
                "Invalid provider '{}'. Must be 'openai', 'elevenlabs', or 'custom'",
                other
            ),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Config {
    openai_api_key: Option<String>,
    elevenlabs_api_key: Option<String>,
}

impl Config {
    fn config_path() -> PathBuf {
        PathBuf::from(".ttsrs_config.json")
    }

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

    fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(&config_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn get_api_key(&self, provider: &TtsProvider) -> Option<&String> {
        match provider {
            TtsProvider::OpenAI => self.openai_api_key.as_ref(),
            TtsProvider::ElevenLabs => self.elevenlabs_api_key.as_ref(),
            _ => None,
        }
    }

    fn set_api_key(&mut self, provider: &TtsProvider, api_key: String) {
        match provider {
            TtsProvider::OpenAI => self.openai_api_key = Some(api_key),
            TtsProvider::ElevenLabs => self.elevenlabs_api_key = Some(api_key),
            _ => {}
        }
    }
}

/// Command-line arguments for ttsrs.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Text-to-speech CLI using OpenAI or ElevenLabs APIs"
)]
struct Args {
    /// Use classic command-line mode (default launches interactive TUI)
    #[arg(long)]
    cli: bool,

    /// Input text file path
    #[arg()]
    input_file: Option<String>,

    /// TTS model to use
    #[arg(short, long, default_value = "tts-1-hd")]
    model: String,

    /// Voice to use for TTS (OpenAI only)
    #[arg(short, long, default_value = "alloy")]
    voice: String,

    /// Output audio format
    #[arg(short, long, default_value = "flac")]
    format: String,

    /// Speaking speed (0.25–4.0, OpenAI only)
    #[arg(long, default_value = "1.0")]
    speed: f32,

    /// API key (can also be set via OPENAI_API_KEY or ELEVENLABS_API_KEY env vars)
    #[arg(short, long)]
    apikey: Option<String>,

    /// Custom API endpoint URL
    #[arg(long)]
    endpoint_url: Option<String>,

    /// TTS provider
    #[arg(long, default_value = "openai")]
    provider: String,

    /// ElevenLabs voice ID (required for ElevenLabs provider)
    #[arg(long)]
    elevenlabs_voice_id: Option<String>,

    /// ElevenLabs model ID
    #[arg(long, default_value = "eleven_turbo_v2_5")]
    elevenlabs_model: String,

    /// ElevenLabs voice stability (0.0–1.0)
    #[arg(long, default_value = "0.5")]
    elevenlabs_stability: f32,

    /// ElevenLabs voice similarity boost (0.0–1.0)
    #[arg(long, default_value = "0.75")]
    elevenlabs_similarity: f32,

    /// Launch TUI mode for interactive configuration
    #[arg(long)]
    tui: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();

    let launch_tui = !args.cli || args.tui;
    if launch_tui {
        match tui::run_tui()? {
            Some(tui_args) => {
                args = tui_args;
            }
            None => {
                println!("TUI cancelled.");
                return Ok(());
            }
        }
    }

    let provider: TtsProvider = args.provider.parse()?;

    let mut config = Config::load().unwrap_or_default();

    let api_key = args.apikey.clone()
        .or_else(|| match provider {
            TtsProvider::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
            TtsProvider::ElevenLabs => std::env::var("ELEVENLABS_API_KEY").ok(),
            TtsProvider::Custom => None,
        })
        .or_else(|| config.get_api_key(&provider).cloned())
        .or_else(|| {
            let prompt = match provider {
                TtsProvider::OpenAI => "Enter your OpenAI API Key",
                TtsProvider::ElevenLabs => "Enter your ElevenLabs API Key",
                TtsProvider::Custom => "Enter your Custom API Key (or press Enter to skip)",
            };

            if provider == TtsProvider::Custom {
                let input: String = Input::new()
                    .with_prompt(prompt)
                    .allow_empty(true)
                    .interact_text()
                    .unwrap_or_default();
                Some(input)
            } else {
                let input: String = Input::new()
                    .with_prompt(prompt)
                    .interact_text()
                    .ok()?;

                config.set_api_key(&provider, input.clone());
                if let Err(e) = config.save() {
                    eprintln!("Warning: Failed to save API key to config file (.ttsrs_config.json): {}", e);
                } else {
                    println!("API key saved to config file (.ttsrs_config.json) for future use.");
                }

                Some(input)
            }
        })
        .context(
            "API key not provided. Set it via the --apikey flag, the appropriate environment variable, or input it when prompted."
        )?;

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

    // Prompt for voice selection (OpenAI only; ElevenLabs uses voice_id)
    match provider {
        TtsProvider::OpenAI | TtsProvider::Custom => {
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
        }
        TtsProvider::ElevenLabs => {
            // For ElevenLabs, ensure voice_id is provided
            if args.elevenlabs_voice_id.is_none() {
                let input: String = Input::new()
                    .with_prompt("Enter the ElevenLabs voice ID (or run with --elevenlabs-voice-id)")
                    .interact_text()?;
                args.elevenlabs_voice_id = Some(input);
            }
        }
    }

    // Prompt for output format
    let formats = match provider {
        TtsProvider::OpenAI | TtsProvider::Custom => vec!["mp3", "flac", "wav", "pcm", "opus", "aac"],
        TtsProvider::ElevenLabs => vec![
            "mp3_44100_128",
            "mp3_44100_192",
            "pcm_16000",
            "pcm_22050",
            "pcm_24000",
            "pcm_44100",
        ],
    };

    // For ElevenLabs, set default format if using the OpenAI default
    if provider == TtsProvider::ElevenLabs && args.format.to_lowercase() == "flac" {
        args.format = "mp3_44100_128".to_string();
    }

    if ((provider == TtsProvider::OpenAI || provider == TtsProvider::Custom) && args.format.to_lowercase() == "flac")
        || (provider == TtsProvider::ElevenLabs && args.format == "mp3_44100_128")
    {
        // Only prompt if default is used
        let default_idx = if provider == TtsProvider::OpenAI || provider == TtsProvider::Custom {
            1
        } else {
            0
        };
        let selection = Select::new()
            .with_prompt("Select an output format")
            .items(&formats)
            .default(default_idx)
            .interact()?;
        args.format = formats[selection].to_string();
    }

    // Initialize HTTP client
    let client = Client::new();

    let input_file_path = Path::new(args.input_file.as_ref().context("Input file path is missing")?);
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

    let text_content = read_text_file(input_file_path)?;
    let chunks = chunk_text(&text_content);

    // Determine the API endpoint URL
    let api_endpoint = if let Some(custom_url) = args.endpoint_url.as_deref() {
        custom_url.to_string()
    } else {
        match provider {
            TtsProvider::OpenAI => "https://api.openai.com/v1/audio/speech".to_string(),
            TtsProvider::Custom => {
                let input: String = Input::new()
                    .with_prompt("Enter the custom API endpoint URL")
                    .interact_text()?;
                input
            }
            TtsProvider::ElevenLabs => {
                let voice_id = args.elevenlabs_voice_id.as_ref()
                    .context("ElevenLabs voice ID is required when using ElevenLabs provider. Use --elevenlabs-voice-id")?;
                format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice_id)
            }
        }
    };

    let gen_config = AudioGenConfig {
        model: &args.model,
        voice: &args.voice,
        format: &args.format,
        api_key: &api_key,
        speed: args.speed,
        api_endpoint: &api_endpoint,
        provider: &provider,
        elevenlabs_model: &args.elevenlabs_model,
        elevenlabs_stability: args.elevenlabs_stability,
        elevenlabs_similarity: args.elevenlabs_similarity,
    };

    let (timestamp, voice_used) =
        generate_audio_files(&chunks, &output_dir, &client, &gen_config).await?;

    println!(
        "Chunk {} files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        args.format,
        green_text(input_file_name)
    );

    let output_ext = format_to_extension(&args.format);

    combine_audio_files(&output_dir, output_ext, &timestamp, &voice_used)?;

    remove_tmp(&output_dir, output_ext, &timestamp, &voice_used)?;

    println!(
        "\nThe File [ {}/output.{} ] is ready for you. \n",
        green_text(input_file_name),
        output_ext
    );

    Ok(())
}

/// Returns the file extension for the given audio format string.
fn format_to_extension(format: &str) -> &str {
    if format.starts_with("mp3") {
        "mp3"
    } else if format.starts_with("pcm") {
        "pcm"
    } else if format.starts_with("ulaw") {
        "wav"
    } else {
        format
    }
}

/// Returns the ffmpeg encoder name for the given output format.
fn encoder_for_format(format: &str) -> &str {
    match format {
        "mp3" => "libmp3lame",
        "flac" => "flac",
        "wav" | "pcm" => "pcm_s16le",
        "opus" => "libopus",
        "aac" => "aac",
        _ => "libmp3lame",
    }
}

fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

/// Reads a text file and returns its content as a single string.
fn read_text_file(file_path: &Path) -> Result<String> {
    let content = fs::read_to_string(file_path)?;
    Ok(content)
}

/// Chunks the input text into smaller pieces, each containing up to `MAX_TOKENS_PER_CHUNK` tokens.
fn chunk_text(text: &str) -> Vec<String> {
    let bpe = cl100k_base().unwrap();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;
    const MAX_TOKENS_PER_CHUNK: usize = 500;

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let line_token_count = bpe.encode_ordinary(line).len();

        if line_token_count > MAX_TOKENS_PER_CHUNK {
            if !current_chunk.is_empty() {
                chunks.push(std::mem::take(&mut current_chunk));
            }
            chunks.push(line.to_string());
            current_token_count = 0;
            continue;
        }

        if current_token_count + line_token_count > MAX_TOKENS_PER_CHUNK {
            if !current_chunk.is_empty() {
                chunks.push(std::mem::take(&mut current_chunk));
            }
            current_token_count = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(line);
        current_token_count += line_token_count;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_TOKENS_PER_CHUNK: usize = 500;

    fn token_count(text: &str) -> usize {
        cl100k_base().unwrap().encode_ordinary(text).len()
    }

    fn build_line_with_min_tokens(min_tokens: usize) -> String {
        let mut line = String::from("word");
        while token_count(&line) < min_tokens {
            line.push_str(" word");
        }
        line
    }

    fn build_line_with_max_tokens(max_tokens: usize) -> String {
        let mut line = String::new();
        for _ in 0..max_tokens {
            let candidate = if line.is_empty() {
                "word".to_string()
            } else {
                format!("{} word", line)
            };

            if token_count(&candidate) > max_tokens {
                break;
            }

            line = candidate;
        }
        line
    }

    #[test]
    fn chunk_text_starts_new_chunk_at_token_boundary() {
        let line1 = build_line_with_max_tokens(200);
        let line2 = build_line_with_max_tokens(200);
        let line3 = build_line_with_max_tokens(200);
        let input = format!("{}\n{}\n{}", line1, line2, line3);

        let chunks = chunk_text(&input);

        assert_eq!(chunks, vec![format!("{} {}", line1, line2), line3]);
        assert!(token_count(&chunks[0]) <= MAX_TOKENS_PER_CHUNK);
        assert!(token_count(&chunks[1]) <= MAX_TOKENS_PER_CHUNK);
    }

    #[test]
    fn chunk_text_skips_blank_lines_and_joins_with_spaces() {
        let input = "first line\n\n   \nsecond line\n\t\nthird line";

        let chunks = chunk_text(input);

        assert_eq!(chunks, vec!["first line second line third line"]);
    }

    #[test]
    fn chunk_text_keeps_oversized_line_as_its_own_chunk() {
        let prefix = build_line_with_max_tokens(100);
        let oversized = build_line_with_min_tokens(MAX_TOKENS_PER_CHUNK + 1);
        let suffix = build_line_with_max_tokens(100);
        let input = format!("{}\n{}\n{}", prefix, oversized, suffix);

        let chunks = chunk_text(&input);

        assert_eq!(chunks, vec![prefix.clone(), oversized.clone(), suffix.clone()]);
        assert!(token_count(&oversized) > MAX_TOKENS_PER_CHUNK);
        assert!(token_count(&chunks[0]) <= MAX_TOKENS_PER_CHUNK);
        assert!(token_count(&chunks[2]) <= MAX_TOKENS_PER_CHUNK);
    }
}

fn preview_prefix(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

/// Configuration for audio generation, grouping all parameters needed by the TTS API call.
struct AudioGenConfig<'a> {
    model: &'a str,
    voice: &'a str,
    format: &'a str,
    api_key: &'a str,
    speed: f32,
    api_endpoint: &'a str,
    provider: &'a TtsProvider,
    elevenlabs_model: &'a str,
    elevenlabs_stability: f32,
    elevenlabs_similarity: f32,
}

/// Generates audio files for each chunk of text using the specified API endpoint.
/// Returns `(timestamp, voice_lowercase)` for identifying the generated files.
async fn generate_audio_files(
    chunks: &[String],
    output_dir: &Path,
    client: &Client,
    config: &AudioGenConfig<'_>,
) -> Result<(String, String)> {
    let date_time_string = Local::now().format("%Y%m%d%H%M%S").to_string();
    let voice_lowercase = config.voice.to_lowercase();

    for (i, chunk_string) in chunks.iter().enumerate() {
        println!("〰️〰️〰️〰️〰️〰️");
        println!(
            "{} {:06} of {}",
            green_text("Processing chunk"),
            i + 1,
            chunks.len()
        );
        println!("Input String: {}...", preview_prefix(&chunk_string, 60));

        let max_chars = match config.provider {
            TtsProvider::OpenAI => 4096,
            TtsProvider::ElevenLabs => 5000,
            _ => 5000,
        };
        if chunk_string.len() > max_chars {
            eprintln!(
                "Warning: Chunk {:06} exceeds {} characters ({}). Attempting to process, but it might fail.",
                i + 1,
                max_chars,
                chunk_string.len()
            );
        }

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.green} {msg}")?,
        );
        pb.set_message(format!("Sending chunk {} to API...", i + 1));

        let (request_body, auth_header) = match config.provider {
            TtsProvider::OpenAI | TtsProvider::Custom => {
                let body = serde_json::json!({
                    "model": config.model,
                    "voice": voice_lowercase,
                    "input": chunk_string,
                    "speed": config.speed,
                    "response_format": config.format,
                });
                let auth_value = if config.api_key.is_empty() && *config.provider == TtsProvider::Custom {
                    "".to_string()
                } else {
                    format!("Bearer {}", config.api_key)
                };
                (
                    body,
                    (
                        "Authorization".to_string(),
                        auth_value,
                    ),
                )
            }
            TtsProvider::ElevenLabs => {
                let body = serde_json::json!({
                    "text": chunk_string,
                    "model_id": config.elevenlabs_model,
                    "voice_settings": {
                        "stability": config.elevenlabs_stability,
                        "similarity_boost": config.elevenlabs_similarity,
                    }
                });
                (body, ("xi-api-key".to_string(), config.api_key.to_string()))
            }
        };

        let mut request = client
            .post(config.api_endpoint)
            .header("Content-Type", "application/json")
            .json(&request_body);

        if !auth_header.1.is_empty() {
            request = request.header(&auth_header.0, &auth_header.1);
        }

        let response = request.send().await;

        pb.set_message(format!("Waiting for response for chunk {}...", i + 1));

        let response = match response {
            Ok(resp) => resp,
            Err(e) => {
                pb.finish_with_message(format!(
                    "❌ Error sending request for chunk {}: {}",
                    i + 1,
                    e
                ));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Could not read error response body".to_string());

            pb.finish_with_message(format!(
                "❌ API Error for chunk {}: Status Code: {}. Response: {}",
                i + 1,
                status,
                error_text
            ));
            continue;
        }

        let file_ext = format_to_extension(config.format);
        let file_name = format!(
            "tmp_{}_{}_chunk{:06}.{}",
            date_time_string,
            voice_lowercase,
            i + 1,
            file_ext
        );
        let file_path = output_dir.join(&file_name);

        let file_result = async {
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::File::create(&file_path)
                .await
                .context("Failed to create file")?;
            let mut stream = response.bytes_stream();
            while let Some(item) = stream.next().await {
                let chunk_bytes = item.context("Failed to read chunk from response stream")?;
                file.write_all(&chunk_bytes)
                    .await
                    .context("Failed to write chunk to file")?;
            }
            Ok::<(), anyhow::Error>(())
        }
        .await;

        match file_result {
            Ok(()) => pb.finish_with_message(format!(
                "✅ Chunk {} audio saved as {}",
                i + 1,
                file_path.display()
            )),
            Err(e) => {
                pb.finish_with_message(format!("❌ Error saving audio for chunk {}: {}", i + 1, e));
                if let Err(rm_err) = fs::remove_file(&file_path) {
                    eprintln!(
                        "Warning: Failed to clean up partial file {}: {}",
                        file_path.display(),
                        rm_err
                    );
                }
            }
        }
    }

    Ok((date_time_string, voice_lowercase))
}

/// Combines all the generated temporary audio files into a single file using ffmpeg.
fn combine_audio_files(
    output_dir: &Path,
    format: &str,
    timestamp: &str,
    voice: &str,
) -> Result<()> {
    let mut input_files = Vec::new();
    let prefix = format!("tmp_{}_{}_", timestamp, voice);
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .map(|ext| ext.to_str() == Some(format))
                .unwrap_or(false)
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name_str| name_str.starts_with(&prefix))
                .unwrap_or(false)
        {
            input_files.push(path);
        }
    }

    if input_files.is_empty() {
        println!(
            "No temporary audio files found to combine for format '{}'. Skipping combination.",
            format
        );
        return Ok(());
    }

    println!(
        "Combining {} temporary {} files...",
        input_files.len(),
        format
    );

    input_files.sort();

    let list_file_path = output_dir.join("ffmpeg_list.txt");
    {
        let mut list_file = File::create(&list_file_path)?;
        for input_file in &input_files {
            let filename = input_file
                .file_name()
                .and_then(|name| name.to_str())
                .context("Failed to get filename")?;
            writeln!(list_file, "file '{}'", filename)?;
        }
    }

    let output_file_path = output_dir.join(format!("output.{}", format));
    let encoder = encoder_for_format(format);

    println!("Checking for internal ffmpeg...");
    // Auto-download handles checking if it already exists and downloading if not
    if let Err(e) = auto_download() {
        eprintln!("Warning: Failed to ensure internal ffmpeg is available: {}", e);
        eprintln!("It will fallback to attempting to use a system-installed ffmpeg.");
    }

    let ffmpeg_args = vec![
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file_path.to_str().context("Invalid UTF-8 in list file path")?,
        "-c:a",
        encoder,
        "-y",
        output_file_path.to_str().context("Invalid UTF-8 in output file path")?,
    ];

    println!("Running internal ffmpeg command...");
    let mut ffmpeg_command = FfmpegCommand::new();
    ffmpeg_command.args(&ffmpeg_args);

    // Fallback to calling inner process output directly
    let ffmpeg_output = ffmpeg_command.as_inner_mut()
        .output()
        .context("Failed to execute ffmpeg command. Is ffmpeg available?")?;

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

#[cfg(test)]
mod tests {
    use super::{encoder_for_format, format_to_extension, preview_prefix};

    #[test]
    fn test_encoder_selection() {
        let test_cases = vec![
            ("mp3", "libmp3lame"),
            ("flac", "flac"),
            ("wav", "pcm_s16le"),
            ("pcm", "pcm_s16le"),
            ("opus", "libopus"),
            ("aac", "aac"),
            ("unknown", "libmp3lame"),
            ("mp3_44100_128", "libmp3lame"),
        ];

        for (format, expected_encoder) in test_cases {
            assert_eq!(
                encoder_for_format(format),
                expected_encoder,
                "Format '{}' should use encoder '{}'",
                format,
                expected_encoder
            );
        }
    }

    #[test]
    fn test_format_to_extension() {
        assert_eq!(format_to_extension("mp3"), "mp3");
        assert_eq!(format_to_extension("mp3_44100_128"), "mp3");
        assert_eq!(format_to_extension("pcm_16000"), "pcm");
        assert_eq!(format_to_extension("ulaw_8000"), "wav");
        assert_eq!(format_to_extension("flac"), "flac");
        assert_eq!(format_to_extension("opus"), "opus");
    }

    #[test]
    fn test_preview_prefix_handles_multibyte_characters() {
        let text = "Lord Durham’s report – Québec et Montréal 😊";
        let preview = preview_prefix(text, 10);
        assert_eq!(preview, "Lord Durha");
    }
}
