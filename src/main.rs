use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tiktoken_rs::cl100k_base;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input text file name
    input_file: String,

    /// TTS model to use
    #[arg(short, long, default_value = "tts-1-hd")]
    model: String,

    /// Voice to use for TTS
    #[arg(short, long, default_value = "fable")]
    voice: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    error: Option<OpenAIError>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
}

/// The main function of the program.
/// 
/// This function orchestrates the entire process of text-to-speech conversion:
/// 1. Parses command-line arguments
/// 2. Reads the input text file
/// 3. Chunks the text into smaller pieces
/// 4. Generates audio files for each chunk
/// 5. Combines the audio files
/// 6. Cleans up temporary files
///
/// # Returns
///
/// Returns a `Result<()>` which is `Ok(())` if the process completes successfully,
/// or an `Err` containing the error information if something goes wrong.

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY not set")?;
    let client = Client::new();

    let input_file_path = Path::new(&args.input_file);
    let input_file_name = input_file_path
        .file_stem()
        .context("Invalid input file")?
        .to_str()
        .context("Invalid input file name")?;

    println!("Now creating a folder called {} for you.", green_text(input_file_name));
    let output_dir = Path::new("./").join(input_file_name);
    fs::create_dir_all(&output_dir)?;

    let lines = read_text_file(input_file_path)?;
    let chunks = chunk_text(&lines);

    generate_audio_files(&chunks, &output_dir, &args.model, &args.voice, &client, &api_key).await?;

    println!(
        "Chunk flac files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        green_text(input_file_name)
    );

    combine_audio_files(&output_dir)?;

    remove_tmp(&output_dir)?;

    println!("\nThe File [ {} ] is ready for you. \n", green_text(input_file_name));

    Ok(())
}

/// Formats the given text in green color for console output.
///
/// # Arguments
///
/// * `text` - A string slice that holds the text to be colored.
///
/// # Returns
///
/// A `String` containing the input text wrapped in ANSI escape codes for green color.

fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

/// Reads a text file and returns its contents as a vector of strings.
///
/// # Arguments
///
/// * `file_path` - A `Path` reference to the file to be read.
///
/// # Returns
///
/// A `Result` containing a `Vec<String>` where each element is a non-empty line from the file.

fn read_text_file(file_path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(file_path)?;
    Ok(
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(String::from)
            .collect()
    )
}

/// Chunks the input text into smaller pieces, each containing up to 500 tokens.
///
/// # Arguments
///
/// * `lines` - A slice of `String`s, each representing a line of text.
///
/// # Returns
///
/// A `Vec<Vec<String>>` where each inner `Vec<String>` is a chunk of the input text.

fn chunk_text(lines: &[String]) -> Vec<Vec<String>> {
    let bpe = cl100k_base().unwrap();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_token_count = 0;

    for line in lines {
        let line_token_count = bpe.encode_ordinary(line).len();

        if current_token_count + line_token_count > 500 {
            chunks.push(std::mem::take(&mut current_chunk));
            current_token_count = 0;
        }

        current_chunk.push(line.clone());
        current_token_count += line_token_count;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Generates audio files for each chunk of text using the OpenAI API.
///
/// # Arguments
///
/// * `chunks` - A slice of text chunks, where each chunk is a `Vec<String>`.
/// * `output_dir` - The directory where the audio files will be saved.
/// * `model` - The TTS model to use.
/// * `voice` - The voice to use for TTS.
/// * `client` - An HTTP client for making API requests.
/// * `api_key` - The OpenAI API key.
///
/// # Returns
///
/// A `Result<()>` which is `Ok(())` if all audio files are generated successfully,
/// or an `Err` containing the error information if something goes wrong.

async fn generate_audio_files(
    chunks: &[Vec<String>],
    output_dir: &Path,
    model: &str,
    voice: &str,
    client: &Client,
    api_key: &str
) -> Result<()> {
    let date_time_string = Local::now().format("%Y%m%d%H%M").to_string();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_string = chunk.join(" ");
        println!("〰️〰️〰️〰️〰️〰️");
        println!(
            "{} {} of {}",
            green_text("Prepare for the chunk"),
            format!("{:06}", i + 1),
            chunks.len()
        );
        println!("Input String: {}...", &chunk_string[..chunk_string.len().min(60)]);

        if chunk_string.len() > 4000 {
            anyhow::bail!(
                "Chunk {:06}: {} is more than 4000 characters, please make it shorter",
                i + 1,
                &chunk_string[..60]
            );
        }

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.green} {msg}")?
        );
        pb.set_message("Generating audio...");

        let response = client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(
                &serde_json::json!({
                "model": model,
                "voice": voice,
                "input": chunk_string,
            })
            )
            .send().await?;

        if !response.status().is_success() {
            let error: OpenAIResponse = response.json().await?;
            if let Some(error) = error.error {
                anyhow::bail!("OpenAI API error: {}", error.message);
            } else {
                anyhow::bail!("Unknown OpenAI API error");
            }
        }

        let file_name = format!("tmp_{}_chunk{:06}.flac", date_time_string, i + 1);
        let file_path = output_dir.join(&file_name);
        let mut file = File::create(&file_path)?;

        let mut stream = response.bytes_stream();
        while let Some(item) = stream.next().await {
            file.write_all(&item?)?;
        }

        pb.finish_with_message(format!("Audio file saved as {}", file_path.display()));
    }

    Ok(())
}

/// Combines all the generated audio files into a single file using ffmpeg.
///
/// # Arguments
///
/// * `output_dir` - The directory containing the audio files to be combined.
///
/// # Returns
///
/// A `Result<()>` which is `Ok(())` if the audio files are combined successfully,
/// or an `Err` containing the error information if something goes wrong.

fn combine_audio_files(output_dir: &Path) -> Result<()> {
    let mut input_files = Vec::new();
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|ext| ext == "flac").unwrap_or(false) && path.file_name().unwrap().to_str().unwrap().starts_with("tmp") {
            input_files.push(path);
        }
    }

    input_files.sort();

    let mut ffmpeg_args = Vec::new();
    for input_file in &input_files {
        ffmpeg_args.push("-i".to_string());
        ffmpeg_args.push(input_file.to_str().unwrap().to_string());
    }
    ffmpeg_args.push("-filter_complex".to_string());
    ffmpeg_args.push(format!(
        "concat=n={}:v=0:a=1[outa]",
        input_files.len()
    ));
    ffmpeg_args.push("-map".to_string());
    ffmpeg_args.push("[outa]".to_string());
    ffmpeg_args.push("-c:a".to_string());
    ffmpeg_args.push("flac".to_string());
    ffmpeg_args.push("-y".to_string()); // Overwrite output files without asking
    ffmpeg_args.push(output_dir.join("combined.flac").to_str().unwrap().to_string());

    let status = Command::new("ffmpeg")
        .args(&ffmpeg_args)
        .status()?;

    if !status.success() {
        anyhow::bail!("ffmpeg command failed");
    }

    Ok(())
}

/// Removes temporary files from the output directory.
///
/// # Arguments
///
/// * `output_dir` - The directory containing the temporary files to be removed.
///
/// # Returns
///
/// A `Result<()>` which is `Ok(())` if all temporary files are removed successfully,
/// or an `Err` containing the error information if something goes wrong.

fn remove_tmp(output_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.file_name().unwrap().to_str().unwrap().starts_with("tmp") {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
