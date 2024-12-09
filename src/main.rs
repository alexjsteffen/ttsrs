// Import necessary crates and modules
use anyhow::{ Context, Result };
use chrono::Local;
use clap::Parser;
use futures::stream::StreamExt;
use indicatif::{ ProgressBar, ProgressStyle };
use reqwest::Client;
use serde::Deserialize;
use std::fs::{ self, File };
use std::io::Write;
use std::path::Path;
use std::process::Command;
use tiktoken_rs::cl100k_base;
use dialoguer::{Input, Select};

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

    /// Output audio format (options: mp3, flac, wav, pcm)
    #[arg(short, long, default_value = "flac")]
    format: String,

    /// OpenAI API key (optional, can also be set via the OPENAI_API_KEY environment variable)
    #[arg(short, long)]
    apikey: Option<String>,
}

// Structs for deserializing OpenAI API responses and errors
#[derive(Deserialize)]
struct OpenAIResponse {
    error: Option<OpenAIError>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
}

/// The main function of the program.
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let mut args = Args::parse();

    // Get the API key from either the command-line argument or the environment variable
    let api_key = args.apikey.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| {
            // Prompt the user for the API key if not provided
            let input: String = Input::new()
                .with_prompt("Enter your OpenAI API Key")
                .interact_text()
                .ok()?;
            Some(input)
        })
        .context(
            "OpenAI API key not provided. Set it via the --apikey flag, the OPENAI_API_KEY environment variable, or input it when prompted."
        )?;

    // Prompt for input file if not provided
    let input_file = args.input_file.clone().or_else(|| {
        let input: String = Input::new()
            .with_prompt("Enter the path to the input text file")
            .interact_text()
            .ok()?;
        Some(input)
    }).context("Input file not provided.")?;

    args.input_file = Some(input_file);

    // Prompt for voice selection
    let voices = vec![
        "Echo - A clear and bright voice ideal for announcements.",
        "Fable - Great for storytelling.",
        "Onyx - A deep and resonant voice.",
        "Nova - A youthful and energetic voice.",
        "Shimmer - A soft and soothing voice.",
        "Alloy - A versatile and natural-sounding voice.",
    ];
    if args.voice == "alloy" {
        let selection = Select::new()
            .with_prompt("Select a voice")
            .items(&voices)
            .default(0)
            .interact()?;
        args.voice = voices[selection].split(" - ").next().unwrap().to_string();
    }

    // Prompt for output format
    let formats = vec!["mp3", "flac", "wav", "pcm"];
    if args.format == "flac" {
        let selection = Select::new()
            .with_prompt("Select an output format")
            .items(&formats)
            .default(1)
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
    println!("Now creating a folder called {} for you.", green_text(input_file_name));
    let output_dir = Path::new("./").join(input_file_name);
    fs::create_dir_all(&output_dir)?;

    // Read the input file and chunk the text
    let lines = read_text_file(input_file_path)?;
    let chunks = chunk_text(&lines);

    // Generate audio files for each chunk
    generate_audio_files(
        &chunks,
        &output_dir,
        &args.model,
        &args.voice,
        &args.format,
        &client,
        &api_key
    ).await?;

    // Notify the user about the generated files
    println!(
        "Chunk flac files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        green_text(input_file_name)
    );

    // Combine the audio files into a single output file
    combine_audio_files(&output_dir, &args.format)?;

    // Remove temporary files
    remove_tmp(&output_dir)?;

    // Final message
    println!("\nThe File [ {} ] is ready for you. \n", green_text(input_file_name));

    Ok(())
}

// Formats text in green color for console output
fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

// Reads a text file and returns its contents as a vector of strings
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

// Chunks the input text into smaller pieces, each containing up to 500 tokens
fn chunk_text(lines: &[String]) -> Vec<Vec<String>> {
    // Initialize the tokenizer
    let bpe = cl100k_base().unwrap();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_token_count = 0;

    // Iterate over each line of text
    for line in lines {
        // Calculate the number of tokens in the current line
        let line_token_count = bpe.encode_ordinary(line).len();

        // If adding this line exceeds the token limit, start a new chunk
        if current_token_count + line_token_count > 500 {
            chunks.push(std::mem::take(&mut current_chunk));
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

// Generates audio files for each chunk of text using the OpenAI API
async fn generate_audio_files(
    chunks: &[Vec<String>],
    output_dir: &Path,
    model: &str,
    voice: &str,
    format: &str,
    client: &Client,
    api_key: &str
) -> Result<()> {
    // Generate a timestamp for file naming
    let date_time_string = Local::now().format("%Y%m%d%H%M").to_string();
    
    // Convert voice name to lowercase for API call
    let voice_lowercase = voice.to_lowercase();

    // Iterate over each chunk
    for (i, chunk) in chunks.iter().enumerate() {
        // Join the lines in the chunk into a single string
        let chunk_string = chunk.join(" ");
        println!("〰️〰️〰️〰️〰️〰️");
        println!(
            "{} {} of {}",
            green_text("Prepare for the chunk"),
            format!("{:06}", i + 1),
            chunks.len()
        );
        println!("Input String: {}...", &chunk_string[..chunk_string.len().min(60)]);

        // Check if the chunk exceeds the character limit
        if chunk_string.len() > 4000 {
            anyhow::bail!(
                "Chunk {:06}: {} is more than 4000 characters, please make it shorter",
                i + 1,
                &chunk_string[..60]
            );
        }

        // Show a progress bar while generating audio
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.green} {msg}")?
        );
        pb.set_message("Generating audio...");

        // Make the API request to OpenAI
        let response = client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(
                &serde_json::json!({
                "model": model,
                "voice": voice_lowercase,  // Use lowercase voice name
                "input": chunk_string,
            })
            )
            .send().await?;

        // Handle API errors
        if !response.status().is_success() {
            let error: OpenAIResponse = response.json().await?;
            if let Some(error) = error.error {
                anyhow::bail!("OpenAI API error: {}", error.message);
            } else {
                anyhow::bail!("Unknown OpenAI API error");
            }
        }

        // Save the audio response to a file
        let file_name = format!("tmp_{}_chunk{:06}.{}", date_time_string, i + 1, format);
        let file_path = output_dir.join(&file_name);
        let mut file = File::create(&file_path)?;

        // Stream the response and write it to the file
        let mut stream = response.bytes_stream();
        while let Some(item) = stream.next().await {
            file.write_all(&item?)?;
        }

        pb.finish_with_message(format!("Audio file saved as {}", file_path.display()));
    }

    Ok(())
}

/// Combines all the generated audio files into a single file using ffmpeg.
fn combine_audio_files(output_dir: &Path, format: &str) -> Result<()> {
    // Collect all the temporary flac files in the output directory
    let mut input_files = Vec::new();
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if
            path
                .extension()
                .map(|ext| ext == format)
                .unwrap_or(false) &&
            path.file_name().unwrap().to_str().unwrap().starts_with("tmp")
        {
            input_files.push(path);
        }
    }

    // Sort the files to ensure they are combined in the correct order
    input_files.sort();

    // Construct the ffmpeg command arguments
    let mut ffmpeg_args = Vec::new();
    for input_file in &input_files {
        ffmpeg_args.push("-i".to_string());
        ffmpeg_args.push(input_file.to_str().unwrap().to_string());
    }
    ffmpeg_args.push("-filter_complex".to_string());
    ffmpeg_args.push(format!("concat=n={}:v=0:a=1[outa]", input_files.len()));
    ffmpeg_args.push("-map".to_string());
    ffmpeg_args.push("[outa]".to_string());
    ffmpeg_args.push("-c:a".to_string());
    ffmpeg_args.push(format.to_string());
    ffmpeg_args.push("-y".to_string()); // Overwrite output files without asking
    ffmpeg_args.push(output_dir.join(format!("output.{}", format)).to_str().unwrap().to_string());

    // Execute the ffmpeg command
    let status = Command::new("ffmpeg").args(&ffmpeg_args).status()?;

    if !status.success() {
        anyhow::bail!("ffmpeg command failed");
    }

    Ok(())
}

/// Removes temporary files from the output directory.
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
