// Import necessary crates and modules
use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use dialoguer::{Input, Select};
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use tiktoken_rs::cl100k_base;

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

    // Prompt for voice selection
    let voices = vec![
        "Echo - Clear and professional, ideal for announcements.",
        "Fable - Warm and engaging, perfect for storytelling.",
        "Onyx - Deep and authoritative.",
        "Nova - Young and energetic.",
        "Shimmer - Soft and soothing.",
        "Alloy - Versatile and well-balanced.",
        "Ballad - New!",
        "Coral - New!",
        "Sage - New!",
    ];
    if args.voice == "alloy" {
        // Only prompt if default is used
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

    // Prompt for output format
    let formats = vec!["mp3", "flac", "wav", "pcm", "opus", "aac"];
    if args.format == "flac" {
        // Only prompt if default is used
        let selection = Select::new()
            .with_prompt("Select an output format")
            .items(&formats)
            .default(1) // Set default index for flac
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
    let api_endpoint = args
        .endpoint_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1/audio/speech"); // Use custom URL or default

    // Generate audio files for each chunk
    generate_audio_files(
        &chunks,
        &output_dir,
        &args.model,
        &args.voice,
        &args.format,
        &client,
        &api_key,
        args.speed,
        api_endpoint, // Pass the determined endpoint URL
    )
    .await?; //

    // Notify the user about the generated files
    println!(
        "Chunk {} files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        args.format, // Use the selected format in the message
        green_text(input_file_name)
    );

    // Combine the audio files into a single output file
    combine_audio_files(&output_dir, &args.format)?; //

    // Remove temporary files
    remove_tmp(&output_dir, &args.format)?; // Pass format to remove correct tmp files

    // Final message
    println!(
        "\nThe File [ {}.{} ] is ready for you. \n",
        green_text(input_file_name),
        args.format
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
) -> Result<()> {
    //
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
            green_text("Processing chunk"), // Changed message slightly
            format!("{:06}", i + 1),
            chunks.len()
        );
        println!(
            "Input String: {}...",
            &chunk_string[..chunk_string.len().min(60)]
        );

        // Check if the chunk exceeds the character limit (OpenAI specific limit)
        const MAX_CHARS_PER_CHUNK: usize = 4096; // Use OpenAI's documented limit
        if chunk_string.len() > MAX_CHARS_PER_CHUNK {
            eprintln!( // Use eprintln for errors
                "Warning: Chunk {:06} exceeds {} characters ({}). Attempting to process, but it might fail.",
                i + 1,
                MAX_CHARS_PER_CHUNK,
                chunk_string.len()
            );
            // Optionally, you could truncate here:
            // chunk_string = chunk_string[..MAX_CHARS_PER_CHUNK].to_string();
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
        let request_body = serde_json::json!({
            "model": model,
            "voice": voice_lowercase,
            "input": chunk_string,
            "speed": speed,
            "response_format": format,
        });

        let response = client
            .post(api_endpoint) // Use the passed endpoint URL
            .header("Authorization", format!("Bearer {}", api_key))
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

        // Save the audio response to a file
        let file_name = format!("tmp_{}_chunk{:06}.{}", date_time_string, i + 1, format);
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

    Ok(())
}

/// Combines all the generated temporary audio files into a single file using ffmpeg.
fn combine_audio_files(output_dir: &Path, format: &str) -> Result<()> {
    //
    // Collect all the temporary files of the specified format in the output directory
    let mut input_files = Vec::new();
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
                .map(|name_str| name_str.starts_with("tmp_")) // Check prefix
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
            // Ensure paths are properly quoted/escaped if needed, especially on Windows
            // For simplicity here, assuming paths don't contain problematic characters.
            // A robust solution might involve more complex path handling.
            writeln!(
                list_file,
                "file '{}'",
                input_file.to_str().unwrap().replace('\\', "/")
            )?; // Use forward slashes
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

/// Removes temporary files from the output directory matching the specified format.
fn remove_tmp(output_dir: &Path, format: &str) -> Result<()> {
    //
    let mut removed_count = 0;
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name_str| name_str.starts_with("tmp_"))
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
