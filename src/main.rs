use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
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

#[tokio::main]
/// The main function in this Rust code reads text from a file, chunks the text, generates audio files
/// based on the chunks using a specified model and voice, combines the audio files, and then cleans up
/// temporary files, ultimately creating a final audio file.
///
/// Returns:
///
/// The `main` function is returning a `Result` with a unit type `()` indicating that it can return
/// either `Ok(())` if the program execution is successful or an `Err` containing an error if any of the
/// operations encounter an issue.
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

    println!(
        "Now creating a folder called {} for you.",
        green_text(input_file_name)
    );
    let output_dir = Path::new("./").join(input_file_name);
    fs::create_dir_all(&output_dir)?;

    let lines = read_text_file(input_file_path)?;
    let chunks = chunk_text(&lines);

    generate_audio_files(
        &chunks,
        &output_dir,
        &args.model,
        &args.voice,
        &client,
        &api_key,
    )
    .await?;

    println!(
        "Chunk flac files are already in [ ./{} ] for ffmpeg to combine.\n\n",
        green_text(input_file_name)
    );

    combine_audio_files(input_file_path, &output_dir)?;
    remove_tmp(&output_dir)?;

    println!(
        "\nThe File [ {} ] is ready for you. \n",
        green_text(input_file_name)
    );

    Ok(())
}

/// The function `green_text` takes a string input and returns the same text with green color
/// formatting.
///
/// Arguments:
///
/// * `text`: The `green_text` function takes a reference to a string (`&str`) as input and returns a
/// new string with the input text formatted in green color. The function achieves this by using ANSI
/// escape codes for colors.
///
/// Returns:
///
/// A string with the input text formatted in green color using ANSI escape codes.
fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

/// The function `read_text_file` reads the content of a text file, filters out empty lines, and returns
/// the non-empty lines as a vector of strings.
///
/// Arguments:
///
/// * `file_path`: The `file_path` parameter is a reference to a `Path` object, which represents the
/// location of the text file that you want to read.
///
/// Returns:
///
/// The function `read_text_file` is returning a `Result` containing a `Vec<String>`.
fn read_text_file(file_path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(file_path)?;
    Ok(content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(String::from)
        .collect())
}

/// The `chunk_text` function in Rust splits a list of text lines into chunks based on a token count
/// limit of 500 using a Byte Pair Encoding (BPE) model.
///
/// Arguments:
///
/// * `lines`: The function `chunk_text` takes a slice of `String` lines as input and splits them into
/// chunks based on a token count limit of 500. Each chunk is a vector of strings.
///
/// Returns:
///
/// The `chunk_text` function returns a `Vec<Vec<String>>`, which is a vector of vectors of strings.
/// Each inner vector represents a chunk of text lines that have been grouped together based on a token
/// count limit of 500 tokens per chunk.
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

/// The function `generate_audio_files` asynchronously generates audio files from text chunks using the
/// OpenAI API and saves them to a specified output directory.
///
/// Arguments:
///
/// * `chunks`: The `chunks` parameter is a slice of vectors of strings. Each vector represents a chunk
/// of text that needs to be converted into audio files. The function processes each chunk sequentially,
/// generating an audio file for each chunk.
/// * `output_dir`: The `output_dir` parameter in the `generate_audio_files` function represents the
/// directory where the audio files will be saved. It is of type `&Path`, which is a reference to a
/// `Path` object that specifies the location where the audio files will be stored. You can provide the
/// function
/// * `model`: The `model` parameter in the `generate_audio_files` function represents the model used
/// for generating audio in the OpenAI API. This model determines the style and characteristics of the
/// generated speech. It could be a specific language model, a voice style, or any other model supported
/// by the OpenAI API
/// * `voice`: The `voice` parameter in the `generate_audio_files` function represents the voice style
/// or type that will be used for generating the audio files. It is a string that specifies the voice
/// model to be used by the OpenAI API for converting the input text into speech. This parameter allows
/// you to choose
/// * `client`: The `client` parameter in the `generate_audio_files` function is of type `Client`, which
/// is likely an HTTP client used to make requests to the OpenAI API for generating audio files. It is
/// used to send a POST request to the OpenAI API endpoint `https://api.openai.com
/// * `api_key`: The `api_key` parameter in the `generate_audio_files` function is the API key required
/// for authentication when making requests to the OpenAI API. This key is used to authorize and
/// identify the user making the API calls. Make sure to keep your API key secure and not expose it
/// publicly.
///
/// Returns:
///
/// The `generate_audio_files` function returns a `Result` with an empty tuple `()` as the success type.
async fn generate_audio_files(
    chunks: &[Vec<String>],
    output_dir: &Path,
    model: &str,
    voice: &str,
    client: &Client,
    api_key: &str,
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
        println!(
            "Input String: {}...",
            &chunk_string[..chunk_string.len().min(60)]
        );

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
                .template("{spinner:.green} {msg}")?,
        );
        pb.set_message("Generating audio...");

        let response = client
            .post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": model,
                "voice": voice,
                "input": chunk_string,
            }))
            .send()
            .await?;

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

/// The function `combine_audio_files` in Rust combines multiple FLAC audio files into a single FLAC
/// file using ffmpeg.
///
/// Arguments:
///
/// * `input_file_path`: The `input_file_path` parameter is the path to the input audio file that you
/// want to combine with the existing FLAC files in the `output_dir`.
/// * `output_dir`: The `output_dir` parameter in the `combine_audio_files` function represents the
/// directory where the output files will be stored. This function reads all FLAC files from this
/// directory, creates a concatenation file (`concat.txt`) listing these files, and then uses `ffmpeg`
/// to combine them into a single
///
/// Returns:
///
/// The `combine_audio_files` function returns a `Result` with the unit type `()` as the success type.
fn combine_audio_files(input_file_path: &Path, output_dir: &Path) -> Result<()> {
    let input_file_name = input_file_path
        .file_stem()
        .context("Invalid input file")?
        .to_str()
        .context("Invalid input file name")?;
    let flac_files: Vec<_> = fs::read_dir(output_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "flac" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    let concat_file_path = output_dir.join("concat.txt");
    let mut concat_file = File::create(&concat_file_path)?;
    for flac_file in &flac_files {
        writeln!(
            concat_file,
            "file '{}'",
            flac_file.strip_prefix(output_dir)?.to_str().unwrap()
        )?;
    }

    let output_file_path = output_dir.join(format!("{}.flac", input_file_name));
    let status = Command::new("ffmpeg")
        .args(&[
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file_path.to_str().unwrap(),
            "-c:a",
            "flac",
            output_file_path.to_str().unwrap(),
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("ffmpeg command failed");
    }

    Ok(())
}

/// The above Rust code defines a function `remove_tmp` that takes a reference to a `Path` as input and
/// returns a `Result`. The function iterates over the entries in the directory specified by
/// `output_dir`. For each entry, it checks if the file name starts with "tmp". If it does, the file is
/// removed using `fs::remove_file`. After iterating through all entries, the code then attempts to
/// remove the file named "concat.txt" in the `output_dir`. Finally, the function returns `Ok(())` to
/// indicate successful completion.
fn remove_tmp(output_dir: &Path) -> Result<()> {
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("tmp")
        {
            fs::remove_file(path)?;
        }
    }
    fs::remove_file(output_dir.join("concat.txt"))?;
    Ok(())
}
