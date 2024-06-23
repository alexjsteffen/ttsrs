#[macro_use]
extern crate serde_derive;
extern crate reqwest;
extern crate tokio;
extern crate clap;
extern crate tokio_stream;

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime};
use std::path::Path;
use clap::{Arg, App};
use tokio::sync::mpsc;
use tokio::time::timeout;

#[derive(Serialize, Deserialize)]
struct TTSResponse {
    url: String,
}

async fn generate_audio_files(chunks: Vec<Vec<String>>, output_dir: &str, model: &str, voice: &str, api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs()
        .to_string();
    
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_string = chunk.join(" ");
        if chunk_string.len() > 4000 {
            println!("Chunk {} is more than 4000 characters, please make it shorter", i + 1);
            std::process::exit(1);
        }

        let (sx, rx) = mpsc::channel(1);
        thread::spawn(move || start_animation(rx));

        let response = client.post("https://api.openai.com/v1/audio/speech")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({ "model": model, "voice": voice, "response_format": "flac", "input": chunk_string }))
            .send()
            .await?
            .json::<TTSResponse>()
            .await?;
        
        sx.send(true).unwrap();

        // Download the returned audio file
        let response_bytes = client
            .get(&response.url)
            .send()
            .await?
            .bytes()
            .await?;
        
        let file_path = format!("{}/tmp_{}_chunk{:06}.flac", output_dir, now, i + 1);
        let mut file = File::create(file_path)?;
        file.write_all(&response_bytes)?;

        println!("A flac file saved as {}/tmp_{}_chunk{:06}.flac", output_dir, now, i + 1);
    }

    Ok(())
}

fn start_animation(rx: mpsc::Receiver<bool>) {
    let braille_chars = ["⡿", "⣟", "⣯", "⣷", "⣾", "⣽", "⣻", "⢿"];
    let mut i = 0;
    loop {
        if let Ok(true) = timeout(Duration::from_millis(100), rx.recv()).await {
            break;
        }
        print!("\r{}", braille_chars[i % braille_chars.len()]);
        thread::sleep(Duration::from_millis(100));
        i += 1;
    }
    println!("\nDone");
}

fn green_text(text: &str) -> String {
    format!("\x1b[92m{}\x1b[0m", text)
}

fn read_text_file(file_path: &str) -> Vec<String> {
    BufReader::new(File::open(file_path).unwrap())
        .lines()
        .filter_map(Result::ok)
        .collect()
}

fn chunk_text(lines: &[String]) -> Vec<Vec<String>> {
    // Mock function for encoding length calculation.
    // Replace this with actual tokenization calculation. 
    let encoding_fn = |line: &str| -> usize { line.len() };

    let mut chunks = vec![];
    let mut current_chunk = vec![];
    let mut current_token_count = 0;

    for line in lines {
        let line_token_count = encoding_fn(line);

        if current_token_count + line_token_count > 500 {
            chunks.push(current_chunk);
            current_chunk = vec![];
            current_token_count = 0;
        }

        current_chunk.push(line.to_string());
        current_token_count += line_token_count;
    }

    chunks.push(current_chunk);
    chunks
}

fn find_ffmpeg_path() -> Option<String> {
    let output = Command::new("which")
        .arg("ffmpeg")
        .output()
        .expect("Failed to execute `which ffmpeg` command!");
    
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

fn combine_audio_files(input_file_path: &str, output_dir: &str) {
    let ffmpeg_path = find_ffmpeg_path().expect("FFmpeg executable not found in system PATH");
    let input_file_name = input_file_path.rsplit('.').nth(1).unwrap();
    let flac_files: Vec<_> = fs::read_dir(output_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|f| f.file_name().to_str().unwrap().ends_with(".flac"))
        .collect();

    let concat_file_path = format!("{}/concat.txt", output_dir);
    let mut concat_file = File::create(&concat_file_path).unwrap();
    for flac_file in flac_files {
        writeln!(concat_file, "file '{}'", flac_file.path().display()).unwrap();
    }

    let output_file_path = format!("{}/{}.flac", output_dir, input_file_name);
    Command::new(ffmpeg_path)
        .arg("-f")
        .arg("concat")
        .arg("-safe")
        .arg("0")
        .arg("-i")
        .arg(&concat_file_path)
        .arg("-c")
        .arg("copy")
        .arg(&output_file_path)
        .status()
        .expect("Failed to execute ffmpeg");
}

fn remove_tmp_files(output_dir: &str) {
    for entry in fs::read_dir(output_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_str().unwrap().starts_with("tmp") || entry.file_name().to_str().unwrap() == "concat.txt" {
            fs::remove_file(entry.path()).unwrap();
        }
    }
}

#[tokio::main]
async fn main() {
    let matches = App::new("Generate audio files from text using OpenAI TTS.")
        .arg(Arg::with_name("input_file").help("Input text file name").required(true))
        .arg(Arg::with_name("model").short("m").long("model").default_value("tts-1-hd").possible_values(&["tts-1-hd", "tts-1"]).help("TTS model to use"))
        .arg(Arg::with_name("voice").short("v").long("voice").default_value("fable").possible_values(&["alloy", "fable", "echo", "onyx", "shimmer", "nova"]).help("Voice to use for TTS"))
        .get_matches();

    let input_file_path = matches.value_of("input_file").unwrap();
    let model = matches.value_of("model").unwrap();
    let voice = matches.value_of("voice").unwrap();

    let input_file_name = input_file_path.rsplit('/').next().unwrap().rsplit('.').nth(1).unwrap();
    println!("Now Create a folder called {} for you.", green_text(input_file_name));
    let output_dir = format!("./{}", input_file_name);
    if !fs::metadata(&output_dir).is_ok() {
        fs::create_dir(&output_dir).unwrap();
    }

    let lines = read_text_file(input_file_path);
    let chunks = chunk_text(&lines);

    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    generate_audio_files(chunks, &output_dir, model, voice, &api_key).await.unwrap();

    println!("Chunk flac files are already in [ {} ] for ffmpeg to combine.\n\n", green_text(&output_dir));
    combine_audio_files(input_file_path, &output_dir);
    remove_tmp_files(&output_dir);
    println!("\nThe File [ {} ] is ready for you. \n", green_text(input_file_name));
}