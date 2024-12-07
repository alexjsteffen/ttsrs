
use anyhow::Result;
use std::path::Path;
use std::fs::File;
use std::io::Write;

pub enum AudioFormat {
    WAV,
    FLAC,
    MP3,
    PCM,
}

impl AudioFormat {
    pub fn from_str(format: &str) -> Option<Self> {
        match format.to_lowercase().as_str() {
            "wav" => Some(AudioFormat::WAV),
            "flac" => Some(AudioFormat::FLAC),
            "mp3" => Some(AudioFormat::MP3),
            "pcm" => Some(AudioFormat::PCM),
            _ => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::WAV => "wav",
            AudioFormat::FLAC => "flac",
            AudioFormat::MP3 => "mp3",
            AudioFormat::PCM => "pcm",
        }
    }
}

pub fn save_audio_file(data: bytes::Bytes, output_path: &Path, format: &AudioFormat) -> Result<()> {
    let mut file = File::create(output_path)?;

    match format {
        AudioFormat::WAV => {
            file.write_all(&data)?;
        },
        AudioFormat::FLAC => {
            file.write_all(&data)?;
        },
        AudioFormat::MP3 => {
            file.write_all(&data)?;
        },
        AudioFormat::PCM => {
            file.write_all(&data)?;
        },
    }

    Ok(())
}