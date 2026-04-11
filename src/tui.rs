use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;

use crate::editor::{run_editor, save_file_with_prompt, EditorResult};
use crate::Args;

#[derive(Debug, Clone, PartialEq)]
enum FocusedField {
    Provider,
    InputFile,
    CreateTextFile,
    Voice,
    Model,
    Format,
    Speed,
    ApiKey,
    // Custom specific fields
    CustomEndpointUrl,
    // ElevenLabs specific fields
    ElevenLabsVoiceId,
    ElevenLabsModel,
    Stability,
    Similarity,
    Submit,
}

pub struct TuiApp {
    focused_field: FocusedField,
    fields: Vec<FocusedField>,
    input_file: String,
    provider: usize, // 0 = OpenAI, 1 = ElevenLabs, 2 = Custom
    voice: usize,
    model: String,
    format: usize,
    speed: String,
    api_key: String,
    custom_endpoint_url: String,
    elevenlabs_voice_id: String,
    elevenlabs_model: String,
    stability: String,
    similarity: String,
    editing_field: Option<String>,
    editor_help_shown: bool, // Track if editor help has been shown this session
}

impl Default for TuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiApp {
    pub fn new() -> Self {
        let fields = vec![
            FocusedField::Provider,
            FocusedField::InputFile,
            FocusedField::CreateTextFile,
            FocusedField::Voice,
            FocusedField::Model,
            FocusedField::Format,
            FocusedField::Speed,
            FocusedField::ApiKey,
            FocusedField::Submit,
        ];

        TuiApp {
            focused_field: FocusedField::Provider,
            fields,
            input_file: String::new(),
            provider: 0,
            voice: 5, // Default to Alloy
            model: "tts-1-hd".to_string(),
            format: 1, // Default to flac for OpenAI
            speed: "1.0".to_string(),
            api_key: String::new(),
            custom_endpoint_url: "http://localhost:1234/v1/audio/speech".to_string(),
            elevenlabs_voice_id: String::new(),
            elevenlabs_model: "eleven_turbo_v2_5".to_string(),
            stability: "0.5".to_string(),
            similarity: "0.75".to_string(),
            editing_field: None,
            editor_help_shown: false,
        }
    }

    fn update_fields(&mut self) {
        // Update available fields based on provider
        self.fields = if self.provider == 0 {
            // OpenAI (built-in provider)
            vec![
                FocusedField::Provider,
                FocusedField::InputFile,
                FocusedField::CreateTextFile,
                FocusedField::Voice,
                FocusedField::Model,
                FocusedField::Format,
                FocusedField::Speed,
                FocusedField::ApiKey,
                FocusedField::Submit,
            ]
        } else if self.provider == 1 {
            // ElevenLabs
            vec![
                FocusedField::Provider,
                FocusedField::InputFile,
                FocusedField::CreateTextFile,
                FocusedField::ElevenLabsVoiceId,
                FocusedField::ElevenLabsModel,
                FocusedField::Format,
                FocusedField::Stability,
                FocusedField::Similarity,
                FocusedField::ApiKey,
                FocusedField::Submit,
            ]
        } else {
            // Custom (OpenAI-compatible endpoint; also supports Speed)
            vec![
                FocusedField::Provider,
                FocusedField::InputFile,
                FocusedField::CreateTextFile,
                FocusedField::CustomEndpointUrl,
                FocusedField::Voice,
                FocusedField::Model,
                FocusedField::Format,
                FocusedField::Speed,
                FocusedField::ApiKey,
                FocusedField::Submit,
            ]
        };

        // Make sure focused field is still valid
        if !self.fields.contains(&self.focused_field) {
            self.focused_field = self.fields[0].clone();
        }
    }

    fn next_field(&mut self) {
        if let Some(current_idx) = self.fields.iter().position(|f| f == &self.focused_field) {
            let next_idx = (current_idx + 1) % self.fields.len();
            self.focused_field = self.fields[next_idx].clone();
        }
    }

    fn prev_field(&mut self) {
        if let Some(current_idx) = self.fields.iter().position(|f| f == &self.focused_field) {
            let prev_idx = if current_idx == 0 {
                self.fields.len() - 1
            } else {
                current_idx - 1
            };
            self.focused_field = self.fields[prev_idx].clone();
        }
    }

    fn handle_char_input(&mut self, c: char) {
        if let Some(ref mut text) = self.editing_field {
            text.push(c);
        }
    }

    fn handle_backspace(&mut self) {
        if let Some(ref mut text) = self.editing_field {
            text.pop();
        }
    }

    fn start_editing(&mut self) {
        let text = match self.focused_field {
            FocusedField::InputFile => &self.input_file,
            FocusedField::Model => &self.model,
            FocusedField::Speed => &self.speed,
            FocusedField::ApiKey => &self.api_key,
            FocusedField::CustomEndpointUrl => &self.custom_endpoint_url,
            FocusedField::ElevenLabsVoiceId => &self.elevenlabs_voice_id,
            FocusedField::ElevenLabsModel => &self.elevenlabs_model,
            FocusedField::Stability => &self.stability,
            FocusedField::Similarity => &self.similarity,
            _ => return,
        };
        self.editing_field = Some(text.clone());
    }

    fn finish_editing(&mut self) {
        if let Some(text) = self.editing_field.take() {
            match self.focused_field {
                FocusedField::InputFile => self.input_file = text,
                FocusedField::Model => self.model = text,
                FocusedField::Speed => self.speed = text,
                FocusedField::ApiKey => self.api_key = text,
                FocusedField::CustomEndpointUrl => self.custom_endpoint_url = text,
                FocusedField::ElevenLabsVoiceId => self.elevenlabs_voice_id = text,
                FocusedField::ElevenLabsModel => self.elevenlabs_model = text,
                FocusedField::Stability => self.stability = text,
                FocusedField::Similarity => self.similarity = text,
                _ => {}
            }
        }
    }

    fn handle_left_right(&mut self, is_right: bool) {
        match self.focused_field {
            FocusedField::Provider => {
                self.provider = if is_right {
                    (self.provider + 1) % 3
                } else if self.provider == 0 {
                    2
                } else {
                    self.provider - 1
                };
                self.update_fields();
            }
            FocusedField::Voice if self.provider == 0 || self.provider == 2 => {
                let voices = get_openai_voices();
                self.voice = if is_right {
                    (self.voice + 1) % voices.len()
                } else if self.voice == 0 {
                    voices.len() - 1
                } else {
                    self.voice - 1
                };
            }
            FocusedField::Format => {
                let formats = get_formats(self.provider);
                self.format = if is_right {
                    (self.format + 1) % formats.len()
                } else if self.format == 0 {
                    formats.len() - 1
                } else {
                    self.format - 1
                };
            }
            _ => {}
        }
    }

    pub fn to_args(&self) -> Result<Args> {
        let provider = if self.provider == 0 {
            "openai".to_string()
        } else if self.provider == 1 {
            "elevenlabs".to_string()
        } else {
            "custom".to_string()
        };

        let voice = if self.provider == 0 || self.provider == 2 {
            get_openai_voices()[self.voice]
                .split(" - ")
                .next()
                .unwrap()
                .to_lowercase()
        } else {
            String::new()
        };

        let format = get_formats(self.provider)[self.format].to_string();

        let speed: f32 = self.speed.parse().context("Invalid speed value")?;

        let elevenlabs_stability: f32 = if self.provider == 1 {
            self.stability.parse().context("Invalid stability value")?
        } else {
            0.5
        };

        let elevenlabs_similarity: f32 = if self.provider == 1 {
            self.similarity
                .parse()
                .context("Invalid similarity value")?
        } else {
            0.75
        };

        Ok(Args {
            cli: false,
            input_file: Some(self.input_file.clone()),
            model: self.model.clone(),
            voice,
            format,
            speed,
            apikey: Some(self.api_key.clone()),
            endpoint_url: if self.provider == 2 { Some(self.custom_endpoint_url.clone()) } else { None },
            provider,
            elevenlabs_voice_id: if self.provider == 1 {
                Some(self.elevenlabs_voice_id.clone())
            } else {
                None
            },
            elevenlabs_model: self.elevenlabs_model.clone(),
            elevenlabs_stability,
            elevenlabs_similarity,
            tui: false, // Already in TUI mode
        })
    }
}

fn get_openai_voices() -> Vec<&'static str> {
    vec![
        "Echo - Clear and professional, ideal for announcements",
        "Fable - Warm and engaging, perfect for storytelling",
        "Onyx - Deep and authoritative",
        "Nova - Young and energetic",
        "Shimmer - Soft and soothing",
        "Alloy - Versatile and well-balanced",
        "Ash - Clear and conversational",
        "Coral - Warm and friendly",
        "Sage - Calm and measured",
    ]
}

fn get_formats(provider: usize) -> Vec<&'static str> {
    if provider == 0 || provider == 2 {
        vec!["mp3", "flac", "wav", "pcm", "opus", "aac"]
    } else {
        vec![
            "mp3_44100_128",
            "mp3_44100_192",
            "pcm_16000",
            "pcm_22050",
            "pcm_24000",
            "pcm_44100",
        ]
    }
}

/// Result of the TUI app loop
enum TuiAppResult {
    /// User wants to submit the form
    Submit,
    /// User cancelled
    Cancelled,
    /// User wants to open the text editor
    OpenEditor,
}

pub fn run_tui() -> Result<Option<Args>> {
    // Create app state (needs to persist across editor invocations)
    let mut app = TuiApp::new();

    loop {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = run_app(&mut terminal, &mut app);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        match result {
            Ok(TuiAppResult::Submit) => {
                return Ok(Some(app.to_args()?));
            }
            Ok(TuiAppResult::Cancelled) => {
                return Ok(None);
            }
            Ok(TuiAppResult::OpenEditor) => {
                // Run the editor
                let show_help = !app.editor_help_shown;
                app.editor_help_shown = true;

                match run_editor(show_help)? {
                    EditorResult::Saved(content) => {
                        // Prompt for filename and save
                        if let Some(path) = save_file_with_prompt(&content)? {
                            // Set the input file to the saved file
                            app.input_file = path.to_string_lossy().to_string();
                            println!("\nPress Enter to continue...");
                            let mut input = String::new();
                            std::io::stdin().read_line(&mut input)?;
                        }
                    }
                    EditorResult::Cancelled => {
                        // Just continue back to the TUI
                    }
                }
                // Continue the loop to go back to the TUI
            }
            Err(e) => return Err(e),
        }
    }
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut TuiApp,
) -> Result<TuiAppResult>
where
    B::Error: Send + Sync + 'static,
{
    loop {
        terminal.draw(|f| ui(f, app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }

            // If editing, handle input differently
            if app.editing_field.is_some() {
                match key.code {
                    KeyCode::Enter => {
                        app.finish_editing();
                    }
                    KeyCode::Esc => {
                        app.editing_field = None;
                    }
                    KeyCode::Char(c) => {
                        app.handle_char_input(c);
                    }
                    KeyCode::Backspace => {
                        app.handle_backspace();
                    }
                    _ => {}
                }
            } else {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        return Ok(TuiAppResult::Cancelled);
                    }
                    KeyCode::Down | KeyCode::Tab => {
                        app.next_field();
                    }
                    KeyCode::Up | KeyCode::BackTab => {
                        app.prev_field();
                    }
                    KeyCode::Left => {
                        app.handle_left_right(false);
                    }
                    KeyCode::Right => {
                        app.handle_left_right(true);
                    }
                    KeyCode::Enter => {
                        if app.focused_field == FocusedField::Submit {
                            return Ok(TuiAppResult::Submit);
                        } else if app.focused_field == FocusedField::CreateTextFile {
                            return Ok(TuiAppResult::OpenEditor);
                        } else {
                            app.start_editing();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Returns the highlighted style for the focused field or the normal color for unfocused fields.
fn field_style(is_focused: bool, normal_color: Color) -> Style {
    if is_focused {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(normal_color)
    }
}

/// Returns the display text for a text-editable field, taking editing state into account.
fn editing_text<'a>(app: &'a TuiApp, field: &FocusedField, stored_value: &'a str) -> String {
    if let Some(ref editing) = app.editing_field {
        if &app.focused_field == field {
            return editing.clone();
        }
    }
    stored_value.to_string()
}

fn ui(f: &mut Frame, app: &mut TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(5),
            ]
            .as_ref(),
        )
        .split(f.area());

    // Title
    let title = Paragraph::new("TTSRS - Text-to-Speech TUI")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Main form
    let form_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(1),
            ]
            .as_ref(),
        )
        .split(chunks[1]);

    let mut chunk_idx = 0;

    // Provider selector
    if app.fields.contains(&FocusedField::Provider) {
        let provider_text = if app.provider == 0 {
            "OpenAI"
        } else if app.provider == 1 {
            "ElevenLabs"
        } else {
            "Custom"
        };
        let style = field_style(app.focused_field == FocusedField::Provider, Color::Cyan);
        let provider = Paragraph::new(format!("{}  [Options: OpenAI, ElevenLabs, Custom]", provider_text))
            .style(style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("TTS Provider (Use ← → to switch)"),
            );
        f.render_widget(provider, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Input file
    if app.fields.contains(&FocusedField::InputFile) {
        let text = editing_text(app, &FocusedField::InputFile, &app.input_file);
        let display_text = if text.is_empty() {
            "[Enter path to text file]".to_string()
        } else {
            text
        };
        let style = field_style(app.focused_field == FocusedField::InputFile, Color::White);
        let input_file = Paragraph::new(display_text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Input Text File (Press Enter to type path)"),
        );
        f.render_widget(input_file, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Create Text File option (opens internal text editor)
    if app.fields.contains(&FocusedField::CreateTextFile) {
        let style = field_style(
            app.focused_field == FocusedField::CreateTextFile,
            Color::LightGreen,
        );
        let create_file = Paragraph::new("[ Open Text Editor ]").style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Create Text File (Press Enter to open editor)"),
        );
        f.render_widget(create_file, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Custom Endpoint URL
    if app.fields.contains(&FocusedField::CustomEndpointUrl) {
        let text = editing_text(app, &FocusedField::CustomEndpointUrl, &app.custom_endpoint_url);
        let style = field_style(app.focused_field == FocusedField::CustomEndpointUrl, Color::White);
        let custom_endpoint_url = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Custom Endpoint URL (Press Enter to edit)"),
        );
        f.render_widget(custom_endpoint_url, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Voice selector (OpenAI only)
    if app.fields.contains(&FocusedField::Voice) {
        let voices = get_openai_voices();
        let voice_text = voices[app.voice];
        let voice_count = voices.len();
        let style = field_style(app.focused_field == FocusedField::Voice, Color::Magenta);
        let voice = Paragraph::new(format!(
            "{}  [{}/{} voices - Use ← → to browse]",
            voice_text,
            app.voice + 1,
            voice_count
        ))
        .style(style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Voice Selection"),
        );
        f.render_widget(voice, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Model
    if app.fields.contains(&FocusedField::Model) {
        let text = editing_text(app, &FocusedField::Model, &app.model);
        let style = field_style(app.focused_field == FocusedField::Model, Color::White);
        let model = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("TTS Model (Press Enter to edit, e.g., tts-1, tts-1-hd)"),
        );
        f.render_widget(model, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Format
    if app.fields.contains(&FocusedField::Format) {
        let formats = get_formats(app.provider);
        let format_text = formats[app.format];
        let format_count = formats.len();
        let style = field_style(app.focused_field == FocusedField::Format, Color::Blue);
        let format_opts = if app.provider == 0 || app.provider == 2 {
            "mp3, flac, wav, pcm, opus, aac"
        } else {
            "mp3_44100_128/192, pcm_16000/22050/24000/44100"
        };
        let format = Paragraph::new(format!(
            "{}  [{}/{}] Options: {}",
            format_text,
            app.format + 1,
            format_count,
            format_opts
        ))
        .style(style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Audio Output Format (Use ← → to select)"),
        );
        f.render_widget(format, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Speed (OpenAI only)
    if app.fields.contains(&FocusedField::Speed) {
        let text = editing_text(app, &FocusedField::Speed, &app.speed);
        let style = field_style(app.focused_field == FocusedField::Speed, Color::White);
        let speed = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Speaking Speed (Enter number 0.25-4.0, default 1.0)"),
        );
        f.render_widget(speed, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // ElevenLabs Voice ID
    if app.fields.contains(&FocusedField::ElevenLabsVoiceId) {
        let text = editing_text(
            app,
            &FocusedField::ElevenLabsVoiceId,
            &app.elevenlabs_voice_id,
        );
        let display_text = if text.is_empty() {
            "[Enter ElevenLabs voice ID from voice library]".to_string()
        } else {
            text
        };
        let style = field_style(
            app.focused_field == FocusedField::ElevenLabsVoiceId,
            Color::White,
        );
        let voice_id = Paragraph::new(display_text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("ElevenLabs Voice ID (Press Enter, e.g., 21m00Tcm4TlvDq8ikWAM)"),
        );
        f.render_widget(voice_id, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // ElevenLabs Model
    if app.fields.contains(&FocusedField::ElevenLabsModel) {
        let text = editing_text(app, &FocusedField::ElevenLabsModel, &app.elevenlabs_model);
        let style = field_style(
            app.focused_field == FocusedField::ElevenLabsModel,
            Color::White,
        );
        let el_model = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("ElevenLabs Model ID (Press Enter to edit, default: eleven_turbo_v2_5)"),
        );
        f.render_widget(el_model, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Stability (ElevenLabs only)
    if app.fields.contains(&FocusedField::Stability) {
        let text = editing_text(app, &FocusedField::Stability, &app.stability);
        let style = field_style(app.focused_field == FocusedField::Stability, Color::White);
        let stability = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Voice Stability (Enter 0.0-1.0, controls consistency, default 0.5)"),
        );
        f.render_widget(stability, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Similarity (ElevenLabs only)
    if app.fields.contains(&FocusedField::Similarity) {
        let text = editing_text(app, &FocusedField::Similarity, &app.similarity);
        let style = field_style(app.focused_field == FocusedField::Similarity, Color::White);
        let similarity = Paragraph::new(text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Similarity Boost (Enter 0.0-1.0, voice likeness, default 0.75)"),
        );
        f.render_widget(similarity, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // API Key
    if app.fields.contains(&FocusedField::ApiKey) {
        let masked_text = if let Some(ref editing) = app.editing_field {
            if app.focused_field == FocusedField::ApiKey {
                "*".repeat(editing.len())
            } else {
                "*".repeat(app.api_key.len())
            }
        } else {
            "*".repeat(app.api_key.len())
        };
        let display_text = if masked_text.is_empty() {
            if app.provider == 2 {
                "[Press Enter to enter API key - Optional for Custom endpoints]".to_string()
            } else {
                "[Press Enter to enter your API key securely]".to_string()
            }
        } else {
            masked_text
        };
        let style = field_style(app.focused_field == FocusedField::ApiKey, Color::Red);
        let title_text = if app.provider == 2 {
            "API Key (Optional for some custom endpoints)"
        } else {
            "API Key (Required - will be saved to config file)"
        };
        let api_key = Paragraph::new(display_text).style(style).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title_text),
        );
        f.render_widget(api_key, form_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    // Submit button
    if app.fields.contains(&FocusedField::Submit) {
        let style = field_style(app.focused_field == FocusedField::Submit, Color::Green);
        let submit = Paragraph::new("[ Generate Audio ]")
            .style(style)
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(submit, form_chunks[chunk_idx]);
    }

    // Help text - expanded to two lines for more context
    let help_text = if app.editing_field.is_some() {
        "Editing Mode: Type your input | Enter to confirm | Esc to cancel\nYellow highlight = current field being edited"
    } else {
        "Navigation: ↑↓ or Tab to move | ← → to cycle options | Enter to edit/submit | q or Esc to quit\nLightGreen=Create Text File (opens editor) | Colors: Cyan=provider, Magenta=voice, Blue=format, Red=API key"
    };
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Help & Navigation"),
        );
    f.render_widget(help, chunks[2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_app_new() {
        let app = TuiApp::new();
        assert_eq!(app.provider, 0);
        assert_eq!(app.voice, 5);
        assert_eq!(app.model, "tts-1-hd");
        assert_eq!(app.format, 1);
        assert_eq!(app.speed, "1.0");
        assert_eq!(app.focused_field, FocusedField::Provider);
    }

    #[test]
    fn test_provider_switch() {
        let mut app = TuiApp::new();
        assert_eq!(app.provider, 0); // OpenAI
                                     // Verify OpenAI specific fields are present
        assert!(app.fields.contains(&FocusedField::Provider));
        assert!(app.fields.contains(&FocusedField::InputFile));
        assert!(app.fields.contains(&FocusedField::CreateTextFile));
        assert!(app.fields.contains(&FocusedField::Voice));
        assert!(app.fields.contains(&FocusedField::Submit));

        // Switch to ElevenLabs
        app.provider = 1;
        app.update_fields();
        // Verify ElevenLabs specific fields are present
        assert!(app.fields.contains(&FocusedField::CreateTextFile));
        assert!(app.fields.contains(&FocusedField::ElevenLabsVoiceId));
        assert!(!app.fields.contains(&FocusedField::Voice));
    }

    #[test]
    fn test_next_field_navigation() {
        let mut app = TuiApp::new();
        let initial_field = app.focused_field.clone();
        app.next_field();
        assert_ne!(app.focused_field, initial_field);
    }

    #[test]
    fn test_prev_field_navigation() {
        let mut app = TuiApp::new();
        app.next_field();
        app.prev_field();
        assert_eq!(app.focused_field, FocusedField::Provider);
    }

    #[test]
    fn test_char_input() {
        let mut app = TuiApp::new();
        app.focused_field = FocusedField::InputFile;
        app.start_editing();
        app.handle_char_input('t');
        app.handle_char_input('e');
        app.handle_char_input('s');
        app.handle_char_input('t');
        app.finish_editing();
        assert_eq!(app.input_file, "test");
    }

    #[test]
    fn test_backspace() {
        let mut app = TuiApp::new();
        app.focused_field = FocusedField::InputFile;
        app.start_editing();
        app.handle_char_input('a');
        app.handle_char_input('b');
        app.handle_backspace();
        app.finish_editing();
        assert_eq!(app.input_file, "a");
    }

    #[test]
    fn test_to_args_openai() {
        let mut app = TuiApp::new();
        app.input_file = "test.txt".to_string();
        app.api_key = "test_key".to_string();

        let args = app.to_args().unwrap();
        assert_eq!(args.input_file, Some("test.txt".to_string()));
        assert_eq!(args.provider, "openai");
        assert_eq!(args.voice, "alloy");
        assert_eq!(args.apikey, Some("test_key".to_string()));
    }

    #[test]
    fn test_to_args_elevenlabs() {
        let mut app = TuiApp::new();
        app.provider = 1;
        app.input_file = "test.txt".to_string();
        app.api_key = "test_key".to_string();
        app.elevenlabs_voice_id = "voice123".to_string();

        let args = app.to_args().unwrap();
        assert_eq!(args.provider, "elevenlabs");
        assert_eq!(args.elevenlabs_voice_id, Some("voice123".to_string()));
    }

    #[test]
    fn test_get_openai_voices() {
        let voices = get_openai_voices();
        assert_eq!(voices.len(), 9);
        assert!(voices[5].contains("Alloy"));
    }

    #[test]
    fn test_get_formats() {
        let openai_formats = get_formats(0);
        assert_eq!(openai_formats.len(), 6);
        assert!(openai_formats.contains(&"flac"));

        let elevenlabs_formats = get_formats(1);
        assert_eq!(elevenlabs_formats.len(), 6);
        assert!(elevenlabs_formats.contains(&"mp3_44100_128"));
    }

    #[test]
    fn test_handle_left_right_provider() {
        let mut app = TuiApp::new();
        app.focused_field = FocusedField::Provider;
        assert_eq!(app.provider, 0);

        app.handle_left_right(true); // Right
        assert_eq!(app.provider, 1);

        app.handle_left_right(true); // Right again
        assert_eq!(app.provider, 2);

        app.handle_left_right(true); // Right again (should wrap)
        assert_eq!(app.provider, 0);

        app.handle_left_right(false); // Left
        assert_eq!(app.provider, 2);
    }

    #[test]
    fn test_handle_left_right_voice() {
        let mut app = TuiApp::new();
        app.focused_field = FocusedField::Voice;
        let initial_voice = app.voice;

        app.handle_left_right(true); // Right
        assert_ne!(app.voice, initial_voice);

        app.handle_left_right(false); // Left
        assert_eq!(app.voice, initial_voice);
    }

    #[test]
    fn test_create_text_file_field_exists() {
        let app = TuiApp::new();
        assert!(app.fields.contains(&FocusedField::CreateTextFile));
        assert!(!app.editor_help_shown);
    }

    #[test]
    fn test_editor_help_shown_flag() {
        let mut app = TuiApp::new();
        assert!(!app.editor_help_shown);
        app.editor_help_shown = true;
        assert!(app.editor_help_shown);
    }
}
