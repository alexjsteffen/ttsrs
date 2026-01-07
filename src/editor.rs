use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use edtui::{EditorEventHandler, EditorState, EditorTheme, EditorView, Lines};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap},
    Frame, Terminal,
};
use std::io::{self, Write};
use std::fs::File;
use std::path::PathBuf;

/// Editor application state
pub struct EditorApp {
    pub editor_state: EditorState,
    pub event_handler: EditorEventHandler,
    pub should_quit: bool,
    pub should_save: bool,
    pub show_help_modal: bool,
}

impl EditorApp {
    pub fn new(show_help: bool) -> Self {
        Self {
            editor_state: EditorState::new(Lines::from("")),
            event_handler: EditorEventHandler::default(),
            should_quit: false,
            should_save: false,
            show_help_modal: show_help,
        }
    }
}

/// Result of the editor session
pub enum EditorResult {
    /// User saved the file with this content
    Saved(String),
    /// User cancelled without saving
    Cancelled,
}

/// Run the text editor and return the result
/// `show_help_modal` indicates whether to show the help modal at startup
pub fn run_editor(show_help_modal: bool) -> Result<EditorResult> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = EditorApp::new(show_help_modal);
    let result = run_editor_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    match result {
        Ok(()) => {
            if app.should_save {
                // Get the text content from the editor
                let content = app.editor_state.lines.to_string();
                Ok(EditorResult::Saved(content))
            } else {
                Ok(EditorResult::Cancelled)
            }
        }
        Err(e) => Err(e),
    }
}

fn run_editor_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut EditorApp,
) -> Result<()>
where
    B::Error: Send + Sync + 'static,
{
    loop {
        terminal.draw(|f| editor_ui(f, app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }

            // If showing help modal, dismiss it with any key
            if app.show_help_modal {
                app.show_help_modal = false;
                continue;
            }

            // Handle special keys for quitting/saving
            match key.code {
                KeyCode::Esc => {
                    // Quit without saving
                    app.should_quit = true;
                    app.should_save = false;
                    return Ok(());
                }
                KeyCode::F(2) => {
                    // Save and exit
                    app.should_quit = true;
                    app.should_save = true;
                    return Ok(());
                }
                _ => {
                    // Pass event to edtui
                    app.event_handler.on_key_event(key, &mut app.editor_state);
                }
            }
        }
    }
}

fn editor_ui(f: &mut Frame, app: &mut EditorApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Min(10),    // Editor
            Constraint::Length(3),  // Help bar
        ])
        .split(f.area());

    // Title
    let title = Paragraph::new("Text Editor - Create your text file")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Editor area
    let editor_theme = EditorTheme::default()
        .block(Block::default().borders(Borders::ALL).title("Editor"))
        .base(Style::default().bg(Color::Black).fg(Color::White))
        .cursor_style(Style::default().bg(Color::White).fg(Color::Black))
        .selection_style(Style::default().bg(Color::Yellow).fg(Color::Black));

    EditorView::new(&mut app.editor_state)
        .theme(editor_theme)
        .wrap(true)
        .render(chunks[1], f.buffer_mut());

    // Help bar
    let help_text = "Vim keybindings: i=insert, Esc=normal mode | F2=Save and Exit | Esc (in normal mode)=Cancel";
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::ALL).title("Help"));
    f.render_widget(help, chunks[2]);

    // Help modal (shown once per session)
    if app.show_help_modal {
        render_help_modal(f);
    }
}

fn render_help_modal(f: &mut Frame) {
    let area = f.area();
    
    // Calculate centered popup area
    let popup_width = 60.min(area.width.saturating_sub(4));
    let popup_height = 16.min(area.height.saturating_sub(4));
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;
    
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear the background
    f.render_widget(Clear, popup_area);

    let help_text = r#"Welcome to the Text Editor!

This is a Vim-inspired text editor.

BASIC USAGE:
• Press 'i' to enter Insert mode (type text)
• Press 'Esc' to return to Normal mode
• Navigate with h/j/k/l or arrow keys
• Press 'w' to move forward by word
• Press 'b' to move backward by word

SAVING:
• Press F2 to save and exit
• Press Esc (in Normal mode) to cancel

Press any key to continue..."#;

    let modal = Paragraph::new(help_text)
        .style(Style::default().fg(Color::White).bg(Color::DarkGray))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title("Editor Instructions")
                .title_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        )
        .wrap(Wrap { trim: false });
    
    f.render_widget(modal, popup_area);
}

/// Prompt user for filename and save the content
pub fn save_file_with_prompt(content: &str) -> Result<Option<PathBuf>> {
    // Restore terminal for prompting
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;

    println!("\n📝 Save your text file");
    println!("Enter filename (without .txt extension), or press Enter to cancel:");
    print!("> ");
    io::stdout().flush()?;

    let mut filename = String::new();
    io::stdin().read_line(&mut filename)?;
    let filename = filename.trim();

    if filename.is_empty() {
        println!("Save cancelled.");
        return Ok(None);
    }

    // Add .txt extension
    let filename_with_ext = format!("{}.txt", filename);
    let path = PathBuf::from(&filename_with_ext);

    // Save the file
    let mut file = File::create(&path)
        .with_context(|| format!("Failed to create file: {}", filename_with_ext))?;
    file.write_all(content.as_bytes())
        .with_context(|| format!("Failed to write to file: {}", filename_with_ext))?;

    println!("✅ File saved as: {}", filename_with_ext);
    
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_app_new() {
        let app = EditorApp::new(true);
        assert!(app.show_help_modal);
        assert!(!app.should_quit);
        assert!(!app.should_save);
    }

    #[test]
    fn test_editor_app_new_without_help() {
        let app = EditorApp::new(false);
        assert!(!app.show_help_modal);
    }
}
