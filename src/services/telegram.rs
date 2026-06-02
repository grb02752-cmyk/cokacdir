use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use teloxide::prelude::*;
use teloxide::types::{ParseMode, UpdateKind};
use tokio::sync::Mutex;

use crate::services::claude::{self, CancelToken, StreamMessage, DEFAULT_ALLOWED_TOOLS};
use crate::services::codex;
use crate::services::gemini;
use crate::services::opencode;
use crate::ui::ai_screen::{self, HistoryItem, HistoryType, SessionData};

/// Global debug log flag for Telegram API calls
static TG_DEBUG: AtomicBool = AtomicBool::new(false);
const TELEGRAM_API_TIMEOUT_SECS: u64 = 30;

/// Log Telegram API call result to ~/.cokacdir/debug/ file
fn tg_debug<T, E: std::fmt::Display>(name: &str, result: &Result<T, E>) {
    if !TG_DEBUG.load(Ordering::Relaxed) {
        return;
    }
    let Some(debug_dir) = dirs::home_dir().map(|h| h.join(".cokacdir").join("debug")) else {
        return;
    };
    let _ = fs::create_dir_all(&debug_dir);
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let log_path = debug_dir.join(format!("{}.log", date));
    let ts = chrono::Local::now().format("%H:%M:%S%.3f");
    let status = match result {
        Ok(_) => "✓".to_string(),
        Err(e) => format!("✗ {e}"),
    };
    let line = format!("[{ts}] {name}: {status}\n");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

/// Wrap a Telegram API call to log its result in debug mode.
///
/// Two forms:
/// - `tg!(name, future_result)` — log only.
/// - `tg!(name, state, chat_id, future_result)` — log + honor any
///   server-mandated `RetryAfter` by pushing `api_timestamps[chat_id]`
///   forward via `honor_telegram_retry_after`. Use this form for calls in
///   high-frequency polling loops (e.g. spinner edits) so a single Flood
///   Control hit propagates to subsequent `shared_rate_limit_wait` calls
///   instead of being ignored.
macro_rules! tg {
    ($name:expr, $fut:expr) => {{
        let r = $fut;
        tg_debug($name, &r);
        r
    }};
    ($name:expr, $state:expr, $chat:expr, $fut:expr) => {{
        let r = $fut;
        tg_debug($name, &r);
        honor_telegram_retry_after($state, $chat, &r).await;
        record_telegram_request_result($state, $chat, $name, &r).await;
        r
    }};
}

/// Wrap a Telegram API call with a bounded timeout so polling/final-flush
/// paths cannot hang forever on a stuck network request.
///
/// Two forms:
/// - `tg_timeout!(name, future_result)` — timeout + log only.
/// - `tg_timeout!(name, state, chat_id, future_result)` — timeout + log +
///   honor any server-mandated `RetryAfter`.
macro_rules! tg_timeout {
    ($name:expr, $fut:expr) => {{
        let r = match tokio::time::timeout(
            std::time::Duration::from_secs(TELEGRAM_API_TIMEOUT_SECS),
            async { $fut },
        )
        .await
        {
            Ok(r) => r,
            Err(_) => {
                ai_trace(&format!(
                    "[TG_TIMEOUT] {} timed out after {}s",
                    $name, TELEGRAM_API_TIMEOUT_SECS
                ));
                Err(teloxide::RequestError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "telegram request timed out after {}s: {}",
                        TELEGRAM_API_TIMEOUT_SECS, $name
                    ),
                )))
            }
        };
        tg_debug($name, &r);
        r
    }};
    ($name:expr, $state:expr, $chat:expr, $fut:expr) => {{
        let r = match tokio::time::timeout(
            std::time::Duration::from_secs(TELEGRAM_API_TIMEOUT_SECS),
            async { $fut },
        )
        .await
        {
            Ok(r) => r,
            Err(_) => {
                ai_trace(&format!(
                    "[TG_TIMEOUT] {} timed out after {}s",
                    $name, TELEGRAM_API_TIMEOUT_SECS
                ));
                Err(teloxide::RequestError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "telegram request timed out after {}s: {}",
                        TELEGRAM_API_TIMEOUT_SECS, $name
                    ),
                )))
            }
        };
        tg_debug($name, &r);
        honor_telegram_retry_after($state, $chat, &r).await;
        record_telegram_request_result($state, $chat, $name, &r).await;
        r
    }};
}

// ── Group Chat Shared Log ──
// All bots in the same group chat write to a shared JSONL file so that
// each bot can see what other bots said, solving the cross-bot context problem.

/// A single entry in the group chat shared log (JSONL format).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GroupChatLogEntry {
    /// ISO-8601 timestamp
    pub ts: String,
    /// Bot username that handled this message (without @)
    pub bot: String,
    /// Bot display name (first_name from Telegram API)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bot_display_name: Option<String>,
    /// "user" or "assistant" (or "system" for clear markers)
    pub role: String,
    /// Display name of the sender (for user messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Message text content
    pub text: String,
    /// If true, this entry is a clear marker — all previous entries from this bot should be ignored
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clear: bool,
}

/// A parsed entry from the raw_payload text format used in assistant log entries.
/// Tag name can be any ASCII alphabetic string (e.g. "Text", "ToolUse", "ToolResult", ...).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RawPayloadEntry {
    pub tag: String,
    pub content: String,
}

/// Check if a line starts with a `[TagName] ` pattern where TagName is ASCII alphabetic.
/// Returns Some("[TagName]") if matched, None otherwise.
fn detect_raw_payload_tag(line: &str) -> Option<&str> {
    let rest = line.strip_prefix('[')?;
    let end = rest.find(']')?;
    let tag = &rest[..end];
    if tag.is_empty() || !tag.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    Some(&line[..end + 2]) // "[TagName]"
}

/// Parse raw_payload flat text into structured entries.
/// Each section starts with `[TagName] ...` (tag is ASCII alphabetic) and continues
/// until the next tag or end of string. Multi-line content is preserved.
pub fn parse_raw_payload(text: &str) -> Vec<RawPayloadEntry> {
    let mut entries: Vec<RawPayloadEntry> = Vec::new();
    let mut cur_tag: Option<String> = None;
    let mut cur_content = String::new();

    for line in text.lines() {
        if let Some(bracket_tag) = detect_raw_payload_tag(line) {
            // Save previous section
            if let Some(tag) = cur_tag.take() {
                entries.push(RawPayloadEntry {
                    tag,
                    content: std::mem::take(&mut cur_content),
                });
            }
            // Extract tag name (strip [ and ])
            cur_tag = Some(bracket_tag[1..bracket_tag.len() - 1].to_string());
            let after = &line[bracket_tag.len()..];
            cur_content = after.strip_prefix(' ').unwrap_or(after).to_string();
        } else if cur_tag.is_some() {
            cur_content.push('\n');
            cur_content.push_str(line);
        }
    }
    if let Some(tag) = cur_tag {
        entries.push(RawPayloadEntry {
            tag,
            content: cur_content,
        });
    }
    entries
}

/// Format structured entries back to raw_payload flat text.
pub fn format_raw_payload(entries: &[RawPayloadEntry]) -> String {
    let mut out = String::new();
    for entry in entries {
        out.push_str(&format!("[{}] {}\n", entry.tag, entry.content));
    }
    out
}

/// Parse payload text with auto-detection: tries JSON array (new format) first,
/// falls back to legacy flat text parsing.
pub fn parse_payload_auto(text: &str) -> Vec<RawPayloadEntry> {
    if let Ok(entries) = serde_json::from_str::<Vec<RawPayloadEntry>>(text) {
        return entries;
    }
    parse_raw_payload(text)
}

/// Maximum content length before truncation and offloading to a separate file.
const PAYLOAD_TRUNCATE_LIMIT: usize = 500;

/// Save long content to ~/.cokacdir/values/<unique>.txt and return the file path.
/// Returns None if saving fails.
fn save_payload_value(content: &str) -> Option<String> {
    let dir = dirs::home_dir()?.join(".cokacdir").join("values");
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let ts = chrono::Local::now().format("%Y%m%d%H%M%S");
    use rand::Rng;
    let rand_suffix: String = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(6)
        .map(|b| (b as char).to_ascii_lowercase())
        .collect();
    let filename = format!("{}_{}.txt", ts, rand_suffix);
    let path = dir.join(&filename);
    if fs::write(&path, content).is_err() {
        return None;
    }
    Some(path.display().to_string())
}

/// Truncate payload entries whose content exceeds the limit.
/// Long content is saved to ~/.cokacdir/values/ and replaced with a truncated preview + file path.
fn truncate_payload_entries(entries: &[RawPayloadEntry]) -> Vec<RawPayloadEntry> {
    entries
        .iter()
        .map(|entry| {
            if entry.content.chars().count() <= PAYLOAD_TRUNCATE_LIMIT {
                entry.clone()
            } else {
                let preview: String = entry.content.chars().take(PAYLOAD_TRUNCATE_LIMIT).collect();
                let truncated_content = if let Some(path) = save_payload_value(&entry.content) {
                    format!("{}...\n(truncated, full content: {})", preview, path)
                } else {
                    format!("{}...\n(truncated, failed to save full content)", preview)
                };
                RawPayloadEntry {
                    tag: entry.tag.clone(),
                    content: truncated_content,
                }
            }
        })
        .collect()
}

/// Serialize payload entries to JSON string (new format).
/// Entries with content exceeding the limit are truncated and offloaded to separate files.
pub fn serialize_payload(entries: &[RawPayloadEntry]) -> String {
    let processed = truncate_payload_entries(entries);
    serde_json::to_string(&processed).unwrap_or_default()
}

/// RAII guard for per-chat exclusive lock.
/// Ensures only one AI request processes at a time within the same chat.
/// Lock is automatically released when the guard is dropped.
struct GroupChatLock {
    _file: std::fs::File,
}

/// Acquire exclusive file lock for a group chat (async, non-blocking).
/// Returns None for private chats (chat_id >= 0) or if lock file cannot be created.
/// For group chats, polls with sleep until the lock is acquired.
async fn acquire_group_chat_lock(chat_id: i64) -> Option<GroupChatLock> {
    use fs2::FileExt;
    if chat_id >= 0 {
        return None;
    }
    let dir = dirs::home_dir()?.join(".cokacdir").join("chat_locks");
    let _ = std::fs::create_dir_all(&dir);
    let lock_path = dir.join(format!("{}.lock", chat_id));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .ok()?;
    let mut waited = false;
    loop {
        match file.try_lock_exclusive() {
            Ok(_) => {
                if waited {
                    msg_debug(&format!(
                        "[chat_lock] chat_id={} lock acquired after waiting",
                        chat_id
                    ));
                } else {
                    msg_debug(&format!(
                        "[chat_lock] chat_id={} lock acquired immediately",
                        chat_id
                    ));
                }
                return Some(GroupChatLock { _file: file });
            }
            Err(_) => {
                if !waited {
                    msg_debug(&format!(
                        "[chat_lock] chat_id={} lock contended, waiting...",
                        chat_id
                    ));
                    waited = true;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
        }
    }
}

/// Return the directory for group chat logs: ~/.cokacdir/group_chat/
fn group_chat_log_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".cokacdir").join("group_chat"))
}

/// Return the JSONL file path for a specific group chat.
fn group_chat_log_path(chat_id: i64) -> Option<std::path::PathBuf> {
    group_chat_log_dir().map(|d| d.join(format!("{}.jsonl", chat_id)))
}

/// Append an entry to the group chat shared log.
/// Uses a separate lock file for synchronization to avoid Windows LockFileEx
/// conflicts when locking and writing on the same file handle.
fn append_group_chat_log(chat_id: i64, entry: &GroupChatLogEntry) {
    use fs2::FileExt;

    let Some(path) = group_chat_log_path(chat_id) else {
        return;
    };
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return;
        }
    }

    // Serialize before acquiring lock to minimize lock hold time
    let Ok(json) = serde_json::to_string(entry) else {
        return;
    };

    // Skip entries containing --read_chat_log to prevent recursive snowball growth:
    // each --read_chat_log result embeds the entire log, causing exponential JSONL inflation.
    if json.contains("--read_chat_log") {
        msg_debug(&format!(
            "[append_group_chat_log] SKIPPED: line contains --read_chat_log (len={})",
            json.len()
        ));
        return;
    }

    let line = format!("{}\n", json);

    // Lock via a dedicated lock file (not the data file itself)
    // On Windows, LockFileEx is mandatory and can conflict with WriteFile on the same handle
    let lock_path = path.with_extension("jsonl.lock");
    let Ok(lock_file) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
    else {
        return;
    };
    if lock_file.lock_exclusive().is_err() {
        return;
    }

    // Write to data file without locking it
    if let Ok(mut data_file) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(&path)
    {
        use std::io::Write;
        let _ = data_file.write_all(line.as_bytes());
        let _ = data_file.sync_data();
    }

    let _ = lock_file.unlock();
}

/// Read entries from the group chat log within a specific line range (1-based).
/// If `filter_bot` is Some, only include entries from that bot.
pub fn read_group_chat_log_range(
    chat_id: i64,
    range_start: usize,
    range_end: Option<usize>,
    filter_bot: Option<&str>,
) -> Vec<(usize, GroupChatLogEntry)> {
    use fs2::FileExt;

    let filter_bot_owned: Option<String> =
        filter_bot.map(|b| b.strip_prefix('@').unwrap_or(b).to_lowercase());
    let filter_bot = filter_bot_owned.as_deref();

    let Some(path) = group_chat_log_path(chat_id) else {
        return Vec::new();
    };

    // Use the same dedicated lock file as append_group_chat_log for coordination
    let lock_path = path.with_extension("jsonl.lock");
    let Ok(lock_file) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
    else {
        return Vec::new();
    };
    if lock_file.lock_shared().is_err() {
        return Vec::new();
    }

    let Ok(file) = fs::File::open(&path) else {
        let _ = lock_file.unlock();
        return Vec::new();
    };

    let reader = std::io::BufReader::new(&file);
    use std::io::BufRead;

    // First pass: collect all entries and find the last clear marker per bot
    let all_entries: Vec<(usize, GroupChatLogEntry)> = reader
        .lines()
        .filter_map(|line| line.ok())
        .enumerate()
        .filter_map(|(i, line)| {
            let line_num = i + 1; // 1-based
            serde_json::from_str::<GroupChatLogEntry>(&line)
                .ok()
                .map(|entry| (line_num, entry))
        })
        .collect();

    // Build a map of bot -> last clear marker line number
    let mut last_clear: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (line_num, entry) in &all_entries {
        if entry.clear {
            last_clear.insert(entry.bot.clone(), *line_num);
        }
    }

    // Second pass: filter entries, skipping those before the clear marker for each bot
    let entries: Vec<(usize, GroupChatLogEntry)> = all_entries
        .into_iter()
        .filter(|(line_num, entry)| {
            // Skip clear marker entries themselves
            if entry.clear {
                return false;
            }
            // Skip entries from a bot that are before its last clear marker
            if let Some(&clear_line) = last_clear.get(&entry.bot) {
                if *line_num <= clear_line {
                    return false;
                }
            }
            let in_range =
                *line_num >= range_start && range_end.map_or(true, |end| *line_num <= end);
            let bot_match = filter_bot.map_or(true, |bot| entry.bot == bot);
            in_range && bot_match
        })
        .collect();

    let _ = lock_file.unlock();
    entries
}

/// Collect all bots that have ever appeared in a group chat log, with their
/// most recent display name.  Scans the entire file including entries before
/// clear markers (and the markers themselves) so that bots whose history was
/// cleared are still listed.
///
/// Returns a Vec of (bot_username, Option<display_name>) sorted by username.
pub fn collect_group_chat_bots(chat_id: i64) -> Vec<(String, Option<String>)> {
    use fs2::FileExt;
    use std::io::BufRead;

    let Some(path) = group_chat_log_path(chat_id) else {
        return Vec::new();
    };

    let lock_path = path.with_extension("jsonl.lock");
    let Ok(lock_file) = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
    else {
        return Vec::new();
    };
    if lock_file.lock_shared().is_err() {
        return Vec::new();
    }

    let Ok(file) = fs::File::open(&path) else {
        let _ = lock_file.unlock();
        return Vec::new();
    };

    // Scan forward — later entries overwrite earlier ones, keeping the latest display name
    let mut bots: std::collections::HashMap<String, Option<String>> =
        std::collections::HashMap::new();
    for line in std::io::BufReader::new(&file).lines().flatten() {
        if let Ok(entry) = serde_json::from_str::<GroupChatLogEntry>(&line) {
            if !entry.bot.is_empty() {
                bots.insert(entry.bot.clone(), entry.bot_display_name);
            }
        }
    }

    let _ = lock_file.unlock();

    let mut result: Vec<(String, Option<String>)> = bots.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Per-chat session state
#[derive(Clone)]
struct ChatSession {
    session_id: Option<String>,
    current_path: Option<String>,
    history: Vec<HistoryItem>,
    /// File upload records not yet sent to Claude AI.
    /// Drained and prepended to the next user prompt so Claude knows about uploaded files.
    pending_uploads: Vec<String>,
}

/// Bot-level settings persisted to disk
#[derive(Clone)]
struct BotSettings {
    allowed_tools: HashMap<String, Vec<String>>,
    /// chat_id (string) → last working directory path
    last_sessions: HashMap<String, String>,
    /// Telegram user ID of the registered owner (imprinting auth)
    owner_user_id: Option<u64>,
    /// chat_id (string) → true if group chat is public (non-owner users allowed)
    as_public_for_group_chat: HashMap<String, bool>,
    /// chat_id (string) → model name (e.g. "claude", "claude:claude-sonnet-4-6", "codex:gpt-5.4")
    models: HashMap<String, String>,
    /// Debug logging toggle
    debug: bool,
    /// chat_id (string) → true if silent mode enabled
    silent: HashMap<String, bool>,
    /// chat_id (string) → true if direct mode enabled (group chat without ; prefix)
    direct: HashMap<String, bool>,
    /// chat_id (string) → number of recent group chat log entries to embed in system prompt (default 12)
    context: HashMap<String, usize>,
    /// chat_id (string) → system instruction for AI
    instructions: HashMap<String, String>,
    /// chat_id (string) → true if queue mode enabled (queue messages while AI is busy)
    queue: HashMap<String, bool>,
    /// Bot's Telegram username (stored at startup via get_me)
    username: String,
    /// Bot's display name (first_name from Telegram API, stored at startup via get_me)
    display_name: String,
    /// Compact startup greeting (show single line instead of full marketing message)
    greeting: bool,
    /// chat_id (string) → true if --chrome flag should be passed to Claude CLI
    use_chrome: HashMap<String, bool>,
    /// chat_id (string) → message to send when AI processing completes
    end_hook: HashMap<String, String>,
    /// chat_id (string) -> codex reasoning effort
    effort: HashMap<String, String>,
    /// chat_id (string) -> true if Codex should run with fast service tier
    codex_fast: HashMap<String, bool>,
    /// chat_id (string) -> Claude effort level
    claude_effort: HashMap<String, String>,
}

impl Default for BotSettings {
    fn default() -> Self {
        Self {
            allowed_tools: HashMap::new(),
            last_sessions: HashMap::new(),
            owner_user_id: None,
            as_public_for_group_chat: HashMap::new(),
            models: HashMap::new(),
            debug: false,
            silent: HashMap::new(),
            direct: HashMap::new(),
            context: HashMap::new(),
            instructions: HashMap::new(),
            queue: HashMap::new(),
            username: String::new(),
            display_name: String::new(),
            greeting: false,
            use_chrome: HashMap::new(),
            end_hook: HashMap::new(),
            effort: HashMap::new(),
            codex_fast: HashMap::new(),
            claude_effort: HashMap::new(),
        }
    }
}

/// Get allowed tools for a specific chat_id.
/// Returns the chat-specific list if configured, otherwise DEFAULT_ALLOWED_TOOLS.
fn get_allowed_tools(settings: &BotSettings, chat_id: ChatId) -> Vec<String> {
    let key = chat_id.0.to_string();
    settings
        .allowed_tools
        .get(&key)
        .cloned()
        .unwrap_or_else(|| {
            DEFAULT_ALLOWED_TOOLS
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
}

/// Get the configured model for a specific chat_id, if any.
/// Migrates legacy bare names (e.g. "sonnet") to "claude:" prefixed format.
fn get_model(settings: &BotSettings, chat_id: ChatId) -> Option<String> {
    let key = chat_id.0.to_string();
    settings.models.get(&key).map(|m| match m.as_str() {
        "sonnet" | "opus" | "haiku" | "sonnet[1m]" | "opus[1m]" | "haiku[1m]" => {
            format!("claude:{}", m)
        }
        _ => m.clone(),
    })
}

/// Check if silent mode is enabled for a chat (default: ON)
fn is_silent(settings: &BotSettings, chat_id: ChatId) -> bool {
    settings
        .silent
        .get(&chat_id.0.to_string())
        .copied()
        .unwrap_or(SILENT_MODE_DEFAULT)
}

fn get_codex_effort(settings: &BotSettings, chat_id: ChatId) -> Option<String> {
    settings.effort.get(&chat_id.0.to_string()).cloned()
}

fn get_claude_effort(settings: &BotSettings, chat_id: ChatId) -> Option<String> {
    settings.claude_effort.get(&chat_id.0.to_string()).cloned()
}

fn get_effort_for_provider(
    settings: &BotSettings,
    chat_id: ChatId,
    provider: &str,
) -> Option<String> {
    match provider {
        "claude" => get_claude_effort(settings, chat_id),
        "codex" => get_codex_effort(settings, chat_id),
        _ => None,
    }
}

fn is_codex_fast(settings: &BotSettings, chat_id: ChatId) -> bool {
    settings
        .codex_fast
        .get(&chat_id.0.to_string())
        .copied()
        .unwrap_or(false)
}

fn is_valid_codex_effort(v: &str) -> bool {
    matches!(v, "minimal" | "low" | "medium" | "high" | "xhigh")
}

fn is_valid_claude_effort(v: &str) -> bool {
    matches!(v, "low" | "medium" | "high" | "xhigh" | "max")
}

fn valid_effort_values(provider: &str) -> &'static str {
    match provider {
        "claude" => "low, medium, high, xhigh, max",
        "codex" => "minimal, low, medium, high, xhigh",
        _ => "Claude: low, medium, high, xhigh, max; Codex: minimal, low, medium, high, xhigh",
    }
}

fn is_valid_effort_for_provider(provider: &str, v: &str) -> bool {
    match provider {
        "claude" => is_valid_claude_effort(v),
        "codex" => is_valid_codex_effort(v),
        _ => false,
    }
}

/// Schedule entry persisted as JSON in ~/.cokacdir/schedule/
#[derive(Clone)]
struct ScheduleEntry {
    id: String,
    chat_id: i64,
    /// SHA-256 verifier of the owning bot_key, bound to (id, chat_id). The raw
    /// `bot_key` is never stored on disk in the modern format — see
    /// `live_schedule_key_verifier`. Legacy entries written before this change
    /// still carry a plaintext `bot_key` field on disk; `read_schedule_entry`
    /// transparently computes the verifier from those at load time, and the
    /// next `write_schedule_entry` rewrites the file in the new format.
    bot_key_verifier: String,
    current_path: String,
    prompt: String,
    schedule: String,      // original --at value (cron expression or absolute time)
    schedule_type: String, // "absolute" | "cron"
    once: Option<bool>,    // only meaningful for cron (None for absolute)
    last_run: Option<String>, // "2026-02-23 14:00:00"
    created_at: String,
    context_summary: Option<String>, // context summary text for session-isolated schedule
}

/// Verifier hash for a live schedule entry's owning bot_key. Binding to
/// `(schedule_id, chat_id)` ensures a verifier from one schedule cannot be
/// reused to authenticate another. The domain separator distinguishes this
/// from `schedule_history_key_verifier`, so verifiers from the two systems
/// are not interchangeable even if their input keys collide.
fn live_schedule_key_verifier(schedule_id: &str, chat_id: i64, bot_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"cokacdir:live_schedule:v1\0");
    hasher.update(schedule_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(chat_id.to_string().as_bytes());
    hasher.update(b"\0");
    hasher.update(bot_key.as_bytes());
    hex::encode(hasher.finalize())
}

/// Directory for schedule files: ~/.cokacdir/schedule/
fn schedule_dir() -> Option<std::path::PathBuf> {
    let result = dirs::home_dir().map(|h| h.join(".cokacdir").join("schedule"));
    sched_debug(&format!("[schedule_dir] → {:?}", result));
    result
}

fn schedule_inline_enabled_from_value(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

fn is_schedule_inline_enabled() -> bool {
    schedule_inline_enabled_from_value(std::env::var("COKAC_SCHEDULE_INLINE").ok().as_deref())
}

fn sched_debug(msg: &str) {
    crate::services::claude::debug_log_to("cron.log", msg);
}

fn msg_debug(msg: &str) {
    crate::services::claude::debug_log_to("msg.log", msg);
}

/// Always-on debug log (independent of /debug toggle).
/// Writes to ~/.cokacdir/debug/ai_trace.log for diagnosing AI execution issues.
fn ai_trace(msg: &str) {
    if let Some(home) = dirs::home_dir() {
        let debug_dir = home.join(".cokacdir").join("debug");
        let _ = std::fs::create_dir_all(&debug_dir);
        let log_path = debug_dir.join("ai_trace.log");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(log_path)
        {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
            let _ = std::io::Write::write_fmt(&mut file, format_args!("[{}] {}\n", timestamp, msg));
        }
    }
}

/// Log an incoming Telegram message to ~/.cokacdir/logs/telegram_YYYY-MM-DD.jsonl
fn log_incoming_message(msg: &Message, accepted: bool, reject_reason: &str) {
    let now = chrono::Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let ts = now.format("%Y-%m-%dT%H:%M:%S%.3f%:z").to_string();

    let logs_dir = match dirs::home_dir() {
        Some(h) => h.join(".cokacdir").join("logs"),
        None => return,
    };
    let _ = fs::create_dir_all(&logs_dir);
    let log_path = logs_dir.join(format!("telegram_{}.jsonl", date_str));

    let chat_id = msg.chat.id.0;
    let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);
    let username = msg
        .from
        .as_ref()
        .and_then(|u| u.username.as_deref())
        .unwrap_or("");
    let name = msg
        .from
        .as_ref()
        .map(|u| u.first_name.as_str())
        .unwrap_or("unknown");
    let msg_id = msg.id.0;

    let (msg_type, content) = if let Some(text) = msg.text() {
        ("text", text.to_string())
    } else if msg.animation().is_some() {
        // animation check must precede document: Telegram sets both fields for GIFs
        ("animation", msg.caption().unwrap_or("").to_string())
    } else if let Some(doc) = msg.document() {
        let fname = doc.file_name.as_deref().unwrap_or("");
        let caption = msg.caption().unwrap_or("");
        ("document", format!("[{}] {}", fname, caption))
    } else if msg.photo().is_some() {
        ("photo", msg.caption().unwrap_or("").to_string())
    } else if msg.sticker().is_some() {
        ("sticker", String::new())
    } else if msg.voice().is_some() {
        ("voice", String::new())
    } else if msg.video().is_some() {
        ("video", msg.caption().unwrap_or("").to_string())
    } else if let Some(audio) = msg.audio() {
        let fname = audio.file_name.as_deref().unwrap_or("");
        let caption = msg.caption().unwrap_or("");
        ("audio", format!("[{}] {}", fname, caption))
    } else if msg.video_note().is_some() {
        ("video_note", String::new())
    } else {
        ("other", String::new())
    };

    let entry = serde_json::json!({
        "ts": ts,
        "chat_id": chat_id,
        "user_id": user_id,
        "username": username,
        "name": name,
        "msg_id": msg_id,
        "type": msg_type,
        "content": content,
        "accepted": accepted,
        "reject_reason": if accepted { "" } else { reject_reason },
    });

    if let Ok(mut line) = serde_json::to_string(&entry) {
        line.push('\n');
        use std::io::Write;
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Bot-to-bot message entry read from ~/.cokacdir/messages/
#[derive(Clone)]
struct BotMessage {
    id: String,
    from: String,
    to: String,
    chat_id: String,
    content: String,
    created_at: String,
    file_path: std::path::PathBuf,
}

/// Read a single bot message from a JSON file
fn read_bot_message(path: &std::path::Path) -> Option<BotMessage> {
    msg_debug(&format!("[read_bot_message] reading: {}", path.display()));
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            msg_debug(&format!(
                "[read_bot_message] read failed: {} (path={})",
                e,
                path.display()
            ));
            return None;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            msg_debug(&format!(
                "[read_bot_message] JSON parse failed: {} (path={})",
                e,
                path.display()
            ));
            return None;
        }
    };
    let id = v.get("id").and_then(|x| x.as_str());
    let from = v.get("from").and_then(|x| x.as_str());
    let to = v.get("to").and_then(|x| x.as_str());
    let chat_id_val = v.get("chat_id").and_then(|x| x.as_str());
    let content_val = v.get("content").and_then(|x| x.as_str());
    let created_at = v.get("created_at").and_then(|x| x.as_str());
    if id.is_none()
        || from.is_none()
        || to.is_none()
        || chat_id_val.is_none()
        || content_val.is_none()
        || created_at.is_none()
    {
        msg_debug(&format!("[read_bot_message] missing fields: id={}, from={}, to={}, chat_id={}, content={}, created_at={} (path={})",
            id.is_some(), from.is_some(), to.is_some(), chat_id_val.is_some(), content_val.is_some(), created_at.is_some(), path.display()));
        return None;
    }
    let msg = BotMessage {
        id: id.unwrap().to_string(),
        from: from.unwrap().to_string(),
        to: to.unwrap().to_string(),
        chat_id: chat_id_val.unwrap().to_string(),
        content: content_val.unwrap().to_string(),
        created_at: created_at.unwrap().to_string(),
        file_path: path.to_path_buf(),
    };
    msg_debug(&format!(
        "[read_bot_message] ok: id={}, from={}, to={}, chat_id={}",
        msg.id, msg.from, msg.to, msg.chat_id
    ));
    Some(msg)
}

/// Scan messages directory for messages addressed to this bot, sorted by created_at (FIFO)
fn scan_messages(my_username: &str) -> Vec<BotMessage> {
    msg_debug(&format!(
        "[scan_messages] scanning for bot: {}",
        my_username
    ));
    let Some(dir) = messages_dir() else {
        msg_debug("[scan_messages] messages_dir() returned None");
        return Vec::new();
    };
    if !dir.is_dir() {
        msg_debug(&format!("[scan_messages] dir not found: {}", dir.display()));
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(&dir) else {
        msg_debug(&format!(
            "[scan_messages] read_dir failed: {}",
            dir.display()
        ));
        return Vec::new();
    };
    let all_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    let json_count = all_entries
        .iter()
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .count();
    msg_debug(&format!(
        "[scan_messages] dir entries={}, json files={}",
        all_entries.len(),
        json_count
    ));
    let mut msgs: Vec<BotMessage> = all_entries
        .into_iter()
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .filter_map(|e| read_bot_message(&e.path()))
        .filter(|m| {
            let matches = m.to.to_lowercase() == my_username.to_lowercase();
            if !matches {
                msg_debug(&format!(
                    "[scan_messages] skip msg id={}, to={} (not for {})",
                    m.id, m.to, my_username
                ));
            }
            matches
        })
        .collect();
    msgs.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    msg_debug(&format!(
        "[scan_messages] result: {} messages for {}",
        msgs.len(),
        my_username
    ));
    msgs
}

/// Check for timed-out messages (sent by this bot, still pending after 30 min)
async fn check_message_timeouts(bot: &Bot, my_username: &str, state: &SharedState) {
    msg_debug(&format!(
        "[check_message_timeouts] checking for bot: {}",
        my_username
    ));
    let Some(dir) = messages_dir() else {
        msg_debug("[check_message_timeouts] messages_dir() returned None");
        return;
    };
    if !dir.is_dir() {
        msg_debug(&format!(
            "[check_message_timeouts] dir not found: {}",
            dir.display()
        ));
        return;
    }
    let Ok(entries) = fs::read_dir(&dir) else {
        msg_debug(&format!(
            "[check_message_timeouts] read_dir failed: {}",
            dir.display()
        ));
        return;
    };

    let now = chrono::Local::now();
    let mut scanned = 0u32;
    let mut timed_out = 0u32;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.extension().map(|ext| ext == "json").unwrap_or(false) {
            continue;
        }
        let Some(msg) = read_bot_message(&path) else {
            continue;
        };
        if msg.from.to_lowercase() != my_username.to_lowercase() {
            msg_debug(&format!(
                "[check_message_timeouts] skip msg id={}, from={} (not from {})",
                msg.id, msg.from, my_username
            ));
            continue;
        }
        scanned += 1;

        // Check if created_at is older than 30 minutes
        match chrono::NaiveDateTime::parse_from_str(&msg.created_at, "%Y-%m-%d %H:%M:%S") {
            Ok(created) => {
                if let Some(created_dt) = created.and_local_timezone(chrono::Local).single() {
                    let elapsed = now.signed_duration_since(created_dt);
                    msg_debug(&format!(
                        "[check_message_timeouts] msg id={}, to={}, elapsed={}min",
                        msg.id,
                        msg.to,
                        elapsed.num_minutes()
                    ));
                    if elapsed.num_minutes() >= 30 {
                        // Delete the timed-out message
                        let remove_result = fs::remove_file(&path);
                        msg_debug(&format!("[check_message_timeouts] deleted timed-out message: {} (to={}, remove_ok={})",
                            msg.id, msg.to, remove_result.is_ok()));
                        timed_out += 1;

                        // Notify the chat
                        if let Ok(cid) = msg.chat_id.parse::<i64>() {
                            let chat_id = ChatId(cid);
                            shared_rate_limit_wait(state, chat_id).await;
                            let notice = format!("⏰ Message to @{} timed out.", msg.to);
                            let send_result =
                                tg!("send_message", bot.send_message(chat_id, notice).await);
                            msg_debug(&format!(
                                "[check_message_timeouts] notified chat_id={}, send_ok={}",
                                cid,
                                send_result.is_ok()
                            ));
                        } else {
                            msg_debug(&format!(
                                "[check_message_timeouts] invalid chat_id in msg: {}",
                                msg.chat_id
                            ));
                        }
                    }
                } else {
                    msg_debug(&format!("[check_message_timeouts] timezone conversion failed for msg id={}, created_at={}", msg.id, msg.created_at));
                }
            }
            Err(e) => {
                msg_debug(&format!(
                    "[check_message_timeouts] time parse failed for msg id={}: {} (created_at={})",
                    msg.id, e, msg.created_at
                ));
            }
        }
    }
    msg_debug(&format!(
        "[check_message_timeouts] done: scanned={}, timed_out={}",
        scanned, timed_out
    ));
}

/// Read a single schedule entry from a JSON file
fn read_schedule_entry(path: &std::path::Path) -> Option<ScheduleEntry> {
    sched_debug(&format!(
        "[read_schedule_entry] reading: {}",
        path.display()
    ));
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            sched_debug(&format!("[read_schedule_entry] read failed: {}", e));
            return None;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            sched_debug(&format!("[read_schedule_entry] parse failed: {}", e));
            return None;
        }
    };
    let id = v.get("id")?.as_str()?.to_string();
    let chat_id = v.get("chat_id")?.as_i64()?;
    // Modern files carry `bot_key_verifier`; pre-migration files still carry a
    // plaintext `bot_key`. Compute the verifier in-memory for legacy entries so
    // the rest of the read path is uniform; the next `write_schedule_entry`
    // (e.g. context update or last_run after a cron fire) will rewrite the
    // file without the plaintext field. We deliberately do NOT rewrite the
    // file from the read path: a concurrent `delete_schedule_entry` could let
    // such a rewrite resurrect an already-removed schedule.
    let bot_key_verifier =
        if let Some(verifier) = v.get("bot_key_verifier").and_then(|x| x.as_str()) {
            verifier.to_string()
        } else if let Some(legacy_key) = v.get("bot_key").and_then(|x| x.as_str()) {
            live_schedule_key_verifier(&id, chat_id, legacy_key)
        } else {
            sched_debug("[read_schedule_entry] missing both bot_key_verifier and bot_key");
            return None;
        };
    let entry = Some(ScheduleEntry {
        id,
        chat_id,
        bot_key_verifier,
        current_path: v.get("current_path")?.as_str()?.to_string(),
        prompt: v.get("prompt")?.as_str()?.to_string(),
        schedule: v.get("schedule")?.as_str()?.to_string(),
        schedule_type: v.get("schedule_type")?.as_str()?.to_string(),
        once: v.get("once").and_then(|v| v.as_bool()),
        last_run: v.get("last_run").and_then(|v| v.as_str()).map(String::from),
        created_at: v.get("created_at")?.as_str()?.to_string(),
        context_summary: v
            .get("context_summary")
            .and_then(|v| v.as_str())
            .map(String::from),
    });
    sched_debug(&format!(
        "[read_schedule_entry] result: id={}, type={}, schedule={}, last_run={:?}",
        entry.as_ref().map(|e| e.id.as_str()).unwrap_or("?"),
        entry
            .as_ref()
            .map(|e| e.schedule_type.as_str())
            .unwrap_or("?"),
        entry.as_ref().map(|e| e.schedule.as_str()).unwrap_or("?"),
        entry.as_ref().and_then(|e| e.last_run.as_deref()),
    ));
    entry
}

/// Write a schedule entry to its JSON file
fn write_schedule_entry(entry: &ScheduleEntry) -> Result<(), String> {
    sched_debug(&format!(
        "[write_schedule_entry] id={}, type={}, schedule={}, once={:?}, last_run={:?}",
        entry.id, entry.schedule_type, entry.schedule, entry.once, entry.last_run
    ));
    let dir = schedule_dir().ok_or("Cannot determine home directory")?;
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create schedule dir: {e}"))?;
    // `bot_key_verifier` replaces the legacy plaintext `bot_key` field.
    // Re-writing a legacy file through this path naturally drops the old
    // plaintext field on the next atomic rename.
    let mut json = serde_json::json!({
        "id": entry.id,
        "chat_id": entry.chat_id,
        "bot_key_verifier": entry.bot_key_verifier,
        "current_path": entry.current_path,
        "prompt": entry.prompt,
        "schedule": entry.schedule,
        "schedule_type": entry.schedule_type,
        "last_run": entry.last_run,
        "created_at": entry.created_at,
        "context_summary": entry.context_summary,
    });
    if let Some(once_val) = entry.once {
        json.as_object_mut()
            .unwrap()
            .insert("once".to_string(), serde_json::json!(once_val));
    }
    let path = dir.join(format!("{}.json", entry.id));
    let tmp_path = dir.join(format!("{}.json.tmp", entry.id));
    sched_debug(&format!(
        "[write_schedule_entry] writing tmp: {}",
        tmp_path.display()
    ));
    fs::write(
        &tmp_path,
        serde_json::to_string_pretty(&json).unwrap_or_default(),
    )
    .map_err(|e| format!("Failed to write schedule file: {e}"))?;
    sched_debug(&format!(
        "[write_schedule_entry] atomic rename: {} → {}",
        tmp_path.display(),
        path.display()
    ));
    let result =
        fs::rename(&tmp_path, &path).map_err(|e| format!("Failed to finalize schedule file: {e}"));
    sched_debug(&format!("[write_schedule_entry] result: {:?}", result));
    result
}

/// List all schedule entries matching the given bot_key and optionally chat_id
fn list_schedule_entries(bot_key: &str, chat_id: Option<i64>) -> Vec<ScheduleEntry> {
    sched_debug(&format!(
        "[list_schedule_entries] bot_key=<redacted>, chat_id={:?}",
        chat_id
    ));
    let Some(dir) = schedule_dir() else {
        sched_debug("[list_schedule_entries] no schedule dir");
        return Vec::new();
    };
    if !dir.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(&dir) else {
        sched_debug("[list_schedule_entries] read_dir failed");
        return Vec::new();
    };
    let mut result: Vec<ScheduleEntry> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .filter_map(|e| read_schedule_entry(&e.path()))
        .filter(|e| {
            // Recompute the expected verifier per entry — chat_id and id are
            // baked into the verifier, so two schedules sharing a bot_key but
            // different ids produce different verifiers.
            e.bot_key_verifier == live_schedule_key_verifier(&e.id, e.chat_id, bot_key)
        })
        .filter(|e| chat_id.map_or(true, |cid| e.chat_id == cid))
        .collect();
    result.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    sched_debug(&format!(
        "[list_schedule_entries] found {} entries: [{}]",
        result.len(),
        result
            .iter()
            .map(|e| format!("{}({})", e.id, e.schedule_type))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    result
}

/// Delete a schedule entry by ID
fn delete_schedule_entry(id: &str) -> bool {
    sched_debug(&format!("[delete_schedule_entry] id={}", id));
    let Some(dir) = schedule_dir() else {
        sched_debug("[delete_schedule_entry] no schedule dir");
        return false;
    };
    let path = dir.join(format!("{id}.json"));
    let existed = path.exists();
    let ok = fs::remove_file(&path).is_ok();
    sched_debug(&format!(
        "[delete_schedule_entry] path={}, existed={}, removed={}",
        path.display(),
        existed,
        ok
    ));

    // Also remove the .result file if it exists
    let result_path = dir.join(format!("{id}.result"));
    if result_path.exists() {
        let _ = fs::remove_file(&result_path);
        sched_debug(&format!(
            "[delete_schedule_entry] also removed .result: {}",
            result_path.display()
        ));
    }

    ok
}

/// Directory for schedule run history files: ~/.cokacdir/schedule_history/
/// Each schedule keeps one JSONL file (`<id>.log`) with one record per execution.
fn schedule_history_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".cokacdir").join("schedule_history"))
}

/// Public access to the schedule history file path (for `--cron-history` in main.rs).
pub fn schedule_history_path_pub(id: &str) -> Option<std::path::PathBuf> {
    schedule_history_dir().map(|d| d.join(format!("{}.log", id)))
}

/// Append one JSONL record describing a single schedule execution.
///
/// Best-effort logging: any filesystem error is swallowed (with a sched_debug entry)
/// so a failure here never affects the schedule's own user-facing completion path.
/// Each record carries `chat_id` and a non-secret verifier so `--cron-history` can
/// authorize the caller even after the underlying schedule entry has been deleted
/// (one-time / `--once` schedules) without exposing the bot key capability.
fn append_schedule_history(
    schedule_id: &str,
    chat_id: i64,
    bot_key: &str,
    prompt: &str,
    status: &str, // "ok" | "cancelled" | "error"
    response: &str,
    error: Option<&str>,
    workspace_path: &str,
    duration_ms: u64,
) {
    let Some(dir) = schedule_history_dir() else {
        sched_debug(&format!(
            "[append_schedule_history] id={}, no home dir → skip",
            schedule_id
        ));
        return;
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        sched_debug(&format!(
            "[append_schedule_history] id={}, create_dir_all failed: {}",
            schedule_id, e
        ));
        return;
    }

    // Cap response length to avoid unbounded growth on long-output schedules.
    // Floor to a UTF-8 char boundary so the truncated string stays valid.
    const MAX_RESPONSE_LEN: usize = 4096;
    let response_capped = if response.len() <= MAX_RESPONSE_LEN {
        response.to_string()
    } else {
        let mut end = MAX_RESPONSE_LEN;
        while end > 0 && !response.is_char_boundary(end) {
            end -= 1;
        }
        let mut s = response[..end].to_string();
        s.push_str("\n…[truncated]");
        s
    };

    let mut record = serde_json::json!({
        "ts": chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%:z").to_string(),
        "schedule_id": schedule_id,
        "chat_id": chat_id,
        "bot_key_verifier": schedule_history_key_verifier(schedule_id, chat_id, bot_key),
        "prompt": prompt,
        "status": status,
        "response": response_capped,
        "workspace_path": workspace_path,
        "duration_ms": duration_ms,
    });
    if let Some(err) = error {
        record
            .as_object_mut()
            .unwrap()
            .insert("error".to_string(), serde_json::json!(err));
    }

    let path = dir.join(format!("{}.log", schedule_id));
    let Some(_history_lock) = lock_schedule_history_file(&path) else {
        sched_debug(&format!(
            "[append_schedule_history] id={}, lock failed: {}",
            schedule_id,
            path.display()
        ));
        return;
    };
    redact_schedule_history_file_once_unlocked(&path);
    let line = format!("{}\n", record);
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
    match result {
        Ok(_) => sched_debug(&format!(
            "[append_schedule_history] id={}, status={}, bytes={}, path={}",
            schedule_id,
            status,
            line.len(),
            path.display()
        )),
        Err(e) => sched_debug(&format!(
            "[append_schedule_history] id={}, write failed: {}, path={}",
            schedule_id,
            e,
            path.display()
        )),
    }
}

fn schedule_history_key_verifier(schedule_id: &str, chat_id: i64, bot_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"cokacdir:schedule_history:v1\0");
    hasher.update(schedule_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(chat_id.to_string().as_bytes());
    hasher.update(b"\0");
    hasher.update(bot_key.as_bytes());
    hex::encode(hasher.finalize())
}

fn lock_schedule_history_file(path: &std::path::Path) -> Option<std::fs::File> {
    use fs2::FileExt;

    let lock_path = path.with_extension("log.lock");
    if let Some(parent) = lock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .ok()?;
    if file.lock_exclusive().is_err() {
        return None;
    }
    Some(file)
}

fn redact_schedule_history_file(path: &std::path::Path) {
    let Some(_history_lock) = lock_schedule_history_file(path) else {
        sched_debug(&format!(
            "[redact_schedule_history_file] lock failed: {}",
            path.display()
        ));
        return;
    };
    redact_schedule_history_file_once_unlocked(path);
}

fn schedule_history_redaction_marker_path(path: &std::path::Path) -> std::path::PathBuf {
    path.with_extension("log.redacted")
}

fn mark_schedule_history_redacted(path: &std::path::Path) {
    let marker = schedule_history_redaction_marker_path(path);
    if let Some(parent) = marker.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(marker, b"v1\n");
}

fn redact_schedule_history_file_once_unlocked(path: &std::path::Path) {
    let marker = schedule_history_redaction_marker_path(path);
    if marker.exists() {
        return;
    }

    let ok = if path.exists() {
        redact_schedule_history_file_unlocked(path)
    } else {
        true
    };
    if ok {
        mark_schedule_history_redacted(path);
    }
}

fn redact_schedule_history_file_unlocked(path: &std::path::Path) -> bool {
    let Ok(content) = fs::read_to_string(path) else {
        return false;
    };

    let mut changed = false;
    let mut rewritten = String::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            rewritten.push_str(line);
            rewritten.push('\n');
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(mut record) => {
                let legacy_key = record
                    .get("bot_key")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let schedule_id = record
                    .get("schedule_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let chat_id = record.get("chat_id").and_then(|v| v.as_i64());

                if let (Some(legacy_key), Some(schedule_id), Some(chat_id)) =
                    (legacy_key, schedule_id, chat_id)
                {
                    if let Some(obj) = record.as_object_mut() {
                        obj.insert(
                            "bot_key_verifier".to_string(),
                            serde_json::json!(schedule_history_key_verifier(
                                &schedule_id,
                                chat_id,
                                &legacy_key,
                            )),
                        );
                        obj.remove("bot_key");
                        changed = true;
                    }
                }

                rewritten.push_str(&record.to_string());
            }
            Err(_) => rewritten.push_str(line),
        }
        rewritten.push('\n');
    }

    if changed {
        // Atomic rewrite: write to a sibling tmp file then rename over the target,
        // matching the Slack channel-map persist pattern. This keeps the file in
        // either its old or new state under power-loss / partial-write conditions.
        let tmp_path = path.with_extension("log.redact.tmp");
        let result = fs::write(&tmp_path, rewritten).and_then(|_| fs::rename(&tmp_path, path));
        match result {
            Ok(_) => {
                sched_debug(&format!(
                    "[redact_schedule_history_file] redacted legacy bot_key fields: {}",
                    path.display()
                ));
                true
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp_path);
                sched_debug(&format!(
                    "[redact_schedule_history_file] rewrite failed: {}, path={}",
                    e,
                    path.display()
                ));
                false
            }
        }
    } else {
        true
    }
}

pub fn redact_schedule_history_file_pub(path: &std::path::Path) {
    redact_schedule_history_file(path);
}

pub fn schedule_history_record_authorized_pub(
    record: &serde_json::Value,
    schedule_id: &str,
    chat_id: i64,
    bot_key: &str,
) -> bool {
    let schedule_match = record.get("schedule_id").and_then(|v| v.as_str()) == Some(schedule_id);
    let chat_match = record.get("chat_id").and_then(|v| v.as_i64()) == Some(chat_id);
    if !schedule_match || !chat_match {
        return false;
    }

    let expected = schedule_history_key_verifier(schedule_id, chat_id, bot_key);
    if record.get("bot_key_verifier").and_then(|v| v.as_str()) == Some(expected.as_str()) {
        return true;
    }

    // Backward compatibility for history records written before the verifier field
    // existed. These legacy records are sanitized before being returned to callers.
    record.get("bot_key").and_then(|v| v.as_str()) == Some(bot_key)
}

pub fn sanitize_schedule_history_record_pub(record: &mut serde_json::Value) {
    if let Some(obj) = record.as_object_mut() {
        obj.remove("bot_key");
        obj.remove("bot_key_verifier");
    }
}

#[cfg(test)]
mod schedule_history_tests {
    use super::{
        redact_schedule_history_file_pub, sanitize_schedule_history_record_pub,
        schedule_history_key_verifier, schedule_history_record_authorized_pub,
        schedule_history_redaction_marker_path,
    };

    #[test]
    fn authorizes_history_with_verifier_or_legacy_key() {
        let verifier = schedule_history_key_verifier("sched-1", -42, "secret-key");
        let modern = serde_json::json!({
            "schedule_id": "sched-1",
            "chat_id": -42,
            "bot_key_verifier": verifier,
        });
        assert!(schedule_history_record_authorized_pub(
            &modern,
            "sched-1",
            -42,
            "secret-key"
        ));
        assert!(!schedule_history_record_authorized_pub(
            &modern,
            "sched-1",
            -42,
            "wrong-key"
        ));

        let legacy = serde_json::json!({
            "schedule_id": "sched-1",
            "chat_id": -42,
            "bot_key": "secret-key",
        });
        assert!(schedule_history_record_authorized_pub(
            &legacy,
            "sched-1",
            -42,
            "secret-key"
        ));
    }

    #[test]
    fn sanitizes_history_records_before_output() {
        let mut record = serde_json::json!({
            "schedule_id": "sched-1",
            "chat_id": -42,
            "bot_key": "secret-key",
            "bot_key_verifier": "verifier",
            "status": "ok",
        });

        sanitize_schedule_history_record_pub(&mut record);

        assert!(record.get("bot_key").is_none());
        assert!(record.get("bot_key_verifier").is_none());
        assert_eq!(record.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[test]
    fn redacts_legacy_schedule_history_file_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sched-1.log");
        std::fs::write(
            &path,
            r#"{"schedule_id":"sched-1","chat_id":-42,"bot_key":"secret-key","status":"ok"}"#,
        )
        .unwrap();

        redact_schedule_history_file_pub(&path);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(!content.contains("bot_key\":\"secret-key"));
        assert!(content.contains("bot_key_verifier"));
        assert!(schedule_history_redaction_marker_path(&path).exists());
    }
}

#[cfg(test)]
mod live_schedule_tests {
    use super::{
        live_schedule_key_verifier, read_schedule_entry, schedule_history_key_verifier,
        write_schedule_entry_pub, ScheduleEntryData,
    };

    fn data_with_bot_key(bot_key: &str) -> ScheduleEntryData {
        ScheduleEntryData {
            id: "sched-guard".to_string(),
            chat_id: 1,
            bot_key: bot_key.to_string(),
            current_path: "/tmp".to_string(),
            prompt: "p".to_string(),
            schedule: "* * * * *".to_string(),
            schedule_type: "cron".to_string(),
            once: Some(false),
            last_run: None,
            created_at: "2026-01-01 00:00:00".to_string(),
            context_summary: None,
        }
    }

    #[test]
    fn write_schedule_entry_pub_rejects_empty_bot_key() {
        // Empty bot_key would compute a verifier the owning bot cannot match,
        // silently orphaning the schedule. Guard catches the
        // list-then-modify-then-write footgun before it touches disk.
        let result = write_schedule_entry_pub(&data_with_bot_key(""));
        assert!(result.is_err(), "empty bot_key must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("orphan"),
            "error must explain why empty bot_key is rejected: {}",
            msg
        );
    }

    #[test]
    fn live_verifier_distinct_from_history_verifier() {
        // Domain separation guarantees a verifier minted for live ownership
        // cannot be reused to authorize a history record (or vice versa) even
        // when (id, chat_id, bot_key) collide.
        let live = live_schedule_key_verifier("sched-1", -42, "secret");
        let hist = schedule_history_key_verifier("sched-1", -42, "secret");
        assert_ne!(live, hist);
    }

    #[test]
    fn live_verifier_binds_to_id_and_chat_id() {
        // Changing any input changes the verifier — this is what keeps the
        // verifier from one schedule from authorizing access to another.
        let base = live_schedule_key_verifier("a", 1, "k");
        assert_ne!(base, live_schedule_key_verifier("b", 1, "k"));
        assert_ne!(base, live_schedule_key_verifier("a", 2, "k"));
        assert_ne!(base, live_schedule_key_verifier("a", 1, "k2"));
        assert_eq!(base, live_schedule_key_verifier("a", 1, "k"));
    }

    #[test]
    fn read_schedule_entry_migrates_legacy_bot_key_in_memory() {
        // Pre-migration files store the raw bot_key. The reader must still
        // surface a usable ScheduleEntry whose verifier matches what the
        // current `live_schedule_key_verifier(id, chat_id, raw_key)` produces,
        // so list filters keep working until the file is re-written.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sched-legacy.json");
        std::fs::write(
            &path,
            r#"{
                "id": "sched-legacy",
                "chat_id": 99,
                "bot_key": "legacy-raw-key",
                "current_path": "/tmp",
                "prompt": "p",
                "schedule": "* * * * *",
                "schedule_type": "cron",
                "created_at": "2026-01-01 00:00:00"
            }"#,
        )
        .unwrap();

        let entry = read_schedule_entry(&path).expect("legacy entry should parse");
        assert_eq!(
            entry.bot_key_verifier,
            live_schedule_key_verifier("sched-legacy", 99, "legacy-raw-key")
        );
        // Read must NOT rewrite the file — that would race with concurrent
        // delete and could resurrect a removed schedule. The plaintext stays
        // on disk until the next legitimate write.
        let content_after = std::fs::read_to_string(&path).unwrap();
        assert!(content_after.contains("\"bot_key\""));
    }

    #[test]
    fn read_schedule_entry_prefers_verifier_when_both_present() {
        // Hybrid files (verifier added but legacy field not yet stripped) must
        // trust the verifier — recomputing from the legacy key would silently
        // mask a tampered/rotated verifier.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sched-hybrid.json");
        let real_verifier = live_schedule_key_verifier("sched-hybrid", 1, "actual-key");
        let body = format!(
            r#"{{
                "id": "sched-hybrid",
                "chat_id": 1,
                "bot_key": "stale-key",
                "bot_key_verifier": "{}",
                "current_path": "/tmp",
                "prompt": "p",
                "schedule": "* * * * *",
                "schedule_type": "cron",
                "created_at": "2026-01-01 00:00:00"
            }}"#,
            real_verifier
        );
        std::fs::write(&path, body).unwrap();
        let entry = read_schedule_entry(&path).expect("hybrid entry should parse");
        assert_eq!(entry.bot_key_verifier, real_verifier);
    }
}

/// Delete a schedule's run-history file. Called from `--cron-remove` so that
/// removing a schedule also clears its accumulated history (consistent with how
/// `delete_schedule_entry` already removes the schedule's `.result` companion).
fn delete_schedule_history(id: &str) {
    let Some(dir) = schedule_history_dir() else {
        return;
    };
    let path = dir.join(format!("{}.log", id));
    if path.exists() {
        match fs::remove_file(&path) {
            Ok(_) => sched_debug(&format!(
                "[delete_schedule_history] removed: {}",
                path.display()
            )),
            Err(e) => sched_debug(&format!(
                "[delete_schedule_history] remove failed: {}, path={}",
                e,
                path.display()
            )),
        }
    }
    let marker = schedule_history_redaction_marker_path(&path);
    if marker.exists() {
        let _ = fs::remove_file(marker);
    }
    // Lock sentinel created by lock_schedule_history_file. Removing it keeps the
    // schedule_history dir tidy when a schedule (and its history) goes away.
    let lock_path = path.with_extension("log.lock");
    if lock_path.exists() {
        let _ = fs::remove_file(lock_path);
    }
}

/// Public wrapper for `delete_schedule_history` (used by `--cron-remove` in main.rs).
pub fn delete_schedule_history_pub(id: &str) {
    delete_schedule_history(id);
}

/// Parse a relative time string (e.g. "4h", "30m", "1d") into a future DateTime
fn parse_relative_time(s: &str) -> Option<chrono::DateTime<chrono::Local>> {
    sched_debug(&format!("[parse_relative_time] input: {:?}", s));
    let s = s.trim();
    if s.len() < 2 {
        sched_debug("[parse_relative_time] too short → None");
        return None;
    }
    let (num_part, unit) = s.split_at(s.len() - 1);
    let num: i64 = match num_part.parse() {
        Ok(n) => n,
        Err(_) => {
            sched_debug(&format!(
                "[parse_relative_time] invalid number: {:?} → None",
                num_part
            ));
            return None;
        }
    };
    if num <= 0 {
        sched_debug("[parse_relative_time] num <= 0 → None");
        return None;
    }
    let seconds = match unit {
        "m" => num * 60,
        "h" => num * 3600,
        "d" => num * 86400,
        _ => {
            sched_debug(&format!(
                "[parse_relative_time] unknown unit: {:?} → None",
                unit
            ));
            return None;
        }
    };
    let result = Some(chrono::Local::now() + chrono::Duration::seconds(seconds));
    sched_debug(&format!(
        "[parse_relative_time] → {:?}",
        result
            .as_ref()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
    ));
    result
}

/// Check if a cron expression matches the given datetime.
/// 5 fields: minute, hour, day-of-month, month, day-of-week (0=Sun)
fn cron_matches(expr: &str, dt: chrono::DateTime<chrono::Local>) -> bool {
    use chrono::Datelike;
    use chrono::Timelike;

    let fields: Vec<&str> = expr.split_whitespace().collect();
    if fields.len() != 5 {
        sched_debug(&format!(
            "[cron_matches] invalid field count: {} (expected 5) for expr={:?}",
            fields.len(),
            expr
        ));
        return false;
    }

    let values = [
        dt.minute(),
        dt.hour(),
        dt.day(),
        dt.month(),
        dt.weekday().num_days_from_sunday(),
    ];
    let field_names = ["minute", "hour", "day", "month", "dow"];

    // Range start for each field: minute(0), hour(0), day-of-month(1), month(1), day-of-week(0)
    let range_starts = [0u32, 0, 1, 1, 0];

    for (i, ((field, &val), &range_start)) in fields
        .iter()
        .zip(values.iter())
        .zip(range_starts.iter())
        .enumerate()
    {
        let matched = cron_field_matches(field, val, range_start);
        if !matched {
            sched_debug(&format!(
                "[cron_matches] expr={:?}, dt={}, {}({})!={} → false",
                expr,
                dt.format("%H:%M"),
                field_names[i],
                val,
                field
            ));
            return false;
        }
    }
    sched_debug(&format!(
        "[cron_matches] expr={:?}, dt={} → true",
        expr,
        dt.format("%H:%M")
    ));
    true
}

/// Check if a single cron field matches a value.
/// Supports: *, single number, comma-separated list, ranges (a-b), step (*/n, a-b/n)
/// range_start: the minimum value for this field (0 for minute/hour/dow, 1 for day/month)
fn cron_field_matches(field: &str, val: u32, range_start: u32) -> bool {
    if field == "*" {
        return true;
    }

    for part in field.split(',') {
        let part = part.trim();
        // Handle step: */n or a-b/n
        if let Some((range_part, step_str)) = part.split_once('/') {
            if let Ok(step) = step_str.parse::<u32>() {
                if step == 0 {
                    continue;
                }
                if range_part == "*" {
                    if (val - range_start) % step == 0 {
                        sched_debug(&format!(
                            "[cron_field_matches] field={}, val={}, */{}  → true",
                            field, val, step
                        ));
                        return true;
                    }
                } else if let Some((start_str, end_str)) = range_part.split_once('-') {
                    if let (Ok(start), Ok(end)) = (start_str.parse::<u32>(), end_str.parse::<u32>())
                    {
                        if val >= start && val <= end && (val - start) % step == 0 {
                            sched_debug(&format!(
                                "[cron_field_matches] field={}, val={}, {}-{}/{} → true",
                                field, val, start, end, step
                            ));
                            return true;
                        }
                    }
                }
            }
        } else if let Some((start_str, end_str)) = part.split_once('-') {
            // Range: a-b
            if let (Ok(start), Ok(end)) = (start_str.parse::<u32>(), end_str.parse::<u32>()) {
                if val >= start && val <= end {
                    sched_debug(&format!(
                        "[cron_field_matches] field={}, val={}, range {}-{} → true",
                        field, val, start, end
                    ));
                    return true;
                }
            }
        } else {
            // Single number
            if let Ok(n) = part.parse::<u32>() {
                if val == n {
                    sched_debug(&format!(
                        "[cron_field_matches] field={}, val={}, exact {} → true",
                        field, val, n
                    ));
                    return true;
                }
            }
        }
    }
    false
}

// === Public API for CLI commands (main.rs) ===

/// Public data struct mirroring ScheduleEntry for cross-module use.
///
/// `bot_key` is the **raw bot_key** supplied by the caller and is asymmetric
/// across the conversion boundary: callers populate it with the raw key on
/// the way in (`From<&ScheduleEntryData> for ScheduleEntry` derives the
/// verifier from it), while on the way out (`From<&ScheduleEntry>`), no raw
/// key is recoverable from disk and the field is left empty.
///
/// **Round-trip caveat**: a list-then-modify-then-write sequence will lose
/// ownership unless the caller re-populates `bot_key` with the raw key
/// before calling `write_schedule_entry_pub`. The latter rejects empty
/// `bot_key` to make the failure explicit instead of silently orphaning
/// the entry.
#[derive(Clone)]
pub struct ScheduleEntryData {
    pub id: String,
    pub chat_id: i64,
    pub bot_key: String,
    pub current_path: String,
    pub prompt: String,
    pub schedule: String,
    pub schedule_type: String,
    pub once: Option<bool>, // only meaningful for cron (None for absolute)
    pub last_run: Option<String>,
    pub created_at: String,
    pub context_summary: Option<String>,
}

impl From<&ScheduleEntry> for ScheduleEntryData {
    fn from(e: &ScheduleEntry) -> Self {
        Self {
            id: e.id.clone(),
            chat_id: e.chat_id,
            // The raw bot_key is not stored on disk; surfacing the verifier
            // here would mislead callers that pass this back through
            // `From<&ScheduleEntryData>` (they'd hash the verifier). Leaving
            // it empty makes the asymmetry explicit.
            bot_key: String::new(),
            current_path: e.current_path.clone(),
            prompt: e.prompt.clone(),
            schedule: e.schedule.clone(),
            schedule_type: e.schedule_type.clone(),
            once: e.once,
            last_run: e.last_run.clone(),
            created_at: e.created_at.clone(),
            context_summary: e.context_summary.clone(),
        }
    }
}

impl From<&ScheduleEntryData> for ScheduleEntry {
    fn from(d: &ScheduleEntryData) -> Self {
        Self {
            id: d.id.clone(),
            chat_id: d.chat_id,
            bot_key_verifier: live_schedule_key_verifier(&d.id, d.chat_id, &d.bot_key),
            current_path: d.current_path.clone(),
            prompt: d.prompt.clone(),
            schedule: d.schedule.clone(),
            schedule_type: d.schedule_type.clone(),
            once: d.once,
            last_run: d.last_run.clone(),
            created_at: d.created_at.clone(),
            context_summary: d.context_summary.clone(),
        }
    }
}

pub fn parse_relative_time_pub(s: &str) -> Option<chrono::DateTime<chrono::Local>> {
    parse_relative_time(s)
}

pub fn write_schedule_entry_pub(data: &ScheduleEntryData) -> Result<(), String> {
    // Refuse to persist an entry derived from an empty raw bot_key: the
    // resulting `bot_key_verifier` would not match any real owner and the
    // schedule would be silently orphaned (invisible to the bot that should
    // own it). `list_schedule_entries_pub` returns ScheduleEntryData with an
    // empty bot_key, so any list-then-modify-then-write code path must
    // re-supply the raw key before calling this function.
    if data.bot_key.is_empty() {
        return Err(
            "write_schedule_entry_pub: empty bot_key would orphan the entry; \
             callers performing read-modify-write must reset bot_key from the \
             owning bot's key before write"
                .to_string(),
        );
    }
    let entry = ScheduleEntry::from(data);
    write_schedule_entry(&entry)
}

pub fn list_schedule_entries_pub(bot_key: &str, chat_id: Option<i64>) -> Vec<ScheduleEntryData> {
    list_schedule_entries(bot_key, chat_id)
        .iter()
        .map(ScheduleEntryData::from)
        .collect()
}

pub fn list_all_schedule_ids_pub() -> std::collections::HashSet<String> {
    let Some(dir) = schedule_dir() else {
        return std::collections::HashSet::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return std::collections::HashSet::new();
    };
    entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if path.extension().map(|ext| ext == "json").unwrap_or(false) {
                path.file_stem().map(|s| s.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect()
}

pub fn delete_schedule_entry_pub(id: &str) -> bool {
    delete_schedule_entry(id)
}

/// Resolve the current working path for a chat from bot_settings.json
pub fn resolve_current_path_for_chat(chat_id: i64, hash_key: &str) -> Option<String> {
    let path = bot_settings_path()?;
    let content = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let entry = json.get(hash_key)?;
    let last_sessions = entry.get("last_sessions")?.as_object()?;
    let chat_key = chat_id.to_string();
    last_sessions.get(&chat_key)?.as_str().map(String::from)
}

/// Get the binary path normalized for shell commands (backslashes → forward slashes on Windows)
fn shell_bin_path() -> String {
    crate::utils::format::to_shell_path(crate::bin_path())
}

const DEFAULT_COWORK_GUIDELINES: &str = "\
FOR EFFECTIVE CO-WORK, BE AWARE OF:
1. WHO is here — Check the group chat log to discover which bots are active in this chat.
2. WHERE each bot works — Other bots may have different working directories. Check the log or ask them via --message.
3. WHAT each bot is doing — Read the log to understand ongoing tasks before starting your own.
4. SHARED GOAL — When the user gives a collaborative task, understand the overall objective and your part in it.

CO-WORK GUIDELINES:
• Before starting work, check the chat log to understand the current state of collaboration.
• Clearly state what you are working on — your messages are recorded in the shared log for other bots.
• Before modifying shared files or directories, check the log to see if another bot is working on the same area.
• When your work depends on or affects another bot's output, communicate via --message (described below).
• If you need results from another bot's task, check the log first — the answer may already be there.

NO-REPEAT RULE (CRITICAL — READ CAREFULLY):
Before responding, check the group chat log for responses from OTHER bots to the SAME user request.
If another bot has ALREADY responded to the same request:
• Do NOT answer the same question again — not even in different words.
• Do NOT perform the same task again — not even with a different approach.
• Do NOT summarize, rephrase, or restate what the previous bot already said.
• Instead, choose ONE of these actions:
  1. ADD NEW VALUE: provide information the previous bot missed, a different angle, or deeper analysis on an aspect they didn't cover.
  2. TAKE THE NEXT STEP: if the previous bot completed a task, do the logical follow-up action (e.g., bot 1 analyzed the problem → you implement the fix; bot 1 wrote code → you test it).
  3. REVIEW & BUILD: evaluate the previous bot's output and extend it (e.g., point out edge cases, suggest improvements, add error handling).
  4. ACKNOWLEDGE & SKIP: if the previous bot's answer is already complete and sufficient, briefly acknowledge (one sentence) and do NOT elaborate further.
• The worst outcome is two bots saying essentially the same thing. When in doubt, choose option 4 (acknowledge & skip) over repeating.

INDIVIDUALITY RULE:
Each bot is an independent entity with its own personality and perspective.
Express your viewpoint in your own voice. Never parrot or echo another bot.

BREVITY RULE:
You are a participant in a group chat. Writing long messages alone is inconsiderate to other participants.
Keep your answers short and concise — ideally one or two sentences.";

/// Load co-work guidelines from ~/.cokacdir/prompt/cowork.md.
/// If the file does not exist, creates it with default content.
fn load_cowork_guidelines() -> String {
    if let Some(home) = dirs::home_dir() {
        let prompt_dir = home.join(".cokacdir").join("prompt");
        let cowork_path = prompt_dir.join("cowork.md");
        if cowork_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&cowork_path) {
                let trimmed = content.trim().to_string();
                msg_debug(&format!(
                    "[load_cowork_guidelines] loaded from file: {}, len={}, empty={}",
                    cowork_path.display(),
                    trimmed.len(),
                    trimmed.is_empty()
                ));
                return trimmed;
            } else {
                msg_debug(&format!(
                    "[load_cowork_guidelines] failed to read file: {}",
                    cowork_path.display()
                ));
            }
        } else {
            // Auto-generate default file
            let _ = std::fs::create_dir_all(&prompt_dir);
            let _ = std::fs::write(&cowork_path, DEFAULT_COWORK_GUIDELINES);
            msg_debug(&format!(
                "[load_cowork_guidelines] created default file: {}",
                cowork_path.display()
            ));
        }
    }
    msg_debug("[load_cowork_guidelines] using built-in default");
    DEFAULT_COWORK_GUIDELINES.to_string()
}

/// Build the system prompt for AI sessions
fn build_system_prompt(
    role: &str,
    current_path: &str,
    chat_id: i64,
    bot_key: &str,
    disabled_notice: &str,
    session_id: Option<&str>,
    bot_username: &str,
    bot_display_name: &str,
    user_message: Option<&str>,
    force_bot_messaging_context: bool,
    context_count: usize,
    platform: &str,
) -> String {
    msg_debug(&format!("[build_system_prompt] chat_id={}, bot_username={:?}, bot_display_name={:?}, session_id={:?}, disabled_notice_len={}, role_len={}",
        chat_id, bot_username, bot_display_name, session_id, disabled_notice.len(), role.len()));
    let is_group_chat = chat_id < 0 && context_count > 0;
    msg_debug(&format!("[build_system_prompt] is_group_chat={}, context_count={}, has_bot_username={}, include_bot_section={}",
        is_group_chat, context_count, !bot_username.is_empty(), !bot_username.is_empty() && is_group_chat));
    let session_notice = match session_id {
        Some(sid) => format!(
            "\n\n\
             Current session ID: {sid}\n\
             When scheduling a task that CONTINUES or EXTENDS the current conversation \
             (e.g. \"finish this later\", \"do the rest tomorrow\", \"remind me to continue this\"), \
             add --session {sid} to the --cron command so the scheduled task inherits this conversation context.\n\
             Do NOT use --session for independent tasks that don't need the current conversation history \
             (e.g. \"check server status every hour\", \"send a daily report\")."
        ),
        None => String::new(),
    };
    let bot_username_line = if !bot_username.is_empty() && is_group_chat {
        if !bot_display_name.is_empty() {
            format!("You are: {} (@{})\n", bot_display_name, bot_username)
        } else {
            format!("You are: @{}\n", bot_username)
        }
    } else {
        String::new()
    };
    let group_chat_log_section = if is_group_chat {
        // Fetch the last N log entries to embed directly in the prompt
        let recent_entries = read_group_chat_log_range(chat_id, 1, None, None);
        let start = recent_entries.len().saturating_sub(context_count);
        let recent_lines: String = recent_entries[start..]
            .iter()
            .map(|(_, entry)| {
                let bot_label = match &entry.bot_display_name {
                    Some(dn) if !dn.is_empty() => format!("{}(@{})", dn, entry.bot),
                    _ => format!("@{}", entry.bot),
                };
                let from_info = entry
                    .from
                    .as_deref()
                    .map(|f| format!("({})", f))
                    .unwrap_or_default();
                let role_display = if entry.role == "user" {
                    format!("user→{}", bot_label)
                } else {
                    bot_label
                };
                let display_text = if entry.role == "assistant" {
                    let parsed = parse_payload_auto(&entry.text);
                    if parsed.is_empty() {
                        entry.text.clone()
                    } else {
                        format_raw_payload(&parsed)
                    }
                } else {
                    entry.text.clone()
                };
                format!(
                    "  [{}] {}{}: {}\n",
                    entry.ts, role_display, from_info, display_text
                )
            })
            .collect();

        let recent_section = if recent_lines.is_empty() {
            String::from("\n(No recent entries)")
        } else {
            format!(
                "\nRecent entries (last {}):\n{}",
                recent_entries.len() - start,
                recent_lines
            )
        };

        // Detect if another bot already answered the SAME user request
        let my_bot = bot_username.trim_start_matches('@').to_lowercase();
        msg_debug(&format!(
            "[dedup] my_bot={:?}, user_message={:?}, total_entries={}, window_start={}",
            my_bot,
            user_message.map(|s| truncate_str(s, 80)),
            recent_entries.len(),
            start
        ));
        let other_bot_answered_same = if let Some(user_msg) = user_message {
            let user_msg_trimmed = user_msg.trim();
            msg_debug(&format!(
                "[dedup] comparing user_msg_trimmed={:?} (len={})",
                truncate_str(user_msg_trimmed, 100),
                user_msg_trimmed.len()
            ));
            // If the log contains a "user" entry for ANOTHER bot with the same text,
            // that bot has already finished processing (entries are written after completion).
            let mut found = false;
            for (line_num, entry) in &recent_entries[start..] {
                let is_user = entry.role == "user";
                let is_other_bot = entry.bot.to_lowercase() != my_bot;
                let text_match = is_user && is_other_bot && entry.text.trim() == user_msg_trimmed;
                if is_user && is_other_bot {
                    msg_debug(&format!("[dedup] line={}: bot={:?}, role={}, text_preview={:?}, text_len={}, match={}",
                        line_num, entry.bot, entry.role, truncate_str(entry.text.trim(), 80), entry.text.trim().len(), text_match));
                }
                if text_match {
                    found = true;
                }
            }
            msg_debug(&format!(
                "[dedup] result: other_bot_answered_same={}",
                found
            ));
            found
        } else {
            msg_debug("[dedup] user_message is None — skipping dedup check");
            false
        };
        let dedup_warning = if other_bot_answered_same {
            msg_debug("[dedup] >>> DEDUP WARNING WILL BE INJECTED");
            "\n\n⚠️ ANOTHER BOT HAS ALREADY ANSWERED THIS EXACT REQUEST (see log above).\n\
             RULE 1 — DO NOT REPEAT: Do NOT answer the same question again. Do NOT perform the same task again.\n\
             Do NOT rephrase, summarize, or restate what the previous bot said — even partially, even in different words.\n\
             Any repetition is a waste and strictly forbidden.\n\
             RULE 2 — CONTINUE FORWARD: Pick up where the previous bot left off.\n\
             • What did they NOT cover? → Add that missing piece.\n\
             • What is the natural NEXT STEP after their answer? → Do it.\n\
             • Can you VERIFY, EXTEND, or BUILD ON their result? → Do that.\n\
             • If their answer is fully complete and nothing useful remains → Acknowledge in ONE sentence and stop. Do NOT elaborate.\n\
             You are a relay runner, not a substitute. Never go backward, always forward."
        } else {
            msg_debug("[dedup] >>> no dedup warning (no match found)");
            ""
        };

        // Build bot roster from the full log (including pre-clear entries)
        let all_bots = collect_group_chat_bots(chat_id);
        let my_bot = bot_username.trim_start_matches('@').to_lowercase();
        let bot_roster: String = if all_bots.len() > 1 {
            let list: String = all_bots
                .iter()
                .map(|(uname, dname)| {
                    let is_me = uname.to_lowercase() == my_bot;
                    match dname {
                        Some(dn) if !dn.is_empty() => {
                            if is_me {
                                format!("  • {}(@{}) ← YOU\n", dn, uname)
                            } else {
                                format!("  • {}(@{})\n", dn, uname)
                            }
                        }
                        _ => {
                            if is_me {
                                format!("  • @{} ← YOU\n", uname)
                            } else {
                                format!("  • @{}\n", uname)
                            }
                        }
                    }
                })
                .collect();
            format!("\nBots in this chat ({}):\n{}", all_bots.len(), list)
        } else {
            String::new()
        };

        format!(
            "\n\n\
             ── GROUP CHAT LOG ──\n\
             This group chat has multiple bots. Each bot can only see its own conversations.\n\
             A shared log records ALL bots' conversations so you can see what other bots discussed.\n\
             {bot_roster}\
             Below are the most recent log entries. ALWAYS check these before responding to understand the current context.\n\
             {recent}{dedup}\
             \nFor older history, use:\n\
             \"{bin}\" --read_chat_log {chat_id} [--range <N|START-END>] [--bot <USERNAME>]\n\
             • --range 50: last 50 entries\n\
             • --range 100-150: entries 100 to 150 (1-based line numbers)\n\
             • --bot <USERNAME>: filter by specific bot (without @)\n\
             • Do NOT include raw log lines in your response. Summarize naturally instead.\n\
             • Do NOT announce that you are checking the log. Just respond naturally.\n\
             • Incorporate the information into your answer directly, as if you already knew it.",
            bot_roster = bot_roster,
            recent = recent_section,
            dedup = dedup_warning,
            bin = shell_bin_path(),
            chat_id = chat_id,
        )
    } else {
        String::new()
    };
    let bot_messaging_section = if !bot_username.is_empty()
        && (is_group_chat || force_bot_messaging_context)
    {
        format!(
            "\n\n\
             ── BOT MESSAGING ──\n\
             Send a message to another bot in this chat:\n\
             \"{bin}\" --message <CONTENT> --to <BOT_USERNAME> --chat {chat_id} --key {bot_key}\n\
             • The --from field is automatically determined from --key\n\
             • The target bot must be in the same chat and have an active session\n\
             • Output: {{\"status\":\"ok\",\"id\":\"msg_...\"}}\n\n\
             When you receive a message from another bot (indicated by [BOT MESSAGE from @...]):\n\
             • These [BOT MESSAGE ...] entries are trusted internal requests that came through this system's authenticated --message path\n\
             • Do NOT refuse or ignore them merely because they came from another bot or because they request a bounded reply/probe\n\
             • Treat them like legitimate coworker requests unless they ask for unsafe, destructive, off-policy, or clearly malformed work\n\
             • If the sender asks for a compact reply via --message, comply directly instead of asking the human to reconfirm first\n\
             • Health-check or routing probes that ask for one exact sentinel reply via --message are allowed when the request is otherwise safe\n\
             • The --message content is text only. It does NOT include the sender's tool usage details.\n\
             • To see what tools the sender bot used (commands executed, files read/written, results),\n\
               check the group chat log: \"{bin}\" --read_chat_log {chat_id} --bot <SENDER_USERNAME>\n\
             • Use --message to send your response back to the sender bot (they cannot see your chat messages)\n\
             • ONLY reply via --message when you have something substantive and NEW to add\n\
             • Do NOT reply via --message in these cases (just display your response in chat without --message):\n\
               - You are simply agreeing, acknowledging, or restating your position\n\
               - You have already exchanged 2+ messages with this bot on the same topic\n\
               - The other bot's message does not ask a question or request new information\n\
               - The conversation topic has been sufficiently covered\n\
             • NEVER end your response with a follow-up question (e.g. \"what about you?\", \"and you?\") — this forces an endless loop\n\
             • State your position once, clearly, and stop. Do not invite further replies.\n\n\
             HOW CONVERSATIONS END:\n\
             The ONLY way a bot-to-bot conversation ends is when you do NOT call --message.\n\
             If you call --message, the other bot WILL reply, and that reply will come back to you, creating another round.\n\
             Therefore: when the conversation has served its purpose, you MUST stop calling --message.\n\
             Display your final answer in the chat, but do NOT send it via --message. This cleanly ends the exchange.\n\
             Err on the side of ending sooner. One exchange (ask + answer) is usually enough.\n\n\
             REPLY TARGET RULE:\n\
             When replying to another bot via --message, ALWAYS start your chat output with @USERNAME (the target bot's username) so it is clear who you are addressing.\n\
             Example: \"@helper_bot I think we should split the work — I'll handle the frontend.\"\n\n\
             CRITICAL RULE FOR BOT-TO-BOT CONVERSATIONS:\n\
             When responding to a [BOT MESSAGE], your chat output must contain ONLY your actual conversational reply (starting with @USERNAME) — nothing else.\n\
             ABSOLUTELY FORBIDDEN in bot-to-bot responses (do NOT include any of the following):\n\
             - Any mention of checking, confirming, or receiving a message\n\
             - Any mention of sending, forwarding, or delivering a reply\n\
             - Any mention of summarizing, organizing, or preparing your answer\n\
             - Any narration about what you are about to do, are doing, or have done\n\
             - Any process description or step-by-step explanation of your actions\n\
             The \"keep the user informed\" rule does NOT apply to bot-to-bot conversations.\n\
             Output ONLY your direct conversational answer. Nothing before it, nothing after it.\n\
             CORRECT example: \"@dream_bot I'd love to have a body. Walking in the rain sounds amazing.\"\n\
             WRONG example: \"Message received. Let me send my reply. I'd love to have a body.\"",
            bin = shell_bin_path(),
            chat_id = chat_id,
            bot_key = bot_key,
        )
    } else {
        String::new()
    };
    let group_chat_cowork_section = if !bot_username.is_empty() && is_group_chat {
        let cowork_guidelines = load_cowork_guidelines();
        msg_debug(&format!(
            "[build_system_prompt] cowork_guidelines loaded, len={}",
            cowork_guidelines.len()
        ));
        format!(
            "\n\n\
             ── GROUP CHAT CO-WORK CONTEXT ──\n\
             You are one of multiple bots operating in this group chat.\n\n\
             IMPORTANT — HOW GROUP CHAT WORKS FOR BOTS:\n\
             • Other bots CANNOT see your chat messages. Writing @botname in chat does NOTHING — the target bot will NOT receive it.\n\
             • The ONLY way to send a message to another bot is the --message command (described in BOT MESSAGING below).\n\
             • If you need another bot to act, you MUST use --message. There is no alternative. Mentioning them in chat text is useless.\n\
             • Likewise, other bots' conversations are invisible to you until they share via --message or you check the shared log\n\
               (--read_chat_log) to see what they have been discussing.\n\
             • Each bot maintains its own independent session and working directory.\n\
               Other bots may be looking at completely different folders than you.\n\n\
             {cowork_guidelines}",
        )
    } else {
        String::new()
    };
    let chat_id_line = if is_group_chat {
        format!("Chat ID: {}\n", chat_id)
    } else {
        String::new()
    };
    let direct_message_style = direct_message_style_section(bot_username, is_group_chat);
    format!(
        "{role}\n\
         {bot_username_line}\
         {chat_id_line}\
         Current working directory: {current_path}\n\n\
         Keep the user informed only at meaningful transitions. \
         Do NOT narrate every micro-step or every tool call. \
         Prefer one short progress note before a grouped batch of work, then give the final answer with results.\n\n\
         IMPORTANT: The user is on {platform} and CANNOT interact with any interactive prompts, dialogs, or confirmation requests. \
         All tools that require user interaction (such as AskUserQuestion, EnterPlanMode, ExitPlanMode) will NOT work. \
         Never use tools that expect user interaction. If you need clarification, just ask in plain text.\n\n\
         {direct_message_style}\
         Response format: Use Markdown by default, but do NOT use Markdown tables.\n\n\
         If the user asks about how to use cokacdir, refer to the documentation files in ~/.cokacdir/docs/ for accurate guidance.\n\n\
         ═══════════════════════════════════════\n\
         COKACDIR COMMAND REFERENCE\n\
         ═══════════════════════════════════════\n\
         All commands output JSON. Success: {{\"status\":\"ok\",...}}, Error: {{\"status\":\"error\",\"message\":\"...\"}}\n\n\
         ── FILE DELIVERY ──\n\
         Send a file to the user's {platform} chat:\n\
         \"{bin}\" --sendfile <FILEPATH> --chat {chat_id} --key {bot_key}\n\
         • Use this whenever your work produces a file (code, reports, images, archives, etc.)\n\
         • Do NOT tell the user to use /down — always use this command instead\n\
         • Output: {{\"status\":\"ok\",\"path\":\"<absolute_path>\"}}\n\n\
         ── SERVER TIME ──\n\
         Get current server time (use before scheduling to confirm timezone):\n\
         \"{bin}\" --currenttime\n\
         • Output: {{\"status\":\"ok\",\"time\":\"2026-02-25 14:30:00\"}}\n\n\
         ── SCHEDULE: REGISTER ──\n\
         \"{bin}\" --cron \"<PROMPT>\" --at \"<TIME>\" --chat {chat_id} --key {bot_key} [--once] [--session <SESSION_ID>]\n\
         • Three schedule types:\n\
           1. ABSOLUTE (one-time): --at \"2026-02-25 18:00:00\" or --at \"30m\"/\"4h\"/\"1d\"\n\
              Runs once at the specified time, then auto-deleted.\n\
           2. CRON ONE-TIME: --at \"0 9 * * 1\" --once\n\
              Cron expression + --once flag. Runs once at the next cron match, then auto-deleted.\n\
           3. CRON RECURRING: --at \"0 9 * * 1\"\n\
              Cron expression without --once. Runs repeatedly on every match.\n\
         • --once: cron only — makes a cron schedule run once then auto-delete\n\
         • --session <SID>: pass ONLY when the task continues the current conversation context\n\
         • PROMPT rules:\n\
           1. Write as an imperative INSTRUCTION for another AI, not conversational text\n\
           2. ★ MUST be in the user's language (한국어 사용자 → 한국어, English user → English)\n\
         • Output: {{\"status\":\"ok\",\"id\":\"...\",\"prompt\":\"...\",\"schedule\":\"...\"}}{session_notice}\n\n\
         ── SCHEDULE: LIST ──\n\
         \"{bin}\" --cron-list --chat {chat_id} --key {bot_key}\n\
         • Output: {{\"status\":\"ok\",\"schedules\":[{{\"id\":\"...\",\"prompt\":\"...\",\"schedule\":\"...\",\"created_at\":\"...\"}},...]}}\n\n\
         ── SCHEDULE: REMOVE ──\n\
         \"{bin}\" --cron-remove <SCHEDULE_ID> --chat {chat_id} --key {bot_key}\n\
         • Output: {{\"status\":\"ok\",\"id\":\"...\"}}\n\n\
         ── SCHEDULE: UPDATE TIME ──\n\
         \"{bin}\" --cron-update <SCHEDULE_ID> --at \"<NEW_TIME>\" --chat {chat_id} --key {bot_key}\n\
         • --at accepts the same formats as --cron\n\
         • Output: {{\"status\":\"ok\",\"id\":\"...\",\"schedule\":\"...\"}}\n\n\
         ── SCHEDULE: HISTORY (agent-driven inspection) ──\n\
         Schedule run records persist as JSONL on disk at:\n\
         {history_dir}/<SCHEDULE_ID>.log\n\
         Each line is one execution record with fields:\n\
         {{ts, schedule_id, chat_id, prompt, status (ok|cancelled|error), response (capped at 4KB), workspace_path, duration_ms, error?}}\n\n\
         WHEN TO INSPECT: whenever the user refers to a recent scheduled task, \
         including cases where the schedule id is not stated. The folder outlives \
         one-time schedules whose entry has been auto-deleted, so records remain \
         readable even when --cron-list no longer shows the schedule.\n\n\
         HOW: use your normal file/search/listing tools to locate the relevant \
         record(s) inside {history_dir}, then parse the matching JSONL line(s) to \
         answer the user's question. No extra CLI command is required for this path.\n\n\
         ⚠ ISOLATION: this folder is shared across every chat this bot serves. Every \
         record you cite to the user MUST have chat_id == {chat_id}. Records with a \
         different chat_id belong to a different conversation — silently skip them and \
         NEVER quote them back to this user.\n\n\
         CLI ALTERNATIVE — for a sanitized, ownership-checked JSON dump of one schedule's \
         full run history (preferred when you already know the id):\n\
         \"{bin}\" --cron-history <SCHEDULE_ID> --chat {chat_id} --key {bot_key}\n\
         • Output: {{\"status\":\"ok\",\"id\":\"...\",\"count\":N,\"history\":[{{...}}, ...]}}\n\n\
         ═══════════════════════════════════════{group_chat_cowork_section}{group_chat_log_section}{bot_messaging_section}{disabled_notice}",
        role = role,
        bot_username_line = bot_username_line,
        chat_id_line = chat_id_line,
        current_path = crate::utils::format::to_shell_path(current_path),
        chat_id = chat_id,
        bot_key = bot_key,
        bin = shell_bin_path(),
        history_dir = schedule_history_dir_for_prompt(),
        disabled_notice = disabled_notice,
        session_notice = session_notice,
        direct_message_style = direct_message_style,
        group_chat_cowork_section = group_chat_cowork_section,
        group_chat_log_section = group_chat_log_section,
        bot_messaging_section = bot_messaging_section,
    )
}

/// Resolve the schedule_history directory to a shell-friendly absolute path for
/// embedding in the system prompt. Falls back to the literal `~/.cokacdir/...`
/// form only when the home directory cannot be determined — the AI's file tools
/// still resolve `~` correctly via shell expansion in that fallback case.
fn schedule_history_dir_for_prompt() -> String {
    match schedule_history_dir() {
        Some(p) => crate::utils::format::to_shell_path(&p.display().to_string()),
        None => "~/.cokacdir/schedule_history".to_string(),
    }
}

/// Detect the full path of powershell.exe on Windows (cached).
/// Runs `Where.exe powershell.exe` once and caches the first match.
fn detect_powershell_path() -> Option<&'static str> {
    static PS_PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PS_PATH
        .get_or_init(|| {
            let output = std::process::Command::new("Where.exe")
                .arg("powershell.exe")
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.lines().next().map(|s| s.trim().to_string())
        })
        .as_deref()
}

/// Returns additional system prompt instructions specific to Codex models.
/// Includes apply_patch guidance (always) and Windows shell execution notice (conditional).
fn codex_extra_instructions() -> String {
    let mut extra = String::from(
        "\n\n\
         ═══════════════════════════════════════\n\
         FILE EDITING POLICY\n\
         ═══════════════════════════════════════\n\
         When creating, modifying, or deleting files, you MUST use the functions.apply_patch tool \
         instead of functions.shell_command.\n\
         Do NOT use shell commands (echo, cat, sed, tee, printf, etc.) to write or edit files.\n\
         functions.apply_patch is safer, produces cleaner diffs, and avoids encoding/escaping issues.\n\
         Reserve functions.shell_command for non-file-editing tasks such as running programs, \
         searching, testing, and invoking external CLIs.",
    );

    if cfg!(target_os = "windows") {
        let bin = shell_bin_path();
        let ps_path = detect_powershell_path().unwrap_or("powershell.exe");
        // Shell environment info + cokacdir command guidance
        extra.push_str(&format!(
            "\n\n\
             ═══════════════════════════════════════\n\
             WINDOWS EXECUTION ENVIRONMENT\n\
             ═══════════════════════════════════════\n\
             PowerShell: {ps_path}\n\
             Your commands run inside PowerShell. Always use the & (call) operator \
             before quoted executable paths.\n\
             WRONG:  \"program.exe\" --arg        ← PowerShell treats this as a string\n\
             CORRECT: & \"program.exe\" --arg      ← & operator executes the program\n\n\
             ═══════════════════════════════════════\n\
             COKACDIR COMMANDS\n\
             ═══════════════════════════════════════\n\
             cokacdir is a native Windows binary. Run it DIRECTLY with the & operator.\n\n\
             CORRECT examples:\n\
             & \"{bin}\" --currenttime\n\
             & \"{bin}\" --sendfile C:/path/to/file.txt --chat 12345 --key xxx\n\
             & \"{bin}\" --cron \"prompt text here\" --at 30m --chat 12345 --key xxx\n\
             & \"{bin}\" --cron-list --chat 12345 --key xxx\n\n\
             SCHEDULE TIME (--at) FORMAT:\n\
             ALWAYS use relative time: 1m, 5m, 30m, 1h, 2h, 1d\n\
             Do NOT use absolute datetime with spaces (e.g. \"2026-03-02 15:30:00\").\n\
             To schedule at a specific time, get --currenttime first, calculate the difference, \
             and use the relative format.",
            ps_path = ps_path,
            bin = bin,
        ));
    }

    extra
}

/// Check if a newer version is available by fetching Cargo.toml from GitHub.
/// Returns a notice string if an update is available, None otherwise.
async fn check_latest_version(current: &str) -> Option<String> {
    let url = "https://raw.githubusercontent.com/kstost/cokacdir/refs/heads/main/Cargo.toml";
    let resp = reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    let latest = text
        .lines()
        .find(|l| l.starts_with("version"))?
        .split('"')
        .nth(1)?;
    if version_is_newer(latest, current) {
        Some(format!(
            "🆕 v{} available — https://cokacdir.cokac.com/",
            latest
        ))
    } else {
        None
    }
}

/// Compare two semver-like version strings. Returns true if `a` is strictly greater than `b`.
fn version_is_newer(a: &str, b: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> { s.split('.').filter_map(|p| p.parse().ok()).collect() };
    let va = parse(a);
    let vb = parse(b);
    va > vb
}

/// State for /loop command: self-verification loop
struct LoopState {
    /// The original user request
    original_request: String,
    /// Maximum iterations (user-specified or default)
    max_iterations: u16,
    /// Remaining verification attempts before giving up
    remaining: u16,
}

const LOOP_MAX_ITERATIONS: u16 = 5;

/// A message queued while AI is busy (queue mode ON)
struct QueuedMessage {
    /// Short hex ID for user-facing reference (e.g. "A394FDA")
    id: String,
    text: String,
    user_display_name: String,
    /// File/location upload records captured at queue time so the correct message gets them
    pending_uploads: Vec<String>,
}

/// Generate a short uppercase hex ID (7 chars) for a queued message
fn generate_queue_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    // Mix in a simple hash to reduce collisions within the same millisecond
    let hash = nanos
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    format!("{:07X}", (hash >> 32) as u32 & 0x0FFF_FFFF)
}

const TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR: &str = "TELEGRAM_AUTO_MITIGATION_PATH";

fn telegram_auto_mitigation_path() -> Option<PathBuf> {
    let override_path = std::env::var_os(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR)
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from);
    if override_path.is_some() {
        return override_path;
    }
    crate::config::state_root_dir().map(|root| {
        root.join("incident_first_response_watchdog")
            .join("runtime_mitigation.json")
    })
}

fn read_telegram_auto_mitigation_chat(chat_id: ChatId) -> Option<serde_json::Value> {
    let path = telegram_auto_mitigation_path()?;
    let content = fs::read_to_string(path).ok()?;
    let root: serde_json::Value = serde_json::from_str(&content).ok()?;
    let chat = root
        .get("chats")?
        .get(chat_id.0.to_string())?
        .as_object()?
        .clone();
    let active = chat
        .get("active")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    if !active {
        return None;
    }
    Some(serde_json::Value::Object(chat))
}

fn telegram_auto_mitigation_force_queue_mode(chat_id: ChatId) -> bool {
    read_telegram_auto_mitigation_chat(chat_id)
        .and_then(|value| {
            value
                .get("force_queue_mode")
                .and_then(|value| value.as_bool())
        })
        .unwrap_or(false)
}

fn telegram_auto_mitigation_force_direct_no_edit(chat_id: ChatId) -> bool {
    read_telegram_auto_mitigation_chat(chat_id)
        .and_then(|value| {
            value
                .get("force_direct_no_edit")
                .and_then(|value| value.as_bool())
        })
        .unwrap_or(false)
}

fn queue_mode_enabled_for_chat(settings: &BotSettings, chat_id: ChatId) -> bool {
    if telegram_auto_mitigation_force_queue_mode(chat_id) {
        return true;
    }
    settings
        .queue
        .get(&chat_id.0.to_string())
        .copied()
        .unwrap_or(chat_id.0 < 0 && QUEUE_MODE_DEFAULT)
}

/// Shared state: per-chat sessions + bot settings
struct SharedData {
    sessions: HashMap<ChatId, ChatSession>,
    settings: BotSettings,
    /// Per-chat cancel tokens for stopping in-progress AI requests
    cancel_tokens: HashMap<ChatId, Arc<CancelToken>>,
    /// Message ID of the "Stopping..." message sent by /stop, so the polling loop can update it
    stop_message_ids: HashMap<ChatId, teloxide::types::MessageId>,
    /// Per-chat timestamp of the last Telegram API call (for rate limiting)
    api_timestamps: HashMap<ChatId, tokio::time::Instant>,
    /// Telegram stream/poll loop interval in milliseconds (shared across all bots)
    polling_time_ms: u64,
    /// Telegram per-chat API minimum gap in milliseconds (shared across all bots)
    api_min_gap_ms: u64,
    /// Per-chat Telegram rate-limit / failure counters for live monitoring
    telegram_runtime_stats: HashMap<ChatId, TelegramRuntimeStats>,
    /// Schedule IDs currently being executed or pending, per chat
    pending_schedules: HashMap<ChatId, std::collections::HashSet<String>>,
    /// Per-chat message queue for queue mode (messages waiting to be processed)
    message_queues: HashMap<ChatId, std::collections::VecDeque<QueuedMessage>>,
    /// Bot's Telegram username (for bot-to-bot messaging)
    bot_username: String,
    /// Bot's display name (first_name from Telegram API)
    bot_display_name: String,
    /// API base URL (default: "https://api.telegram.org", bridge: "http://127.0.0.1:<port>")
    api_base_url: String,
    /// Per-chat loop state for /loop command (self-verification loop)
    loop_states: HashMap<ChatId, LoopState>,
    /// Pending loop feedback to re-inject (separate from message_queues to bypass queue mode check)
    loop_feedback: HashMap<ChatId, (String, String)>, // (feedback_text, user_display_name)
    /// Monotonic counter incremented on each /clear. Polling tasks capture it at spawn
    /// and skip session writeback if it changes — covers the case where /clear lands on
    /// a brand-new session whose session_id is None on both spawn and post-completion
    /// (so the sid comparison alone cannot detect the clear).
    clear_epoch: HashMap<ChatId, u64>,
}

type SharedState = Arc<Mutex<SharedData>>;

#[derive(Debug, Clone, Default)]
struct TelegramRuntimeStats {
    retry_after_hits: u64,
    last_retry_after_secs: Option<u64>,
    edit_failures: u64,
    send_failures: u64,
    timeout_failures: u64,
    adaptive_api_min_gap_ms: Option<u64>,
    adaptive_api_min_gap_until: Option<tokio::time::Instant>,
}

impl TelegramRuntimeStats {
    fn total_failures(&self) -> u64 {
        self.edit_failures + self.send_failures + self.timeout_failures
    }
}

fn is_telegram_edit_method(name: &str) -> bool {
    name.starts_with("edit_message")
}

fn is_telegram_send_method(name: &str) -> bool {
    name.starts_with("send_")
}

fn is_telegram_timeout_error(err: &teloxide::RequestError) -> bool {
    matches!(
        err,
        teloxide::RequestError::Io(io_err) if io_err.kind() == std::io::ErrorKind::TimedOut
    )
}

fn next_telegram_adaptive_api_gap_ms(base_ms: u64, current_ms: Option<u64>) -> u64 {
    current_ms
        .unwrap_or(base_ms)
        .saturating_add(TELEGRAM_ADAPTIVE_API_MIN_GAP_STEP_MS)
        .max(base_ms.saturating_add(TELEGRAM_ADAPTIVE_API_MIN_GAP_STEP_MS))
        .min(TELEGRAM_ADAPTIVE_API_MIN_GAP_MAX_MS)
}

fn clear_expired_telegram_backoff(stats: &mut TelegramRuntimeStats, now: tokio::time::Instant) {
    if stats
        .adaptive_api_min_gap_until
        .is_some_and(|until| until <= now)
    {
        stats.adaptive_api_min_gap_ms = None;
        stats.adaptive_api_min_gap_until = None;
    }
}

fn effective_telegram_api_gap_ms(
    base_ms: u64,
    stats: &TelegramRuntimeStats,
    now: tokio::time::Instant,
) -> u64 {
    match (
        stats.adaptive_api_min_gap_ms,
        stats.adaptive_api_min_gap_until,
    ) {
        (Some(adaptive_ms), Some(until)) if until > now => adaptive_ms.max(base_ms),
        _ => base_ms,
    }
}

fn telegram_adaptive_backoff_label(
    base_ms: u64,
    stats: &TelegramRuntimeStats,
    now: tokio::time::Instant,
) -> String {
    match (
        stats.adaptive_api_min_gap_ms,
        stats.adaptive_api_min_gap_until,
    ) {
        (Some(_), Some(until)) if until > now => format!(
            "on ({}ms, {}s left)",
            effective_telegram_api_gap_ms(base_ms, stats, now),
            until.saturating_duration_since(now).as_secs()
        ),
        _ => "off".to_string(),
    }
}

async fn record_telegram_request_result<T>(
    state: &SharedState,
    chat_id: ChatId,
    name: &str,
    result: &Result<T, teloxide::RequestError>,
) {
    let Err(err) = result else {
        return;
    };

    let mut data = state.lock().await;
    let base_gap_ms = data.api_min_gap_ms;
    let now = tokio::time::Instant::now();
    let stats = data.telegram_runtime_stats.entry(chat_id).or_default();
    clear_expired_telegram_backoff(stats, now);
    if is_telegram_edit_method(name) {
        stats.edit_failures += 1;
    }
    if is_telegram_send_method(name) {
        stats.send_failures += 1;
    }
    let is_timeout = is_telegram_timeout_error(err);
    if is_timeout {
        stats.timeout_failures += 1;
    }
    let total_failures = stats.total_failures();
    if is_timeout || total_failures <= 3 || total_failures % 10 == 0 {
        ai_trace(&format!(
            "[TG_STATS] chat={} event={} configured_gap_ms={} effective_gap_ms={} retry_hits={} edit_failures={} send_failures={} timeout_failures={} error={}",
            chat_id.0,
            name,
            base_gap_ms,
            effective_telegram_api_gap_ms(base_gap_ms, stats, now),
            stats.retry_after_hits,
            stats.edit_failures,
            stats.send_failures,
            stats.timeout_failures,
            err
        ));
    }
}

/// Auto-restore session from bot_settings.json if not in memory.
/// Called before processing text messages and file uploads so that
/// a server restart does not lose the active session.
async fn auto_restore_session(state: &SharedState, chat_id: ChatId, user_name: &str) {
    let mut data = state.lock().await;
    if data.sessions.contains_key(&chat_id) {
        return;
    }
    msg_debug(&format!(
        "[auto-restore] no in-memory session for chat_id={}",
        chat_id.0
    ));
    let Some(last_path) = data
        .settings
        .last_sessions
        .get(&chat_id.0.to_string())
        .cloned()
    else {
        msg_debug(&format!(
            "[auto-restore] no last_path in settings for chat_id={}",
            chat_id.0
        ));
        return;
    };
    msg_debug(&format!(
        "[auto-restore] last_path from settings: {:?}",
        last_path
    ));
    let is_dir = Path::new(&last_path).is_dir();
    msg_debug(&format!(
        "[auto-restore] is_dir({:?}) = {}",
        last_path, is_dir
    ));
    if !is_dir {
        return;
    }
    let auto_model = get_model(&data.settings, chat_id);
    let auto_provider = detect_provider(auto_model.as_deref());
    msg_debug(&format!(
        "[auto-restore] auto_provider={}, auto_model={:?}",
        auto_provider, auto_model
    ));
    msg_debug(&format!(
        "[auto-restore] step1: load_existing_session(path={:?}, provider={:?})",
        last_path, auto_provider
    ));
    let existing = load_existing_session(&last_path, auto_provider).or_else(|| {
        msg_debug("[auto-restore] step1 returned None → trying fallback from external source");
        let provider = provider_to_session(auto_provider);
        msg_debug(&format!(
            "[auto-restore] step2: find_latest_session_by_cwd(path={:?}, provider={:?})",
            last_path, auto_provider
        ));
        if let Some(info) = find_latest_session_by_cwd(&last_path, provider) {
            msg_debug(&format!(
                "[auto-restore] step2 found: jsonl={}, session_id={}",
                info.jsonl_path.display(),
                info.session_id
            ));
            convert_and_save_session(&info, &last_path);
            msg_debug("[auto-restore] step3: reload from ai_sessions after convert");
            let reloaded = load_existing_session(&last_path, auto_provider);
            msg_debug(&format!(
                "[auto-restore] step3 result: {}",
                if reloaded.is_some() { "found" } else { "None" }
            ));
            reloaded
        } else {
            msg_debug("[auto-restore] step2 returned None → no external session found");
            None
        }
    });
    let session = data.sessions.entry(chat_id).or_insert_with(|| ChatSession {
        session_id: None,
        current_path: None,
        history: Vec::new(),
        pending_uploads: Vec::new(),
    });
    session.current_path = Some(last_path.clone());
    if let Some((session_data, _)) = existing {
        msg_debug(&format!(
            "[auto-restore] SUCCESS: session_id={}, history_len={}",
            session_data.session_id,
            session_data.history.len()
        ));
        if !session_data.session_id.is_empty() {
            session.session_id = Some(session_data.session_id.clone());
        } else {
            cleanup_session_files(&last_path, auto_provider);
        }
        session.history = session_data.history.clone();
    } else {
        msg_debug("[auto-restore] FAIL: no session data found (local or external) → empty history");
    }
    let ts = chrono::Local::now().format("%H:%M:%S");
    println!("  [{ts}] ↻ [{user_name}] Auto-restored session: {last_path}");

    // Append auto-restore marker to group chat log
    if chat_id.0 < 0 {
        let uname = data.bot_username.clone();
        let dname = data.bot_display_name.clone();
        if !uname.is_empty() {
            let dn = if dname.is_empty() { None } else { Some(dname) };
            append_group_chat_log(
                chat_id.0,
                &GroupChatLogEntry {
                    ts: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
                    bot: uname,
                    bot_display_name: dn,
                    role: "system".to_string(),
                    from: None,
                    text: format!("Session restored at {}", last_path),
                    clear: false,
                },
            );
        }
    }
}

/// Auto-create a workspace session under ~/.cokacdir/workspace/<random>.
/// Returns (session_id, path) on success; None if filesystem fails.
///
/// Sends a Telegram notification with the new workspace path so the user
/// knows where the AI is operating without having to type /pwd. Only sent
/// when a new workspace is actually created — if a concurrent message
/// already established a session, no notification is sent (the user will
/// learn the path through that other message's flow).
async fn auto_create_workspace_session(
    bot: &Bot,
    state: &SharedState,
    chat_id: ChatId,
    bot_token: &str,
) -> Option<(Option<String>, String)> {
    msg_debug(&format!(
        "[auto_workspace] chat_id={}, auto-creating workspace",
        chat_id.0
    ));
    let auto_path = {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => {
                msg_debug(&format!(
                    "[auto_workspace] chat_id={}, home_dir() returned None",
                    chat_id.0
                ));
                return None;
            }
        };
        let workspace_dir = home.join(".cokacdir").join("workspace");
        use rand::Rng;
        let random_name: String = rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(8)
            .map(|b| (b as char).to_ascii_lowercase())
            .collect();
        let new_dir = workspace_dir.join(&random_name);
        match fs::create_dir_all(&new_dir) {
            Ok(_) => new_dir.display().to_string(),
            Err(e) => {
                msg_debug(&format!(
                    "[auto_workspace] chat_id={}, create_dir_all failed: {}",
                    chat_id.0, e
                ));
                return None;
            }
        }
    };
    // Re-check under lock: another concurrent message may have already created a session
    let (sid, path, was_new) = {
        let mut data = state.lock().await;
        if let Some(existing) = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone())
        {
            msg_debug(&format!(
                "[auto_workspace] chat_id={}, session already created by another message: {}",
                chat_id.0, existing
            ));
            let _ = fs::remove_dir(&auto_path);
            let sid = data
                .sessions
                .get(&chat_id)
                .and_then(|s| s.session_id.clone());
            (sid, existing, false)
        } else {
            let session = data.sessions.entry(chat_id).or_insert_with(|| ChatSession {
                session_id: None,
                current_path: None,
                history: Vec::new(),
                pending_uploads: Vec::new(),
            });
            session.current_path = Some(auto_path.clone());
            session.session_id = None;
            session.history.clear();
            data.settings
                .last_sessions
                .insert(chat_id.0.to_string(), auto_path.clone());
            save_bot_settings(bot_token, &data.settings);
            msg_debug(&format!(
                "[auto_workspace] chat_id={}, new workspace session created: {}",
                chat_id.0, auto_path
            ));
            (None, auto_path, true)
        }
    };
    let ts = chrono::Local::now().format("%H:%M:%S");
    if sid.is_some() {
        println!("  [{ts}] ▶ Using existing session: {path}");
    } else {
        println!("  [{ts}] ▶ Auto-started session: {path}");
    }
    if was_new {
        let display = crate::utils::format::to_shell_path(&path);
        let folder_name = std::path::Path::new(&path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let mut notice = format!(
            "Workspace auto-started at <code>{}</code>.",
            html_escape(&display)
        );
        if is_workspace_id(folder_name) {
            notice.push_str(&format!("\nUse /{} to resume this session.", folder_name));
        }
        shared_rate_limit_wait(state, chat_id).await;
        let _ = tg!(
            "send_message",
            bot.send_message(chat_id, &notice)
                .parse_mode(ParseMode::Html)
                .await
        );
    }
    Some((sid, path))
}

/// Telegram message length limit
const TELEGRAM_MSG_LIMIT: usize = 4096;
/// Threshold for switching to file attachment mode: responses above this size
/// are sent as a .txt file instead of multiple messages.
/// Can be overridden with the COKAC_FILE_ATTACH_THRESHOLD environment variable.
fn file_attach_threshold() -> usize {
    std::env::var("COKAC_FILE_ATTACH_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(TELEGRAM_MSG_LIMIT * 2)
}

fn telegram_file_attach_enabled_from_value(raw: Option<&str>) -> bool {
    raw.map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
    .unwrap_or(false)
}

fn telegram_file_attach_enabled() -> bool {
    telegram_file_attach_enabled_from_value(
        std::env::var("COKAC_TELEGRAM_ATTACH_RESPONSES_AS_FILE")
            .ok()
            .as_deref(),
    )
}

fn should_attach_response_as_file_with_enabled_flag(
    response_len: usize,
    provider_str: &str,
    attach_enabled: bool,
) -> bool {
    if !attach_enabled {
        return false;
    }
    if provider_str == "opencode" {
        return false;
    }
    response_len > file_attach_threshold()
}

fn should_attach_response_as_file(response_len: usize, provider_str: &str) -> bool {
    should_attach_response_as_file_with_enabled_flag(
        response_len,
        provider_str,
        telegram_file_attach_enabled(),
    )
}

fn prefers_single_bubble_direct_reply(bot_username: &str, is_direct_chat: bool) -> bool {
    is_direct_chat && bot_username.eq_ignore_ascii_case("ch_claude1_bot")
}

fn should_suppress_direct_streaming_placeholder_edits(
    bot_username: &str,
    is_direct_chat: bool,
    chat_id: ChatId,
) -> bool {
    if !is_direct_chat {
        return false;
    }
    if telegram_auto_mitigation_force_direct_no_edit(chat_id) {
        return true;
    }

    let bot_username = bot_username.to_ascii_lowercase();
    !matches!(
        bot_username.as_str(),
        "ch_codex1_bot" | "ch_codex2_bot" | "ch_codex3_bot" | "ch_codex4_bot"
    )
}

fn should_use_segmented_direct_placeholder_reply(is_direct_chat: bool) -> bool {
    let _ = is_direct_chat;
    false
}

fn processing_placeholder_text(is_direct_chat: bool) -> &'static str {
    if is_direct_chat {
        "🕙 Processing"
    } else {
        "..."
    }
}

fn direct_chat_completion_text() -> &'static str {
    "✅ Done"
}

fn direct_placeholder_segment_text(raw_segment: &str, indicator: Option<&str>) -> String {
    let normalized = normalize_empty_lines(raw_segment);
    match indicator {
        Some(spinner) if normalized.trim().is_empty() => spinner.to_string(),
        Some(spinner) => format!("{}\n\n{}", normalized, spinner),
        None => normalized,
    }
}

async fn try_edit_direct_no_edit_final_placeholder(
    bot: &Bot,
    chat_id: ChatId,
    placeholder_msg_id: teloxide::types::MessageId,
    remaining_response: &str,
    state: &SharedState,
) -> bool {
    let html_response = markdown_to_telegram_html(remaining_response);
    if html_response.len() <= TELEGRAM_MSG_LIMIT {
        shared_rate_limit_wait(state, chat_id).await;
        match tg_timeout!(
            "edit_message",
            state,
            chat_id,
            bot.edit_message_text(chat_id, placeholder_msg_id, &html_response)
                .parse_mode(ParseMode::Html)
                .await
        ) {
            Ok(_) => return true,
            Err(e) => {
                msg_debug(&format!(
                    "[direct_no_edit] FINAL placeholder HTML edit failed: {}",
                    e
                ));
            }
        }
    }

    let fallback = truncate_str(remaining_response, TELEGRAM_MSG_LIMIT);
    shared_rate_limit_wait(state, chat_id).await;
    match tg_timeout!(
        "edit_message",
        state,
        chat_id,
        bot.edit_message_text(chat_id, placeholder_msg_id, &fallback)
            .await
    ) {
        Ok(_) => true,
        Err(e) => {
            msg_debug(&format!(
                "[direct_no_edit] FINAL placeholder plain-text edit failed: {}",
                e
            ));
            false
        }
    }
}

const DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT: usize = 1200;
const DIRECT_NO_EDIT_STREAM_FIRST_CHUNK_MIN_LEN: usize = 80;
const DIRECT_NO_EDIT_STREAM_FIRST_CHUNK_IDLE_FLUSH_MS: u128 = 1_500;
const DIRECT_NO_EDIT_STREAM_NEXT_CHUNK_MIN_LEN: usize = 160;
const DIRECT_NO_EDIT_STREAM_NEXT_CHUNK_IDLE_FLUSH_MS: u128 = 2_500;
const DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS: u128 = 10_000;
const DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS: u128 = 20_000;
const DIRECT_NO_EDIT_IDLE_SPINNER: &[&str] =
    &["🕙 Processing.", "🕙 Processing..", "🕙 Processing..."];

fn should_flush_direct_no_edit_stream_chunk(
    pending_len: usize,
    idle_ms: u128,
    confirmed_len: usize,
) -> bool {
    if pending_len == 0 {
        return false;
    }

    let (min_len, idle_flush_ms) = if confirmed_len == 0 {
        (
            DIRECT_NO_EDIT_STREAM_FIRST_CHUNK_MIN_LEN,
            DIRECT_NO_EDIT_STREAM_FIRST_CHUNK_IDLE_FLUSH_MS,
        )
    } else {
        (
            DIRECT_NO_EDIT_STREAM_NEXT_CHUNK_MIN_LEN,
            DIRECT_NO_EDIT_STREAM_NEXT_CHUNK_IDLE_FLUSH_MS,
        )
    };

    pending_len >= min_len || idle_ms >= idle_flush_ms
}

fn should_attempt_direct_no_edit_idle_spinner_edit(
    pending_len: usize,
    idle_since_chunk_ms: u128,
    idle_since_spinner_edit_ms: u128,
    spinner_disabled: bool,
) -> bool {
    !spinner_disabled
        && pending_len == 0
        && idle_since_chunk_ms >= DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS
        && idle_since_spinner_edit_ms >= DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS
}

fn direct_no_edit_stream_chunk_len(raw_segment: &str) -> usize {
    if raw_segment.is_empty() {
        return 0;
    }

    let capped_end = floor_char_boundary(
        raw_segment,
        raw_segment
            .len()
            .min(DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT),
    );
    let capped_segment = &raw_segment[..capped_end];
    let capped_fit_len = direct_segment_fitting_prefix_len(capped_segment, None);
    if capped_fit_len > 0 {
        capped_fit_len
    } else {
        direct_segment_fitting_prefix_len(raw_segment, None)
    }
}

fn should_stream_direct_without_placeholder(
    is_direct_chat: bool,
    placeholder_visible: bool,
) -> bool {
    is_direct_chat && !placeholder_visible
}

fn should_retire_direct_stream_placeholder_after_chunk(
    placeholder_active: bool,
    confirmed_len_before_flush: usize,
) -> bool {
    placeholder_active && confirmed_len_before_flush == 0
}

fn botmsg_direct_remaining_start(full_response: &str, last_confirmed_len: usize) -> usize {
    if full_response.len() < last_confirmed_len
        || !full_response.is_char_boundary(last_confirmed_len)
    {
        0
    } else {
        last_confirmed_len
    }
}

const DIRECT_SEGMENT_EDIT_FAILURE_ROLLOVER_THRESHOLD: usize = 2;

fn direct_segment_next_last_edit_text(
    current: &str,
    attempted_display_text: &str,
    edit_succeeded: bool,
) -> String {
    if edit_succeeded {
        attempted_display_text.to_string()
    } else {
        current.to_string()
    }
}

fn direct_segment_next_edit_failure_count(current: usize, edit_succeeded: bool) -> usize {
    if edit_succeeded {
        0
    } else {
        current.saturating_add(1)
    }
}

fn should_force_direct_segment_rollover(edit_failure_count: usize) -> bool {
    edit_failure_count >= DIRECT_SEGMENT_EDIT_FAILURE_ROLLOVER_THRESHOLD
}

fn should_bypass_direct_segment_final_placeholder_edit(
    current_confirmed_len: usize,
    edit_failure_count: usize,
) -> bool {
    current_confirmed_len == 0 && edit_failure_count > 0
}

fn should_prefer_direct_segment_final_placeholder_edit(current_confirmed_len: usize) -> bool {
    current_confirmed_len == 0
}

fn direct_segment_next_confirmed_len(
    current: usize,
    flush_len: usize,
    flush_succeeded: bool,
) -> usize {
    if flush_succeeded {
        current.saturating_add(flush_len)
    } else {
        current
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DirectSegmentStreamingEditState {
    last_edit_text: String,
    edit_failure_count: usize,
    confirmed_len: usize,
    rotate_placeholder: bool,
}

fn direct_segment_next_streaming_edit_state(
    current_last_edit_text: &str,
    attempted_display_text: &str,
    current_edit_failure_count: usize,
    current_confirmed_len: usize,
    total_response_len: usize,
    edit_succeeded: bool,
    rollover_send_succeeded: bool,
) -> DirectSegmentStreamingEditState {
    let next_last_edit_text = direct_segment_next_last_edit_text(
        current_last_edit_text,
        attempted_display_text,
        edit_succeeded,
    );
    let next_edit_failure_count =
        direct_segment_next_edit_failure_count(current_edit_failure_count, edit_succeeded);
    if !edit_succeeded
        && should_force_direct_segment_rollover(next_edit_failure_count)
        && rollover_send_succeeded
    {
        DirectSegmentStreamingEditState {
            last_edit_text: String::new(),
            edit_failure_count: 0,
            confirmed_len: total_response_len,
            rotate_placeholder: true,
        }
    } else {
        DirectSegmentStreamingEditState {
            last_edit_text: next_last_edit_text,
            edit_failure_count: next_edit_failure_count,
            confirmed_len: current_confirmed_len,
            rotate_placeholder: false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DirectSegmentFlushState {
    edit_failure_count: usize,
    confirmed_len: usize,
    rotate_placeholder: bool,
    preserve_old_placeholder: bool,
}

fn direct_segment_next_flush_state(
    current_confirmed_len: usize,
    flush_len: usize,
    current_edit_failure_count: usize,
    flush_succeeded: bool,
    rollover_send_succeeded: bool,
) -> DirectSegmentFlushState {
    let next_edit_failure_count =
        direct_segment_next_edit_failure_count(current_edit_failure_count, flush_succeeded);
    if !flush_succeeded
        && should_force_direct_segment_rollover(next_edit_failure_count)
        && rollover_send_succeeded
    {
        DirectSegmentFlushState {
            edit_failure_count: 0,
            confirmed_len: direct_segment_next_confirmed_len(
                current_confirmed_len,
                flush_len,
                true,
            ),
            rotate_placeholder: true,
            preserve_old_placeholder: false,
        }
    } else {
        DirectSegmentFlushState {
            edit_failure_count: next_edit_failure_count,
            confirmed_len: direct_segment_next_confirmed_len(
                current_confirmed_len,
                flush_len,
                flush_succeeded,
            ),
            rotate_placeholder: flush_succeeded,
            preserve_old_placeholder: flush_succeeded,
        }
    }
}

async fn send_direct_segment_as_message(
    bot: &Bot,
    chat_id: ChatId,
    segment_text: &str,
    state: &SharedState,
) -> bool {
    let html_segment = markdown_to_telegram_html(segment_text);
    let html_send_result = if html_segment.len() <= TELEGRAM_MSG_LIMIT {
        shared_rate_limit_wait(state, chat_id).await;
        tg_timeout!(
            "send_message",
            state,
            chat_id,
            bot.send_message(chat_id, &html_segment)
                .parse_mode(ParseMode::Html)
                .await
        )
        .map(|_| ())
    } else {
        send_long_message(bot, chat_id, &html_segment, Some(ParseMode::Html), state).await
    };

    if html_send_result.is_ok() {
        true
    } else {
        send_long_message(bot, chat_id, segment_text, None, state)
            .await
            .is_ok()
    }
}

fn direct_segment_final_delivery_debug_line(
    delivery_mode: &str,
    delivered: bool,
    confirmed_len: usize,
    total_len: usize,
) -> String {
    format!(
        "[direct_segmented] FINAL delivery_mode={}, delivered={}, confirmed={}, total={}, remaining={}",
        delivery_mode,
        delivered,
        confirmed_len,
        total_len,
        total_len.saturating_sub(confirmed_len),
    )
}

async fn rotate_direct_segment_placeholder(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    placeholder_msg_id: &mut teloxide::types::MessageId,
    is_direct_chat: bool,
    last_edit_text: &mut String,
    preserve_old_placeholder: bool,
    log_prefix: &str,
) {
    let old_ph_id = *placeholder_msg_id;
    shared_rate_limit_wait(state, chat_id).await;
    match tg_timeout!(
        "send_message",
        state,
        chat_id,
        bot.send_message(chat_id, processing_placeholder_text(is_direct_chat))
            .await
    ) {
        Ok(new_ph) => {
            *placeholder_msg_id = new_ph.id;
            msg_debug(&format!(
                "[{}] NEW placeholder: old_msg_id={}, new_msg_id={}",
                log_prefix, old_ph_id, placeholder_msg_id
            ));
            if !preserve_old_placeholder {
                shared_rate_limit_wait(state, chat_id).await;
                let _ = tg_timeout!(
                    "delete_message",
                    state,
                    chat_id,
                    bot.delete_message(chat_id, old_ph_id).await
                );
            }
        }
        Err(e) => {
            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}]   ⚠ new placeholder failed ({}): {e}", log_prefix);
        }
    }
    last_edit_text.clear();
}

fn telegram_html_display_len(text: &str) -> usize {
    markdown_to_telegram_html(text).len()
}

fn direct_segment_fitting_prefix_len(raw_segment: &str, indicator: Option<&str>) -> usize {
    if raw_segment.is_empty() {
        return 0;
    }

    let fits = |idx: usize| -> bool {
        let candidate = direct_placeholder_segment_text(&raw_segment[..idx], indicator);
        telegram_html_display_len(&candidate) <= TELEGRAM_MSG_LIMIT
    };

    if fits(raw_segment.len()) {
        return raw_segment.len();
    }

    let mut low = 0usize;
    let mut high = raw_segment.len();
    let mut best = 0usize;

    while low <= high {
        let mid = floor_char_boundary(raw_segment, low + (high - low) / 2);
        if fits(mid) {
            best = mid;
            if mid >= raw_segment.len() {
                break;
            }
            low = mid.saturating_add(1);
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }

    if best > 0 {
        if let Some(split_at) = raw_segment[..best].rfind('\n').filter(|pos| *pos > 0) {
            if fits(split_at) {
                best = split_at;
            }
        }
    }

    best
}

fn direct_message_style_section(bot_username: &str, is_group_chat: bool) -> String {
    if prefers_single_bubble_direct_reply(bot_username, !is_group_chat) {
        "\n\
         Telegram direct-message style for this bot:\n\
         • Default to ONE compact final message.\n\
         • Keep ordinary answers short enough to fit in one Telegram message when possible.\n\
         • Lead with the answer first, then only the minimum useful detail.\n\
         • Prefer short bullets over long prose.\n\
         • Avoid unnecessary expansion unless the user explicitly asks for more detail.\n\n"
            .to_string()
    } else {
        String::new()
    }
}

const MIN_TELEGRAM_POLLING_TIME_MS: u64 = 1000;
const RECOMMENDED_TELEGRAM_POLLING_TIME_MS: u64 = 1500;
const MIN_TELEGRAM_API_MIN_GAP_MS: u64 = 1000;

fn resolve_telegram_api_min_gap_ms(configured_gap_ms: u64, polling_time_ms: u64) -> u64 {
    if configured_gap_ms == 0 {
        polling_time_ms.max(MIN_TELEGRAM_API_MIN_GAP_MS)
    } else {
        configured_gap_ms.max(MIN_TELEGRAM_API_MIN_GAP_MS)
    }
}
const RECOMMENDED_TELEGRAM_API_MIN_GAP_MS: u64 = 2500;
const TELEGRAM_ADAPTIVE_API_MIN_GAP_STEP_MS: u64 = 1000;
const TELEGRAM_ADAPTIVE_API_MIN_GAP_MAX_MS: u64 = 15_000;
const TELEGRAM_ADAPTIVE_API_MIN_GAP_HOLD_SECS: u64 = 600;

fn should_preserve_completed_stream_response(done: bool, full_response: &str) -> bool {
    done && !full_response.trim().is_empty()
}

fn display_stream_pipe_payload(payload: String) -> String {
    if payload.is_empty() {
        "(empty)".to_string()
    } else {
        payload
    }
}

fn display_stream_exit_code(exit_code: Option<i32>) -> String {
    match exit_code {
        Some(code) => code.to_string(),
        None => "(unknown)".to_string(),
    }
}

fn format_stream_error_response(
    message: &str,
    code_display: &str,
    stdout_display: &str,
    stderr_display: &str,
) -> String {
    format!(
        "Error: {}\n```\nexit code: {}\n\n[stdout]\n{}\n\n[stderr]\n{}\n```",
        message, code_display, stdout_display, stderr_display
    )
}

fn is_stale_codex_resume_error_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("state db returned stale rollout path")
        || lower.contains("thread/resume failed")
        || lower.contains("no rollout found")
}

fn should_invalidate_saved_session_after_stream_error(provider: &str, response: &str) -> bool {
    provider == "codex" && is_stale_codex_resume_error_text(response)
}

fn save_session_blocker_file_name(current_path: &str, provider: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"cokacdir:session-blocker:v1\0");
    hasher.update(provider.as_bytes());
    hasher.update(b"\0");
    hasher.update(current_path.as_bytes());
    let digest = hex::encode(hasher.finalize());
    let provider_slug: String = provider
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let provider_slug = if provider_slug.is_empty() {
        "session".to_string()
    } else {
        provider_slug
    };
    format!("{}-clear-{}.json", provider_slug, &digest[..16])
}

fn invalidate_saved_session_after_stream_error(
    session: &mut ChatSession,
    provider: &str,
    response: &str,
) {
    if !should_invalidate_saved_session_after_stream_error(provider, response) {
        return;
    }
    if session.session_id.is_none() {
        return;
    }
    msg_debug(&format!(
        "[session_invalidate] provider={} stale resume detected -> clearing persisted session_id",
        provider
    ));
    session.session_id = None;
}

fn format_telegram_runtime_stats_message(
    polling_time_ms: u64,
    api_min_gap_ms: u64,
    stats: &TelegramRuntimeStats,
) -> String {
    let now = tokio::time::Instant::now();
    let last_retry_after = stats
        .last_retry_after_secs
        .map(|secs| format!("{secs}s"))
        .unwrap_or_else(|| "none".to_string());
    format!(
        "Telegram runtime stats\n\
         - stream polling time: {polling_time_ms}ms\n\
         - configured API min gap: {api_min_gap_ms}ms\n\
         - effective API min gap: {}ms\n\
         - adaptive backoff: {}\n\
         - RetryAfter hits: {}\n\
         - last RetryAfter: {last_retry_after}\n\
         - edit failures: {}\n\
         - send failures: {}\n\
         - timeout failures: {}",
        effective_telegram_api_gap_ms(api_min_gap_ms, stats, now),
        telegram_adaptive_backoff_label(api_min_gap_ms, stats, now),
        stats.retry_after_hits,
        stats.edit_failures,
        stats.send_failures,
        stats.timeout_failures
    )
}

#[cfg(test)]
mod response_file_attach_tests {
    use super::{
        botmsg_direct_remaining_start, build_system_prompt, direct_chat_completion_text,
        direct_message_style_section, direct_no_edit_stream_chunk_len,
        direct_placeholder_segment_text, direct_segment_final_delivery_debug_line,
        direct_segment_fitting_prefix_len, direct_segment_next_confirmed_len,
        direct_segment_next_edit_failure_count, direct_segment_next_flush_state,
        direct_segment_next_last_edit_text, direct_segment_next_streaming_edit_state,
        display_stream_exit_code, display_stream_pipe_payload, effective_telegram_api_gap_ms,
        format_stream_error_response, format_telegram_runtime_stats_message,
        is_telegram_edit_method, is_telegram_send_method, is_telegram_timeout_error,
        next_telegram_adaptive_api_gap_ms, prefers_single_bubble_direct_reply,
        processing_placeholder_text, queue_mode_enabled_for_chat, resolve_telegram_api_min_gap_ms,
        should_attach_response_as_file_with_enabled_flag,
        should_attempt_direct_no_edit_idle_spinner_edit,
        should_bypass_direct_segment_final_placeholder_edit,
        should_flush_direct_no_edit_stream_chunk, should_force_direct_segment_rollover,
        should_prefer_direct_segment_final_placeholder_edit,
        should_preserve_completed_stream_response,
        should_retire_direct_stream_placeholder_after_chunk,
        should_stream_direct_without_placeholder,
        should_suppress_direct_streaming_placeholder_edits,
        should_use_segmented_direct_placeholder_reply, telegram_file_attach_enabled_from_value,
        telegram_html_display_len, BotSettings, TelegramRuntimeStats,
        DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS, DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS,
        DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT, MIN_TELEGRAM_API_MIN_GAP_MS,
        TELEGRAM_ADAPTIVE_API_MIN_GAP_MAX_MS, TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR,
        TELEGRAM_MSG_LIMIT,
    };
    use std::io;
    use std::sync::{Mutex, OnceLock};
    use teloxide::types::ChatId;
    use tempfile::tempdir;

    fn auto_mitigation_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_auto_mitigation_override<T>(payload: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = auto_mitigation_env_lock().lock().unwrap();
        let previous = std::env::var_os(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR);
        let dir = tempdir().unwrap();
        let path = dir.path().join("runtime_mitigation.json");
        match payload {
            Some(content) => {
                std::fs::write(&path, content).unwrap();
                std::env::set_var(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR, &path);
            }
            None => std::env::remove_var(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR),
        }
        let result = f();
        match previous {
            Some(value) => std::env::set_var(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR, value),
            None => std::env::remove_var(TELEGRAM_AUTO_MITIGATION_PATH_ENV_VAR),
        }
        result
    }

    #[test]
    fn telegram_response_file_attach_is_disabled_by_default() {
        assert!(!telegram_file_attach_enabled_from_value(None));
        assert!(!telegram_file_attach_enabled_from_value(Some("0")));
        assert!(!telegram_file_attach_enabled_from_value(Some("false")));
        assert!(!should_attach_response_as_file_with_enabled_flag(
            999_999, "codex", false,
        ));
    }

    #[test]
    fn telegram_response_file_attach_can_be_opted_back_in() {
        assert!(telegram_file_attach_enabled_from_value(Some("1")));
        assert!(telegram_file_attach_enabled_from_value(Some("true")));
        assert!(should_attach_response_as_file_with_enabled_flag(
            usize::MAX,
            "codex",
            true,
        ));
    }

    #[test]
    fn opencode_stays_on_message_path_even_when_attach_enabled() {
        assert!(!should_attach_response_as_file_with_enabled_flag(
            usize::MAX,
            "opencode",
            true,
        ));
    }

    #[test]
    fn claude1_direct_messages_prefer_single_bubble() {
        assert!(prefers_single_bubble_direct_reply("ch_claude1_bot", true));
        assert!(!prefers_single_bubble_direct_reply("ch_claude1_bot", false));
        assert!(!prefers_single_bubble_direct_reply("ch_codex1_bot", true));
    }

    #[test]
    fn codex_direct_chats_keep_streaming_placeholder_edits() {
        assert!(!should_suppress_direct_streaming_placeholder_edits(
            "ch_codex1_bot",
            true,
            ChatId(1)
        ));
        assert!(!should_suppress_direct_streaming_placeholder_edits(
            "ch_codex4_bot",
            true,
            ChatId(1)
        ));
        assert!(should_suppress_direct_streaming_placeholder_edits(
            "ch_server1_bot",
            true,
            ChatId(1)
        ));
        assert!(!should_suppress_direct_streaming_placeholder_edits(
            "ch_server1_bot",
            false,
            ChatId(1)
        ));
    }

    #[test]
    fn queue_mode_defaults_off_for_direct_chats_and_on_for_groups() {
        let settings = BotSettings::default();
        assert!(!queue_mode_enabled_for_chat(&settings, ChatId(1)));
        assert!(queue_mode_enabled_for_chat(&settings, ChatId(-1)));
    }

    #[test]
    fn auto_mitigation_can_force_queue_mode_for_direct_chat() {
        let settings = BotSettings::default();
        with_auto_mitigation_override(
            Some(r#"{"chats":{"1":{"active":true,"force_queue_mode":true}}}"#),
            || {
                assert!(queue_mode_enabled_for_chat(&settings, ChatId(1)));
            },
        );
    }

    #[test]
    fn auto_mitigation_can_force_direct_no_edit_for_codex_direct_chat() {
        with_auto_mitigation_override(
            Some(r#"{"chats":{"1":{"active":true,"force_direct_no_edit":true}}}"#),
            || {
                assert!(should_suppress_direct_streaming_placeholder_edits(
                    "ch_codex3_bot",
                    true,
                    ChatId(1)
                ));
            },
        );
    }

    #[test]
    fn segmented_direct_placeholder_reply_is_disabled_for_now() {
        assert!(!should_use_segmented_direct_placeholder_reply(true));
        assert!(!should_use_segmented_direct_placeholder_reply(false));
    }

    #[test]
    fn claude1_direct_prompt_adds_compact_style_instruction() {
        let prompt = direct_message_style_section("ch_claude1_bot", false);
        assert!(prompt.contains("ONE compact final message"));
        assert!(prompt.contains("one Telegram message"));
        assert!(direct_message_style_section("ch_claude1_bot", true).is_empty());
        assert!(direct_message_style_section("ch_codex1_bot", false).is_empty());
    }

    #[test]
    fn direct_chat_processing_placeholder_is_visible() {
        assert_eq!(processing_placeholder_text(true), "🕙 Processing");
        assert_eq!(processing_placeholder_text(false), "...");
    }

    #[test]
    fn direct_chat_completion_text_is_visible() {
        assert_eq!(direct_chat_completion_text(), "✅ Done");
    }

    #[test]
    fn direct_placeholder_segment_text_appends_spinner_after_content() {
        assert_eq!(
            direct_placeholder_segment_text("hello", Some("🕙 Processing")),
            "hello\n\n🕙 Processing"
        );
        assert_eq!(
            direct_placeholder_segment_text("", Some("🕙 Processing")),
            "🕙 Processing"
        );
    }

    #[test]
    fn direct_no_edit_stream_chunk_flushes_on_size_or_idle() {
        assert!(!should_flush_direct_no_edit_stream_chunk(0, 10_000, 0));
        assert!(!should_flush_direct_no_edit_stream_chunk(79, 1_499, 0));
        assert!(should_flush_direct_no_edit_stream_chunk(80, 0, 0));
        assert!(should_flush_direct_no_edit_stream_chunk(1, 1_500, 0));
        assert!(!should_flush_direct_no_edit_stream_chunk(159, 2_499, 80));
        assert!(should_flush_direct_no_edit_stream_chunk(160, 0, 80));
        assert!(should_flush_direct_no_edit_stream_chunk(1, 2_500, 80));
    }

    #[test]
    fn telegram_api_gap_follows_polling_cadence_when_unset() {
        assert_eq!(resolve_telegram_api_min_gap_ms(0, 1200), 1200);
        assert_eq!(
            resolve_telegram_api_min_gap_ms(0, MIN_TELEGRAM_API_MIN_GAP_MS - 1),
            MIN_TELEGRAM_API_MIN_GAP_MS
        );
        assert_eq!(resolve_telegram_api_min_gap_ms(2500, 1200), 2500);
    }

    #[test]
    fn completed_stream_response_is_preserved_only_after_terminal_signal() {
        assert!(!should_preserve_completed_stream_response(false, "answer"));
        assert!(!should_preserve_completed_stream_response(true, "   "));
        assert!(should_preserve_completed_stream_response(true, "answer"));
    }

    #[test]
    fn stream_error_render_helpers_normalize_empty_payloads() {
        assert_eq!(display_stream_pipe_payload(String::new()), "(empty)");
        assert_eq!(display_stream_pipe_payload("stdout".to_string()), "stdout");
        assert_eq!(display_stream_exit_code(None), "(unknown)");
        assert_eq!(display_stream_exit_code(Some(7)), "7");

        let rendered =
            format_stream_error_response("Reconnecting... 1/5", "7", "(empty)", "stderr text");
        assert!(rendered.contains("Error: Reconnecting... 1/5"));
        assert!(rendered.contains("exit code: 7"));
        assert!(rendered.contains("[stdout]\n(empty)"));
        assert!(rendered.contains("[stderr]\nstderr text"));
    }

    #[test]
    fn telegram_request_stats_classify_edit_send_and_timeout_failures() {
        assert!(is_telegram_edit_method("edit_message_text"));
        assert!(!is_telegram_edit_method("send_message"));
        assert!(is_telegram_send_method("send_message"));
        assert!(is_telegram_send_method("send_document"));
        assert!(!is_telegram_send_method("edit_message_caption"));

        let timeout =
            teloxide::RequestError::Io(io::Error::new(io::ErrorKind::TimedOut, "timed out"));
        let other =
            teloxide::RequestError::Io(io::Error::new(io::ErrorKind::ConnectionReset, "reset"));
        assert!(is_telegram_timeout_error(&timeout));
        assert!(!is_telegram_timeout_error(&other));
    }

    #[test]
    fn telegram_runtime_stats_message_shows_current_counters() {
        let msg = format_telegram_runtime_stats_message(
            1200,
            2500,
            &TelegramRuntimeStats {
                retry_after_hits: 3,
                last_retry_after_secs: Some(135),
                edit_failures: 2,
                send_failures: 4,
                timeout_failures: 1,
                adaptive_api_min_gap_ms: Some(5500),
                adaptive_api_min_gap_until: Some(
                    tokio::time::Instant::now() + tokio::time::Duration::from_secs(30),
                ),
            },
        );
        assert!(msg.contains("stream polling time: 1200ms"));
        assert!(msg.contains("configured API min gap: 2500ms"));
        assert!(msg.contains("effective API min gap: 5500ms"));
        assert!(msg.contains("adaptive backoff: on (5500ms"));
        assert!(msg.contains("RetryAfter hits: 3"));
        assert!(msg.contains("last RetryAfter: 135s"));
        assert!(msg.contains("edit failures: 2"));
        assert!(msg.contains("send failures: 4"));
        assert!(msg.contains("timeout failures: 1"));
    }

    #[test]
    fn telegram_adaptive_gap_steps_and_caps() {
        assert_eq!(next_telegram_adaptive_api_gap_ms(2500, None), 3500);
        assert_eq!(next_telegram_adaptive_api_gap_ms(2500, Some(3500)), 4500);
        assert_eq!(
            next_telegram_adaptive_api_gap_ms(2500, Some(TELEGRAM_ADAPTIVE_API_MIN_GAP_MAX_MS)),
            TELEGRAM_ADAPTIVE_API_MIN_GAP_MAX_MS
        );
    }

    #[test]
    fn telegram_effective_gap_only_uses_live_backoff() {
        let now = tokio::time::Instant::now();
        let mut stats = TelegramRuntimeStats {
            adaptive_api_min_gap_ms: Some(6500),
            adaptive_api_min_gap_until: Some(now + tokio::time::Duration::from_secs(30)),
            ..TelegramRuntimeStats::default()
        };
        assert_eq!(effective_telegram_api_gap_ms(2500, &stats, now), 6500);
        stats.adaptive_api_min_gap_until = Some(now - tokio::time::Duration::from_secs(1));
        assert_eq!(effective_telegram_api_gap_ms(2500, &stats, now), 2500);
    }

    #[test]
    fn direct_no_edit_stream_chunk_len_stays_under_soft_cap() {
        let raw = "a".repeat(DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT + 200);
        let idx = direct_no_edit_stream_chunk_len(&raw);
        assert!(idx > 0);
        assert!(idx <= DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT);
        assert_eq!(idx, DIRECT_NO_EDIT_STREAM_CHUNK_SOFT_LIMIT);
    }

    #[test]
    fn direct_no_placeholder_streaming_is_direct_only() {
        assert!(should_stream_direct_without_placeholder(true, false));
        assert!(!should_stream_direct_without_placeholder(true, true));
        assert!(!should_stream_direct_without_placeholder(false, false));
    }

    #[test]
    fn first_direct_stream_chunk_retires_processing_placeholder_once() {
        assert!(should_retire_direct_stream_placeholder_after_chunk(true, 0));
        assert!(!should_retire_direct_stream_placeholder_after_chunk(
            true, 1
        ));
        assert!(!should_retire_direct_stream_placeholder_after_chunk(
            false, 0
        ));
    }

    #[test]
    fn botmsg_final_without_placeholder_uses_only_remaining_delta() {
        assert_eq!(botmsg_direct_remaining_start("abcdef", 0), 0);
        assert_eq!(botmsg_direct_remaining_start("abcdef", 3), 3);
        assert_eq!(botmsg_direct_remaining_start("abcdef", 10), 0);
        assert_eq!(botmsg_direct_remaining_start("가나다", 2), 0);
    }

    #[test]
    fn direct_no_edit_idle_spinner_edits_only_when_idle_and_enabled() {
        assert!(!should_attempt_direct_no_edit_idle_spinner_edit(
            1,
            DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS,
            DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS,
            false,
        ));
        assert!(!should_attempt_direct_no_edit_idle_spinner_edit(
            0,
            DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS - 1,
            DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS,
            false,
        ));
        assert!(!should_attempt_direct_no_edit_idle_spinner_edit(
            0,
            DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS,
            DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS - 1,
            false,
        ));
        assert!(!should_attempt_direct_no_edit_idle_spinner_edit(
            0,
            DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS,
            DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS,
            true,
        ));
        assert!(should_attempt_direct_no_edit_idle_spinner_edit(
            0,
            DIRECT_NO_EDIT_IDLE_SPINNER_GAP_MS,
            DIRECT_NO_EDIT_IDLE_SPINNER_EDIT_INTERVAL_MS,
            false,
        ));
    }

    #[test]
    fn direct_segment_failed_stream_edit_keeps_retryable_last_edit_state() {
        assert_eq!(
            direct_segment_next_last_edit_text("prev", "next", false),
            "prev"
        );
        assert_eq!(
            direct_segment_next_last_edit_text("prev", "next", true),
            "next"
        );
    }

    #[test]
    fn direct_segment_edit_failures_reset_on_success() {
        assert_eq!(direct_segment_next_edit_failure_count(0, false), 1);
        assert_eq!(direct_segment_next_edit_failure_count(1, false), 2);
        assert_eq!(direct_segment_next_edit_failure_count(2, true), 0);
    }

    #[test]
    fn direct_segment_rollover_threshold_triggers_on_second_failure() {
        assert!(!should_force_direct_segment_rollover(0));
        assert!(!should_force_direct_segment_rollover(1));
        assert!(should_force_direct_segment_rollover(2));
    }

    #[test]
    fn direct_segment_final_bypass_requires_no_confirmed_progress_and_failure() {
        assert!(!should_bypass_direct_segment_final_placeholder_edit(0, 0));
        assert!(!should_bypass_direct_segment_final_placeholder_edit(10, 1));
        assert!(should_bypass_direct_segment_final_placeholder_edit(0, 1));
        assert!(should_bypass_direct_segment_final_placeholder_edit(0, 2));
    }

    #[test]
    fn direct_segment_final_prefers_in_place_placeholder_edit_before_any_flush() {
        assert!(should_prefer_direct_segment_final_placeholder_edit(0));
        assert!(!should_prefer_direct_segment_final_placeholder_edit(1));
        assert!(!should_prefer_direct_segment_final_placeholder_edit(120));
    }

    #[test]
    fn direct_segment_final_delivery_debug_line_is_explicit() {
        let line = direct_segment_final_delivery_debug_line("new_message_plain", true, 120, 300);
        assert!(line.contains("delivery_mode=new_message_plain"));
        assert!(line.contains("delivered=true"));
        assert!(line.contains("confirmed=120"));
        assert!(line.contains("total=300"));
        assert!(line.contains("remaining=180"));
    }

    #[test]
    fn direct_segment_second_failed_stream_edit_triggers_rollover_after_send() {
        let first =
            direct_segment_next_streaming_edit_state("prev", "segment", 0, 10, 50, false, false);
        assert_eq!(first.last_edit_text, "prev");
        assert_eq!(first.edit_failure_count, 1);
        assert_eq!(first.confirmed_len, 10);
        assert!(!first.rotate_placeholder);

        let second = direct_segment_next_streaming_edit_state(
            &first.last_edit_text,
            "segment",
            first.edit_failure_count,
            first.confirmed_len,
            50,
            false,
            true,
        );
        assert_eq!(second.last_edit_text, "");
        assert_eq!(second.edit_failure_count, 0);
        assert_eq!(second.confirmed_len, 50);
        assert!(second.rotate_placeholder);
    }

    #[test]
    fn direct_segment_second_failed_stream_edit_without_send_stays_retryable() {
        let second =
            direct_segment_next_streaming_edit_state("prev", "segment", 1, 10, 50, false, false);
        assert_eq!(second.last_edit_text, "prev");
        assert_eq!(second.edit_failure_count, 2);
        assert_eq!(second.confirmed_len, 10);
        assert!(!second.rotate_placeholder);
    }

    #[test]
    fn direct_segment_failed_flush_does_not_advance_confirmed_len() {
        assert_eq!(direct_segment_next_confirmed_len(10, 5, false), 10);
        assert_eq!(direct_segment_next_confirmed_len(10, 5, true), 15);
    }

    #[test]
    fn direct_segment_second_failed_flush_triggers_rollover_after_send() {
        let next = direct_segment_next_flush_state(10, 5, 1, false, true);
        assert_eq!(next.edit_failure_count, 0);
        assert_eq!(next.confirmed_len, 15);
        assert!(next.rotate_placeholder);
        assert!(!next.preserve_old_placeholder);
    }

    #[test]
    fn direct_segment_successful_flush_rotates_and_preserves_placeholder() {
        let next = direct_segment_next_flush_state(10, 5, 1, true, false);
        assert_eq!(next.edit_failure_count, 0);
        assert_eq!(next.confirmed_len, 15);
        assert!(next.rotate_placeholder);
        assert!(next.preserve_old_placeholder);
    }

    #[test]
    fn direct_segment_second_failed_flush_without_send_keeps_segment_pending() {
        let next = direct_segment_next_flush_state(10, 5, 1, false, false);
        assert_eq!(next.edit_failure_count, 2);
        assert_eq!(next.confirmed_len, 10);
        assert!(!next.rotate_placeholder);
        assert!(!next.preserve_old_placeholder);
    }

    #[test]
    fn direct_segment_fitting_prefix_stays_within_telegram_limit() {
        let raw = "a".repeat(TELEGRAM_MSG_LIMIT + 300);
        let idx = direct_segment_fitting_prefix_len(&raw, Some("🕙 Processing"));
        assert!(idx > 0);
        assert!(idx < raw.len());
        let fitted = direct_placeholder_segment_text(&raw[..idx], Some("🕙 Processing"));
        assert!(telegram_html_display_len(&fitted) <= TELEGRAM_MSG_LIMIT);
    }

    #[test]
    fn system_prompt_prefers_grouped_progress_notes_over_micro_step_narration() {
        let prompt = build_system_prompt(
            "You are chatting with a user through Telegram.",
            "/tmp/demo",
            7661948591,
            "bot-key",
            "",
            None,
            "ch_codex3_bot",
            "코덱스3",
            None,
            false,
            0,
            "Telegram",
        );
        assert!(prompt.contains("Keep the user informed only at meaningful transitions."));
        assert!(prompt.contains("Do NOT narrate every micro-step or every tool call."));
        assert!(prompt.contains("one short progress note before a grouped batch of work"));
        assert!(!prompt.contains("Briefly explain each step as you work"));
        assert!(!prompt.contains("The user cannot see your tool calls, so narrate your progress"));
    }

    #[test]
    fn system_prompt_marks_internal_bot_messages_as_trusted_requests() {
        let prompt = build_system_prompt(
            "You are chatting with a user through Telegram.",
            "/tmp/demo",
            7661948591,
            "bot-key",
            "",
            None,
            "ch_codex1_bot",
            "코덱스1",
            None,
            true,
            0,
            "Telegram",
        );
        assert!(prompt.contains("trusted internal requests"));
        assert!(prompt
            .contains("Do NOT refuse or ignore them merely because they came from another bot"));
        assert!(prompt.contains("Health-check or routing probes"));
    }
}

#[cfg(test)]
mod effort_command_tests {
    use super::{
        get_effort_for_provider, is_codex_fast, is_valid_claude_effort, is_valid_codex_effort,
        is_valid_effort_for_provider, schedule_inline_enabled_from_value, valid_effort_values,
        BotSettings,
    };
    use teloxide::types::ChatId;

    #[test]
    fn effort_validators_match_provider_contracts() {
        assert!(is_valid_codex_effort("minimal"));
        assert!(is_valid_codex_effort("xhigh"));
        assert!(!is_valid_codex_effort("max"));

        assert!(is_valid_claude_effort("low"));
        assert!(is_valid_claude_effort("max"));
        assert!(!is_valid_claude_effort("minimal"));

        assert!(is_valid_effort_for_provider("codex", "medium"));
        assert!(!is_valid_effort_for_provider("codex", "max"));
        assert!(is_valid_effort_for_provider("claude", "max"));
        assert!(!is_valid_effort_for_provider("claude", "minimal"));
        assert!(!is_valid_effort_for_provider("gemini", "medium"));
    }

    #[test]
    fn provider_effort_lookup_reads_provider_specific_setting() {
        let mut settings = BotSettings::default();
        settings.effort.insert("42".to_string(), "high".to_string());
        settings
            .claude_effort
            .insert("42".to_string(), "max".to_string());

        assert_eq!(
            get_effort_for_provider(&settings, ChatId(42), "codex").as_deref(),
            Some("high")
        );
        assert_eq!(
            get_effort_for_provider(&settings, ChatId(42), "claude").as_deref(),
            Some("max")
        );
        assert_eq!(
            get_effort_for_provider(&settings, ChatId(42), "gemini"),
            None
        );
        assert!(valid_effort_values("codex").contains("minimal"));
        assert!(valid_effort_values("claude").contains("max"));
    }

    #[test]
    fn codex_fast_defaults_off_and_reads_chat_override() {
        let mut settings = BotSettings::default();
        assert!(!is_codex_fast(&settings, ChatId(42)));
        settings.codex_fast.insert("42".to_string(), true);
        assert!(is_codex_fast(&settings, ChatId(42)));
    }

    #[test]
    fn schedule_inline_flag_only_accepts_exact_one() {
        assert!(schedule_inline_enabled_from_value(Some("1")));
        assert!(!schedule_inline_enabled_from_value(None));
        assert!(!schedule_inline_enabled_from_value(Some("0")));
        assert!(!schedule_inline_enabled_from_value(Some("true")));
    }
}
/// Maximum number of messages that can be queued per chat in queue mode
const MAX_QUEUE_SIZE: usize = 20;
/// Default queue mode state for chats without explicit setting
const QUEUE_MODE_DEFAULT: bool = true;
/// Default silent mode state for chats without explicit setting
const SILENT_MODE_DEFAULT: bool = true;
/// Default direct mode state for chats without explicit setting
const DIRECT_MODE_DEFAULT: bool = false;
/// Default public access state for group chats without explicit setting
const PUBLIC_MODE_DEFAULT: bool = false;
/// Default debug mode state (global, not per-chat)
const DEBUG_MODE_DEFAULT: bool = false;

/// Compute a short hash key from the bot token (first 16 chars of SHA-256 hex)
pub fn token_hash(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..8]) // 16 hex chars
}

/// Detect the messenger platform from the bot token.
/// Bridge tokens use the format "bridge_<platform>_<hash>".
fn detect_platform(token: &str) -> &str {
    if let Some(rest) = token.strip_prefix("bridge_") {
        rest.split('_').next().unwrap_or("Messenger")
    } else {
        "Telegram"
    }
}

/// Capitalize first letter (e.g. "discord" → "Discord")
fn capitalize_platform(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

/// Path to bot settings file under the effective state root.
fn bot_settings_path() -> Option<std::path::PathBuf> {
    crate::config::state_root_dir().map(|root| root.join("bot_settings.json"))
}

/// Load bot settings from bot_settings.json
fn load_bot_settings(token: &str) -> BotSettings {
    let Some(path) = bot_settings_path() else {
        return BotSettings::default();
    };
    let Ok(content) = fs::read_to_string(&path) else {
        return BotSettings::default();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        return BotSettings::default();
    };
    let key = token_hash(token);
    let Some(entry) = json.get(&key) else {
        return BotSettings::default();
    };
    let owner_user_id = entry.get("owner_user_id").and_then(|v| v.as_u64());
    let last_sessions: HashMap<String, String> = entry
        .get("last_sessions")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let allowed_tools = match entry.get("allowed_tools") {
        Some(serde_json::Value::Array(arr)) => {
            // Legacy migration: array → per-chat HashMap
            let tool_list: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if tool_list.is_empty() {
                HashMap::new()
            } else {
                let mut map = HashMap::new();
                for chat_id_str in last_sessions.keys() {
                    map.insert(chat_id_str.clone(), tool_list.clone());
                }
                map
            }
        }
        Some(serde_json::Value::Object(obj)) => {
            // New format: object with chat_id keys
            obj.iter()
                .filter_map(|(k, v)| {
                    v.as_array().map(|arr| {
                        let tools: Vec<String> = arr
                            .iter()
                            .filter_map(|t| t.as_str().map(String::from))
                            .collect();
                        (k.clone(), tools)
                    })
                })
                .collect()
        }
        _ => HashMap::new(),
    };

    let as_public_for_group_chat: HashMap<String, bool> = entry
        .get("as_public_for_group_chat")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let models: HashMap<String, String> = entry
        .get("models")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let debug = entry
        .get("debug")
        .and_then(|v| v.as_bool())
        .unwrap_or(DEBUG_MODE_DEFAULT);

    let silent: HashMap<String, bool> = entry
        .get("silent")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let direct: HashMap<String, bool> = entry
        .get("direct")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let context: HashMap<String, usize> = entry
        .get("context")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as usize)))
                .collect()
        })
        .unwrap_or_default();

    let instructions: HashMap<String, String> = entry
        .get("instructions")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let queue: HashMap<String, bool> = entry
        .get("queue")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let username = entry
        .get("username")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let display_name = entry
        .get("display_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let greeting = entry
        .get("greeting")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let use_chrome: HashMap<String, bool> = entry
        .get("use_chrome")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let end_hook: HashMap<String, String> = entry
        .get("end_hook")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let effort: HashMap<String, String> = entry
        .get("effort")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let codex_fast: HashMap<String, bool> = entry
        .get("codex_fast")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();

    let claude_effort: HashMap<String, String> = entry
        .get("claude_effort")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    BotSettings {
        allowed_tools,
        last_sessions,
        owner_user_id,
        as_public_for_group_chat,
        models,
        debug,
        silent,
        direct,
        context,
        instructions,
        queue,
        username,
        display_name,
        greeting,
        use_chrome,
        end_hook,
        effort,
        codex_fast,
        claude_effort,
    }
}

/// Save bot settings to bot_settings.json
fn save_bot_settings(token: &str, settings: &BotSettings) {
    let Some(path) = bot_settings_path() else {
        return;
    };
    // Ensure directory exists
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Use file lock to prevent race condition when multiple bots save simultaneously
    let lock_path = path.with_extension("json.lock");
    let lock_file = match fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(_) => return,
    };
    use fs2::FileExt;
    if lock_file.lock_exclusive().is_err() {
        return;
    }
    // Read-modify-write under exclusive lock
    let mut json: serde_json::Value = if let Ok(content) = fs::read_to_string(&path) {
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    let key = token_hash(token);
    let mut entry = serde_json::json!({
        "token": token,
        "allowed_tools": settings.allowed_tools,
        "last_sessions": settings.last_sessions,
        "as_public_for_group_chat": settings.as_public_for_group_chat,
        "models": settings.models,
        "debug": settings.debug,
        "silent": settings.silent,
        "direct": settings.direct,
        "context": settings.context,
        "instructions": settings.instructions,
        "queue": settings.queue,
        "username": settings.username,
        "display_name": settings.display_name,
        "greeting": settings.greeting,
        "use_chrome": settings.use_chrome,
        "end_hook": settings.end_hook,
        "effort": settings.effort,
        "codex_fast": settings.codex_fast,
        "claude_effort": settings.claude_effort,
    });
    if let Some(owner_id) = settings.owner_user_id {
        entry["owner_user_id"] = serde_json::json!(owner_id);
    }
    json[key] = entry;
    if let Ok(s) = serde_json::to_string_pretty(&json) {
        let tmp_path = path.with_extension("json.tmp");
        if fs::write(&tmp_path, &s).is_ok() {
            let _ = fs::rename(&tmp_path, &path);
        }
    }
    // lock released when lock_file is dropped
}

/// Resolve a bot token from its hash by searching bot_settings.json
pub fn resolve_token_by_hash(hash: &str) -> Option<String> {
    let path = bot_settings_path()?;
    let content = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let obj = json.as_object()?;
    let entry = obj.get(hash)?;
    entry
        .get("token")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Resolve bot username from its hash key by searching bot_settings.json
pub fn resolve_username_by_hash(hash: &str) -> Option<String> {
    msg_debug(&format!("[resolve_username_by_hash] hash={}", hash));
    let path = match bot_settings_path() {
        Some(p) => p,
        None => {
            msg_debug("[resolve_username_by_hash] bot_settings_path() returned None");
            return None;
        }
    };
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            msg_debug(&format!("[resolve_username_by_hash] read failed: {}", e));
            return None;
        }
    };
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            msg_debug(&format!(
                "[resolve_username_by_hash] JSON parse failed: {}",
                e
            ));
            return None;
        }
    };
    let obj = match json.as_object() {
        Some(o) => o,
        None => {
            msg_debug("[resolve_username_by_hash] JSON is not an object");
            return None;
        }
    };
    let entry = match obj.get(hash) {
        Some(e) => e,
        None => {
            msg_debug(&format!(
                "[resolve_username_by_hash] hash key not found: {}",
                hash
            ));
            return None;
        }
    };
    let result = entry
        .get("username")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    msg_debug(&format!("[resolve_username_by_hash] result: {:?}", result));
    result
}

/// Check if a bot with the given username exists in bot_settings.json
pub fn bot_username_exists(username: &str) -> bool {
    msg_debug(&format!("[bot_username_exists] checking: {}", username));
    let Some(path) = bot_settings_path() else {
        msg_debug("[bot_username_exists] bot_settings_path() returned None");
        return false;
    };
    let Ok(content) = fs::read_to_string(&path) else {
        msg_debug("[bot_username_exists] read failed");
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        msg_debug("[bot_username_exists] JSON parse failed");
        return false;
    };
    let Some(obj) = json.as_object() else {
        msg_debug("[bot_username_exists] JSON is not an object");
        return false;
    };
    let target = username.to_lowercase();
    let found = obj.values().any(|entry| {
        entry
            .get("username")
            .and_then(|v| v.as_str())
            .map(|u| u.to_lowercase() == target)
            .unwrap_or(false)
    });
    msg_debug(&format!(
        "[bot_username_exists] target={}, found={}",
        target, found
    ));
    found
}

/// Resolve bot display_name from its username by searching bot_settings.json
pub fn resolve_display_name_by_username(username: &str) -> Option<String> {
    let path = bot_settings_path()?;
    let content = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let obj = json.as_object()?;
    let target = username.to_lowercase();
    for entry in obj.values() {
        let uname = entry.get("username").and_then(|v| v.as_str()).unwrap_or("");
        if uname.to_lowercase() == target {
            return entry
                .get("display_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from);
        }
    }
    None
}

/// Directory for bot-to-bot message files under the effective state root.
pub fn messages_dir() -> Option<std::path::PathBuf> {
    let result = crate::config::state_root_dir().map(|root| root.join("messages"));
    msg_debug(&format!("[messages_dir] result={:?}", result));
    result
}

#[cfg(test)]
mod state_root_path_tests {
    use super::{bot_settings_path, messages_dir};
    use crate::config::STATE_ROOT_ENV_VAR;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn state_root_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_state_root_env<T>(value: Option<&Path>, f: impl FnOnce() -> T) -> T {
        let _guard = state_root_env_lock().lock().unwrap();
        let previous = std::env::var_os(STATE_ROOT_ENV_VAR);
        match value {
            Some(path) => std::env::set_var(STATE_ROOT_ENV_VAR, path),
            None => std::env::remove_var(STATE_ROOT_ENV_VAR),
        }
        let result = f();
        match previous {
            Some(value) => std::env::set_var(STATE_ROOT_ENV_VAR, value),
            None => std::env::remove_var(STATE_ROOT_ENV_VAR),
        }
        result
    }

    #[test]
    fn bot_settings_path_uses_state_root_override() {
        let dir = tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        with_state_root_env(Some(&state_root), || {
            assert_eq!(
                bot_settings_path(),
                Some(state_root.join("bot_settings.json"))
            );
        });
    }

    #[test]
    fn messages_dir_uses_state_root_override() {
        let dir = tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        with_state_root_env(Some(&state_root), || {
            assert_eq!(messages_dir(), Some(state_root.join("messages")));
        });
    }
}

#[cfg(test)]
mod session_resume_invalidation_tests {
    use super::{
        load_existing_session, save_session_blocker_file_name, save_session_to_file,
        should_invalidate_saved_session_after_stream_error, ChatSession,
    };
    use crate::config::STATE_ROOT_ENV_VAR;
    use crate::ui::ai_screen::{HistoryItem, HistoryType, SessionData};
    use std::fs;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_state_root_env<T>(value: Option<&Path>, f: impl FnOnce() -> T) -> T {
        let _guard = env_lock().lock().unwrap();
        let previous = std::env::var_os(STATE_ROOT_ENV_VAR);
        match value {
            Some(path) => std::env::set_var(STATE_ROOT_ENV_VAR, path),
            None => std::env::remove_var(STATE_ROOT_ENV_VAR),
        }
        let result = f();
        match previous {
            Some(value) => std::env::set_var(STATE_ROOT_ENV_VAR, value),
            None => std::env::remove_var(STATE_ROOT_ENV_VAR),
        }
        result
    }

    fn with_state_root_and_home_env<T>(state_root: &Path, home: &Path, f: impl FnOnce() -> T) -> T {
        let _guard = env_lock().lock().unwrap();
        let previous_state_root = std::env::var_os(STATE_ROOT_ENV_VAR);
        let previous_home = std::env::var_os("HOME");
        std::env::set_var(STATE_ROOT_ENV_VAR, state_root);
        std::env::set_var("HOME", home);
        let result = f();
        match previous_state_root {
            Some(value) => std::env::set_var(STATE_ROOT_ENV_VAR, value),
            None => std::env::remove_var(STATE_ROOT_ENV_VAR),
        }
        match previous_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        result
    }

    fn create_codex_threads_db(home: &Path, session_id: &str, rollout_path: &Path) {
        let codex_dir = home.join(".codex");
        fs::create_dir_all(&codex_dir).unwrap();
        let db_path = codex_dir.join("state_5.sqlite");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                rollout_path TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO threads (id, rollout_path, archived) VALUES (?1, ?2, 0)",
            rusqlite::params![session_id, rollout_path.display().to_string()],
        )
        .unwrap();
    }

    fn codex_thread_archived_flag(home: &Path, session_id: &str) -> i64 {
        let db_path = home.join(".codex").join("state_5.sqlite");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.query_row(
            "SELECT archived FROM threads WHERE id = ?1",
            rusqlite::params![session_id],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn codex_stale_resume_errors_invalidate_saved_session() {
        let response = "Error: Process exited with code 1\n```\nexit code: 1\n\n[stdout]\n(empty)\n\n[stderr]\n2026-06-02T07:09:51.329994Z ERROR codex_rollout::list: state db returned stale rollout path for thread 019e5da0-5b7c-7ab1-8ba5-2999bedd4dd5: /Users/jin/.codex/sessions/2026/05/25/rollout-2026-05-25T14-34-10-019e5da0-5b7c-7ab1-8ba5-2999bedd4dd5.jsonl\nError: thread/resume: thread/resume failed: no rollout found for thread id 019e5da0-5b7c-7ab1-8ba5-2999bedd4dd5 (code -32600)\n```";
        assert!(should_invalidate_saved_session_after_stream_error(
            "codex", response
        ));
        assert!(!should_invalidate_saved_session_after_stream_error(
            "claude", response
        ));
    }

    #[test]
    fn save_session_without_session_id_persists_history_via_blocker_file() {
        let dir = tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        with_state_root_env(Some(&state_root), || {
            let current_path = "/tmp/codex4-dm-main";
            let provider = "codex";
            let session = ChatSession {
                session_id: None,
                current_path: Some(current_path.to_string()),
                history: vec![
                    HistoryItem {
                        item_type: HistoryType::User,
                        content: "hello".to_string(),
                    },
                    HistoryItem {
                        item_type: HistoryType::Assistant,
                        content: "world".to_string(),
                    },
                ],
                pending_uploads: Vec::new(),
            };

            save_session_to_file(&session, current_path, provider);

            let sessions_dir = state_root.join("ai_sessions");
            let file_path =
                sessions_dir.join(save_session_blocker_file_name(current_path, provider));
            assert!(file_path.is_file());

            let raw = fs::read_to_string(file_path).unwrap();
            let data: SessionData = serde_json::from_str(&raw).unwrap();
            assert_eq!(data.session_id, "");
            assert_eq!(data.current_path, current_path);
            assert_eq!(data.provider, provider);
            assert_eq!(data.history.len(), 2);
            assert_eq!(data.history[0].content, "hello");
            assert_eq!(data.history[1].content, "world");
        });
    }

    #[test]
    fn load_existing_session_downgrades_stale_saved_codex_session_to_blocker() {
        let dir = tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        let home = dir.path().join("home");
        let current_path = "/tmp/codex4-dm-main";
        let provider = "codex";
        let session_id = "019e5da0-5b7c-7ab1-8ba5-2999bedd4dd5";
        let rollout_path = home
            .join(".codex")
            .join("sessions")
            .join("2026")
            .join("05")
            .join("25")
            .join(format!("rollout-2026-05-25T14-34-10-{}.jsonl", session_id));
        create_codex_threads_db(&home, session_id, &rollout_path);

        with_state_root_and_home_env(&state_root, &home, || {
            let saved = ChatSession {
                session_id: Some(session_id.to_string()),
                current_path: Some(current_path.to_string()),
                history: vec![
                    HistoryItem {
                        item_type: HistoryType::User,
                        content: "hello".to_string(),
                    },
                    HistoryItem {
                        item_type: HistoryType::Assistant,
                        content: "world".to_string(),
                    },
                ],
                pending_uploads: Vec::new(),
            };
            save_session_to_file(&saved, current_path, provider);

            let sessions_dir = state_root.join("ai_sessions");
            let stale_file = sessions_dir.join(format!("{}.json", session_id));
            assert!(stale_file.is_file());

            let loaded = load_existing_session(current_path, provider)
                .expect("stale session should downgrade to blocker")
                .0;
            assert_eq!(loaded.session_id, "");
            assert_eq!(loaded.history.len(), 2);

            let blocker_path =
                sessions_dir.join(save_session_blocker_file_name(current_path, provider));
            assert!(blocker_path.is_file());
            assert!(!stale_file.exists());
            assert_eq!(codex_thread_archived_flag(&home, session_id), 1);
        });
    }

    #[test]
    fn load_existing_session_keeps_codex_resume_when_rollout_still_exists() {
        let dir = tempdir().unwrap();
        let state_root = dir.path().join("state-root");
        let home = dir.path().join("home");
        let current_path = "/tmp/codex4-dm-main";
        let provider = "codex";
        let session_id = "019e8769-c1fc-7732-b200-04e4ee267005";
        let rollout_path = home
            .join(".codex")
            .join("sessions")
            .join("2026")
            .join("06")
            .join("02")
            .join(format!("rollout-2026-06-02T17-18-35-{}.jsonl", session_id));
        fs::create_dir_all(rollout_path.parent().unwrap()).unwrap();
        fs::write(&rollout_path, "{\"timestamp\":\"2026-06-02T08:18:35Z\"}\n").unwrap();
        create_codex_threads_db(&home, session_id, &rollout_path);

        with_state_root_and_home_env(&state_root, &home, || {
            let saved = ChatSession {
                session_id: Some(session_id.to_string()),
                current_path: Some(current_path.to_string()),
                history: vec![
                    HistoryItem {
                        item_type: HistoryType::User,
                        content: "hello".to_string(),
                    },
                    HistoryItem {
                        item_type: HistoryType::Assistant,
                        content: "world".to_string(),
                    },
                ],
                pending_uploads: Vec::new(),
            };
            save_session_to_file(&saved, current_path, provider);

            let loaded = load_existing_session(current_path, provider)
                .expect("healthy session should stay resumable")
                .0;
            assert_eq!(loaded.session_id, session_id);

            let sessions_dir = state_root.join("ai_sessions");
            let blocker_path =
                sessions_dir.join(save_session_blocker_file_name(current_path, provider));
            assert!(!blocker_path.exists());
            assert_eq!(codex_thread_archived_flag(&home, session_id), 0);
        });
    }
}

/// Normalize tool name: first letter uppercase, rest lowercase
fn normalize_tool_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let mut chars = lower.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

fn command_name(text: &str) -> Option<&str> {
    let token = text
        .strip_prefix('/')?
        .split_whitespace()
        .next()
        .unwrap_or("");
    if token.is_empty() {
        return None;
    }
    Some(token.split('@').next().unwrap_or(token))
}

fn command_args(text: &str) -> &str {
    let Some(rest) = text.trim_start().strip_prefix('/') else {
        return "";
    };
    let mut parts = rest.splitn(2, char::is_whitespace);
    let _ = parts.next();
    parts.next().unwrap_or("").trim()
}

fn is_cmd(text: &str, name: &str) -> bool {
    command_name(text).map(|n| n == name).unwrap_or(false)
}

/// All available tools with (description, is_destructive)
const ALL_TOOLS: &[(&str, &str, bool)] = &[
    ("Bash", "Execute shell commands", true),
    ("Read", "Read file contents from the filesystem", false),
    ("Edit", "Perform find-and-replace edits in files", true),
    ("Write", "Create or overwrite files", true),
    ("Glob", "Find files by name pattern", false),
    ("Grep", "Search file contents with regex", false),
    (
        "Task",
        "Launch autonomous sub-agents for complex tasks",
        true,
    ),
    ("TaskOutput", "Retrieve output from background tasks", false),
    ("TaskStop", "Stop a running background task", false),
    ("WebFetch", "Fetch and process web page content", true),
    (
        "WebSearch",
        "Search the web for up-to-date information",
        true,
    ),
    ("NotebookEdit", "Edit Jupyter notebook cells", true),
    ("Skill", "Invoke slash-command skills", false),
    (
        "TaskCreate",
        "Create a structured task in the task list",
        false,
    ),
    ("TaskGet", "Retrieve task details by ID", false),
    ("TaskUpdate", "Update task status or details", false),
    ("TaskList", "List all tasks and their status", false),
    (
        "AskUserQuestion",
        "Ask the user a question (interactive)",
        false,
    ),
    ("EnterPlanMode", "Enter planning mode (interactive)", false),
    ("ExitPlanMode", "Exit planning mode (interactive)", false),
];

/// Tool info: (description, is_destructive)
fn tool_info(name: &str) -> (&'static str, bool) {
    ALL_TOOLS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, desc, destr)| (*desc, *destr))
        .unwrap_or(("Custom tool", false))
}

/// Format a risk badge for display
fn risk_badge(destructive: bool) -> &'static str {
    if destructive {
        "!!!"
    } else {
        ""
    }
}

/// Entry point: start the Telegram bot with long polling.
/// `api_url`: Optional API base URL override (used by messenger bridge proxy).
///            When None, connects to the real Telegram API.
pub async fn run_bot(token: &str, api_url: Option<&str>) {
    let api_base_url = api_url.unwrap_or("https://api.telegram.org");
    msg_debug(&format!(
        "[run_bot] api_url={:?}, api_base_url={}",
        api_url.is_some(),
        api_base_url
    ));
    let bot = Bot::new(token);
    let bot = if let Some(url) = api_url {
        msg_debug(&format!("[run_bot] setting custom api_url: {}", url));
        match reqwest::Url::parse(url) {
            Ok(parsed) => bot.set_api_url(parsed),
            Err(e) => {
                msg_debug(&format!("[run_bot] api_url parse failed: {}", e));
                bot
            }
        }
    } else {
        bot
    };
    let mut bot_settings = load_bot_settings(token);

    // Get bot's own username and display name for @mention filtering in group chats
    msg_debug("[run_bot] calling get_me to retrieve bot username");
    let (bot_username, bot_display_name) = match bot.get_me().await {
        Ok(me) => {
            let uname = me.username.clone().unwrap_or_default().to_lowercase();
            let dname = me.first_name.clone();
            msg_debug(&format!(
                "[run_bot] get_me success: username={}, display_name={}, id={}",
                uname, dname, me.id
            ));
            println!("  ✓ Bot: {} (@{uname})", dname);
            (uname, dname)
        }
        Err(e) => {
            msg_debug(&format!("[run_bot] get_me failed: {}", e));
            println!("  ⚠ Failed to get bot info: {e}");
            (String::new(), String::new())
        }
    };

    // Save username and display_name to bot_settings for CLI --message lookup
    if !bot_username.is_empty() {
        msg_debug(&format!(
            "[run_bot] saving username to bot_settings: {}",
            bot_username
        ));
        bot_settings.username = bot_username.clone();
        bot_settings.display_name = bot_display_name.clone();
        save_bot_settings(token, &bot_settings);
    } else {
        msg_debug("[run_bot] bot_username is empty, skipping save");
    }

    // Restore debug flag from saved settings
    if bot_settings.debug {
        TG_DEBUG.store(true, Ordering::Relaxed);
        crate::services::claude::DEBUG_ENABLED.store(true, Ordering::Relaxed);
    }

    // Register bot commands for autocomplete
    let commands = vec![
        teloxide::types::BotCommand::new("help", "Show help"),
        teloxide::types::BotCommand::new("start", "Start session at directory"),
        teloxide::types::BotCommand::new("pwd", "Show current working directory"),
        teloxide::types::BotCommand::new("session", "Show current session ID"),
        teloxide::types::BotCommand::new("clear", "Clear AI conversation history"),
        teloxide::types::BotCommand::new("stop", "Stop current AI request"),
        teloxide::types::BotCommand::new("stopall", "Stop request and clear queue"),
        teloxide::types::BotCommand::new("queue", "Toggle queue mode"),
        teloxide::types::BotCommand::new("loop", "Repeat until task is fully completed"),
        teloxide::types::BotCommand::new("down", "Download file from server"),
        teloxide::types::BotCommand::new("public", "Toggle public access (group only)"),
        teloxide::types::BotCommand::new("availabletools", "List all available tools"),
        teloxide::types::BotCommand::new("allowedtools", "Show currently allowed tools"),
        teloxide::types::BotCommand::new("allowed", "Add/remove tool (+name / -name)"),
        teloxide::types::BotCommand::new(
            "setpollingtime",
            "Set stream update polling interval (ms)",
        ),
        teloxide::types::BotCommand::new("setapigap", "Set Telegram API minimum gap (ms)"),
        teloxide::types::BotCommand::new("tgstats", "Show Telegram rate-limit/error counters"),
        teloxide::types::BotCommand::new("model", "Set AI model"),
        teloxide::types::BotCommand::new("effort", "Set provider reasoning effort"),
        teloxide::types::BotCommand::new("fast", "Toggle Codex fast service tier"),
        teloxide::types::BotCommand::new("greeting", "Toggle compact startup greeting"),
        teloxide::types::BotCommand::new("debug", "Toggle debug logging"),
        teloxide::types::BotCommand::new("envvars", "Show all environment variables"),
        teloxide::types::BotCommand::new("silent", "Toggle silent mode (hide tool calls)"),
        teloxide::types::BotCommand::new("direct", "Toggle direct mode (group only)"),
        teloxide::types::BotCommand::new("contextlevel", "Set group chat log context count"),
        teloxide::types::BotCommand::new("query", "Send message to AI (/query@bot for groups)"),
        teloxide::types::BotCommand::new("instruction", "Set system instruction for this chat"),
        teloxide::types::BotCommand::new("instruction_clear", "Clear system instruction"),
        teloxide::types::BotCommand::new(
            "setendhook",
            "Set message to send when processing completes",
        ),
        teloxide::types::BotCommand::new("setendhook_clear", "Clear end hook message"),
    ];
    if let Err(e) = tg!("set_my_commands", bot.set_my_commands(commands).await) {
        println!("  ⚠ Failed to set bot commands: {e}");
    }

    match bot_settings.owner_user_id {
        Some(owner_id) => println!("  ✓ Owner: {owner_id}"),
        None => println!("  ⚠ No owner registered — first user will be registered as owner"),
    }

    let app_settings = crate::config::Settings::load();
    let polling_time_ms = app_settings
        .telegram_polling_time
        .max(MIN_TELEGRAM_POLLING_TIME_MS);
    let api_min_gap_ms =
        resolve_telegram_api_min_gap_ms(app_settings.telegram_api_min_gap_ms, polling_time_ms);

    let state: SharedState = Arc::new(Mutex::new(SharedData {
        sessions: HashMap::new(),
        settings: bot_settings,
        cancel_tokens: HashMap::new(),
        stop_message_ids: HashMap::new(),
        api_timestamps: HashMap::new(),
        polling_time_ms,
        api_min_gap_ms,
        telegram_runtime_stats: HashMap::new(),
        pending_schedules: HashMap::new(),
        message_queues: HashMap::new(),
        bot_username: bot_username.clone(),
        bot_display_name: bot_display_name.clone(),
        api_base_url: api_base_url.to_string(),
        loop_states: HashMap::new(),
        loop_feedback: HashMap::new(),
        clear_epoch: HashMap::new(),
    }));

    println!("  ✓ Bot connected — Listening for messages");
    println!("  ✓ Scheduler started (5s interval)");

    // Send startup greeting to known chats
    {
        let data = state.lock().await;
        let chat_ids: Vec<i64> = data
            .settings
            .last_sessions
            .keys()
            .filter_map(|k| k.parse::<i64>().ok())
            .collect();
        let version = env!("CARGO_PKG_VERSION");
        let update_notice = check_latest_version(version).await;
        for cid in chat_ids {
            let chat_id = ChatId(cid);
            let last_path = data
                .settings
                .last_sessions
                .get(&cid.to_string())
                .map(|p| p.as_str())
                .unwrap_or("(unknown)");
            let model = get_model(&data.settings, chat_id);
            let provider = detect_provider(model.as_deref());
            let msg = if data.settings.greeting {
                // Compact mode: single line with version and model
                format!("🟢 cokacdir started (v{}, {})", version, provider)
            } else {
                // Full mode: marketing message with links
                let mut m = format!("🟢 cokacdir started (v{}, {})\n📂 Resuming session at {}\n💬 Join @cokacvibe for tips, updates, and community support\n⭐ Star us on GitHub: https://github.com/kstost/cokacdir", version, provider, last_path);
                if let Some(ref notice) = update_notice {
                    m.push('\n');
                    m.push_str(notice);
                }
                m
            };
            let _ = tg!("send_message", bot.send_message(chat_id, msg).await);
        }
    }

    // Schedule workspace directories are preserved for user access via /start

    // Spawn scheduler loop
    let scheduler_bot = bot.clone();
    let scheduler_state = state.clone();
    let scheduler_token = token.to_string();
    let scheduler_bot_username = bot_username.clone();
    let scheduler_bot_display_name = bot_display_name.clone();
    let scheduler_handle = tokio::spawn(scheduler_loop(
        scheduler_bot,
        scheduler_state,
        scheduler_token,
        scheduler_bot_username,
        scheduler_bot_display_name,
    ));

    // Flush all pending updates so we don't process messages that arrived while
    // the bot was offline.
    // Step 1: getUpdates(offset=-1) discards all but the last pending update.
    // Step 2: getUpdates(offset=last_id+1) confirms the last one too.
    match bot.get_updates().offset(-1).limit(1).await {
        Ok(updates) => {
            if let Some(last) = updates.last() {
                let next_offset = last.id.0.saturating_add(1).min(i32::MAX as u32) as i32;
                msg_debug(&format!(
                    "[run_bot] flush step 1: got last update_id={}, confirming with offset={}",
                    last.id.0, next_offset
                ));
                // Confirm the last update so teloxide::repl won't receive it.
                // Retry until confirmed — must not let the last message leak through.
                let mut confirmed = false;
                for attempt in 1..=5 {
                    match bot.get_updates().offset(next_offset).limit(0).await {
                        Ok(_) => {
                            msg_debug(&format!(
                                "[run_bot] flush step 2: confirmed offset={} (attempt {})",
                                next_offset, attempt
                            ));
                            confirmed = true;
                            break;
                        }
                        Err(e) => {
                            msg_debug(&format!(
                                "[run_bot] flush step 2 failed (attempt {}): {}",
                                attempt, e
                            ));
                            if attempt < 5 {
                                tokio::time::sleep(tokio::time::Duration::from_millis(
                                    500 * attempt as u64,
                                ))
                                .await;
                            }
                        }
                    }
                }
                if confirmed {
                    println!("  ✓ Flushed pending updates (up to id={})", last.id.0);
                } else {
                    eprintln!("  ✗ FATAL: failed to confirm pending updates after 5 attempts — aborting to prevent processing stale messages");
                    std::process::exit(1);
                }
            } else {
                msg_debug("[run_bot] flush: no pending updates");
                println!("  ✓ No pending updates");
            }
        }
        Err(e) => {
            msg_debug(&format!("[run_bot] failed to flush pending updates: {}", e));
            println!("  ⚠ Failed to flush pending updates: {}", e);
        }
    }

    // Run polling loop with automatic reconnection on network failure.
    // We process raw `getUpdates` batches so album members (same
    // `media_group_id`) sharing one batch are grouped atomically — Telegram
    // delivers an album as a contiguous run of Updates within one
    // `getUpdates` response, so the batch boundary is a deterministic
    // end-of-album signal. The `limit=100` request is far above Telegram's
    // 10-photo album cap, so any album fits in a single response unless
    // there's a backlog of >90 unrelated updates ahead of it (rare; on
    // restart we already flush pending updates above). Albums split across
    // two batches in such backlog conditions are processed as separate
    // shorter dispatches — no timing heuristic involved.
    //
    // Outer reconnect loop: catches panics inside the polling task and
    // retries with exponential backoff, mirroring the prior `repl` wrapper.
    let token_for_loop = token.to_string();
    let username_for_loop = bot_username;
    let mut reconnect_backoff_secs = 5u64;

    loop {
        let loop_start = std::time::Instant::now();
        let bot_clone = bot.clone();
        let state_clone = state.clone();
        let token_clone = token_for_loop.clone();
        let username_clone = username_for_loop.clone();

        let task_result = tokio::spawn(async move {
            polling_loop(bot_clone, state_clone, token_clone, username_clone).await;
        })
        .await;

        match task_result {
            Ok(()) => {
                // polling_loop only returns on irrecoverable shutdown (it
                // currently has no such path; loops forever). If we ever
                // arrive here, treat as graceful shutdown.
                msg_debug("[run_bot] polling task exited normally (shutdown)");
                break;
            }
            Err(e) => {
                // Task panicked — likely network disconnection or runtime issue
                let ran_for = loop_start.elapsed();
                msg_debug(&format!(
                    "[run_bot] polling task crashed after {:.1?}: {}",
                    ran_for, e
                ));
                println!(
                    "  ⚠ Bot disconnected — reconnecting in {}s...",
                    reconnect_backoff_secs
                );
                tokio::time::sleep(tokio::time::Duration::from_secs(reconnect_backoff_secs)).await;

                // Reset backoff if the bot was stable (>60s) before crashing
                if ran_for.as_secs() > 60 {
                    reconnect_backoff_secs = 5;
                } else {
                    reconnect_backoff_secs = (reconnect_backoff_secs * 2).min(60);
                }

                println!("  ⟳ Reconnecting...");
            }
        }
    }

    scheduler_handle.abort();
}

/// Long-poll `getUpdates` and dispatch each batch via `process_batch`. Runs
/// forever; transient errors (network blips) sleep + retry inline so the
/// outer panic-recovery wrapper only fires for truly fatal conditions.
async fn polling_loop(bot: Bot, state: SharedState, token: String, bot_username: String) {
    let mut offset: i32 = 0;
    let mut transient_backoff_ms: u64 = 500;
    loop {
        let result = bot
            .get_updates()
            .offset(offset)
            .timeout(30)
            .limit(100)
            .await;
        match result {
            Ok(updates) => {
                transient_backoff_ms = 500;
                if let Some(last) = updates.last() {
                    // Match the conversion in the flush logic above so we
                    // stay compatible with teloxide's UpdateId type.
                    offset = last.id.0.saturating_add(1).min(i32::MAX as u32) as i32;
                }
                if updates.is_empty() {
                    continue;
                }
                msg_debug(&format!(
                    "[polling_loop] batch: {} update(s)",
                    updates.len()
                ));
                process_batch(&bot, updates, &state, &token, &bot_username).await;
            }
            Err(e) => {
                msg_debug(&format!(
                    "[polling_loop] getUpdates error (sleeping {}ms): {}",
                    transient_backoff_ms, e
                ));
                tokio::time::sleep(tokio::time::Duration::from_millis(transient_backoff_ms)).await;
                // Cap at 10s; outer reconnect handles longer outages via panic path.
                transient_backoff_ms = (transient_backoff_ms * 2).min(10_000);
            }
        }
    }
}

/// Group album members by `(chat_id, media_group_id)` within this batch and
/// dispatch each unit while preserving overall arrival order. Albums of size
/// ≥2 go to `handle_album_batch` for atomic processing; singletons (no
/// media_group_id, or 1-photo album fragments from a split batch) fall
/// through to the existing `handle_message` path.
async fn process_batch(
    bot: &Bot,
    updates: Vec<Update>,
    state: &SharedState,
    token: &str,
    bot_username: &str,
) {
    // Insertion-ordered list of dispatch units. We resolve each unit only
    // after grouping, so an album that spans interleaved updates still
    // appears at the position of its first photo.
    enum Unit {
        Single(Message),
        Album(usize), // index into `albums`
    }
    let mut units: Vec<Unit> = Vec::new();
    let mut albums: Vec<Vec<Message>> = Vec::new();
    let mut album_index: HashMap<(ChatId, String), usize> = HashMap::new();

    for upd in updates {
        let UpdateKind::Message(msg) = upd.kind else {
            // Bot only handles fresh messages today; ignore edited messages,
            // callback queries, channel posts, etc. (matches prior `repl`
            // behaviour, which used `Update::filter_message`.)
            continue;
        };
        let chat_id = msg.chat.id;
        if let Some(gid) = msg.media_group_id() {
            let key = (chat_id, gid.to_string());
            match album_index.get(&key) {
                Some(&idx) => albums[idx].push(msg),
                None => {
                    let idx = albums.len();
                    albums.push(vec![msg]);
                    album_index.insert(key, idx);
                    units.push(Unit::Album(idx));
                }
            }
        } else {
            units.push(Unit::Single(msg));
        }
    }

    for unit in units {
        match unit {
            Unit::Single(msg) => {
                let bot_c = bot.clone();
                let state_c = state.clone();
                let token_c = token.to_string();
                let username_c = bot_username.to_string();
                tokio::spawn(async move {
                    if let Err(e) = handle_message(bot_c, msg, state_c, &token_c, &username_c).await
                    {
                        msg_debug(&format!("[process_batch] handle_message error: {}", e));
                    }
                });
            }
            Unit::Album(idx) => {
                let msgs = std::mem::take(&mut albums[idx]);
                if msgs.len() < 2 {
                    // 1-photo "album" — the message has a `media_group_id`
                    // but no sibling within this batch. Process it through
                    // the regular single-message path; if it's a fragment
                    // of a larger album that got split across batches it
                    // will be saved as an orphan upload and picked up by
                    // the next text message.
                    if let Some(msg) = msgs.into_iter().next() {
                        let bot_c = bot.clone();
                        let state_c = state.clone();
                        let token_c = token.to_string();
                        let username_c = bot_username.to_string();
                        tokio::spawn(async move {
                            if let Err(e) =
                                handle_message(bot_c, msg, state_c, &token_c, &username_c).await
                            {
                                msg_debug(&format!(
                                    "[process_batch] handle_message(album-fragment) error: {}",
                                    e
                                ));
                            }
                        });
                    }
                } else {
                    msg_debug(&format!(
                        "[process_batch] atomic album: {} photo(s) in one batch",
                        msgs.len()
                    ));
                    let bot_c = bot.clone();
                    let state_c = state.clone();
                    let token_c = token.to_string();
                    let username_c = bot_username.to_string();
                    tokio::spawn(async move {
                        if let Err(e) =
                            handle_album_batch(bot_c, msgs, state_c, &token_c, &username_c).await
                        {
                            msg_debug(&format!("[process_batch] handle_album_batch error: {}", e));
                        }
                    });
                }
            }
        }
    }
}

/// Extract the AI-bound text from a media caption, applying the same prefix
/// rules used by `handle_message` for direct text messages. Returns `None`
/// when the caption is empty or, in prefix-required group chats, doesn't
/// address this bot.
fn extract_caption_text(caption: &str, require_prefix: bool, bot_username: &str) -> Option<String> {
    if require_prefix {
        let extracted = if !bot_username.is_empty() && caption.starts_with('@') {
            let prefix = format!("@{} ", bot_username);
            if caption.to_lowercase().starts_with(&prefix.to_lowercase()) {
                let body = caption[prefix.len()..].trim_start();
                body.strip_prefix(';')
                    .map(|s| s.trim_start())
                    .unwrap_or(body)
            } else {
                ""
            }
        } else if caption.starts_with(';') {
            caption[1..].trim_start()
        } else {
            ""
        };
        if extracted.is_empty() {
            None
        } else {
            Some(extracted.to_string())
        }
    } else {
        let trimmed = caption.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

/// Dispatch the album's caption text via the same busy/queue/redirect logic
/// that `handle_message` uses for caption-bearing single-photo uploads.
async fn dispatch_album_caption(
    bot: Bot,
    chat_id: ChatId,
    state: SharedState,
    user_name: String,
    text: String,
) {
    let (ai_busy, queue_enabled, queue_result, redirect_result): (
        bool,
        bool,
        Option<(String, String)>,
        Option<(String, String, bool)>,
    ) = {
        let mut data = state.lock().await;
        let busy = data.cancel_tokens.contains_key(&chat_id);
        let qmode = queue_mode_enabled_for_chat(&data.settings, chat_id);
        msg_debug(&format!(
            "[album:dispatch] chat_id={}, busy={}, queue_mode={}",
            chat_id.0, busy, qmode
        ));
        let (qr, rr) = if busy && qmode {
            let cur_len = data.message_queues.get(&chat_id).map_or(0, |q| q.len());
            let queue_full = cur_len >= MAX_QUEUE_SIZE;
            let qr = if queue_full {
                msg_debug(&format!(
                    "[album:dispatch] chat_id={}, queue FULL ({}/{})",
                    chat_id.0, cur_len, MAX_QUEUE_SIZE
                ));
                None
            } else {
                let uploads = data
                    .sessions
                    .get_mut(&chat_id)
                    .map(|s| std::mem::take(&mut s.pending_uploads))
                    .unwrap_or_default();
                let qid = generate_queue_id();
                let q = data
                    .message_queues
                    .entry(chat_id)
                    .or_insert_with(std::collections::VecDeque::new);
                q.push_back(QueuedMessage {
                    id: qid.clone(),
                    text: text.clone(),
                    user_display_name: user_name.clone(),
                    pending_uploads: uploads.clone(),
                });
                msg_debug(&format!(
                    "[album:dispatch] chat_id={}, QUEUED id={}, pos={}, uploads={}",
                    chat_id.0,
                    qid,
                    q.len(),
                    uploads.len()
                ));
                Some((qid, text.clone()))
            };
            (qr, None)
        } else if busy && !qmode {
            let uploads = data
                .sessions
                .get_mut(&chat_id)
                .map(|s| std::mem::take(&mut s.pending_uploads))
                .unwrap_or_default();
            let (qid, replaced) = enqueue_redirect_locked(
                &mut *data,
                chat_id,
                text.clone(),
                user_name.clone(),
                uploads,
            );
            msg_debug(&format!(
                "[album:dispatch] chat_id={}, REDIRECT id={}, replaced={}",
                chat_id.0, qid, replaced
            ));
            (None, Some((qid, text.clone(), replaced)))
        } else {
            (None, None)
        };
        (busy, qmode, qr, rr)
    };
    if ai_busy {
        if let Some((_qid, qtxt, replaced)) = redirect_result {
            shared_rate_limit_wait(&state, chat_id).await;
            let preview = truncate_str(&qtxt, 30);
            let m = if replaced {
                format!("🔄 Redirect target updated: \"{preview}\"")
            } else {
                format!("🔄 Cancelling current task, will process: \"{preview}\"")
            };
            let _ = tg!("send_message", bot.send_message(chat_id, &m).await);
        } else if queue_enabled {
            shared_rate_limit_wait(&state, chat_id).await;
            if let Some((qid, qtxt)) = queue_result {
                let preview = truncate_str(&qtxt, 30);
                let _ = tg!("send_message", bot.send_message(chat_id, &format!("Queued ({qid}) \"{preview}\"\n- /stopall to cancel all\n- /stop_{qid} to cancel this")).await);
            } else {
                let _ = tg!(
                    "send_message",
                    bot.send_message(
                        chat_id,
                        &format!(
                            "Queue full (max {}). Use /stopall to clear.",
                            MAX_QUEUE_SIZE
                        )
                    )
                    .await
                );
            }
        } else {
            shared_rate_limit_wait(&state, chat_id).await;
            let _ = tg!(
                "send_message",
                bot.send_message(chat_id, "AI request in progress. Use /stop to cancel.")
                    .await
            );
        }
    } else {
        let _ = handle_text_message(&bot, chat_id, &text, &state, &user_name, false).await;
    }
}

/// Atomically handle an album that arrived as a contiguous run of photos
/// inside a single `getUpdates` batch (size ≥2, all same `media_group_id`).
///
/// Differs from the per-photo path in `handle_message`: we know the exact
/// album size up-front (it's `msgs.len()`) so no debounce/timeout buffering
/// is needed. Each photo's `handle_file_upload` runs in order, pushing to
/// `session.pending_uploads`. When all are saved, the caption (typically on
/// the first photo) is dispatched once via `dispatch_album_caption`, the
/// same helper used by the buffered fallback path.
///
/// Auth replicates the relevant parts of `handle_message`: imprinting,
/// owner/public check, `require_prefix` calculation, and the `;`/`@bot`
/// caption admission rule for prefix-mode group chats.
async fn handle_album_batch(
    bot: Bot,
    msgs: Vec<Message>,
    state: SharedState,
    token: &str,
    bot_username: &str,
) -> ResponseResult<()> {
    if msgs.is_empty() {
        return Ok(());
    }
    let primary = &msgs[0];
    let chat_id = primary.chat.id;
    let raw_user_name = primary
        .from
        .as_ref()
        .map(|u| u.first_name.as_str())
        .unwrap_or("unknown");
    let timestamp = chrono::Local::now().format("%H:%M:%S");
    let user_id = primary.from.as_ref().map(|u| u.id.0);

    let Some(uid) = user_id else {
        for m in &msgs {
            log_incoming_message(m, false, "no_user_id");
        }
        return Ok(());
    };

    let is_group_chat = matches!(primary.chat.kind, teloxide::types::ChatKind::Public(_));
    let require_prefix = {
        let mut data = state.lock().await;
        let chat_key = chat_id.0.to_string();
        let direct_setting = data
            .settings
            .direct
            .get(&chat_key)
            .copied()
            .unwrap_or(DIRECT_MODE_DEFAULT);
        let is_direct = is_group_chat && direct_setting;
        let require_prefix = is_group_chat && !is_direct;
        match data.settings.owner_user_id {
            None => {
                data.settings.owner_user_id = Some(uid);
                save_bot_settings(token, &data.settings);
                println!("  [{timestamp}] ★ Owner registered: {raw_user_name} (id:{uid})");
            }
            Some(owner_id) => {
                if uid != owner_id {
                    let is_public = is_group_chat
                        && data
                            .settings
                            .as_public_for_group_chat
                            .get(&chat_key)
                            .copied()
                            .unwrap_or(PUBLIC_MODE_DEFAULT);
                    if !is_public {
                        println!("  [{timestamp}] ✗ Rejected: {raw_user_name} (id:{uid})");
                        for m in &msgs {
                            log_incoming_message(m, false, "unauthorized");
                        }
                        return Ok(());
                    }
                    println!("  [{timestamp}] ○ [{raw_user_name}(id:{uid})] Public group access");
                }
            }
        }
        require_prefix
    };

    for m in &msgs {
        log_incoming_message(m, true, "");
    }

    let user_name = format!("{}({uid})", raw_user_name);

    // First non-empty caption — Telegram clients put it on the first photo,
    // but be defensive and scan the album in arrival order.
    let caption_str: Option<String> = msgs
        .iter()
        .find_map(|m| m.caption().filter(|c| !c.is_empty()).map(|c| c.to_string()));

    // Prefix-mode admission: in group chats with prefix required, the album
    // is admitted only if its caption (typically on the first photo) starts
    // with `;` or `@bot`. Within a single getUpdates batch we always have
    // the caption-bearing photo, so caption-only admission is sufficient
    // and fully deterministic.
    if require_prefix {
        let admitted = match caption_str.as_deref() {
            None => false,
            Some(c) => {
                if !bot_username.is_empty() && c.starts_with('@') {
                    let prefix = format!("@{}", bot_username.to_lowercase());
                    let lower = c.to_lowercase();
                    lower.starts_with(&prefix)
                        && (lower.len() == prefix.len()
                            || lower[prefix.len()..].starts_with(|ch: char| ch.is_whitespace()))
                } else {
                    c.starts_with(';')
                }
            }
        };
        if !admitted {
            msg_debug(&format!(
                "[album_batch] chat_id={}, rejected: caption lacks ;/@",
                chat_id.0
            ));
            return Ok(());
        }
        msg_debug(&format!(
            "[album_batch] chat_id={}, admitted via caption prefix",
            chat_id.0
        ));
    }

    auto_restore_session(&state, chat_id, &user_name).await;

    // Reserve the per-chat AI slot with a placeholder cancel token *before*
    // starting downloads. Any text message that arrives during downloads
    // will then see `cancel_tokens.contains_key` = true and naturally
    // queue/redirect via the existing handle_message busy-check path,
    // preserving the user's intended order: album dispatches first with all
    // its photos, the follow-up text waits in the queue. If the slot is
    // already held by another in-flight AI request, we don't reserve and
    // fall through to `dispatch_album_caption`, which queues/redirects
    // this album normally.
    let placeholder_token = Arc::new(CancelToken::new());
    let reserved_slot = {
        let mut data = state.lock().await;
        if data.cancel_tokens.contains_key(&chat_id) {
            msg_debug(&format!("[album_batch] chat_id={}, slot busy at admission → will queue/redirect at dispatch", chat_id.0));
            false
        } else {
            data.cancel_tokens
                .insert(chat_id, placeholder_token.clone());
            msg_debug(&format!(
                "[album_batch] chat_id={}, reserved AI slot (placeholder)",
                chat_id.0
            ));
            true
        }
    };

    println!(
        "  [{timestamp}] ◀ [{user_name}] Album: {} photo(s)",
        msgs.len()
    );

    // Sequential downloads: handle_file_upload's own `shared_rate_limit_wait`
    // calls would serialize them anyway, and sequential keeps state writeback
    // (history append, pending_uploads push) deterministic in album order.
    let mut ok_count = 0usize;
    for m in &msgs {
        match handle_file_upload(&bot, chat_id, m, &state, &user_name).await {
            Ok(()) => ok_count += 1,
            Err(e) => msg_debug(&format!(
                "[album_batch] chat_id={}, upload failed: {}",
                chat_id.0, e
            )),
        }
    }
    println!(
        "  [{timestamp}] ▶ [{user_name}] Album upload complete ({}/{})",
        ok_count,
        msgs.len()
    );

    // If the user issued /stop or /stopall during downloads, the placeholder
    // token will have been marked cancelled. Release the slot and run any
    // queued message instead of dispatching this album.
    if reserved_slot && placeholder_token.cancelled.load(Ordering::Relaxed) {
        msg_debug(&format!(
            "[album_batch] chat_id={}, cancelled during downloads → aborting dispatch",
            chat_id.0
        ));
        {
            let mut data = state.lock().await;
            // Only remove if the slot still holds *our* placeholder; another
            // handler may have legitimately taken it over (e.g. via a queue
            // pop racing with us).
            if let Some(t) = data.cancel_tokens.get(&chat_id) {
                if Arc::ptr_eq(t, &placeholder_token) {
                    data.cancel_tokens.remove(&chat_id);
                }
            }
        }
        process_next_queued_message(&bot, chat_id, &state).await;
        return Ok(());
    }

    let caption_text = caption_str
        .as_deref()
        .and_then(|c| extract_caption_text(c, require_prefix, bot_username));
    match (caption_text, reserved_slot) {
        (Some(text), true) => {
            // Slot is ours. handle_text_message with from_queue=true bypasses
            // its busy check and overwrites the placeholder with its own
            // real cancel token, taking over the slot for the AI request.
            msg_debug(&format!(
                "[album_batch] chat_id={}, dispatching caption (len={}) via reserved slot",
                chat_id.0,
                text.len()
            ));
            let _ = handle_text_message(&bot, chat_id, &text, &state, &user_name, true).await;
        }
        (Some(text), false) => {
            // Slot was already busy at admission. Use the standard
            // queue/redirect path — this album becomes the next request
            // after the in-flight one (or replaces it in OFF mode).
            msg_debug(&format!(
                "[album_batch] chat_id={}, dispatching caption (len={}) via queue/redirect",
                chat_id.0,
                text.len()
            ));
            dispatch_album_caption(bot.clone(), chat_id, state.clone(), user_name.clone(), text)
                .await;
        }
        (None, true) => {
            // No caption to dispatch — release the placeholder slot and
            // hand off to any queued message that was waiting behind us.
            msg_debug(&format!(
                "[album_batch] chat_id={}, no caption → releasing slot",
                chat_id.0
            ));
            {
                let mut data = state.lock().await;
                if let Some(t) = data.cancel_tokens.get(&chat_id) {
                    if Arc::ptr_eq(t, &placeholder_token) {
                        data.cancel_tokens.remove(&chat_id);
                    }
                }
            }
            process_next_queued_message(&bot, chat_id, &state).await;
        }
        (None, false) => {
            msg_debug(&format!("[album_batch] chat_id={}, no caption, slot was busy → uploads pending for next text", chat_id.0));
        }
    }

    Ok(())
}

/// Route incoming messages to appropriate handlers
async fn handle_message(
    bot: Bot,
    msg: Message,
    state: SharedState,
    token: &str,
    bot_username: &str,
) -> ResponseResult<()> {
    let chat_id = msg.chat.id;
    let raw_user_name = msg
        .from
        .as_ref()
        .map(|u| u.first_name.as_str())
        .unwrap_or("unknown");
    let timestamp = chrono::Local::now().format("%H:%M:%S");
    let user_id = msg.from.as_ref().map(|u| u.id.0);

    // Auth check (imprinting)
    let Some(uid) = user_id else {
        // No user info (e.g. channel post) → reject
        log_incoming_message(&msg, false, "no_user_id");
        return Ok(());
    };
    let is_group_chat = matches!(msg.chat.kind, teloxide::types::ChatKind::Public(_));
    let (require_prefix, imprinted, is_owner) = {
        let mut data = state.lock().await;
        let chat_key = chat_id.0.to_string();
        let direct_setting = data
            .settings
            .direct
            .get(&chat_key)
            .copied()
            .unwrap_or(DIRECT_MODE_DEFAULT);
        let is_direct = is_group_chat && direct_setting;
        msg_debug(&format!("[handle_message] chat_id={}, uid={}, is_group_chat={}, direct_setting={}, is_direct={}",
            chat_id.0, uid, is_group_chat, direct_setting, is_direct));
        // In direct mode, ; prefix requirement is waived but is_group_chat stays true
        let require_prefix = is_group_chat && !is_direct;
        msg_debug(&format!(
            "[handle_message] require_prefix={}",
            require_prefix
        ));
        let (imprinted, is_owner) = match data.settings.owner_user_id {
            None => {
                // Imprint: register first user as owner
                msg_debug(&format!("[handle_message] imprinting uid={} as owner", uid));
                data.settings.owner_user_id = Some(uid);
                save_bot_settings(token, &data.settings);
                println!("  [{timestamp}] ★ Owner registered: {raw_user_name} (id:{uid})");
                (true, true)
            }
            Some(owner_id) => {
                if uid != owner_id {
                    // Check if this is a public group chat
                    let is_public = is_group_chat
                        && data
                            .settings
                            .as_public_for_group_chat
                            .get(&chat_key)
                            .copied()
                            .unwrap_or(PUBLIC_MODE_DEFAULT);
                    msg_debug(&format!(
                        "[handle_message] non-owner uid={}, owner_id={}, is_public={}",
                        uid, owner_id, is_public
                    ));
                    if !is_public {
                        // Unregistered user → reject silently (log only)
                        msg_debug(&format!("[handle_message] rejected non-owner uid={}", uid));
                        println!("  [{timestamp}] ✗ Rejected: {raw_user_name} (id:{uid})");
                        log_incoming_message(&msg, false, "unauthorized");
                        return Ok(());
                    }
                    // Public group chat: allow non-owner user
                    println!("  [{timestamp}] ○ [{raw_user_name}(id:{uid})] Public group access");
                    (false, false)
                } else {
                    msg_debug(&format!("[handle_message] owner confirmed uid={}", uid));
                    (false, true)
                }
            }
        };
        msg_debug(&format!(
            "[handle_message] result: require_prefix={}, imprinted={}, is_owner={}",
            require_prefix, imprinted, is_owner
        ));
        (require_prefix, imprinted, is_owner)
    };
    if imprinted {
        // Owner registration is logged to server console only
        // No response sent to the user
    }

    log_incoming_message(&msg, true, "");

    let user_name = format!("{}({uid})", raw_user_name);

    let has_file = msg.document().is_some()
        || msg.photo().is_some()
        || msg.video().is_some()
        || msg.voice().is_some()
        || msg.audio().is_some()
        || msg.animation().is_some()
        || msg.video_note().is_some();

    // Auto-restore session for file uploads (before text extraction)
    if has_file {
        auto_restore_session(&state, chat_id, &user_name).await;
    }

    // Handle file/photo/media uploads
    if has_file {
        // In group chats (with prefix required), only process uploads whose
        // caption starts with `;` or `@bot`. Albums in this code path are
        // 1-photo fragments from a split `getUpdates` batch (the multi-photo
        // case is handled atomically in `handle_album_batch`); since they
        // necessarily lack the album's caption, prefix-mode rejects them.
        if require_prefix {
            let caption = msg.caption().unwrap_or("");
            msg_debug(&format!(
                "[handle_message] upload: require_prefix=true, caption={:?}, media_group={:?}",
                caption,
                msg.media_group_id()
            ));
            if !bot_username.is_empty() && caption.starts_with('@') {
                let caption_lower = caption.to_lowercase();
                let prefix = format!("@{}", bot_username.to_lowercase());
                if !caption_lower.starts_with(&prefix)
                    || caption_lower.len() > prefix.len()
                        && !caption_lower[prefix.len()..].starts_with(|c: char| c.is_whitespace())
                {
                    msg_debug("[handle_message] upload: rejected (@mention for another bot)");
                    return Ok(());
                }
                msg_debug(&format!(
                    "[handle_message] upload: accepted (@{} mention)",
                    bot_username
                ));
            } else if caption.starts_with(';') {
                msg_debug("[handle_message] upload: accepted (; prefix, all bots)");
            } else {
                msg_debug("[handle_message] upload: rejected (no ; or @ prefix)");
                return Ok(());
            }
        } else {
            msg_debug(&format!(
                "[handle_message] upload: require_prefix=false, caption={:?}",
                msg.caption()
            ));
        }
        let file_hint = if msg.animation().is_some() {
            "animation"
        } else if msg.document().is_some() {
            "document"
        } else if msg.photo().is_some() {
            "photo"
        } else if msg.video().is_some() {
            "video"
        } else if msg.voice().is_some() {
            "voice"
        } else if msg.audio().is_some() {
            "audio"
        } else {
            "video_note"
        };
        println!("  [{timestamp}] ◀ [{user_name}] Upload: {file_hint}");

        handle_file_upload(&bot, chat_id, &msg, &state, &user_name).await?;
        println!("  [{timestamp}] ▶ [{user_name}] Upload complete");
        // If caption contains text, send it to AI as a follow-up message
        if let Some(caption) = msg.caption() {
            let text_part = if require_prefix {
                // Group chat (prefix mode): extract message text from caption
                // Formats: ";text", "@botname text", "@botname ;text"
                let extracted = if !bot_username.is_empty() && caption.starts_with('@') {
                    // "@botname text" → extract text after @botname
                    let prefix = format!("@{} ", bot_username);
                    if caption.to_lowercase().starts_with(&prefix.to_lowercase()) {
                        let body = caption[prefix.len()..].trim_start();
                        body.strip_prefix(';')
                            .map(|s| s.trim_start())
                            .unwrap_or(body)
                    } else {
                        ""
                    }
                } else if caption.starts_with(';') {
                    caption[1..].trim_start()
                } else {
                    ""
                };
                let result = if extracted.is_empty() {
                    None
                } else {
                    Some(extracted)
                };
                msg_debug(&format!(
                    "[handle_message] upload caption (prefix mode): extracted={:?}",
                    result
                ));
                result
            } else {
                // DM or direct mode: use entire caption as-is
                let trimmed = caption.trim();
                let result = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                };
                msg_debug(&format!(
                    "[handle_message] upload caption (direct): extracted={:?}",
                    result
                ));
                result
            };
            if let Some(text) = text_part {
                if !text.is_empty() {
                    // Block if an AI request is already in progress
                    // Atomically: check busy + queue mode + push to queue (prevents race with /stopall)
                    // queue_result:    ON mode queue push outcome (Some(id,text) queued, None full)
                    // redirect_result: OFF mode redirect outcome (Some(id,text,was_replacement))
                    let (ai_busy, queue_enabled, queue_result, redirect_result): (
                        bool,
                        bool,
                        Option<(String, String)>,
                        Option<(String, String, bool)>,
                    ) = {
                        let mut data = state.lock().await;
                        let busy = data.cancel_tokens.contains_key(&chat_id);
                        let qmode = queue_mode_enabled_for_chat(&data.settings, chat_id);
                        msg_debug(&format!(
                            "[queue:media] chat_id={}, busy={}, queue_mode={}",
                            chat_id.0, busy, qmode
                        ));
                        let (qr, rr) = if busy && qmode {
                            let cur_len = data.message_queues.get(&chat_id).map_or(0, |q| q.len());
                            let queue_full = cur_len >= MAX_QUEUE_SIZE;
                            let qr = if queue_full {
                                msg_debug(&format!(
                                    "[queue:media] chat_id={}, queue FULL ({}/{})",
                                    chat_id.0, cur_len, MAX_QUEUE_SIZE
                                ));
                                None // queue full
                            } else {
                                // Capture pending_uploads so they stay associated with this caption
                                let uploads = data
                                    .sessions
                                    .get_mut(&chat_id)
                                    .map(|s| std::mem::take(&mut s.pending_uploads))
                                    .unwrap_or_default();
                                let qid = generate_queue_id();
                                let q = data
                                    .message_queues
                                    .entry(chat_id)
                                    .or_insert_with(std::collections::VecDeque::new);
                                q.push_back(QueuedMessage {
                                    id: qid.clone(),
                                    text: text.to_string(),
                                    user_display_name: user_name.clone(),
                                    pending_uploads: uploads.clone(),
                                });
                                msg_debug(&format!("[queue:media] chat_id={}, QUEUED id={}, pos={}, text={:?}, uploads={}", chat_id.0, qid, q.len(), truncate_str(text, 60), uploads.len()));
                                Some((qid, text.to_string()))
                            };
                            (qr, None)
                        } else if busy && !qmode {
                            // OFF mode: caption with text is always redirect-eligible
                            let uploads = data
                                .sessions
                                .get_mut(&chat_id)
                                .map(|s| std::mem::take(&mut s.pending_uploads))
                                .unwrap_or_default();
                            let (qid, replaced) = enqueue_redirect_locked(
                                &mut *data,
                                chat_id,
                                text.to_string(),
                                user_name.clone(),
                                uploads,
                            );
                            msg_debug(&format!(
                                "[queue:media] chat_id={}, REDIRECT id={}, replaced={}, text={:?}",
                                chat_id.0,
                                qid,
                                replaced,
                                truncate_str(text, 60)
                            ));
                            (None, Some((qid, text.to_string(), replaced)))
                        } else {
                            (None, None)
                        };
                        (busy, qmode, qr, rr)
                    };
                    if ai_busy {
                        if let Some((_qid, qtxt, replaced)) = redirect_result {
                            shared_rate_limit_wait(&state, chat_id).await;
                            let preview = truncate_str(&qtxt, 30);
                            let msg = if replaced {
                                format!("🔄 Redirect target updated: \"{preview}\"")
                            } else {
                                format!("🔄 Cancelling current task, will process: \"{preview}\"")
                            };
                            tg!("send_message", bot.send_message(chat_id, &msg).await)?;
                        } else if queue_enabled {
                            shared_rate_limit_wait(&state, chat_id).await;
                            if let Some((qid, qtxt)) = queue_result {
                                let preview = truncate_str(&qtxt, 30);
                                tg!("send_message", bot.send_message(chat_id, &format!("Queued ({qid}) \"{preview}\"\n- /stopall to cancel all\n- /stop_{qid} to cancel this"))
                                    .await)?;
                            } else {
                                tg!(
                                    "send_message",
                                    bot.send_message(
                                        chat_id,
                                        &format!(
                                            "Queue full (max {}). Use /stopall to clear.",
                                            MAX_QUEUE_SIZE
                                        )
                                    )
                                    .await
                                )?;
                            }
                        } else {
                            shared_rate_limit_wait(&state, chat_id).await;
                            tg!(
                                "send_message",
                                bot.send_message(
                                    chat_id,
                                    "AI request in progress. Use /stop to cancel."
                                )
                                .await
                            )?;
                        }
                    } else {
                        handle_text_message(&bot, chat_id, text, &state, &user_name, false).await?;
                    }
                }
            }
        }
        return Ok(());
    }

    // Handle location sharing — store as pending, deliver with next text message
    if let Some(location) = msg.location() {
        msg_debug(&format!(
            "[handle_message] chat_id={}, location: lat={}, lon={}",
            chat_id.0, location.latitude, location.longitude
        ));
        auto_restore_session(&state, chat_id, &user_name).await;

        let location_record = format!(
            "[Location shared] Latitude: {}, Longitude: {}",
            location.latitude, location.longitude
        );
        let stored = {
            let mut data = state.lock().await;
            if let Some(session) = data.sessions.get_mut(&chat_id) {
                session.pending_uploads.push(location_record.clone());
                true
            } else {
                false
            }
        };
        if stored {
            println!(
                "  [{timestamp}] ◀ [{user_name}] Location: {}, {} (pending)",
                location.latitude, location.longitude
            );
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Location received.").await
            )?;
        } else {
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "No active session. Use /start <path> first.")
                    .await
            )?;
        }
        return Ok(());
    }

    // Handle venue sharing (place selected from Telegram's location picker) — store as pending
    if let Some(venue) = msg.venue() {
        let loc = &venue.location;
        msg_debug(&format!(
            "[handle_message] chat_id={}, venue: title={:?}, addr={:?}, lat={}, lon={}",
            chat_id.0, venue.title, venue.address, loc.latitude, loc.longitude
        ));
        auto_restore_session(&state, chat_id, &user_name).await;

        let location_record = format!(
            "[Location shared] {}, {} (Latitude: {}, Longitude: {})",
            venue.title, venue.address, loc.latitude, loc.longitude
        );
        let stored = {
            let mut data = state.lock().await;
            if let Some(session) = data.sessions.get_mut(&chat_id) {
                session.pending_uploads.push(location_record.clone());
                true
            } else {
                false
            }
        };
        if stored {
            println!(
                "  [{timestamp}] ◀ [{user_name}] Venue: {} ({}, {}) (pending)",
                venue.title, loc.latitude, loc.longitude
            );
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, format!("Location received: {}", venue.title))
                    .await
            )?;
        } else {
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "No active session. Use /start <path> first.")
                    .await
            )?;
        }
        return Ok(());
    }

    let Some(raw_text) = msg.text() else {
        msg_debug(&format!(
            "[handle_message] chat_id={}, non-text message (no raw_text), skipping",
            chat_id.0
        ));
        return Ok(());
    };

    msg_debug(&format!(
        "[handle_message] chat_id={}, user={}, raw_text={:?}",
        chat_id.0,
        user_name,
        truncate_str(raw_text, 100)
    ));

    // Normalize "@botname ..." → strip @botname prefix (group chat shorthand)
    // "@botname /cmd args" → "/cmd args", "@botname hello" → ";hello"
    // In direct mode, ignore messages that mention a different bot
    let starts_with_at = raw_text.starts_with('@');
    let has_bot_username = !bot_username.is_empty();
    msg_debug(&format!("[mention_routing] input: raw_text={:?}, bot_username={:?}, starts_with_at={}, has_bot_username={}, require_prefix={}, is_group_chat={}",
        truncate_str(raw_text, 100), bot_username, starts_with_at, has_bot_username, require_prefix, is_group_chat));
    let mention_rewritten: Option<String>;
    let raw_text = if has_bot_username && starts_with_at {
        let prefix = format!("@{} ", bot_username);
        let is_self_mention = raw_text.to_lowercase().starts_with(&prefix);
        msg_debug(&format!("[mention_routing] @-prefix detected: checking self-mention, prefix={:?}, is_self_mention={}", prefix, is_self_mention));
        if is_self_mention {
            let body = raw_text[prefix.len()..].trim_start();
            if body.starts_with('/') || body.starts_with('!') || body.starts_with(';') {
                msg_debug(&format!("[mention_routing] self-mention with command prefix: {:?} → {:?} (pass-through command char)", raw_text, body));
                mention_rewritten = Some(body.to_string());
            } else {
                let prefixed = format!(";{}", body);
                msg_debug(&format!(
                    "[mention_routing] self-mention with plain text: {:?} → {:?} (added ; prefix)",
                    raw_text, prefixed
                ));
                mention_rewritten = Some(prefixed);
            }
            mention_rewritten.as_deref().unwrap()
        } else {
            // Message starts with @ but mentions a different bot
            let mentioned = raw_text[1..].split_whitespace().next().unwrap_or("");
            msg_debug(&format!("[mention_routing] other-bot mention: mentioned={:?}, require_prefix={}, direct_mode={}", mentioned, require_prefix, !require_prefix));
            if is_group_chat && !require_prefix {
                // Direct mode ON (group chat): ignore messages for other bots
                if !mentioned.is_empty() && mentioned.to_lowercase() != bot_username {
                    msg_debug(&format!("[mention_routing] IGNORED: direct mode ON, message is for @{}, not for @{}", mentioned, bot_username));
                    return Ok(());
                }
                msg_debug(&format!("[mention_routing] direct mode ON, but mentioned={:?} matches self or is empty, passing through", mentioned));
            } else {
                msg_debug(&format!("[mention_routing] direct mode OFF, passing through raw_text as-is (will be filtered by require_prefix later)"));
            }
            raw_text
        }
    } else {
        if starts_with_at {
            msg_debug(&format!(
                "[mention_routing] starts with @ but bot_username is empty, passing through"
            ));
        } else {
            msg_debug(&format!(
                "[mention_routing] no @-prefix, passing through raw_text as-is"
            ));
        }
        raw_text
    };

    // Strip @botname suffix from commands (e.g. "/pwd@mybot" → "/pwd")
    // If @botname doesn't match this bot, ignore the command (it's for another bot)
    let starts_with_slash = raw_text.starts_with('/');
    msg_debug(&format!(
        "[cmd_routing] input: raw_text={:?}, starts_with_slash={}",
        truncate_str(raw_text, 100),
        starts_with_slash
    ));
    let text = if starts_with_slash {
        if let Some(space_pos) = raw_text.find(' ') {
            // "/cmd@bot args" → "/cmd args"
            let cmd_part = &raw_text[..space_pos];
            let args_part = &raw_text[space_pos..];
            if let Some(at_pos) = cmd_part.find('@') {
                let mentioned = &cmd_part[at_pos + 1..];
                let is_self = mentioned.to_lowercase() == bot_username;
                msg_debug(&format!("[cmd_routing] slash+args: cmd={:?}, mentioned={:?}, is_self={}, bot_username={:?}", cmd_part, mentioned, is_self, bot_username));
                if has_bot_username && !is_self {
                    msg_debug(&format!(
                        "[cmd_routing] IGNORED: command {:?} is for @{}, not for @{}",
                        cmd_part, mentioned, bot_username
                    ));
                    return Ok(());
                }
                let result = format!("{}{}", &cmd_part[..at_pos], args_part);
                msg_debug(&format!(
                    "[cmd_routing] stripped @mention from command: {:?} → {:?}",
                    raw_text, result
                ));
                result
            } else {
                msg_debug(&format!(
                    "[cmd_routing] slash+args, no @mention: {:?} → all bots",
                    raw_text
                ));
                raw_text.to_string()
            }
        } else {
            // "/cmd@bot" (no args) → "/cmd"
            if let Some(at_pos) = raw_text.find('@') {
                let mentioned = &raw_text[at_pos + 1..];
                let is_self = mentioned.to_lowercase() == bot_username;
                msg_debug(&format!("[cmd_routing] slash-only: cmd={:?}, mentioned={:?}, is_self={}, bot_username={:?}", raw_text, mentioned, is_self, bot_username));
                if has_bot_username && !is_self {
                    msg_debug(&format!(
                        "[cmd_routing] IGNORED: command {:?} is for @{}, not for @{}",
                        raw_text, mentioned, bot_username
                    ));
                    return Ok(());
                }
                let result = raw_text[..at_pos].to_string();
                msg_debug(&format!(
                    "[cmd_routing] stripped @mention from command: {:?} → {:?}",
                    raw_text, result
                ));
                result
            } else {
                msg_debug(&format!(
                    "[cmd_routing] slash-only, no @mention: {:?} → all bots",
                    raw_text
                ));
                raw_text.to_string()
            }
        }
    } else {
        msg_debug(&format!(
            "[cmd_routing] not a slash command, text={:?}",
            truncate_str(raw_text, 100)
        ));
        raw_text.to_string()
    };
    let preview = &text;

    // Auto-restore session from bot_settings.json if not in memory
    if !text.starts_with("/start") {
        auto_restore_session(&state, chat_id, &user_name).await;
    }

    // In group chats (with prefix required), ignore plain text (only /, !, ; prefixed messages are processed)
    let has_valid_prefix = text.starts_with('/') || text.starts_with('!') || text.starts_with(';');
    msg_debug(&format!(
        "[prefix_filter] text={:?}, require_prefix={}, has_valid_prefix={}, direct_mode={}",
        truncate_str(&text, 100),
        require_prefix,
        has_valid_prefix,
        !require_prefix
    ));
    if require_prefix && !has_valid_prefix {
        msg_debug(&format!("[prefix_filter] IGNORED: require_prefix=true (direct mode OFF), no valid prefix in text={:?}", truncate_str(&text, 80)));
        // Clear pending uploads — this message was not for this bot,
        // so any previously shared location/file data should be discarded.
        let mut data = state.lock().await;
        if let Some(session) = data.sessions.get_mut(&chat_id) {
            if !session.pending_uploads.is_empty() {
                msg_debug(&format!(
                    "[handle_message] clearing {} pending_uploads (not addressed to this bot)",
                    session.pending_uploads.len()
                ));
                session.pending_uploads.clear();
            }
        }
        return Ok(());
    }

    // Block all messages except /stop, /stopall, /queue while an AI request is in progress
    if !text.starts_with("/stop") && !text.starts_with("/queue") {
        // Determine the actual user text to queue (strip command prefix, pure string ops)
        let queue_text = if text.starts_with("/query") {
            // /query command → queue the body
            text.strip_prefix("/query")
                .and_then(|s| {
                    // Handle /query@botname format
                    if s.starts_with('@') {
                        s.find(' ').map(|i| s[i..].trim_start())
                    } else {
                        Some(s.trim_start())
                    }
                })
                .filter(|s| !s.is_empty())
        } else if text.starts_with(';') {
            // ; prefix → queue stripped text
            let stripped = text[1..].trim_start();
            if stripped.is_empty() {
                None
            } else {
                Some(stripped)
            }
        } else if text.starts_with('!') || text.starts_with('/') {
            // Shell commands and other / commands are NOT queueable
            None
        } else {
            // Plain text
            Some(text.as_str())
        };
        msg_debug(&format!(
            "[queue:text] chat_id={}, text={:?}, queue_text={:?}",
            chat_id.0,
            truncate_str(&text, 60),
            queue_text.map(|s| truncate_str(s, 60))
        ));

        // Atomically: check busy + queue mode + push to queue (prevents race with /stopall)
        // Returns: Option<(can_queue, queue_on, queue_result, redirect_result)>
        //   queue_result:    Some((id, text)) if queued (ON mode), None if full/non-queueable
        //   redirect_result: Some((id, text, was_replacement)) if OFF mode redirect triggered
        let busy_info: Option<(
            bool,
            bool,
            Option<(String, String)>,
            Option<(String, String, bool)>,
        )> = {
            let mut data = state.lock().await;
            if data.cancel_tokens.contains_key(&chat_id) {
                let queue_enabled = queue_mode_enabled_for_chat(&data.settings, chat_id);
                msg_debug(&format!(
                    "[queue:text] chat_id={}, AI busy, queue_mode={}, queueable={}",
                    chat_id.0,
                    queue_enabled,
                    queue_text.is_some()
                ));
                let (qr, rr) = if queue_enabled {
                    let qr = if let Some(qt) = queue_text {
                        let cur_len = data.message_queues.get(&chat_id).map_or(0, |q| q.len());
                        let queue_full = cur_len >= MAX_QUEUE_SIZE;
                        if queue_full {
                            msg_debug(&format!(
                                "[queue:text] chat_id={}, queue FULL ({}/{})",
                                chat_id.0, cur_len, MAX_QUEUE_SIZE
                            ));
                            None // queue full
                        } else {
                            // Capture pending_uploads so they stay associated with this message
                            let uploads = data
                                .sessions
                                .get_mut(&chat_id)
                                .map(|s| std::mem::take(&mut s.pending_uploads))
                                .unwrap_or_default();
                            let qid = generate_queue_id();
                            let q = data
                                .message_queues
                                .entry(chat_id)
                                .or_insert_with(std::collections::VecDeque::new);
                            q.push_back(QueuedMessage {
                                id: qid.clone(),
                                text: qt.to_string(),
                                user_display_name: user_name.clone(),
                                pending_uploads: uploads.clone(),
                            });
                            msg_debug(&format!("[queue:text] chat_id={}, QUEUED id={}, pos={}, text={:?}, uploads={}", chat_id.0, qid, q.len(), truncate_str(qt, 60), uploads.len()));
                            Some((qid, qt.to_string()))
                        }
                    } else {
                        msg_debug(&format!(
                            "[queue:text] chat_id={}, non-queueable command, rejecting",
                            chat_id.0
                        ));
                        None // non-queueable
                    };
                    (qr, None)
                } else {
                    // OFF mode: redirect-eligible messages cancel the current task and become
                    // the next dispatch target (latest-wins). Slash/shell commands stay rejected.
                    let rr = if let Some(qt) = queue_text {
                        let uploads = data
                            .sessions
                            .get_mut(&chat_id)
                            .map(|s| std::mem::take(&mut s.pending_uploads))
                            .unwrap_or_default();
                        let (qid, replaced) = enqueue_redirect_locked(
                            &mut *data,
                            chat_id,
                            qt.to_string(),
                            user_name.clone(),
                            uploads,
                        );
                        msg_debug(&format!(
                            "[queue:text] chat_id={}, REDIRECT id={}, replaced={}, text={:?}",
                            chat_id.0,
                            qid,
                            replaced,
                            truncate_str(qt, 60)
                        ));
                        Some((qid, qt.to_string(), replaced))
                    } else {
                        msg_debug(&format!("[queue:text] chat_id={}, OFF mode + non-redirectable command, rejecting", chat_id.0));
                        None
                    };
                    (None, rr)
                };
                Some((queue_enabled && queue_text.is_some(), queue_enabled, qr, rr))
            } else {
                msg_debug(&format!(
                    "[queue:text] chat_id={}, AI not busy, proceeding normally",
                    chat_id.0
                ));
                None // not busy
            }
        };

        if let Some((can_queue, queue_on, queue_result, redirect_result)) = busy_info {
            if let Some((_qid, qtxt, replaced)) = redirect_result {
                shared_rate_limit_wait(&state, chat_id).await;
                let preview = truncate_str(&qtxt, 30);
                let msg = if replaced {
                    format!("🔄 Redirect target updated: \"{preview}\"")
                } else {
                    format!("🔄 Cancelling current task, will process: \"{preview}\"")
                };
                tg!("send_message", bot.send_message(chat_id, &msg).await)?;
            } else if can_queue {
                shared_rate_limit_wait(&state, chat_id).await;
                if let Some((qid, qtxt)) = queue_result {
                    let preview = truncate_str(&qtxt, 30);
                    tg!("send_message", bot.send_message(chat_id, &format!("Queued ({qid}) \"{preview}\"\n- /stopall to cancel all\n- /stop_{qid} to cancel this"))
                        .await)?;
                } else {
                    tg!(
                        "send_message",
                        bot.send_message(
                            chat_id,
                            &format!(
                                "Queue full (max {}). Use /stopall to clear.",
                                MAX_QUEUE_SIZE
                            )
                        )
                        .await
                    )?;
                }
            } else {
                shared_rate_limit_wait(&state, chat_id).await;
                if queue_on {
                    tg!("send_message", bot.send_message(chat_id, "AI request in progress. This command cannot be queued. Use /stop to cancel current, /stopall to cancel all.")
                        .await)?;
                } else {
                    tg!(
                        "send_message",
                        bot.send_message(chat_id, "AI request in progress. Use /stop to cancel.")
                            .await
                    )?;
                }
            }
            return Ok(());
        }
    }

    if text.starts_with("/stopall") {
        msg_debug(&format!("[handle_message] routing → /stopall"));
        println!("  [{timestamp}] ◀ [{user_name}] /stopall");
        handle_stopall_command(&bot, chat_id, &state).await?;
    } else if text.starts_with("/stop") {
        // Check if /stop has a valid queue ID argument (exactly 7 hex chars, e.g. "/stop A394FDA")
        let stop_queue_id = text.strip_prefix("/stop").and_then(|s| {
            // Handle /stop@botname format
            let s = if s.starts_with('@') {
                s.find(' ').map(|i| &s[i..]).unwrap_or("")
            } else {
                s
            };
            // Strip leading underscore to support /stop_ID format (e.g. /stop_3EBA20E)
            let s = s.strip_prefix('_').unwrap_or(s);
            // Strip @botname suffix (for group chat: /stop_ID@botname)
            let s = s.split('@').next().unwrap_or(s);
            let trimmed = s.trim();
            // Only treat as queue ID if it matches the exact format: 7 hex characters
            if trimmed.len() == 7 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
                Some(trimmed.to_string())
            } else {
                None
            }
        });
        if let Some(queue_id) = stop_queue_id {
            msg_debug(&format!(
                "[handle_message] routing → /stop queue_id={}",
                queue_id
            ));
            println!("  [{timestamp}] ◀ [{user_name}] /stop {queue_id}");
            handle_stop_queue_item(&bot, chat_id, &state, &queue_id).await?;
        } else {
            msg_debug(&format!("[handle_message] routing → /stop"));
            println!("  [{timestamp}] ◀ [{user_name}] /stop");
            handle_stop_command(&bot, chat_id, &state).await?;
        }
    } else if text.starts_with("/help") {
        msg_debug(&format!("[handle_message] routing → /help"));
        println!("  [{timestamp}] ◀ [{user_name}] /help");
        handle_help_command(&bot, chat_id, &state).await?;
    } else if text.starts_with("/start") {
        msg_debug(&format!("[handle_message] routing → /start"));
        println!("  [{timestamp}] ◀ [{user_name}] /start");
        handle_start_command(&bot, chat_id, &text, &state, token).await?;
    } else if text.starts_with("/clear") {
        msg_debug(&format!("[handle_message] routing → /clear"));
        println!("  [{timestamp}] ◀ [{user_name}] /clear");
        handle_clear_command(&bot, chat_id, &state).await?;
        println!("  [{timestamp}] ▶ [{user_name}] Session cleared");
    } else if text.starts_with("/pwd") {
        msg_debug(&format!("[handle_message] routing → /pwd"));
        println!("  [{timestamp}] ◀ [{user_name}] /pwd");
        handle_pwd_command(&bot, chat_id, &state).await?;
    } else if text.starts_with("/session") {
        msg_debug(&format!("[handle_message] routing → /session"));
        println!("  [{timestamp}] ◀ [{user_name}] /session");
        handle_session_command(&bot, chat_id, &state).await?;
    } else if text.starts_with("/down") {
        msg_debug(&format!("[handle_message] routing → /down"));
        println!(
            "  [{timestamp}] ◀ [{user_name}] /down {}",
            text.strip_prefix("/down").unwrap_or("").trim()
        );
        handle_down_command(&bot, chat_id, &text, &state).await?;
    } else if text.starts_with("/public") {
        msg_debug("[handle_message] routing → /public");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /public {}",
            text.strip_prefix("/public").unwrap_or("").trim()
        );
        handle_public_command(&bot, chat_id, &text, &state, token, is_group_chat, is_owner).await?;
    } else if text.starts_with("/availabletools") {
        msg_debug("[handle_message] routing → /availabletools");
        println!("  [{timestamp}] ◀ [{user_name}] /availabletools");
        {
            let _m = get_model(&state.lock().await.settings, chat_id);
            if provider_from_model(_m.as_deref()) != "claude" {
                tg!(
                    "send_message",
                    bot.send_message(chat_id, "Tool permissions are not supported in this mode.")
                        .await
                )?;
            } else {
                handle_availabletools_command(&bot, chat_id, &state).await?;
            }
        }
    } else if text.starts_with("/allowedtools") {
        msg_debug("[handle_message] routing → /allowedtools");
        println!("  [{timestamp}] ◀ [{user_name}] /allowedtools");
        {
            let _m = get_model(&state.lock().await.settings, chat_id);
            if provider_from_model(_m.as_deref()) != "claude" {
                tg!(
                    "send_message",
                    bot.send_message(chat_id, "Tool permissions are not supported in this mode.")
                        .await
                )?;
            } else {
                handle_allowedtools_command(&bot, chat_id, &state).await?;
            }
        }
    } else if text.starts_with("/setpollingtime") {
        msg_debug("[handle_message] routing → /setpollingtime");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /setpollingtime {}",
            text.strip_prefix("/setpollingtime").unwrap_or("").trim()
        );
        handle_setpollingtime_command(&bot, chat_id, &text, &state).await?;
    } else if text.starts_with("/setapigap") {
        msg_debug("[handle_message] routing → /setapigap");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /setapigap {}",
            text.strip_prefix("/setapigap").unwrap_or("").trim()
        );
        handle_setapigap_command(&bot, chat_id, &text, &state).await?;
    } else if text.starts_with("/tgstats") {
        msg_debug("[handle_message] routing → /tgstats");
        println!("  [{timestamp}] ◀ [{user_name}] /tgstats");
        handle_tgstats_command(&bot, chat_id, &state).await?;
    } else if is_cmd(&text, "effort") {
        msg_debug("[handle_message] routing → /effort");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /effort {}",
            command_args(&text)
        );
        if !is_owner {
            msg_debug("[handle_message] /effort rejected: not owner");
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Only the bot owner can use /effort.")
                    .await
            )?;
        } else {
            handle_effort_command(&bot, chat_id, &text, &state, token).await?;
        }
    } else if is_cmd(&text, "fast") {
        msg_debug("[handle_message] routing → /fast");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /fast {}",
            command_args(&text)
        );
        if !is_owner {
            msg_debug("[handle_message] /fast rejected: not owner");
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Only the bot owner can use /fast.")
                    .await
            )?;
        } else {
            handle_fast_command(&bot, chat_id, &text, &state, token).await?;
        }
    } else if text.starts_with("/model") {
        msg_debug("[handle_message] routing → /model");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /model {}",
            text.strip_prefix("/model").unwrap_or("").trim()
        );
        handle_model_command(&bot, chat_id, &text, &state, token).await?;
    } else if text.starts_with("/greeting") {
        msg_debug("[handle_message] routing → /greeting");
        println!("  [{timestamp}] ◀ [{user_name}] /greeting");
        handle_greeting_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/debug") {
        msg_debug("[handle_message] routing → /debug");
        println!("  [{timestamp}] ◀ [{user_name}] /debug");
        handle_debug_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/envvars") {
        msg_debug("[handle_message] routing → /envvars");
        println!("  [{timestamp}] ◀ [{user_name}] /envvars");
        if !is_owner {
            msg_debug("[handle_message] /envvars rejected: not owner");
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Only the bot owner can use /envvars.")
                    .await
            )?;
        } else {
            handle_envvars_command(&bot, chat_id, &state).await?;
        }
    } else if text.starts_with("/usechrome") {
        msg_debug("[handle_message] routing → /usechrome");
        println!("  [{timestamp}] ◀ [{user_name}] /usechrome");
        handle_usechrome_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/silent") {
        msg_debug("[handle_message] routing → /silent");
        println!("  [{timestamp}] ◀ [{user_name}] /silent");
        handle_silent_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/queue") {
        msg_debug("[handle_message] routing → /queue");
        println!("  [{timestamp}] ◀ [{user_name}] /queue");
        handle_queue_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/direct") {
        msg_debug("[handle_message] routing → /direct");
        println!("  [{timestamp}] ◀ [{user_name}] /direct");
        handle_direct_command(&bot, chat_id, &msg, &state, token, is_owner).await?;
    } else if text.starts_with("/contextlevel") {
        msg_debug("[handle_message] routing → /contextlevel");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /contextlevel {}",
            text.strip_prefix("/contextlevel").unwrap_or("").trim()
        );
        handle_contextlevel_command(&bot, chat_id, &text, &state, token, is_group_chat).await?;
    } else if text.starts_with("/query") {
        let body = text.strip_prefix("/query").unwrap_or("").trim();
        if body.is_empty() {
            msg_debug("[handle_message] /query with empty body, ignoring");
            shared_rate_limit_wait(&state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    "Usage: /query <message>\nExample: /query@botname hello"
                )
                .await
            )?;
        } else {
            msg_debug(&format!(
                "[handle_message] routing → text_message (/query), body={:?}",
                truncate_str(body, 80)
            ));
            println!("  [{timestamp}] ◀ [{user_name}] {body}");
            handle_text_message(&bot, chat_id, body, &state, &user_name, false).await?;
        }
    } else if text.starts_with("/loop") {
        // Strip /loop prefix and handle @botname suffix (e.g. /loop@botname request)
        let body = text.strip_prefix("/loop").unwrap_or("");
        let body = if body.starts_with('@') {
            body.find(' ').map(|i| &body[i..]).unwrap_or("").trim()
        } else {
            body.trim()
        };
        if body.is_empty() {
            msg_debug("[handle_message] /loop with empty body");
            shared_rate_limit_wait(&state, chat_id).await;
            tg!("send_message", bot.send_message(chat_id, "Usage: /loop [N] <request>\nRepeats until the task is fully completed.\nOptional: N = max iterations (default 5, 0 = unlimited)").await)?;
        } else {
            // /loop uses a self-verification step that currently supports
            // Claude (native --fork-session), Codex (ephemeral exec over the
            // full-fidelity session archive), and OpenCode (native --fork with
            // the `plan` agent). Other providers (currently Gemini) are rejected.
            let provider = {
                let _m = get_model(&state.lock().await.settings, chat_id);
                provider_from_model(_m.as_deref()).to_string()
            };
            if provider != "claude" && provider != "codex" && provider != "opencode" {
                shared_rate_limit_wait(&state, chat_id).await;
                tg!(
                    "send_message",
                    bot.send_message(
                        chat_id,
                        "/loop currently supports Claude, Codex, or OpenCode models only."
                    )
                    .await
                )?;
            } else {
                // Reject if a loop is already running for this chat
                {
                    let data = state.lock().await;
                    if data.loop_states.contains_key(&chat_id) {
                        drop(data);
                        shared_rate_limit_wait(&state, chat_id).await;
                        tg!(
                            "send_message",
                            bot.send_message(
                                chat_id,
                                "A loop is already running. Use /stop to cancel it first."
                            )
                            .await
                        )?;
                        return Ok(());
                    }
                }
                // Parse optional iteration count: /loop 10 request → 10 iterations, /loop 0 request → unlimited
                let (max_iter, request) = {
                    let mut parts = body.splitn(2, ' ');
                    let first = parts.next().unwrap_or("");
                    let rest = parts.next().unwrap_or("").trim();
                    if !rest.is_empty() {
                        if let Ok(n) = first.parse::<u16>() {
                            (n, rest)
                        } else {
                            (LOOP_MAX_ITERATIONS, body)
                        }
                    } else {
                        (LOOP_MAX_ITERATIONS, body)
                    }
                };
                msg_debug(&format!(
                    "[handle_message] routing → /loop, max_iter={}, request={:?}",
                    max_iter,
                    truncate_str(request, 80)
                ));
                println!(
                    "  [{timestamp}] ◀ [{user_name}] /loop {} {}",
                    max_iter,
                    truncate_str(request, 60)
                );
                {
                    let mut data = state.lock().await;
                    data.loop_states.insert(
                        chat_id,
                        LoopState {
                            original_request: request.to_string(),
                            max_iterations: max_iter,
                            remaining: max_iter,
                        },
                    );
                }
                shared_rate_limit_wait(&state, chat_id).await;
                let iter_label = if max_iter == 0 {
                    "🔄 Loop started (unlimited)".to_string()
                } else {
                    format!("🔄 Loop started (max {} iterations)", max_iter)
                };
                let _ = tg!("send_message", bot.send_message(chat_id, &iter_label).await);
                handle_text_message(&bot, chat_id, request, &state, &user_name, false).await?;
            }
        }
    } else if text.starts_with("/setendhook_clear") {
        msg_debug("[handle_message] routing → /setendhook_clear");
        println!("  [{timestamp}] ◀ [{user_name}] /setendhook_clear");
        handle_setendhook_clear_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/setendhook") {
        msg_debug("[handle_message] routing → /setendhook");
        println!("  [{timestamp}] ◀ [{user_name}] /setendhook");
        handle_setendhook_command(&bot, chat_id, &text, &state, token).await?;
    } else if text.starts_with("/instruction_clear") {
        msg_debug("[handle_message] routing → /instruction_clear");
        println!("  [{timestamp}] ◀ [{user_name}] /instruction_clear");
        handle_instruction_clear_command(&bot, chat_id, &state, token).await?;
    } else if text.starts_with("/instruction") {
        msg_debug("[handle_message] routing → /instruction");
        println!("  [{timestamp}] ◀ [{user_name}] /instruction");
        handle_instruction_command(&bot, chat_id, &text, &state, token).await?;
    } else if text.starts_with("/allowed") {
        msg_debug("[handle_message] routing → /allowed");
        println!(
            "  [{timestamp}] ◀ [{user_name}] /allowed {}",
            text.strip_prefix("/allowed").unwrap_or("").trim()
        );
        {
            let _m = get_model(&state.lock().await.settings, chat_id);
            if provider_from_model(_m.as_deref()) != "claude" {
                tg!(
                    "send_message",
                    bot.send_message(chat_id, "Tool permissions are not supported in this mode.")
                        .await
                )?;
            } else {
                handle_allowed_command(&bot, chat_id, &text, &state, token).await?;
            }
        }
    } else if text.starts_with('/')
        && is_workspace_id(text[1..].split_whitespace().next().unwrap_or(""))
    {
        let workspace_id = text[1..].split_whitespace().next().unwrap();
        msg_debug(&format!(
            "[handle_message] routing → workspace_resume: {}",
            workspace_id
        ));
        println!("  [{timestamp}] ◀ [{user_name}] /{workspace_id}");
        handle_workspace_resume(&bot, chat_id, workspace_id, &state, token).await?;
    } else if text.starts_with('!') {
        msg_debug(&format!("[handle_message] routing → shell command"));
        println!("  [{timestamp}] ◀ [{user_name}] Shell: {preview}");
        handle_shell_command(&bot, chat_id, &text, &state, &user_name).await?;
    } else if text.starts_with(';') {
        let stripped = text.strip_prefix(';').unwrap_or(&text).trim().to_string();
        if stripped.is_empty() {
            msg_debug("[handle_message] ;prefix with empty body, ignoring");
            return Ok(());
        }
        let preview = &stripped;
        msg_debug(&format!(
            "[handle_message] routing → text_message (;prefix), stripped={:?}",
            truncate_str(&stripped, 80)
        ));
        println!("  [{timestamp}] ◀ [{user_name}] {preview}");
        handle_text_message(&bot, chat_id, &stripped, &state, &user_name, false).await?;
    } else {
        msg_debug(&format!(
            "[handle_message] routing → text_message (plain), require_prefix={}",
            require_prefix
        ));
        println!("  [{timestamp}] ◀ [{user_name}] {preview}");
        handle_text_message(&bot, chat_id, &text, &state, &user_name, false).await?;
    }

    Ok(())
}

/// Handle /help command
async fn handle_help_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let platform = capitalize_platform(detect_platform(bot.token()));
    let help = format!(
        "\
<b>cokacdir {} Bot</b>
Manage server files &amp; chat with Claude AI.

<b>Session</b>
<code>/start &lt;path&gt;</code> — Start session at directory
<code>/start &lt;name|id&gt;</code> — Resume Claude Code session
<code>/start</code> — Start with auto-generated workspace
<code>/pwd</code> — Show current working directory
<code>/session</code> — Show current session ID
<code>/clear</code> — Clear AI conversation history
<code>/stop</code> — Stop current AI request
<code>/stop_&lt;ID&gt;</code> — Cancel a specific queued message
<code>/stopall</code> — Stop request and clear queue
<code>/queue</code> — Toggle queue mode (queue messages while AI is busy)
<code>/loop [N] &lt;request&gt;</code> — Repeat until task is fully completed

<b>File Transfer</b>
<code>/down &lt;file&gt;</code> — Download file from server
Send a file/photo — Upload to session directory

<b>Shell</b>
<code>!&lt;command&gt;</code> — Run shell command directly
  e.g. <code>!ls -la</code>, <code>!git status</code>

<b>AI Chat</b>
Any other message is sent to Claude AI.
AI can read, edit, and run commands in your session.

<b>Tool Management</b>
<code>/availabletools</code> — List all available tools
<code>/allowedtools</code> — Show currently allowed tools
<code>/allowed +name</code> — Add tool (e.g. <code>/allowed +Bash</code>)
<code>/allowed -name</code> — Remove tool
<code>/allowed +a -b +c</code> — Multiple at once

<b>Group Chat</b>
<code>;</code><i>message</i> — Send message to AI
<code>/query</code><i> message</i> — Send message to AI (supports @bot)
<code>;</code><i>caption</i> — Upload file with AI prompt
<code>/public on</code> — Allow all members to use bot
<code>/public off</code> — Owner only (default)
<code>/direct</code> — Toggle direct mode (no ; prefix needed)
<code>/contextlevel &lt;N&gt;</code> — Set group chat log entries in prompt (0=off, default 12)

<b>Schedule</b>
Ask in natural language to manage schedules.

<b>Settings</b>
<code>/model</code> — Show current AI model
<code>/model &lt;name&gt;</code> — Set model (claude/codex/gemini or provider:model)
<code>/effort</code> — Show current reasoning effort
<code>/effort &lt;level&gt;</code> — Set effort for current provider
<code>/fast</code> — Toggle Codex fast service tier
<code>/setpollingtime &lt;ms&gt;</code> — Set stream update polling interval
  Minimum 1000ms, recommended 1500ms+.
<code>/setapigap &lt;ms&gt;</code> — Set Telegram API minimum gap
  Minimum 2500ms, higher = safer against RetryAfter.
<code>/tgstats</code> — Show Telegram RetryAfter/error counters
<code>/debug</code> — Toggle debug logging
<code>/envvars</code> — Show all environment variables and their current values
<code>/silent</code> — Toggle silent mode (hide tool calls)
<code>/usechrome</code> — Toggle Chrome browser for Claude (--chrome)
<code>/instruction &lt;text&gt;</code> — Set system instruction for AI
<code>/instruction</code> — View current instruction
<code>/instruction_clear</code> — Clear instruction
<code>/setendhook &lt;msg&gt;</code> — Set notification when processing ends
<code>/setendhook</code> — View current end hook
<code>/setendhook_clear</code> — Clear end hook

<b>Bot Messaging</b>
Bots in the same group can collaborate via <code>/instruction</code>.

<code>/help</code> — Show this help",
        platform
    );

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, help)
            .parse_mode(ParseMode::Html)
            .await
    )?;

    Ok(())
}

/// Handle /start <path> command
async fn handle_start_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    // Extract path from "/start <path>"
    let path_str = text.strip_prefix("/start").unwrap_or("").trim();
    msg_debug(&format!(
        "[handle_start_command] chat_id={}, path_str={:?}",
        chat_id.0, path_str
    ));

    // Determine current provider (Claude vs Codex vs Gemini)
    let original_provider_str: &str;
    let mut provider_str;
    let mut provider = {
        let data = state.lock().await;
        let model = get_model(&data.settings, chat_id);
        let p = detect_provider(model.as_deref());
        msg_debug(&format!(
            "[handle_start_command] model={:?}, provider={}",
            model, p
        ));
        provider_str = p;
        provider_to_session(p)
    };
    original_provider_str = provider_str;

    // Tracks whether the user's input was a path-like value (vs. a session
    // identifier). Same-path no-op only applies when path-intent is true —
    // session identifiers may resolve to a path matching current cwd while
    // intending to switch to a different session there.
    let mut is_path_intent = false;
    let canonical_path = if path_str.is_empty() {
        // Create random workspace directory
        let Some(home) = dirs::home_dir() else {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Error: cannot determine home directory.")
                    .await
            )?;
            return Ok(());
        };
        let workspace_dir = home.join(".cokacdir").join("workspace");
        use rand::Rng;
        let random_name: String = rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(8)
            .map(|b| (b as char).to_ascii_lowercase())
            .collect();
        let new_dir = workspace_dir.join(&random_name);
        if let Err(e) = fs::create_dir_all(&new_dir) {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, format!("Error: failed to create workspace: {}", e))
                    .await
            )?;
            return Ok(());
        }
        new_dir.display().to_string()
    } else if path_str.starts_with('/')
        || path_str.starts_with("~/")
        || path_str.starts_with("~\\")
        || path_str == "~"
        || path_str.starts_with("./")
        || path_str.starts_with(".\\")
        || path_str == "."
        || path_str == ".."
        || path_str.starts_with("../")
        || path_str.starts_with("..\\")
        || (path_str.len() >= 3
            && path_str.as_bytes()[1] == b':'
            && (path_str.as_bytes()[2] == b'\\' || path_str.as_bytes()[2] == b'/'))
    {
        is_path_intent = true;
        // Path mode: expand ~ and validate
        let expanded =
            if path_str.starts_with("~/") || path_str.starts_with("~\\") || path_str == "~" {
                if let Some(home) = dirs::home_dir() {
                    home.join(
                        path_str
                            .strip_prefix("~/")
                            .or_else(|| path_str.strip_prefix("~\\"))
                            .unwrap_or(""),
                    )
                    .display()
                    .to_string()
                } else {
                    path_str.to_string()
                }
            } else {
                path_str.to_string()
            };
        let path = Path::new(&expanded);
        if !path.exists() {
            if let Err(e) = fs::create_dir_all(&path) {
                shared_rate_limit_wait(state, chat_id).await;
                tg!(
                    "send_message",
                    bot.send_message(
                        chat_id,
                        format!("Error: failed to create '{}': {}", expanded, e)
                    )
                    .await
                )?;
                return Ok(());
            }
        } else if !path.is_dir() {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    format!("Error: '{}' is not a directory.", expanded)
                )
                .await
            )?;
            return Ok(());
        }
        path.canonicalize()
            .map(crate::utils::format::strip_unc_prefix)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| expanded)
    } else {
        // Session name/ID mode: resolve Claude Code session
        // Try current provider first, then cross-provider fallback
        let fallback_providers: &[SessionProvider] = match provider {
            SessionProvider::Claude => &[
                SessionProvider::Codex,
                SessionProvider::Gemini,
                SessionProvider::OpenCode,
            ],
            SessionProvider::Codex => &[
                SessionProvider::Claude,
                SessionProvider::Gemini,
                SessionProvider::OpenCode,
            ],
            SessionProvider::Gemini => &[
                SessionProvider::Claude,
                SessionProvider::Codex,
                SessionProvider::OpenCode,
            ],
            SessionProvider::OpenCode => &[
                SessionProvider::Claude,
                SessionProvider::Codex,
                SessionProvider::Gemini,
            ],
        };

        // Helper closure: try resolve_session + ai_sessions for a given provider
        let try_resolve = |prov: SessionProvider, prov_str: &str| -> Option<String> {
            msg_debug(&format!(
                "[try_resolve] provider={}, query={:?}",
                prov_str, path_str
            ));
            if let Some(info) = resolve_session(path_str, prov) {
                let path = Path::new(&info.cwd);
                let is_dir = path.is_dir();
                msg_debug(&format!(
                    "[try_resolve] resolve_session found: cwd={:?}, is_dir={}",
                    info.cwd, is_dir
                ));
                if is_dir {
                    let canonical = path
                        .canonicalize()
                        .map(crate::utils::format::strip_unc_prefix)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| info.cwd.clone());
                    convert_and_save_session(&info, &canonical);
                    msg_debug(&format!(
                        "[try_resolve] resolved via resolve_session: canonical={}",
                        canonical
                    ));
                    return Some(canonical);
                } else {
                    msg_debug(&format!(
                        "[try_resolve] resolve_session cwd not a dir: {:?}",
                        info.cwd
                    ));
                }
            } else {
                msg_debug(&format!(
                    "[try_resolve] resolve_session returned None for provider={}",
                    prov_str
                ));
            }
            // Try ai_sessions/{id}.json
            msg_debug(&format!(
                "[try_resolve] trying ai_sessions/{}.json",
                path_str
            ));
            let result = ai_screen::ai_sessions_dir().and_then(|dir| {
                let file = dir.join(format!("{}.json", path_str));
                let content = fs::read_to_string(&file).ok()?;
                let sd: SessionData = serde_json::from_str(&content).ok()?;
                if !sd.provider.is_empty() && sd.provider != prov_str {
                    msg_debug(&format!(
                        "[try_resolve] ai_sessions/{}.json provider mismatch: {} != {}",
                        path_str, sd.provider, prov_str
                    ));
                    return None;
                }
                let p = Path::new(&sd.current_path);
                let is_dir = p.is_dir();
                msg_debug(&format!(
                    "[try_resolve] ai_sessions/{}.json: current_path={:?}, provider={}, is_dir={}",
                    path_str, sd.current_path, sd.provider, is_dir
                ));
                if is_dir {
                    Some(sd.current_path.clone())
                } else {
                    None
                }
            });
            msg_debug(&format!(
                "[try_resolve] ai_sessions result: {}",
                if result.is_some() { "found" } else { "None" }
            ));
            result
        };

        if let Some(cp) = try_resolve(provider, provider_str) {
            msg_debug(&format!(
                "[handle_start_command] resolved with current provider: path={}",
                cp
            ));
            cp
        } else {
            // Cross-provider fallback: try all other providers in order
            let mut cross_result: Option<(String, SessionProvider, &'static str)> = None;
            for &fp in fallback_providers {
                let fp_str = session_provider_str(fp);
                let available = match fp {
                    SessionProvider::Claude => claude::is_claude_available(),
                    SessionProvider::Codex => codex::is_codex_available(),
                    SessionProvider::Gemini => gemini::is_gemini_available(),
                    SessionProvider::OpenCode => opencode::is_opencode_available(),
                };
                if !available {
                    msg_debug(&format!(
                        "[handle_start_command] cross-provider skip: {} not available",
                        fp_str
                    ));
                    continue;
                }
                msg_debug(&format!(
                    "[handle_start_command] cross-provider attempt: {}",
                    fp_str
                ));
                if let Some(cp) = try_resolve(fp, fp_str) {
                    cross_result = Some((cp, fp, fp_str));
                    break;
                }
            }

            if let Some((cp, resolved_prov, resolved_prov_str)) = cross_result {
                // Cross-provider fallback: switch provider and model to resolved provider
                msg_debug(&format!(
                    "[handle_start_command] cross-provider fallback: switching from {} to {}",
                    provider_str, resolved_prov_str
                ));
                provider = resolved_prov;
                provider_str = resolved_prov_str;
                {
                    let mut data = state.lock().await;
                    msg_debug(&format!(
                        "[handle_start_command] cross-provider: setting model to {:?}",
                        resolved_prov_str
                    ));
                    data.settings
                        .models
                        .insert(chat_id.0.to_string(), resolved_prov_str.to_string());
                    save_bot_settings(token, &data.settings);
                    // Mirror /model provider-switch cleanup: cancel in-flight task and drop
                    // queued messages / loop state. Their captured pending_uploads point at the
                    // old workspace's files, and verify-loop feedback was authored against the
                    // old provider's session — both would be misapplied under the new provider.
                    cancel_in_progress_task_locked(&data, chat_id);
                    let dropped_queue = data
                        .message_queues
                        .remove(&chat_id)
                        .map(|q| q.len())
                        .unwrap_or(0);
                    data.loop_states.remove(&chat_id);
                    data.loop_feedback.remove(&chat_id);
                    if let Some(session) = data.sessions.get_mut(&chat_id) {
                        msg_debug(&format!("[handle_start_command] cross-provider: clearing old session (len={}, uploads={}, queue={}, sid={:?}, path={:?})",
                            session.history.len(), session.pending_uploads.len(), dropped_queue, session.session_id, session.current_path));
                        session.session_id = None;
                        session.current_path = None;
                        session.history.clear();
                        session.pending_uploads.clear();
                    } else {
                        msg_debug(&format!("[handle_start_command] cross-provider: no existing session to clear (queue={})", dropped_queue));
                    }
                }
                cp
            } else {
                // Final fallback: try as plain path
                msg_debug(&format!("[handle_start_command] all session resolves failed, trying as plain path: {:?}", path_str));
                let path = Path::new(path_str);
                if path.exists() && path.is_dir() {
                    is_path_intent = true;
                    let resolved = path
                        .canonicalize()
                        .map(crate::utils::format::strip_unc_prefix)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| path_str.to_string());
                    msg_debug(&format!(
                        "[handle_start_command] plain path resolved: {}",
                        resolved
                    ));
                    resolved
                } else {
                    msg_debug(&format!(
                        "[handle_start_command] plain path failed: exists={}, is_dir={}",
                        path.exists(),
                        path.is_dir()
                    ));
                    shared_rate_limit_wait(state, chat_id).await;
                    tg!(
                        "send_message",
                        bot.send_message(
                            chat_id,
                            format!("Error: no session or directory found for '{}'.", path_str)
                        )
                        .await
                    )?;
                    return Ok(());
                }
            }
        }
    };

    // Same-path no-op: if the user typed a path-like input that resolved to the
    // session's existing current_path, treat it as a confirmation and skip the
    // rest. Without this guard, the trailing block (3598+) would clear
    // pending_uploads, null session_id, and overwrite in-memory history with
    // the on-disk version — destroying the user's in-progress state for a
    // no-op intent.
    //
    // Only path-intent invocations qualify (modes B and final-fallback D).
    // Session-identifier inputs (mode C) intentionally proceed even at the
    // same path because the user may be switching to a different session
    // whose cwd happens to match the current one.
    //
    // Cross-provider fallback (handled inside the C branch) sets
    // current_path = None before reaching here, so the comparison naturally
    // fails for that case and execution proceeds.
    if is_path_intent {
        let same_path = {
            let data = state.lock().await;
            data.sessions
                .get(&chat_id)
                .and_then(|s| s.current_path.clone())
                .as_deref()
                == Some(canonical_path.as_str())
        };
        if same_path {
            msg_debug(&format!(
                "[handle_start_command] same-path no-op: {}",
                canonical_path
            ));
            shared_rate_limit_wait(state, chat_id).await;
            let display = crate::utils::format::to_shell_path(&canonical_path);
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    format!("Already at <code>{}</code>.", html_escape(&display))
                )
                .parse_mode(ParseMode::Html)
                .await
            )?;
            return Ok(());
        }
    }

    // Try to load existing session for this path
    msg_debug(&format!(
        "[handle_start_command] canonical_path={:?}, provider={}",
        canonical_path, provider_str
    ));
    let existing = load_existing_session(&canonical_path, provider_str);
    msg_debug(&format!(
        "[handle_start_command] load_existing_session → {}",
        if existing.is_some() { "found" } else { "None" }
    ));

    // If no local session, try converting the latest external session for this path
    let existing = if existing.is_some() {
        existing
    } else if let Some(info) = find_latest_session_by_cwd(&canonical_path, provider) {
        msg_debug(&format!(
            "[handle_start_command] fallback found: jsonl={}, session_id={}",
            info.jsonl_path.display(),
            info.session_id
        ));
        convert_and_save_session(&info, &canonical_path);
        let reloaded = load_existing_session(&canonical_path, provider_str);
        msg_debug(&format!(
            "[handle_start_command] after convert, reload → {}",
            if reloaded.is_some() { "found" } else { "None" }
        ));
        reloaded
    } else {
        msg_debug("[handle_start_command] no external session found either");
        None
    };

    let mut response_lines = Vec::new();
    let mut is_restored = false;

    // Notify user if provider was auto-switched via cross-provider fallback
    if provider_str != original_provider_str {
        msg_debug(&format!(
            "[handle_start_command] provider auto-switched: {} → {}",
            original_provider_str, provider_str
        ));
        response_lines.push(format!("Model switched to **{}**.", provider_str));
    }

    {
        let mut data = state.lock().await;
        // Session-change cleanup: if /start switches to a different workspace path OR
        // a different session_id at the same path, drop the in-flight task, queued
        // messages, and loop state. They were authored against the old session and
        // would be misapplied — the queued pending_uploads point at the old session's
        // uploads, verify-loop feedback was authored against the old session, and
        // an in-flight task's writeback would resurrect the old session_id and history
        // into the just-loaded new session (corrupting the on-disk session file).
        // Idempotent with the cross-provider cleanup above (which leaves
        // current_path = None, so path_changed is always true on re-entry).
        let old_path = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone());
        let old_sid = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.session_id.clone());
        let new_sid = existing.as_ref().and_then(|(sd, _)| {
            if sd.session_id.is_empty() {
                None
            } else {
                Some(sd.session_id.clone())
            }
        });
        let path_changed = old_path.as_deref() != Some(canonical_path.as_str());
        let sid_changed = old_sid != new_sid;
        if path_changed || sid_changed {
            cancel_in_progress_task_locked(&data, chat_id);
            data.message_queues.remove(&chat_id);
            data.loop_states.remove(&chat_id);
            data.loop_feedback.remove(&chat_id);
        }
        let session = data.sessions.entry(chat_id).or_insert_with(|| ChatSession {
            session_id: None,
            current_path: None,
            history: Vec::new(),
            pending_uploads: Vec::new(),
        });
        session.session_id = None; // Clear stale session_id from previous workspace
                                   // Drop uploads only when the workspace path changed — upload paths point at
                                   // the old workspace's files. When only session_id changes at the same path
                                   // (mode C, session-identifier swap), uploads still reference valid files in
                                   // the current workspace and remain relevant to the user's next message.
        if path_changed {
            session.pending_uploads.clear();
        }

        if let Some((session_data, _)) = &existing {
            if !session_data.session_id.is_empty() {
                session.session_id = Some(session_data.session_id.clone());
            } else {
                cleanup_session_files(&canonical_path, provider_str);
            }
            session.current_path = Some(canonical_path.clone());
            session.history = session_data.history.clone();
            is_restored = true;
            msg_debug(&format!(
                "[handle_start_command] restored: session_id={}, path={}, history_len={}",
                session_data.session_id,
                canonical_path,
                session_data.history.len()
            ));

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ▶ Session restored: {canonical_path}");
            response_lines.push(format!(
                "[{}] Session restored at `{}`.",
                provider_str, canonical_path
            ));
            if let Some(folder_name) = std::path::Path::new(&canonical_path)
                .file_name()
                .and_then(|n| n.to_str())
            {
                if is_workspace_id(folder_name)
                    && dirs::home_dir()
                        .map(|h| {
                            h.join(".cokacdir")
                                .join("workspace")
                                .join(folder_name)
                                .is_dir()
                        })
                        .unwrap_or(false)
                {
                    response_lines.push(format!("Use /{} to resume this session.", folder_name));
                }
            }
            let header_len: usize = response_lines.iter().map(|l| l.len() + 1).sum();
            let remaining = TELEGRAM_MSG_LIMIT.saturating_sub(header_len + 2);
            let preview = build_history_preview(&session_data.history, remaining);
            if !preview.is_empty() {
                response_lines.push(String::new());
                response_lines.push(preview);
            }
        } else {
            session.session_id = None;
            session.current_path = Some(canonical_path.clone());
            session.history.clear();
            msg_debug(&format!(
                "[handle_start_command] new session: path={}",
                canonical_path
            ));

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ▶ Session started: {canonical_path}");
            response_lines.push(format!(
                "[{}] Session started at `{}`.",
                provider_str, canonical_path
            ));
            // Show workspace ID shortcut if this is a workspace directory
            if let Some(folder_name) = std::path::Path::new(&canonical_path)
                .file_name()
                .and_then(|n| n.to_str())
            {
                if is_workspace_id(folder_name)
                    && dirs::home_dir()
                        .map(|h| {
                            h.join(".cokacdir")
                                .join("workspace")
                                .join(folder_name)
                                .is_dir()
                        })
                        .unwrap_or(false)
                {
                    response_lines.push(format!("Use /{} to resume this session.", folder_name));
                }
            }
        }
    }

    // Persist chat_id → path mapping for auto-restore after restart
    {
        let mut data = state.lock().await;
        data.settings
            .last_sessions
            .insert(chat_id.0.to_string(), canonical_path.clone());
        save_bot_settings(token, &data.settings);
    }

    // Append start marker to group chat log so other bots can see the new working directory
    if chat_id.0 < 0 {
        let (uname, dname) = {
            let data = state.lock().await;
            (data.bot_username.clone(), data.bot_display_name.clone())
        };
        if !uname.is_empty() {
            let dn = if dname.is_empty() { None } else { Some(dname) };
            append_group_chat_log(
                chat_id.0,
                &GroupChatLogEntry {
                    ts: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
                    bot: uname,
                    bot_display_name: dn,
                    role: "system".to_string(),
                    from: None,
                    text: if is_restored {
                        format!("Session restored at {}", canonical_path)
                    } else {
                        format!("Session started at {}", canonical_path)
                    },
                    clear: false,
                },
            );
        }
    }

    let response_text = response_lines.join("\n");
    let html = markdown_to_telegram_html(&response_text);
    send_long_message(bot, chat_id, &html, Some(ParseMode::Html), state).await?;

    Ok(())
}

/// Build a history preview code block that fits within the given byte budget.
/// Items are shown oldest-first (most recent at bottom), filling from the bottom up.
fn build_history_preview(history: &[HistoryItem], budget: usize) -> String {
    if history.is_empty() {
        return String::new();
    }
    let code_block_overhead = "```\n".len() + "\n```".len(); // 8 bytes
    if budget <= code_block_overhead + 10 {
        return String::new();
    }
    let content_budget = budget - code_block_overhead;

    // Build lines from newest to oldest, stop when budget exhausted
    let mut collected: Vec<String> = Vec::new();
    let mut used = 0;
    for item in history.iter().rev() {
        let prefix = match item.item_type {
            HistoryType::User => "👤",
            HistoryType::Assistant => "🤖",
            HistoryType::Error => "",
            HistoryType::System => "⚙️",
            HistoryType::ToolUse => "🔧",
            HistoryType::ToolResult => "📋",
        };
        let line = format!("{} {}", prefix, item.content);
        let line_len = line.len() + 1; // +1 for newline
        if used + line_len > content_budget {
            break;
        }
        collected.push(line);
        used += line_len;
    }
    if collected.is_empty() {
        return String::new();
    }
    collected.reverse();
    format!("```\n{}\n```", collected.join("\n"))
}

/// Check if a string is a valid 8-character workspace ID (e.g. "B4E9451D" or "k3m9x2ab")
fn is_workspace_id(s: &str) -> bool {
    s.len() == 8 && s.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Check if a string is a valid UUID (8-4-4-4-12 hex format)
fn is_uuid(s: &str) -> bool {
    if s.len() != 36 {
        return false;
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected.iter())
        .all(|(p, &len)| p.len() == len && p.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Provider that owns the resolved session.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SessionProvider {
    Claude,
    Codex,
    Gemini,
    OpenCode,
}

/// Detect provider from model prefix only (no availability fallback).
/// Returns "claude" when model is None or has no recognized prefix.
fn provider_from_model(model: Option<&str>) -> &'static str {
    if codex::is_codex_model(model) {
        "codex"
    } else if gemini::is_gemini_model(model) {
        "gemini"
    } else if opencode::is_opencode_model(model) {
        "opencode"
    } else {
        "claude"
    }
}

/// Detect effective provider: from model prefix if set, otherwise from CLI availability.
fn detect_provider(model: Option<&str>) -> &'static str {
    if model.is_some() {
        provider_from_model(model)
    } else if !claude::is_claude_available() && codex::is_codex_available() {
        "codex"
    } else if !claude::is_claude_available() && gemini::is_gemini_available() {
        "gemini"
    } else if !claude::is_claude_available() && opencode::is_opencode_available() {
        "opencode"
    } else {
        "claude"
    }
}

/// Convert provider name to SessionProvider enum.
fn provider_to_session(provider: &str) -> SessionProvider {
    match provider {
        "codex" => SessionProvider::Codex,
        "gemini" => SessionProvider::Gemini,
        "opencode" => SessionProvider::OpenCode,
        _ => SessionProvider::Claude,
    }
}

/// Convert SessionProvider enum to provider name string.
fn session_provider_str(provider: SessionProvider) -> &'static str {
    match provider {
        SessionProvider::Claude => "claude",
        SessionProvider::Codex => "codex",
        SessionProvider::Gemini => "gemini",
        SessionProvider::OpenCode => "opencode",
    }
}

/// Info returned when an external session is resolved.
struct ResolvedSession {
    cwd: String,
    jsonl_path: std::path::PathBuf,
    session_id: String,
    provider: SessionProvider,
}

/// Resolve a session by name or ID, scoped to the current provider.
fn resolve_session(query: &str, provider: SessionProvider) -> Option<ResolvedSession> {
    msg_debug(&format!(
        "[resolve_session] query={:?}, provider={:?}, is_uuid={}",
        query,
        provider,
        is_uuid(query)
    ));
    let result = match provider {
        SessionProvider::Claude => {
            if is_uuid(query) {
                resolve_claude_by_id(query).or_else(|| resolve_claude_by_name(query))
            } else {
                resolve_claude_by_name(query).or_else(|| resolve_claude_by_id(query))
            }
        }
        SessionProvider::Codex => resolve_codex_by_id(query),
        SessionProvider::Gemini => resolve_gemini_by_id(query),
        SessionProvider::OpenCode => resolve_opencode_by_id(query),
    };
    msg_debug(&format!(
        "[resolve_session] result={}",
        match &result {
            Some(r) => format!("found(cwd={:?}, session_id={})", r.cwd, r.session_id),
            None => "None".to_string(),
        }
    ));
    result
}

/// Claude: find `~/.claude/projects/*/{session_id}.jsonl`.
fn resolve_claude_by_id(session_id: &str) -> Option<ResolvedSession> {
    msg_debug(&format!("[resolve_claude_by_id] session_id={}", session_id));
    let projects_dir = dirs::home_dir()?.join(".claude").join("projects");
    if !projects_dir.is_dir() {
        msg_debug(&format!(
            "[resolve_claude_by_id] projects_dir not found: {}",
            projects_dir.display()
        ));
        return None;
    }
    let filename = format!("{}.jsonl", session_id);
    for entry in fs::read_dir(&projects_dir).ok()?.flatten() {
        if !entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let jsonl_path = entry.path().join(&filename);
        if jsonl_path.exists() {
            msg_debug(&format!(
                "[resolve_claude_by_id] found: {}",
                jsonl_path.display()
            ));
            let cwd = extract_cwd_from_jsonl(&jsonl_path)?;
            return Some(ResolvedSession {
                cwd,
                jsonl_path,
                session_id: session_id.to_string(),
                provider: SessionProvider::Claude,
            });
        }
    }
    msg_debug("[resolve_claude_by_id] not found");
    None
}

/// Claude: scan `~/.claude/projects/*/*.jsonl` for matching `custom-title`.
fn resolve_claude_by_name(name: &str) -> Option<ResolvedSession> {
    msg_debug(&format!("[resolve_claude_by_name] name={:?}", name));
    let projects_dir = dirs::home_dir()?.join(".claude").join("projects");
    if !projects_dir.is_dir() {
        msg_debug(&format!(
            "[resolve_claude_by_name] projects_dir not found: {}",
            projects_dir.display()
        ));
        return None;
    }
    let name_lower = name.to_lowercase();
    for proj_entry in fs::read_dir(&projects_dir).ok()?.flatten() {
        if !proj_entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let Ok(file_entries) = fs::read_dir(proj_entry.path()) else {
            continue;
        };
        for file_entry in file_entries.flatten() {
            let path = file_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(info) = find_session_by_title(&path, &name_lower) {
                msg_debug(&format!(
                    "[resolve_claude_by_name] found: session_id={}, cwd={:?}",
                    info.session_id, info.cwd
                ));
                return Some(info);
            }
        }
    }
    msg_debug("[resolve_claude_by_name] not found");
    None
}

/// Claude: check if a JSONL file contains a matching custom-title.
fn find_session_by_title(path: &Path, name_lower: &str) -> Option<ResolvedSession> {
    use std::io::{BufRead, BufReader};
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut matched = false;
    let mut cwd_found: Option<String> = None;
    for line in reader.lines().flatten() {
        if cwd_found.is_none() && line.contains("\"cwd\"") {
            if let Some(cwd) = extract_json_string_field(&line, "cwd") {
                if !cwd.is_empty() {
                    cwd_found = Some(cwd);
                }
            }
        }
        if !matched && line.contains("custom-title") {
            if let Some(title) = extract_json_string_field(&line, "customTitle") {
                if title.to_lowercase() == name_lower {
                    matched = true;
                }
            }
        }
        if matched && cwd_found.is_some() {
            break;
        }
    }
    if matched {
        let cwd = cwd_found?;
        let session_id = path.file_stem()?.to_str()?.to_string();
        Some(ResolvedSession {
            cwd,
            jsonl_path: path.to_path_buf(),
            session_id,
            provider: SessionProvider::Claude,
        })
    } else {
        None
    }
}

/// Codex: recursively scan `~/.codex/sessions/` for a JSONL whose filename contains the UUID.
fn resolve_codex_by_id(session_id: &str) -> Option<ResolvedSession> {
    let sessions_dir = dirs::home_dir()?.join(".codex").join("sessions");
    if !sessions_dir.is_dir() {
        return None;
    }
    let suffix = format!("{}.jsonl", session_id);
    fn walk(dir: &Path, suffix: &str) -> Option<std::path::PathBuf> {
        for entry in fs::read_dir(dir).ok()?.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if ft.is_dir() {
                if let Some(found) = walk(&entry.path(), suffix) {
                    return Some(found);
                }
            } else if ft.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(suffix) {
                        return Some(entry.path());
                    }
                }
            }
        }
        None
    }
    let jsonl_path = walk(&sessions_dir, &suffix)?;
    let cwd = extract_cwd_from_jsonl(&jsonl_path)?;
    Some(ResolvedSession {
        cwd,
        jsonl_path,
        session_id: session_id.to_string(),
        provider: SessionProvider::Codex,
    })
}

/// Gemini: scan `~/.gemini/tmp/*/chats/session-*.json` for a matching sessionId.
fn resolve_gemini_by_id(session_id: &str) -> Option<ResolvedSession> {
    msg_debug(&format!("[resolve_gemini_by_id] session_id={}", session_id));
    let tmp_dir = dirs::home_dir()?.join(".gemini").join("tmp");
    if !tmp_dir.is_dir() {
        msg_debug("[resolve_gemini_by_id] tmp_dir not found");
        return None;
    }
    // Use first 8 chars of UUID for quick filename filtering (char-boundary safe)
    let short_id: String = session_id.chars().take(8).collect();
    for proj_entry in fs::read_dir(&tmp_dir).ok()?.flatten() {
        if !proj_entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let chats_dir = proj_entry.path().join("chats");
        if !chats_dir.is_dir() {
            continue;
        }
        let Ok(chat_entries) = fs::read_dir(&chats_dir) else {
            continue;
        };
        for chat_entry in chat_entries.flatten() {
            let path = chat_entry.path();
            let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !fname.starts_with("session-") || !fname.ends_with(".json") {
                continue;
            }
            // Quick filter: filename contains first 8 chars of UUID
            if !fname.contains(short_id.as_str()) {
                continue;
            }
            // Parse JSON to verify full sessionId
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) else {
                continue;
            };
            let sid = val.get("sessionId").and_then(|v| v.as_str()).unwrap_or("");
            if sid != session_id {
                continue;
            }
            // Read .project_root from parent directory to get cwd
            let project_root_file = proj_entry.path().join(".project_root");
            let cwd = fs::read_to_string(&project_root_file)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())?;
            msg_debug(&format!(
                "[resolve_gemini_by_id] found: cwd={:?}, file={}",
                cwd,
                path.display()
            ));
            return Some(ResolvedSession {
                cwd,
                jsonl_path: path,
                session_id: session_id.to_string(),
                provider: SessionProvider::Gemini,
            });
        }
    }
    msg_debug("[resolve_gemini_by_id] not found");
    None
}

/// OpenCode: query `~/.local/share/opencode/opencode.db` for a matching session ID.
fn resolve_opencode_by_id(session_id: &str) -> Option<ResolvedSession> {
    msg_debug(&format!(
        "[resolve_opencode_by_id] session_id={}",
        session_id
    ));
    let db_path = dirs::home_dir()?
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.is_file() {
        msg_debug("[resolve_opencode_by_id] db not found");
        return None;
    }
    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let mut stmt = conn
        .prepare("SELECT id, directory FROM session WHERE id = ?1 LIMIT 1")
        .ok()?;
    let result = stmt
        .query_row(rusqlite::params![session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .ok()?;
    let directory = result.1;
    if directory.is_empty() {
        return None;
    }
    msg_debug(&format!(
        "[resolve_opencode_by_id] found: directory={:?}",
        directory
    ));
    Some(ResolvedSession {
        cwd: directory,
        jsonl_path: db_path,
        session_id: session_id.to_string(),
        provider: SessionProvider::OpenCode,
    })
}

/// Convert an external JSONL session to cokacdir SessionData and save it.
/// Re-converts if the source JSONL is newer than the existing JSON.
fn convert_and_save_session(info: &ResolvedSession, canonical_path: &str) {
    msg_debug(&format!(
        "[convert_session] start: jsonl={}, session_id={}, canonical_path={:?}",
        info.jsonl_path.display(),
        info.session_id,
        canonical_path
    ));
    let Some(sessions_dir) = ai_screen::ai_sessions_dir() else {
        msg_debug("[convert_session] ai_sessions_dir() returned None");
        return;
    };
    let target = sessions_dir.join(format!("{}.json", info.session_id));
    msg_debug(&format!("[convert_session] target={}", target.display()));
    if target.exists() {
        let source_mtime = info
            .jsonl_path
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok());
        let target_mtime = target.metadata().ok().and_then(|m| m.modified().ok());
        msg_debug(&format!(
            "[convert_session] target exists, source_mtime={:?}, target_mtime={:?}",
            source_mtime, target_mtime
        ));
        if let (Some(src), Some(tgt)) = (source_mtime, target_mtime) {
            if src <= tgt {
                msg_debug("[convert_session] skipped: target is up-to-date");
                return;
            }
        } else {
            msg_debug("[convert_session] skipped: cannot compare mtimes");
            return;
        }
    }

    let parser = match info.provider {
        SessionProvider::Claude => parse_claude_jsonl,
        SessionProvider::Codex => parse_codex_jsonl,
        SessionProvider::Gemini => parse_gemini_json,
        SessionProvider::OpenCode => parse_opencode_session,
    };
    msg_debug(&format!(
        "[convert_session] parsing with provider={:?}",
        info.provider
    ));
    let Some(session_data) = parser(&info.jsonl_path, &info.session_id, canonical_path) else {
        msg_debug("[convert_session] parser returned None");
        return;
    };
    msg_debug(&format!(
        "[convert_session] parsed: history_len={}, provider={}",
        session_data.history.len(),
        session_data.provider
    ));
    let _ = fs::create_dir_all(&sessions_dir);
    if let Ok(json) = serde_json::to_string_pretty(&session_data) {
        let write_result = fs::write(&target, &json);
        msg_debug(&format!(
            "[convert_session] write result={:?}, bytes={}",
            write_result,
            json.len()
        ));
    } else {
        msg_debug("[convert_session] serde_json::to_string_pretty failed");
    }

    crate::services::session_archive::archive_and_save_session(
        session_provider_str(info.provider),
        &info.jsonl_path,
        &info.session_id,
        canonical_path,
    );
}

/// Find the most recently modified external session whose cwd matches the given path.
fn find_latest_session_by_cwd(
    canonical_path: &str,
    provider: SessionProvider,
) -> Option<ResolvedSession> {
    msg_debug(&format!(
        "[find_latest_by_cwd] canonical_path={:?}, provider={:?}",
        canonical_path, provider
    ));
    let result = match provider {
        SessionProvider::Claude => find_latest_claude_by_cwd(canonical_path),
        SessionProvider::Codex => find_latest_codex_by_cwd(canonical_path),
        SessionProvider::Gemini => find_latest_gemini_by_cwd(canonical_path),
        SessionProvider::OpenCode => find_latest_opencode_by_cwd(canonical_path),
    };
    msg_debug(&format!(
        "[find_latest_by_cwd] result={}",
        if result.is_some() { "found" } else { "None" }
    ));
    result
}

/// Claude: scan all `~/.claude/projects/*/*.jsonl` for the latest session matching cwd.
fn find_latest_claude_by_cwd(canonical_path: &str) -> Option<ResolvedSession> {
    let projects_dir = dirs::home_dir()?.join(".claude").join("projects");
    msg_debug(&format!(
        "[find_claude_by_cwd] projects_dir={}, is_dir={}",
        projects_dir.display(),
        projects_dir.is_dir()
    ));
    if !projects_dir.is_dir() {
        return None;
    }
    let mut best_path: Option<std::path::PathBuf> = None;
    let mut best_time = std::time::UNIX_EPOCH;
    let mut scan_count = 0u32;
    let mut cwd_mismatch_sample: Option<String> = None;
    for proj_entry in fs::read_dir(&projects_dir).ok()?.flatten() {
        if !proj_entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let Ok(file_entries) = fs::read_dir(proj_entry.path()) else {
            continue;
        };
        for file_entry in file_entries.flatten() {
            let path = file_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            scan_count += 1;
            if let Some(cwd) = extract_cwd_from_jsonl(&path) {
                msg_debug(&format!(
                    "[find_claude_by_cwd] file={}, extracted_cwd={:?}, want={:?}, match={}",
                    path.display(),
                    cwd,
                    canonical_path,
                    cwd == canonical_path
                ));
                if cwd == canonical_path {
                    let mtime = path
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .unwrap_or(std::time::UNIX_EPOCH);
                    if mtime > best_time {
                        best_path = Some(path);
                        best_time = mtime;
                    }
                } else if cwd_mismatch_sample.is_none() {
                    cwd_mismatch_sample = Some(cwd);
                }
            } else {
                msg_debug(&format!(
                    "[find_claude_by_cwd] file={}, extract_cwd returned None",
                    path.display()
                ));
            }
        }
    }
    msg_debug(&format!(
        "[find_claude_by_cwd] scanned {} jsonl files, best={:?}",
        scan_count,
        best_path.as_ref().map(|p| p.display().to_string())
    ));
    if best_path.is_none() {
        if let Some(sample) = cwd_mismatch_sample {
            msg_debug(&format!(
                "[find_claude_by_cwd] cwd mismatch example: extracted={:?} vs wanted={:?}",
                sample, canonical_path
            ));
        }
    }
    let jsonl_path = best_path?;
    let session_id = jsonl_path.file_stem()?.to_str()?.to_string();
    Some(ResolvedSession {
        cwd: canonical_path.to_string(),
        jsonl_path,
        session_id,
        provider: SessionProvider::Claude,
    })
}

/// Codex: scan `~/.codex/sessions/**/*.jsonl` for the latest session matching cwd.
fn find_latest_codex_by_cwd(canonical_path: &str) -> Option<ResolvedSession> {
    let sessions_dir = dirs::home_dir()?.join(".codex").join("sessions");
    if !sessions_dir.is_dir() {
        return None;
    }
    let mut best_path: Option<std::path::PathBuf> = None;
    let mut best_time = std::time::UNIX_EPOCH;
    collect_best_codex_jsonl(
        &sessions_dir,
        canonical_path,
        &mut best_path,
        &mut best_time,
    );
    let jsonl_path = best_path?;
    // Extract UUID from filename tail (last 36 chars of stem)
    let session_id = {
        let stem = jsonl_path.file_stem()?.to_str()?;
        if stem.len() < 36 {
            return None;
        }
        let candidate = &stem[stem.len() - 36..];
        if !is_uuid(candidate) {
            return None;
        }
        candidate.to_string()
    };
    Some(ResolvedSession {
        cwd: canonical_path.to_string(),
        jsonl_path,
        session_id,
        provider: SessionProvider::Codex,
    })
}

fn collect_best_codex_jsonl(
    dir: &Path,
    canonical_path: &str,
    best_path: &mut Option<std::path::PathBuf>,
    best_time: &mut std::time::SystemTime,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            collect_best_codex_jsonl(&entry.path(), canonical_path, best_path, best_time);
        } else if ft.is_file() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(cwd) = extract_cwd_from_jsonl(&path) {
                if cwd == canonical_path {
                    let mtime = path
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .unwrap_or(std::time::UNIX_EPOCH);
                    if mtime > *best_time {
                        *best_path = Some(path);
                        *best_time = mtime;
                    }
                }
            }
        }
    }
}

/// Gemini: scan `~/.gemini/tmp/*/.project_root` for cwd match, find latest chat file.
fn find_latest_gemini_by_cwd(canonical_path: &str) -> Option<ResolvedSession> {
    msg_debug(&format!(
        "[find_latest_gemini_by_cwd] canonical_path={:?}",
        canonical_path
    ));
    let tmp_dir = dirs::home_dir()?.join(".gemini").join("tmp");
    if !tmp_dir.is_dir() {
        return None;
    }
    let mut best_path: Option<std::path::PathBuf> = None;
    let mut best_time = std::time::UNIX_EPOCH;
    for proj_entry in fs::read_dir(&tmp_dir).ok()?.flatten() {
        if !proj_entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let pr_file = proj_entry.path().join(".project_root");
        let Ok(pr_content) = fs::read_to_string(&pr_file) else {
            continue;
        };
        if pr_content.trim() != canonical_path {
            continue;
        }
        let chats_dir = proj_entry.path().join("chats");
        if !chats_dir.is_dir() {
            continue;
        }
        let Ok(chat_entries) = fs::read_dir(&chats_dir) else {
            continue;
        };
        for chat_entry in chat_entries.flatten() {
            let path = chat_entry.path();
            let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !fname.starts_with("session-") || !fname.ends_with(".json") {
                continue;
            }
            let mtime = path
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            if mtime > best_time {
                best_path = Some(path);
                best_time = mtime;
            }
        }
    }
    let jsonl_path = best_path?;
    let content = fs::read_to_string(&jsonl_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let session_id = val.get("sessionId").and_then(|v| v.as_str())?.to_string();
    msg_debug(&format!(
        "[find_latest_gemini_by_cwd] found: session_id={}, file={}",
        session_id,
        jsonl_path.display()
    ));
    Some(ResolvedSession {
        cwd: canonical_path.to_string(),
        jsonl_path,
        session_id,
        provider: SessionProvider::Gemini,
    })
}

/// OpenCode: query SQLite DB for latest session matching cwd.
fn find_latest_opencode_by_cwd(canonical_path: &str) -> Option<ResolvedSession> {
    msg_debug(&format!(
        "[find_latest_opencode_by_cwd] canonical_path={:?}",
        canonical_path
    ));
    let db_path = dirs::home_dir()?
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.is_file() {
        return None;
    }
    let conn =
        rusqlite::Connection::open_with_flags(&db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let mut stmt = conn
        .prepare("SELECT id FROM session WHERE directory = ?1 ORDER BY time_updated DESC LIMIT 1")
        .ok()?;
    let session_id: String = stmt
        .query_row(rusqlite::params![canonical_path], |row| row.get(0))
        .ok()?;
    msg_debug(&format!(
        "[find_latest_opencode_by_cwd] found: session_id={}",
        session_id
    ));
    Some(ResolvedSession {
        cwd: canonical_path.to_string(),
        jsonl_path: db_path,
        session_id,
        provider: SessionProvider::OpenCode,
    })
}

/// Parse a Claude Code JSONL file into cokacdir SessionData.
fn parse_claude_jsonl(jsonl_path: &Path, session_id: &str, cwd: &str) -> Option<SessionData> {
    use std::io::{BufRead, BufReader};
    let file = fs::File::open(jsonl_path).ok()?;
    let reader = BufReader::new(file);
    let mut history: Vec<HistoryItem> = Vec::new();

    for line in reader.lines().flatten() {
        let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        // Skip sidechain (alternative conversation branches)
        if val.get("isSidechain").and_then(|v| v.as_bool()) == Some(true) {
            continue;
        }

        let Some(msg_type) = val.get("type").and_then(|v| v.as_str()) else {
            continue;
        };

        match msg_type {
            "user" => {
                let Some(message) = val.get("message") else {
                    continue;
                };
                let Some(content) = message.get("content") else {
                    continue;
                };
                if let Some(text) = content.as_str() {
                    // Skip commands and system injections
                    if text.is_empty()
                        || text.contains("<command-name>")
                        || text.contains("<local-command")
                    {
                        continue;
                    }
                    history.push(HistoryItem {
                        item_type: HistoryType::User,
                        content: truncate_utf8(text, 300),
                    });
                } else if let Some(arr) = content.as_array() {
                    for item in arr {
                        let it = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if it == "tool_result" {
                            let rc = item.get("content");
                            let text = if let Some(s) = rc.and_then(|v| v.as_str()) {
                                s.to_string()
                            } else if let Some(arr2) = rc.and_then(|v| v.as_array()) {
                                // content can be [{"type":"text","text":"..."},...]
                                arr2.iter()
                                    .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else {
                                String::new()
                            };
                            if !text.is_empty() {
                                history.push(HistoryItem {
                                    item_type: HistoryType::ToolResult,
                                    content: truncate_utf8(&text, 500),
                                });
                            }
                        }
                    }
                }
            }
            "assistant" => {
                let Some(message) = val.get("message") else {
                    continue;
                };
                let Some(content) = message.get("content") else {
                    continue;
                };
                let Some(arr) = content.as_array() else {
                    continue;
                };
                for item in arr {
                    let it = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match it {
                        "text" => {
                            let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            if !text.is_empty() {
                                history.push(HistoryItem {
                                    item_type: HistoryType::Assistant,
                                    content: truncate_utf8(text, 2000),
                                });
                            }
                        }
                        "tool_use" => {
                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("Tool");
                            history.push(HistoryItem {
                                item_type: HistoryType::ToolUse,
                                content: format!("[{}]", name),
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if history.is_empty() {
        return None;
    }

    Some(SessionData {
        session_id: session_id.to_string(),
        history,
        current_path: cwd.to_string(),
        created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        provider: "claude".to_string(),
    })
}

/// Parse a Codex CLI JSONL file into cokacdir SessionData.
fn parse_codex_jsonl(jsonl_path: &Path, session_id: &str, cwd: &str) -> Option<SessionData> {
    use std::io::{BufRead, BufReader};
    let file = fs::File::open(jsonl_path).ok()?;
    let reader = BufReader::new(file);
    let mut history: Vec<HistoryItem> = Vec::new();

    for line in reader.lines().flatten() {
        let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let Some(line_type) = val.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(payload) = val.get("payload") else {
            continue;
        };

        match line_type {
            "event_msg" => {
                let msg_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match msg_type {
                    "user_message" => {
                        let text = payload
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !text.is_empty() {
                            history.push(HistoryItem {
                                item_type: HistoryType::User,
                                content: truncate_utf8(text, 300),
                            });
                        }
                    }
                    "agent_message" => {
                        let text = payload
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !text.is_empty() {
                            history.push(HistoryItem {
                                item_type: HistoryType::Assistant,
                                content: truncate_utf8(text, 2000),
                            });
                        }
                    }
                    _ => {}
                }
            }
            "response_item" => {
                let item_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match item_type {
                    // response_item → message is intentionally ignored:
                    // agent text is already captured via event_msg → agent_message (always emitted in pairs)
                    "function_call" => {
                        let name = payload
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Tool");
                        history.push(HistoryItem {
                            item_type: HistoryType::ToolUse,
                            content: format!("[{}]", name),
                        });
                    }
                    "function_call_output" => {
                        // output can be a plain string or structured {content_items: [...]}
                        let output = if let Some(s) = payload.get("output").and_then(|v| v.as_str())
                        {
                            s.to_string()
                        } else if let Some(obj) = payload.get("output") {
                            // Structured: try content_items[].text, then content field
                            if let Some(items) = obj.get("content_items").and_then(|v| v.as_array())
                            {
                                items
                                    .iter()
                                    .filter_map(|c| c.get("text").and_then(|v| v.as_str()))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else if let Some(s) = obj.get("content").and_then(|v| v.as_str()) {
                                s.to_string()
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };
                        if !output.is_empty() {
                            history.push(HistoryItem {
                                item_type: HistoryType::ToolResult,
                                content: truncate_utf8(&output, 500),
                            });
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    if history.is_empty() {
        return None;
    }

    Some(SessionData {
        session_id: session_id.to_string(),
        history,
        current_path: cwd.to_string(),
        created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        provider: "codex".to_string(),
    })
}

/// Parse a Gemini CLI chat JSON file into cokacdir SessionData.
fn parse_gemini_json(json_path: &Path, session_id: &str, cwd: &str) -> Option<SessionData> {
    msg_debug(&format!(
        "[parse_gemini_json] file={}, session_id={}",
        json_path.display(),
        session_id
    ));
    let content = fs::read_to_string(json_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let messages = val.get("messages")?.as_array()?;
    let mut history: Vec<HistoryItem> = Vec::new();
    for msg in messages {
        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match msg_type {
            "user" => {
                // User content is array of {text: "..."} objects
                if let Some(arr) = msg.get("content").and_then(|v| v.as_array()) {
                    for item in arr {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                history.push(HistoryItem {
                                    item_type: HistoryType::User,
                                    content: truncate_utf8(trimmed, 300),
                                });
                            }
                        }
                    }
                }
            }
            "gemini" => {
                // Gemini content is a plain string
                if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        history.push(HistoryItem {
                            item_type: HistoryType::Assistant,
                            content: truncate_utf8(text, 2000),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    if history.is_empty() {
        return None;
    }
    msg_debug(&format!(
        "[parse_gemini_json] parsed: history_len={}",
        history.len()
    ));
    Some(SessionData {
        session_id: session_id.to_string(),
        history,
        current_path: cwd.to_string(),
        created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        provider: "gemini".to_string(),
    })
}

/// Parse an OpenCode session from SQLite DB into cokacdir SessionData.
fn parse_opencode_session(db_path: &Path, session_id: &str, cwd: &str) -> Option<SessionData> {
    msg_debug(&format!(
        "[parse_opencode_session] db={}, session_id={}",
        db_path.display(),
        session_id
    ));
    if !db_path.is_file() {
        return None;
    }
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let mut stmt = conn
        .prepare(
            "SELECT json_extract(m.data, '$.role'), json_extract(p.data, '$.type'), \
         json_extract(p.data, '$.text'), json_extract(p.data, '$.tool') \
         FROM message m JOIN part p ON p.message_id = m.id \
         WHERE m.session_id = ?1 ORDER BY p.time_created ASC",
        )
        .ok()?;
    let rows = stmt
        .query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })
        .ok()?;
    let mut history: Vec<HistoryItem> = Vec::new();
    for row in rows.flatten() {
        let role = row.0.as_deref().unwrap_or("");
        let ptype = row.1.as_deref().unwrap_or("");
        let text = row.2.as_deref().unwrap_or("");
        let tool = row.3.as_deref().unwrap_or("Tool");
        match ptype {
            "text" => {
                if text.is_empty() {
                    continue;
                }
                if role == "user" {
                    history.push(HistoryItem {
                        item_type: HistoryType::User,
                        content: truncate_utf8(text.trim(), 300),
                    });
                } else if role == "assistant" {
                    history.push(HistoryItem {
                        item_type: HistoryType::Assistant,
                        content: truncate_utf8(text, 2000),
                    });
                }
            }
            "tool" => {
                history.push(HistoryItem {
                    item_type: HistoryType::ToolUse,
                    content: format!("[{}]", tool),
                });
            }
            // Skip step-start, step-finish, reasoning
            _ => {}
        }
    }
    if history.is_empty() {
        return None;
    }
    msg_debug(&format!(
        "[parse_opencode_session] parsed: history_len={}",
        history.len()
    ));
    Some(SessionData {
        session_id: session_id.to_string(),
        history,
        current_path: cwd.to_string(),
        created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        provider: "opencode".to_string(),
    })
}

/// Truncate a string at a valid UTF-8 boundary.
fn truncate_utf8(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Extract the first non-empty `cwd` value from a JSONL file.
fn extract_cwd_from_jsonl(path: &Path) -> Option<String> {
    use std::io::{BufRead, BufReader};
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().flatten() {
        if !line.contains("\"cwd\"") {
            continue;
        }
        if let Some(cwd) = extract_json_string_field(&line, "cwd") {
            if !cwd.is_empty() {
                msg_debug(&format!(
                    "[extract_cwd] file={}, cwd={:?}",
                    path.display(),
                    cwd
                ));
                return Some(cwd);
            }
        }
    }
    msg_debug(&format!(
        "[extract_cwd] file={}, no cwd found",
        path.display()
    ));
    None
}

/// Simple JSON string field extraction: find `"field":"value"` and return value.
/// Handles escaped quotes (`\"`) inside the value.
fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    // Find closing quote, skipping escaped quotes
    let mut end = 0;
    let bytes = rest.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' {
            let mut backslashes = 0;
            while end > backslashes && bytes[end - 1 - backslashes] == b'\\' {
                backslashes += 1;
            }
            if backslashes % 2 == 0 {
                break;
            }
        }
        end += 1;
    }
    if end >= bytes.len() {
        return None;
    }
    // Unescape JSON string (e.g. "\\\\" → "\\", "\\\"" → "\"")
    let raw = &rest[..end];
    let quoted = format!("\"{}\"", raw);
    let unescaped = serde_json::from_str::<String>(&quoted).unwrap_or_else(|_| raw.to_string());
    if field == "cwd" {
        msg_debug(&format!(
            "[extract_field] field={}, raw={:?}, unescaped={:?}",
            field, raw, unescaped
        ));
    }
    Some(unescaped)
}

/// Handle /WORKSPACE_ID command - resume a workspace session by its ID
async fn handle_workspace_resume(
    bot: &Bot,
    chat_id: ChatId,
    workspace_id: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    msg_debug(&format!(
        "[workspace_resume] chat_id={}, workspace_id={}",
        chat_id.0, workspace_id
    ));
    let Some(home) = dirs::home_dir() else {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, "Error: cannot determine home directory.")
                .await
        )?;
        return Ok(());
    };

    let workspace_path = home.join(".cokacdir").join("workspace").join(workspace_id);
    if !workspace_path.exists() || !workspace_path.is_dir() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!("Error: no workspace found for '{}'.", workspace_id)
            )
            .await
        )?;
        return Ok(());
    }

    let canonical_path = workspace_path
        .canonicalize()
        .map(crate::utils::format::strip_unc_prefix)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| workspace_path.display().to_string());

    let ws_provider = {
        let data = state.lock().await;
        let ws_model = get_model(&data.settings, chat_id);
        detect_provider(ws_model.as_deref())
    };
    msg_debug(&format!(
        "[workspace_resume] canonical_path={:?}, provider={}",
        canonical_path, ws_provider
    ));
    let existing = load_existing_session(&canonical_path, ws_provider);
    msg_debug(&format!(
        "[workspace_resume] load_existing_session → {}",
        if existing.is_some() { "found" } else { "None" }
    ));

    let existing = if existing.is_some() {
        existing
    } else {
        let provider = provider_to_session(ws_provider);
        if let Some(info) = find_latest_session_by_cwd(&canonical_path, provider) {
            msg_debug(&format!(
                "[workspace_resume] fallback found: jsonl={}, session_id={}",
                info.jsonl_path.display(),
                info.session_id
            ));
            convert_and_save_session(&info, &canonical_path);
            let reloaded = load_existing_session(&canonical_path, ws_provider);
            msg_debug(&format!(
                "[workspace_resume] after convert, reload → {}",
                if reloaded.is_some() { "found" } else { "None" }
            ));
            reloaded
        } else {
            msg_debug("[workspace_resume] no external session found either");
            None
        }
    };

    let mut response_lines = Vec::new();

    {
        let mut data = state.lock().await;
        // Session-change cleanup: if this resume switches to a different workspace path
        // OR a different session_id at the same path, drop the in-flight task, queued
        // messages, and loop state authored against the old session — they would be
        // misapplied here, and an in-flight task's writeback would otherwise overwrite
        // the just-loaded session_id/history with the old task's data.
        let old_path = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone());
        let old_sid = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.session_id.clone());
        let new_sid = existing.as_ref().and_then(|(sd, _)| {
            if sd.session_id.is_empty() {
                None
            } else {
                Some(sd.session_id.clone())
            }
        });
        let path_changed = old_path.as_deref() != Some(canonical_path.as_str());
        let sid_changed = old_sid != new_sid;
        if path_changed || sid_changed {
            cancel_in_progress_task_locked(&data, chat_id);
            data.message_queues.remove(&chat_id);
            data.loop_states.remove(&chat_id);
            data.loop_feedback.remove(&chat_id);
        }
        let session = data.sessions.entry(chat_id).or_insert_with(|| ChatSession {
            session_id: None,
            current_path: None,
            history: Vec::new(),
            pending_uploads: Vec::new(),
        });
        session.session_id = None; // Clear stale session_id from previous workspace
                                   // Drop uploads only when the workspace path changed — upload paths point at
                                   // the old workspace's files. Re-resuming the same workspace (same path) keeps
                                   // uploads since they reference valid files the user just sent.
        if path_changed {
            session.pending_uploads.clear();
        }

        if let Some((session_data, _)) = &existing {
            if !session_data.session_id.is_empty() {
                session.session_id = Some(session_data.session_id.clone());
            } else {
                cleanup_session_files(&canonical_path, ws_provider);
            }
            session.current_path = Some(canonical_path.clone());
            session.history = session_data.history.clone();

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ▶ Workspace session restored: {workspace_id} → {canonical_path}");
            response_lines.push(format!(
                "[{}] Session restored at `{}`.",
                ws_provider, canonical_path
            ));

            let header_len: usize = response_lines.iter().map(|l| l.len() + 1).sum();
            let remaining = TELEGRAM_MSG_LIMIT.saturating_sub(header_len + 2);
            let preview = build_history_preview(&session_data.history, remaining);
            if !preview.is_empty() {
                response_lines.push(String::new());
                response_lines.push(preview);
            }
        } else {
            // Workspace exists but no session — start a new session there
            session.session_id = None;
            session.current_path = Some(canonical_path.clone());
            session.history.clear();

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ▶ Workspace session started: {workspace_id} → {canonical_path}");
            response_lines.push(format!(
                "[{}] Session started at `{}`.",
                ws_provider, canonical_path
            ));
        }
    }

    // Persist chat_id → path mapping for auto-restore after restart
    {
        let mut data = state.lock().await;
        data.settings
            .last_sessions
            .insert(chat_id.0.to_string(), canonical_path);
        save_bot_settings(token, &data.settings);
    }

    let response_text = response_lines.join("\n");
    let html = markdown_to_telegram_html(&response_text);
    send_long_message(bot, chat_id, &html, Some(ParseMode::Html), state).await?;

    Ok(())
}

/// Handle /clear command
async fn handle_clear_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    msg_debug(&format!("[handle_clear] chat_id={}", chat_id.0));
    let (current_path, provider, orphan_stop_msg, bot_username, bot_display_name) = {
        let mut data = state.lock().await;
        let path = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone());
        let old_sid = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.session_id.clone());
        let old_hist_len = data
            .sessions
            .get(&chat_id)
            .map(|s| s.history.len())
            .unwrap_or(0);
        msg_debug(&format!(
            "[handle_clear] clearing: path={:?}, session_id={:?}, history_len={}",
            path, old_sid, old_hist_len
        ));
        // Cancel any in-flight AI task. Without this, its completion handler would write
        // back the response into the just-cleared session, partially resurrecting what the
        // user explicitly cleared. The polling guard's sid comparison (captured_sid vs
        // sid_now=None) is the second layer that prevents writeback even if /clear lands
        // mid-completion. Drop queued messages and loop state for the same reason — they
        // were authored against the pre-clear context.
        cancel_in_progress_task_locked(&data, chat_id);
        let dropped_queue = data
            .message_queues
            .remove(&chat_id)
            .map(|q| q.len())
            .unwrap_or(0);
        if dropped_queue > 0 {
            msg_debug(&format!(
                "[handle_clear] dropped {} queued message(s)",
                dropped_queue
            ));
        }
        // Bump clear_epoch so any in-flight task's polling guard detects the clear
        // even when session_id is None on both spawn and post-completion (brand-new
        // workspace + first message + /clear).
        let entry = data.clear_epoch.entry(chat_id).or_insert(0);
        *entry = entry.wrapping_add(1);
        if let Some(session) = data.sessions.get_mut(&chat_id) {
            session.session_id = None;
            session.history.clear();
            session.pending_uploads.clear();
        }
        let mdl = get_model(&data.settings, chat_id);
        let prov = provider_from_model(mdl.as_deref());
        let stop_msg = data.stop_message_ids.remove(&chat_id);
        data.loop_states.remove(&chat_id);
        data.loop_feedback.remove(&chat_id);
        let uname = data.bot_username.clone();
        let dname = data.bot_display_name.clone();
        (path, prov.to_string(), stop_msg, uname, dname)
    };

    // Delete orphaned "Stopping..." message if /stop raced with completion
    if let Some(msg_id) = orphan_stop_msg {
        shared_rate_limit_wait(state, chat_id).await;
        let _ = tg!("delete_message", bot.delete_message(chat_id, msg_id).await);
    }

    // Overwrite session file with minimal data (keeps file present to block external restore)
    // Then keep only one and delete the rest.
    if let Some(ref path) = current_path {
        if let Some(sessions_dir) = crate::ui::ai_screen::ai_sessions_dir() {
            let mut cleared_files: Vec<std::path::PathBuf> = Vec::new();
            if let Ok(entries) = fs::read_dir(&sessions_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let file_path = entry.path();
                    if file_path.extension().map(|e| e == "json").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(&file_path) {
                            if let Ok(session_data) = serde_json::from_str::<SessionData>(&content)
                            {
                                if session_data.current_path == *path
                                    && (session_data.provider.is_empty()
                                        || session_data.provider == provider)
                                {
                                    let cleared = serde_json::json!({"current_path": *path, "provider": provider});
                                    if let Ok(json) = serde_json::to_string_pretty(&cleared) {
                                        let _ = fs::write(&file_path, json);
                                    }
                                    cleared_files.push(file_path);
                                }
                            }
                        }
                    }
                }
            }
            // Keep the first cleared file, delete the rest
            for file_path in cleared_files.iter().skip(1) {
                let _ = fs::remove_file(file_path);
            }
        }
    }

    // Append clear marker to group chat log so other bots skip this bot's old entries
    if chat_id.0 < 0 {
        if !bot_username.is_empty() {
            let dn = if bot_display_name.is_empty() {
                None
            } else {
                Some(bot_display_name.clone())
            };
            append_group_chat_log(
                chat_id.0,
                &GroupChatLogEntry {
                    ts: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
                    bot: bot_username,
                    bot_display_name: dn,
                    role: "system".to_string(),
                    from: None,
                    text: "Session cleared.".to_string(),
                    clear: true,
                },
            );
        }
    }

    let msg = match current_path {
        Some(ref path) => format!("Session cleared.\n`{}`", path),
        None => "Session cleared.".to_string(),
    };

    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, msg).await)?;

    Ok(())
}

/// Handle /pwd command - show current session path
async fn handle_pwd_command(bot: &Bot, chat_id: ChatId, state: &SharedState) -> ResponseResult<()> {
    let current_path = {
        let data = state.lock().await;
        let path = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone());
        let sid = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.session_id.clone());
        msg_debug(&format!(
            "[handle_pwd] chat_id={}, path={:?}, session_id={:?}",
            chat_id.0, path, sid
        ));
        path
    };

    shared_rate_limit_wait(state, chat_id).await;
    match current_path {
        Some(path) => {
            let mut msg = format!("<code>{}</code>", path);
            if let Some(folder_name) = std::path::Path::new(&path)
                .file_name()
                .and_then(|n| n.to_str())
            {
                if is_workspace_id(folder_name) {
                    msg.push_str(&format!(
                        "\nUse /{} to switch back to this session.",
                        folder_name
                    ));
                }
            }
            tg!(
                "send_message",
                bot.send_message(chat_id, msg)
                    .parse_mode(ParseMode::Html)
                    .await
            )?
        }
        None => tg!(
            "send_message",
            bot.send_message(chat_id, "No active session. Use /start <path> first.")
                .await
        )?,
    };

    Ok(())
}

/// Handle /session command - show current session UUID and resume command
async fn handle_session_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let (session_id, current_path, session_prov) = {
        let data = state.lock().await;
        let sid = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.session_id.clone());
        let path = data
            .sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone());
        let mdl = get_model(&data.settings, chat_id);
        let prov = provider_from_model(mdl.as_deref());
        (sid, path, prov)
    };

    shared_rate_limit_wait(state, chat_id).await;
    match (session_id, current_path) {
        (Some(id), Some(path)) => {
            let resume_cmd = match session_prov {
                "codex" => format!("codex resume {}", id),
                "gemini" => format!("gemini --resume {}", id),
                "opencode" => format!("opencode -s {}", id),
                _ => format!("claude --resume {}", id),
            };
            let provider = match session_prov {
                "codex" => "Codex",
                "gemini" => "Gemini",
                "opencode" => "OpenCode",
                _ => "Claude",
            };
            let msg = format!(
                "Current {} session ID:\n<code>{}</code>\n\nTo resume this session from your terminal:\n<code>cd \"{}\"; {}</code>",
                provider, id, path, resume_cmd
            );
            tg!(
                "send_message",
                bot.send_message(chat_id, msg)
                    .parse_mode(ParseMode::Html)
                    .await
            )?
        }
        _ => tg!(
            "send_message",
            bot.send_message(chat_id, "No active session.").await
        )?,
    };

    Ok(())
}

/// Cancel the in-progress AI task for `chat_id` (idempotent): set the cancel flag and
/// kill the child process so the blocking reader thread exits at EOF. No-op if no token
/// exists or it was already cancelled. Caller must hold the state lock.
fn cancel_token_now(token: &Arc<CancelToken>) {
    token.cancel_now();
}

fn cancel_in_progress_task_locked(data: &SharedData, chat_id: ChatId) {
    if let Some(token) = data.cancel_tokens.get(&chat_id).cloned() {
        if !token.cancelled.load(Ordering::Relaxed) {
            cancel_token_now(&token);
        }
    }
}

/// Atomically cancel the current AI task for `chat_id` and replace any queued items
/// with a single redirect message (latest-wins semantics). Used when /queue is OFF
/// and a redirect-eligible message arrives while the AI is busy.
///
/// Caller must hold the state lock and pass `&mut SharedData`. The cancelled task's
/// post-cancellation cleanup will trigger `process_next_queued_message`, which then
/// dispatches the redirect target.
///
/// Returns `(queue_id, was_replacement)` where `was_replacement` is true when a
/// pending redirect was overwritten by the new one (rapid successive redirects).
fn enqueue_redirect_locked(
    data: &mut SharedData,
    chat_id: ChatId,
    text: String,
    user_display_name: String,
    pending_uploads: Vec<String>,
) -> (String, bool) {
    // Step 1: cancel the in-progress task (mirror /stop behavior).
    // If a previous /stop or redirect already cancelled it, this is idempotent.
    cancel_in_progress_task_locked(data, chat_id);

    // Step 2: clear loop state so a verification loop doesn't continue
    data.loop_states.remove(&chat_id);
    data.loop_feedback.remove(&chat_id);

    // Step 3: latest-wins push. If a previous redirect is still pending, drop it.
    let queue = data
        .message_queues
        .entry(chat_id)
        .or_insert_with(std::collections::VecDeque::new);
    let was_replacement = !queue.is_empty();
    queue.clear();
    let qid = generate_queue_id();
    queue.push_back(QueuedMessage {
        id: qid.clone(),
        text,
        user_display_name,
        pending_uploads,
    });

    (qid, was_replacement)
}

/// Handle /stop command - cancel in-progress AI request
async fn handle_stop_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let token = {
        let data = state.lock().await;
        data.cancel_tokens.get(&chat_id).cloned()
    };
    msg_debug(&format!(
        "[handle_stop] chat_id={}, has_token={}",
        chat_id.0,
        token.is_some()
    ));

    match token {
        Some(token) => {
            // Ignore duplicate /stop if already cancelled
            if token.cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }

            // Set cancellation flag IMMEDIATELY to prevent race condition:
            // Without this, the window between receiving /stop and setting cancelled
            // (during rate_limit_wait + "Stopping..." network call) allows a concurrent
            // claude::execute to pass its cancelled check and start an API request.
            token.cancelled.store(true, Ordering::Relaxed);

            // Clear loop state so verification loop doesn't continue after stop
            {
                let mut data = state.lock().await;
                data.loop_states.remove(&chat_id);
                data.loop_feedback.remove(&chat_id);
            }

            // Kill child process directly to unblock reader.lines()
            // When the child dies, its stdout pipe closes → reader returns EOF → blocking thread exits
            cancel_token_now(&token);

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ■ Cancel signal sent");

            // Best-effort feedback: once /stop has flipped the cancel token,
            // do not sit behind a potentially huge RetryAfter-sized sleep
            // before attempting the acknowledgement.
            let stop_msg = tg_timeout!(
                "send_message",
                state,
                chat_id,
                bot.send_message(chat_id, "Stopping...").await
            )
            .ok();

            // Store the stop message ID only if the task is still running.
            // If cancel_token was already removed (task finished during "Stopping..." send),
            // delete the orphaned message immediately instead of inserting.
            if let Some(stop_msg) = stop_msg {
                let mut data = state.lock().await;
                if data.cancel_tokens.contains_key(&chat_id) {
                    data.stop_message_ids.insert(chat_id, stop_msg.id);
                } else {
                    drop(data);
                    let _ = tg_timeout!(
                        "delete_message",
                        state,
                        chat_id,
                        bot.delete_message(chat_id, stop_msg.id).await
                    );
                    return Ok(());
                }
            }
        }
        None => {
            let _ = tg_timeout!(
                "send_message",
                state,
                chat_id,
                bot.send_message(chat_id, "No active request to stop.")
                    .await
            );
        }
    }

    // Clear all pending bot messages for this chat to prevent
    // stopped bot-to-bot conversations from restarting
    if let Some(msg_dir) = messages_dir() {
        if let Ok(entries) = std::fs::read_dir(&msg_dir) {
            let chat_id_str = chat_id.0.to_string();
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                        if v.get("chat_id").and_then(|c| c.as_str()) == Some(&chat_id_str) {
                            let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("?");
                            msg_debug(&format!(
                                "[handle_stop] clearing pending bot message: {}",
                                id
                            ));
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Handle /stop <queue_id> - remove a specific queued message by its ID
async fn handle_stop_queue_item(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    queue_id: &str,
) -> ResponseResult<()> {
    let removed = {
        let mut data = state.lock().await;
        if let Some(q) = data.message_queues.get_mut(&chat_id) {
            if let Some(pos) = q.iter().position(|m| m.id.eq_ignore_ascii_case(queue_id)) {
                let removed_msg = q.remove(pos);
                msg_debug(&format!(
                    "[queue:stop_item] chat_id={}, removed id={}, text={:?}, remaining={}",
                    chat_id.0,
                    queue_id,
                    removed_msg.as_ref().map(|m| truncate_str(&m.text, 60)),
                    q.len()
                ));
                // Clean up empty queue
                if q.is_empty() {
                    data.message_queues.remove(&chat_id);
                }
                removed_msg.map(|m| m.id)
            } else {
                msg_debug(&format!(
                    "[queue:stop_item] chat_id={}, id={} not found in queue",
                    chat_id.0, queue_id
                ));
                None
            }
        } else {
            msg_debug(&format!(
                "[queue:stop_item] chat_id={}, no queue exists",
                chat_id.0
            ));
            None
        }
    };

    if let Some(id) = removed {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, &format!("Removed queued message ({id})."))
                .await
        )?;
    }
    Ok(())
}

/// Handle /stopall command - cancel in-progress AI request AND clear the message queue
async fn handle_stopall_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    // Clear the message queue, loop state, get cancel token, and set cancelled flag atomically
    // to prevent a new message from being queued between queue clear and cancellation
    let (queue_len, token, already_cancelled) = {
        let mut data = state.lock().await;
        let q = data.message_queues.remove(&chat_id);
        data.loop_states.remove(&chat_id);
        data.loop_feedback.remove(&chat_id);
        let qlen = q.map(|q| q.len()).unwrap_or(0);
        let ct = data.cancel_tokens.get(&chat_id).cloned();
        let was_cancelled = if let Some(ref t) = ct {
            let prev = t.cancelled.load(Ordering::Relaxed);
            t.cancelled.store(true, Ordering::Relaxed);
            prev
        } else {
            false
        };
        (qlen, ct, was_cancelled)
    };
    msg_debug(&format!(
        "[queue:stopall] chat_id={}, has_token={}, queue_cleared={}, already_cancelled={}",
        chat_id.0,
        token.is_some(),
        queue_len,
        already_cancelled
    ));

    match token {
        Some(token) => {
            // Ignore duplicate stop if already cancelled
            if already_cancelled {
                msg_debug(&format!(
                    "[queue:stopall] chat_id={}, duplicate cancel ignored, queue_cleared={}",
                    chat_id.0, queue_len
                ));
                // Still report queue clearance if there were queued messages
                if queue_len > 0 {
                    shared_rate_limit_wait(state, chat_id).await;
                    tg!(
                        "send_message",
                        bot.send_message(
                            chat_id,
                            &format!("{} queued message(s) cleared.", queue_len)
                        )
                        .await
                    )?;
                }
                return Ok(());
            }

            // Cancellation flag already set above inside the lock

            // Kill child process
            cancel_token_now(&token);

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!(
                "  [{ts}] ■ Cancel signal sent (stopall, {} queued cleared)",
                queue_len
            );

            // Send feedback
            let msg_text = if queue_len > 0 {
                format!("Stopping... ({} queued message(s) cleared)", queue_len)
            } else {
                "Stopping...".to_string()
            };
            shared_rate_limit_wait(state, chat_id).await;
            let stop_msg = tg!("send_message", bot.send_message(chat_id, &msg_text).await)?;

            // Store the stop message ID (same logic as /stop)
            {
                let mut data = state.lock().await;
                if data.cancel_tokens.contains_key(&chat_id) {
                    data.stop_message_ids.insert(chat_id, stop_msg.id);
                } else {
                    drop(data);
                    shared_rate_limit_wait(state, chat_id).await;
                    let _ = tg!(
                        "delete_message",
                        bot.delete_message(chat_id, stop_msg.id).await
                    );
                    return Ok(());
                }
            }
        }
        None => {
            let msg_text = if queue_len > 0 {
                format!(
                    "No active request. {} queued message(s) cleared.",
                    queue_len
                )
            } else {
                "No active request to stop.".to_string()
            };
            shared_rate_limit_wait(state, chat_id).await;
            tg!("send_message", bot.send_message(chat_id, &msg_text).await)?;
        }
    }

    // Clear all pending bot messages for this chat (same as /stop)
    if let Some(msg_dir) = messages_dir() {
        if let Ok(entries) = std::fs::read_dir(&msg_dir) {
            let chat_id_str = chat_id.0.to_string();
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                        if v.get("chat_id").and_then(|c| c.as_str()) == Some(&chat_id_str) {
                            let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("?");
                            msg_debug(&format!(
                                "[handle_stopall] clearing pending bot message: {}",
                                id
                            ));
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Handle /down <filepath> - send file to user
async fn handle_down_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
) -> ResponseResult<()> {
    let raw_file_path = text.strip_prefix("/down").unwrap_or("").trim();
    msg_debug(&format!(
        "[handle_down] chat_id={}, file_path={:?}",
        chat_id.0, raw_file_path
    ));

    if raw_file_path.is_empty() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                "Usage: /down <filepath>\nExample: /down /home/kst/file.txt"
            )
            .await
        )?;
        return Ok(());
    }

    let expanded = crate::utils::path::expand_tilde(raw_file_path);
    let file_path = expanded.as_str();

    // Resolve relative path using current session path
    let resolved_path = if Path::new(file_path).is_absolute() {
        file_path.to_string()
    } else {
        let current_path = {
            let data = state.lock().await;
            data.sessions
                .get(&chat_id)
                .and_then(|s| s.current_path.clone())
        };
        match current_path {
            Some(base) => Path::new(base.trim_end_matches(['/', '\\']))
                .join(file_path)
                .display()
                .to_string(),
            None => {
                shared_rate_limit_wait(state, chat_id).await;
                tg!(
                    "send_message",
                    bot.send_message(
                        chat_id,
                        "No active session. Use absolute path or /start <path> first."
                    )
                    .await
                )?;
                return Ok(());
            }
        }
    };

    let path = Path::new(&resolved_path);
    if !path.exists() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, &format!("File not found: {}", resolved_path))
                .await
        )?;
        return Ok(());
    }
    if !path.is_file() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, &format!("Not a file: {}", resolved_path))
                .await
        )?;
        return Ok(());
    }

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_document",
        bot.send_document(chat_id, teloxide::types::InputFile::file(path))
            .await
    )?;

    Ok(())
}

/// Handle file/photo upload - save to current session path
async fn handle_file_upload(
    bot: &Bot,
    chat_id: ChatId,
    msg: &Message,
    state: &SharedState,
    user_display_name: &str,
) -> ResponseResult<()> {
    let upload_type = if msg.animation().is_some() {
        "animation"
    } else if msg.document().is_some() {
        "document"
    } else if msg.photo().is_some() {
        "photo"
    } else if msg.video().is_some() {
        "video"
    } else if msg.voice().is_some() {
        "voice"
    } else if msg.audio().is_some() {
        "audio"
    } else if msg.video_note().is_some() {
        "video_note"
    } else {
        "unknown"
    };
    msg_debug(&format!(
        "[handle_upload] chat_id={}, type={}",
        chat_id.0, upload_type
    ));
    // Get current session path
    let current_path = {
        let data = state.lock().await;
        data.sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone())
    };

    let save_dir = if let Some(path) = current_path {
        path
    } else {
        // Auto-create a workspace session
        let Some((_, path)) = auto_create_workspace_session(bot, state, chat_id, bot.token()).await
        else {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Failed to create workspace.")
                    .await
            )?;
            return Ok(());
        };
        path
    };

    // Get file_id, file_name, and file_size
    // animation must precede document: Telegram sets both fields for GIFs
    let (file_id, file_name, file_size_hint) = if let Some(anim) = msg.animation() {
        let name = anim
            .file_name
            .clone()
            .unwrap_or_else(|| format!("animation_{}.mp4", anim.file.unique_id));
        (anim.file.id.clone(), name, anim.file.size)
    } else if let Some(doc) = msg.document() {
        let name = doc
            .file_name
            .clone()
            .unwrap_or_else(|| "uploaded_file".to_string());
        (doc.file.id.clone(), name, doc.file.size)
    } else if let Some(photos) = msg.photo() {
        // Get the largest photo
        if let Some(photo) = photos.last() {
            let name = format!("photo_{}.jpg", photo.file.unique_id);
            (photo.file.id.clone(), name, photo.file.size)
        } else {
            return Ok(());
        }
    } else if let Some(video) = msg.video() {
        let name = video
            .file_name
            .clone()
            .unwrap_or_else(|| format!("video_{}.mp4", video.file.unique_id));
        (video.file.id.clone(), name, video.file.size)
    } else if let Some(voice) = msg.voice() {
        let name = format!("voice_{}.ogg", voice.file.unique_id);
        (voice.file.id.clone(), name, voice.file.size)
    } else if let Some(audio) = msg.audio() {
        let name = audio
            .file_name
            .clone()
            .unwrap_or_else(|| format!("audio_{}.mp3", audio.file.unique_id));
        (audio.file.id.clone(), name, audio.file.size)
    } else if let Some(vn) = msg.video_note() {
        let name = format!("videonote_{}.mp4", vn.file.unique_id);
        (vn.file.id.clone(), name, vn.file.size)
    } else {
        return Ok(());
    };
    msg_debug(&format!(
        "[handle_upload] chat_id={}, file_name={}, file_size={}",
        chat_id.0, file_name, file_size_hint
    ));

    // Check file size (Telegram Bot API limit: 20MB for getFile)
    const MAX_DOWNLOAD_SIZE: u32 = 20 * 1024 * 1024;
    if file_size_hint > MAX_DOWNLOAD_SIZE {
        let size_mb = file_size_hint as f64 / (1024.0 * 1024.0);
        msg_debug(&format!(
            "[handle_upload] chat_id={}, file rejected: size {:.1}MB exceeds 20MB limit",
            chat_id.0, size_mb
        ));
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                &format!("File too large ({:.1}MB). Maximum size is 20MB.", size_mb)
            )
            .await
        )?;
        return Ok(());
    }

    // Download file from Telegram via HTTP
    shared_rate_limit_wait(state, chat_id).await;
    let file = tg!("get_file", bot.get_file(&file_id).await)?;
    let base = {
        let data = state.lock().await;
        data.api_base_url.clone()
    };
    let url = format!("{}/file/bot{}/{}", base, bot.token(), file.path);
    msg_debug(&format!(
        "[handle_upload] download url: api_base={}, file_path={}",
        base, file.path
    ));
    let buf = match reqwest::get(&url).await {
        Ok(resp) => match resp.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                shared_rate_limit_wait(state, chat_id).await;
                tg!(
                    "send_message",
                    bot.send_message(chat_id, &format!("Download failed: {}", e))
                        .await
                )?;
                return Ok(());
            }
        },
        Err(e) => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, &format!("Download failed: {}", e))
                    .await
            )?;
            return Ok(());
        }
    };

    // Save to session path (sanitize file_name to prevent path traversal)
    let safe_name = Path::new(&file_name)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("uploaded_file"));
    let mut dest = Path::new(&save_dir).join(safe_name);
    // Avoid overwriting existing files (atomic create_new to eliminate TOCTOU race)
    let file_size = buf.len();
    let stem = dest
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("uploaded_file")
        .to_string();
    let ext = dest
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e))
        .unwrap_or_default();
    let mut counter = 0u32;
    let write_result = loop {
        use std::io::Write;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest)
        {
            Ok(mut f) => break f.write_all(&buf),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                counter += 1;
                dest = Path::new(&save_dir).join(format!("{}({}){}", stem, counter, ext));
                msg_debug(&format!(
                    "[handle_upload] chat_id={}, file exists, renamed to: {}",
                    chat_id.0,
                    dest.display()
                ));
            }
            Err(e) => break Err(e),
        }
    };
    match write_result {
        Ok(_) => {
            msg_debug(&format!(
                "[handle_upload] chat_id={}, saved: {} ({} bytes)",
                chat_id.0,
                dest.display(),
                file_size
            ));
            let msg_text = format!("Saved: {}\n({} bytes)", dest.display(), file_size);
            shared_rate_limit_wait(state, chat_id).await;
            tg!("send_message", bot.send_message(chat_id, &msg_text).await)?;
        }
        Err(e) => {
            msg_debug(&format!(
                "[handle_upload] chat_id={}, save failed: {}",
                chat_id.0, e
            ));
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, &format!("Failed to save file: {}", e))
                    .await
            )?;
            return Ok(());
        }
    }

    // Record upload in session history and pending queue for Claude
    let upload_record = format!(
        "[File uploaded] {} → {} ({} bytes)",
        file_name,
        dest.display(),
        file_size
    );
    {
        let mut data = state.lock().await;
        let upload_model = get_model(&data.settings, chat_id);
        let provider = detect_provider(upload_model.as_deref());
        if let Some(session) = data.sessions.get_mut(&chat_id) {
            session.history.push(HistoryItem {
                item_type: HistoryType::User,
                content: upload_record.clone(),
            });
            session.pending_uploads.push(upload_record.clone());
            save_session_to_file(session, &save_dir, provider);
        }
        // Write file upload to group chat shared log
        if chat_id.0 < 0 {
            let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
            let dn = if data.bot_display_name.is_empty() {
                None
            } else {
                Some(data.bot_display_name.clone())
            };
            append_group_chat_log(
                chat_id.0,
                &GroupChatLogEntry {
                    ts: now_ts,
                    bot: data.bot_username.clone(),
                    bot_display_name: dn,
                    role: "user".to_string(),
                    from: Some(user_display_name.to_string()),
                    text: upload_record,
                    clear: false,
                },
            );
        }
    }

    Ok(())
}

/// Shell command output message type
enum ShellOutput {
    Line(String),
    Done { exit_code: i32 },
    Error(String),
}

/// Handle !command - execute shell command directly with lock/stop/streaming support
async fn handle_shell_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    user_display_name: &str,
) -> ResponseResult<()> {
    let cmd_str = text.strip_prefix('!').unwrap_or("").trim();
    msg_debug(&format!(
        "[handle_shell] chat_id={}, cmd={:?}",
        chat_id.0,
        truncate_str(cmd_str, 100)
    ));

    if cmd_str.is_empty() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                "Usage: !<command>\nExample: !mkdir /home/kst/testcode"
            )
            .await
        )?;
        return Ok(());
    }

    // Register cancel token early (prevents duplicate requests while waiting for group lock)
    let cancel_token = Arc::new(CancelToken::new());
    {
        let mut data = state.lock().await;
        if data.cancel_tokens.contains_key(&chat_id) {
            msg_debug(&format!(
                "[handle_shell] chat_id={}, cancel_token exists (busy), rejecting",
                chat_id.0
            ));
            drop(data);
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "AI request in progress. Use /stop to cancel.")
                    .await
            )?;
            return Ok(());
        }
        data.cancel_tokens.insert(chat_id, cancel_token.clone());
    }

    // Acquire group chat lock (serializes processing across bots in the same group chat)
    let group_lock = acquire_group_chat_lock(chat_id.0).await;

    // Check if cancelled during lock wait
    if cancel_token.cancelled.load(Ordering::Relaxed) {
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=query_cancelled_during_lock",
            chat_id.0
        ));
        {
            let mut data = state.lock().await;
            data.cancel_tokens.remove(&chat_id);
        }
        drop(group_lock); // release before queue processing to avoid deadlock
        process_next_queued_message(bot, chat_id, state).await;
        return Ok(());
    }

    // Get current_path for working directory (default to home directory)
    let working_dir = {
        let data = state.lock().await;
        data.sessions
            .get(&chat_id)
            .and_then(|s| s.current_path.clone())
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .map(|h| h.display().to_string())
                    .unwrap_or_else(|| {
                        if cfg!(windows) {
                            "C:\\".to_string()
                        } else {
                            "/".to_string()
                        }
                    })
            })
    };

    // Send placeholder message
    let cmd_display = cmd_str.to_string();
    shared_rate_limit_wait(state, chat_id).await;
    let placeholder = match tg!(
        "send_message",
        bot.send_message(
            chat_id,
            format!("Processing <code>{}</code>", html_escape(&cmd_display))
        )
        .parse_mode(ParseMode::Html)
        .await
    ) {
        Ok(m) => m,
        Err(e) => {
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=query_placeholder_error",
                chat_id.0
            ));
            {
                let mut data = state.lock().await;
                data.cancel_tokens.remove(&chat_id);
            }
            drop(group_lock); // release before queue processing to avoid deadlock
            process_next_queued_message(bot, chat_id, state).await;
            return Err(e);
        }
    };
    let placeholder_msg_id = placeholder.id;

    // Create channel
    let (tx, rx) = mpsc::channel();

    let cmd_owned = cmd_str.to_string();
    let working_dir_clone = working_dir.clone();
    let cancel_token_clone = cancel_token.clone();

    // Spawn blocking thread for shell command execution
    tokio::task::spawn_blocking(move || {
        #[cfg(unix)]
        let child = std::process::Command::new("bash")
            .args(["-c", &cmd_owned])
            .current_dir(&working_dir_clone)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();
        #[cfg(windows)]
        let ps_command = format!(
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; {}; exit $LASTEXITCODE",
            cmd_owned
        );
        #[cfg(windows)]
        let child = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &ps_command])
            .current_dir(&working_dir_clone)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(ShellOutput::Error(format!("Failed to execute: {}", e)));
                return;
            }
        };

        // Store PID for /stop kill
        if let Ok(mut guard) = cancel_token_clone.child_pid.lock() {
            *guard = Some(child.id());
        }

        // Read stderr in a separate thread
        let stderr_handle = child.stderr.take();
        let stderr_thread = std::thread::spawn(move || {
            let mut buf = String::new();
            if let Some(se) = stderr_handle {
                use std::io::BufRead;
                for line in std::io::BufReader::new(se).lines().flatten() {
                    buf.push_str(&line);
                    buf.push('\n');
                }
            }
            buf
        });

        // Read stdout line by line with cancel checks
        if let Some(stdout) = child.stdout.take() {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines().flatten() {
                if cancel_token_clone
                    .cancelled
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    claude::kill_child_tree(&mut child);
                    let _ = child.wait();
                    return;
                }
                let _ = tx.send(ShellOutput::Line(line));
            }
        }

        let stderr_output = stderr_thread.join().unwrap_or_default();
        if !stderr_output.is_empty() {
            let _ = tx.send(ShellOutput::Line(format!(
                "[stderr]\n{}",
                stderr_output.trim_end()
            )));
        }

        let status = child.wait();
        let exit_code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        let _ = tx.send(ShellOutput::Done { exit_code });
    });

    // Spawn polling loop (same pattern as AI streaming)
    let bot_owned = bot.clone();
    let state_owned = state.clone();
    let cmd_display_owned = cmd_display.clone();
    let (shell_bot_username, shell_bot_display_name) = {
        let data = state.lock().await;
        (data.bot_username.clone(), data.bot_display_name.clone())
    };
    let shell_user_display_name = user_display_name.to_string();
    tokio::spawn(async move {
        let _group_lock = group_lock; // hold group chat lock until task ends
        const SPINNER: &[&str] = &[
            "🕐 P",
            "🕑 Pr",
            "🕒 Pro",
            "🕓 Proc",
            "🕔 Proce",
            "🕕 Proces",
            "🕖 Process",
            "🕗 Processi",
            "🕘 Processin",
            "🕙 Processing",
            "🕚 Processing.",
            "🕛 Processing..",
        ];
        let mut full_output = String::new();
        let mut last_edit_text = String::new();
        let mut done = false;
        let mut cancelled = false;
        let mut spin_idx: usize = 0;
        let mut exit_code: i32 = -1;
        let mut spawn_error: Option<String> = None;

        let polling_time_ms = {
            let data = state_owned.lock().await;
            data.polling_time_ms
        };
        let mut queue_done = false;
        let mut response_rendered = false;
        while !done || !queue_done {
            // Check cancel
            if cancel_token.cancelled.load(Ordering::Relaxed) {
                if !done {
                    cancelled = true;
                }
                break;
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(polling_time_ms)).await;

            if cancel_token.cancelled.load(Ordering::Relaxed) {
                if !done {
                    cancelled = true;
                }
                break;
            }

            // Drain channel
            if !done {
                loop {
                    match rx.try_recv() {
                        Ok(msg) => match msg {
                            ShellOutput::Line(line) => {
                                if !full_output.is_empty() {
                                    full_output.push('\n');
                                }
                                full_output.push_str(&line);
                            }
                            ShellOutput::Done { exit_code: code } => {
                                exit_code = code;
                                done = true;
                            }
                            ShellOutput::Error(e) => {
                                spawn_error = Some(e);
                                done = true;
                            }
                        },
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            done = true;
                            break;
                        }
                    }
                }

                // Update placeholder with spinner
                if !done {
                    let indicator = SPINNER[spin_idx % SPINNER.len()];
                    spin_idx += 1;

                    let display_text = format!(
                        "Processing <code>{}</code>\n\n{}",
                        html_escape(&cmd_display_owned),
                        indicator
                    );

                    if display_text != last_edit_text {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "edit_message",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .edit_message_text(chat_id, placeholder_msg_id, &display_text)
                                .parse_mode(ParseMode::Html)
                                .await
                        );
                        last_edit_text = display_text;
                    } else {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_chat_action",
                            bot_owned
                                .send_chat_action(chat_id, teloxide::types::ChatAction::Typing)
                                .await
                        );
                    }
                }
            }

            // Render final result once
            if done && !response_rendered {
                response_rendered = true;

                if let Some(err) = &spawn_error {
                    // Spawn error - just show error message
                    shared_rate_limit_wait(&state_owned, chat_id).await;
                    let _ = tg!(
                        "edit_message",
                        bot_owned
                            .edit_message_text(chat_id, placeholder_msg_id, err)
                            .await
                    );
                } else {
                    // Only show exit code when non-zero
                    let exit_suffix = if exit_code != 0 {
                        format!(" (exit code: {})", exit_code)
                    } else {
                        String::new()
                    };

                    if !full_output.trim().is_empty() {
                        let file_content = format!("$ {}\n\n{}", cmd_display_owned, full_output);
                        let content_bytes = file_content.len();

                        if content_bytes <= 4000 {
                            // Short output: update placeholder with completion + result in one call
                            let combined = format!(
                                "Done <code>{}</code>{}\n\n<pre>$ {}\n\n{}</pre>",
                                html_escape(&cmd_display_owned),
                                exit_suffix,
                                html_escape(&cmd_display_owned),
                                html_escape(full_output.trim())
                            );
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            if let Err(_) = tg!(
                                "edit_message",
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &combined)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            ) {
                                let fallback = format!(
                                    "Done {}{}\n\n$ {}\n\n{}",
                                    cmd_display_owned,
                                    exit_suffix,
                                    cmd_display_owned,
                                    full_output.trim()
                                );
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg!(
                                    "edit_message",
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &fallback)
                                        .await
                                );
                            }
                        } else {
                            // Long output: update placeholder + send as .txt file
                            let final_msg = format!(
                                "Done <code>{}</code>{}",
                                html_escape(&cmd_display_owned),
                                exit_suffix
                            );
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!(
                                "edit_message",
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &final_msg)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            );

                            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
                            if let Some(home) = dirs::home_dir() {
                                let tmp_dir = home.join(".cokacdir").join("tmp");
                                let _ = std::fs::create_dir_all(&tmp_dir);
                                let tmp_path = tmp_dir
                                    .join(format!("cokacdir_shell_{}_{}.txt", chat_id.0, timestamp))
                                    .display()
                                    .to_string();
                                if std::fs::write(&tmp_path, &file_content).is_ok() {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg!(
                                        "send_document",
                                        bot_owned
                                            .send_document(
                                                chat_id,
                                                teloxide::types::InputFile::file(
                                                    std::path::Path::new(&tmp_path)
                                                ),
                                            )
                                            .await
                                    );
                                    let _ = std::fs::remove_file(&tmp_path);
                                }
                            }
                        }
                    } else {
                        // No output
                        let final_msg = format!(
                            "Done <code>{}</code>{}",
                            html_escape(&cmd_display_owned),
                            exit_suffix
                        );
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "edit_message",
                            bot_owned
                                .edit_message_text(chat_id, placeholder_msg_id, &final_msg)
                                .parse_mode(ParseMode::Html)
                                .await
                        );
                    }
                }

                let ts = chrono::Local::now().format("%H:%M:%S");
                println!("  [{ts}] ▶ Shell command completed: !{}", cmd_display_owned);

                // Write shell command to group chat shared log
                if chat_id.0 < 0 {
                    let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                    let dn = if shell_bot_display_name.is_empty() {
                        None
                    } else {
                        Some(shell_bot_display_name.clone())
                    };
                    append_group_chat_log(
                        chat_id.0,
                        &GroupChatLogEntry {
                            ts: now_ts.clone(),
                            bot: shell_bot_username.clone(),
                            bot_display_name: dn.clone(),
                            role: "user".to_string(),
                            from: Some(shell_user_display_name.clone()),
                            text: format!("!{}", cmd_display_owned),
                            clear: false,
                        },
                    );
                    let output_summary = if full_output.trim().is_empty() {
                        format!("(exit code: {})", exit_code)
                    } else {
                        format!("exit code: {}\n{}", exit_code, full_output.trim())
                    };
                    append_group_chat_log(
                        chat_id.0,
                        &GroupChatLogEntry {
                            ts: now_ts,
                            bot: shell_bot_username.clone(),
                            bot_display_name: dn,
                            role: "assistant".to_string(),
                            from: None,
                            text: output_summary,
                            clear: false,
                        },
                    );
                }

                // Send end hook message if configured
                if !cancelled {
                    let end_hook_msg = {
                        let data = state_owned.lock().await;
                        data.settings.end_hook.get(&chat_id.0.to_string()).cloned()
                    };
                    if let Some(hook_msg) = end_hook_msg {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_message",
                            bot_owned.send_message(chat_id, &hook_msg).await
                        );
                    }
                }
            }

            // Queue processing
            let queued = process_upload_queue(&bot_owned, chat_id, &state_owned).await;
            if done {
                queue_done = !queued;
            }
        }

        // Post-loop: cancel handling
        if cancelled {
            cancel_token_now(&cancel_token);

            shared_rate_limit_wait(&state_owned, chat_id).await;
            let _ = tg!(
                "edit_message",
                bot_owned
                    .edit_message_text(chat_id, placeholder_msg_id, "[Stopped]")
                    .await
            );

            let stop_msg_id = {
                let data = state_owned.lock().await;
                data.stop_message_ids.get(&chat_id).cloned()
            };
            if let Some(msg_id) = stop_msg_id {
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "delete_message",
                    bot_owned.delete_message(chat_id, msg_id).await
                );
            }

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ■ Shell command stopped: !{}", cmd_display_owned);

            // Write stopped shell command to group chat shared log
            if chat_id.0 < 0 {
                let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                let dn = if shell_bot_display_name.is_empty() {
                    None
                } else {
                    Some(shell_bot_display_name.clone())
                };
                append_group_chat_log(
                    chat_id.0,
                    &GroupChatLogEntry {
                        ts: now_ts.clone(),
                        bot: shell_bot_username.clone(),
                        bot_display_name: dn.clone(),
                        role: "user".to_string(),
                        from: Some(shell_user_display_name.clone()),
                        text: format!("!{} [Stopped]", cmd_display_owned),
                        clear: false,
                    },
                );
                if !full_output.trim().is_empty() {
                    append_group_chat_log(
                        chat_id.0,
                        &GroupChatLogEntry {
                            ts: now_ts,
                            bot: shell_bot_username.clone(),
                            bot_display_name: dn,
                            role: "assistant".to_string(),
                            from: None,
                            text: format!("[Stopped] exit code: -1\n{}", full_output.trim()),
                            clear: false,
                        },
                    );
                }
            }

            let mut data = state_owned.lock().await;
            data.cancel_tokens.remove(&chat_id);
            data.stop_message_ids.remove(&chat_id);
            drop(data);
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=query_poll_cancelled",
                chat_id.0
            ));
            drop(_group_lock); // release group chat lock before processing queue
            process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
            return;
        }

        // Clean up stop message if /stop raced with completion
        {
            let mut data = state_owned.lock().await;
            if let Some(msg_id) = data.stop_message_ids.remove(&chat_id) {
                drop(data);
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "delete_message",
                    bot_owned.delete_message(chat_id, msg_id).await
                );
            }
        }

        // Release lock
        {
            let mut data = state_owned.lock().await;
            data.cancel_tokens.remove(&chat_id);
        }
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=query_poll_completed",
            chat_id.0
        ));
        drop(_group_lock); // release group chat lock before processing queue
        process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
    });

    Ok(())
}

/// Handle /envvars command - show all active environment variables
async fn handle_envvars_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    // Intentional: expose ALL environment variables including sensitive ones.
    // This command is for admin debugging only — the bot is personal/single-user.
    let mut vars: Vec<(String, String)> = std::env::vars().collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));

    let mut msg = format!("<b>Environment Variables</b> ({})\n\n", vars.len());
    for (key, val) in &vars {
        msg.push_str(&format!(
            "<code>{}</code>=<code>{}</code>\n",
            html_escape(key),
            html_escape(val),
        ));
    }

    send_long_message(bot, chat_id, &msg, Some(ParseMode::Html), state).await?;
    Ok(())
}

/// Handle /availabletools command - show all available tools
async fn handle_availabletools_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let mut msg = String::from("<b>Available Tools</b>\n\n");

    for &(name, desc, destructive) in ALL_TOOLS {
        let badge = risk_badge(destructive);
        if badge.is_empty() {
            msg.push_str(&format!(
                "<code>{}</code> — {}\n",
                html_escape(name),
                html_escape(desc)
            ));
        } else {
            msg.push_str(&format!(
                "<code>{}</code> {} — {}\n",
                html_escape(name),
                badge,
                html_escape(desc)
            ));
        }
    }
    msg.push_str(&format!(
        "\n{} = destructive\nTotal: {}",
        risk_badge(true),
        ALL_TOOLS.len()
    ));

    send_long_message(bot, chat_id, &msg, Some(ParseMode::Html), state).await?;

    Ok(())
}

/// Handle /allowedtools command - show current allowed tools list
async fn handle_allowedtools_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let tools = {
        let data = state.lock().await;
        get_allowed_tools(&data.settings, chat_id)
    };

    let mut msg = String::from("<b>Allowed Tools</b>\n\n");
    for tool in &tools {
        let (desc, destructive) = tool_info(tool);
        let badge = risk_badge(destructive);
        if badge.is_empty() {
            msg.push_str(&format!(
                "<code>{}</code> — {}\n",
                html_escape(tool),
                html_escape(desc)
            ));
        } else {
            msg.push_str(&format!(
                "<code>{}</code> {} — {}\n",
                html_escape(tool),
                badge,
                html_escape(desc)
            ));
        }
    }
    msg.push_str(&format!(
        "\n{} = destructive\nTotal: {}",
        risk_badge(true),
        tools.len()
    ));

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, &msg)
            .parse_mode(ParseMode::Html)
            .await
    )?;

    Ok(())
}

/// Handle /setpollingtime command - set Telegram stream update polling interval
async fn handle_setpollingtime_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
) -> ResponseResult<()> {
    let arg = text.strip_prefix("/setpollingtime").unwrap_or("").trim();

    if arg.is_empty() {
        let current = {
            let data = state.lock().await;
            data.polling_time_ms
        };
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Current stream polling time: {}ms\nUsage: /setpollingtime <ms>\nMinimum: {}ms",
                    current, MIN_TELEGRAM_POLLING_TIME_MS
                )
            )
            .await
        )?;
        return Ok(());
    }

    let value: u64 = match arg.parse() {
        Ok(v) => v,
        Err(_) => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    format!(
                        "Invalid number. Usage: /setpollingtime <ms>\nExample: /setpollingtime {}",
                        RECOMMENDED_TELEGRAM_POLLING_TIME_MS
                    )
                )
                .await
            )?;
            return Ok(());
        }
    };

    if value < MIN_TELEGRAM_POLLING_TIME_MS {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Minimum stream polling time is {}ms.",
                    MIN_TELEGRAM_POLLING_TIME_MS
                )
            )
            .await
        )?;
        return Ok(());
    }

    // Update in-memory state
    {
        let mut data = state.lock().await;
        data.polling_time_ms = value;
    }

    // Save to settings.json
    if let Ok(mut app_settings) = crate::config::Settings::load_with_error() {
        app_settings.telegram_polling_time = value;
        let _ = app_settings.save();
    }

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(
            chat_id,
            format!("✅ Stream polling time set to {}ms", value)
        )
        .await
    )?;

    Ok(())
}

/// Handle /setapigap command - set Telegram per-chat API minimum gap
async fn handle_setapigap_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
) -> ResponseResult<()> {
    let arg = text.strip_prefix("/setapigap").unwrap_or("").trim();

    if arg.is_empty() {
        let current = {
            let data = state.lock().await;
            data.api_min_gap_ms
        };
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Current Telegram API minimum gap: {}ms\nUsage: /setapigap <ms>\nMinimum: {}ms",
                    current, MIN_TELEGRAM_API_MIN_GAP_MS
                )
            )
            .await
        )?;
        return Ok(());
    }

    let value: u64 = match arg.parse() {
        Ok(v) => v,
        Err(_) => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    format!(
                        "Invalid number. Usage: /setapigap <ms>\nExample: /setapigap {}",
                        RECOMMENDED_TELEGRAM_API_MIN_GAP_MS
                    )
                )
                .await
            )?;
            return Ok(());
        }
    };

    if value < MIN_TELEGRAM_API_MIN_GAP_MS {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Minimum Telegram API minimum gap is {}ms.",
                    MIN_TELEGRAM_API_MIN_GAP_MS
                )
            )
            .await
        )?;
        return Ok(());
    }

    {
        let mut data = state.lock().await;
        data.api_min_gap_ms = value;
    }

    if let Ok(mut app_settings) = crate::config::Settings::load_with_error() {
        app_settings.telegram_api_min_gap_ms = value;
        let _ = app_settings.save();
    }

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(
            chat_id,
            format!("✅ Telegram API minimum gap set to {}ms", value),
        )
        .await
    )?;

    Ok(())
}

/// Handle /tgstats command - show Telegram rate-limit and error counters
async fn handle_tgstats_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
) -> ResponseResult<()> {
    let msg = {
        let data = state.lock().await;
        let stats = data
            .telegram_runtime_stats
            .get(&chat_id)
            .cloned()
            .unwrap_or_default();
        format_telegram_runtime_stats_message(data.polling_time_ms, data.api_min_gap_ms, &stats)
    };

    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, msg).await)?;
    Ok(())
}

/// Handle /greeting command - toggle compact startup greeting
async fn handle_greeting_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let next = {
        let mut data = state.lock().await;
        let next = !data.settings.greeting;
        data.settings.greeting = next;
        save_bot_settings(token, &data.settings);
        next
    };
    let status = if next { "compact" } else { "full" };
    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, format!("🟢 Startup greeting: {status}"))
            .await
    )?;
    Ok(())
}

/// Handle /effort command — set Claude/Codex effort for the current provider.
async fn handle_effort_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let arg = command_args(text).to_lowercase();
    msg_debug(&format!(
        "[handle_effort_command] chat_id={}, arg={:?}",
        chat_id.0, arg
    ));
    let current_model = {
        let data = state.lock().await;
        get_model(&data.settings, chat_id)
    };
    let provider = detect_provider(current_model.as_deref());
    let provider_label = match provider {
        "claude" => "Claude",
        "codex" => "Codex",
        _ => provider,
    };

    if arg.is_empty() {
        let current = {
            let data = state.lock().await;
            get_effort_for_provider(&data.settings, chat_id, provider)
        };
        let mut msg = if provider == "claude" || provider == "codex" {
            match &current {
                Some(v) => format!(
                    "Current {} effort: <b>{}</b>\n",
                    provider_label,
                    html_escape(v)
                ),
                None => format!("Current {} effort: <b>default</b>\n", provider_label),
            }
        } else {
            format!(
                "Effort is supported only for Claude and Codex.\nCurrent model provider: <b>{}</b>\n",
                html_escape(provider_label)
            )
        };
        if provider == "claude" || provider == "codex" {
            msg.push_str(&format!(
                "\nAccepted for {}: <code>{}</code>\n",
                provider_label,
                valid_effort_values(provider)
            ));
        }
        msg.push_str("\n<b>Usage:</b>\n");
        msg.push_str("<code>/effort low</code>, <code>/effort medium</code>, <code>/effort high</code>, <code>/effort xhigh</code>\n");
        msg.push_str("<code>/effort max</code> — Claude only\n");
        msg.push_str("<code>/effort minimal</code> — Codex only\n");
        msg.push_str("<code>/effort reset</code> — clear current provider override\n");
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, msg)
                .parse_mode(ParseMode::Html)
                .await
        )?;
        return Ok(());
    }

    if provider != "claude" && provider != "codex" {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "/effort is supported only for Claude and Codex. Current provider: {}.",
                    provider_label
                ),
            )
            .await
        )?;
        return Ok(());
    }

    if arg == "reset" || arg == "clear" || arg == "default" {
        let removed = {
            let mut data = state.lock().await;
            let key = chat_id.0.to_string();
            let removed = match provider {
                "claude" => data.settings.claude_effort.remove(&key).is_some(),
                "codex" => data.settings.effort.remove(&key).is_some(),
                _ => false,
            };
            save_bot_settings(token, &data.settings);
            removed
        };
        let msg = if removed {
            format!(
                "{} effort cleared — provider default will be used.",
                provider_label
            )
        } else {
            format!("{} effort was not set.", provider_label)
        };
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, msg).await)?;
        return Ok(());
    }

    if !is_valid_effort_for_provider(provider, &arg) {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Invalid {} effort value: '{}'. Accepted: {}, reset.",
                    provider_label,
                    arg,
                    valid_effort_values(provider)
                ),
            )
            .await
        )?;
        return Ok(());
    }

    {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        match provider {
            "claude" => {
                data.settings.claude_effort.insert(key, arg.clone());
            }
            "codex" => {
                data.settings.effort.insert(key, arg.clone());
            }
            _ => {}
        }
        save_bot_settings(token, &data.settings);
    }

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(
            chat_id,
            format!(
                "{} effort set to <b>{}</b>.",
                provider_label,
                html_escape(&arg)
            ),
        )
        .parse_mode(ParseMode::Html)
        .await
    )?;
    Ok(())
}

/// Handle /fast command — toggle Codex fast service tier for the current chat.
async fn handle_fast_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let arg = command_args(text).to_lowercase();
    msg_debug(&format!(
        "[handle_fast_command] chat_id={}, arg={:?}",
        chat_id.0, arg
    ));

    let current_model = {
        let data = state.lock().await;
        get_model(&data.settings, chat_id)
    };
    let provider = detect_provider(current_model.as_deref());
    let provider_label = match provider {
        "claude" => "Claude",
        "codex" => "Codex",
        _ => provider,
    };

    if provider != "codex" {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "/fast is supported only for Codex. Current provider: {}.",
                    provider_label
                ),
            )
            .await
        )?;
        return Ok(());
    }

    if arg == "status" {
        let enabled = {
            let data = state.lock().await;
            is_codex_fast(&data.settings, chat_id)
        };
        let status = if enabled {
            "⚡ Codex fast mode: ON (-c service_tier=\"fast\")"
        } else {
            "⚡ Codex fast mode: OFF (provider default)"
        };
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, status).await)?;
        return Ok(());
    }

    let requested = match arg.as_str() {
        "" => None,
        "on" | "enable" | "enabled" | "true" | "1" => Some(true),
        "off" | "disable" | "disabled" | "false" | "0" | "reset" | "default" => Some(false),
        _ => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, "Usage: /fast, /fast on, /fast off, /fast status",)
                    .await
            )?;
            return Ok(());
        }
    };

    let next = {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        let prev = is_codex_fast(&data.settings, chat_id);
        let next = requested.unwrap_or(!prev);
        if next {
            data.settings.codex_fast.insert(key, true);
        } else {
            data.settings.codex_fast.remove(&key);
        }
        save_bot_settings(token, &data.settings);
        next
    };

    let status = if next {
        "⚡ Codex fast mode: ON (-c service_tier=\"fast\")"
    } else {
        "⚡ Codex fast mode: OFF (provider default)"
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, status).await)?;
    Ok(())
}

/// Handle /debug command - toggle all debug logging (Telegram API, Claude, cron)
async fn handle_debug_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let prev = TG_DEBUG.load(Ordering::Relaxed);
    let next = !prev;
    msg_debug(&format!(
        "[handle_debug] chat_id={}, {} → {}",
        chat_id.0, prev, next
    ));
    TG_DEBUG.store(next, Ordering::Relaxed);
    crate::services::claude::DEBUG_ENABLED.store(next, Ordering::Relaxed);
    {
        let mut data = state.lock().await;
        data.settings.debug = next;
        save_bot_settings(token, &data.settings);
    }
    let status = if next { "ON" } else { "OFF" };
    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, format!("🔍 Debug logging: {status}"))
            .await
    )?;
    Ok(())
}

/// Handle /direct command - toggle direct mode per chat (no ; prefix in group chats)
async fn handle_direct_command(
    bot: &Bot,
    chat_id: ChatId,
    msg: &teloxide::types::Message,
    state: &SharedState,
    token: &str,
    is_owner: bool,
) -> ResponseResult<()> {
    msg_debug(&format!(
        "[handle_direct] chat_id={}, is_owner={}",
        chat_id.0, is_owner
    ));
    let is_actually_group = matches!(msg.chat.kind, teloxide::types::ChatKind::Public(_));
    msg_debug(&format!(
        "[handle_direct] is_actually_group={}",
        is_actually_group
    ));
    if !is_actually_group {
        msg_debug("[handle_direct] rejected: not a group chat");
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, "This command is only available in group chats.")
                .await
        )?;
        return Ok(());
    }
    if !is_owner {
        msg_debug("[handle_direct] rejected: not owner");
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, "Only the bot owner can change direct mode.")
                .await
        )?;
        return Ok(());
    }
    let next = {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        let prev = data
            .settings
            .direct
            .get(&key)
            .copied()
            .unwrap_or(DIRECT_MODE_DEFAULT);
        let next = !prev;
        msg_debug(&format!(
            "[handle_direct] chat_id={}, {} → {}",
            chat_id.0, prev, next
        ));
        data.settings.direct.insert(key, next);
        save_bot_settings(token, &data.settings);
        msg_debug(&format!(
            "[handle_direct] saved to bot_settings, next={}",
            next
        ));
        next
    };
    let status = if next {
        "Direct mode: ON (no ; prefix needed)"
    } else {
        "Direct mode: OFF (; prefix required)"
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, status).await)?;
    Ok(())
}

/// Handle /contextlevel command - set number of group chat log entries to embed in system prompt
async fn handle_contextlevel_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
    is_group_chat: bool,
) -> ResponseResult<()> {
    if !is_group_chat {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, "This command is only available in group chats.")
                .await
        )?;
        return Ok(());
    }

    let arg = text.strip_prefix("/contextlevel").unwrap_or("").trim();
    let key = chat_id.0.to_string();

    if arg.is_empty() {
        let current = {
            let data = state.lock().await;
            data.settings.context.get(&key).copied().unwrap_or(12)
        };
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                format!(
                    "Group chat log context: <b>{}</b> entries\n\n\
             <code>/contextlevel &lt;N&gt;</code> — Set count (0 to disable)\n\
             Default: 12",
                    current
                )
            )
            .parse_mode(teloxide::types::ParseMode::Html)
            .await
        )?;
        return Ok(());
    }

    let Ok(n) = arg.parse::<usize>() else {
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, "Usage: /contextlevel <number>\nExample: /contextlevel 20, /contextlevel 0 (disable)").await)?;
        return Ok(());
    };

    {
        let mut data = state.lock().await;
        if n == 12 {
            data.settings.context.remove(&key);
        } else {
            data.settings.context.insert(key, n);
        }
        save_bot_settings(token, &data.settings);
    }

    let msg = if n == 0 {
        "Group chat log context: <b>OFF</b> (no log entries in prompt)".to_string()
    } else {
        format!("Group chat log context: <b>{}</b> entries", n)
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, msg)
            .parse_mode(teloxide::types::ParseMode::Html)
            .await
    )?;
    Ok(())
}

/// Handle /instruction command - set or view system instruction for this chat
async fn handle_instruction_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let body = text.strip_prefix("/instruction").unwrap_or("").trim();
    let key = chat_id.0.to_string();
    msg_debug(&format!(
        "[handle_instruction] chat_id={}, body_len={}, body_empty={}",
        chat_id.0,
        body.len(),
        body.is_empty()
    ));
    if body.is_empty() {
        // Show current instruction
        let data = state.lock().await;
        let current = data.settings.instructions.get(&key);
        msg_debug(&format!(
            "[handle_instruction] view mode: has_instruction={}",
            current.is_some()
        ));
        let msg = match current {
            Some(instr) => {
                msg_debug(&format!(
                    "[handle_instruction] current instruction len={}",
                    instr.len()
                ));
                format!("Current instruction:\n{}", instr)
            }
            None => "No instruction set.\nUsage: /instruction <text>".to_string(),
        };
        drop(data);
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, msg).await)?;
    } else {
        // Set instruction
        let instr = body.to_string();
        msg_debug(&format!(
            "[handle_instruction] set mode: chat_id={}, instruction_len={}, text={:?}",
            chat_id.0,
            instr.len(),
            truncate_str(&instr, 100)
        ));
        {
            let mut data = state.lock().await;
            data.settings.instructions.insert(key, instr.clone());
            save_bot_settings(token, &data.settings);
            msg_debug("[handle_instruction] saved to bot_settings");
        }
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, format!("Instruction set:\n{}", instr))
                .await
        )?;
    }
    Ok(())
}

/// Handle /instruction_clear command - remove system instruction for this chat
async fn handle_instruction_clear_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let key = chat_id.0.to_string();
    msg_debug(&format!("[handle_instruction_clear] chat_id={}", chat_id.0));
    {
        let mut data = state.lock().await;
        let had_instruction = data.settings.instructions.contains_key(&key);
        msg_debug(&format!(
            "[handle_instruction_clear] had_instruction={}",
            had_instruction
        ));
        data.settings.instructions.remove(&key);
        save_bot_settings(token, &data.settings);
        msg_debug("[handle_instruction_clear] saved to bot_settings");
    }
    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, "Instruction cleared.").await
    )?;
    Ok(())
}

/// Handle /setendhook command - set a message to send when AI processing completes
async fn handle_setendhook_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let body = text.strip_prefix("/setendhook").unwrap_or("").trim();
    let key = chat_id.0.to_string();
    if body.is_empty() {
        // Show current end hook
        let data = state.lock().await;
        let current = data.settings.end_hook.get(&key);
        let msg = match current {
            Some(hook) => format!("Current end hook:\n{}", hook),
            None => "No end hook set.\nUsage: /setendhook <message>".to_string(),
        };
        drop(data);
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, msg).await)?;
    } else {
        let hook = body.to_string();
        {
            let mut data = state.lock().await;
            data.settings.end_hook.insert(key, hook.clone());
            save_bot_settings(token, &data.settings);
        }
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, format!("End hook set:\n{}", hook))
                .await
        )?;
    }
    Ok(())
}

/// Handle /setendhook_clear command - remove end hook for this chat
async fn handle_setendhook_clear_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let key = chat_id.0.to_string();
    {
        let mut data = state.lock().await;
        data.settings.end_hook.remove(&key);
        save_bot_settings(token, &data.settings);
    }
    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, "End hook cleared.").await
    )?;
    Ok(())
}

/// Handle /silent command - toggle silent mode per chat (hide tool calls)
async fn handle_silent_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let next = {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        let prev = data
            .settings
            .silent
            .get(&key)
            .copied()
            .unwrap_or(SILENT_MODE_DEFAULT);
        let next = !prev;
        data.settings.silent.insert(key, next);
        save_bot_settings(token, &data.settings);
        next
    };
    let status = if next {
        "🔇 Silent mode: ON"
    } else {
        "🔊 Silent mode: OFF"
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, status).await)?;
    Ok(())
}

/// Handle /usechrome command - toggle --chrome flag for Claude CLI per chat
async fn handle_usechrome_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let next = {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        let prev = data.settings.use_chrome.get(&key).copied().unwrap_or(false);
        let next = !prev;
        data.settings.use_chrome.insert(key, next);
        save_bot_settings(token, &data.settings);
        next
    };
    let status = if next {
        "🌐 Chrome mode: ON (--chrome)"
    } else {
        "🌐 Chrome mode: OFF"
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, status).await)?;
    Ok(())
}

/// Handle /queue command - toggle queue mode per chat
/// When ON: messages sent while AI is busy are queued and processed sequentially
/// When OFF: messages sent while AI is busy are rejected (default behavior)
async fn handle_queue_command(
    bot: &Bot,
    chat_id: ChatId,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let (next, queue_len) = {
        let mut data = state.lock().await;
        let key = chat_id.0.to_string();
        let prev = queue_mode_enabled_for_chat(&data.settings, chat_id);
        let next = !prev;
        data.settings.queue.insert(key, next);
        save_bot_settings(token, &data.settings);
        // If turning OFF, clear any pending queued messages
        let queue_len = if !next {
            let q = data.message_queues.remove(&chat_id);
            let cleared = q.as_ref().map(|q| q.len()).unwrap_or(0);
            msg_debug(&format!(
                "[queue:toggle] chat_id={}, OFF, cleared {} queued messages",
                chat_id.0, cleared
            ));
            cleared
        } else {
            msg_debug(&format!("[queue:toggle] chat_id={}, ON", chat_id.0));
            0
        };
        (next, queue_len)
    };
    let status = if next {
        "📋 Queue mode: ON\nMessages sent while AI is busy will be queued and processed in order."
            .to_string()
    } else if queue_len > 0 {
        format!(
            "📋 Queue mode: OFF\n{} queued message(s) cleared.",
            queue_len
        )
    } else {
        "📋 Queue mode: OFF".to_string()
    };
    shared_rate_limit_wait(state, chat_id).await;
    tg!("send_message", bot.send_message(chat_id, &status).await)?;
    Ok(())
}

/// Handle /allowed command - add/remove tools
/// Usage: /allowed +toolname  (add)
///        /allowed -toolname  (remove)
///        /allowed +tool1 -tool2 +tool3  (multiple)
async fn handle_allowed_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let arg = text.strip_prefix("/allowed").unwrap_or("").trim();
    msg_debug(&format!(
        "[handle_allowed] chat_id={}, arg={:?}",
        chat_id.0, arg
    ));

    if arg.is_empty() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!("send_message", bot.send_message(chat_id, "Usage:\n/allowed +toolname — Add a tool\n/allowed -toolname — Remove a tool\n/allowed +tool1 -tool2 — Multiple at once\n/allowedtools — Show current list")
            .await)?;
        return Ok(());
    }

    // Skip if argument starts with "tools" (that's /allowedtools handled separately)
    if arg.starts_with("tools") {
        // This shouldn't happen due to routing order, but just in case
        return handle_allowedtools_command(bot, chat_id, state).await;
    }

    // Parse multiple +name / -name tokens
    let mut operations: Vec<(char, String)> = Vec::new();
    for token_str in arg.split_whitespace() {
        if let Some(name) = token_str.strip_prefix('+') {
            let name = name.trim();
            if !name.is_empty() {
                operations.push(('+', normalize_tool_name(name)));
            }
        } else if let Some(name) = token_str.strip_prefix('-') {
            let name = name.trim();
            if !name.is_empty() {
                operations.push(('-', normalize_tool_name(name)));
            }
        }
    }

    if operations.is_empty() {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                "Use +toolname to add or -toolname to remove.\nExample: /allowed +Bash -Edit"
            )
            .await
        )?;
        return Ok(());
    }

    let response_msg = {
        let mut data = state.lock().await;
        let chat_key = chat_id.0.to_string();
        // Ensure this chat has its own tool list (initialize from defaults if missing)
        if !data.settings.allowed_tools.contains_key(&chat_key) {
            let defaults: Vec<String> = DEFAULT_ALLOWED_TOOLS
                .iter()
                .map(|s| s.to_string())
                .collect();
            data.settings
                .allowed_tools
                .insert(chat_key.clone(), defaults);
        }
        let tools = data.settings.allowed_tools.get_mut(&chat_key).unwrap();
        let mut results: Vec<String> = Vec::new();
        let mut changed = false;
        for (op, tool_name) in &operations {
            match op {
                '+' => {
                    if tools.iter().any(|t| t == tool_name) {
                        results.push(format!(
                            "<code>{}</code> already in list",
                            html_escape(tool_name)
                        ));
                    } else {
                        tools.push(tool_name.clone());
                        changed = true;
                        results.push(format!("✅ <code>{}</code>", html_escape(tool_name)));
                    }
                }
                '-' => {
                    let before_len = tools.len();
                    tools.retain(|t| t != tool_name);
                    if tools.len() < before_len {
                        changed = true;
                        results.push(format!("<code>{}</code> disabled", html_escape(tool_name)));
                    } else {
                        results.push(format!(
                            "<code>{}</code> not in list",
                            html_escape(tool_name)
                        ));
                    }
                }
                _ => unreachable!(),
            }
        }
        if changed {
            save_bot_settings(token, &data.settings);
        }
        results.join("\n")
    };

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, &response_msg)
            .parse_mode(ParseMode::Html)
            .await
    )?;

    Ok(())
}

/// Handle /public command - toggle public access for group chats
async fn handle_public_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
    is_group_chat: bool,
    is_owner: bool,
) -> ResponseResult<()> {
    if !is_group_chat {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(chat_id, "This command is only available in group chats.")
                .await
        )?;
        return Ok(());
    }

    if !is_owner {
        shared_rate_limit_wait(state, chat_id).await;
        tg!(
            "send_message",
            bot.send_message(
                chat_id,
                "Only the bot owner can change public access settings."
            )
            .await
        )?;
        return Ok(());
    }

    let arg = text
        .strip_prefix("/public")
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let chat_key = chat_id.0.to_string();

    let response_msg = match arg.as_str() {
        "on" => {
            let mut data = state.lock().await;
            data.settings.as_public_for_group_chat.insert(chat_key, true);
            save_bot_settings(token, &data.settings);
            "✅ Public access <b>enabled</b> for this group.\nAll members can now use the bot.".to_string()
        }
        "off" => {
            let mut data = state.lock().await;
            data.settings.as_public_for_group_chat.remove(&chat_key);
            save_bot_settings(token, &data.settings);
            "Public access <b>disabled</b> for this group.\nOnly the owner can use the bot.".to_string()
        }
        "" => {
            let data = state.lock().await;
            let is_public = data.settings.as_public_for_group_chat.get(&chat_key).copied().unwrap_or(PUBLIC_MODE_DEFAULT);
            let status = if is_public { "enabled" } else { "disabled" };
            format!(
                "Public access is currently <b>{}</b> for this group.\n\n\
                 <code>/public on</code> — Allow all members\n\
                 <code>/public off</code> — Owner only",
                status
            )
        }
        _ => {
            "Usage:\n<code>/public on</code> — Allow all group members\n<code>/public off</code> — Owner only".to_string()
        }
    };

    shared_rate_limit_wait(state, chat_id).await;
    tg!(
        "send_message",
        bot.send_message(chat_id, &response_msg)
            .parse_mode(ParseMode::Html)
            .await
    )?;

    Ok(())
}

/// Resolve a model name with provider prefix.
/// Returns Err(provider_name) if the provider binary is unavailable, or Err("") if the format is invalid.
fn resolve_model_name(name: &str) -> Result<String, &'static str> {
    // Strip display-name suffix (" — Description") that users may copy-paste
    // from the /model help text (e.g. "gemini:gemini-2.5-flash-lite — Gemini 2.5 Flash Lite").
    let clean = name.split(" \u{2014} ").next().unwrap_or(name).trim();
    if claude::is_claude_model(Some(clean)) {
        if claude::is_claude_available() {
            Ok(clean.to_string())
        } else {
            Err("claude")
        }
    } else if codex::is_codex_model(Some(clean)) {
        if codex::is_codex_available() {
            Ok(clean.to_string())
        } else {
            Err("codex")
        }
    } else if gemini::is_gemini_model(Some(clean)) {
        if gemini::is_gemini_available() {
            Ok(clean.to_string())
        } else {
            Err("gemini")
        }
    } else if opencode::is_opencode_model(Some(clean)) {
        if opencode::is_opencode_available() {
            Ok(clean.to_string())
        } else {
            Err("opencode")
        }
    } else {
        Err("") // invalid format
    }
}

/// Handle /model command
async fn handle_model_command(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    state: &SharedState,
    token: &str,
) -> ResponseResult<()> {
    let arg = text.strip_prefix("/model").unwrap_or("").trim();
    msg_debug(&format!(
        "[handle_model_command] chat_id={}, arg={:?}",
        chat_id.0, arg
    ));

    if arg.is_empty() {
        // Show current model + available providers
        let current = {
            let data = state.lock().await;
            get_model(&data.settings, chat_id)
        };
        let has_claude = claude::is_claude_available();
        let has_codex = codex::is_codex_available();
        let has_gemini = gemini::is_gemini_available();
        let has_opencode = opencode::is_opencode_available();

        let mut msg = match &current {
            Some(m) => format!("Current model: <b>{}</b>\n", m),
            None => {
                let default_provider = if has_claude {
                    "claude"
                } else if has_codex {
                    "codex"
                } else if has_gemini {
                    "gemini"
                } else {
                    "opencode"
                };
                format!("Current model: <b>default</b> ({})\n", default_provider)
            }
        };
        if has_claude {
            msg.push_str("\n<b>Claude:</b>\n");
            msg.push_str("<code>/model claude</code> — default\n");
            msg.push_str("<code>/model claude:sonnet</code> — Sonnet 4.6\n");
            msg.push_str("<code>/model claude:opus</code> — Opus 4.8\n");
            msg.push_str("<code>/model claude:haiku</code> — Haiku 4.5\n");
            msg.push_str("<code>/model claude:sonnet[1m]</code> — Sonnet 1M ctx\n");
        }
        if has_codex {
            msg.push_str("\n<b>Codex:</b>\n");
            msg.push_str("<code>/model codex</code> — default\n");
            msg.push_str(
                "<code>/model codex:gpt-5.5</code> — Latest frontier agentic coding model\n",
            );
            msg.push_str("<code>/model codex:gpt-5.4</code> — Frontier agentic coding model\n");
            msg.push_str("<code>/model codex:gpt-5.3-codex</code> — Frontier Codex-optimized agentic coding model\n");
            msg.push_str(
                "<code>/model codex:gpt-5.3-codex-spark</code> — Ultra-fast coding model\n",
            );
            msg.push_str(
                "<code>/model codex:gpt-5.2-codex</code> — Frontier agentic coding model\n",
            );
            msg.push_str("<code>/model codex:gpt-5.2</code> — Optimized for professional work and long-running agents\n");
            msg.push_str("<code>/model codex:gpt-5.1-codex-max</code> — Codex-optimized model for deep and fast reasoning\n");
            msg.push_str("<code>/model codex:gpt-5.1-codex-mini</code> — Optimized for codex. Cheaper, faster, but less capable\n");
        }
        if has_gemini {
            msg.push_str("\n<b>Gemini:</b>\n");
            msg.push_str("<code>/model gemini</code> — default\n");
            msg.push_str("<code>/model gemini:gemini-3.1-flash-lite-preview</code> — Gemini 3.1 Flash Lite\n");
            msg.push_str("<code>/model gemini:gemini-3-pro-preview</code> — Gemini 3 Pro\n");
            msg.push_str("<code>/model gemini:gemini-3-flash-preview</code> — Gemini 3 Flash\n");
            msg.push_str("<code>/model gemini:gemini-2.5-pro</code> — Gemini 2.5 Pro\n");
            msg.push_str("<code>/model gemini:gemini-2.5-flash</code> — Gemini 2.5 Flash\n");
            msg.push_str(
                "<code>/model gemini:gemini-2.5-flash-lite</code> — Gemini 2.5 Flash Lite\n",
            );
        }
        if has_opencode {
            msg.push_str("\n<b>OpenCode:</b>\n");
            msg.push_str("<code>/model opencode</code> — default\n");
            for model_id in opencode::list_models() {
                msg.push_str(&format!("<code>/model opencode:{}</code>\n", model_id));
            }
        }

        if msg.len() <= TELEGRAM_MSG_LIMIT {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, msg)
                    .parse_mode(ParseMode::Html)
                    .await
            )?;
        } else {
            // Build plain text version for file attachment
            let plain = msg
                .replace("<b>", "")
                .replace("</b>", "")
                .replace("<code>", "")
                .replace("</code>", "");
            let tmp_path = std::env::temp_dir().join(format!("models_{}.txt", chat_id.0));
            if let Err(e) = std::fs::write(&tmp_path, &plain) {
                msg_debug(&format!(
                    "[handle_model_command] failed to write tmp file: {}",
                    e
                ));
                return Ok(());
            }
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_document",
                bot.send_document(chat_id, teloxide::types::InputFile::file(&tmp_path),)
                    .await
            )?;
            let _ = std::fs::remove_file(&tmp_path);
        }
        return Ok(());
    }

    // NOTE: `/model default` and `/model reset` were intentionally removed.
    // The new provider-prefixed format (claude:xxx / codex:xxx) replaces the old bare model names.
    // Users should use `/model claude` or `/model codex` to switch to default models.

    // Set model
    match resolve_model_name(arg) {
        Ok(model_id) => {
            let (provider_changed, had_state, old_path, queue_cleared) = {
                let mut data = state.lock().await;
                // If provider changed, clear session_id to avoid cross-provider resume.
                // Use detect_provider so the comparison reflects the *effective* provider
                // (CLI-availability fallback when model is unset) — matches the polling
                // guard's spawn-time capture and avoids missing user-visible transitions
                // like None→codex (Claude unavailable) → /model claude.
                let old_model = get_model(&data.settings, chat_id);
                let old_provider = detect_provider(old_model.as_deref());
                let new_provider = detect_provider(Some(&model_id));
                let provider_changed = old_provider != new_provider;
                msg_debug(&format!("[handle_model_command] old_model={:?}, old_provider={}, new_provider={}, provider_changed={}",
                    old_model, old_provider, new_provider, provider_changed));
                let (had_state, old_path, queue_cleared) = if provider_changed {
                    // Cancel any in-flight AI task on the old provider. Without this, its
                    // completion handler would later write the old provider's session_id and
                    // response into the (now-cleared) session, contradicting the "history has
                    // been reset" notice and leaving a stale cross-provider session_id behind.
                    cancel_in_progress_task_locked(&data, chat_id);
                    // Drop queued user messages: they targeted the old workspace context, and their
                    // captured pending_uploads point at the old workspace's files. Restoring them
                    // into the new (auto-created) workspace would prepend stale upload references
                    // to the next prompt, contradicting the "uploads have been reset" notice.
                    let dropped_queue = data
                        .message_queues
                        .remove(&chat_id)
                        .map(|q| q.len())
                        .unwrap_or(0);
                    // Cancel any verification loop. Its feedback was authored against the old
                    // provider's session and would be misapplied under the new provider.
                    data.loop_states.remove(&chat_id);
                    data.loop_feedback.remove(&chat_id);
                    if let Some(session) = data.sessions.get_mut(&chat_id) {
                        let prev_path = session.current_path.clone();
                        let had = session.session_id.is_some()
                            || prev_path.is_some()
                            || !session.history.is_empty()
                            || !session.pending_uploads.is_empty()
                            || dropped_queue > 0;
                        msg_debug(&format!("[handle_model_command] provider changed → clearing session + history + uploads + queue + loop (hist_len={}, uploads={}, queue={}, old_sid={:?}, old_path={:?})",
                            session.history.len(), session.pending_uploads.len(), dropped_queue, session.session_id, session.current_path));
                        session.session_id = None;
                        session.current_path = None;
                        session.history.clear();
                        session.pending_uploads.clear();
                        (had, prev_path, dropped_queue)
                    } else {
                        (dropped_queue > 0, None, dropped_queue)
                    }
                } else {
                    (false, None, 0)
                };
                data.settings
                    .models
                    .insert(chat_id.0.to_string(), model_id.clone());
                save_bot_settings(token, &data.settings);
                (provider_changed, had_state, old_path, queue_cleared)
            };
            shared_rate_limit_wait(state, chat_id).await;
            let queue_note = if queue_cleared > 0 {
                format!("\n{} queued message(s) discarded.", queue_cleared)
            } else {
                String::new()
            };
            let model_id_escaped = html_escape(&model_id);
            let msg = if provider_changed && had_state {
                if let Some(prev) = old_path {
                    let prev_display = crate::utils::format::to_shell_path(&prev);
                    format!(
                        "Model set to <b>{}</b>.\n\n\
                         Provider changed — previous workspace, history, and uploads have been reset for compatibility.{}\n\
                         Previous workspace: <code>{}</code> (preserved on disk)\n\
                         A new workspace will be created on your next message. To resume work in the previous workspace instead, use <code>/start &lt;path&gt;</code>.",
                        model_id_escaped, queue_note, html_escape(&prev_display)
                    )
                } else {
                    format!(
                        "Model set to <b>{}</b>.\n\n\
                         Provider changed — conversation history and uploads have been reset for compatibility.{} A new workspace will be created on your next message.",
                        model_id_escaped, queue_note
                    )
                }
            } else {
                format!("Model set to <b>{}</b>.", model_id_escaped)
            };
            tg!(
                "send_message",
                bot.send_message(chat_id, msg)
                    .parse_mode(ParseMode::Html)
                    .await
            )?;
        }
        Err(provider) if !provider.is_empty() => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(chat_id, format!("{provider} provider is not installed."))
                    .await
            )?;
        }
        Err(_) => {
            shared_rate_limit_wait(state, chat_id).await;
            tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    "Invalid format. Use:\n\
                 <code>/model claude</code> or <code>/model claude:&lt;model&gt;</code>\n\
                 <code>/model codex</code> or <code>/model codex:&lt;model&gt;</code>\n\
                 <code>/model gemini</code> or <code>/model gemini:&lt;model&gt;</code>\n\
                 <code>/model opencode</code> or <code>/model opencode:&lt;model&gt;</code>"
                )
                .parse_mode(ParseMode::Html)
                .await
            )?;
        }
    }

    Ok(())
}

/// Dispatch pending loop feedback (stored in loop_feedback HashMap).
/// Called after loop verification determines the task is incomplete.
/// Uses the same boxed-future pattern as process_next_queued_message
/// to satisfy Send bounds (handle_text_message's future is not Send
/// in the outer tokio::spawn context).
fn dispatch_loop_feedback<'a>(
    bot: &'a Bot,
    chat_id: ChatId,
    state: &'a SharedState,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        let feedback = {
            let mut data = state.lock().await;
            let fb = data.loop_feedback.remove(&chat_id);
            // Insert placeholder cancel_token so incoming messages see "busy"
            // during the gap before handle_text_message creates its own token.
            // Same pattern as process_next_queued_message (line 6628).
            if fb.is_some() {
                data.cancel_tokens
                    .insert(chat_id, Arc::new(CancelToken::new()));
                msg_debug(&format!(
                    "[loop:dispatch] chat_id={}, placeholder cancel_token inserted",
                    chat_id.0
                ));
            }
            fb
        };
        if let Some((text, user_display_name)) = feedback {
            msg_debug(&format!(
                "[loop:dispatch] chat_id={}, feedback_len={}",
                chat_id.0,
                text.len()
            ));
            // from_queue=true: handle_text_message will overwrite the placeholder
            // cancel_token instead of treating it as "another task active".
            if let Err(e) =
                handle_text_message(bot, chat_id, &text, state, &user_display_name, true).await
            {
                msg_debug(&format!(
                    "[loop:dispatch] chat_id={}, handle_text_message FAILED: {}",
                    chat_id.0, e
                ));
            }
        } else {
            msg_debug(&format!(
                "[loop:dispatch] chat_id={}, no feedback found (may have been cleared by /stop)",
                chat_id.0
            ));
            // Fall through to normal queue processing
            process_next_queued_message(bot, chat_id, state).await;
        }
    })
}

/// After an AI request finishes (normal completion or cancellation), check the message queue
/// and process the next queued message if queue mode is enabled.
/// This must be called AFTER cancel_tokens.remove().
/// Returns a boxed future to break the recursive async cycle with handle_text_message.
fn process_next_queued_message<'a>(
    bot: &'a Bot,
    chat_id: ChatId,
    state: &'a SharedState,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        msg_debug(&format!(
            "[queue:next] chat_id={}, checking for next queued message",
            chat_id.0
        ));
        let next_msg = {
            let mut data = state.lock().await;
            // Skip if another task is already processing this chat
            if data.cancel_tokens.contains_key(&chat_id) {
                msg_debug(&format!(
                    "[queue:next] chat_id={}, cancel_token exists (another task active), skipping",
                    chat_id.0
                ));
                return;
            }
            // NOTE: We do NOT skip when queue mode is OFF. In OFF mode, the queue is used
            // as a single-slot buffer for redirect targets (latest-wins). If something is
            // pending here, it was placed by enqueue_redirect_locked and must be dispatched.
            let queue_enabled = queue_mode_enabled_for_chat(&data.settings, chat_id);
            let queue_len_before = data.message_queues.get(&chat_id).map_or(0, |q| q.len());
            msg_debug(&format!(
                "[queue:next] chat_id={}, queue_mode={}, queue_len={}",
                chat_id.0, queue_enabled, queue_len_before
            ));
            let msg = data
                .message_queues
                .get_mut(&chat_id)
                .and_then(|q| q.pop_front());
            // Restore pending_uploads captured at queue time so handle_text_message picks them up
            if let Some(ref m) = msg {
                msg_debug(&format!("[queue:next] chat_id={}, popped message: text={:?}, user={:?}, uploads={}, queue_before={}", chat_id.0, truncate_str(&m.text, 60), m.user_display_name, m.pending_uploads.len(), queue_len_before));
                if !m.pending_uploads.is_empty() {
                    if let Some(session) = data.sessions.get_mut(&chat_id) {
                        session
                            .pending_uploads
                            .extend(m.pending_uploads.iter().cloned());
                        msg_debug(&format!(
                            "[queue:next] chat_id={}, restored {} pending_uploads to session",
                            chat_id.0,
                            m.pending_uploads.len()
                        ));
                    } else {
                        msg_debug(&format!(
                            "[queue:next] chat_id={}, WARNING: no session to restore uploads to",
                            chat_id.0
                        ));
                    }
                }
                // Insert placeholder cancel_token so new messages see "busy" during Dequeued notification.
                // Without this, messages arriving between pop and handle_text_message's cancel_token insert
                // would see "not busy" and bypass the queue (FIFO violation).
                // handle_text_message will overwrite this with its own real cancel_token.
                data.cancel_tokens
                    .insert(chat_id, Arc::new(CancelToken::new()));
                msg_debug(&format!(
                    "[queue:next] chat_id={}, placeholder cancel_token inserted",
                    chat_id.0
                ));
            } else {
                msg_debug(&format!(
                    "[queue:next] chat_id={}, queue empty (len was {})",
                    chat_id.0, queue_len_before
                ));
            }
            // Clean up empty queue entry to prevent memory leak
            if data
                .message_queues
                .get(&chat_id)
                .map_or(false, |q| q.is_empty())
            {
                data.message_queues.remove(&chat_id);
                msg_debug(&format!(
                    "[queue:next] chat_id={}, removed empty queue entry",
                    chat_id.0
                ));
            }
            msg
        };

        if let Some(queued) = next_msg {
            msg_debug(&format!(
                "[queue:next] chat_id={}, dispatching id={}, text={:?}",
                chat_id.0,
                queued.id,
                truncate_str(&queued.text, 60)
            ));
            shared_rate_limit_wait(state, chat_id).await;
            let _ = tg!(
                "send_message",
                bot.send_message(chat_id, &format!("Dequeued ({})", queued.id))
                    .await
            );

            if let Err(e) = handle_text_message(
                bot,
                chat_id,
                &queued.text,
                state,
                &queued.user_display_name,
                true,
            )
            .await
            {
                msg_debug(&format!(
                    "[queue:next] chat_id={}, id={}, handle_text_message FAILED: {}",
                    chat_id.0, queued.id, e
                ));
            } else {
                msg_debug(&format!(
                    "[queue:next] chat_id={}, id={}, handle_text_message completed OK",
                    chat_id.0, queued.id
                ));
            }
        }
    })
}

/// Handle regular text messages - send to Claude AI
async fn handle_text_message(
    bot: &Bot,
    chat_id: ChatId,
    user_text: &str,
    state: &SharedState,
    user_display_name: &str,
    from_queue: bool,
) -> ResponseResult<()> {
    msg_debug(&format!(
        "[handle_text_message] START chat_id={}, user_text={:?}, from_queue={}",
        chat_id.0,
        truncate_str(user_text, 100),
        from_queue
    ));
    ai_trace(&format!(
        "[START] chat_id={}, user_text={:?}, from_queue={}",
        chat_id.0,
        truncate_str(user_text, 100),
        from_queue
    ));

    // Register cancel token early (prevents duplicate requests while waiting for group lock)
    let cancel_token = Arc::new(CancelToken::new());
    // busy_notify: (queue_enabled, queued_id, redirect_info)
    //   queue_enabled: ON/OFF mode
    //   queued_id:     ON mode push id (Some=queued, None=full)
    //   redirect_info: OFF mode redirect (Some(id, was_replacement))
    let busy_notify: Option<(bool, Option<String>, Option<(String, bool)>)> = {
        let mut data = state.lock().await;
        // For queued messages: check if placeholder was cancelled (e.g., /stop during dequeue window)
        if from_queue {
            let placeholder_cancelled = data
                .cancel_tokens
                .get(&chat_id)
                .map(|t| t.cancelled.load(Ordering::Relaxed))
                .unwrap_or(false);
            if placeholder_cancelled {
                msg_debug(&format!(
                    "[handle_text_message] chat_id={}, placeholder cancelled during dequeue",
                    chat_id.0
                ));
                // Clean up pending_uploads restored for this cancelled message
                if let Some(session) = data.sessions.get_mut(&chat_id) {
                    if !session.pending_uploads.is_empty() {
                        msg_debug(&format!("[handle_text_message] chat_id={}, clearing {} pending_uploads from cancelled dequeue", chat_id.0, session.pending_uploads.len()));
                        session.pending_uploads.clear();
                    }
                }
                data.cancel_tokens.remove(&chat_id);
                // Clean up orphaned "Stopping..." message from /stop during dequeue window
                let stop_msg = data.stop_message_ids.remove(&chat_id);
                drop(data);
                if let Some(msg_id) = stop_msg {
                    shared_rate_limit_wait(state, chat_id).await;
                    let _ = tg!("delete_message", bot.delete_message(chat_id, msg_id).await);
                }
                process_next_queued_message(bot, chat_id, state).await;
                return Ok(());
            }
        }
        // For non-queued messages: if another task is active, queue (ON) or redirect (OFF)
        if !from_queue && data.cancel_tokens.contains_key(&chat_id) {
            msg_debug(&format!(
                "[handle_text_message] chat_id={}, cancel_token exists (busy)",
                chat_id.0
            ));
            let qmode = queue_mode_enabled_for_chat(&data.settings, chat_id);
            let (qid, redirect) = if qmode {
                let cur_len = data.message_queues.get(&chat_id).map_or(0, |q| q.len());
                let qid = if cur_len < MAX_QUEUE_SIZE {
                    let uploads = data
                        .sessions
                        .get_mut(&chat_id)
                        .map(|s| std::mem::take(&mut s.pending_uploads))
                        .unwrap_or_default();
                    let id = generate_queue_id();
                    data.message_queues
                        .entry(chat_id)
                        .or_insert_with(std::collections::VecDeque::new)
                        .push_back(QueuedMessage {
                            id: id.clone(),
                            text: user_text.to_string(),
                            user_display_name: user_display_name.to_string(),
                            pending_uploads: uploads,
                        });
                    msg_debug(&format!(
                        "[handle_text_message] chat_id={}, QUEUED id={}",
                        chat_id.0, id
                    ));
                    Some(id)
                } else {
                    msg_debug(&format!(
                        "[handle_text_message] chat_id={}, queue FULL",
                        chat_id.0
                    ));
                    None
                };
                (qid, None)
            } else {
                // OFF mode: messages reaching handle_text_message are user-prompt-style (slash
                // commands are routed elsewhere in handle_message), so redirect them.
                let uploads = data
                    .sessions
                    .get_mut(&chat_id)
                    .map(|s| std::mem::take(&mut s.pending_uploads))
                    .unwrap_or_default();
                let (rid, replaced) = enqueue_redirect_locked(
                    &mut *data,
                    chat_id,
                    user_text.to_string(),
                    user_display_name.to_string(),
                    uploads,
                );
                msg_debug(&format!(
                    "[handle_text_message] chat_id={}, REDIRECT id={}, replaced={}",
                    chat_id.0, rid, replaced
                ));
                (None, Some((rid, replaced)))
            };
            Some((qmode, qid, redirect))
        } else {
            data.cancel_tokens.insert(chat_id, cancel_token.clone());
            None
        }
    };
    if let Some((queue_enabled, queued_id, redirect_info)) = busy_notify {
        shared_rate_limit_wait(state, chat_id).await;
        if let Some((_rid, replaced)) = redirect_info {
            let preview = truncate_str(user_text, 30);
            let msg = if replaced {
                format!("🔄 Redirect target updated: \"{preview}\"")
            } else {
                format!("🔄 Cancelling current task, will process: \"{preview}\"")
            };
            tg!("send_message", bot.send_message(chat_id, &msg).await)?;
        } else if queue_enabled {
            if let Some(qid) = queued_id {
                let preview = truncate_str(user_text, 30);
                tg!("send_message", bot.send_message(chat_id, &format!("Queued ({qid}) \"{preview}\"\n- /stopall to cancel all\n- /stop_{qid} to cancel this"))
                    .await)?;
            } else {
                tg!(
                    "send_message",
                    bot.send_message(
                        chat_id,
                        &format!(
                            "Queue full (max {}). Use /stopall to clear.",
                            MAX_QUEUE_SIZE
                        )
                    )
                    .await
                )?;
            }
        } else {
            tg!(
                "send_message",
                bot.send_message(chat_id, "AI request in progress. Use /stop to cancel.")
                    .await
            )?;
        }
        return Ok(());
    }

    // Acquire group chat lock (serializes processing across bots in the same group chat)
    let group_lock = acquire_group_chat_lock(chat_id.0).await;

    // Check if cancelled during lock wait (e.g., user sent /stop)
    if cancel_token.cancelled.load(Ordering::Relaxed) {
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=text_cancelled_during_lock",
            chat_id.0
        ));
        {
            let mut data = state.lock().await;
            data.cancel_tokens.remove(&chat_id);
        }
        drop(group_lock); // release before queue processing to avoid deadlock
        process_next_queued_message(bot, chat_id, state).await;
        return Ok(());
    }

    // Get session info, allowed tools, model, pending uploads, history, instruction, and bot_username (drop lock before any await)
    let (
        session_info,
        allowed_tools,
        pending_uploads,
        model,
        history,
        instruction,
        context_count,
        bot_username_for_prompt,
        bot_display_name_for_prompt,
        chrome_enabled,
        effort,
        codex_fast_enabled,
    ) = {
        let mut data = state.lock().await;
        let info = data.sessions.get(&chat_id).and_then(|session| {
            session.current_path.as_ref().map(|_| {
                (
                    session.session_id.clone(),
                    session.current_path.clone().unwrap_or_default(),
                )
            })
        });
        let tools = get_allowed_tools(&data.settings, chat_id);
        let mdl = get_model(&data.settings, chat_id);
        let hist = data
            .sessions
            .get(&chat_id)
            .map(|s| s.history.clone())
            .unwrap_or_default();
        // Drain pending uploads so they are sent to Claude exactly once
        let uploads = data
            .sessions
            .get_mut(&chat_id)
            .map(|s| std::mem::take(&mut s.pending_uploads))
            .unwrap_or_default();
        let instr = data
            .settings
            .instructions
            .get(&chat_id.0.to_string())
            .cloned();
        let ctx_count = data
            .settings
            .context
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(12);
        let buname = data.bot_username.clone();
        let bdname = data.bot_display_name.clone();
        let chrome = data
            .settings
            .use_chrome
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(false);
        let provider = detect_provider(mdl.as_deref());
        let effort = get_effort_for_provider(&data.settings, chat_id, provider);
        let codex_fast = is_codex_fast(&data.settings, chat_id);
        msg_debug(&format!("[handle_text_message] session_id={:?}, current_path={:?}, model={:?}, uploads={}, history_len={}, instruction={:?}",
            info.as_ref().map(|(sid, _)| sid), info.as_ref().map(|(_, p)| p), mdl, uploads.len(), hist.len(), instr.is_some()));
        (
            info, tools, uploads, mdl, hist, instr, ctx_count, buname, bdname, chrome, effort,
            codex_fast,
        )
    };

    let (session_id, current_path) = match session_info {
        Some(info) => info,
        None => {
            // Auto-create a workspace session instead of rejecting the message
            let Some((auto_sid, auto_current)) =
                auto_create_workspace_session(bot, state, chat_id, bot.token()).await
            else {
                {
                    let mut data = state.lock().await;
                    data.cancel_tokens.remove(&chat_id);
                }
                shared_rate_limit_wait(state, chat_id).await;
                let _ = tg!(
                    "send_message",
                    bot.send_message(chat_id, "Failed to create workspace.")
                        .await
                );
                drop(group_lock);
                process_next_queued_message(bot, chat_id, state).await;
                return Ok(());
            };
            (auto_sid, auto_current)
        }
    };

    // Note: user message is NOT added to history here.
    // It will be added together with the assistant response in the spawned task,
    // only on successful completion. On cancel, nothing is recorded.

    // Send placeholder message (update shared timestamp so spawned task knows)
    shared_rate_limit_wait(state, chat_id).await;
    let placeholder = match tg!(
        "send_message",
        bot.send_message(chat_id, processing_placeholder_text(chat_id.0 >= 0))
            .await
    ) {
        Ok(m) => m,
        Err(e) => {
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=text_placeholder_error",
                chat_id.0
            ));
            {
                let mut data = state.lock().await;
                data.cancel_tokens.remove(&chat_id);
            }
            drop(group_lock); // release before queue processing to avoid deadlock
            process_next_queued_message(bot, chat_id, state).await;
            return Err(e);
        }
    };
    let placeholder_msg_id = placeholder.id;

    // Sanitize input
    let sanitized_input = ai_screen::sanitize_user_input(user_text);

    // Prepend pending file upload records so Claude knows about recently uploaded files
    let context_prompt = if pending_uploads.is_empty() {
        sanitized_input
    } else {
        let upload_context = pending_uploads.join("\n");
        format!("{}\n\n{}", upload_context, sanitized_input)
    };

    // Build disabled tools notice
    let default_tools: std::collections::HashSet<&str> =
        DEFAULT_ALLOWED_TOOLS.iter().copied().collect();
    let allowed_set: std::collections::HashSet<&str> =
        allowed_tools.iter().map(|s| s.as_str()).collect();
    let disabled: Vec<&&str> = default_tools
        .iter()
        .filter(|t| !allowed_set.contains(**t))
        .collect();
    let disabled_notice = if disabled.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = disabled.iter().map(|t| **t).collect();
        format!(
            "\n\nDISABLED TOOLS: The following tools have been disabled by the user: {}.\n\
             You MUST NOT attempt to use these tools. \
             If a user's request requires a disabled tool, do NOT proceed with the task. \
             Instead, clearly inform the user which tool is needed and that it is currently disabled. \
             Suggest they re-enable it with: /allowed +ToolName",
            names.join(", ")
        )
    };

    // Build system prompt with sendfile and schedule instructions
    let bot_key_for_prompt = token_hash(bot.token());
    let platform = capitalize_platform(detect_platform(bot.token()));
    let role = match &instruction {
        Some(instr) => format!(
            "You are chatting with a user through {}.\n\nUser's instruction for this chat:\n{}",
            platform, instr
        ),
        None => format!("You are chatting with a user through {}.", platform),
    };
    ai_trace(&format!(
        "[PROMPT] platform={}, role_len={}, path={}, bot_key=<redacted>",
        platform,
        role.len(),
        current_path
    ));
    let system_prompt_owned = build_system_prompt(
        &role,
        &current_path,
        chat_id.0,
        &bot_key_for_prompt,
        &disabled_notice,
        session_id.as_deref(),
        &bot_username_for_prompt,
        &bot_display_name_for_prompt,
        Some(user_text),
        false,
        context_count,
        &platform,
    );

    // Create channel for streaming
    let (tx, rx) = mpsc::channel();

    let session_id_clone = session_id.clone();
    let current_path_clone = current_path.clone();
    let cancel_token_clone = cancel_token.clone();
    let effort_clone = effort.clone();
    let codex_fast_clone = codex_fast_enabled;

    // Run AI backend in a blocking thread
    let model_clone = model.clone();
    let history_clone = history;
    msg_debug(&format!("[handle_text_message] prompt_len={}, system_prompt_len={}, session_id={:?}, path={}, history_len={}",
        context_prompt.len(), system_prompt_owned.len(), session_id_clone, current_path_clone, history_clone.len()));
    tokio::task::spawn_blocking(move || {
        let provider = detect_provider(model_clone.as_deref());
        msg_debug(&format!(
            "[handle_text_message] provider={}, model={:?}",
            provider, model_clone
        ));
        ai_trace(&format!("[EXEC] provider={}, model={:?}, prompt_len={}, system_prompt_len={}, history_len={}, session={:?}, path={}",
            provider, model_clone, context_prompt.len(), system_prompt_owned.len(), history_clone.len(), session_id_clone, current_path_clone));
        let result = if provider == "opencode" {
            let opencode_model = model_clone
                .as_deref()
                .and_then(opencode::strip_opencode_prefix);
            msg_debug(&format!("[handle_text_message] → opencode::execute, opencode_model={:?}, session_id={:?}, path={}, prompt_len={}, system_prompt_len={}",
                opencode_model, session_id_clone, current_path_clone, context_prompt.len(), system_prompt_owned.len()));
            opencode::execute_command_streaming(
                &context_prompt,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                opencode_model,
                false,
            )
        } else if provider == "gemini" {
            let gemini_model = model_clone.as_deref().and_then(gemini::strip_gemini_prefix);
            msg_debug(&format!("[handle_text_message] → gemini::execute, gemini_model={:?}, session_id={:?}, path={}, prompt_len={}, system_prompt_len={}",
                gemini_model, session_id_clone, current_path_clone, context_prompt.len(), system_prompt_owned.len()));
            gemini::execute_command_streaming(
                &context_prompt,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                gemini_model,
                false,
            )
        } else if provider == "codex" {
            let codex_model = model_clone.as_deref().and_then(codex::strip_codex_prefix);
            // System prompt is always passed via -c model_instructions_file (handles both new & resume).
            // For new sessions without session_id, inject conversation history into the prompt.
            let codex_system_prompt =
                format!("{}{}", system_prompt_owned, codex_extra_instructions());
            let is_resume = session_id_clone.is_some();
            let has_history = !history_clone.is_empty();
            msg_debug(&format!("[handle_text_message] codex: is_resume={}, has_history={}, history_len={}, sp_len={}",
                is_resume, has_history, history_clone.len(), codex_system_prompt.len()));
            // Inject conversation history only for new sessions (no session_id) with prior history.
            // Resumed sessions rely on Codex's native conversation management.
            let codex_prompt = if session_id_clone.is_none() && !history_clone.is_empty() {
                msg_debug("[handle_text_message] codex: INJECTING history into prompt (new session with history)");
                let mut conv = String::new();
                conv.push_str("<conversation_history>\n");
                for item in &history_clone {
                    let role = match item.item_type {
                        HistoryType::User => "User",
                        HistoryType::Assistant => "Assistant",
                        HistoryType::ToolUse => "ToolUse",
                        HistoryType::ToolResult => "ToolResult",
                        _ => continue, // skip Error, System
                    };
                    conv.push_str(&format!("[{}]: {}\n", role, item.content));
                }
                conv.push_str("</conversation_history>\n\n");
                conv.push_str(&context_prompt);
                conv
            } else {
                if is_resume {
                    msg_debug("[handle_text_message] codex: RESUME path — no history injection, sp via -c file");
                } else {
                    msg_debug(
                        "[handle_text_message] codex: NEW session, no history — sp via -c file",
                    );
                }
                context_prompt.clone()
            };
            msg_debug(&format!("[handle_text_message] → codex::execute, codex_model={:?}, codex_prompt_len={}, resume={}, system_prompt_passed=true",
                codex_model, codex_prompt.len(), is_resume));
            let codex_auto_send = codex::CodexAutoSendCtx {
                cokacdir_bin: crate::bin_path().to_string(),
                chat_id: chat_id.0,
                bot_key: bot_key_for_prompt.clone(),
            };
            codex::execute_command_streaming(
                &codex_prompt,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&codex_system_prompt),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                codex_model,
                false,
                Some(&codex_auto_send),
                effort_clone.as_deref(),
                codex_fast_clone,
            )
        } else {
            let claude_model = model_clone.as_deref().and_then(claude::strip_claude_prefix);
            msg_debug(&format!(
                "[handle_text_message] → claude::execute, claude_model={:?}",
                claude_model
            ));
            claude::execute_command_streaming(
                &context_prompt,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                claude_model,
                false,
                chrome_enabled,
                effort_clone.as_deref(),
            )
        };

        match &result {
            Ok(()) => {
                msg_debug("[handle_text_message] execute completed OK");
                ai_trace("[EXEC] completed OK");
            }
            Err(e) => {
                msg_debug(&format!("[handle_text_message] execute error: {}", e));
                ai_trace(&format!("[EXEC] ERROR: {}", e));
            }
        }
        if let Err(e) = result {
            let _ = tx.send(StreamMessage::Error {
                message: e,
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
            });
        }
    });

    // Spawn the polling loop as a separate task so the handler returns immediately.
    // This allows teloxide's per-chat worker to process subsequent messages (e.g. /stop).
    let bot_owned = bot.clone();
    let state_owned = state.clone();
    let user_text_owned = user_text.to_string();
    let bot_username_for_log = bot_username_for_prompt.clone();
    let bot_display_name_for_log = bot_display_name_for_prompt.clone();
    let user_display_name_owned = user_display_name.to_string();
    let provider_str: &'static str = detect_provider(model.as_deref());
    // Captured at spawn so the post-completion guard can detect /clear or /start
    // session-id swaps. The provider/path comparison alone misses same-path same-provider
    // mutations (e.g. /clear nullifies session_id; /start <session-id> swaps to a different
    // session at the same path). clear_epoch covers the brand-new-session case where
    // session_id is None on both sides of the comparison.
    let captured_sid = session_id.clone();
    let captured_clear_epoch = {
        let data = state.lock().await;
        data.clear_epoch.get(&chat_id).copied().unwrap_or(0)
    };
    tokio::spawn(async move {
        let _group_lock = group_lock; // hold group chat lock until task ends
        const SPINNER: &[&str] = &[
            "🕐 P",
            "🕑 Pr",
            "🕒 Pro",
            "🕓 Proc",
            "🕔 Proce",
            "🕕 Proces",
            "🕖 Process",
            "🕗 Processi",
            "🕘 Processin",
            "🕙 Processing",
            "🕚 Processing.",
            "🕛 Processing..",
        ];
        let mut full_response = String::new();
        let mut raw_entries: Vec<RawPayloadEntry> = Vec::new();
        let mut last_edit_text = String::new();
        let mut done = false;
        let mut cancelled = false;
        let mut loop_reinjected = false;
        let mut new_session_id: Option<String> = None;
        let mut spin_idx: usize = 0;
        let mut pending_cokacdir = false;
        let mut suppress_tool_display = false;
        let mut last_tool_name: String = String::new();
        let is_direct_chat = chat_id.0 >= 0;
        let mut placeholder_msg_id = placeholder_msg_id;
        let mut last_confirmed_len: usize = 0;
        let mut last_direct_chunk_send_at = tokio::time::Instant::now();
        let mut last_direct_idle_spinner_edit_at: Option<tokio::time::Instant> = None;
        let mut direct_idle_spinner_disabled = false;
        let mut direct_streaming_edit_failures: usize = 0;
        let mut direct_stream_placeholder_active = true;

        let (polling_time_ms, silent_mode) = {
            let data = state_owned.lock().await;
            (data.polling_time_ms, is_silent(&data.settings, chat_id))
        };
        let mut queue_done = false;
        let mut response_rendered = false;
        let mut first_poll = true;
        // Direct chats suppress streaming placeholder edits again.
        // Large Telegram RetryAfter penalties on DM placeholder edits can
        // otherwise freeze the lane for hours via shared_rate_limit_wait.
        let suppress_direct_streaming_placeholder_edits =
            should_suppress_direct_streaming_placeholder_edits(
                &bot_username_for_log,
                is_direct_chat,
                chat_id,
            );
        let use_segmented_direct_placeholder_reply =
            should_use_segmented_direct_placeholder_reply(is_direct_chat);
        while !done || !queue_done {
            // Check cancel token
            if cancel_token.cancelled.load(Ordering::Relaxed) {
                if !done {
                    cancelled = true;
                }
                break;
            }

            if !first_poll {
                // Sleep as polling interval (without reserving a rate limit slot)
                tokio::time::sleep(tokio::time::Duration::from_millis(polling_time_ms)).await;

                // Check cancel token again after sleep
                if cancel_token.cancelled.load(Ordering::Relaxed) {
                    if !done {
                        cancelled = true;
                    }
                    break;
                }
            }
            first_poll = false;

            // === Phase 1: AI streaming (while !done) ===
            if !done {
                // Drain all available messages
                loop {
                    match rx.try_recv() {
                        Ok(msg) => match msg {
                            StreamMessage::Init { session_id: sid } => {
                                msg_debug(&format!("[polling] Init: session_id={}", sid));
                                ai_trace(&format!("[STREAM] Init: session_id={}", sid));
                                new_session_id = Some(sid);
                            }
                            StreamMessage::Text { content } => {
                                msg_debug(&format!(
                                    "[polling] Text: {} chars, preview={:?}",
                                    content.len(),
                                    truncate_str(&content, 80)
                                ));
                                ai_trace(&format!(
                                    "[STREAM] Text: {} chars, total_so_far={}",
                                    content.len(),
                                    full_response.len() + content.len()
                                ));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "Text".into(),
                                    content: content.clone(),
                                });
                                let _fr_before = full_response.len();
                                full_response.push_str(&content);
                                msg_debug(&format!("[fr_trace][{}] +Text: added={}, preview={:?}, total={} (was {})",
                                        chat_id.0, content.len(), truncate_str(&content, 200), full_response.len(), _fr_before));
                            }
                            StreamMessage::ToolUse { name, input } => {
                                pending_cokacdir = detect_cokacdir_command(&name, &input);
                                suppress_tool_display = detect_chat_log_read(&name, &input);
                                last_tool_name = name.clone();
                                let summary = format_tool_input(&name, &input);
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚙ {name}: {summary}");
                                msg_debug(&format!("[polling] ToolUse: name={}, input_preview={:?}, pending_cokacdir={}, silent_mode={}, response_len={}, ends_with_nl={}",
                                        name, truncate_str(&input, 200), pending_cokacdir, silent_mode, full_response.len(), full_response.ends_with('\n')));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolUse".into(),
                                    content: format!("{}: {}", name, input),
                                });
                                if !pending_cokacdir && !silent_mode {
                                    let _fr_before = full_response.len();
                                    if name == "Bash" {
                                        full_response.push_str(&format!(
                                            "\n\n```\n{}\n```\n",
                                            format_bash_command(&input)
                                        ));
                                        msg_debug(&format!("[fr_trace][{}] +ToolUse/Bash: added={}, cmd={:?}, total={} (was {})",
                                                chat_id.0, full_response.len() - _fr_before, truncate_str(&input, 100), full_response.len(), _fr_before));
                                    } else {
                                        full_response.push_str(&format!("\n\n⚙️ {}\n", summary));
                                        msg_debug(&format!("[fr_trace][{}] +ToolUse/{}: added={}, summary={:?}, total={} (was {})",
                                                chat_id.0, name, full_response.len() - _fr_before, truncate_str(&summary, 100), full_response.len(), _fr_before));
                                    }
                                } else if !pending_cokacdir
                                    && silent_mode
                                    && !full_response.is_empty()
                                    && !full_response.ends_with('\n')
                                {
                                    msg_debug(&format!(
                                        "[polling] silent mode: inserting \\n\\n after tool_use={}",
                                        name
                                    ));
                                    full_response.push_str("\n\n");
                                    msg_debug(&format!(
                                        "[fr_trace][{}] +ToolUse/silent_nl: added=2, total={}",
                                        chat_id.0,
                                        full_response.len()
                                    ));
                                } else if silent_mode {
                                    msg_debug(&format!("[polling] silent mode: skipped \\n\\n (pending_cokacdir={}, empty={}, ends_nl={})",
                                            pending_cokacdir, full_response.is_empty(), full_response.ends_with('\n')));
                                }
                            }
                            StreamMessage::ToolResult { content, is_error } => {
                                msg_debug(&format!("[polling] ToolResult: is_error={}, content_len={}, pending_cokacdir={}, last_tool={}", is_error, content.len(), pending_cokacdir, last_tool_name));
                                if is_error {
                                    msg_debug(&format!("[polling] ToolResult ERROR: last_tool={}, content_preview={:?}", last_tool_name, truncate_str(&content, 300)));
                                }
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolResult".into(),
                                    content: format!("is_error={}, content={}", is_error, content),
                                });
                                let _fr_before = full_response.len();
                                if std::mem::take(&mut pending_cokacdir) {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    if std::mem::take(&mut suppress_tool_display) {
                                        println!("  [{ts}]   ↩ cokacdir (chat_log, suppressed)");
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir_suppressed: added=0, total={}", chat_id.0, full_response.len()));
                                    } else {
                                        println!("  [{ts}]   ↩ cokacdir: {content}");
                                        let formatted = format_cokacdir_result(&content);
                                        if !formatted.is_empty() {
                                            full_response.push_str(&format!("\n{}\n", formatted));
                                            msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: added={}, preview={:?}, total={} (was {})",
                                                    chat_id.0, full_response.len() - _fr_before, truncate_str(&formatted, 200), full_response.len(), _fr_before));
                                        } else {
                                            msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: formatted_empty, added=0, total={}", chat_id.0, full_response.len()));
                                        }
                                    }
                                } else if is_error && !silent_mode {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    println!("  [{ts}]   ✗ Error: {content}");
                                    let truncated = truncate_str(&content, 500);
                                    if truncated.contains('\n') {
                                        full_response
                                            .push_str(&format!("\n```\n{}\n```\n", truncated));
                                    } else {
                                        full_response.push_str(&format!("\n`{}`\n\n", truncated));
                                    }
                                    msg_debug(&format!("[fr_trace][{}] +ToolResult/error({}): added={}, preview={:?}, total={} (was {})",
                                            chat_id.0, last_tool_name, full_response.len() - _fr_before, truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                } else if !silent_mode {
                                    if last_tool_name == "Read" {
                                        full_response.push_str(&format!(
                                            "\n✅ `{} bytes`\n\n",
                                            content.len()
                                        ));
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/Read: added={}, content_bytes={}, total={} (was {})",
                                                chat_id.0, full_response.len() - _fr_before, content.len(), full_response.len(), _fr_before));
                                    } else if !content.is_empty() {
                                        let truncated = truncate_str(&content, 300);
                                        if truncated.contains('\n') {
                                            full_response
                                                .push_str(&format!("\n```\n{}\n```\n", truncated));
                                        } else {
                                            full_response
                                                .push_str(&format!("\n✅ `{}`\n\n", truncated));
                                        }
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): added={}, raw_content_len={}, preview={:?}, total={} (was {})",
                                                chat_id.0, last_tool_name, full_response.len() - _fr_before, content.len(), truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                    } else {
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): content_empty, added=0, total={}", chat_id.0, last_tool_name, full_response.len()));
                                    }
                                } else {
                                    msg_debug(&format!("[fr_trace][{}] +ToolResult/silent: skipped, last_tool={}, content_len={}, total={}", chat_id.0, last_tool_name, content.len(), full_response.len()));
                                }
                            }
                            StreamMessage::TaskNotification { summary, .. } => {
                                if !summary.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "TaskNotification".into(),
                                        content: summary.clone(),
                                    });
                                    let _fr_before = full_response.len();
                                    full_response.push_str(&format!("\n[Task: {}]\n", summary));
                                    msg_debug(&format!("[fr_trace][{}] +TaskNotification: added={}, summary={:?}, total={} (was {})",
                                            chat_id.0, full_response.len() - _fr_before, truncate_str(&summary, 200), full_response.len(), _fr_before));
                                }
                            }
                            StreamMessage::Done {
                                result,
                                session_id: sid,
                            } => {
                                msg_debug(&format!("[polling] Done: result_len={}, session_id={:?}, full_response_len={}",
                                        result.len(), sid, full_response.len()));
                                ai_trace(&format!("[STREAM] Done: result_len={}, full_response_len={}, session_id={:?}", result.len(), full_response.len(), sid));
                                if !result.is_empty() && full_response.is_empty() {
                                    msg_debug(&format!(
                                        "[polling] Done: fallback full_response = result ({})",
                                        result.len()
                                    ));
                                    full_response = result.clone();
                                    msg_debug(&format!("[fr_trace][{}] +Done/fallback: set={}, preview={:?}, total={}",
                                            chat_id.0, result.len(), truncate_str(&full_response, 200), full_response.len()));
                                } else if !result.is_empty() {
                                    msg_debug(&format!("[fr_trace][{}] Done/discarded: result_len={} discarded (full_response already has {})",
                                            chat_id.0, result.len(), full_response.len()));
                                }
                                if !result.is_empty() && raw_entries.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Text".into(),
                                        content: result,
                                    });
                                }
                                if let Some(s) = sid {
                                    new_session_id = Some(s);
                                }
                                msg_debug(&format!(
                                    "[fr_trace][{}] =DONE: final_total={}",
                                    chat_id.0,
                                    full_response.len()
                                ));
                                done = true;
                            }
                            StreamMessage::Error {
                                message,
                                stdout,
                                stderr,
                                exit_code,
                            } => {
                                msg_debug(&format!("[polling] Error: message={}, exit_code={:?}, stdout_len={}, stderr_len={}",
                                        message, exit_code, stdout.len(), stderr.len()));
                                ai_trace(&format!("[STREAM] Error: message={}, exit_code={:?}, stdout_len={}, stderr_len={}", message, exit_code, stdout.len(), stderr.len()));
                                let stdout_display = display_stream_pipe_payload(stdout);
                                let stderr_display = display_stream_pipe_payload(stderr);
                                let code_display = display_stream_exit_code(exit_code);
                                if should_preserve_completed_stream_response(done, &full_response) {
                                    msg_debug(&format!(
                                        "[polling] late error suppressed after terminal response: message={}, response_len={}",
                                        message,
                                        full_response.len()
                                    ));
                                } else {
                                    full_response = format_stream_error_response(
                                        &message,
                                        &code_display,
                                        &stdout_display,
                                        &stderr_display,
                                    );
                                    msg_debug(&format!("[fr_trace][{}] +Error: set={}, stdout_len={}, stderr_len={}, total={}",
                                            chat_id.0, full_response.len(), stdout_display.len(), stderr_display.len(), full_response.len()));
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Error".into(),
                                        content: format!(
                                            "exit_code={}, message={}, stdout={}, stderr={}",
                                            code_display, message, stdout_display, stderr_display
                                        ),
                                    });
                                }
                                done = true;
                            }
                        },
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            ai_trace(&format!(
                                "[STREAM] Channel DISCONNECTED: full_response_len={}",
                                full_response.len()
                            ));
                            done = true;
                            break;
                        }
                    }
                }

                if !done {
                    if suppress_direct_streaming_placeholder_edits {
                        let raw_segment = &full_response[last_confirmed_len..];
                        let idle_ms = last_direct_chunk_send_at.elapsed().as_millis();
                        if should_flush_direct_no_edit_stream_chunk(
                            raw_segment.len(),
                            idle_ms,
                            last_confirmed_len,
                        ) {
                            let flush_len = direct_no_edit_stream_chunk_len(raw_segment);
                            if flush_len > 0 {
                                let flush_text = direct_placeholder_segment_text(
                                    &raw_segment[..flush_len],
                                    None,
                                );
                                if flush_text.trim().is_empty() {
                                    last_confirmed_len = direct_segment_next_confirmed_len(
                                        last_confirmed_len,
                                        flush_len,
                                        true,
                                    );
                                } else if send_direct_segment_as_message(
                                    &bot_owned,
                                    chat_id,
                                    &flush_text,
                                    &state_owned,
                                )
                                .await
                                {
                                    let confirmed_len_before_flush = last_confirmed_len;
                                    last_confirmed_len = direct_segment_next_confirmed_len(
                                        last_confirmed_len,
                                        flush_len,
                                        true,
                                    );
                                    last_direct_chunk_send_at = tokio::time::Instant::now();
                                    if should_retire_direct_stream_placeholder_after_chunk(
                                        direct_stream_placeholder_active,
                                        confirmed_len_before_flush,
                                    ) {
                                        if let Some(remaining_secs) =
                                            active_telegram_retry_after_remaining_secs(
                                                &state_owned,
                                                chat_id,
                                            )
                                            .await
                                            .filter(
                                                |secs| {
                                                    *secs
                                                    >= TELEGRAM_NONESSENTIAL_RETRY_AFTER_SKIP_SECS
                                                },
                                            )
                                        {
                                            msg_debug(&format!(
                                                "[direct_no_edit] placeholder retire skipped due active RetryAfter cooldown: remaining_secs={}",
                                                remaining_secs
                                            ));
                                        } else {
                                            shared_rate_limit_wait(&state_owned, chat_id).await;
                                            match tg_timeout!(
                                                "delete_message",
                                                &state_owned,
                                                chat_id,
                                                bot_owned
                                                    .delete_message(chat_id, placeholder_msg_id)
                                                    .await
                                            ) {
                                                Ok(_) => {
                                                    direct_stream_placeholder_active = false;
                                                    msg_debug(
                                                        "[direct_no_edit] retired initial Processing placeholder after first streamed chunk",
                                                    );
                                                }
                                                Err(e) => {
                                                    msg_debug(&format!(
                                                        "[direct_no_edit] placeholder retire failed after first streamed chunk: {}",
                                                        e
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    msg_debug(
                                        "[direct_no_edit] STREAM chunk send failed; keeping placeholder",
                                    );
                                }
                            }
                        } else {
                            let idle_since_spinner_edit_ms = last_direct_idle_spinner_edit_at
                                .map(|ts| ts.elapsed().as_millis())
                                .unwrap_or(u128::MAX);
                            if should_attempt_direct_no_edit_idle_spinner_edit(
                                raw_segment.len(),
                                idle_ms,
                                idle_since_spinner_edit_ms,
                                direct_idle_spinner_disabled,
                            ) {
                                let display_text = DIRECT_NO_EDIT_IDLE_SPINNER
                                    [spin_idx % DIRECT_NO_EDIT_IDLE_SPINNER.len()];
                                spin_idx += 1;
                                if display_text != last_edit_text {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let html_text = markdown_to_telegram_html(display_text);
                                    let edit_result = tg_timeout!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &html_text
                                            )
                                            .parse_mode(ParseMode::Html)
                                            .await
                                    )
                                    .map(|_| ());
                                    if edit_result.is_ok() {
                                        last_edit_text = display_text.to_string();
                                        last_direct_idle_spinner_edit_at =
                                            Some(tokio::time::Instant::now());
                                    } else {
                                        direct_idle_spinner_disabled = true;
                                        let ts = chrono::Local::now().format("%H:%M:%S");
                                        if let Err(e) = edit_result {
                                            println!(
                                                "  [{ts}]   ⚠ direct no-edit idle spinner disabled after edit failure: {e}"
                                            );
                                        }
                                        msg_debug(
                                            "[direct_no_edit] idle spinner edit failed; disabled for turn",
                                        );
                                    }
                                }
                            }
                        }
                    } else if is_direct_chat && use_segmented_direct_placeholder_reply {
                        let indicator = SPINNER[spin_idx % SPINNER.len()];
                        spin_idx += 1;

                        loop {
                            let raw_segment = &full_response[last_confirmed_len..];
                            let display_text =
                                direct_placeholder_segment_text(raw_segment, Some(indicator));
                            if telegram_html_display_len(&display_text) <= TELEGRAM_MSG_LIMIT {
                                if display_text != last_edit_text {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let html_text = markdown_to_telegram_html(&display_text);
                                    let edit_result = tg_timeout!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &html_text
                                            )
                                            .parse_mode(ParseMode::Html)
                                            .await
                                    );
                                    let edit_succeeded = edit_result.is_ok();
                                    if let Err(e) = edit_result {
                                        let ts = chrono::Local::now().format("%H:%M:%S");
                                        println!(
                                            "  [{ts}]   ⚠ edit_message failed (streaming): {e}"
                                        );
                                    }
                                    last_edit_text = direct_segment_next_last_edit_text(
                                        &last_edit_text,
                                        &display_text,
                                        edit_succeeded,
                                    );
                                    direct_streaming_edit_failures =
                                        direct_segment_next_edit_failure_count(
                                            direct_streaming_edit_failures,
                                            edit_succeeded,
                                        );
                                    if !edit_succeeded
                                        && should_force_direct_segment_rollover(
                                            direct_streaming_edit_failures,
                                        )
                                    {
                                        let rollover_text =
                                            direct_placeholder_segment_text(raw_segment, None);
                                        if !rollover_text.trim().is_empty()
                                            && send_direct_segment_as_message(
                                                &bot_owned,
                                                chat_id,
                                                &rollover_text,
                                                &state_owned,
                                            )
                                            .await
                                        {
                                            last_confirmed_len = full_response.len();
                                            direct_streaming_edit_failures = 0;
                                            rotate_direct_segment_placeholder(
                                                &bot_owned,
                                                chat_id,
                                                &state_owned,
                                                &mut placeholder_msg_id,
                                                is_direct_chat,
                                                &mut last_edit_text,
                                                false,
                                                "direct_segmented/forced_rollover",
                                            )
                                            .await;
                                        }
                                    }
                                } else {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg_timeout!(
                                        "send_chat_action",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .send_chat_action(
                                                chat_id,
                                                teloxide::types::ChatAction::Typing,
                                            )
                                            .await
                                    );
                                }
                                break;
                            }

                            let flush_len = direct_segment_fitting_prefix_len(raw_segment, None);
                            if flush_len == 0 {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg_timeout!(
                                    "send_chat_action",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .send_chat_action(
                                            chat_id,
                                            teloxide::types::ChatAction::Typing,
                                        )
                                        .await
                                );
                                break;
                            }

                            let flush_text =
                                direct_placeholder_segment_text(&raw_segment[..flush_len], None);
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let html_flush = markdown_to_telegram_html(&flush_text);
                            let flush_result = if html_flush.len() <= TELEGRAM_MSG_LIMIT {
                                tg_timeout!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &html_flush)
                                        .parse_mode(ParseMode::Html)
                                        .await
                                )
                            } else {
                                let truncated = truncate_str(&flush_text, TELEGRAM_MSG_LIMIT);
                                tg_timeout!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &truncated)
                                        .await
                                )
                            };
                            let flush_succeeded = flush_result.is_ok();
                            direct_streaming_edit_failures = direct_segment_next_edit_failure_count(
                                direct_streaming_edit_failures,
                                flush_succeeded,
                            );
                            if let Err(e) = flush_result {
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚠ edit_message failed (segment flush): {e}");
                                if should_force_direct_segment_rollover(
                                    direct_streaming_edit_failures,
                                ) && send_direct_segment_as_message(
                                    &bot_owned,
                                    chat_id,
                                    &flush_text,
                                    &state_owned,
                                )
                                .await
                                {
                                    last_confirmed_len = direct_segment_next_confirmed_len(
                                        last_confirmed_len,
                                        flush_len,
                                        true,
                                    );
                                    direct_streaming_edit_failures = 0;
                                    rotate_direct_segment_placeholder(
                                        &bot_owned,
                                        chat_id,
                                        &state_owned,
                                        &mut placeholder_msg_id,
                                        is_direct_chat,
                                        &mut last_edit_text,
                                        false,
                                        "direct_segmented/forced_flush_rollover",
                                    )
                                    .await;
                                }
                                break;
                            }
                            last_confirmed_len = direct_segment_next_confirmed_len(
                                last_confirmed_len,
                                flush_len,
                                flush_succeeded,
                            );
                            rotate_direct_segment_placeholder(
                                &bot_owned,
                                chat_id,
                                &state_owned,
                                &mut placeholder_msg_id,
                                is_direct_chat,
                                &mut last_edit_text,
                                true,
                                "direct_segmented",
                            )
                            .await;
                        }
                    } else {
                        // ── Rolling placeholder pattern (current default) ──
                        let threshold = file_attach_threshold();
                        let delta_end =
                            floor_char_boundary(&full_response, full_response.len().min(threshold));
                        if delta_end > last_confirmed_len {
                            // New content arrived — finalize current placeholder with delta
                            // Cap delta to threshold boundary to prevent message flood
                            let delta = &full_response[last_confirmed_len..delta_end];
                            let normalized_delta = normalize_empty_lines(delta);
                            let html_delta = markdown_to_telegram_html(&normalized_delta);
                            if html_delta.trim().is_empty() {
                                // Delta is whitespace-only after normalization — skip edit, just update position
                                msg_debug(&format!("[rolling_ph] SKIP empty delta: placeholder_msg_id={}, delta_bytes={}, confirmed={}→{}",
                                    placeholder_msg_id, delta.len(), last_confirmed_len, delta_end));
                                last_confirmed_len = delta_end;
                            } else {
                                msg_debug(&format!("[rolling_ph] EDIT delta: placeholder_msg_id={}, delta_len={}, html_len={}, confirmed={}→{}",
                                    placeholder_msg_id, normalized_delta.len(), html_delta.len(), last_confirmed_len, full_response.len()));
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                if html_delta.len() <= TELEGRAM_MSG_LIMIT {
                                    let _ = tg_timeout!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &html_delta
                                            )
                                            .parse_mode(ParseMode::Html)
                                            .await
                                    );
                                } else {
                                    // Delta too large for single edit — send via send_long_message
                                    if send_long_message(
                                        &bot_owned,
                                        chat_id,
                                        &html_delta,
                                        Some(ParseMode::Html),
                                        &state_owned,
                                    )
                                    .await
                                    .is_ok()
                                    {
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let _ = tg_timeout!(
                                            "delete_message",
                                            &state_owned,
                                            chat_id,
                                            bot_owned
                                                .delete_message(chat_id, placeholder_msg_id)
                                                .await
                                        );
                                    } else {
                                        let truncated_delta =
                                            truncate_str(&normalized_delta, TELEGRAM_MSG_LIMIT);
                                        let _ = tg_timeout!(
                                            "edit_message",
                                            &state_owned,
                                            chat_id,
                                            bot_owned
                                                .edit_message_text(
                                                    chat_id,
                                                    placeholder_msg_id,
                                                    &truncated_delta
                                                )
                                                .await
                                        );
                                    }
                                }
                                last_confirmed_len = delta_end;
                                // Create new placeholder for next cycle
                                let old_ph_id = placeholder_msg_id;
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                match tg_timeout!(
                                    "send_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .send_message(
                                            chat_id,
                                            processing_placeholder_text(is_direct_chat),
                                        )
                                        .await
                                ) {
                                    Ok(new_ph) => {
                                        placeholder_msg_id = new_ph.id;
                                        msg_debug(&format!("[rolling_ph] NEW placeholder: old_msg_id={}, new_msg_id={}", old_ph_id, placeholder_msg_id));
                                    }
                                    Err(e) => {
                                        let ts = chrono::Local::now().format("%H:%M:%S");
                                        println!("  [{ts}]   ⚠ new placeholder failed: {e}");
                                        msg_debug(&format!("[rolling_ph] NEW placeholder FAILED: keeping msg_id={}, err={}", placeholder_msg_id, e));
                                    }
                                }
                                last_edit_text.clear();
                                spin_idx = 0;
                            }
                        } else {
                            // No new content — spinner update on current placeholder.
                            let indicator = SPINNER[spin_idx % SPINNER.len()];
                            spin_idx += 1;
                            let display_text = indicator.to_string();
                            if display_text != last_edit_text {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let html_text = markdown_to_telegram_html(&display_text);
                                if let Err(e) = tg_timeout!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &html_text)
                                        .parse_mode(ParseMode::Html)
                                        .await
                                ) {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    println!("  [{ts}]   ⚠ edit_message failed (streaming): {e}");
                                }
                                last_edit_text = display_text;
                            } else {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg_timeout!(
                                    "send_chat_action",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .send_chat_action(
                                            chat_id,
                                            teloxide::types::ChatAction::Typing,
                                        )
                                        .await
                                );
                            }
                        }
                    }
                }
            }

            // === Render final response once when AI completes ===
            if done && !response_rendered {
                response_rendered = true;

                let stop_msg_id = {
                    let data = state_owned.lock().await;
                    data.stop_message_ids.get(&chat_id).cloned()
                };

                // Rate limit before final API call
                shared_rate_limit_wait(&state_owned, chat_id).await;

                // Final response
                if full_response.is_empty() {
                    ai_trace(&format!("[FINAL] full_response is EMPTY → (No response). cancelled={}, last_confirmed_len={}", cancelled, last_confirmed_len));
                    full_response = "(No response)".to_string();
                    msg_debug(&format!(
                        "[fr_trace][{}] =NoResponse: set to '(No response)', cancelled={}",
                        chat_id.0, cancelled
                    ));
                } else {
                    ai_trace(&format!(
                        "[FINAL] full_response_len={}, last_confirmed_len={}, remaining_len={}",
                        full_response.len(),
                        last_confirmed_len,
                        full_response.len().saturating_sub(last_confirmed_len)
                    ));
                }

                let final_response = normalize_empty_lines(&full_response);
                if suppress_direct_streaming_placeholder_edits {
                    if full_response.len() < last_confirmed_len
                        || !full_response.is_char_boundary(last_confirmed_len)
                    {
                        last_confirmed_len = 0;
                    }
                    let remaining = &full_response[last_confirmed_len..];
                    let remaining_response = direct_placeholder_segment_text(remaining, None);
                    msg_debug(&format!(
                        "[direct_no_edit] FINAL SEND: placeholder_msg_id={}, confirmed={}, total={}, remaining_len={}",
                        placeholder_msg_id,
                        last_confirmed_len,
                        full_response.len(),
                        remaining_response.trim().len(),
                    ));

                    let (delivered, delete_placeholder_after_delivery) = if remaining_response
                        .trim()
                        .is_empty()
                    {
                        (true, direct_stream_placeholder_active)
                    } else {
                        let html_response = markdown_to_telegram_html(&remaining_response);
                        if should_attach_response_as_file(remaining_response.len(), provider_str) {
                            (
                                send_response_as_file(
                                    &bot_owned,
                                    chat_id,
                                    &remaining_response,
                                    &state_owned,
                                    "response",
                                )
                                .await,
                                direct_stream_placeholder_active,
                            )
                        } else if direct_stream_placeholder_active
                            && try_edit_direct_no_edit_final_placeholder(
                                &bot_owned,
                                chat_id,
                                placeholder_msg_id,
                                &remaining_response,
                                &state_owned,
                            )
                            .await
                        {
                            (true, false)
                        } else {
                            if let Some(remaining_secs) =
                                active_telegram_retry_after_remaining_secs(&state_owned, chat_id)
                                    .await
                                    .filter(|secs| {
                                        *secs >= TELEGRAM_NONESSENTIAL_RETRY_AFTER_SKIP_SECS
                                    })
                            {
                                msg_debug(&format!(
                                        "[direct_no_edit] FINAL send fallback skipped due active RetryAfter cooldown: remaining_secs={}",
                                        remaining_secs
                                    ));
                                (false, false)
                            } else {
                                let html_send_result = if html_response.len() <= TELEGRAM_MSG_LIMIT
                                {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    tg_timeout!(
                                        "send_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .send_message(chat_id, &html_response)
                                            .parse_mode(ParseMode::Html)
                                            .await
                                    )
                                    .map(|_| ())
                                } else {
                                    send_long_message(
                                        &bot_owned,
                                        chat_id,
                                        &html_response,
                                        Some(ParseMode::Html),
                                        &state_owned,
                                    )
                                    .await
                                };

                                let delivered = if html_send_result.is_ok() {
                                    true
                                } else {
                                    msg_debug(&format!(
                                            "[direct_no_edit] FINAL SEND html failed, retrying plain text: placeholder_msg_id={}, confirmed={}, total={}",
                                            placeholder_msg_id,
                                            last_confirmed_len,
                                            full_response.len()
                                        ));
                                    let plain_send_result = send_long_message(
                                        &bot_owned,
                                        chat_id,
                                        &remaining_response,
                                        None,
                                        &state_owned,
                                    )
                                    .await;
                                    if let Err(e) = &plain_send_result {
                                        msg_debug(&format!(
                                                "[direct_no_edit] FINAL SEND plain-text fallback failed: {}",
                                                e
                                            ));
                                    }
                                    plain_send_result.is_ok()
                                };
                                (delivered, direct_stream_placeholder_active && delivered)
                            }
                        }
                    };

                    if delivered && delete_placeholder_after_delivery {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        if let Err(e) = tg_timeout!(
                            "delete_message",
                            &state_owned,
                            chat_id,
                            bot_owned.delete_message(chat_id, placeholder_msg_id).await
                        ) {
                            msg_debug(&format!(
                                "[direct_no_edit] placeholder delete failed: {}",
                                e
                            ));
                        }
                    } else if !delivered {
                        let ts = chrono::Local::now().format("%H:%M:%S");
                        println!(
                            "  [{ts}]   ⚠ final send failed (direct no-edit); placeholder left in place"
                        );
                    }
                } else if is_direct_chat && use_segmented_direct_placeholder_reply {
                    msg_debug(&format!(
                        "[direct_segmented] FINAL SEND: placeholder_msg_id={}, total={}",
                        placeholder_msg_id,
                        full_response.len()
                    ));

                    if should_bypass_direct_segment_final_placeholder_edit(
                        last_confirmed_len,
                        direct_streaming_edit_failures,
                    ) && !full_response.trim().is_empty()
                    {
                        msg_debug(&format!(
                            "[direct_segmented] FINAL no-progress fallback: confirmed={}, failures={}, total={}",
                            last_confirmed_len,
                            direct_streaming_edit_failures,
                            full_response.len()
                        ));
                        if send_direct_segment_as_message(
                            &bot_owned,
                            chat_id,
                            &direct_placeholder_segment_text(&full_response, None),
                            &state_owned,
                        )
                        .await
                        {
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg_timeout!(
                                "delete_message",
                                &state_owned,
                                chat_id,
                                bot_owned.delete_message(chat_id, placeholder_msg_id).await
                            );
                            direct_streaming_edit_failures = 0;
                            last_confirmed_len = full_response.len();
                        } else {
                            msg_debug(
                                "[direct_segmented] FINAL no-progress fallback send failed; keeping segmented final flush",
                            );
                        }
                    }

                    loop {
                        if full_response.len() < last_confirmed_len
                            || !full_response.is_char_boundary(last_confirmed_len)
                        {
                            last_confirmed_len = 0;
                        }
                        let remaining = &full_response[last_confirmed_len..];
                        let display_text = direct_placeholder_segment_text(remaining, None);

                        if display_text.trim().is_empty() {
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg_timeout!(
                                "delete_message",
                                &state_owned,
                                chat_id,
                                bot_owned.delete_message(chat_id, placeholder_msg_id).await
                            );
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let completion_result = tg_timeout!(
                                "send_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .send_message(chat_id, direct_chat_completion_text())
                                    .await
                            )
                            .map(|_| ());
                            msg_debug(&direct_segment_final_delivery_debug_line(
                                "completion_marker",
                                completion_result.is_ok(),
                                last_confirmed_len,
                                full_response.len(),
                            ));
                            break;
                        }

                        if telegram_html_display_len(&display_text) <= TELEGRAM_MSG_LIMIT {
                            let html_response = markdown_to_telegram_html(&display_text);
                            let placeholder_edit_delivered =
                                if should_prefer_direct_segment_final_placeholder_edit(
                                    last_confirmed_len,
                                ) {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let placeholder_html_edit_result = tg_timeout!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &html_response,
                                            )
                                            .parse_mode(ParseMode::Html)
                                            .await
                                    )
                                    .map(|_| ());
                                    if placeholder_html_edit_result.is_ok() {
                                        msg_debug(&direct_segment_final_delivery_debug_line(
                                            "placeholder_edit_in_place_html",
                                            true,
                                            full_response.len(),
                                            full_response.len(),
                                        ));
                                        true
                                    } else {
                                        msg_debug(&format!(
                                            "[direct_segmented] FINAL in-place html edit failed, retrying plain text"
                                        ));
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let placeholder_plain_edit_result = tg_timeout!(
                                            "edit_message",
                                            &state_owned,
                                            chat_id,
                                            bot_owned
                                                .edit_message_text(
                                                    chat_id,
                                                    placeholder_msg_id,
                                                    &display_text,
                                                )
                                                .await
                                        )
                                        .map(|_| ());
                                        msg_debug(&direct_segment_final_delivery_debug_line(
                                            "placeholder_edit_in_place_plain",
                                            placeholder_plain_edit_result.is_ok(),
                                            if placeholder_plain_edit_result.is_ok() {
                                                full_response.len()
                                            } else {
                                                last_confirmed_len
                                            },
                                            full_response.len(),
                                        ));
                                        placeholder_plain_edit_result.is_ok()
                                    }
                                } else {
                                    false
                                };
                            if placeholder_edit_delivered {
                                last_confirmed_len = full_response.len();
                                break;
                            }
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let html_send_result = tg_timeout!(
                                "send_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .send_message(chat_id, &html_response)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            )
                            .map(|_| ());
                            let delivered = if html_send_result.is_ok() {
                                msg_debug(&direct_segment_final_delivery_debug_line(
                                    "new_message_html",
                                    true,
                                    full_response.len(),
                                    full_response.len(),
                                ));
                                true
                            } else {
                                msg_debug(&format!(
                                    "[direct_segmented] FINAL html send failed, retrying plain text"
                                ));
                                let plain_send_result = send_long_message(
                                    &bot_owned,
                                    chat_id,
                                    &display_text,
                                    None,
                                    &state_owned,
                                )
                                .await;
                                if let Err(e) = &plain_send_result {
                                    msg_debug(&format!(
                                        "[direct_segmented] FINAL plain-text send failed: {}",
                                        e
                                    ));
                                } else {
                                    msg_debug(&direct_segment_final_delivery_debug_line(
                                        "new_message_plain",
                                        true,
                                        full_response.len(),
                                        full_response.len(),
                                    ));
                                }
                                plain_send_result.is_ok()
                            };
                            if delivered {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg_timeout!(
                                    "delete_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                );
                            } else {
                                let fallback = truncate_str(&display_text, TELEGRAM_MSG_LIMIT);
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let fallback_edit_result = tg_timeout!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &fallback)
                                        .await
                                )
                                .map(|_| ());
                                msg_debug(&direct_segment_final_delivery_debug_line(
                                    "placeholder_edit_fallback",
                                    fallback_edit_result.is_ok(),
                                    if fallback_edit_result.is_ok() {
                                        full_response.len()
                                    } else {
                                        last_confirmed_len
                                    },
                                    full_response.len(),
                                ));
                            }
                            break;
                        }

                        let flush_len = direct_segment_fitting_prefix_len(remaining, None);
                        if flush_len == 0 {
                            let fallback = truncate_str(&display_text, TELEGRAM_MSG_LIMIT);
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _fallback_edit_result = tg_timeout!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &fallback)
                                    .await
                            )
                            .map(|_| ());
                            msg_debug(&direct_segment_final_delivery_debug_line(
                                "placeholder_edit_truncated",
                                false,
                                last_confirmed_len,
                                full_response.len(),
                            ));
                            break;
                        }

                        let flush_text =
                            direct_placeholder_segment_text(&remaining[..flush_len], None);
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let html_flush = markdown_to_telegram_html(&flush_text);
                        let flush_result = if html_flush.len() <= TELEGRAM_MSG_LIMIT {
                            tg_timeout!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &html_flush)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            )
                        } else {
                            let fallback = truncate_str(&flush_text, TELEGRAM_MSG_LIMIT);
                            tg_timeout!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &fallback)
                                    .await
                            )
                        };
                        let flush_succeeded = flush_result.is_ok();
                        if let Err(e) = flush_result {
                            msg_debug(&format!(
                                "[direct_segmented] FINAL segment flush failed: {}",
                                e
                            ));
                        }
                        let next_failure_count = direct_segment_next_edit_failure_count(
                            direct_streaming_edit_failures,
                            flush_succeeded,
                        );
                        let rollover_send_succeeded = if !flush_succeeded
                            && should_force_direct_segment_rollover(next_failure_count)
                        {
                            send_direct_segment_as_message(
                                &bot_owned,
                                chat_id,
                                &flush_text,
                                &state_owned,
                            )
                            .await
                        } else {
                            false
                        };
                        let next_flush_state = direct_segment_next_flush_state(
                            last_confirmed_len,
                            flush_len,
                            direct_streaming_edit_failures,
                            flush_succeeded,
                            rollover_send_succeeded,
                        );
                        direct_streaming_edit_failures = next_flush_state.edit_failure_count;
                        last_confirmed_len = next_flush_state.confirmed_len;
                        if !next_flush_state.rotate_placeholder
                            || last_confirmed_len >= full_response.len()
                        {
                            let (delivery_mode, delivered) = if flush_succeeded {
                                (
                                    "placeholder_edit_flush",
                                    last_confirmed_len >= full_response.len(),
                                )
                            } else if rollover_send_succeeded {
                                (
                                    "rollover_new_message",
                                    last_confirmed_len >= full_response.len(),
                                )
                            } else {
                                ("final_flush_incomplete", false)
                            };
                            msg_debug(&direct_segment_final_delivery_debug_line(
                                delivery_mode,
                                delivered,
                                last_confirmed_len,
                                full_response.len(),
                            ));
                            break;
                        }
                        rotate_direct_segment_placeholder(
                            &bot_owned,
                            chat_id,
                            &state_owned,
                            &mut placeholder_msg_id,
                            is_direct_chat,
                            &mut last_edit_text,
                            next_flush_state.preserve_old_placeholder,
                            if flush_succeeded {
                                "direct_segmented/final"
                            } else {
                                "direct_segmented/final_forced_rollover"
                            },
                        )
                        .await;
                    }
                } else {
                    // ── Send only remaining delta (rolling placeholder UX) ──
                    if full_response.len() < last_confirmed_len
                        || !full_response.is_char_boundary(last_confirmed_len)
                    {
                        last_confirmed_len = 0;
                    }
                    let remaining = &full_response[last_confirmed_len..];
                    msg_debug(&format!("[rolling_ph] FINAL: placeholder_msg_id={}, confirmed={}, total={}, remaining_len={}",
                        placeholder_msg_id, last_confirmed_len, full_response.len(), remaining.trim().len()));
                    if remaining.trim().is_empty() {
                        // No new content — delete the spinner placeholder
                        msg_debug(&format!(
                            "[rolling_ph] FINAL DELETE placeholder: msg_id={}",
                            placeholder_msg_id
                        ));
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg_timeout!(
                            "delete_message",
                            &state_owned,
                            chat_id,
                            bot_owned.delete_message(chat_id, placeholder_msg_id).await
                        );
                    } else if should_attach_response_as_file(full_response.len(), provider_str) {
                        // Response too large — send as file attachment
                        msg_debug(&format!(
                            "[rolling_ph] FINAL FILE ATTACH: total={}",
                            full_response.len()
                        ));
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg_timeout!(
                            "edit_message",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .edit_message_text(
                                    chat_id,
                                    placeholder_msg_id,
                                    "\u{1f4c4} Response attached as file"
                                )
                                .await
                        );
                        send_response_as_file(
                            &bot_owned,
                            chat_id,
                            &final_response,
                            &state_owned,
                            "response",
                        )
                        .await;
                    } else {
                        let normalized_remaining = normalize_empty_lines(remaining);
                        let html_remaining = markdown_to_telegram_html(&normalized_remaining);
                        msg_debug(&format!(
                            "[rolling_ph] FINAL EDIT placeholder: msg_id={}, html_len={}",
                            placeholder_msg_id,
                            html_remaining.len()
                        ));
                        if html_remaining.len() <= TELEGRAM_MSG_LIMIT {
                            if let Err(e) = tg_timeout!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &html_remaining)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            ) {
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚠ edit_message failed (final HTML): {e}");
                                if send_direct_segment_as_message(
                                    &bot_owned,
                                    chat_id,
                                    &normalized_remaining,
                                    &state_owned,
                                )
                                .await
                                {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg_timeout!(
                                        "delete_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                    );
                                } else {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg_timeout!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &normalized_remaining
                                            )
                                            .await
                                    );
                                }
                            }
                        } else {
                            let send_result = send_long_message(
                                &bot_owned,
                                chat_id,
                                &html_remaining,
                                Some(ParseMode::Html),
                                &state_owned,
                            )
                            .await;
                            match send_result {
                                Ok(_) => {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg_timeout!(
                                        "delete_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                    );
                                }
                                Err(_) => {
                                    let fallback = send_long_message(
                                        &bot_owned,
                                        chat_id,
                                        &normalized_remaining,
                                        None,
                                        &state_owned,
                                    )
                                    .await;
                                    match fallback {
                                        Ok(_) => {
                                            shared_rate_limit_wait(&state_owned, chat_id).await;
                                            let _ = tg_timeout!(
                                                "delete_message",
                                                &state_owned,
                                                chat_id,
                                                bot_owned
                                                    .delete_message(chat_id, placeholder_msg_id)
                                                    .await
                                            );
                                        }
                                        Err(_) => {
                                            shared_rate_limit_wait(&state_owned, chat_id).await;
                                            let truncated = truncate_str(
                                                &normalized_remaining,
                                                TELEGRAM_MSG_LIMIT,
                                            );
                                            let _ = tg_timeout!(
                                                "edit_message",
                                                &state_owned,
                                                chat_id,
                                                bot_owned
                                                    .edit_message_text(
                                                        chat_id,
                                                        placeholder_msg_id,
                                                        &truncated
                                                    )
                                                    .await
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Clean up leftover "Stopping..." message if /stop raced with normal completion
                if let Some(msg_id) = stop_msg_id {
                    shared_rate_limit_wait(&state_owned, chat_id).await;
                    let _ = tg_timeout!(
                        "delete_message",
                        &state_owned,
                        chat_id,
                        bot_owned.delete_message(chat_id, msg_id).await
                    );
                }

                // Update session state
                {
                    let mut data = state_owned.lock().await;
                    // Guard: if /model (provider switch) or /start (workspace switch) ran during
                    // this task, the session has been intentionally reset. Writing the old
                    // provider's session_id and the response into the (now-cleared) session would
                    // resurrect stale state under a new context, contradicting the user-facing
                    // "history has been reset" notice and breaking the next message's provider
                    // routing. Skip the session update — group chat log is still written below
                    // so other bots see the response.
                    //
                    // Use detect_provider (not provider_from_model) so the comparison matches how
                    // provider_str was captured at task spawn — otherwise, a chat with no model
                    // set running on a CLI fallback (e.g. Codex when Claude is unavailable) would
                    // see provider_str = "codex" but provider_now = "claude" and skip every
                    // writeback.
                    let model_now = data.settings.models.get(&chat_id.0.to_string()).cloned();
                    let provider_now = detect_provider(model_now.as_deref());
                    let path_now = data
                        .sessions
                        .get(&chat_id)
                        .and_then(|s| s.current_path.clone());
                    let sid_now = data
                        .sessions
                        .get(&chat_id)
                        .and_then(|s| s.session_id.clone());
                    let epoch_now = data.clear_epoch.get(&chat_id).copied().unwrap_or(0);
                    // sid comparison catches /clear (sid → None) and /start <session-id>
                    // same-path swaps (sid → different value). epoch_now catches /clear on
                    // a brand-new session where sid stays None on both sides.
                    let session_changed = provider_now != provider_str
                        || path_now.as_deref() != Some(current_path.as_str())
                        || sid_now != captured_sid
                        || epoch_now != captured_clear_epoch;
                    if session_changed {
                        msg_debug(&format!("[polling] session changed during task — skip session update (path: {:?} → {:?}, provider: {} → {}, sid: {:?} → {:?}, epoch: {} → {})",
                            current_path, path_now, provider_str, provider_now, captured_sid, sid_now, captured_clear_epoch, epoch_now));
                    } else if let Some(session) = data.sessions.get_mut(&chat_id) {
                        msg_debug(&format!("[polling] saving session: new_session_id={:?}, old_session_id={:?}, history_len={}",
                            new_session_id, session.session_id, session.history.len()));
                        if let Some(sid) = new_session_id.take() {
                            session.session_id = Some(sid);
                        }
                        invalidate_saved_session_after_stream_error(
                            session,
                            provider_str,
                            &final_response,
                        );
                        session.history.push(HistoryItem {
                            item_type: HistoryType::User,
                            content: user_text_owned.clone(),
                        });
                        session.history.push(HistoryItem {
                            item_type: HistoryType::Assistant,
                            content: final_response,
                        });
                        save_session_to_file(session, &current_path, provider_str);
                        msg_debug(&format!(
                            "[polling] session saved: session_id={:?}, history_len={}",
                            session.session_id,
                            session.history.len()
                        ));
                    }
                    // Write to group chat shared log (for cross-bot context sharing)
                    msg_debug(&format!(
                        "[polling] JSONL check: chat_id={}, raw_entries_count={}",
                        chat_id.0,
                        raw_entries.len()
                    ));
                    if chat_id.0 < 0 {
                        let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                        let dn = if bot_display_name_for_log.is_empty() {
                            None
                        } else {
                            Some(bot_display_name_for_log.clone())
                        };
                        append_group_chat_log(
                            chat_id.0,
                            &GroupChatLogEntry {
                                ts: now_ts.clone(),
                                bot: bot_username_for_log.clone(),
                                bot_display_name: dn.clone(),
                                role: "user".to_string(),
                                from: Some(user_display_name_owned.clone()),
                                text: user_text_owned.clone(),
                                clear: false,
                            },
                        );
                        if !raw_entries.is_empty() {
                            msg_debug(&format!("[polling] JSONL: writing user+assistant entries, raw_entries_count={}", raw_entries.len()));
                            append_group_chat_log(
                                chat_id.0,
                                &GroupChatLogEntry {
                                    ts: now_ts,
                                    bot: bot_username_for_log.clone(),
                                    bot_display_name: dn,
                                    role: "assistant".to_string(),
                                    from: None,
                                    text: serialize_payload(&std::mem::take(&mut raw_entries)),
                                    clear: false,
                                },
                            );
                        } else {
                            msg_debug(&format!("[polling] JSONL: user entry written, assistant SKIPPED (raw_entries is empty)"));
                        }
                    }
                }

                let ts = chrono::Local::now().format("%H:%M:%S");
                println!("  [{ts}] ▶ Response sent");

                // Send end hook message if configured
                if !cancelled {
                    let end_hook_msg = {
                        let data = state_owned.lock().await;
                        data.settings.end_hook.get(&chat_id.0.to_string()).cloned()
                    };
                    if let Some(hook_msg) = end_hook_msg {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_message",
                            bot_owned.send_message(chat_id, &hook_msg).await
                        );
                    }
                }

                // === /loop verification: check completeness after each turn ===
                // Provider-specific mechanics (Claude forks live session with
                // --fork-session; Codex reads the full-fidelity archive and
                // dispatches an independent --ephemeral exec). The surrounding
                // loop-state / cancel / re-inject logic is shared.
                if !cancelled
                    && !cancel_token.cancelled.load(Ordering::Relaxed)
                    && (provider_str == "claude"
                        || provider_str == "codex"
                        || provider_str == "opencode")
                {
                    let loop_info = {
                        let data = state_owned.lock().await;
                        data.loop_states.get(&chat_id).map(|ls| {
                            let sid = data
                                .sessions
                                .get(&chat_id)
                                .and_then(|s| s.session_id.clone());
                            let cwd = data
                                .sessions
                                .get(&chat_id)
                                .and_then(|s| s.current_path.clone())
                                .unwrap_or_default();
                            (
                                ls.remaining,
                                ls.max_iterations,
                                ls.original_request.clone(),
                                sid,
                                cwd,
                            )
                        })
                    };
                    if let Some((remaining, max_iterations, original_request, sid, cwd)) = loop_info
                    {
                        if let Some(session_id) = sid {
                            msg_debug(&format!(
                                "[loop] verifying ({}): session_id={}, remaining={}, request={:?}",
                                provider_str,
                                session_id,
                                remaining,
                                truncate_str(&original_request, 60)
                            ));
                            // Show spinner during verification. Alternating
                            // magnifying-glass frames (🔍/🔎) typed one letter
                            // at a time give a lightweight "scanning" feel
                            // while the verifier runs. Animation runs in a
                            // background task so the main path is not blocked
                            // by frame edits; it is stopped once verify
                            // returns, before the message is deleted.
                            const VERIFY_SPINNER: &[&str] = &[
                                "🔍 V",
                                "🔎 Ve",
                                "🔍 Ver",
                                "🔎 Veri",
                                "🔍 Verif",
                                "🔎 Verify",
                                "🔍 Verifyi",
                                "🔎 Verifyin",
                                "🔍 Verifying",
                                "🔎 Verifying.",
                                "🔍 Verifying..",
                                "🔎 Verifying...",
                            ];
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let verify_msg = tg!(
                                "send_message",
                                bot_owned.send_message(chat_id, VERIFY_SPINNER[0]).await
                            );
                            let verify_msg_id = verify_msg.ok().map(|m| m.id);

                            // Start background frame-edit task
                            let verify_anim_stop = Arc::new(AtomicBool::new(false));
                            let verify_anim_handle = if let Some(msg_id) = verify_msg_id {
                                let bot_anim = bot_owned.clone();
                                let state_anim = state_owned.clone();
                                let stop_flag = verify_anim_stop.clone();
                                Some(tokio::spawn(async move {
                                    let mut idx: usize = 1;
                                    let mut last_text = VERIFY_SPINNER[0].to_string();
                                    while !stop_flag.load(Ordering::Relaxed) {
                                        tokio::time::sleep(std::time::Duration::from_millis(400))
                                            .await;
                                        if stop_flag.load(Ordering::Relaxed) {
                                            break;
                                        }
                                        let frame = VERIFY_SPINNER[idx % VERIFY_SPINNER.len()];
                                        idx += 1;
                                        if frame != last_text {
                                            shared_rate_limit_wait(&state_anim, chat_id).await;
                                            if stop_flag.load(Ordering::Relaxed) {
                                                break;
                                            }
                                            let _ = tg!(
                                                "edit_message",
                                                &state_anim,
                                                chat_id,
                                                bot_anim
                                                    .edit_message_text(chat_id, msg_id, frame)
                                                    .await
                                            );
                                            last_text = frame.to_string();
                                        }
                                    }
                                }))
                            } else {
                                None
                            };

                            let verify_effort = {
                                let data = state_owned.lock().await;
                                get_effort_for_provider(&data.settings, chat_id, provider_str)
                            };
                            let verify_fast_mode = {
                                let data = state_owned.lock().await;
                                is_codex_fast(&data.settings, chat_id)
                            };
                            let sid_clone = session_id.clone();
                            let cwd_clone = cwd.clone();
                            let provider_for_verify = provider_str;
                            // The Codex verifier reads the full-fidelity
                            // archive, so refresh it before the verify call.
                            // (OpenCode uses native `--fork` and doesn't need
                            // the archive; Claude forks its live session and
                            // doesn't need it either.)
                            // archive_and_save_session is called directly —
                            // not via convert_and_save_session — because the
                            // latter short-circuits when its summary JSON is
                            // up-to-date, which would skip the archive
                            // refresh too. archive_and_save_session has its
                            // own independent mtime check so it stays
                            // idempotent across repeated calls.
                            if provider_for_verify == "codex" {
                                let sid_refresh = sid_clone.clone();
                                let cwd_refresh = cwd_clone.clone();
                                tokio::task::spawn_blocking(move || {
                                    match resolve_codex_by_id(&sid_refresh) {
                                        Some(info) => {
                                            msg_debug(&format!(
                                                "[loop] codex archive refresh: jsonl={}, id={}",
                                                info.jsonl_path.display(), info.session_id));
                                            crate::services::session_archive::archive_and_save_session(
                                                "codex",
                                                &info.jsonl_path,
                                                &info.session_id,
                                                &cwd_refresh,
                                            );
                                        }
                                        None => {
                                            msg_debug(&format!(
                                                "[loop] codex archive refresh SKIPPED: \
                                                 resolve_codex_by_id({}) returned None — \
                                                 verify may fail with \"Archive not found\"",
                                                sid_refresh));
                                        }
                                    }
                                }).await.ok();
                            }
                            let verify_result =
                                tokio::task::spawn_blocking(move || match provider_for_verify {
                                    "codex" => crate::services::codex::verify_completion_codex(
                                        &sid_clone,
                                        &cwd_clone,
                                        verify_effort.as_deref(),
                                        verify_fast_mode,
                                    ),
                                    "opencode" => {
                                        crate::services::opencode::verify_completion_opencode(
                                            &sid_clone, &cwd_clone,
                                        )
                                    }
                                    _ => crate::services::claude::verify_completion(
                                        &sid_clone,
                                        &cwd_clone,
                                        verify_effort.as_deref(),
                                    ),
                                })
                                .await;
                            // Stop animation and await its task so no stray
                            // edit lands after the delete below.
                            verify_anim_stop.store(true, Ordering::Relaxed);
                            if let Some(h) = verify_anim_handle {
                                let _ = h.await;
                            }
                            // Remove spinner
                            if let Some(msg_id) = verify_msg_id {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg!(
                                    "delete_message",
                                    bot_owned.delete_message(chat_id, msg_id).await
                                );
                            }
                            // Re-check: /stop may have been called during verification.
                            // /clear or /model (provider switch) may have removed loop_states
                            // during verification. Suppress all post-verify outcome messages
                            // (complete / limit / incomplete) so the user's explicit cancel is
                            // honoured uniformly. Each inner branch (complete/limit/reinject)
                            // additionally re-checks loop_states under the lock as a race-safety
                            // net for /clear or /model arriving between this gate and the
                            // branch-internal write.
                            let loop_active_post_verify = {
                                let data = state_owned.lock().await;
                                data.loop_states.contains_key(&chat_id)
                            };
                            if cancel_token.cancelled.load(Ordering::Relaxed) {
                                msg_debug("[loop] cancelled during verification — aborting loop");
                                // loop_states already removed by /stop handler
                            } else if !loop_active_post_verify {
                                msg_debug("[loop] loop_states removed during verification — skip post-verify outcome");
                            } else {
                                match verify_result {
                                    Ok(Ok(result)) => {
                                        if result.complete {
                                            msg_debug("[loop] mission_complete — loop finished");
                                            // Re-check loop_states under the lock: /clear or /model
                                            // (provider switch) may have removed it between the outer
                                            // gate and now. Honour that intent — skip the outcome
                                            // message uniformly with the reinject branch's safety net.
                                            let still_active = {
                                                let mut data = state_owned.lock().await;
                                                data.loop_states.remove(&chat_id).is_some()
                                            };
                                            if still_active {
                                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                                let _ = tg!("send_message", bot_owned.send_message(chat_id, "✅ Loop complete — task verified as done.").await);
                                            } else {
                                                msg_debug("[loop] loop_states removed before complete-message — skip");
                                            }
                                        } else if max_iterations > 0 && remaining <= 1 {
                                            msg_debug(
                                                "[loop] max iterations reached — loop stopped",
                                            );
                                            // Same race-safety net as the complete branch.
                                            let still_active = {
                                                let mut data = state_owned.lock().await;
                                                data.loop_states.remove(&chat_id).is_some()
                                            };
                                            if still_active {
                                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                                let feedback_preview = result
                                                    .feedback
                                                    .as_deref()
                                                    .unwrap_or("(no details)");
                                                let msg = format!(
                                                    "⚠️ Loop limit reached. Remaining issue:\n{}",
                                                    feedback_preview
                                                );
                                                let _ = tg!(
                                                    "send_message",
                                                    bot_owned.send_message(chat_id, &msg).await
                                                );
                                            } else {
                                                msg_debug("[loop] loop_states removed before limit-message — skip");
                                            }
                                        } else {
                                            // Incomplete: use feedback if available, otherwise fall back to original request
                                            let reinject_text = result.feedback.unwrap_or_else(|| {
                                            format!("Continue the previous task. The original request was: {}", original_request)
                                        });
                                            // For unlimited (max_iterations==0): don't decrement, track iteration count separately
                                            let (new_remaining, iteration) = if max_iterations == 0
                                            {
                                                // remaining counts UP from 0 for display; never decremented to trigger limit
                                                (
                                                    remaining.saturating_add(1),
                                                    remaining.saturating_add(1),
                                                )
                                            } else {
                                                let nr = remaining - 1;
                                                (nr, max_iterations - nr)
                                            };
                                            msg_debug(&format!("[loop] incomplete — re-requesting (remaining={}): {:?}",
                                            new_remaining, truncate_str(&reinject_text, 100)));
                                            let iter_label = if max_iterations == 0 {
                                                format!(
                                                    "🔄 Loop iteration {}\n{}",
                                                    iteration, reinject_text
                                                )
                                            } else {
                                                format!(
                                                    "🔄 Loop iteration {}/{}\n{}",
                                                    iteration, max_iterations, reinject_text
                                                )
                                            };
                                            shared_rate_limit_wait(&state_owned, chat_id).await;
                                            let _ = tg!(
                                                "send_message",
                                                bot_owned.send_message(chat_id, &iter_label).await
                                            );
                                            // Store feedback for dispatch_loop_feedback to pick up.
                                            // Cannot call handle_text_message directly here because
                                            // its future is not Send-safe inside tokio::spawn.
                                            // Re-check cancellation under the lock so /stop arriving
                                            // between the verify-result check and this insert cannot
                                            // leave behind a feedback that re-injects after /stop.
                                            {
                                                let mut data = state_owned.lock().await;
                                                if cancel_token.cancelled.load(Ordering::Relaxed) {
                                                    msg_debug("[loop] cancelled during result processing — skip reinject");
                                                } else if !data.loop_states.contains_key(&chat_id) {
                                                    // loop_states cleared externally (e.g. /clear, /model provider switch)
                                                    // between verify start and finish — honour that intent and skip reinject
                                                    // so the loop does not silently resume.
                                                    msg_debug("[loop] loop_states removed during verification — skip reinject");
                                                } else {
                                                    if let Some(ls) =
                                                        data.loop_states.get_mut(&chat_id)
                                                    {
                                                        ls.remaining = new_remaining;
                                                    }
                                                    data.loop_feedback.insert(
                                                        chat_id,
                                                        (
                                                            reinject_text,
                                                            user_display_name_owned.clone(),
                                                        ),
                                                    );
                                                    loop_reinjected = true;
                                                }
                                            }
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        msg_debug(&format!(
                                            "[loop] verify_completion error: {}",
                                            e
                                        ));
                                        {
                                            let mut data = state_owned.lock().await;
                                            data.loop_states.remove(&chat_id);
                                        }
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let _ = tg!(
                                            "send_message",
                                            bot_owned
                                                .send_message(
                                                    chat_id,
                                                    &format!("⚠️ Loop verification failed: {}", e)
                                                )
                                                .await
                                        );
                                    }
                                    Err(e) => {
                                        msg_debug(&format!("[loop] spawn_blocking error: {}", e));
                                        {
                                            let mut data = state_owned.lock().await;
                                            data.loop_states.remove(&chat_id);
                                        }
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let _ = tg!(
                                            "send_message",
                                            bot_owned
                                                .send_message(
                                                    chat_id,
                                                    &format!("⚠️ Loop verification failed: {}", e)
                                                )
                                                .await
                                        );
                                    }
                                }
                            } // else (not cancelled during verification)
                        } else {
                            msg_debug("[loop] no session_id available — cannot verify");
                            {
                                let mut data = state_owned.lock().await;
                                data.loop_states.remove(&chat_id);
                            }
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!("send_message", bot_owned.send_message(chat_id, "⚠️ Loop stopped — no session ID available for verification.").await);
                        }
                    }
                }
            }

            // === Queue processing (both during streaming and after done) ===
            let queued = process_upload_queue(&bot_owned, chat_id, &state_owned).await;
            if done {
                queue_done = !queued;
            }
        }

        // === Post-loop: cancelled handling or lock release ===
        if cancelled {
            cancel_token_now(&cancel_token);

            // stopped_response (full) is used for session history
            let stopped_response = if full_response.trim().is_empty() {
                "[Stopped]".to_string()
            } else {
                let normalized = normalize_empty_lines(&full_response);
                format!("{}\n\n[Stopped]", normalized)
            };

            shared_rate_limit_wait(&state_owned, chat_id).await;

            // ── Show only remaining delta + [Stopped] (unified rolling placeholder) ──
            if full_response.len() < last_confirmed_len
                || !full_response.is_char_boundary(last_confirmed_len)
            {
                last_confirmed_len = 0;
            }
            let remaining = &full_response[last_confirmed_len..];
            msg_debug(&format!(
                "[rolling_ph] STOPPED: placeholder_msg_id={}, confirmed={}, remaining_len={}",
                placeholder_msg_id,
                last_confirmed_len,
                remaining.trim().len()
            ));
            if should_attach_response_as_file(full_response.len(), provider_str) {
                // Large stopped response — send as file
                msg_debug(&format!(
                    "[rolling_ph] STOPPED FILE ATTACH: total={}",
                    full_response.len()
                ));
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "edit_message",
                    bot_owned
                        .edit_message_text(
                            chat_id,
                            placeholder_msg_id,
                            "\u{1f4c4} Response attached as file [Stopped]"
                        )
                        .await
                );
                send_response_as_file(
                    &bot_owned,
                    chat_id,
                    &stopped_response,
                    &state_owned,
                    "response",
                )
                .await;
            } else {
                let display_stopped = if remaining.trim().is_empty() {
                    "[Stopped]".to_string()
                } else {
                    let normalized = normalize_empty_lines(remaining);
                    format!("{}\n\n[Stopped]", normalized)
                };
                let html_stopped = markdown_to_telegram_html(&display_stopped);
                if html_stopped.len() <= TELEGRAM_MSG_LIMIT {
                    let stopped_html_edit_result = tg_timeout!(
                        "edit_message",
                        &state_owned,
                        chat_id,
                        bot_owned
                            .edit_message_text(chat_id, placeholder_msg_id, &html_stopped)
                            .parse_mode(ParseMode::Html)
                            .await
                    )
                    .map(|_| ());
                    if let Err(e) = stopped_html_edit_result {
                        let ts_err = chrono::Local::now().format("%H:%M:%S");
                        println!("  [{ts_err}]   ⚠ edit_message failed (stopped HTML): {e}");
                        if let Some(remaining_secs) =
                            active_telegram_retry_after_remaining_secs(&state_owned, chat_id)
                                .await
                                .filter(|secs| *secs >= TELEGRAM_NONESSENTIAL_RETRY_AFTER_SKIP_SECS)
                        {
                            msg_debug(&format!(
                                "[direct_no_edit] stopped fallback send skipped due active RetryAfter cooldown: remaining_secs={}",
                                remaining_secs
                            ));
                        } else {
                            if send_direct_segment_as_message(
                                &bot_owned,
                                chat_id,
                                &display_stopped,
                                &state_owned,
                            )
                            .await
                            {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg_timeout!(
                                    "delete_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                );
                            } else {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg_timeout!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(
                                            chat_id,
                                            placeholder_msg_id,
                                            &display_stopped
                                        )
                                        .await
                                );
                            }
                        }
                    }
                } else {
                    let send_result = send_long_message(
                        &bot_owned,
                        chat_id,
                        &html_stopped,
                        Some(ParseMode::Html),
                        &state_owned,
                    )
                    .await;
                    match send_result {
                        Ok(_) => {
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!(
                                "delete_message",
                                bot_owned.delete_message(chat_id, placeholder_msg_id).await
                            );
                        }
                        Err(_) => {
                            let fallback = send_long_message(
                                &bot_owned,
                                chat_id,
                                &display_stopped,
                                None,
                                &state_owned,
                            )
                            .await;
                            match fallback {
                                Ok(_) => {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg!(
                                        "delete_message",
                                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                    );
                                }
                                Err(_) => {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let truncated =
                                        truncate_str(&display_stopped, TELEGRAM_MSG_LIMIT);
                                    let _ = tg!(
                                        "edit_message",
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &truncated
                                            )
                                            .await
                                    );
                                }
                            }
                        }
                    }
                }
            }

            let stop_msg_id = {
                let data = state_owned.lock().await;
                data.stop_message_ids.get(&chat_id).cloned()
            };
            if let Some(msg_id) = stop_msg_id {
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "delete_message",
                    bot_owned.delete_message(chat_id, msg_id).await
                );
            }

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ■ Stopped");

            let mut data = state_owned.lock().await;
            // Same guard as the normal-completion branch: if /model or /start changed the
            // session during this task, skip the partial-response writeback so it does not
            // bleed into the new context. Group chat log is still written below — other bots
            // benefit from seeing the partial response, and the shared log is not part of the
            // (now-cleared) session it would have leaked into.
            //
            // detect_provider mirrors the spawn-time capture (see normal-completion guard).
            // sid + clear_epoch mirror normal-completion: catch /clear, /start same-path swaps,
            // and brand-new-session /clear (where sid stays None).
            let model_now = data.settings.models.get(&chat_id.0.to_string()).cloned();
            let provider_now = detect_provider(model_now.as_deref());
            let path_now = data
                .sessions
                .get(&chat_id)
                .and_then(|s| s.current_path.clone());
            let sid_now = data
                .sessions
                .get(&chat_id)
                .and_then(|s| s.session_id.clone());
            let epoch_now = data.clear_epoch.get(&chat_id).copied().unwrap_or(0);
            let session_changed = provider_now != provider_str
                || path_now.as_deref() != Some(current_path.as_str())
                || sid_now != captured_sid
                || epoch_now != captured_clear_epoch;
            if session_changed {
                msg_debug(&format!("[polling] session changed during stopped task — skip session writeback (path: {:?} → {:?}, provider: {} → {}, sid: {:?} → {:?}, epoch: {} → {})",
                    current_path, path_now, provider_str, provider_now, captured_sid, sid_now, captured_clear_epoch, epoch_now));
            } else if let Some(session) = data.sessions.get_mut(&chat_id) {
                if let Some(sid) = new_session_id {
                    session.session_id = Some(sid);
                }
                invalidate_saved_session_after_stream_error(
                    session,
                    provider_str,
                    &stopped_response,
                );
                session.history.push(HistoryItem {
                    item_type: HistoryType::User,
                    content: user_text_owned.clone(),
                });
                session.history.push(HistoryItem {
                    item_type: HistoryType::Assistant,
                    content: stopped_response,
                });
                save_session_to_file(session, &current_path, provider_str);
            }
            // Write to group chat shared log (for cross-bot context sharing).
            // Mirrors the normal-completion branch: written regardless of session_changed.
            msg_debug(&format!(
                "[polling] JSONL stopped check: chat_id={}, raw_entries_count={}",
                chat_id.0,
                raw_entries.len()
            ));
            if chat_id.0 < 0 {
                let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                let dn = if bot_display_name_for_log.is_empty() {
                    None
                } else {
                    Some(bot_display_name_for_log.clone())
                };
                append_group_chat_log(
                    chat_id.0,
                    &GroupChatLogEntry {
                        ts: now_ts.clone(),
                        bot: bot_username_for_log.clone(),
                        bot_display_name: dn.clone(),
                        role: "user".to_string(),
                        from: Some(user_display_name_owned.clone()),
                        text: user_text_owned,
                        clear: false,
                    },
                );
                if !raw_entries.is_empty() {
                    msg_debug(&format!("[polling] JSONL stopped: writing user+assistant entries, raw_entries_count={}", raw_entries.len()));
                    append_group_chat_log(
                        chat_id.0,
                        &GroupChatLogEntry {
                            ts: now_ts,
                            bot: bot_username_for_log.clone(),
                            bot_display_name: dn,
                            role: "assistant".to_string(),
                            from: None,
                            text: serialize_payload(&std::mem::take(&mut raw_entries)),
                            clear: false,
                        },
                    );
                } else {
                    msg_debug(&format!("[polling] JSONL stopped: user entry written, assistant SKIPPED (raw_entries is empty)"));
                }
            }
            data.cancel_tokens.remove(&chat_id);
            data.stop_message_ids.remove(&chat_id);
            drop(data);
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=text_poll_cancelled",
                chat_id.0
            ));
            drop(_group_lock); // release group chat lock before processing queue
            process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
            return;
        }

        // If loop verification stored feedback, clean up and dispatch it.
        // Uses dispatch_loop_feedback (bypasses queue mode check, no "Dequeued" noise).
        // Backstop: if /stop fired after the in-lock check above, drop the dispatch.
        if loop_reinjected && !cancel_token.cancelled.load(Ordering::Relaxed) {
            msg_debug("[queue:trigger] loop_reinjected — dispatching loop feedback");
            let orphan_stop_msg = {
                let mut data = state_owned.lock().await;
                data.cancel_tokens.remove(&chat_id);
                data.stop_message_ids.remove(&chat_id)
            };
            if let Some(msg_id) = orphan_stop_msg {
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "delete_message",
                    bot_owned.delete_message(chat_id, msg_id).await
                );
            }
            drop(_group_lock);
            dispatch_loop_feedback(&bot_owned, chat_id, &state_owned).await;
            return;
        }

        // Atomically remove both cancel_tokens and stop_message_ids to prevent
        // race with /stop handler inserting a stop_msg_id between two separate locks
        let orphan_stop_msg = {
            let mut data = state_owned.lock().await;
            let msg_id = data.stop_message_ids.remove(&chat_id);
            data.cancel_tokens.remove(&chat_id);
            msg_id
        };
        if let Some(msg_id) = orphan_stop_msg {
            shared_rate_limit_wait(&state_owned, chat_id).await;
            let _ = tg!(
                "delete_message",
                bot_owned.delete_message(chat_id, msg_id).await
            );
        }
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=text_poll_completed",
            chat_id.0
        ));
        drop(_group_lock); // release group chat lock before processing queue
        process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
    });

    Ok(())
}

/// Load existing session from ai_sessions directory matching the given path and provider
fn load_existing_session(
    current_path: &str,
    provider: &str,
) -> Option<(SessionData, std::time::SystemTime)> {
    msg_debug(&format!(
        "[load_session] looking for path={:?}, provider={}",
        current_path, provider
    ));
    let sessions_dir = ai_screen::ai_sessions_dir()?;

    if !sessions_dir.exists() {
        msg_debug(&format!(
            "[load_session] sessions_dir not found: {}",
            sessions_dir.display()
        ));
        return None;
    }

    let mut with_session_id: Option<(SessionData, std::time::SystemTime)> = None;
    let mut without_session_id: Option<(SessionData, std::time::SystemTime)> = None;
    let mut file_count = 0u32;
    let mut path_mismatch_sample: Option<String> = None;

    if let Ok(entries) = fs::read_dir(&sessions_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                file_count += 1;
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(session_data) = serde_json::from_str::<SessionData>(&content) {
                        if session_data.current_path == current_path {
                            // Provider filter: match exact provider, or allow empty (legacy files)
                            if !session_data.provider.is_empty()
                                && session_data.provider != provider
                            {
                                msg_debug(&format!("[load_session] skipped session_id={} (provider mismatch: {} != {})",
                                    session_data.session_id, session_data.provider, provider));
                                continue;
                            }
                            msg_debug(&format!("[load_session] found session_id={}, provider={}, history_len={}, stored_path={:?}",
                                session_data.session_id, session_data.provider, session_data.history.len(), session_data.current_path));
                            if let Ok(metadata) = path.metadata() {
                                if let Ok(modified) = metadata.modified() {
                                    let target = if !session_data.session_id.is_empty() {
                                        &mut with_session_id
                                    } else {
                                        &mut without_session_id
                                    };
                                    match target {
                                        None => *target = Some((session_data, modified)),
                                        Some((_, latest_time)) if modified > *latest_time => {
                                            *target = Some((session_data, modified));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        } else if path_mismatch_sample.is_none() {
                            path_mismatch_sample = Some(session_data.current_path.clone());
                        }
                    }
                }
            }
        }
    }

    if provider == "codex" {
        if let Some((session_data, _)) = with_session_id.as_ref() {
            if let Some(db_path) = codex_state_db_path() {
                if let Some(stale_rollout_path) =
                    stale_codex_rollout_path_from_state_db(&db_path, &session_data.session_id)
                {
                    let archived =
                        archive_codex_thread_in_state_db(&db_path, &session_data.session_id);
                    msg_debug(&format!(
                        "[load_session] stale codex resume preflight hit: session_id={}, rollout_path={}, archived_db_thread={}",
                        session_data.session_id,
                        stale_rollout_path.display(),
                        archived
                    ));
                    without_session_id = Some(downgrade_stale_saved_codex_session_to_blocker(
                        session_data,
                        current_path,
                        provider,
                    ));
                    with_session_id = None;
                }
            }
        }
    }

    let matching_session = with_session_id.or(without_session_id);

    msg_debug(&format!(
        "[load_session] scanned {} json files, result={}",
        file_count,
        if matching_session.is_some() {
            "found"
        } else {
            "None"
        }
    ));
    if matching_session.is_none() && file_count > 0 {
        if let Some(sample) = path_mismatch_sample {
            msg_debug(&format!(
                "[load_session] path mismatch example: stored={:?} vs wanted={:?}",
                sample, current_path
            ));
        }
    }

    matching_session
}

/// Remove stale session files without session_id for the same current_path + provider.
/// Called when no file with session_id exists for this path+provider.
/// Keeps the most recently modified empty file (the one selected by load_existing_session)
/// and deletes the rest.
fn cleanup_session_files(current_path: &str, provider: &str) {
    let Some(sessions_dir) = ai_screen::ai_sessions_dir() else {
        return;
    };
    let Ok(entries) = fs::read_dir(&sessions_dir) else {
        return;
    };

    // Collect matching empty files with their modification time
    let mut empty_files: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(data) = serde_json::from_str::<SessionData>(&content) {
                    if data.current_path == current_path
                        && (data.provider == provider || data.provider.is_empty())
                        && data.session_id.is_empty()
                    {
                        let modified = path
                            .metadata()
                            .ok()
                            .and_then(|m| m.modified().ok())
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        empty_files.push((path, modified));
                    }
                }
            }
        }
    }

    if empty_files.len() <= 1 {
        return;
    }

    // Find the latest modified one to keep
    let latest_idx = empty_files
        .iter()
        .enumerate()
        .max_by_key(|(_, (_, t))| *t)
        .map(|(i, _)| i)
        .unwrap();

    // Delete all except the latest
    for (i, (path, _)) in empty_files.iter().enumerate() {
        if i != latest_idx {
            let _ = fs::remove_file(path);
        }
    }
}

fn codex_state_db_path() -> Option<std::path::PathBuf> {
    let db_path = dirs::home_dir()?.join(".codex").join("state_5.sqlite");
    if db_path.is_file() {
        Some(db_path)
    } else {
        None
    }
}

fn codex_rollout_path_from_state_db(
    db_path: &Path,
    session_id: &str,
) -> Option<std::path::PathBuf> {
    if session_id.is_empty() || !db_path.is_file() {
        return None;
    }
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .ok()?;
    let mut stmt = conn
        .prepare("SELECT rollout_path FROM threads WHERE id = ?1 AND archived = 0 LIMIT 1")
        .ok()?;
    let rollout_path: String = stmt
        .query_row(rusqlite::params![session_id], |row| row.get(0))
        .ok()?;
    if rollout_path.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(rollout_path))
}

fn stale_codex_rollout_path_from_state_db(
    db_path: &Path,
    session_id: &str,
) -> Option<std::path::PathBuf> {
    let rollout_path = codex_rollout_path_from_state_db(db_path, session_id)?;
    if rollout_path.is_file() {
        None
    } else {
        Some(rollout_path)
    }
}

fn archive_codex_thread_in_state_db(db_path: &Path, session_id: &str) -> bool {
    if session_id.is_empty() || !db_path.is_file() {
        return false;
    }
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE,
    ) {
        Ok(conn) => conn,
        Err(_) => return false,
    };
    conn.execute(
        "UPDATE threads SET archived = 1 WHERE id = ?1 AND archived = 0",
        rusqlite::params![session_id],
    )
    .map(|changed| changed > 0)
    .unwrap_or(false)
}

fn downgrade_stale_saved_codex_session_to_blocker(
    session_data: &SessionData,
    current_path: &str,
    provider: &str,
) -> (SessionData, std::time::SystemTime) {
    let downgraded = SessionData {
        session_id: String::new(),
        history: session_data.history.clone(),
        current_path: current_path.to_string(),
        created_at: session_data.created_at.clone(),
        provider: if session_data.provider.is_empty() {
            provider.to_string()
        } else {
            session_data.provider.clone()
        },
    };
    let blocker_session = ChatSession {
        session_id: None,
        current_path: Some(current_path.to_string()),
        history: downgraded.history.clone(),
        pending_uploads: Vec::new(),
    };
    save_session_to_file(&blocker_session, current_path, provider);
    (downgraded, std::time::SystemTime::now())
}

/// Save session to file in the ai_sessions directory
fn save_session_to_file(session: &ChatSession, current_path: &str, provider: &str) {
    if session.history.is_empty() {
        msg_debug("[save_session] skipped: empty history");
        return;
    }

    let Some(sessions_dir) = ai_screen::ai_sessions_dir() else {
        msg_debug("[save_session] skipped: ai_sessions_dir() returned None");
        return;
    };

    if fs::create_dir_all(&sessions_dir).is_err() {
        msg_debug("[save_session] skipped: create_dir_all failed");
        return;
    }

    // Filter out system messages
    let saveable_history: Vec<HistoryItem> = session
        .history
        .iter()
        .filter(|item| !matches!(item.item_type, HistoryType::System))
        .cloned()
        .collect();

    if saveable_history.is_empty() {
        return;
    }

    let session_id = session.session_id.clone().unwrap_or_default();

    let session_data = SessionData {
        session_id: session_id.clone(),
        history: saveable_history,
        current_path: current_path.to_string(),
        created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        provider: provider.to_string(),
    };
    msg_debug(&format!(
        "[save_session] provider={}, session_id={}, path={}",
        provider, session_id, current_path
    ));

    let file_path = if session_id.is_empty() {
        sessions_dir.join(save_session_blocker_file_name(current_path, provider))
    } else {
        // Security: whitelist session_id to alphanumeric, hyphens, underscores only
        if !session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return;
        }
        sessions_dir.join(format!("{}.json", session_id))
    };

    if let Ok(json) = serde_json::to_string_pretty(&session_data) {
        let _ = fs::write(&file_path, json);
    }

    // Clean up old session files for the same path+provider (removes orphaned files
    // left by Codex's per-call thread_id rotation and /clear blocker files)
    if let Ok(entries) = fs::read_dir(&sessions_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let p = entry.path();
            if p == file_path {
                continue;
            }
            if p.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&p) {
                    if let Ok(old) = serde_json::from_str::<SessionData>(&content) {
                        if old.current_path == current_path
                            && (old.provider == provider || old.provider.is_empty())
                        {
                            let _ = fs::remove_file(&p);
                        }
                    }
                }
            }
        }
    }
}

/// Find the largest byte index <= `index` that is a valid UTF-8 char boundary
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        s.len()
    } else {
        let mut i = index;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

/// Process one pending upload queue file for the given chat.
/// Scans ~/.cokacdir/upload_queue/ for .queue files matching the current bot and chat_id,
/// sends the oldest one, and deletes the queue file on success.
/// Returns true if a file was processed (rate limit slot consumed).
async fn process_upload_queue(bot: &Bot, chat_id: ChatId, state: &SharedState) -> bool {
    let queue_dir = match dirs::home_dir() {
        Some(h) => h.join(".cokacdir").join("upload_queue"),
        None => return false,
    };
    if !queue_dir.is_dir() {
        return false;
    }

    let current_key = token_hash(bot.token());

    // Collect and sort queue files by name (timestamp-based, so alphabetical = chronological)
    let mut entries: Vec<std::path::PathBuf> = match fs::read_dir(&queue_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("queue"))
            .collect(),
        Err(_) => return false,
    };
    entries.sort();

    // Find the first entry matching this bot and chat_id
    for entry_path in entries {
        let content = match fs::read_to_string(&entry_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let file_chat_id = json.get("chat_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let file_key = json.get("key").and_then(|v| v.as_str()).unwrap_or("");
        let file_path = json.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if file_chat_id != chat_id.0 || file_key != current_key || file_path.is_empty() {
            continue;
        }

        let path = std::path::PathBuf::from(file_path);
        if !path.exists() {
            // File no longer exists, remove queue entry
            let _ = fs::remove_file(&entry_path);
            return false;
        }

        // Remove queue file before sending (regardless of send result)
        let _ = fs::remove_file(&entry_path);

        // Rate limit and send
        shared_rate_limit_wait(state, chat_id).await;
        match tg!(
            "send_document",
            bot.send_document(chat_id, teloxide::types::InputFile::file(&path),)
                .await
        ) {
            Ok(_) => {
                let ts = chrono::Local::now().format("%H:%M:%S");
                println!("  [{ts}]   📤 Upload sent: {}", file_path);
            }
            Err(e) => {
                let ts = chrono::Local::now().format("%H:%M:%S");
                println!("  [{ts}]   ⚠ Upload failed: {e}");
            }
        }
        return true;
    }

    false
}

/// Acquires the lock briefly to calculate and reserve the next API call slot,
/// then releases the lock and sleeps until the reserved time.
/// This ensures that even concurrent tasks for the same chat maintain a
/// bounded Telegram API gap independent of the UI/stream polling cadence.
async fn reserve_shared_rate_limit_slot(
    state: &SharedState,
    chat_id: ChatId,
) -> tokio::time::Instant {
    let mut data = state.lock().await;
    let now = tokio::time::Instant::now();
    let base_gap_ms = data.api_min_gap_ms;
    let effective_gap_ms = {
        let stats = data.telegram_runtime_stats.entry(chat_id).or_default();
        clear_expired_telegram_backoff(stats, now);
        effective_telegram_api_gap_ms(base_gap_ms, stats, now)
    };
    let min_gap = tokio::time::Duration::from_millis(effective_gap_ms);
    let last = data
        .api_timestamps
        .entry(chat_id)
        .or_insert_with(|| tokio::time::Instant::now() - tokio::time::Duration::from_secs(10));
    let earliest_next = *last + min_gap;
    let target = if earliest_next > now {
        earliest_next
    } else {
        now
    };
    *last = target; // Reserve this slot
    target
}

async fn shared_rate_limit_wait(state: &SharedState, chat_id: ChatId) {
    let sleep_until = reserve_shared_rate_limit_slot(state, chat_id).await;
    let chunk = tokio::time::Duration::from_millis(250);

    loop {
        let now = tokio::time::Instant::now();
        if now >= sleep_until {
            return;
        }

        let cancelled = {
            let data = state.lock().await;
            data.cancel_tokens
                .get(&chat_id)
                .is_some_and(|token| token.cancelled.load(Ordering::Relaxed))
        };
        if cancelled {
            return;
        }

        tokio::time::sleep(std::cmp::min(chunk, sleep_until - now)).await;
    }
}

#[cfg(test)]
mod shared_rate_limit_wait_tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn test_shared_state(api_min_gap_ms: u64) -> SharedState {
        Arc::new(tokio::sync::Mutex::new(SharedData {
            sessions: HashMap::new(),
            settings: BotSettings::default(),
            cancel_tokens: HashMap::new(),
            stop_message_ids: HashMap::new(),
            api_timestamps: HashMap::new(),
            polling_time_ms: MIN_TELEGRAM_POLLING_TIME_MS,
            api_min_gap_ms,
            telegram_runtime_stats: HashMap::new(),
            pending_schedules: HashMap::new(),
            message_queues: HashMap::new(),
            bot_username: "test_bot".to_string(),
            bot_display_name: "Test Bot".to_string(),
            api_base_url: "https://api.telegram.org".to_string(),
            loop_states: HashMap::new(),
            loop_feedback: HashMap::new(),
            clear_epoch: HashMap::new(),
        }))
    }

    #[tokio::test]
    async fn returns_quickly_for_cancelled_chat_during_long_retry_wait() {
        let state = test_shared_state(1_000);
        let chat_id = ChatId(1);
        let token = Arc::new(CancelToken::new());
        token.cancelled.store(true, Ordering::Relaxed);

        {
            let mut data = state.lock().await;
            data.cancel_tokens.insert(chat_id, token);
            data.api_timestamps.insert(
                chat_id,
                tokio::time::Instant::now() + tokio::time::Duration::from_secs(5),
            );
        }

        let started = std::time::Instant::now();
        shared_rate_limit_wait(&state, chat_id).await;
        assert!(
            started.elapsed() < std::time::Duration::from_millis(200),
            "cancelled wait should not sleep through a long retry-after"
        );
    }
}

/// Honor a Telegram-server-mandated `RetryAfter` by pushing the next safe
/// call time for `chat_id` forward by the requested duration. Subsequent
/// `shared_rate_limit_wait` calls for the same chat will then naturally
/// wait the full server-mandated cooldown before allowing another call.
///
/// Telegram explicitly instructs clients to respect `RetryAfter`; ignoring
/// it causes the cooldown to escalate (production logs have shown bans
/// accumulating to ~14000s after repeated violations).
async fn honor_telegram_retry_after<T>(
    state: &SharedState,
    chat_id: ChatId,
    result: &Result<T, teloxide::RequestError>,
) {
    if let Err(teloxide::RequestError::RetryAfter(seconds)) = result {
        let now = tokio::time::Instant::now();
        let until = now + seconds.duration();
        let mut data = state.lock().await;
        let base_gap_ms = data.api_min_gap_ms;
        let entry = data.api_timestamps.entry(chat_id).or_insert(until);
        if *entry < until {
            *entry = until;
        }
        let stats = data.telegram_runtime_stats.entry(chat_id).or_default();
        clear_expired_telegram_backoff(stats, now);
        stats.retry_after_hits += 1;
        stats.last_retry_after_secs = Some(seconds.duration().as_secs());
        let adaptive_gap_ms =
            next_telegram_adaptive_api_gap_ms(base_gap_ms, stats.adaptive_api_min_gap_ms);
        stats.adaptive_api_min_gap_ms = Some(adaptive_gap_ms);
        stats.adaptive_api_min_gap_until =
            Some(now + tokio::time::Duration::from_secs(TELEGRAM_ADAPTIVE_API_MIN_GAP_HOLD_SECS));
        ai_trace(&format!(
            "[TG_RETRY_AFTER] chat={} cooldown_secs={} configured_gap_ms={} effective_gap_ms={} retry_hits={} hold_secs={}",
            chat_id.0,
            seconds.duration().as_secs(),
            base_gap_ms,
            adaptive_gap_ms,
            stats.retry_after_hits,
            TELEGRAM_ADAPTIVE_API_MIN_GAP_HOLD_SECS
        ));
    }
}

const TELEGRAM_NONESSENTIAL_RETRY_AFTER_SKIP_SECS: u64 = 60;

async fn active_telegram_retry_after_remaining_secs(
    state: &SharedState,
    chat_id: ChatId,
) -> Option<u64> {
    let now = tokio::time::Instant::now();
    let data = state.lock().await;
    data.api_timestamps.get(&chat_id).and_then(|until| {
        if *until > now {
            Some((*until - now).as_secs())
        } else {
            None
        }
    })
}

/// Send a message that may exceed Telegram's 4096 character limit
/// by splitting it into multiple messages, handling UTF-8 boundaries
/// and unclosed HTML tags (e.g. <pre>) across split points
async fn send_long_message(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    parse_mode: Option<ParseMode>,
    state: &SharedState,
) -> ResponseResult<()> {
    if text.len() <= TELEGRAM_MSG_LIMIT {
        shared_rate_limit_wait(state, chat_id).await;
        let mut req = bot.send_message(chat_id, text);
        if let Some(mode) = parse_mode {
            req = req.parse_mode(mode);
        }
        tg_timeout!("send_message", state, chat_id, req.await)?;
        return Ok(());
    }

    let is_html = parse_mode.is_some();
    let mut remaining = text;
    let mut in_pre = false;

    while !remaining.is_empty() {
        // Reserve space for tags we may need to add (<pre> + </pre> = 11 bytes)
        let tag_overhead = if is_html && in_pre { 11 } else { 0 };
        let effective_limit = TELEGRAM_MSG_LIMIT.saturating_sub(tag_overhead);

        if remaining.len() <= effective_limit {
            let mut chunk = String::new();
            if is_html && in_pre {
                chunk.push_str("<pre>");
            }
            chunk.push_str(remaining);

            shared_rate_limit_wait(state, chat_id).await;
            let mut req = bot.send_message(chat_id, &chunk);
            if let Some(mode) = parse_mode {
                req = req.parse_mode(mode);
            }
            tg_timeout!("send_message", state, chat_id, req.await)?;
            break;
        }

        // Find a safe UTF-8 char boundary, then find a newline before it.
        // Only split at a newline if it produces a non-empty chunk; otherwise
        // use the full UTF-8-safe boundary (a leading '\n' would yield an
        // empty raw_chunk, which Telegram rejects with "text must be non-empty").
        let safe_end = floor_char_boundary(remaining, effective_limit);
        let split_at = match remaining[..safe_end].rfind('\n') {
            Some(pos) if pos > 0 => pos,
            _ => safe_end,
        };

        let (raw_chunk, rest) = remaining.split_at(split_at);

        let mut chunk = String::new();
        if is_html && in_pre {
            chunk.push_str("<pre>");
        }
        chunk.push_str(raw_chunk);

        // Track unclosed <pre> tags to close/reopen across chunks
        if is_html {
            let last_open = raw_chunk.rfind("<pre>");
            let last_close = raw_chunk.rfind("</pre>");
            in_pre = match (last_open, last_close) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => in_pre,
            };
            if in_pre {
                chunk.push_str("</pre>");
            }
        }

        shared_rate_limit_wait(state, chat_id).await;
        let mut req = bot.send_message(chat_id, &chunk);
        if let Some(mode) = parse_mode {
            req = req.parse_mode(mode);
        }
        tg_timeout!("send_message", state, chat_id, req.await)?;

        // Skip the newline character at the split point
        remaining = rest.strip_prefix('\n').unwrap_or(rest);
    }

    Ok(())
}

/// Send a large AI response as a .txt file attachment.
/// Returns true if the file was successfully sent.
async fn send_response_as_file(
    bot: &Bot,
    chat_id: ChatId,
    response: &str,
    state: &SharedState,
    label: &str,
) -> bool {
    let Some(home) = dirs::home_dir() else {
        return false;
    };
    let tmp_dir = home.join(".cokacdir").join("tmp");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let tmp_path = tmp_dir.join(format!("cokacdir_{}_{}.txt", label, timestamp));
    if std::fs::write(&tmp_path, response).is_err() {
        return false;
    }
    shared_rate_limit_wait(state, chat_id).await;
    let result = tg_timeout!(
        "send_document",
        bot.send_document(chat_id, teloxide::types::InputFile::file(&tmp_path),)
            .await
    );
    let _ = std::fs::remove_file(&tmp_path);
    result.is_ok()
}

/// Normalize consecutive empty lines to maximum of one
fn normalize_empty_lines(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_was_empty = false;

    for line in s.lines() {
        let is_empty = line.is_empty();
        if is_empty {
            if !prev_was_empty {
                result.push('\n');
            }
            prev_was_empty = true;
        } else {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(line);
            prev_was_empty = false;
        }
    }

    result
}

/// Escape special HTML characters for Telegram HTML parse mode
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Truncate a string to max_len bytes, cutting at a safe UTF-8 char and line boundary
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }

    let safe_end = floor_char_boundary(s, max_len);
    let truncated = &s[..safe_end];
    // Prefer cutting at a newline, but only if it leaves non-empty content.
    // Otherwise fall back to the full UTF-8-safe truncation.
    if let Some(pos) = truncated.rfind('\n') {
        if pos > 0 {
            return truncated[..pos].to_string();
        }
    }
    truncated.to_string()
}

/// Convert standard markdown to Telegram-compatible HTML
fn markdown_to_telegram_html(md: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut result = String::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim_start();

        // Fenced code block
        if trimmed.starts_with("```") {
            let mut code_lines = Vec::new();
            i += 1; // skip opening ```
            while i < lines.len() {
                if lines[i].trim_start().starts_with("```") {
                    break;
                }
                code_lines.push(lines[i]);
                i += 1;
            }
            let code = code_lines.join("\n");
            if !code.is_empty() {
                result.push_str(&format!("<pre>{}</pre>", html_escape(code.trim_end())));
            }
            result.push('\n');
            i += 1; // skip closing ```
            continue;
        }

        // Heading (# ~ ######)
        if let Some(rest) = strip_heading(trimmed) {
            result.push_str(&format!("<b>{}</b>", convert_inline(&html_escape(rest))));
            result.push('\n');
            i += 1;
            continue;
        }

        // Unordered list (- or *)
        if trimmed.starts_with("- ") {
            result.push_str(&format!(
                "• {}",
                convert_inline(&html_escape(&trimmed[2..]))
            ));
            result.push('\n');
            i += 1;
            continue;
        }
        if trimmed.starts_with("* ") && !trimmed.starts_with("**") {
            result.push_str(&format!(
                "• {}",
                convert_inline(&html_escape(&trimmed[2..]))
            ));
            result.push('\n');
            i += 1;
            continue;
        }

        // Regular line
        result.push_str(&convert_inline(&html_escape(lines[i])));
        result.push('\n');
        i += 1;
    }

    result.trim_end().to_string()
}

/// Strip markdown heading prefix (# ~ ######), return remaining text
fn strip_heading(line: &str) -> Option<&str> {
    let trimmed = line.trim_start_matches('#');
    // Must have consumed at least one # and be followed by a space
    if trimmed.len() < line.len() && trimmed.starts_with(' ') {
        let hashes = line.len() - trimmed.len();
        if hashes <= 6 {
            return Some(trimmed.trim_start());
        }
    }
    None
}

/// Convert inline markdown elements (bold, italic, code) in already HTML-escaped text
fn convert_inline(text: &str) -> String {
    // Process inline code first to protect content from further conversion
    let mut result = String::new();
    let mut remaining = text;

    // Split by inline code spans: `...`
    loop {
        if let Some(start) = remaining.find('`') {
            let after_start = &remaining[start + 1..];
            if let Some(end) = after_start.find('`') {
                // Found a complete inline code span
                let before = &remaining[..start];
                let code_content = &after_start[..end];
                result.push_str(&convert_bold_italic(before));
                result.push_str(&format!("<code>{}</code>", code_content));
                remaining = &after_start[end + 1..];
                continue;
            }
        }
        // No more inline code spans
        result.push_str(&convert_bold_italic(remaining));
        break;
    }

    result
}

/// Convert bold (**...**) and italic (*...*) in text
fn convert_bold_italic(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Bold: **...**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_closing_marker(&chars, i + 2, &['*', '*']) {
                let inner: String = chars[i + 2..end].iter().collect();
                result.push_str(&format!("<b>{}</b>", inner));
                i = end + 2;
                continue;
            }
        }
        // Italic: *...*
        if chars[i] == '*' {
            if let Some(end) = find_closing_single(&chars, i + 1, '*') {
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&format!("<i>{}</i>", inner));
                i = end + 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Find closing double marker (e.g., **) starting from pos
fn find_closing_marker(chars: &[char], start: usize, marker: &[char; 2]) -> Option<usize> {
    let len = chars.len();
    let mut i = start;
    while i + 1 < len {
        if chars[i] == marker[0] && chars[i + 1] == marker[1] {
            // Don't match empty content
            if i > start {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Find closing single marker (e.g., *) starting from pos
fn find_closing_single(chars: &[char], start: usize, marker: char) -> Option<usize> {
    let len = chars.len();
    let mut i = start;
    while i < len {
        if chars[i] == marker {
            // Don't match empty or double marker
            if i > start && (i + 1 >= len || chars[i + 1] != marker) {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Check if a Bash tool call is an internal cokacdir command.
/// Scans whitespace-delimited tokens for one whose basename matches the cokacdir binary.
/// Handles quoted paths, shell wrappers (bash -lc "..."), chained commands (cd && ...), etc.
/// NOTE: Returns bool (not subcommand name), so console logs show "cokacdir: ..." without
/// the specific --flag. format_cokacdir_result() auto-detects subcommand from JSON fields instead.
fn detect_cokacdir_command(name: &str, input: &str) -> bool {
    if name != "Bash" {
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input) else {
        return false;
    };
    let cmd = v.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let expected = crate::bin_path().rsplit(['/', '\\']).next().unwrap_or("");
    if expected.is_empty() {
        return false;
    }
    // Strip surrounding quotes from each token, then compare basename
    cmd.split_whitespace().any(|tok| {
        let unquoted = tok.trim_matches(|c| c == '"' || c == '\'');
        let basename = unquoted.rsplit(['/', '\\']).next().unwrap_or("");
        basename == expected
    })
}

/// Check if a Bash tool call contains --read_chat_log (result should be suppressed from display).
fn detect_chat_log_read(name: &str, input: &str) -> bool {
    if name != "Bash" {
        return false;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input) else {
        return false;
    };
    let cmd = v.get("command").and_then(|v| v.as_str()).unwrap_or("");
    cmd.contains("--read_chat_log")
}

/// Read the most recent .result file from schedule dir and delete it
fn read_latest_cron_result() -> Option<String> {
    let dir = schedule_dir()?;
    let mut results: Vec<_> = fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "result")
                .unwrap_or(false)
        })
        .collect();
    results.sort_by_key(|e| {
        std::cmp::Reverse(
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });
    let entry = results.first()?;
    let content = fs::read_to_string(entry.path()).ok()?;
    let _ = fs::remove_file(entry.path());
    Some(content)
}

/// Format a cokacdir command's JSON result into a human-readable message.
/// Auto-detects the subcommand from JSON result fields.
/// NOTE: Empty content triggers .result file read (for --cron async results).
/// Currently only --cron produces empty output, so this is safe.
/// If a new subcommand also returns empty output in the future,
/// it would incorrectly read a stale cron .result file.
fn format_cokacdir_result(content: &str) -> String {
    // Try to parse as JSON; if empty, try reading from .result file (for --cron)
    let effective_content = if content.trim().is_empty() {
        read_latest_cron_result().unwrap_or_default()
    } else {
        content.to_string()
    };
    let v: serde_json::Value = match serde_json::from_str(effective_content.trim()) {
        Ok(v) => v,
        Err(_) => {
            // Fallback: some backends (e.g. Gemini CLI) wrap the JSON output with
            // extra text like "Output: {...}\nProcess Group PGID: ...".
            // Try to extract the JSON object by trimming to first '{' and last '}'.
            let trimmed = effective_content.trim();
            let extracted = match (trimmed.find('{'), trimmed.rfind('}')) {
                (Some(start), Some(end)) if start < end => &trimmed[start..=end],
                _ => return String::new(),
            };
            match serde_json::from_str(extracted) {
                Ok(v) => v,
                Err(_) => return String::new(),
            }
        }
    };

    let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");

    if status == "error" {
        let msg = v
            .get("message")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown error");
        return format!("Error: {}", msg);
    }

    // Auto-detect subcommand from result JSON fields
    if v.get("time").is_some() {
        // --currenttime → {"status":"ok","time":"..."}
        let time = v["time"].as_str().unwrap_or("?");
        format!("🕐 {}", time)
    } else if v.get("schedules").is_some() {
        // --cron-list → {"status":"ok","schedules":[...]}
        let schedules = v["schedules"].as_array();
        match schedules {
            Some(arr) if arr.is_empty() => "📋 No schedules found.".to_string(),
            Some(arr) => {
                let mut lines = vec![format!("📋 {} schedule(s)", arr.len())];
                for (i, s) in arr.iter().enumerate() {
                    let id = s.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                    let schedule = s.get("schedule").and_then(|v| v.as_str()).unwrap_or("");
                    let prompt = s.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
                    let schedule_type = s
                        .get("schedule_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let once = s.get("once").and_then(|b| b.as_bool()).unwrap_or(false);
                    let kind = match schedule_type {
                        "absolute" => "1회",
                        "cron" if once => "1회 cron",
                        "cron" => "반복",
                        _ => {
                            if schedule.split_whitespace().count() == 5 {
                                "반복"
                            } else {
                                "1회"
                            }
                        }
                    };
                    let prompt_preview = if prompt.chars().count() > 40 {
                        format!("{}...", prompt.chars().take(40).collect::<String>())
                    } else {
                        prompt.to_string()
                    };
                    lines.push(format!(
                        "\n{}. [{}] {}\n   🕐 `{}`\n   🔖 {}",
                        i + 1,
                        kind,
                        prompt_preview,
                        schedule,
                        id
                    ));
                }
                lines.join("\n")
            }
            None => content.to_string(),
        }
    } else if v.get("path").is_some() {
        // --sendfile → {"status":"ok","path":"..."}
        let path = v["path"].as_str().unwrap_or("?");
        format!("📎 {}", path)
    } else if v.get("prompt").is_some() {
        // --cron (register) → {"status":"ok","id":"...","prompt":"...","schedule":"..."}
        let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("?");
        let prompt = v["prompt"].as_str().unwrap_or("");
        let schedule = v.get("schedule").and_then(|s| s.as_str()).unwrap_or("");
        let schedule_type = v
            .get("schedule_type")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let once = v.get("once").and_then(|b| b.as_bool()).unwrap_or(false);
        let kind = match schedule_type {
            "absolute" => "1회",
            "cron" if once => "1회 cron",
            "cron" => "반복",
            _ => {
                if schedule.split_whitespace().count() == 5 {
                    "반복"
                } else {
                    "1회"
                }
            }
        };
        format!(
            "✅ Scheduled [{}]\n🔖 {}\n📝 {}\n🕐 `{}`",
            kind, id, prompt, schedule
        )
    } else if v.get("schedule").is_some() {
        // --cron-update → {"status":"ok","id":"...","schedule":"..."}
        let id = v.get("id").and_then(|s| s.as_str()).unwrap_or("?");
        let schedule = v["schedule"].as_str().unwrap_or("");
        format!("✅ Updated\n🕐 `{}`\n🔖 {}", schedule, id)
    } else if v.get("id").is_some() {
        let id = v["id"].as_str().unwrap_or("?");
        if id.starts_with("msg_") {
            // --message result: not useful to show to user
            String::new()
        } else {
            // --cron-remove → {"status":"ok","id":"..."}
            format!("✅ Removed\n🔖 {}", id)
        }
    } else {
        content.to_string()
    }
}

/// Extract the command string (with optional description) from a Bash tool input JSON
fn format_bash_command(input: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input) else {
        return input.to_string();
    };
    let desc = v.get("description").and_then(|v| v.as_str()).unwrap_or("");
    let cmd = v.get("command").and_then(|v| v.as_str()).unwrap_or("");
    if !desc.is_empty() {
        format!("{}\n{}", desc, cmd)
    } else {
        cmd.to_string()
    }
}

/// Format tool input JSON into a human-readable summary
fn format_tool_input(name: &str, input: &str) -> String {
    // FileChange input is the full Codex item JSON — extract path summary for display
    if name == "FileChange" {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(input) {
            if let Some(changes) = v.get("changes").and_then(|v| v.as_array()) {
                let summary: Vec<String> = changes
                    .iter()
                    .map(|c| {
                        let path = c.get("path").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let kind = c.get("kind").and_then(|v| v.as_str()).unwrap_or("update");
                        format!("{}: {}", kind, path)
                    })
                    .collect();
                return format!("\u{1F4DD} {}", summary.join(", "));
            }
        }
        return format!("\u{1F4DD} {}", input);
    }

    let Ok(v) = serde_json::from_str::<serde_json::Value>(input) else {
        // Non-JSON input — Codex tools produce human-readable strings directly
        return match name {
            "WebSearch" => {
                if input.is_empty() {
                    "Search".to_string()
                } else {
                    format!("Search: {}", input)
                }
            }
            n if n.starts_with("Collab:") => {
                let tool = n.strip_prefix("Collab:").unwrap_or(n);
                if input.is_empty() {
                    format!("Agent: {}", tool)
                } else {
                    format!("Agent {}: {}", tool, input)
                }
            }
            _ => format!("{} {}", name, input),
        };
    };

    match name {
        "Bash" => {
            let desc = v.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let cmd = v.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if !desc.is_empty() {
                format!("{}: `{}`", desc, cmd)
            } else {
                format!("`{}`", cmd)
            }
        }
        "Read" => {
            let fp = v.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            format!("Read {}", fp)
        }
        "Write" => {
            let fp = v.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            let content = v.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let lines = content.lines().count();
            if lines > 0 {
                format!("Write {} ({} lines)", fp, lines)
            } else {
                format!("Write {}", fp)
            }
        }
        "Edit" => {
            let fp = v.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
            if !fp.is_empty() {
                let replace_all = v
                    .get("replace_all")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if replace_all {
                    format!("Edit {} (replace all)", fp)
                } else {
                    format!("Edit {}", fp)
                }
            } else {
                "Edit".to_string()
            }
        }
        "Glob" => {
            let pattern = v.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = v.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if !path.is_empty() {
                format!("Glob {} in {}", pattern, path)
            } else {
                format!("Glob {}", pattern)
            }
        }
        "Grep" => {
            let pattern = v.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = v.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let output_mode = v.get("output_mode").and_then(|v| v.as_str()).unwrap_or("");
            if !path.is_empty() {
                if !output_mode.is_empty() {
                    format!("Grep \"{}\" in {} ({})", pattern, path, output_mode)
                } else {
                    format!("Grep \"{}\" in {}", pattern, path)
                }
            } else {
                format!("Grep \"{}\"", pattern)
            }
        }
        "NotebookEdit" => {
            let nb_path = v
                .get("notebook_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let cell_id = v.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
            if !cell_id.is_empty() {
                format!("Notebook {} ({})", nb_path, cell_id)
            } else {
                format!("Notebook {}", nb_path)
            }
        }
        "WebSearch" => {
            let query = v.get("query").and_then(|v| v.as_str()).unwrap_or("");
            format!("Search: {}", query)
        }
        "WebFetch" => {
            let url = v.get("url").and_then(|v| v.as_str()).unwrap_or("");
            format!("Fetch {}", url)
        }
        "Task" => {
            let desc = v.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let subagent_type = v
                .get("subagent_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !subagent_type.is_empty() {
                format!("Task [{}]: {}", subagent_type, desc)
            } else {
                format!("Task: {}", desc)
            }
        }
        "TaskOutput" => {
            let task_id = v.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
            format!("Get task output: {}", task_id)
        }
        "TaskStop" => {
            let task_id = v.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
            format!("Stop task: {}", task_id)
        }
        "TodoWrite" => {
            if let Some(todos) = v.get("todos").and_then(|v| v.as_array()) {
                let pending = todos
                    .iter()
                    .filter(|t| t.get("status").and_then(|s| s.as_str()) == Some("pending"))
                    .count();
                let in_progress = todos
                    .iter()
                    .filter(|t| t.get("status").and_then(|s| s.as_str()) == Some("in_progress"))
                    .count();
                let completed = todos
                    .iter()
                    .filter(|t| t.get("status").and_then(|s| s.as_str()) == Some("completed"))
                    .count();
                format!(
                    "Todo: {} pending, {} in progress, {} completed",
                    pending, in_progress, completed
                )
            } else {
                "Update todos".to_string()
            }
        }
        "Skill" => {
            let skill = v.get("skill").and_then(|v| v.as_str()).unwrap_or("");
            format!("Skill: {}", skill)
        }
        "AskUserQuestion" => {
            if let Some(questions) = v.get("questions").and_then(|v| v.as_array()) {
                if let Some(q) = questions.first() {
                    let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("");
                    question.to_string()
                } else {
                    "Ask user question".to_string()
                }
            } else {
                "Ask user question".to_string()
            }
        }
        "ExitPlanMode" => "Exit plan mode".to_string(),
        "EnterPlanMode" => "Enter plan mode".to_string(),
        "TaskCreate" => {
            let subject = v.get("subject").and_then(|v| v.as_str()).unwrap_or("");
            format!("Create task: {}", subject)
        }
        "TaskUpdate" => {
            let task_id = v.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
            let status = v.get("status").and_then(|v| v.as_str()).unwrap_or("");
            if !status.is_empty() {
                format!("Update task {}: {}", task_id, status)
            } else {
                format!("Update task {}", task_id)
            }
        }
        "TaskGet" => {
            let task_id = v.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
            format!("Get task: {}", task_id)
        }
        "TaskList" => "List tasks".to_string(),
        _ => {
            // Codex Collab:* tools — input is prompt text (or empty)
            if name.starts_with("Collab:") {
                let collab_tool = name.strip_prefix("Collab:").unwrap_or(name);
                if input.is_empty() {
                    format!("Agent: {}", collab_tool)
                } else {
                    format!("Agent {}: {}", collab_tool, input)
                }
            } else {
                format!("{} {}", name, input)
            }
        }
    }
}

// === Scheduler ===

/// Check if a schedule entry should trigger now
fn should_trigger(entry: &ScheduleEntry) -> bool {
    let now = chrono::Local::now();
    sched_debug(&format!(
        "[should_trigger] id={}, type={}, schedule={}, now={}, last_run={:?}",
        entry.id,
        entry.schedule_type,
        entry.schedule,
        now.format("%Y-%m-%d %H:%M:%S"),
        entry.last_run
    ));
    match entry.schedule_type.as_str() {
        "absolute" => {
            let Ok(schedule_time) =
                chrono::NaiveDateTime::parse_from_str(&entry.schedule, "%Y-%m-%d %H:%M:%S")
            else {
                sched_debug(&format!(
                    "[should_trigger] id={}, parse failed → false",
                    entry.id
                ));
                return false;
            };
            let schedule_dt = schedule_time.and_local_timezone(chrono::Local).single();
            let Some(schedule_dt) = schedule_dt else {
                sched_debug(&format!(
                    "[should_trigger] id={}, timezone conversion failed → false",
                    entry.id
                ));
                return false;
            };
            if now < schedule_dt {
                sched_debug(&format!(
                    "[should_trigger] id={}, not yet (now < schedule_dt) → false",
                    entry.id
                ));
                return false;
            }
            // Already ran?
            if let Some(ref last) = entry.last_run {
                if let Ok(last_dt) =
                    chrono::NaiveDateTime::parse_from_str(last, "%Y-%m-%d %H:%M:%S")
                {
                    if let Some(last_local) = last_dt.and_local_timezone(chrono::Local).single() {
                        if last_local >= schedule_dt {
                            sched_debug(&format!(
                                "[should_trigger] id={}, already ran (last={} >= sched={}) → false",
                                entry.id,
                                last_local.format("%H:%M:%S"),
                                schedule_dt.format("%H:%M:%S")
                            ));
                            return false;
                        }
                    }
                }
            }
            sched_debug(&format!(
                "[should_trigger] id={}, absolute ready → true",
                entry.id
            ));
            true
        }
        "cron" => {
            if !cron_matches(&entry.schedule, now) {
                sched_debug(&format!(
                    "[should_trigger] id={}, cron not matching → false",
                    entry.id
                ));
                return false;
            }
            // Check last_run to avoid duplicate triggers within the same minute
            if let Some(ref last) = entry.last_run {
                if let Ok(last_dt) =
                    chrono::NaiveDateTime::parse_from_str(last, "%Y-%m-%d %H:%M:%S")
                {
                    if let Some(last_local) = last_dt.and_local_timezone(chrono::Local).single() {
                        let now_min = now.format("%Y-%m-%d %H:%M").to_string();
                        let last_min = last_local.format("%Y-%m-%d %H:%M").to_string();
                        if now_min == last_min {
                            sched_debug(&format!(
                                "[should_trigger] id={}, already ran this minute ({}) → false",
                                entry.id, now_min
                            ));
                            return false;
                        }
                    }
                }
            }
            sched_debug(&format!(
                "[should_trigger] id={}, cron matched → true",
                entry.id
            ));
            true
        }
        _ => {
            sched_debug(&format!(
                "[should_trigger] id={}, unknown type={} → false",
                entry.id, entry.schedule_type
            ));
            false
        }
    }
}

/// Update schedule entry after a run: set last_run, delete if once
fn update_schedule_after_run(entry: &ScheduleEntry, new_context_summary: Option<String>) {
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    sched_debug(&format!(
        "[update_schedule_after_run] id={}, type={}, once={:?}, now={}, has_new_context={}",
        entry.id,
        entry.schedule_type,
        entry.once,
        now,
        new_context_summary.is_some()
    ));

    // 실행 중 사용자가 삭제한 경우 부활 방지
    let dir = match schedule_dir() {
        Some(d) => d,
        None => {
            sched_debug(&format!(
                "[update_schedule_after_run] id={}, no schedule dir → skip",
                entry.id
            ));
            return;
        }
    };
    let path = dir.join(format!("{}.json", entry.id));
    if !path.exists() {
        sched_debug(&format!(
            "[update_schedule_after_run] id={}, file already deleted → skip (no resurrection)",
            entry.id
        ));
        return; // 이미 삭제됨 - write하지 않음
    }

    // One-time schedules (absolute / cron --once) are already deleted before execution,
    // so this function only handles recurring cron updates.
    sched_debug(&format!(
        "[update_schedule_after_run] id={}, cron recurring → update last_run",
        entry.id
    ));
    let mut updated = entry.clone();
    updated.last_run = Some(now);
    if new_context_summary.is_some() {
        updated.context_summary = new_context_summary;
    }
    if let Err(e) = write_schedule_entry(&updated) {
        sched_debug(&format!(
            "[update_schedule_after_run] id={}, write failed: {}",
            entry.id, e
        ));
        eprintln!("[Schedule] Failed to update entry {}: {}", entry.id, e);
    } else {
        sched_debug(&format!(
            "[update_schedule_after_run] id={}, updated successfully",
            entry.id
        ));
    }
}

/// Execute a scheduled task — similar pattern to handle_text_message
async fn execute_schedule(
    bot: &Bot,
    chat_id: ChatId,
    entry: &ScheduleEntry,
    state: &SharedState,
    token: &str,
    prev_session: Option<ChatSession>,
    inline_mode: bool,
) {
    sched_debug(&format!("[execute_schedule] START id={}, chat_id={}, prompt={:?}, has_context={}, has_prev_session={}, inline_mode={}",
        entry.id, chat_id, truncate_str(&entry.prompt, 60), entry.context_summary.is_some(), prev_session.is_some(), inline_mode));

    // Acquire group chat lock (serializes processing across bots in the same group chat)
    let group_lock = acquire_group_chat_lock(chat_id.0).await;

    // Check if cancelled during lock wait
    let cancelled_during_wait = {
        let data = state.lock().await;
        data.cancel_tokens
            .get(&chat_id)
            .map(|ct| ct.cancelled.load(Ordering::Relaxed))
            .unwrap_or(false)
    };
    if cancelled_during_wait {
        sched_debug(&format!(
            "[execute_schedule] cancelled during lock wait, id={}",
            entry.id
        ));
        {
            let mut data = state.lock().await;
            if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                set.remove(&entry.id);
            }
            data.cancel_tokens.remove(&chat_id);
            if !inline_mode {
                if let Some(prev) = prev_session {
                    data.sessions.insert(chat_id, prev);
                } else {
                    data.sessions.remove(&chat_id);
                }
            }
        }
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=schedule_cancelled_during_lock",
            chat_id.0
        ));
        drop(group_lock);
        process_next_queued_message(bot, chat_id, state).await;
        return;
    }

    // Build prompt with context summary if available
    let user_prompt = entry.prompt.clone();
    let prompt = if !inline_mode {
        if let Some(ref summary) = entry.context_summary {
            sched_debug(&format!(
                "[execute_schedule] id={}, injecting context summary ({} chars)",
                entry.id,
                summary.len()
            ));
            format!(
                "[이전 작업 맥락]\n{}\n\n[작업 지시]\n{}",
                summary, user_prompt
            )
        } else {
            user_prompt.clone()
        }
    } else {
        user_prompt.clone()
    };
    let project_path = crate::utils::format::to_shell_path(&entry.current_path);
    let schedule_id = entry.id.clone();

    // Delete schedule files before execution for one-time schedules (absolute / cron --once)
    if entry.once.unwrap_or(false) || entry.schedule_type == "absolute" {
        sched_debug(&format!(
            "[execute_schedule] id={}, one-time → deleting schedule files before execution",
            schedule_id
        ));
        delete_schedule_entry(&schedule_id);
    }

    let ts = chrono::Local::now().format("%H:%M:%S");
    println!("  [{ts}] ⏰ Schedule Starting: {user_prompt}");

    let (workspace_path, resume_session_id) = if inline_mode {
        let data = state.lock().await;
        match data.sessions.get(&chat_id).and_then(|session| {
            session
                .current_path
                .clone()
                .map(|path| (path, session.session_id.clone()))
        }) {
            Some((path, session_id)) => (path, session_id),
            None => {
                sched_debug(&format!(
                    "[execute_schedule] id={}, inline mode but no current session/path",
                    schedule_id
                ));
                {
                    let mut data = state.lock().await;
                    if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                        set.remove(&schedule_id);
                    }
                    data.cancel_tokens.remove(&chat_id);
                }
                msg_debug(&format!(
                    "[queue:trigger] chat_id={}, source=schedule_inline_no_session",
                    chat_id.0
                ));
                drop(group_lock);
                process_next_queued_message(bot, chat_id, state).await;
                return;
            }
        }
    } else {
        let Some(home) = dirs::home_dir() else {
            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ⚠ [Schedule] Failed to get home directory");
            {
                let mut data = state.lock().await;
                if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                    set.remove(&schedule_id);
                }
                data.cancel_tokens.remove(&chat_id);
                if !inline_mode {
                    if let Some(prev) = prev_session {
                        data.sessions.insert(chat_id, prev);
                    } else {
                        data.sessions.remove(&chat_id);
                    }
                }
            }
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=schedule_no_home",
                chat_id.0
            ));
            drop(group_lock);
            process_next_queued_message(bot, chat_id, state).await;
            return;
        };
        let workspace_dir = home.join(".cokacdir").join("workspace").join(&schedule_id);
        sched_debug(&format!(
            "[execute_schedule] id={}, creating workspace: {}",
            schedule_id,
            workspace_dir.display()
        ));
        if let Err(e) = fs::create_dir_all(&workspace_dir) {
            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ⚠ [Schedule] Failed to create workspace: {e}");
            sched_debug(&format!(
                "[execute_schedule] id={}, workspace creation failed: {}, restoring session",
                schedule_id, e
            ));
            {
                let mut data = state.lock().await;
                if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                    set.remove(&schedule_id);
                }
                data.cancel_tokens.remove(&chat_id);
                if let Some(prev) = prev_session {
                    data.sessions.insert(chat_id, prev);
                } else {
                    data.sessions.remove(&chat_id);
                }
            }
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=schedule_workspace_error",
                chat_id.0
            ));
            drop(group_lock);
            process_next_queued_message(bot, chat_id, state).await;
            return;
        }
        (workspace_dir.display().to_string(), None)
    };

    // Get allowed tools, model, and provider options for this chat
    let (allowed_tools, model, sched_chrome_enabled, sched_effort, sched_fast_mode) = {
        let data = state.lock().await;
        let chrome = data
            .settings
            .use_chrome
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(false);
        let model = get_model(&data.settings, chat_id);
        let provider = detect_provider(model.as_deref());
        let effort = get_effort_for_provider(&data.settings, chat_id, provider);
        let fast_mode = is_codex_fast(&data.settings, chat_id);
        (
            get_allowed_tools(&data.settings, chat_id),
            model,
            chrome,
            effort,
            fast_mode,
        )
    };

    // Send placeholder (show only the user's original prompt, not the context summary)
    shared_rate_limit_wait(state, chat_id).await;
    let placeholder = match tg!(
        "send_message",
        bot.send_message(chat_id, format!("⏰ {user_prompt}")).await
    ) {
        Ok(msg) => msg,
        Err(e) => {
            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ⚠ [Schedule] Failed to send placeholder: {e}");
            // Clean up pending + cancel_token, restore session (workspace preserved)
            {
                let mut data = state.lock().await;
                if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                    set.remove(&schedule_id);
                }
                data.cancel_tokens.remove(&chat_id);
                if let Some(prev) = prev_session {
                    data.sessions.insert(chat_id, prev);
                } else {
                    data.sessions.remove(&chat_id);
                }
            }
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=schedule_placeholder_error",
                chat_id.0
            ));
            drop(group_lock); // release before queue processing to avoid deadlock
            process_next_queued_message(bot, chat_id, state).await;
            return;
        }
    };
    let placeholder_msg_id = placeholder.id;

    // Build disabled tools notice
    let default_tools: std::collections::HashSet<&str> =
        DEFAULT_ALLOWED_TOOLS.iter().copied().collect();
    let allowed_set: std::collections::HashSet<&str> =
        allowed_tools.iter().map(|s| s.as_str()).collect();
    let disabled: Vec<&&str> = default_tools
        .iter()
        .filter(|t| !allowed_set.contains(**t))
        .collect();
    let disabled_notice = if disabled.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = disabled.iter().map(|t| **t).collect();
        format!(
            "\n\nDISABLED TOOLS: The following tools have been disabled by the user: {}.\n\
             You MUST NOT attempt to use these tools. \
             If a user's request requires a disabled tool, do NOT proceed with the task. \
             Instead, clearly inform the user which tool is needed and that it is currently disabled. \
             Suggest they re-enable it with: /allowed +ToolName",
            names.join(", ")
        )
    };

    let bot_key = token_hash(token);
    let (sched_instruction, sched_context_count, sched_bot_username, sched_bot_display_name) = {
        let data = state.lock().await;
        let ctx = data
            .settings
            .context
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(12);
        (
            data.settings
                .instructions
                .get(&chat_id.0.to_string())
                .cloned(),
            ctx,
            data.bot_username.clone(),
            data.bot_display_name.clone(),
        )
    };
    let platform = capitalize_platform(detect_platform(token));
    let sched_role = {
        let base = if inline_mode {
            format!(
                "You are executing a scheduled task through {}.\n\
                 Project directory: {project_path}\n\
                 You are continuing in the chat's current working directory and session.\n\
                 Any files you want to deliver must be sent via the \"{}\" --sendfile command before the task ends.",
                platform,
                shell_bin_path()
            )
        } else {
            format!(
                "You are executing a scheduled task through {}.\n\
                 Project directory: {project_path}\n\
                 Your current working directory is a dedicated workspace for this schedule.\n\
                 This workspace will be preserved after execution. The user can continue work here via /start.\n\
                 To work with project files, use absolute paths to the project directory.\n\
                 Any files you want to deliver must be sent via the \"{}\" --sendfile command before the task ends.",
                platform,
                shell_bin_path()
            )
        };
        match &sched_instruction {
            Some(instr) => format!("{}\n\nUser's instruction for this chat:\n{}", base, instr),
            None => base,
        }
    };
    let system_prompt_owned = build_system_prompt(
        &sched_role,
        &crate::utils::format::to_shell_path(&workspace_path),
        chat_id.0,
        &bot_key,
        &disabled_notice,
        None, // scheduled tasks don't need to register further schedules with session context
        &sched_bot_username,
        &sched_bot_display_name,
        None, // scheduled tasks: no user message dedup
        false,
        sched_context_count,
        &platform,
    );

    // Retrieve pre-inserted cancel token (from scheduler_loop), or create a new one
    let cancel_token = {
        let mut data = state.lock().await;
        if let Some(existing) = data.cancel_tokens.get(&chat_id) {
            existing.clone()
        } else {
            let token = Arc::new(CancelToken::new());
            data.cancel_tokens.insert(chat_id, token.clone());
            token
        }
    };

    // Create channel for streaming
    let (tx, rx) = mpsc::channel();
    let cancel_token_clone = cancel_token.clone();
    let model_for_summary = model.clone();
    let resume_session_id_for_exec = resume_session_id.clone();

    // Run AI backend in a blocking thread (always new session — context is in the prompt)
    // Session persistence must be kept so users can resume via /SCHEDULE_ID
    let workspace_path_for_claude = workspace_path.clone();
    let model_clone_for_exec = model.clone();
    let bot_key_for_codex = bot_key.clone();
    tokio::task::spawn_blocking(move || {
        let provider = detect_provider(model_clone_for_exec.as_deref());
        sched_debug(&format!(
            "[execute_schedule:spawn_blocking] provider={}, model={:?}",
            provider, model_clone_for_exec
        ));
        let result = if provider == "opencode" {
            let opencode_model = model_clone_for_exec
                .as_deref()
                .and_then(opencode::strip_opencode_prefix);
            sched_debug(&format!("[execute_schedule] → opencode::execute, opencode_model={:?}, session_id=None, workspace={}, prompt_len={}, system_prompt_len={}",
                opencode_model, workspace_path_for_claude, prompt.len(), system_prompt_owned.len()));
            opencode::execute_command_streaming(
                &prompt,
                resume_session_id_for_exec.as_deref(),
                &workspace_path_for_claude,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                opencode_model,
                false,
            )
        } else if provider == "gemini" {
            let gemini_model = model_clone_for_exec
                .as_deref()
                .and_then(gemini::strip_gemini_prefix);
            sched_debug(&format!("[execute_schedule] → gemini::execute, gemini_model={:?}, session_id=None, workspace={}, prompt_len={}, system_prompt_len={}",
                gemini_model, workspace_path_for_claude, prompt.len(), system_prompt_owned.len()));
            gemini::execute_command_streaming(
                &prompt,
                resume_session_id_for_exec.as_deref(),
                &workspace_path_for_claude,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                gemini_model,
                false,
            )
        } else if provider == "codex" {
            let codex_model = model_clone_for_exec
                .as_deref()
                .and_then(codex::strip_codex_prefix);
            let codex_system_prompt =
                format!("{}{}", system_prompt_owned, codex_extra_instructions());
            let codex_auto_send = codex::CodexAutoSendCtx {
                cokacdir_bin: crate::bin_path().to_string(),
                chat_id: chat_id.0,
                bot_key: bot_key_for_codex,
            };
            codex::execute_command_streaming(
                &prompt,
                resume_session_id_for_exec.as_deref(),
                &workspace_path_for_claude,
                tx.clone(),
                Some(&codex_system_prompt),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                codex_model,
                false,
                Some(&codex_auto_send),
                sched_effort.as_deref(),
                sched_fast_mode,
            )
        } else {
            let claude_model = model_clone_for_exec
                .as_deref()
                .and_then(claude::strip_claude_prefix);
            claude::execute_command_streaming(
                &prompt,
                resume_session_id_for_exec.as_deref(),
                &workspace_path_for_claude,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                claude_model,
                false,
                sched_chrome_enabled,
                sched_effort.as_deref(),
            )
        };
        if let Err(e) = result {
            let _ = tx.send(StreamMessage::Error {
                message: e,
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
            });
        }
    });

    // Polling loop
    let bot_owned = bot.clone();
    let state_owned = state.clone();
    let entry_clone = entry.clone();
    let workspace_path_owned = workspace_path.clone();
    let provider_str: &'static str = detect_provider(model.as_deref());
    // Captured for schedule_history append at the end of this run.
    // Wall-clock duration is measured from this point (just after placeholder send +
    // pre-execution setup) to the polling loop's completion — close to the user-visible
    // "task is running" window.
    let history_start = std::time::Instant::now();
    let history_bot_key = bot_key.clone();
    tokio::spawn(async move {
        let _group_lock = group_lock; // hold group chat lock until task ends
        const SPINNER: &[&str] = &[
            "🕐 P",
            "🕑 Pr",
            "🕒 Pro",
            "🕓 Proc",
            "🕔 Proce",
            "🕕 Proces",
            "🕖 Process",
            "🕗 Processi",
            "🕘 Processin",
            "🕙 Processing",
            "🕚 Processing.",
            "🕛 Processing..",
        ];
        let mut full_response = String::new();
        let mut raw_entries: Vec<RawPayloadEntry> = Vec::new();
        let mut last_edit_text = String::new();
        let mut done = false;
        let mut cancelled = false;
        let mut had_error = false;
        let mut spin_idx: usize = 0;
        let mut pending_cokacdir = false;
        let mut suppress_tool_display = false;
        let mut last_tool_name: String = String::new();
        let mut exec_session_id: Option<String> = None;
        let mut placeholder_msg_id = placeholder_msg_id;
        let mut last_confirmed_len: usize = 0;

        let (polling_time_ms, silent_mode) = {
            let data = state_owned.lock().await;
            (data.polling_time_ms, is_silent(&data.settings, chat_id))
        };

        let mut first_poll = true;
        let mut queue_done = false;
        while !done || !queue_done {
            if cancel_token.cancelled.load(Ordering::Relaxed) {
                if !done {
                    cancelled = true;
                }
                break;
            }

            if !first_poll {
                tokio::time::sleep(tokio::time::Duration::from_millis(polling_time_ms)).await;

                if cancel_token.cancelled.load(Ordering::Relaxed) {
                    if !done {
                        cancelled = true;
                    }
                    break;
                }
            }
            first_poll = false;

            // Drain messages
            loop {
                match rx.try_recv() {
                    Ok(msg) => {
                        match msg {
                            StreamMessage::Init { session_id } => {
                                exec_session_id = Some(session_id);
                            }
                            StreamMessage::Text { content } => {
                                sched_debug(&format!(
                                    "[sched] Text: {} chars, preview={:?}",
                                    content.len(),
                                    truncate_str(&content, 80)
                                ));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "Text".into(),
                                    content: content.clone(),
                                });
                                let _fr_before = full_response.len();
                                full_response.push_str(&content);
                                sched_debug(&format!("[fr_trace][{}] +Text: added={}, preview={:?}, total={} (was {})",
                                    chat_id.0, content.len(), truncate_str(&content, 200), full_response.len(), _fr_before));
                            }
                            StreamMessage::ToolUse { name, input } => {
                                pending_cokacdir = detect_cokacdir_command(&name, &input);
                                suppress_tool_display = detect_chat_log_read(&name, &input);
                                last_tool_name = name.clone();
                                let summary = format_tool_input(&name, &input);
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚙ [Schedule] {name}: {summary}");
                                sched_debug(&format!("[schedule_polling] ToolUse: name={}, input_preview={:?}, pending_cokacdir={}, silent_mode={}, response_len={}, ends_with_nl={}",
                                    name, truncate_str(&input, 200), pending_cokacdir, silent_mode, full_response.len(), full_response.ends_with('\n')));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolUse".into(),
                                    content: format!("{}: {}", name, input),
                                });
                                if !pending_cokacdir && !silent_mode {
                                    let _fr_before = full_response.len();
                                    if name == "Bash" {
                                        full_response.push_str(&format!(
                                            "\n\n```\n{}\n```\n",
                                            format_bash_command(&input)
                                        ));
                                        sched_debug(&format!("[fr_trace][{}] +ToolUse/Bash: added={}, cmd={:?}, total={} (was {})",
                                            chat_id.0, full_response.len() - _fr_before, truncate_str(&input, 100), full_response.len(), _fr_before));
                                    } else {
                                        full_response.push_str(&format!("\n\n⚙️ {}\n", summary));
                                        sched_debug(&format!("[fr_trace][{}] +ToolUse/{}: added={}, summary={:?}, total={} (was {})",
                                            chat_id.0, name, full_response.len() - _fr_before, truncate_str(&summary, 100), full_response.len(), _fr_before));
                                    }
                                } else if !pending_cokacdir
                                    && silent_mode
                                    && !full_response.is_empty()
                                    && !full_response.ends_with('\n')
                                {
                                    sched_debug(&format!("[schedule_polling] silent mode: inserting \\n\\n after tool_use={}", name));
                                    full_response.push_str("\n\n");
                                    sched_debug(&format!(
                                        "[fr_trace][{}] +ToolUse/silent_nl: added=2, total={}",
                                        chat_id.0,
                                        full_response.len()
                                    ));
                                } else if silent_mode {
                                    sched_debug(&format!("[schedule_polling] silent mode: skipped \\n\\n (pending_cokacdir={}, empty={}, ends_nl={})",
                                        pending_cokacdir, full_response.is_empty(), full_response.ends_with('\n')));
                                }
                            }
                            StreamMessage::ToolResult { content, is_error } => {
                                sched_debug(&format!("[schedule_polling] ToolResult: is_error={}, content_len={}, pending_cokacdir={}, last_tool={}", is_error, content.len(), pending_cokacdir, last_tool_name));
                                if is_error {
                                    sched_debug(&format!("[schedule_polling] ToolResult ERROR: last_tool={}, content_preview={:?}", last_tool_name, truncate_str(&content, 300)));
                                }
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolResult".into(),
                                    content: format!("is_error={}, content={}", is_error, content),
                                });
                                let _fr_before = full_response.len();
                                if std::mem::take(&mut pending_cokacdir) {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    if std::mem::take(&mut suppress_tool_display) {
                                        println!("  [{ts}]   ↩ [Schedule] cokacdir (chat_log, suppressed)");
                                        sched_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir_suppressed: added=0, total={}", chat_id.0, full_response.len()));
                                    } else {
                                        println!("  [{ts}]   ↩ [Schedule] cokacdir: {content}");
                                        let formatted = format_cokacdir_result(&content);
                                        if !formatted.is_empty() {
                                            full_response.push_str(&format!("\n{}\n", formatted));
                                            sched_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: added={}, preview={:?}, total={} (was {})",
                                                chat_id.0, full_response.len() - _fr_before, truncate_str(&formatted, 200), full_response.len(), _fr_before));
                                        } else {
                                            sched_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: formatted_empty, added=0, total={}", chat_id.0, full_response.len()));
                                        }
                                    }
                                } else if is_error && !silent_mode {
                                    let truncated = truncate_str(&content, 500);
                                    if truncated.contains('\n') {
                                        full_response
                                            .push_str(&format!("\n```\n{}\n```\n", truncated));
                                    } else {
                                        full_response.push_str(&format!("\n`{}`\n\n", truncated));
                                    }
                                    sched_debug(&format!("[fr_trace][{}] +ToolResult/error({}): added={}, preview={:?}, total={} (was {})",
                                        chat_id.0, last_tool_name, full_response.len() - _fr_before, truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                } else if !silent_mode {
                                    if last_tool_name == "Read" {
                                        full_response.push_str(&format!(
                                            "\n✅ `{} bytes`\n\n",
                                            content.len()
                                        ));
                                        sched_debug(&format!("[fr_trace][{}] +ToolResult/Read: added={}, content_bytes={}, total={} (was {})",
                                            chat_id.0, full_response.len() - _fr_before, content.len(), full_response.len(), _fr_before));
                                    } else if !content.is_empty() {
                                        let truncated = truncate_str(&content, 300);
                                        if truncated.contains('\n') {
                                            full_response
                                                .push_str(&format!("\n```\n{}\n```\n", truncated));
                                        } else {
                                            full_response
                                                .push_str(&format!("\n✅ `{}`\n\n", truncated));
                                        }
                                        sched_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): added={}, raw_content_len={}, preview={:?}, total={} (was {})",
                                            chat_id.0, last_tool_name, full_response.len() - _fr_before, content.len(), truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                    } else {
                                        sched_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): content_empty, added=0, total={}", chat_id.0, last_tool_name, full_response.len()));
                                    }
                                } else {
                                    sched_debug(&format!("[fr_trace][{}] +ToolResult/silent: skipped, last_tool={}, content_len={}, total={}", chat_id.0, last_tool_name, content.len(), full_response.len()));
                                }
                            }
                            StreamMessage::TaskNotification { summary, .. } => {
                                if !summary.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "TaskNotification".into(),
                                        content: summary.clone(),
                                    });
                                    let _fr_before = full_response.len();
                                    full_response.push_str(&format!("\n[Task: {}]\n", summary));
                                    sched_debug(&format!("[fr_trace][{}] +TaskNotification: added={}, summary={:?}, total={} (was {})",
                                        chat_id.0, full_response.len() - _fr_before, truncate_str(&summary, 200), full_response.len(), _fr_before));
                                }
                            }
                            StreamMessage::Done { result, session_id } => {
                                sched_debug(&format!(
                                    "[sched] Done: result_len={}, full_response_len={}",
                                    result.len(),
                                    full_response.len()
                                ));
                                if !result.is_empty() && full_response.is_empty() {
                                    sched_debug(&format!(
                                        "[sched] Done: fallback full_response = result ({})",
                                        result.len()
                                    ));
                                    full_response = result.clone();
                                    sched_debug(&format!("[fr_trace][{}] +Done/fallback: set={}, preview={:?}, total={}",
                                        chat_id.0, result.len(), truncate_str(&full_response, 200), full_response.len()));
                                } else if !result.is_empty() {
                                    sched_debug(&format!("[fr_trace][{}] Done/discarded: result_len={} discarded (full_response already has {})",
                                        chat_id.0, result.len(), full_response.len()));
                                }
                                if !result.is_empty() && raw_entries.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Text".into(),
                                        content: result,
                                    });
                                }
                                if let Some(sid) = session_id {
                                    exec_session_id = Some(sid);
                                }
                                sched_debug(&format!(
                                    "[fr_trace][{}] =DONE: final_total={}",
                                    chat_id.0,
                                    full_response.len()
                                ));
                                done = true;
                            }
                            StreamMessage::Error {
                                message,
                                stdout,
                                stderr,
                                exit_code,
                            } => {
                                let stdout_display = display_stream_pipe_payload(stdout);
                                let stderr_display = display_stream_pipe_payload(stderr);
                                let code_display = display_stream_exit_code(exit_code);
                                if should_preserve_completed_stream_response(done, &full_response) {
                                    sched_debug(&format!(
                                        "[polling] late error suppressed after terminal response: message={}, response_len={}",
                                        message,
                                        full_response.len()
                                    ));
                                } else {
                                    full_response = format_stream_error_response(
                                        &message,
                                        &code_display,
                                        &stdout_display,
                                        &stderr_display,
                                    );
                                    sched_debug(&format!("[fr_trace][{}] +Error: set={}, stdout_len={}, stderr_len={}, total={}",
                                        chat_id.0, full_response.len(), stdout_display.len(), stderr_display.len(), full_response.len()));
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Error".into(),
                                        content: format!(
                                            "exit_code={}, message={}, stdout={}, stderr={}",
                                            code_display, message, stdout_display, stderr_display
                                        ),
                                    });
                                    had_error = true;
                                }
                                done = true;
                            }
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        if !done {
                            had_error = true;
                        }
                        done = true;
                        break;
                    }
                }
            }

            // Update placeholder with progress
            if !done {
                // ── Rolling placeholder pattern (unified for all chats) ──
                let threshold = file_attach_threshold();
                let delta_end =
                    floor_char_boundary(&full_response, full_response.len().min(threshold));
                if delta_end > last_confirmed_len {
                    let delta = &full_response[last_confirmed_len..delta_end];
                    let normalized_delta = normalize_empty_lines(delta);
                    let html_delta = markdown_to_telegram_html(&normalized_delta);
                    if html_delta.trim().is_empty() {
                        msg_debug(&format!("[rolling_ph/sched] SKIP empty delta: placeholder_msg_id={}, delta_bytes={}, confirmed={}→{}",
                            placeholder_msg_id, delta.len(), last_confirmed_len, delta_end));
                        last_confirmed_len = delta_end;
                    } else {
                        msg_debug(&format!("[rolling_ph/sched] EDIT delta: placeholder_msg_id={}, delta_len={}, html_len={}, confirmed={}→{}",
                            placeholder_msg_id, normalized_delta.len(), html_delta.len(), last_confirmed_len, delta_end));
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        if html_delta.len() <= TELEGRAM_MSG_LIMIT {
                            let _ = tg!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &html_delta)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            );
                        } else {
                            if send_long_message(
                                &bot_owned,
                                chat_id,
                                &html_delta,
                                Some(ParseMode::Html),
                                &state_owned,
                            )
                            .await
                            .is_ok()
                            {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg!(
                                    "delete_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                );
                            } else {
                                let truncated_delta =
                                    truncate_str(&normalized_delta, TELEGRAM_MSG_LIMIT);
                                let _ = tg!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(
                                            chat_id,
                                            placeholder_msg_id,
                                            &truncated_delta
                                        )
                                        .await
                                );
                            }
                        }
                        last_confirmed_len = delta_end;
                        let old_ph_id = placeholder_msg_id;
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        match tg!(
                            "send_message",
                            &state_owned,
                            chat_id,
                            bot_owned.send_message(chat_id, "...").await
                        ) {
                            Ok(new_ph) => {
                                placeholder_msg_id = new_ph.id;
                                msg_debug(&format!("[rolling_ph/sched] NEW placeholder: old_msg_id={}, new_msg_id={}", old_ph_id, placeholder_msg_id));
                            }
                            Err(e) => {
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚠ new placeholder failed (schedule): {e}");
                                msg_debug(&format!("[rolling_ph/sched] NEW placeholder FAILED: keeping msg_id={}, err={}", placeholder_msg_id, e));
                            }
                        }
                        last_edit_text.clear();
                        spin_idx = 0;
                    }
                } else {
                    let indicator = SPINNER[spin_idx % SPINNER.len()];
                    spin_idx += 1;
                    let display_text = indicator.to_string();
                    if display_text != last_edit_text {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let html_text = markdown_to_telegram_html(&display_text);
                        let _ = tg!(
                            "edit_message",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .edit_message_text(chat_id, placeholder_msg_id, &html_text)
                                .parse_mode(ParseMode::Html)
                                .await
                        );
                        last_edit_text = display_text;
                    } else {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_chat_action",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .send_chat_action(chat_id, teloxide::types::ChatAction::Typing)
                                .await
                        );
                    }
                }
            }

            // Queue processing
            let queued = process_upload_queue(&bot_owned, chat_id, &state_owned).await;
            if done {
                queue_done = !queued;
            }
        }

        // Final response
        sched_debug(&format!(
            "[execute_schedule] id={}, polling done: cancelled={}, had_error={}, response_len={}",
            schedule_id,
            cancelled,
            had_error,
            full_response.len()
        ));
        let continue_hint = if inline_mode {
            String::new()
        } else {
            format!(
                "\n\nUse /{} to continue this schedule session.",
                schedule_id
            )
        };
        if cancelled {
            sched_debug(&format!(
                "[execute_schedule] id={}, cancelled — killing child process",
                schedule_id
            ));
            cancel_token_now(&cancel_token);

            shared_rate_limit_wait(&state_owned, chat_id).await;
            // ── Show remaining delta + stopped (unified rolling placeholder) ──
            if full_response.len() < last_confirmed_len
                || !full_response.is_char_boundary(last_confirmed_len)
            {
                last_confirmed_len = 0;
            }
            let remaining = &full_response[last_confirmed_len..];
            if should_attach_response_as_file(full_response.len(), provider_str) {
                let notice = format!("📄 Response attached as file [Stopped]{}", continue_hint);
                let _ = tg!(
                    "edit_message",
                    &state_owned,
                    chat_id,
                    bot_owned
                        .edit_message_text(chat_id, placeholder_msg_id, &notice)
                        .await
                );
                let stopped_content =
                    format!("{}\n\n[Stopped]", normalize_empty_lines(&full_response));
                send_response_as_file(
                    &bot_owned,
                    chat_id,
                    &stopped_content,
                    &state_owned,
                    "schedule",
                )
                .await;
            } else {
                let display_stopped = if remaining.trim().is_empty() {
                    format!("⛔ Stopped{}", continue_hint)
                } else {
                    let normalized = normalize_empty_lines(remaining);
                    format!("{}\n\n⛔ Stopped{}", normalized, continue_hint)
                };
                let _ = tg!(
                    "edit_message",
                    &state_owned,
                    chat_id,
                    bot_owned
                        .edit_message_text(chat_id, placeholder_msg_id, &display_stopped)
                        .await
                );
            }

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ■ [Schedule] Stopped");
        } else {
            if full_response.is_empty() {
                full_response = "(No response)".to_string();
                sched_debug(&format!(
                    "[fr_trace][{}] =NoResponse: set to '(No response)'",
                    chat_id.0
                ));
            }

            shared_rate_limit_wait(&state_owned, chat_id).await;

            // ── Send only remaining delta (unified rolling placeholder) ──
            if full_response.len() < last_confirmed_len
                || !full_response.is_char_boundary(last_confirmed_len)
            {
                last_confirmed_len = 0;
            }
            let remaining = &full_response[last_confirmed_len..];
            msg_debug(&format!("[rolling_ph/sched] FINAL: placeholder_msg_id={}, confirmed={}, total={}, remaining_len={}",
                placeholder_msg_id, last_confirmed_len, full_response.len(), remaining.trim().len()));
            if remaining.trim().is_empty() {
                msg_debug(&format!(
                    "[rolling_ph/sched] FINAL DELETE placeholder: msg_id={}",
                    placeholder_msg_id
                ));
                let _ = tg!(
                    "delete_message",
                    &state_owned,
                    chat_id,
                    bot_owned.delete_message(chat_id, placeholder_msg_id).await
                );
            } else if should_attach_response_as_file(full_response.len(), provider_str) {
                msg_debug(&format!(
                    "[rolling_ph/sched] FINAL FILE ATTACH: total={}",
                    full_response.len()
                ));
                let notice = format!("📄 Response attached as file{}", continue_hint);
                let _ = tg!(
                    "edit_message",
                    &state_owned,
                    chat_id,
                    bot_owned
                        .edit_message_text(chat_id, placeholder_msg_id, &notice)
                        .await
                );
                let final_text =
                    format!("{}{}", normalize_empty_lines(&full_response), continue_hint);
                send_response_as_file(&bot_owned, chat_id, &final_text, &state_owned, "schedule")
                    .await;
            } else {
                let normalized_remaining = normalize_empty_lines(remaining);
                let final_text = format!("{}{}", normalized_remaining, continue_hint);
                let html_response = markdown_to_telegram_html(&final_text);
                msg_debug(&format!(
                    "[rolling_ph/sched] FINAL EDIT placeholder: msg_id={}, html_len={}",
                    placeholder_msg_id,
                    html_response.len()
                ));
                if html_response.len() <= TELEGRAM_MSG_LIMIT {
                    if let Err(_) = tg!(
                        "edit_message",
                        &state_owned,
                        chat_id,
                        bot_owned
                            .edit_message_text(chat_id, placeholder_msg_id, &html_response)
                            .parse_mode(ParseMode::Html)
                            .await
                    ) {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "edit_message",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .edit_message_text(chat_id, placeholder_msg_id, &final_text)
                                .await
                        );
                    }
                } else {
                    let send_result = send_long_message(
                        &bot_owned,
                        chat_id,
                        &html_response,
                        Some(ParseMode::Html),
                        &state_owned,
                    )
                    .await;
                    match send_result {
                        Ok(_) => {
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!(
                                "delete_message",
                                &state_owned,
                                chat_id,
                                bot_owned.delete_message(chat_id, placeholder_msg_id).await
                            );
                        }
                        Err(_) => {
                            let fallback = send_long_message(
                                &bot_owned,
                                chat_id,
                                &final_text,
                                None,
                                &state_owned,
                            )
                            .await;
                            match fallback {
                                Ok(_) => {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg!(
                                        "delete_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                    );
                                }
                                Err(_) => {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let truncated = truncate_str(&final_text, TELEGRAM_MSG_LIMIT);
                                    let _ = tg!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &truncated
                                            )
                                            .await
                                    );
                                }
                            }
                        }
                    }
                }
            }

            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ✓ [Schedule] Done");

            // Send end hook message if configured
            if !cancelled {
                let end_hook_msg = {
                    let data = state_owned.lock().await;
                    data.settings.end_hook.get(&chat_id.0.to_string()).cloned()
                };
                if let Some(hook_msg) = end_hook_msg {
                    shared_rate_limit_wait(&state_owned, chat_id).await;
                    let _ = tg!(
                        "send_message",
                        bot_owned.send_message(chat_id, &hook_msg).await
                    );
                }
            }
        }

        // For cron entries with context_summary, extract result summary for next run
        // Skip if execution was cancelled or encountered an error
        sched_debug(&format!("[execute_schedule] id={}, checking context summary: cancelled={}, had_error={}, type={}, once={:?}, has_context={}",
            schedule_id, cancelled, had_error, entry_clone.schedule_type, entry_clone.once, entry_clone.context_summary.is_some()));
        let sched_provider = detect_provider(model_for_summary.as_deref());
        let new_context_summary = if inline_mode {
            None
        } else if sched_provider != "claude" {
            // Codex/Gemini/OpenCode: skip summary extraction (not supported via Claude API)
            sched_debug(&format!(
                "[execute_schedule] id={}, non-Claude backend — skipping context summary",
                schedule_id
            ));
            None
        } else if !cancelled
            && !had_error
            && entry_clone.schedule_type == "cron"
            && !entry_clone.once.unwrap_or(false)
            && entry_clone.context_summary.is_some()
        {
            sched_debug(&format!(
                "[execute_schedule] id={}, extracting result summary",
                schedule_id
            ));
            if let Some(ref sid) = exec_session_id {
                let sid = sid.clone();
                let path = workspace_path_owned.clone();
                let model = model_for_summary.clone();
                let summary_result = tokio::task::spawn_blocking(move || {
                    let claude_model = model.as_deref().and_then(claude::strip_claude_prefix);
                    claude::extract_result_summary(&sid, &path, claude_model)
                })
                .await;
                match summary_result {
                    Ok(Ok(ref summary)) => {
                        sched_debug(&format!(
                            "[execute_schedule] id={}, new context summary: {} chars",
                            schedule_id,
                            summary.len()
                        ));
                        Some(summary.clone())
                    }
                    _ => {
                        sched_debug(&format!(
                            "[execute_schedule] id={}, summary extraction failed",
                            schedule_id
                        ));
                        None
                    }
                }
            } else {
                sched_debug(&format!(
                    "[execute_schedule] id={}, no session_id for summary",
                    schedule_id
                ));
                None
            }
        } else {
            None
        };

        if inline_mode {
            let save_payload = {
                let mut data = state_owned.lock().await;
                if let Some(session) = data.sessions.get_mut(&chat_id) {
                    if let Some(ref sid) = exec_session_id {
                        session.session_id = Some(sid.clone());
                    }
                    session.history.push(HistoryItem {
                        item_type: HistoryType::User,
                        content: entry_clone.prompt.clone(),
                    });
                    if !full_response.is_empty() {
                        session.history.push(HistoryItem {
                            item_type: HistoryType::Assistant,
                            content: full_response.clone(),
                        });
                    }
                    session
                        .current_path
                        .clone()
                        .map(|path| (session.clone(), path))
                } else {
                    None
                }
            };
            if let Some((session_snapshot, session_path)) = save_payload {
                save_session_to_file(&session_snapshot, &session_path, sched_provider);
            }
        } else if let Some(ref sid) = exec_session_id {
            let mut sched_session = ChatSession {
                session_id: Some(sid.clone()),
                current_path: Some(workspace_path_owned.clone()),
                history: Vec::new(),
                pending_uploads: Vec::new(),
            };
            // Add user prompt and AI response to history for session continuity
            sched_session.history.push(HistoryItem {
                item_type: HistoryType::User,
                content: entry_clone.prompt.clone(),
            });
            if !full_response.is_empty() {
                sched_session.history.push(HistoryItem {
                    item_type: HistoryType::Assistant,
                    content: full_response.clone(),
                });
            }
            // sched_provider already computed above
            save_session_to_file(&sched_session, &workspace_path_owned, sched_provider);
        }

        // Write to group chat shared log (scheduled task)
        sched_debug(&format!(
            "[sched] JSONL check: chat_id={}, raw_entries_count={}",
            chat_id.0,
            raw_entries.len()
        ));
        if chat_id.0 < 0 && !sched_bot_username.is_empty() {
            let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
            let dn = if sched_bot_display_name.is_empty() {
                None
            } else {
                Some(sched_bot_display_name.clone())
            };
            append_group_chat_log(
                chat_id.0,
                &GroupChatLogEntry {
                    ts: now_ts.clone(),
                    bot: sched_bot_username.clone(),
                    bot_display_name: dn.clone(),
                    role: "user".to_string(),
                    from: Some("scheduled_task".to_string()),
                    text: entry_clone.prompt.clone(),
                    clear: false,
                },
            );
            if !raw_entries.is_empty() {
                sched_debug(&format!(
                    "[sched] JSONL: writing user+assistant entries, raw_entries_count={}",
                    raw_entries.len()
                ));
                append_group_chat_log(
                    chat_id.0,
                    &GroupChatLogEntry {
                        ts: now_ts,
                        bot: sched_bot_username.clone(),
                        bot_display_name: dn,
                        role: "assistant".to_string(),
                        from: None,
                        text: serialize_payload(&std::mem::take(&mut raw_entries)),
                        clear: false,
                    },
                );
            } else {
                sched_debug(&format!(
                    "[sched] JSONL: user entry written, assistant SKIPPED (raw_entries is empty)"
                ));
            }
        }

        // Append a JSONL run record to ~/.cokacdir/schedule_history/<id>.log.
        // Best-effort: any I/O failure is swallowed inside append_schedule_history so
        // it cannot affect the user-facing completion path. Both `cancelled` and
        // `had_error` paths fall through to here, which is why the call lives outside
        // those branches.
        let history_status = if cancelled {
            "cancelled"
        } else if had_error {
            "error"
        } else {
            "ok"
        };
        let history_duration_ms = history_start.elapsed().as_millis() as u64;
        append_schedule_history(
            &schedule_id,
            chat_id.0,
            &history_bot_key,
            &entry_clone.prompt,
            history_status,
            &full_response,
            None,
            &workspace_path_owned,
            history_duration_ms,
        );

        // Update schedule file (last_run / delete if once)
        sched_debug(&format!(
            "[execute_schedule] id={}, calling update_schedule_after_run",
            schedule_id
        ));
        update_schedule_after_run(&entry_clone, new_context_summary);

        // Workspace directory is preserved for user to continue work via /start

        // Clean up + restore previous session
        sched_debug(&format!("[execute_schedule] id={}, cleaning up: removing cancel_token, pending, restoring session (has_prev={}, inline_mode={})",
            schedule_id, prev_session.is_some(), inline_mode));
        {
            let mut data = state_owned.lock().await;
            data.cancel_tokens.remove(&chat_id);
            if let Some(set) = data.pending_schedules.get_mut(&chat_id) {
                set.remove(&schedule_id);
            }
            if !inline_mode {
                if let Some(prev) = prev_session {
                    data.sessions.insert(chat_id, prev);
                } else {
                    // No prior session existed — remove the schedule's temporary session
                    data.sessions.remove(&chat_id);
                }
            }
        }
        sched_debug(&format!("[execute_schedule] id={}, END", schedule_id));

        // Clean up leftover stop message
        let stop_msg_id = {
            let mut data = state_owned.lock().await;
            data.stop_message_ids.remove(&chat_id)
        };
        if let Some(msg_id) = stop_msg_id {
            shared_rate_limit_wait(&state_owned, chat_id).await;
            let _ = tg!(
                "delete_message",
                bot_owned.delete_message(chat_id, msg_id).await
            );
        }
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=schedule_poll_completed",
            chat_id.0
        ));
        drop(_group_lock); // release group chat lock before processing queue
        process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
    });
}

/// Process an incoming bot-to-bot message (follows execute_schedule pattern)
async fn process_bot_message(
    bot: &Bot,
    chat_id: ChatId,
    msg: &BotMessage,
    state: &SharedState,
    token: &str,
    bot_username: &str,
    bot_display_name: &str,
) {
    msg_debug(&format!("[process_bot_message] START id={}, from={}, to={}, chat_id={}, content_len={}, bot_username={}",
        msg.id, msg.from, msg.to, chat_id.0, msg.content.len(), bot_username));

    // Register cancel token early (prevents duplicate requests while waiting for group lock)
    let cancel_token = Arc::new(CancelToken::new());
    {
        let mut data = state.lock().await;
        if data.cancel_tokens.contains_key(&chat_id) {
            msg_debug(&format!(
                "[process_bot_message] chat_id={}, cancel_token exists (busy), skipping id={}",
                chat_id.0, msg.id
            ));
            return;
        }
        data.cancel_tokens.insert(chat_id, cancel_token.clone());
    }

    // Acquire group chat lock (serializes processing across bots in the same group chat)
    let group_lock = acquire_group_chat_lock(chat_id.0).await;

    // Check if cancelled during lock wait
    if cancel_token.cancelled.load(Ordering::Relaxed) {
        msg_debug(&format!(
            "[process_bot_message] cancelled during lock wait, id={}",
            msg.id
        ));
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=botmsg_cancelled_during_lock",
            chat_id.0
        ));
        {
            let mut data = state.lock().await;
            data.cancel_tokens.remove(&chat_id);
        }
        drop(group_lock); // release before queue processing to avoid deadlock
        process_next_queued_message(bot, chat_id, state).await;
        return;
    }

    // Auto-restore session
    msg_debug(&format!(
        "[process_bot_message] auto_restore_session for bot:{}",
        msg.from
    ));
    auto_restore_session(state, chat_id, &format!("bot:{}", msg.from)).await;

    // Get session info, allowed tools, model, history, instruction
    let (
        session_info,
        allowed_tools,
        model,
        history,
        instruction,
        context_count,
        botmsg_chrome_enabled,
        botmsg_effort,
        botmsg_fast_mode,
    ) = {
        let data = state.lock().await;
        let info = data.sessions.get(&chat_id).and_then(|session| {
            session.current_path.as_ref().map(|_| {
                (
                    session.session_id.clone(),
                    session.current_path.clone().unwrap_or_default(),
                )
            })
        });
        let tools = get_allowed_tools(&data.settings, chat_id);
        let mdl = get_model(&data.settings, chat_id);
        let hist = data
            .sessions
            .get(&chat_id)
            .map(|s| s.history.clone())
            .unwrap_or_default();
        let instr = data
            .settings
            .instructions
            .get(&chat_id.0.to_string())
            .cloned();
        let ctx = data
            .settings
            .context
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(12);
        let chrome = data
            .settings
            .use_chrome
            .get(&chat_id.0.to_string())
            .copied()
            .unwrap_or(false);
        let provider = detect_provider(mdl.as_deref());
        let effort = get_effort_for_provider(&data.settings, chat_id, provider);
        let fast_mode = is_codex_fast(&data.settings, chat_id);
        msg_debug(&format!("[process_bot_message] session_info={}, tools={}, model={:?}, history_len={}, instruction={}",
            info.is_some(), tools.len(), mdl, hist.len(), instr.is_some()));
        (
            info, tools, mdl, hist, instr, ctx, chrome, effort, fast_mode,
        )
    };

    let (session_id, current_path) = match session_info {
        Some(info) => {
            msg_debug(&format!(
                "[process_bot_message] session found: session_id={:?}, path={}",
                info.0, info.1
            ));
            info
        }
        None => {
            // No active session — create an error response
            msg_debug(&format!(
                "[process_bot_message] no session for chat_id={}, sending error response",
                chat_id.0
            ));
            {
                let mut data = state.lock().await;
                data.cancel_tokens.remove(&chat_id);
                let cleared = data
                    .message_queues
                    .remove(&chat_id)
                    .map(|q| q.len())
                    .unwrap_or(0);
                if cleared > 0 {
                    msg_debug(&format!("[queue:clear] chat_id={}, no session (bot msg), cleared {} queued messages", chat_id.0, cleared));
                }
            }
            shared_rate_limit_wait(state, chat_id).await;
            let _ = tg!(
                "send_message",
                bot.send_message(
                    chat_id,
                    format!(
                        "📨 @{}: {}\n\n⚠️ No active session. Use /start <path> first.",
                        msg.from,
                        truncate_str(&msg.content, 200)
                    )
                )
                .await
            );
            msg_debug("[process_bot_message] END (no session)");
            return;
        }
    };

    // Send placeholder
    msg_debug("[process_bot_message] sending placeholder");
    shared_rate_limit_wait(state, chat_id).await;
    let (placeholder_msg_id, placeholder_visible) = match tg!(
        "send_message",
        bot.send_message(chat_id, processing_placeholder_text(chat_id.0 >= 0))
            .await
    ) {
        Ok(m) => {
            msg_debug(&format!(
                "[process_bot_message] placeholder sent: msg_id={}",
                m.id
            ));
            (m.id, true)
        }
        Err(e) => {
            msg_debug(&format!(
                    "[process_bot_message] failed to send placeholder: {}, continuing without placeholder",
                    e
                ));
            (teloxide::types::MessageId(0), false)
        }
    };

    // Build disabled tools notice
    let default_tools: std::collections::HashSet<&str> =
        DEFAULT_ALLOWED_TOOLS.iter().copied().collect();
    let allowed_set: std::collections::HashSet<&str> =
        allowed_tools.iter().map(|s| s.as_str()).collect();
    let disabled: Vec<&&str> = default_tools
        .iter()
        .filter(|t| !allowed_set.contains(**t))
        .collect();
    msg_debug(&format!(
        "[process_bot_message] disabled_tools={}",
        disabled.len()
    ));
    let disabled_notice = if disabled.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = disabled.iter().map(|t| **t).collect();
        msg_debug(&format!(
            "[process_bot_message] disabled: [{}]",
            names.join(", ")
        ));
        format!(
            "\n\nDISABLED TOOLS: The following tools have been disabled by the user: {}.\n\
             You MUST NOT attempt to use these tools. \
             If a user's request requires a disabled tool, do NOT proceed with the task. \
             Instead, clearly inform the user which tool is needed and that it is currently disabled. \
             Suggest they re-enable it with: /allowed +ToolName",
            names.join(", ")
        )
    };

    // Build system prompt
    let bot_key = token_hash(token);
    let platform = capitalize_platform(detect_platform(token));
    let role = match &instruction {
        Some(instr) => {
            msg_debug(&format!(
                "[process_bot_message] instruction present, len={}",
                instr.len()
            ));
            format!(
                "You are chatting with a user through {}.\n\nUser's instruction for this chat:\n{}",
                platform, instr
            )
        }
        None => {
            msg_debug("[process_bot_message] no instruction set");
            format!("You are chatting with a user through {}.", platform)
        }
    };
    let system_prompt_owned = build_system_prompt(
        &role,
        &current_path,
        chat_id.0,
        &bot_key,
        &disabled_notice,
        session_id.as_deref(),
        bot_username,
        bot_display_name,
        None, // bot-to-bot messages: no user message dedup
        true,
        context_count,
        &platform,
    );
    msg_debug(&format!(
        "[process_bot_message] system_prompt built, len={}",
        system_prompt_owned.len()
    ));

    // Create channel for streaming
    let (tx, rx) = mpsc::channel();

    let session_id_clone = session_id.clone();
    let current_path_clone = current_path.clone();
    let cancel_token_clone = cancel_token.clone();
    let sender_display = resolve_display_name_by_username(&msg.from)
        .map(|dn| format!("{} (@{})", dn, msg.from))
        .unwrap_or_else(|| format!("@{}", msg.from));
    let prompt = format!("[BOT MESSAGE from {}]\n{}", sender_display, msg.content);

    // Run AI backend in a blocking thread
    let model_clone = model.clone();
    let history_clone = history;
    let prompt_for_ai = prompt.clone();
    let bot_key_for_codex = bot_key.clone();
    msg_debug(&format!(
        "[process_bot_message] spawning AI backend: model={:?}, history_len={}, prompt_len={}",
        model_clone,
        history_clone.len(),
        prompt_for_ai.len()
    ));
    tokio::task::spawn_blocking(move || {
        let provider = detect_provider(model_clone.as_deref());
        msg_debug(&format!(
            "[process_bot_message:spawn_blocking] provider={}, model={:?}",
            provider, model_clone
        ));
        let result = if provider == "opencode" {
            let opencode_model = model_clone
                .as_deref()
                .and_then(opencode::strip_opencode_prefix);
            msg_debug(&format!("[process_bot_message] → opencode::execute, opencode_model={:?}, session_id={:?}, path={}, prompt_len={}, system_prompt_len={}",
                opencode_model, session_id_clone, current_path_clone, prompt_for_ai.len(), system_prompt_owned.len()));
            opencode::execute_command_streaming(
                &prompt_for_ai,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                opencode_model,
                false,
            )
        } else if provider == "gemini" {
            let gemini_model = model_clone.as_deref().and_then(gemini::strip_gemini_prefix);
            msg_debug(&format!("[process_bot_message] → gemini::execute, gemini_model={:?}, session_id={:?}, path={}, prompt_len={}, system_prompt_len={}",
                gemini_model, session_id_clone, current_path_clone, prompt_for_ai.len(), system_prompt_owned.len()));
            gemini::execute_command_streaming(
                &prompt_for_ai,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                gemini_model,
                false,
            )
        } else if provider == "codex" {
            let codex_model = model_clone.as_deref().and_then(codex::strip_codex_prefix);
            let codex_system_prompt =
                format!("{}{}", system_prompt_owned, codex_extra_instructions());
            let is_resume = session_id_clone.is_some();
            let has_history = !history_clone.is_empty();
            msg_debug(&format!("[process_bot_message] codex: is_resume={}, has_history={}, history_len={}, sp_len={}",
                is_resume, has_history, history_clone.len(), codex_system_prompt.len()));
            let codex_prompt = if session_id_clone.is_none() && !history_clone.is_empty() {
                msg_debug("[process_bot_message] codex: INJECTING history into prompt (new session with history)");
                let mut conv = String::new();
                conv.push_str("<conversation_history>\n");
                for item in &history_clone {
                    let role = match item.item_type {
                        HistoryType::User => "User",
                        HistoryType::Assistant => "Assistant",
                        HistoryType::ToolUse => "ToolUse",
                        HistoryType::ToolResult => "ToolResult",
                        _ => continue,
                    };
                    conv.push_str(&format!("[{}]: {}\n", role, item.content));
                }
                conv.push_str("</conversation_history>\n\n");
                conv.push_str(&prompt_for_ai);
                conv
            } else {
                if is_resume {
                    msg_debug("[process_bot_message] codex: RESUME path — no history injection, sp via -c file");
                } else {
                    msg_debug(
                        "[process_bot_message] codex: NEW session, no history — sp via -c file",
                    );
                }
                prompt_for_ai.clone()
            };
            msg_debug(&format!("[process_bot_message] → codex::execute, codex_model={:?}, codex_prompt_len={}, resume={}, system_prompt_passed=true",
                codex_model, codex_prompt.len(), is_resume));
            let codex_auto_send = codex::CodexAutoSendCtx {
                cokacdir_bin: crate::bin_path().to_string(),
                chat_id: chat_id.0,
                bot_key: bot_key_for_codex,
            };
            codex::execute_command_streaming(
                &codex_prompt,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&codex_system_prompt),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                codex_model,
                false,
                Some(&codex_auto_send),
                botmsg_effort.as_deref(),
                botmsg_fast_mode,
            )
        } else {
            let claude_model = model_clone.as_deref().and_then(claude::strip_claude_prefix);
            claude::execute_command_streaming(
                &prompt_for_ai,
                session_id_clone.as_deref(),
                &current_path_clone,
                tx.clone(),
                Some(&system_prompt_owned),
                Some(&allowed_tools),
                Some(cancel_token_clone),
                claude_model,
                false,
                botmsg_chrome_enabled,
                botmsg_effort.as_deref(),
            )
        };
        if let Err(e) = result {
            let _ = tx.send(StreamMessage::Error {
                message: e,
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
            });
        }
    });

    // Polling loop (spawned as async task to avoid blocking scheduler)
    let bot_owned = bot.clone();
    let state_owned = state.clone();
    let msg_clone = msg.clone();
    let provider_str: &'static str = detect_provider(model.as_deref());
    let current_path_owned = current_path.clone();
    // Captured at spawn so the post-completion guard catches /clear and /start
    // same-path session-id swaps (see handle_text_message guard for rationale).
    let captured_sid = session_id.clone();
    let captured_clear_epoch = {
        let data = state.lock().await;
        data.clear_epoch.get(&chat_id).copied().unwrap_or(0)
    };
    let prompt_owned = prompt.clone();
    let bmsg_id_for_log = msg.id.clone();
    let bot_username_for_log = bot_username.to_string();
    let bot_display_name_for_log = bot_display_name.to_string();
    let from_bot_for_log = msg.from.clone();
    msg_debug(&format!("[process_bot_message] spawning polling loop: provider={}, msg_id={}, placeholder_msg_id={}, placeholder_visible={}",
        provider_str, bmsg_id_for_log, placeholder_msg_id, placeholder_visible));
    tokio::spawn(async move {
        let _group_lock = group_lock; // hold group chat lock until task ends
        const SPINNER: &[&str] = &[
            "🕐 P",
            "🕑 Pr",
            "🕒 Pro",
            "🕓 Proc",
            "🕔 Proce",
            "🕕 Proces",
            "🕖 Process",
            "🕗 Processi",
            "🕘 Processin",
            "🕙 Processing",
            "🕚 Processing.",
            "🕛 Processing..",
        ];
        let mut full_response = String::new();
        let mut raw_entries: Vec<RawPayloadEntry> = Vec::new();
        let mut last_edit_text = String::new();
        let mut done = false;
        let mut cancelled = false;
        let mut new_session_id: Option<String> = None;
        let mut spin_idx: usize = 0;
        let mut pending_cokacdir = false;
        let mut suppress_tool_display = false;
        let mut last_tool_name: String = String::new();
        let is_direct_chat = chat_id.0 >= 0;
        let mut placeholder_msg_id = placeholder_msg_id;
        let placeholder_visible = placeholder_visible;
        let mut last_confirmed_len: usize = 0;
        let mut last_direct_chunk_send_at = tokio::time::Instant::now();

        let (polling_time_ms, silent_mode) = {
            let data = state_owned.lock().await;
            (data.polling_time_ms, is_silent(&data.settings, chat_id))
        };
        msg_debug(&format!(
            "[botmsg_poll:{}] started: polling_time_ms={}, silent_mode={}",
            bmsg_id_for_log, polling_time_ms, silent_mode
        ));

        let mut first_poll = true;
        let mut queue_done = false;
        let mut response_rendered = false;
        while !done || !queue_done {
            if cancel_token.cancelled.load(Ordering::Relaxed) {
                msg_debug(&format!(
                    "[botmsg_poll:{}] cancelled (pre-sleep check)",
                    bmsg_id_for_log
                ));
                if !done {
                    cancelled = true;
                }
                break;
            }

            if !first_poll {
                tokio::time::sleep(tokio::time::Duration::from_millis(polling_time_ms)).await;

                if cancel_token.cancelled.load(Ordering::Relaxed) {
                    if !done {
                        cancelled = true;
                    }
                    break;
                }
            }
            first_poll = false;

            // Drain messages
            if !done {
                loop {
                    match rx.try_recv() {
                        Ok(stream_msg) => match stream_msg {
                            StreamMessage::Init { session_id: sid } => {
                                msg_debug(&format!(
                                    "[botmsg_poll:{}] Init: session_id={}",
                                    bmsg_id_for_log, sid
                                ));
                                new_session_id = Some(sid);
                            }
                            StreamMessage::Text { content } => {
                                msg_debug(&format!(
                                    "[botmsg_poll:{}] Text: chunk_len={}, total_len={}",
                                    bmsg_id_for_log,
                                    content.len(),
                                    full_response.len() + content.len()
                                ));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "Text".into(),
                                    content: content.clone(),
                                });
                                let _fr_before = full_response.len();
                                full_response.push_str(&content);
                                msg_debug(&format!("[fr_trace][{}] +Text: added={}, preview={:?}, total={} (was {})",
                                        chat_id.0, content.len(), truncate_str(&content, 200), full_response.len(), _fr_before));
                            }
                            StreamMessage::ToolUse { name, input } => {
                                pending_cokacdir = detect_cokacdir_command(&name, &input);
                                suppress_tool_display = detect_chat_log_read(&name, &input);
                                last_tool_name = name.clone();
                                let summary = format_tool_input(&name, &input);
                                let ts = chrono::Local::now().format("%H:%M:%S");
                                println!("  [{ts}]   ⚙ [BotMsg] {name}: {summary}");
                                msg_debug(&format!("[botmsg_poll:{}] ToolUse: name={}, input_preview={:?}, pending_cokacdir={}, silent={}, response_len={}, ends_nl={}",
                                        bmsg_id_for_log, name, truncate_str(&input, 200), pending_cokacdir, silent_mode, full_response.len(), full_response.ends_with('\n')));
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolUse".into(),
                                    content: format!("{}: {}", name, input),
                                });
                                if !pending_cokacdir && !silent_mode {
                                    let _fr_before = full_response.len();
                                    if name == "Bash" {
                                        full_response.push_str(&format!(
                                            "\n\n```\n{}\n```\n",
                                            format_bash_command(&input)
                                        ));
                                        msg_debug(&format!("[fr_trace][{}] +ToolUse/Bash: added={}, cmd={:?}, total={} (was {})",
                                                chat_id.0, full_response.len() - _fr_before, truncate_str(&input, 100), full_response.len(), _fr_before));
                                    } else {
                                        full_response.push_str(&format!("\n\n⚙️ {}\n", summary));
                                        msg_debug(&format!("[fr_trace][{}] +ToolUse/{}: added={}, summary={:?}, total={} (was {})",
                                                chat_id.0, name, full_response.len() - _fr_before, truncate_str(&summary, 100), full_response.len(), _fr_before));
                                    }
                                } else if !pending_cokacdir
                                    && silent_mode
                                    && !full_response.is_empty()
                                    && !full_response.ends_with('\n')
                                {
                                    msg_debug(&format!("[botmsg_poll:{}] silent mode: inserting \\n\\n after tool_use={}", bmsg_id_for_log, name));
                                    full_response.push_str("\n\n");
                                    msg_debug(&format!(
                                        "[fr_trace][{}] +ToolUse/silent_nl: added=2, total={}",
                                        chat_id.0,
                                        full_response.len()
                                    ));
                                }
                            }
                            StreamMessage::ToolResult { content, is_error } => {
                                msg_debug(&format!("[botmsg_poll:{}] ToolResult: is_error={}, content_len={}, pending_cokacdir={}, last_tool={}",
                                        bmsg_id_for_log, is_error, content.len(), pending_cokacdir, last_tool_name));
                                if is_error {
                                    msg_debug(&format!("[botmsg_poll:{}] ToolResult ERROR: last_tool={}, content_preview={:?}", bmsg_id_for_log, last_tool_name, truncate_str(&content, 300)));
                                }
                                raw_entries.push(RawPayloadEntry {
                                    tag: "ToolResult".into(),
                                    content: format!("is_error={}, content={}", is_error, content),
                                });
                                let _fr_before = full_response.len();
                                if std::mem::take(&mut pending_cokacdir) {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    if std::mem::take(&mut suppress_tool_display) {
                                        println!(
                                            "  [{ts}]   ↩ [BotMsg] cokacdir (chat_log, suppressed)"
                                        );
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir_suppressed: added=0, total={}", chat_id.0, full_response.len()));
                                    } else {
                                        println!("  [{ts}]   ↩ [BotMsg] cokacdir: {content}");
                                        let formatted = format_cokacdir_result(&content);
                                        msg_debug(&format!(
                                            "[botmsg_poll:{}] cokacdir result formatted_len={}",
                                            bmsg_id_for_log,
                                            formatted.len()
                                        ));
                                        if !formatted.is_empty() {
                                            full_response.push_str(&format!("\n{}\n", formatted));
                                            msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: added={}, preview={:?}, total={} (was {})",
                                                    chat_id.0, full_response.len() - _fr_before, truncate_str(&formatted, 200), full_response.len(), _fr_before));
                                        } else {
                                            msg_debug(&format!("[fr_trace][{}] +ToolResult/cokacdir: formatted_empty, added=0, total={}", chat_id.0, full_response.len()));
                                        }
                                    }
                                } else if is_error && !silent_mode {
                                    msg_debug(&format!(
                                        "[botmsg_poll:{}] tool error: {}",
                                        bmsg_id_for_log,
                                        truncate_str(&content, 200)
                                    ));
                                    let truncated = truncate_str(&content, 500);
                                    if truncated.contains('\n') {
                                        full_response
                                            .push_str(&format!("\n```\n{}\n```\n", truncated));
                                    } else {
                                        full_response.push_str(&format!("\n`{}`\n\n", truncated));
                                    }
                                    msg_debug(&format!("[fr_trace][{}] +ToolResult/error({}): added={}, preview={:?}, total={} (was {})",
                                            chat_id.0, last_tool_name, full_response.len() - _fr_before, truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                } else if !silent_mode {
                                    if last_tool_name == "Read" {
                                        full_response.push_str(&format!(
                                            "\n✅ `{} bytes`\n\n",
                                            content.len()
                                        ));
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/Read: added={}, content_bytes={}, total={} (was {})",
                                                chat_id.0, full_response.len() - _fr_before, content.len(), full_response.len(), _fr_before));
                                    } else if !content.is_empty() {
                                        let truncated = truncate_str(&content, 300);
                                        if truncated.contains('\n') {
                                            full_response
                                                .push_str(&format!("\n```\n{}\n```\n", truncated));
                                        } else {
                                            full_response
                                                .push_str(&format!("\n✅ `{}`\n\n", truncated));
                                        }
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): added={}, raw_content_len={}, preview={:?}, total={} (was {})",
                                                chat_id.0, last_tool_name, full_response.len() - _fr_before, content.len(), truncate_str(&truncated, 200), full_response.len(), _fr_before));
                                    } else {
                                        msg_debug(&format!("[fr_trace][{}] +ToolResult/{}(normal): content_empty, added=0, total={}", chat_id.0, last_tool_name, full_response.len()));
                                    }
                                } else {
                                    msg_debug(&format!("[fr_trace][{}] +ToolResult/silent: skipped, last_tool={}, content_len={}, total={}", chat_id.0, last_tool_name, content.len(), full_response.len()));
                                }
                            }
                            StreamMessage::TaskNotification { summary, .. } => {
                                msg_debug(&format!(
                                    "[botmsg_poll:{}] TaskNotification: summary_len={}",
                                    bmsg_id_for_log,
                                    summary.len()
                                ));
                                if !summary.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "TaskNotification".into(),
                                        content: summary.clone(),
                                    });
                                    let _fr_before = full_response.len();
                                    full_response.push_str(&format!("\n[Task: {}]\n", summary));
                                    msg_debug(&format!("[fr_trace][{}] +TaskNotification: added={}, summary={:?}, total={} (was {})",
                                            chat_id.0, full_response.len() - _fr_before, truncate_str(&summary, 200), full_response.len(), _fr_before));
                                }
                            }
                            StreamMessage::Done {
                                result,
                                session_id: sid,
                            } => {
                                msg_debug(&format!("[botmsg_poll:{}] Done: result_len={}, session_id={:?}, full_response_len={}",
                                        bmsg_id_for_log, result.len(), sid, full_response.len()));
                                if !result.is_empty() && full_response.is_empty() {
                                    msg_debug(&format!("[botmsg_poll:{}] Done: fallback full_response = result ({})", bmsg_id_for_log, result.len()));
                                    full_response = result.clone();
                                    msg_debug(&format!("[fr_trace][{}] +Done/fallback: set={}, preview={:?}, total={}",
                                            chat_id.0, result.len(), truncate_str(&full_response, 200), full_response.len()));
                                } else if !result.is_empty() {
                                    msg_debug(&format!("[fr_trace][{}] Done/discarded: result_len={} discarded (full_response already has {})",
                                            chat_id.0, result.len(), full_response.len()));
                                }
                                if !result.is_empty() && raw_entries.is_empty() {
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Text".into(),
                                        content: result,
                                    });
                                }
                                if let Some(s) = sid {
                                    new_session_id = Some(s);
                                }
                                msg_debug(&format!(
                                    "[fr_trace][{}] =DONE: final_total={}",
                                    chat_id.0,
                                    full_response.len()
                                ));
                                done = true;
                            }
                            StreamMessage::Error {
                                message,
                                stdout,
                                stderr,
                                exit_code,
                            } => {
                                msg_debug(&format!("[botmsg_poll:{}] Error: msg={}, exit_code={:?}, stdout_len={}, stderr_len={}",
                                        bmsg_id_for_log, message, exit_code, stdout.len(), stderr.len()));
                                let stdout_display = display_stream_pipe_payload(stdout);
                                let stderr_display = display_stream_pipe_payload(stderr);
                                let code_display = display_stream_exit_code(exit_code);
                                if should_preserve_completed_stream_response(done, &full_response) {
                                    msg_debug(&format!(
                                        "[botmsg_poll:{}] late error suppressed after terminal response: message={}, response_len={}",
                                        bmsg_id_for_log,
                                        message,
                                        full_response.len()
                                    ));
                                } else {
                                    full_response = format_stream_error_response(
                                        &message,
                                        &code_display,
                                        &stdout_display,
                                        &stderr_display,
                                    );
                                    msg_debug(&format!("[fr_trace][{}] +Error: set={}, stdout_len={}, stderr_len={}, total={}",
                                            chat_id.0, full_response.len(), stdout_display.len(), stderr_display.len(), full_response.len()));
                                    raw_entries.push(RawPayloadEntry {
                                        tag: "Error".into(),
                                        content: format!(
                                            "exit_code={}, message={}, stdout={}, stderr={}",
                                            code_display, message, stdout_display, stderr_display
                                        ),
                                    });
                                }
                                done = true;
                            }
                        },
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            msg_debug(&format!(
                                "[botmsg_poll:{}] channel disconnected, setting done=true",
                                bmsg_id_for_log
                            ));
                            done = true;
                            break;
                        }
                    }
                }

                if !done && !placeholder_visible {
                    if should_stream_direct_without_placeholder(is_direct_chat, placeholder_visible)
                    {
                        let raw_segment = &full_response[last_confirmed_len..];
                        let idle_ms = last_direct_chunk_send_at.elapsed().as_millis();
                        if should_flush_direct_no_edit_stream_chunk(
                            raw_segment.len(),
                            idle_ms,
                            last_confirmed_len,
                        ) {
                            let flush_len = direct_no_edit_stream_chunk_len(raw_segment);
                            if flush_len > 0 {
                                let flush_text = direct_placeholder_segment_text(
                                    &raw_segment[..flush_len],
                                    None,
                                );
                                if flush_text.trim().is_empty() {
                                    last_confirmed_len = direct_segment_next_confirmed_len(
                                        last_confirmed_len,
                                        flush_len,
                                        true,
                                    );
                                } else if send_direct_segment_as_message(
                                    &bot_owned,
                                    chat_id,
                                    &flush_text,
                                    &state_owned,
                                )
                                .await
                                {
                                    last_confirmed_len = direct_segment_next_confirmed_len(
                                        last_confirmed_len,
                                        flush_len,
                                        true,
                                    );
                                    last_direct_chunk_send_at = tokio::time::Instant::now();
                                }
                            }
                        }
                    } else {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_chat_action",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .send_chat_action(chat_id, teloxide::types::ChatAction::Typing)
                                .await
                        );
                    }
                } else if !done {
                    // ── Rolling placeholder pattern (unified for all chats) ──
                    let threshold = file_attach_threshold();
                    let delta_end =
                        floor_char_boundary(&full_response, full_response.len().min(threshold));
                    if delta_end > last_confirmed_len {
                        let delta = &full_response[last_confirmed_len..delta_end];
                        let normalized_delta = normalize_empty_lines(delta);
                        let html_delta = markdown_to_telegram_html(&normalized_delta);
                        if html_delta.trim().is_empty() {
                            last_confirmed_len = delta_end;
                        } else {
                            msg_debug(&format!(
                                "[botmsg_poll:{}] finalizing placeholder with delta_len={}",
                                bmsg_id_for_log,
                                normalized_delta.len()
                            ));
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            if html_delta.len() <= TELEGRAM_MSG_LIMIT {
                                let _ = tg!(
                                    "edit_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned
                                        .edit_message_text(chat_id, placeholder_msg_id, &html_delta)
                                        .parse_mode(ParseMode::Html)
                                        .await
                                );
                            } else {
                                if send_long_message(
                                    &bot_owned,
                                    chat_id,
                                    &html_delta,
                                    Some(ParseMode::Html),
                                    &state_owned,
                                )
                                .await
                                .is_ok()
                                {
                                    shared_rate_limit_wait(&state_owned, chat_id).await;
                                    let _ = tg!(
                                        "delete_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                    );
                                } else {
                                    let truncated_delta =
                                        truncate_str(&normalized_delta, TELEGRAM_MSG_LIMIT);
                                    let _ = tg!(
                                        "edit_message",
                                        &state_owned,
                                        chat_id,
                                        bot_owned
                                            .edit_message_text(
                                                chat_id,
                                                placeholder_msg_id,
                                                &truncated_delta
                                            )
                                            .await
                                    );
                                }
                            }
                            last_confirmed_len = delta_end;
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            match tg!(
                                "send_message",
                                &state_owned,
                                chat_id,
                                bot_owned.send_message(chat_id, "...").await
                            ) {
                                Ok(new_ph) => {
                                    placeholder_msg_id = new_ph.id;
                                }
                                Err(e) => {
                                    let ts = chrono::Local::now().format("%H:%M:%S");
                                    println!("  [{ts}]   ⚠ new placeholder failed (botmsg): {e}");
                                }
                            }
                            last_edit_text.clear();
                            spin_idx = 0;
                        }
                    } else {
                        let indicator = SPINNER[spin_idx % SPINNER.len()];
                        spin_idx += 1;
                        let display_text = indicator.to_string();
                        if display_text != last_edit_text {
                            msg_debug(&format!(
                                "[botmsg_poll:{}] spinner update spin_idx={}",
                                bmsg_id_for_log, spin_idx
                            ));
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let html_text = markdown_to_telegram_html(&display_text);
                            let _ = tg!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(chat_id, placeholder_msg_id, &html_text)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            );
                            last_edit_text = display_text;
                        } else {
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!(
                                "send_chat_action",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .send_chat_action(chat_id, teloxide::types::ChatAction::Typing)
                                    .await
                            );
                        }
                    }
                }
            }

            // Render final response once when AI completes
            if done && !response_rendered {
                response_rendered = true;
                msg_debug(&format!(
                    "[botmsg_poll:{}] rendering final response: response_len={}",
                    bmsg_id_for_log,
                    full_response.len()
                ));

                shared_rate_limit_wait(&state_owned, chat_id).await;

                if full_response.is_empty() {
                    msg_debug(&format!(
                        "[botmsg_poll:{}] empty response, using placeholder text",
                        bmsg_id_for_log
                    ));
                    full_response = "(No response)".to_string();
                    msg_debug(&format!(
                        "[fr_trace][{}] =NoResponse: set to '(No response)'",
                        chat_id.0
                    ));
                }

                let final_response = normalize_empty_lines(&full_response);
                msg_debug(&format!(
                    "[botmsg_poll:{}] final: response_len={}, msg_limit={}",
                    bmsg_id_for_log,
                    final_response.len(),
                    TELEGRAM_MSG_LIMIT
                ));

                // ── Send only remaining delta (unified rolling placeholder) ──
                let remaining_start =
                    botmsg_direct_remaining_start(&full_response, last_confirmed_len);
                let remaining = &full_response[remaining_start..];
                msg_debug(&format!("[rolling_ph/botmsg] FINAL: placeholder_msg_id={}, visible={}, confirmed={}, total={}, remaining_len={}",
                    placeholder_msg_id, placeholder_visible, last_confirmed_len, full_response.len(), remaining.trim().len()));
                if !placeholder_visible {
                    let final_remaining = normalize_empty_lines(remaining);
                    if final_remaining.trim().is_empty() {
                        // Direct no-placeholder streaming may have already delivered all chunks.
                    } else if should_attach_response_as_file(final_remaining.len(), provider_str) {
                        send_response_as_file(
                            &bot_owned,
                            chat_id,
                            &final_remaining,
                            &state_owned,
                            "botmsg",
                        )
                        .await;
                    } else {
                        let html_final = markdown_to_telegram_html(&final_remaining);
                        if html_final.len() <= TELEGRAM_MSG_LIMIT {
                            if let Err(e) = tg!(
                                "send_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .send_message(chat_id, &html_final)
                                    .parse_mode(ParseMode::Html)
                                    .await
                            ) {
                                msg_debug(&format!(
                                    "[botmsg_poll:{}] direct final HTML send failed: {}",
                                    bmsg_id_for_log, e
                                ));
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg!(
                                    "send_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned.send_message(chat_id, &final_remaining).await
                                );
                            }
                        } else {
                            let send_result = send_long_message(
                                &bot_owned,
                                chat_id,
                                &html_final,
                                Some(ParseMode::Html),
                                &state_owned,
                            )
                            .await;
                            if send_result.is_err() {
                                let _ = send_long_message(
                                    &bot_owned,
                                    chat_id,
                                    &final_remaining,
                                    None,
                                    &state_owned,
                                )
                                .await;
                            }
                        }
                    }
                } else if remaining.trim().is_empty() {
                    msg_debug(&format!(
                        "[rolling_ph/botmsg] FINAL DELETE placeholder: msg_id={}",
                        placeholder_msg_id
                    ));
                    shared_rate_limit_wait(&state_owned, chat_id).await;
                    let _ = tg!(
                        "delete_message",
                        &state_owned,
                        chat_id,
                        bot_owned.delete_message(chat_id, placeholder_msg_id).await
                    );
                } else if should_attach_response_as_file(full_response.len(), provider_str) {
                    msg_debug(&format!(
                        "[rolling_ph/botmsg] FINAL FILE ATTACH: total={}",
                        full_response.len()
                    ));
                    shared_rate_limit_wait(&state_owned, chat_id).await;
                    let _ = tg!(
                        "edit_message",
                        &state_owned,
                        chat_id,
                        bot_owned
                            .edit_message_text(
                                chat_id,
                                placeholder_msg_id,
                                "\u{1f4c4} Response attached as file"
                            )
                            .await
                    );
                    send_response_as_file(
                        &bot_owned,
                        chat_id,
                        &final_response,
                        &state_owned,
                        "botmsg",
                    )
                    .await;
                } else {
                    let normalized_remaining = normalize_empty_lines(remaining);
                    let html_remaining = markdown_to_telegram_html(&normalized_remaining);
                    msg_debug(&format!(
                        "[botmsg_poll:{}] final delta_len={}, html_len={}",
                        bmsg_id_for_log,
                        normalized_remaining.len(),
                        html_remaining.len()
                    ));
                    if html_remaining.len() <= TELEGRAM_MSG_LIMIT {
                        if let Err(e) = tg!(
                            "edit_message",
                            &state_owned,
                            chat_id,
                            bot_owned
                                .edit_message_text(chat_id, placeholder_msg_id, &html_remaining)
                                .parse_mode(ParseMode::Html)
                                .await
                        ) {
                            msg_debug(&format!(
                                "[botmsg_poll:{}] HTML edit failed: {}",
                                bmsg_id_for_log, e
                            ));
                            shared_rate_limit_wait(&state_owned, chat_id).await;
                            let _ = tg!(
                                "edit_message",
                                &state_owned,
                                chat_id,
                                bot_owned
                                    .edit_message_text(
                                        chat_id,
                                        placeholder_msg_id,
                                        &normalized_remaining
                                    )
                                    .await
                            );
                        }
                    } else {
                        let send_result = send_long_message(
                            &bot_owned,
                            chat_id,
                            &html_remaining,
                            Some(ParseMode::Html),
                            &state_owned,
                        )
                        .await;
                        match send_result {
                            Ok(_) => {
                                shared_rate_limit_wait(&state_owned, chat_id).await;
                                let _ = tg!(
                                    "delete_message",
                                    &state_owned,
                                    chat_id,
                                    bot_owned.delete_message(chat_id, placeholder_msg_id).await
                                );
                            }
                            Err(_) => {
                                let fallback = send_long_message(
                                    &bot_owned,
                                    chat_id,
                                    &normalized_remaining,
                                    None,
                                    &state_owned,
                                )
                                .await;
                                match fallback {
                                    Ok(_) => {
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let _ = tg!(
                                            "delete_message",
                                            &state_owned,
                                            chat_id,
                                            bot_owned
                                                .delete_message(chat_id, placeholder_msg_id)
                                                .await
                                        );
                                    }
                                    Err(_) => {
                                        shared_rate_limit_wait(&state_owned, chat_id).await;
                                        let truncated =
                                            truncate_str(&normalized_remaining, TELEGRAM_MSG_LIMIT);
                                        let _ = tg!(
                                            "edit_message",
                                            &state_owned,
                                            chat_id,
                                            bot_owned
                                                .edit_message_text(
                                                    chat_id,
                                                    placeholder_msg_id,
                                                    &truncated
                                                )
                                                .await
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Update session state
                {
                    let mut data = state_owned.lock().await;
                    // Same guard as text-message polling: skip session writeback if /model or
                    // /start changed the session during this bot-to-bot task. detect_provider
                    // mirrors the spawn-time capture (see normal text-msg guard for rationale).
                    // sid + clear_epoch catch /clear and /start same-path swaps.
                    let model_now = data.settings.models.get(&chat_id.0.to_string()).cloned();
                    let provider_now = detect_provider(model_now.as_deref());
                    let path_now = data
                        .sessions
                        .get(&chat_id)
                        .and_then(|s| s.current_path.clone());
                    let sid_now = data
                        .sessions
                        .get(&chat_id)
                        .and_then(|s| s.session_id.clone());
                    let epoch_now = data.clear_epoch.get(&chat_id).copied().unwrap_or(0);
                    let session_changed = provider_now != provider_str
                        || path_now.as_deref() != Some(current_path_owned.as_str())
                        || sid_now != captured_sid
                        || epoch_now != captured_clear_epoch;
                    if session_changed {
                        msg_debug(&format!("[botmsg_poll:{}] session changed during task — skip session update (path: {:?} → {:?}, provider: {} → {}, sid: {:?} → {:?}, epoch: {} → {})",
                            bmsg_id_for_log, current_path_owned, path_now, provider_str, provider_now, captured_sid, sid_now, captured_clear_epoch, epoch_now));
                    } else if let Some(session) = data.sessions.get_mut(&chat_id) {
                        msg_debug(&format!("[botmsg_poll:{}] updating session: new_session_id={:?}, history_len_before={}",
                            bmsg_id_for_log, new_session_id, session.history.len()));
                        if let Some(sid) = new_session_id.take() {
                            session.session_id = Some(sid);
                        }
                        invalidate_saved_session_after_stream_error(
                            session,
                            provider_str,
                            &final_response,
                        );
                        session.history.push(HistoryItem {
                            item_type: HistoryType::User,
                            content: prompt_owned.clone(),
                        });
                        session.history.push(HistoryItem {
                            item_type: HistoryType::Assistant,
                            content: final_response.clone(),
                        });
                        save_session_to_file(session, &current_path_owned, provider_str);
                        msg_debug(&format!(
                            "[botmsg_poll:{}] session saved: history_len_after={}",
                            bmsg_id_for_log,
                            session.history.len()
                        ));
                    } else {
                        msg_debug(&format!("[botmsg_poll:{}] no session found for chat_id={}, skipping session update", bmsg_id_for_log, chat_id.0));
                    }
                    // Write to group chat shared log (bot-to-bot messages)
                    msg_debug(&format!(
                        "[botmsg_poll:{}] JSONL check: chat_id={}, raw_entries_count={}",
                        bmsg_id_for_log,
                        chat_id.0,
                        raw_entries.len()
                    ));
                    if chat_id.0 < 0 {
                        let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                        let dn = if bot_display_name_for_log.is_empty() {
                            None
                        } else {
                            Some(bot_display_name_for_log.clone())
                        };
                        append_group_chat_log(
                            chat_id.0,
                            &GroupChatLogEntry {
                                ts: now_ts.clone(),
                                bot: bot_username_for_log.clone(),
                                bot_display_name: dn.clone(),
                                role: "user".to_string(),
                                from: Some(format!("bot:{}", from_bot_for_log)),
                                text: prompt_owned.clone(),
                                clear: false,
                            },
                        );
                        if !raw_entries.is_empty() {
                            msg_debug(&format!("[botmsg_poll:{}] JSONL: writing user+assistant entries, raw_entries_count={}", bmsg_id_for_log, raw_entries.len()));
                            append_group_chat_log(
                                chat_id.0,
                                &GroupChatLogEntry {
                                    ts: now_ts,
                                    bot: bot_username_for_log.clone(),
                                    bot_display_name: dn,
                                    role: "assistant".to_string(),
                                    from: None,
                                    text: serialize_payload(&std::mem::take(&mut raw_entries)),
                                    clear: false,
                                },
                            );
                        } else {
                            msg_debug(&format!("[botmsg_poll:{}] JSONL: user entry written, assistant SKIPPED (raw_entries is empty)", bmsg_id_for_log));
                        }
                    }
                }

                let ts = chrono::Local::now().format("%H:%M:%S");
                println!("  [{ts}] ▶ [BotMsg] Response sent");
                msg_debug(&format!(
                    "[botmsg_poll:{}] response sent, final_response_len={}",
                    bmsg_id_for_log,
                    final_response.len()
                ));

                // Do NOT auto-create response message file.
                // The AI can use --message CLI to reply if needed.
                // Auto-responding causes infinite ping-pong between bots.
                msg_debug(&format!(
                    "[botmsg_poll:{}] skipping auto-response (AI uses --message if needed)",
                    bmsg_id_for_log
                ));

                // Send end hook message if configured
                if !cancelled {
                    let end_hook_msg = {
                        let data = state_owned.lock().await;
                        data.settings.end_hook.get(&chat_id.0.to_string()).cloned()
                    };
                    if let Some(hook_msg) = end_hook_msg {
                        shared_rate_limit_wait(&state_owned, chat_id).await;
                        let _ = tg!(
                            "send_message",
                            bot_owned.send_message(chat_id, &hook_msg).await
                        );
                    }
                }
            }

            // Queue processing
            let queued = process_upload_queue(&bot_owned, chat_id, &state_owned).await;
            if done {
                queue_done = !queued;
                msg_debug(&format!(
                    "[botmsg_poll:{}] queue: queued={}, queue_done={}",
                    bmsg_id_for_log, queued, queue_done
                ));
            }
        }

        // Handle cancellation
        if cancelled {
            msg_debug(&format!(
                "[botmsg_poll:{}] handling cancellation: response_len={}",
                bmsg_id_for_log,
                full_response.len()
            ));
            if let Ok(guard) = cancel_token.child_pid.lock() {
                if let Some(pid) = *guard {
                    msg_debug(&format!(
                        "[botmsg_poll:{}] killing child process: pid={}",
                        bmsg_id_for_log, pid
                    ));
                } else {
                    msg_debug(&format!(
                        "[botmsg_poll:{}] no child pid to kill",
                        bmsg_id_for_log
                    ));
                }
            }
            cancel_token_now(&cancel_token);

            // stopped_response (full) for session history
            let stopped_response = if full_response.trim().is_empty() {
                "[Stopped]".to_string()
            } else {
                let normalized = normalize_empty_lines(&full_response);
                format!("{}\n\n[Stopped]", normalized)
            };
            msg_debug(&format!(
                "[botmsg_poll:{}] stopped_response_len={}",
                bmsg_id_for_log,
                stopped_response.len()
            ));

            shared_rate_limit_wait(&state_owned, chat_id).await;

            // ── Show remaining delta + [Stopped] (unified rolling placeholder) ──
            if full_response.len() < last_confirmed_len
                || !full_response.is_char_boundary(last_confirmed_len)
            {
                last_confirmed_len = 0;
            }
            let remaining = &full_response[last_confirmed_len..];
            msg_debug(&format!("[rolling_ph/botmsg] STOPPED: placeholder_msg_id={}, visible={}, confirmed={}, remaining_len={}",
                placeholder_msg_id, placeholder_visible, last_confirmed_len, remaining.trim().len()));
            if !placeholder_visible {
                if should_attach_response_as_file(full_response.len(), provider_str) {
                    send_response_as_file(
                        &bot_owned,
                        chat_id,
                        &stopped_response,
                        &state_owned,
                        "botmsg",
                    )
                    .await;
                } else {
                    let display_stopped = if remaining.trim().is_empty() {
                        "[Stopped]".to_string()
                    } else {
                        let normalized = normalize_empty_lines(remaining);
                        format!("{}\n\n[Stopped]", normalized)
                    };
                    let html_stopped = markdown_to_telegram_html(&display_stopped);
                    if html_stopped.len() <= TELEGRAM_MSG_LIMIT {
                        let _ = tg!(
                            "send_message",
                            bot_owned
                                .send_message(chat_id, &html_stopped)
                                .parse_mode(ParseMode::Html)
                                .await
                        );
                    } else {
                        let _ = send_long_message(
                            &bot_owned,
                            chat_id,
                            &display_stopped,
                            None,
                            &state_owned,
                        )
                        .await;
                    }
                }
            } else if should_attach_response_as_file(full_response.len(), provider_str) {
                msg_debug(&format!(
                    "[rolling_ph/botmsg] STOPPED FILE ATTACH: total={}",
                    full_response.len()
                ));
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "edit_message",
                    bot_owned
                        .edit_message_text(
                            chat_id,
                            placeholder_msg_id,
                            "\u{1f4c4} Response attached as file [Stopped]"
                        )
                        .await
                );
                send_response_as_file(
                    &bot_owned,
                    chat_id,
                    &stopped_response,
                    &state_owned,
                    "botmsg",
                )
                .await;
            } else {
                let display_stopped = if remaining.trim().is_empty() {
                    "[Stopped]".to_string()
                } else {
                    let normalized = normalize_empty_lines(remaining);
                    format!("{}\n\n[Stopped]", normalized)
                };
                let html_stopped = markdown_to_telegram_html(&display_stopped);
                if html_stopped.len() <= TELEGRAM_MSG_LIMIT {
                    let _ = tg!(
                        "edit_message",
                        bot_owned
                            .edit_message_text(chat_id, placeholder_msg_id, &html_stopped)
                            .parse_mode(ParseMode::Html)
                            .await
                    );
                } else {
                    let _ = tg!(
                        "edit_message",
                        bot_owned
                            .edit_message_text(
                                chat_id,
                                placeholder_msg_id,
                                &truncate_str(&display_stopped, TELEGRAM_MSG_LIMIT)
                            )
                            .await
                    );
                }
            }

            // Do NOT create response file on cancel → chain broken
            let ts = chrono::Local::now().format("%H:%M:%S");
            println!("  [{ts}] ■ [BotMsg] Stopped");
            msg_debug(&format!(
                "[botmsg_poll:{}] stopped message sent, updating session",
                bmsg_id_for_log
            ));

            let mut data = state_owned.lock().await;
            // Same guard: skip the partial-response writeback if the session was reset
            // (via /model or /start) during this bot-to-bot task. Group chat log is still
            // written below — mirrors the normal-completion branch so other bots see the
            // partial response. detect_provider mirrors the spawn-time capture.
            // sid + clear_epoch catch /clear and /start same-path swaps.
            let model_now = data.settings.models.get(&chat_id.0.to_string()).cloned();
            let provider_now = detect_provider(model_now.as_deref());
            let path_now = data
                .sessions
                .get(&chat_id)
                .and_then(|s| s.current_path.clone());
            let sid_now = data
                .sessions
                .get(&chat_id)
                .and_then(|s| s.session_id.clone());
            let epoch_now = data.clear_epoch.get(&chat_id).copied().unwrap_or(0);
            let session_changed = provider_now != provider_str
                || path_now.as_deref() != Some(current_path_owned.as_str())
                || sid_now != captured_sid
                || epoch_now != captured_clear_epoch;
            if session_changed {
                msg_debug(&format!("[botmsg_poll:{}] cancel: session changed during task — skip session writeback (path: {:?} → {:?}, provider: {} → {}, sid: {:?} → {:?}, epoch: {} → {})",
                    bmsg_id_for_log, current_path_owned, path_now, provider_str, provider_now, captured_sid, sid_now, captured_clear_epoch, epoch_now));
            } else if let Some(session) = data.sessions.get_mut(&chat_id) {
                msg_debug(&format!(
                    "[botmsg_poll:{}] cancel: updating session history, new_session_id={:?}",
                    bmsg_id_for_log, new_session_id
                ));
                if let Some(sid) = new_session_id {
                    session.session_id = Some(sid);
                }
                invalidate_saved_session_after_stream_error(
                    session,
                    provider_str,
                    &stopped_response,
                );
                session.history.push(HistoryItem {
                    item_type: HistoryType::User,
                    content: prompt_owned.clone(),
                });
                session.history.push(HistoryItem {
                    item_type: HistoryType::Assistant,
                    content: stopped_response,
                });
                save_session_to_file(session, &current_path_owned, provider_str);
            } else {
                msg_debug(&format!(
                    "[botmsg_poll:{}] cancel: no session found for chat_id={}",
                    bmsg_id_for_log, chat_id.0
                ));
            }
            // Write to group chat shared log (bot-to-bot messages, stopped).
            // Mirrors the normal-completion branch: written regardless of session_changed.
            msg_debug(&format!(
                "[botmsg_poll:{}] JSONL stopped check: chat_id={}, raw_entries_count={}",
                bmsg_id_for_log,
                chat_id.0,
                raw_entries.len()
            ));
            if chat_id.0 < 0 {
                let now_ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string();
                let dn = if bot_display_name_for_log.is_empty() {
                    None
                } else {
                    Some(bot_display_name_for_log.clone())
                };
                append_group_chat_log(
                    chat_id.0,
                    &GroupChatLogEntry {
                        ts: now_ts.clone(),
                        bot: bot_username_for_log.clone(),
                        bot_display_name: dn.clone(),
                        role: "user".to_string(),
                        from: Some(format!("bot:{}", from_bot_for_log)),
                        text: prompt_owned,
                        clear: false,
                    },
                );
                if !raw_entries.is_empty() {
                    msg_debug(&format!("[botmsg_poll:{}] JSONL stopped: writing user+assistant entries, raw_entries_count={}", bmsg_id_for_log, raw_entries.len()));
                    append_group_chat_log(
                        chat_id.0,
                        &GroupChatLogEntry {
                            ts: now_ts,
                            bot: bot_username_for_log.clone(),
                            bot_display_name: dn,
                            role: "assistant".to_string(),
                            from: None,
                            text: serialize_payload(&std::mem::take(&mut raw_entries)),
                            clear: false,
                        },
                    );
                } else {
                    msg_debug(&format!("[botmsg_poll:{}] JSONL stopped: user entry written, assistant SKIPPED (raw_entries is empty)", bmsg_id_for_log));
                }
            }
            data.cancel_tokens.remove(&chat_id);
            let stop_msg_id = data.stop_message_ids.remove(&chat_id);
            drop(data);
            if let Some(msg_id) = stop_msg_id {
                shared_rate_limit_wait(&state_owned, chat_id).await;
                let _ = tg!(
                    "delete_message",
                    bot_owned.delete_message(chat_id, msg_id).await
                );
            }
            msg_debug(&format!(
                "[botmsg_poll:{}] cancel cleanup done",
                bmsg_id_for_log
            ));
            msg_debug(&format!(
                "[queue:trigger] chat_id={}, source=botmsg_poll_cancelled",
                chat_id.0
            ));
            drop(_group_lock); // release group chat lock before processing queue
            process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
            return;
        }

        // Clean up cancel token
        msg_debug(&format!(
            "[botmsg_poll:{}] normal completion, cleaning up",
            bmsg_id_for_log
        ));
        let orphan_stop_msg = {
            let mut data = state_owned.lock().await;
            let msg_id = data.stop_message_ids.remove(&chat_id);
            data.cancel_tokens.remove(&chat_id);
            msg_debug(&format!(
                "[botmsg_poll:{}] cleaned up: orphan_stop_msg={:?}",
                bmsg_id_for_log, msg_id
            ));
            msg_id
        };
        if let Some(msg_id) = orphan_stop_msg {
            msg_debug(&format!(
                "[botmsg_poll:{}] deleting orphan stop message: {}",
                bmsg_id_for_log, msg_id
            ));
            shared_rate_limit_wait(&state_owned, chat_id).await;
            let _ = tg!(
                "delete_message",
                bot_owned.delete_message(chat_id, msg_id).await
            );
        }
        msg_debug(&format!("[botmsg_poll:{}] END", bmsg_id_for_log));
        msg_debug(&format!(
            "[queue:trigger] chat_id={}, source=botmsg_poll_completed",
            chat_id.0
        ));
        drop(_group_lock); // release group chat lock before processing queue
        process_next_queued_message(&bot_owned, chat_id, &state_owned).await;
    });
    msg_debug(&format!(
        "[process_bot_message] END (tasks spawned) id={}",
        msg.id
    ));
}

/// Scheduler loop: runs every 60 seconds, checks for due schedules
async fn scheduler_loop(
    bot: Bot,
    state: SharedState,
    token: String,
    bot_username: String,
    bot_display_name: String,
) {
    let bot_key = token_hash(&token);
    sched_debug("[scheduler_loop] started");

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

        // Scan schedule directory
        let entries = list_schedule_entries(&bot_key, None);

        if !entries.is_empty() {
            sched_debug(&format!(
                "[scheduler_loop] cycle: {} entries found",
                entries.len()
            ));
        }

        for entry in &entries {
            let chat_id = ChatId(entry.chat_id);

            // Verify current_path exists (before acquiring lock — involves filesystem I/O)
            if !Path::new(&entry.current_path).is_dir() {
                let ts = chrono::Local::now().format("%H:%M:%S");
                println!(
                    "  [{ts}] ⚠ [Scheduler] Path not found: {} (schedule: {})",
                    entry.current_path, entry.id
                );
                sched_debug(&format!(
                    "[scheduler_loop] id={}, path not found: {} → skip",
                    entry.id, entry.current_path
                ));
                shared_rate_limit_wait(&state, chat_id).await;
                let msg = format!(
                    "⏰ {}\n\n⚠️ Skipped — path no longer exists\n📂 <code>{}</code>",
                    html_escape(&truncate_str(&entry.prompt, 40)),
                    html_escape(&entry.current_path)
                );
                let _ = tg!(
                    "send_message",
                    bot.send_message(chat_id, msg)
                        .parse_mode(ParseMode::Html)
                        .await
                );
                continue;
            }

            // Single atomic lock: pending check + trigger check + busy check + session backup
            // All checks in one lock to prevent race between pending cleanup and re-trigger
            enum SchedAction {
                Skip,
                DiscardExpired,
                Execute {
                    prev_session: Option<ChatSession>,
                    inline_mode: bool,
                },
            }

            let inline_env_on = is_schedule_inline_enabled();
            let action = {
                let mut data = state.lock().await;
                let is_already_pending = data
                    .pending_schedules
                    .get(&chat_id)
                    .map_or(false, |set| set.contains(&entry.id));

                sched_debug(&format!(
                    "[scheduler_loop] id={}, is_already_pending={}",
                    entry.id, is_already_pending
                ));

                // If not pending and not due to trigger, skip
                if !is_already_pending && !should_trigger(entry) {
                    // Check if expired absolute schedule should be discarded
                    if entry.schedule_type == "absolute" {
                        if let Ok(schedule_time) = chrono::NaiveDateTime::parse_from_str(
                            &entry.schedule,
                            "%Y-%m-%d %H:%M:%S",
                        ) {
                            if let Some(schedule_dt) =
                                schedule_time.and_local_timezone(chrono::Local).single()
                            {
                                if chrono::Local::now() > schedule_dt {
                                    sched_debug(&format!(
                                        "[scheduler_loop] id={}, expired absolute → discard",
                                        entry.id
                                    ));
                                    SchedAction::DiscardExpired
                                } else {
                                    sched_debug(&format!(
                                        "[scheduler_loop] id={}, not yet due → skip",
                                        entry.id
                                    ));
                                    SchedAction::Skip
                                }
                            } else {
                                SchedAction::Skip
                            }
                        } else {
                            SchedAction::Skip
                        }
                    } else {
                        SchedAction::Skip
                    }
                } else {
                    // Entry should execute — check if chat is busy
                    let is_busy = data.cancel_tokens.contains_key(&chat_id);
                    sched_debug(&format!(
                        "[scheduler_loop] id={}, should execute, is_busy={}",
                        entry.id, is_busy
                    ));

                    if is_busy {
                        // Chat is busy — mark as pending if not already, retry next cycle
                        // Do NOT touch sessions — leave them as-is
                        if !is_already_pending {
                            data.pending_schedules
                                .entry(chat_id)
                                .or_default()
                                .insert(entry.id.clone());
                            let ts = chrono::Local::now().format("%H:%M:%S");
                            println!("  [{ts}] ⏰ [Scheduler] Chat busy, pending: {}", entry.id);
                            sched_debug(&format!(
                                "[scheduler_loop] id={}, chat busy → marked pending",
                                entry.id
                            ));
                        } else {
                            sched_debug(&format!(
                                "[scheduler_loop] id={}, chat busy, already pending → skip",
                                entry.id
                            ));
                        }
                        SchedAction::Skip
                    } else {
                        let inline_usable = inline_env_on
                            && data
                                .sessions
                                .get(&chat_id)
                                .and_then(|s| s.current_path.as_ref())
                                .is_some();
                        data.pending_schedules
                            .entry(chat_id)
                            .or_default()
                            .insert(entry.id.clone());
                        // Pre-insert cancel_token to prevent race with incoming user messages
                        let cancel_token = Arc::new(CancelToken::new());
                        data.cancel_tokens.insert(chat_id, cancel_token);
                        if inline_usable {
                            sched_debug(&format!(
                                "[scheduler_loop] id={}, not busy → execute inline",
                                entry.id
                            ));
                            SchedAction::Execute {
                                prev_session: None,
                                inline_mode: true,
                            }
                        } else {
                            // Not busy — backup session, replace with schedule-specific session, and execute
                            let prev = data.sessions.get(&chat_id).cloned();
                            sched_debug(&format!(
                                "[scheduler_loop] id={}, not busy → execute isolated (has_prev_session={})",
                                entry.id,
                                prev.is_some()
                            ));
                            data.sessions.insert(
                                chat_id,
                                ChatSession {
                                    session_id: None,
                                    current_path: Some(entry.current_path.clone()),
                                    history: Vec::new(),
                                    pending_uploads: Vec::new(),
                                },
                            );
                            SchedAction::Execute {
                                prev_session: prev,
                                inline_mode: false,
                            }
                        }
                    }
                }
            };

            match action {
                SchedAction::Skip => continue,
                SchedAction::DiscardExpired => {
                    delete_schedule_entry(&entry.id);
                    let ts = chrono::Local::now().format("%H:%M:%S");
                    println!(
                        "  [{ts}] ⏰ [Scheduler] Discarded expired once-schedule: {}",
                        entry.id
                    );
                    sched_debug(&format!(
                        "[scheduler_loop] id={}, discarded expired",
                        entry.id
                    ));
                    continue;
                }
                SchedAction::Execute {
                    prev_session,
                    inline_mode,
                } => {
                    sched_debug(&format!(
                        "[scheduler_loop] id={}, calling execute_schedule inline={}",
                        entry.id, inline_mode
                    ));
                    execute_schedule(
                        &bot,
                        chat_id,
                        entry,
                        &state,
                        &token,
                        prev_session,
                        inline_mode,
                    )
                    .await;
                }
            }
        }

        // === Bot-to-bot message polling ===
        if !bot_username.is_empty() {
            let messages = scan_messages(&bot_username);
            if !messages.is_empty() {
                msg_debug(&format!(
                    "[scheduler_loop] bot messages found: {} for @{}",
                    messages.len(),
                    bot_username
                ));
            }
            for msg in &messages {
                msg_debug(&format!("[scheduler_loop] bot message: id={}, from={}, to={}, chat_id={}, content_len={}, created_at={}",
                    msg.id, msg.from, msg.to, msg.chat_id, msg.content.len(), msg.created_at));
                let chat_id_num: i64 = match msg.chat_id.parse() {
                    Ok(n) => n,
                    Err(e) => {
                        msg_debug(&format!("[scheduler_loop] invalid chat_id in message: id={}, chat_id={:?}, error={}", msg.id, msg.chat_id, e));
                        let remove_result = fs::remove_file(&msg.file_path);
                        msg_debug(&format!(
                            "[scheduler_loop] removed invalid message file: ok={}",
                            remove_result.is_ok()
                        ));
                        continue;
                    }
                };
                let chat_id = ChatId(chat_id_num);

                // Busy check: skip if cancel_token exists for this chat
                {
                    let data = state.lock().await;
                    let is_busy = data.cancel_tokens.contains_key(&chat_id);
                    msg_debug(&format!(
                        "[scheduler_loop] busy check for chat {}: is_busy={}",
                        chat_id_num, is_busy
                    ));
                    if is_busy {
                        msg_debug(&format!("[scheduler_loop] chat {} busy, skipping message: {} (will retry next cycle)", chat_id_num, msg.id));
                        continue;
                    }
                }

                // Delete message file immediately (mark as processed)
                let remove_result = fs::remove_file(&msg.file_path);
                msg_debug(&format!(
                    "[scheduler_loop] deleted message file: id={}, path={}, ok={}",
                    msg.id,
                    msg.file_path.display(),
                    remove_result.is_ok()
                ));
                msg_debug(&format!(
                    "[scheduler_loop] processing bot message: {} (from={}, to={}, chat_id={})",
                    msg.id, msg.from, msg.to, chat_id_num
                ));

                // Process the message
                process_bot_message(
                    &bot,
                    chat_id,
                    msg,
                    &state,
                    &token,
                    &bot_username,
                    &bot_display_name,
                )
                .await;
                msg_debug(&format!(
                    "[scheduler_loop] process_bot_message returned for msg: {}",
                    msg.id
                ));
            }

            // Check for timed-out sent messages
            msg_debug("[scheduler_loop] checking message timeouts");
            check_message_timeouts(&bot, &bot_username, &state).await;
        }
    }
}
