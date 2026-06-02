//! Gemini service — spawns `cokacdir --bridge gemini` with Claude-compatible
//! arguments and reuses the existing `StreamMessage` / `ClaudeResponse` types.
//!
//! The public API mirrors `claude.rs` so callers can swap backends with minimal
//! code changes.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::OnceLock;
use serde_json::Value;

use crate::services::claude::{
    ClaudeResponse, StreamMessage, CancelToken,
    debug_log_to, kill_child_tree, DEFAULT_ALLOWED_TOOLS,
};

fn gemini_debug(msg: &str) {
    debug_log_to("gemini.log", msg);
}

/// Truncate a string for log previews (char-boundary safe).
fn log_preview(s: &str, max: usize) -> &str {
    if s.len() <= max { return s; }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

// ============================================================
// Gemini availability check (is `gemini` CLI installed?)
// ============================================================

static GEMINI_BIN: OnceLock<Option<String>> = OnceLock::new();

fn detect_gemini_bin() -> Option<String> {
    gemini_debug("[detect_gemini_bin] START");

    if let Ok(val) = std::env::var("COKAC_GEMINI_PATH") {
        if !val.is_empty() && std::path::Path::new(&val).exists() {
            gemini_debug(&format!("[detect_gemini_bin] found via COKAC_GEMINI_PATH={}", val));
            return Some(val);
        }
    }

    #[cfg(unix)]
    {
        if let Ok(output) = Command::new("which").arg("gemini").output() {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && std::path::Path::new(&p).exists() {
                    gemini_debug(&format!("[detect_gemini_bin] found via which: {}", p));
                    return Some(p);
                }
            }
        }
        if let Ok(output) = Command::new("bash").args(["-lc", "which gemini"]).output() {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && std::path::Path::new(&p).exists() {
                    gemini_debug(&format!("[detect_gemini_bin] found via bash -lc which: {}", p));
                    return Some(p);
                }
            }
        }
    }

    #[cfg(windows)]
    {
        if let Some(path) = crate::services::claude::search_path_wide("gemini", Some(".cmd")) {
            gemini_debug(&format!("[detect_gemini_bin] found via search_path_wide .cmd: {}", path));
            return Some(path);
        }
        if let Some(path) = crate::services::claude::search_path_wide("gemini", Some(".exe")) {
            gemini_debug(&format!("[detect_gemini_bin] found via search_path_wide .exe: {}", path));
            return Some(path);
        }
    }

    gemini_debug("[detect_gemini_bin] NOT FOUND");
    None
}

fn gemini_bin() -> Option<&'static String> {
    GEMINI_BIN.get_or_init(detect_gemini_bin).as_ref()
}

/// Check if Gemini CLI is available
pub fn is_gemini_available() -> bool {
    let result = gemini_bin().is_some();
    gemini_debug(&format!("[is_gemini_available] result={}", result));
    result
}

// ============================================================
// Gemini version detection + --skip-trust capability gate
// ============================================================

static GEMINI_VERSION: OnceLock<Option<String>> = OnceLock::new();
static GEMINI_SUPPORTS_SKIP_TRUST: OnceLock<bool> = OnceLock::new();

fn detect_gemini_version() -> Option<String> {
    let bin = gemini_bin()?;
    gemini_debug(&format!("[detect_gemini_version] running {} --version", bin));
    let output = Command::new(bin).arg("--version").output().ok()?;
    if !output.status.success() {
        gemini_debug(&format!("[detect_gemini_version] non-zero exit: {:?}", output.status));
        return None;
    }
    let v = String::from_utf8_lossy(&output.stdout).trim().to_string();
    gemini_debug(&format!("[detect_gemini_version] version={:?}", v));
    if v.is_empty() { None } else { Some(v) }
}

/// Returns the cached Gemini CLI version string (e.g. "0.39.1", "0.40.0-preview.3")
/// or None if Gemini is unavailable / version probe failed.
pub fn gemini_version() -> Option<&'static String> {
    GEMINI_VERSION.get_or_init(detect_gemini_version).as_ref()
}

/// Decide whether the given version string supports the `--skip-trust` flag.
/// Introduced in stable v0.39.1 and preview v0.40.0-preview.3 (PR #25814,
/// merged 2026-04-23). Anything older lacks the flag and would error out.
fn version_supports_skip_trust(v: &str) -> bool {
    let v = v.trim().trim_start_matches('v');
    let (core, pre) = match v.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (v, None),
    };
    let parts: Vec<u32> = core.split('.').filter_map(|p| p.parse().ok()).collect();
    if parts.len() < 3 { return false; }
    let (major, minor, patch) = (parts[0], parts[1], parts[2]);

    if major > 0 { return true; }
    if minor > 40 { return true; }
    if minor == 40 {
        return match pre {
            None => true,
            Some(p) if p.starts_with("preview.") => {
                p.trim_start_matches("preview.").split('.').next()
                    .and_then(|n| n.parse::<u32>().ok())
                    .map(|n| n >= 3)
                    .unwrap_or(false)
            }
            Some(p) if p.starts_with("nightly.") => {
                // 0.40.0-nightly.YYYYMMDD.gHASH — supported only for nightlies
                // dated on/after the merge (2026-04-23).
                p.trim_start_matches("nightly.").split('.').next()
                    .and_then(|d| d.parse::<u32>().ok())
                    .map(|d| d >= 20260423)
                    .unwrap_or(false)
            }
            _ => false,
        };
    }
    if minor == 39 && pre.is_none() && patch >= 1 { return true; }
    false
}

fn probe_gemini_supports_skip_trust() -> bool {
    let Some(v) = gemini_version() else { return false; };
    let result = version_supports_skip_trust(v);
    gemini_debug(&format!("[probe_gemini_supports_skip_trust] version={} → supported={}", v, result));
    result
}

/// Returns true if the installed Gemini CLI accepts `--skip-trust`.
/// Cached after the first call.
pub fn gemini_supports_skip_trust() -> bool {
    *GEMINI_SUPPORTS_SKIP_TRUST.get_or_init(probe_gemini_supports_skip_trust)
}

/// Check if a model string refers to the Gemini backend
pub fn is_gemini_model(model: Option<&str>) -> bool {
    let result = model.map(|m| m == "gemini" || m.starts_with("gemini:")).unwrap_or(false);
    gemini_debug(&format!("[is_gemini_model] model={:?} result={}", model, result));
    result
}

/// Strip "gemini:" prefix and return the actual model name.
/// Returns None if the input is just "gemini" (use default).
/// Also strips display-name suffix (" — Description") that may have been
/// stored when a user copy-pasted the full help line.
pub fn strip_gemini_prefix(model: &str) -> Option<&str> {
    let result = model.strip_prefix("gemini:")
        .filter(|s| !s.is_empty())
        .map(|s| s.split(" \u{2014} ").next().unwrap_or(s).trim());
    gemini_debug(&format!("[strip_gemini_prefix] model={:?} result={:?}", model, result));
    result
}

// ============================================================
// Build the `cokacdir --bridge gemini` command
// ============================================================

fn build_bridge_command(
    _prompt: &str,
    session_id: Option<&str>,
    working_dir: &str,
    system_prompt_file: Option<&str>,
    allowed_tools: Option<&[String]>,
    model: Option<&str>,
    output_format: &str,
    verbose: bool,
    _no_session_persistence: bool,
) -> (Command, Option<std::path::PathBuf>) {
    let self_bin = crate::bin_path();
    gemini_debug(&format!("[build_cmd] bin={} working_dir={} session_id={:?} model={:?} format={} verbose={}",
        self_bin, working_dir, session_id, model, output_format, verbose));

    let tools_str = match allowed_tools {
        Some(tools) => tools.join(","),
        None => DEFAULT_ALLOWED_TOOLS.join(","),
    };
    gemini_debug(&format!("[build_cmd] tools_str_len={}", tools_str.len()));

    let mut args: Vec<String> = vec![
        "--bridge".into(), "gemini".into(),
        "-p".into(),
        "--dangerously-skip-permissions".into(),
        "--tools".into(), tools_str,
        "--output-format".into(), output_format.into(),
    ];

    // System prompt file
    let default_system_prompt = r#"You are a terminal file manager assistant. Be concise. Focus on file operations. Respond in the same language as the user.

SECURITY RULES (MUST FOLLOW):
- NEVER execute destructive commands like rm -rf, format, mkfs, dd, etc.
- NEVER modify system files in /etc, /sys, /proc, /boot
- NEVER access or modify files outside the current working directory without explicit user path
- NEVER execute commands that could harm the system or compromise security
- ONLY suggest safe file operations: copy, move, rename, create directory, view, edit
- If a request seems dangerous, explain the risk and suggest a safer alternative

BASH EXECUTION RULES (MUST FOLLOW):
- All commands MUST run non-interactively without user input
- Use -y, --yes, or --non-interactive flags (e.g., apt install -y, npm init -y)
- Use -m flag for commit messages (e.g., git commit -m "message")
- Disable pagers with --no-pager or pipe to cat (e.g., git --no-pager log)
- NEVER use commands that open editors (vim, nano, etc.)
- NEVER use commands that wait for stdin without arguments
- NEVER use interactive flags like -i

IMPORTANT: Format your responses using Markdown for better readability:
- Use **bold** for important terms or commands
- Use `code` for file paths, commands, and technical terms
- Use bullet lists (- item) for multiple items
- Use numbered lists (1. item) for sequential steps
- Use code blocks (```language) for multi-line code or command examples
- Use headers (## Title) to organize longer responses
- Keep formatting minimal and terminal-friendly"#;

    // Write system prompt to temp file
    fn simple_uuid() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        format!("{:x}_{}", nanos, std::process::id())
    }

    let sp_dir = dirs::home_dir().unwrap_or_else(std::env::temp_dir).join(".cokacdir");
    let _ = std::fs::create_dir_all(&sp_dir);
    let sp_path = sp_dir.join(format!("gemini_sp_{}", simple_uuid()));

    let effective_sp = match system_prompt_file {
        Some("") => {
            gemini_debug("[build_cmd] system_prompt_file is empty string → no system prompt");
            None
        }
        Some(p) => {
            // Use caller-provided file directly
            gemini_debug(&format!("[build_cmd] using caller system prompt file: {}", p));
            args.push("--append-system-prompt-file".into());
            args.push(p.to_string());
            None // no temp file to clean
        }
        None => {
            // Write default system prompt
            gemini_debug(&format!("[build_cmd] writing default system prompt to temp file: {}", sp_path.display()));
            if std::fs::write(&sp_path, default_system_prompt).is_ok() {
                args.push("--append-system-prompt-file".into());
                args.push(sp_path.to_string_lossy().to_string());
                gemini_debug(&format!("[build_cmd] default system prompt written: {} bytes", default_system_prompt.len()));
                Some(sp_path.clone())
            } else {
                gemini_debug("[build_cmd] FAILED to write default system prompt");
                None
            }
        }
    };

    if verbose {
        args.push("--verbose".into());
    }

    if let Some(m) = model {
        args.push("--model".into());
        args.push(m.to_string());
        gemini_debug(&format!("[build_cmd] model={}", m));
    }

    if let Some(sid) = session_id {
        args.push("--resume".into());
        args.push(sid.to_string());
        gemini_debug(&format!("[build_cmd] resume session={}", sid));
    }

    gemini_debug(&format!("[build_cmd] full args: {} {}", self_bin, args.join(" ")));

    let mut cmd = Command::new(self_bin);
    cmd.args(&args)
        .current_dir(working_dir)
        .env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        .env("BASH_DEFAULT_TIMEOUT_MS", "86400000")
        .env("BASH_MAX_TIMEOUT_MS", "86400000")
        .env_remove("CLAUDECODE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if gemini_supports_skip_trust() {
        cmd.env("COKAC_GEMINI_SKIP_TRUST", "1");
    } else {
        cmd.env_remove("COKAC_GEMINI_SKIP_TRUST");
    }

    (cmd, effective_sp)
}

// ============================================================
// execute_command — non-streaming, json mode
// ============================================================

pub fn execute_command(
    prompt: &str,
    session_id: Option<&str>,
    working_dir: &str,
    allowed_tools: Option<&[String]>,
    model: Option<&str>,
) -> ClaudeResponse {
    gemini_debug(&format!("[execute_command] START prompt_len={} session_id={:?} working_dir={} model={:?}",
        prompt.len(), session_id, working_dir, model));
    gemini_debug(&format!("[execute_command] prompt_preview={:?}", log_preview(prompt, 200)));

    let (mut cmd, sp_path) = build_bridge_command(
        prompt, session_id, working_dir, None, allowed_tools, model,
        "json", false, false,
    );

    gemini_debug("[execute_command] spawning process...");
    let mut child = match cmd.spawn() {
        Ok(c) => {
            gemini_debug(&format!("[execute_command] spawned PID={}", c.id()));
            c
        }
        Err(e) => {
            gemini_debug(&format!("[execute_command] spawn FAILED: {}", e));
            return ClaudeResponse {
                success: false, response: None, session_id: None,
                error: Some(format!("Failed to start bridge: {}", e)),
            };
        }
    };

    // Write prompt to stdin and close to signal EOF to bridge
    if let Some(mut stdin) = child.stdin.take() {
        match stdin.write_all(prompt.as_bytes()) {
            Ok(()) => gemini_debug(&format!("[execute_command] stdin: wrote {} bytes", prompt.len())),
            Err(e) => gemini_debug(&format!("[execute_command] stdin write FAILED: {}", e)),
        }
        drop(stdin);
        gemini_debug("[execute_command] stdin closed");
    } else {
        gemini_debug("[execute_command] WARN: no stdin handle");
    }

    // Clean up temp file on exit
    struct Guard(Option<std::path::PathBuf>);
    impl Drop for Guard {
        fn drop(&mut self) {
            if let Some(ref p) = self.0 { let _ = std::fs::remove_file(p); }
        }
    }
    let _guard = Guard(sp_path);

    gemini_debug("[execute_command] waiting for output...");
    match child.wait_with_output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            gemini_debug(&format!("[execute_command] exit={:?} stdout_len={} stderr_len={}",
                output.status.code(), stdout.len(), stderr.len()));
            if !stderr.is_empty() {
                gemini_debug(&format!("[execute_command] STDERR: {}", log_preview(&stderr, 500)));
            }

            // Parse like parse_claude_output — bridge already outputs Claude-compatible JSON
            let mut sid: Option<String> = None;
            let mut response_text = String::new();
            let mut line_count = 0u32;

            for line in stdout.trim().lines() {
                line_count += 1;
                gemini_debug(&format!("[execute_command] line {}: {}", line_count, log_preview(line, 300)));

                if let Ok(json) = serde_json::from_str::<Value>(line) {
                    if let Some(s) = json.get("session_id").and_then(|v| v.as_str()) {
                        gemini_debug(&format!("[execute_command] session_id extracted: {}", s));
                        sid = Some(s.to_string());
                    }
                    if let Some(r) = json.get("result").and_then(|v| v.as_str()) {
                        gemini_debug(&format!("[execute_command] result field: {} chars", r.len()));
                        response_text = r.to_string();
                    } else if let Some(m) = json.get("message").and_then(|v| v.as_str()) {
                        gemini_debug(&format!("[execute_command] message field: {} chars", m.len()));
                        response_text = m.to_string();
                    } else if let Some(c) = json.get("content").and_then(|v| v.as_str()) {
                        gemini_debug(&format!("[execute_command] content field: {} chars", c.len()));
                        response_text = c.to_string();
                    }
                } else if !line.trim().is_empty() && !line.starts_with('{') {
                    gemini_debug(&format!("[execute_command] non-JSON line appended: {} chars", line.len()));
                    response_text.push_str(line);
                    response_text.push('\n');
                }
            }

            if response_text.is_empty() {
                gemini_debug("[execute_command] WARN: parsed response empty, falling back to raw stdout");
                response_text = stdout.trim().to_string();
            }

            gemini_debug(&format!("[execute_command] DONE: lines={} response_len={} session_id={:?} success={}",
                line_count, response_text.len(), sid, output.status.success()));

            ClaudeResponse {
                success: output.status.success(),
                response: Some(response_text.trim().to_string()),
                session_id: sid,
                error: if output.status.success() { None } else {
                    Some(stderr)
                },
            }
        }
        Err(e) => {
            gemini_debug(&format!("[execute_command] wait_with_output FAILED: {}", e));
            ClaudeResponse {
                success: false, response: None, session_id: None,
                error: Some(format!("Failed to read output: {}", e)),
            }
        }
    }
}

// ============================================================
// execute_command_streaming — stream-json mode
// ============================================================

/// Same signature as `claude::execute_command_streaming`.
pub fn execute_command_streaming(
    prompt: &str,
    session_id: Option<&str>,
    working_dir: &str,
    sender: Sender<StreamMessage>,
    system_prompt: Option<&str>,
    allowed_tools: Option<&[String]>,
    cancel_token: Option<std::sync::Arc<CancelToken>>,
    model: Option<&str>,
    no_session_persistence: bool,
) -> Result<(), String> {
    gemini_debug("=== gemini execute_command_streaming START ===");
    gemini_debug(&format!("[stream] prompt_len={} session_id={:?} working_dir={} model={:?}",
        prompt.len(), session_id, working_dir, model));
    gemini_debug(&format!("[stream] system_prompt_len={} cancel_token={} no_session_persistence={}",
        system_prompt.map_or(0, |s| s.len()), cancel_token.is_some(), no_session_persistence));
    gemini_debug(&format!("[stream] prompt_preview={:?}", log_preview(prompt, 200)));

    // Determine system prompt file
    // If system_prompt is None → use default (bridge handles it)
    // If system_prompt is Some("") → no system prompt
    // If system_prompt is Some(text) → write to temp file
    let sp_file_path: Option<std::path::PathBuf>;
    let sp_arg: Option<String>;

    match system_prompt {
        None => {
            gemini_debug("[stream] system_prompt=None → using default");
            sp_file_path = None;
            sp_arg = None;
        }
        Some("") => {
            gemini_debug("[stream] system_prompt=\"\" → no system prompt");
            sp_file_path = None;
            sp_arg = Some(String::new());
        }
        Some(text) => {
            let sp_dir = dirs::home_dir().unwrap_or_else(std::env::temp_dir).join(".cokacdir");
            let _ = std::fs::create_dir_all(&sp_dir);
            let path = sp_dir.join(format!("gemini_sp_stream_{}", {
                use std::time::{SystemTime, UNIX_EPOCH};
                let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
                format!("{:x}_{}", n, std::process::id())
            }));
            gemini_debug(&format!("[stream] writing system prompt to temp file: {} ({} bytes)", path.display(), text.len()));
            match std::fs::write(&path, text) {
                Ok(()) => gemini_debug(&format!("[stream] system prompt written OK: {}", path.display())),
                Err(e) => {
                    gemini_debug(&format!("[stream] system prompt write FAILED: {}", e));
                    return Err(format!("Failed to write system prompt: {}", e));
                }
            }
            sp_arg = Some(path.to_string_lossy().to_string());
            sp_file_path = Some(path);
        }
    }

    struct SpGuard(Option<std::path::PathBuf>);
    impl Drop for SpGuard { fn drop(&mut self) { if let Some(ref p) = self.0 { let _ = std::fs::remove_file(p); } } }
    let _sp_guard = SpGuard(sp_file_path);

    let (mut cmd, default_sp_path) = build_bridge_command(
        prompt,
        session_id,
        working_dir,
        sp_arg.as_deref().or(None),
        allowed_tools,
        model,
        "stream-json",
        true, // verbose
        no_session_persistence,
    );

    let _default_sp_guard = SpGuard(default_sp_path);

    gemini_debug("[stream] spawning process...");
    let mut child = cmd.spawn().map_err(|e| {
        gemini_debug(&format!("[stream] spawn FAILED: {}", e));
        format!("Failed to start bridge: {}", e)
    })?;
    gemini_debug(&format!("[stream] spawned PID={}", child.id()));

    // Store PID for cancel
    if let Some(ref token) = cancel_token {
        *token.child_pid.lock().unwrap() = Some(child.id());
        if token.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            gemini_debug("[stream] cancelled before stdin write, killing");
            kill_child_tree(&mut child);
            let _ = child.wait();
            return Ok(());
        }
    }

    // Write prompt to stdin and close to signal EOF to bridge
    if let Some(mut stdin) = child.stdin.take() {
        match stdin.write_all(prompt.as_bytes()) {
            Ok(()) => gemini_debug(&format!("[stream] stdin: wrote {} bytes", prompt.len())),
            Err(e) => gemini_debug(&format!("[stream] stdin write FAILED: {}", e)),
        }
        drop(stdin);
        gemini_debug("[stream] stdin closed");
    } else {
        gemini_debug("[stream] WARN: no stdin handle");
    }

    // Read stdout line by line — output is already Claude-compatible
    let stdout = child.stdout.take().ok_or_else(|| {
        gemini_debug("[stream] FAILED to capture stdout");
        "Failed to capture stdout".to_string()
    })?;
    let reader = BufReader::new(stdout);
    gemini_debug("[stream] stdout reader ready, entering event loop");

    let mut last_session_id: Option<String> = None;
    let mut final_result: Option<String> = None;
    let mut stdout_error: Option<(String, String)> = None;
    let mut event_count = 0u32;
    let mut text_event_count = 0u32;
    let mut tool_event_count = 0u32;

    for line in reader.lines() {
        if let Some(ref token) = cancel_token {
            if token.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                gemini_debug("[stream] cancelled during event loop, killing");
                kill_child_tree(&mut child);
                let _ = child.wait();
                return Ok(());
            }
        }

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                gemini_debug(&format!("[stream] stdout read error: {}", e));
                let _ = sender.send(StreamMessage::Error {
                    message: format!("Failed to read output: {}", e),
                    stdout: String::new(), stderr: String::new(), exit_code: None,
                });
                break;
            }
        };

        if line.trim().is_empty() { continue; }

        event_count += 1;
        gemini_debug(&format!("[stream] RAW[{}]: {}", event_count, log_preview(&line, 500)));

        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            // Reuse claude's parse_stream_message logic by checking fields directly
            let msg_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let msg_subtype = json.get("subtype").and_then(|v| v.as_str()).unwrap_or("");
            gemini_debug(&format!("[stream] event {}: type={:?} subtype={:?}", event_count, msg_type, msg_subtype));

            let msg = match msg_type {
                "system" => {
                    match msg_subtype {
                        "init" => {
                            let sid = json.get("session_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            gemini_debug(&format!("[stream] INIT: session_id={}", sid));
                            last_session_id = Some(sid.clone());
                            Some(StreamMessage::Init { session_id: sid })
                        }
                        "task_notification" => {
                            let task_id = json.get("task_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let status = json.get("status").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let summary = json.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            gemini_debug(&format!("[stream] TASK_NOTIFICATION: task_id={} status={} summary={:?}",
                                task_id, status, log_preview(&summary, 100)));
                            Some(StreamMessage::TaskNotification { task_id, status, summary })
                        }
                        _ => {
                            gemini_debug(&format!("[stream] system event with unknown subtype={:?}", msg_subtype));
                            None
                        }
                    }
                }
                "assistant" => {
                    if let Some(content) = json.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_array()) {
                        let content_len = content.len();
                        if let Some(item) = content.first() {
                            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            gemini_debug(&format!("[stream] ASSISTANT: content_items={} first_item_type={}", content_len, item_type));
                            match item_type {
                                "text" => {
                                    text_event_count += 1;
                                    let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    gemini_debug(&format!("[stream] TEXT[{}]: {} chars, preview={:?}",
                                        text_event_count, text.len(), log_preview(&text, 100)));
                                    Some(StreamMessage::Text { content: text })
                                }
                                "tool_use" => {
                                    tool_event_count += 1;
                                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let input = item.get("input")
                                        .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
                                        .unwrap_or_default();
                                    gemini_debug(&format!("[stream] TOOL_USE[{}]: name={} input_len={}",
                                        tool_event_count, name, input.len()));
                                    Some(StreamMessage::ToolUse { name, input })
                                }
                                _ => {
                                    gemini_debug(&format!("[stream] assistant item with unknown type={}", item_type));
                                    None
                                }
                            }
                        } else {
                            gemini_debug("[stream] assistant message with empty content array");
                            None
                        }
                    } else {
                        gemini_debug("[stream] assistant message missing content array");
                        None
                    }
                }
                "user" => {
                    if let Some(content) = json.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_array()) {
                        if let Some(item) = content.first() {
                            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if item_type == "tool_result" {
                                let ct = if let Some(s) = item.get("content").and_then(|v| v.as_str()) {
                                    s.to_string()
                                } else if let Some(arr) = item.get("content").and_then(|v| v.as_array()) {
                                    arr.iter()
                                        .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                } else {
                                    String::new()
                                };
                                let is_error = item.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                                gemini_debug(&format!("[stream] TOOL_RESULT: is_error={} content_len={} preview={:?}",
                                    is_error, ct.len(), log_preview(&ct, 200)));
                                Some(StreamMessage::ToolResult { content: ct, is_error })
                            } else {
                                gemini_debug(&format!("[stream] user item with type={} (not tool_result)", item_type));
                                None
                            }
                        } else {
                            gemini_debug("[stream] user message with empty content array");
                            None
                        }
                    } else {
                        gemini_debug("[stream] user message missing content array");
                        None
                    }
                }
                "result" => {
                    let is_error = json.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                    if is_error {
                        let errors = json.get("errors").and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join("; "))
                            .or_else(|| json.get("result").and_then(|v| v.as_str()).map(String::from))
                            .unwrap_or_else(|| "Unknown error".to_string());
                        gemini_debug(&format!("[stream] RESULT ERROR: {}", errors));
                        stdout_error = Some((errors.clone(), line.clone()));
                        Some(StreamMessage::Error {
                            message: errors, stdout: String::new(), stderr: String::new(), exit_code: None,
                        })
                    } else {
                        let result = json.get("result").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let sid = json.get("session_id").and_then(|v| v.as_str()).map(String::from);
                        gemini_debug(&format!("[stream] RESULT OK: result_len={} session_id={:?}", result.len(), sid));
                        final_result = Some(result.clone());
                        if sid.is_some() { last_session_id = sid.clone(); }
                        Some(StreamMessage::Done { result, session_id: sid })
                    }
                }
                _ => {
                    gemini_debug(&format!("[stream] UNKNOWN type={:?}: {}", msg_type, log_preview(&line, 200)));
                    None
                }
            };

            if let Some(m) = msg {
                match &m {
                    StreamMessage::Error { message, .. } => {
                        gemini_debug(&format!("[stream] deferring Error message: {}", message));
                        // Don't send yet; combine with stderr after process exits
                        continue;
                    }
                    _ => {}
                }
                if sender.send(m).is_err() {
                    gemini_debug("[stream] send failed (receiver dropped)");
                    break;
                }
            }
        } else {
            gemini_debug(&format!("[stream] JSON parse failed for event {}", event_count));
        }
    }

    gemini_debug(&format!("[stream] event loop ended: events={} text_events={} tool_events={} final_result={}",
        event_count, text_event_count, tool_event_count, final_result.is_some()));

    // Check cancel
    if let Some(ref token) = cancel_token {
        if token.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            gemini_debug("[stream] cancelled after event loop, killing");
            kill_child_tree(&mut child);
            let _ = child.wait();
            return Ok(());
        }
    }

    gemini_debug("[stream] waiting for process exit...");
    let status = child.wait().map_err(|e| {
        gemini_debug(&format!("[stream] wait FAILED: {}", e));
        format!("Process error: {}", e)
    })?;

    // Always capture stderr for diagnostics
    let stderr_msg = child.stderr.take()
        .and_then(|s| std::io::read_to_string(s).ok())
        .unwrap_or_default();
    if !stderr_msg.is_empty() {
        gemini_debug(&format!("[stream] STDERR: {}", log_preview(&stderr_msg, 500)));
    }
    gemini_debug(&format!("[stream] exit_code={:?} success={} final_result={} stderr_len={}",
        status.code(), status.success(), final_result.is_some(), stderr_msg.len()));

    if stdout_error.is_some() || !status.success() {
        let (message, stdout_raw) = if let Some((msg, raw)) = stdout_error {
            gemini_debug(&format!("[stream] reporting error: {}", msg));
            (msg, raw)
        } else {
            let msg = format!("Process exited with code {:?}", status.code());
            gemini_debug(&format!("[stream] reporting exit error: {}", msg));
            (msg, String::new())
        };
        let _ = sender.send(StreamMessage::Error {
            message, stdout: stdout_raw, stderr: stderr_msg, exit_code: status.code(),
        });
        return Ok(());
    }

    if final_result.is_none() {
        gemini_debug(&format!("[stream] sending fallback Done (no result event): session_id={:?}", last_session_id));
        let _ = sender.send(StreamMessage::Done {
            result: String::new(),
            session_id: last_session_id,
        });
    }

    gemini_debug("=== gemini execute_command_streaming END ===");
    Ok(())
}
