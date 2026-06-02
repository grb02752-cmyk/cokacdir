# How to Configure Settings

## /silent

Toggles silent mode for the current chat. Default: **ON**.

- **ON** — Tool calls (Bash, Read, Edit, etc.) are hidden from the response. Only the AI's text output and errors are shown.
- **OFF** — Full tool call details are displayed, including commands run and file contents read.

Silent mode reduces message noise, especially in group chats.

---

## /debug

Toggles debug logging. Default: **OFF**.

When enabled, detailed logs are printed for Telegram API operations, AI service calls, and the cron scheduler. The flag is per-bot (stored once for the running bot token), so it affects every chat served by that bot — but it does not cross to other bots running on the same machine.

---

## /usechrome

Toggles the `--chrome` flag for the Claude CLI for the current chat. Default: **OFF** per chat.

- **ON** (`🌐 Chrome mode: ON (--chrome)`): Claude is invoked with `--chrome`, allowing it to drive a real Chrome browser session for tasks that require web interaction.
- **OFF** (`🌐 Chrome mode: OFF`): Claude runs without the flag.

The setting only takes effect when the active model is Claude. Other providers ignore this toggle.

---

## /effort

Sets the reasoning effort for the current chat's active provider. **Bot owner only.**

- **Claude** accepts: `low`, `medium`, `high`, `xhigh`, `max`
- **Codex** accepts: `minimal`, `low`, `medium`, `high`, `xhigh`
- `reset` / `clear` / `default` removes the override and falls back to the provider's own default

Examples:

```text
/effort high
/effort max
/effort reset
```

With no argument, `/effort` shows the current setting plus the accepted values for the active provider.

---

## /fast

Toggles Codex fast service tier for the current chat. **Bot owner only.**

- Supported only when the active provider is **Codex**
- **ON** passes `-c service_tier="fast"` to the Codex CLI
- **OFF** removes the override and uses the provider default

Examples:

```text
/fast
/fast on
/fast off
/fast status
```

---

## /greeting

Toggles the startup greeting style.

- **Compact**: `cokacdir started (v0.4.80, Claude)`
- **Full**: Includes session path, community links, GitHub URL, and update notices.

---

## /setpollingtime \<ms\>

Sets the stream update polling interval in milliseconds. This controls how frequently streaming responses and shell command output are checked and refreshed on screen.

```
/setpollingtime 3000
```

- **Minimum**: 1000ms
- **Recommended**: 1500ms or higher
- This no longer controls the Telegram API send/edit gap directly.
- Telegram API throttling is guarded separately by an internal per-chat minimum gap.
- Without arguments, shows the current value.

---

## /setapigap \<ms\>

Sets an explicit Telegram API minimum gap in milliseconds. This is the safety guard used between send/edit calls for the same chat.

```
/setapigap 2500
```

- **Minimum explicit override**: 1000ms
- **Recommended**: 2500ms or higher
- **Default config value**: `0` = auto mode, which follows the stream polling cadence with a 1000ms floor
- Raise this if you still see `RetryAfter` cooldowns during long streaming sessions.
- If the bot still hits `RetryAfter`, it now temporarily raises the effective gap automatically for that chat.
- Without arguments, shows the current effective gap.

---

## /tgstats

Shows the current Telegram runtime counters for the chat:

- Stream polling time
- Telegram API minimum gap
- Effective API minimum gap
- Adaptive backoff status
- `RetryAfter` hit count
- Last `RetryAfter` cooldown
- Edit failure count
- Send failure count
- Timeout failure count

Use this after long or throttled streams to see whether the bot is still hitting Telegram-side cooldowns.

---

## /envvars

Prints every environment variable currently visible to the bot process, sorted alphabetically. **Bot owner only.**

Useful for verifying that `~/.cokacdir/.env.json` loaded correctly, or checking whether a `COKAC_*` override is active.

> ⚠ **Security warning:** `/envvars` exposes **all** environment variables with no redaction — including API keys, tokens, and credentials. Telegram stores message history on its servers, so anything printed by this command is persisted until you delete the messages. Use it only for diagnostics, clear the response afterward, and **always use it in a 1:1 chat** — never in a group chat. When the owner runs `/envvars` in a group, the response is a normal group message that every member sees, regardless of the `/public` setting.

See [How to Configure Environment Variables](how-to-configure-environment-variables.md) for the full list of variables cokacdir reads (`COKAC_CLAUDE_PATH`, `COKAC_CODEX_PATH`, `COKAC_GEMINI_PATH`, `COKAC_OPENCODE_PATH`, `COKAC_FILE_ATTACH_THRESHOLD`, `COKACDIR_DEBUG`) and for the `~/.cokacdir/.env.json` auto-loader.

---

## /help

Displays the full command reference with all available commands and usage examples.
