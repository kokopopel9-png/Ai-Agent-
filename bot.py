#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Coding Agent  (Bot API 10.3)
=====================================

An autonomous coding agent that lives in Telegram.  You describe a goal in plain
language; the agent plans, explores the project, edits files, runs commands,
verifies its own work, fixes failures and reports back.

How it differs from a "tool bot"
  * No command zoo.  The LLM drives a tool-calling loop:
        Plan -> Explore -> Edit -> Run/Verify -> Self-correct -> Report
  * Shell, files, search, HTTP, screenshots, skills, memory are *tools the agent
    uses*, not commands you type.
  * Every task is checkpointed (shadow git repo) and can be undone with one tap.
  * Risky commands pause for approval via inline buttons.

Telegram features used
  * Rich Messages (sendRichMessage / editMessageText + rich_message)
        headings, checklists, tables, collapsible <details>, code blocks
  * Live progress with sendRichMessageDraft + <tg-thinking> + native Stop button
        (can_stop, stopped_message_generation update)
  * RichBlockButtons: coloured buttons INSIDE the message body (<tg-button-row>),
        auto-fallback to an inline keyboard if the server rejects them
  * "Cards" composed from rich blocks (heading + quotation + <cite> + button row)
  * Topics in private chats (message_thread_id is honoured everywhere)
  * Message reactions for "seen" acknowledgements
  * Automatic fallback to classic HTML messages if the Rich API is unavailable

Run
  pip install -r requirements.txt
  cp .env.example .env   # fill BOT_TOKEN, ADMIN_IDS and one LLM key
  python bot.py

SECURITY: the agent executes shell commands.  Path checks and approval prompts
reduce risk but are NOT a sandbox.  Run it in Docker / a dedicated VM and keep
ADMIN_IDS / ALLOWED_USER_IDS tight.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fnmatch
import html
import ipaddress
import json
import logging
import os
import random
import re
import secrets
import shlex
import shutil
import signal
import socket
import sqlite3
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

# ============================================================================
# CONFIGURATION  (everything comes from environment / .env — no secrets in code)
# ============================================================================


# Non-secret defaults only. Credentials are read from process environment variables.
_DIRECT_CONFIG = {
    "LLM_PROVIDER": "",
    "ANTHROPIC_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "GEMINI_API_KEY": "",
    "LLM_BASE_URL": "",
    "LLM_API_KEY": "",
    "ANTHROPIC_MODEL": "claude-sonnet-5-5",
    "OPENROUTER_MODEL": "openrouter/auto",
    "GEMINI_MODEL": "gemini-2.5-flash",
    "LLM_MODEL": "",
    "TAVILY_API_KEY": "",
    "BRAVE_API_KEY": "",
    "WORKSPACE_ROOT": "./workspace",
    "DATA_DIR": "./data",
    "AGENT_SHARE_HOME": "false",
    "MAX_STEPS": "150",
    "MAX_CALLS_PER_TURN": "25",
    "CHAT_MAX_STEPS": "4",
    "CMD_TIMEOUT": "120",
    "CMD_TIMEOUT_MAX": "900",
    "MAX_TOOL_OUTPUT": "7000",
    "MAX_CONCURRENT_TASKS": "3",
    "RATE_LIMIT_PER_MIN": "30",
    "DEFAULT_APPROVAL": "smart",
    "APPROVAL_TIMEOUT": "300",
    "BUTTON_STYLES": "true",
    "HISTORY_TURNS": "8",
    "CONTEXT_BUDGET_CHARS": "160000",
    "MAX_IMAGE_BYTES": "4000000",
    "TELEGRAM_API_URL": "https://api.telegram.org",
    "OAUTH_PORT": "8787",
    "OAUTH_PUBLIC_BASE": "",
    "AGY_COMMAND": "/home/ubuntu/.local/bin/agy",
    "OPENCODE_COMMAND": "/home/ubuntu/.opencode/bin/opencode",
}


def _load_dotenv(path: str = ".env") -> None:
    # Credentials should be injected by the hosting environment, not stored in source.
    return


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, _DIRECT_CONFIG.get(name, default))).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name).lower()
    return default if not v else v in ("1", "true", "yes", "on")


def _env_ids(name: str) -> set[int]:
    out: set[int] = set()
    for part in re.split(r"[,\s]+", _env(name)):
        if part.lstrip("-").isdigit():
            out.add(int(part))
    return out


class Cfg:
    BOT_TOKEN = _env("BOT_TOKEN")
    ADMIN_IDS = _env_ids("ADMIN_IDS") or {8353748526}
    ALLOWED_USER_IDS = _env_ids("ALLOWED_USER_IDS") or set(ADMIN_IDS)

    LLM_PROVIDER = _env("LLM_PROVIDER").lower()  # anthropic | openrouter | gemini | custom (auto if empty)
    ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
    GEMINI_API_KEY = _env("GEMINI_API_KEY")
    LLM_BASE_URL = _env("LLM_BASE_URL")  # custom OpenAI-compatible endpoint (…/v1)
    LLM_API_KEY = _env("LLM_API_KEY")
    ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    OPENROUTER_MODEL = _env("OPENROUTER_MODEL", "openrouter/auto")
    GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
    LLM_MODEL = _env("LLM_MODEL")  # model for the custom provider
    TAVILY_API_KEY = _env("TAVILY_API_KEY")  # optional: better web search
    BRAVE_API_KEY = _env("BRAVE_API_KEY")    # optional: alternative web search

    WORKSPACE_ROOT = Path(_env("WORKSPACE_ROOT", "./workspace")).resolve()
    DATA_DIR = Path(_env("DATA_DIR", "./data")).resolve()
    SHARE_HOME = _env_bool("AGENT_SHARE_HOME", False)  # let commands see the real $HOME

    MAX_STEPS = _env_int("MAX_STEPS", 150)  # one task may chain many terminal commands
    MAX_CALLS_PER_TURN = _env_int("MAX_CALLS_PER_TURN", 25)
    CHAT_MAX_STEPS = _env_int("CHAT_MAX_STEPS", 4)  # plain-chat replies (web search only)
    CMD_TIMEOUT = _env_int("CMD_TIMEOUT", 120)
    CMD_TIMEOUT_MAX = _env_int("CMD_TIMEOUT_MAX", 900)
    MAX_TOOL_OUTPUT = _env_int("MAX_TOOL_OUTPUT", 7000)
    MAX_CONCURRENT_TASKS = _env_int("MAX_CONCURRENT_TASKS", 3)
    RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 30)
    PUBLIC_ACCESS = False  # Private deployment: only pre-authorized Telegram IDs are allowed.
    DEFAULT_APPROVAL = _env("DEFAULT_APPROVAL", "smart").lower()  # ask | smart | auto
    APPROVAL_TIMEOUT = _env_int("APPROVAL_TIMEOUT", 300)
    PROGRESS_MODE = _env("PROGRESS_MODE", "message")  # message = live edited message (works everywhere) | draft = Telegram drafts
    BUTTON_STYLES = _env_bool("BUTTON_STYLES", True)  # coloured inline buttons (Bot API 9.4+)
    HISTORY_TURNS = _env_int("HISTORY_TURNS", 8)
    CONTEXT_BUDGET_CHARS = _env_int("CONTEXT_BUDGET_CHARS", 160_000)
    MAX_IMAGE_BYTES = max(100_000, min(_env_int("MAX_IMAGE_BYTES", 4_000_000), 20_000_000))
    TG_API = _env("TELEGRAM_API_URL", "https://api.telegram.org")  # local Bot API server supported
    PREVIEW_URL_TEMPLATE = _env("PREVIEW_URL_TEMPLATE", "https://{port}-ihaoqp0asxc8fyy20s6fu-ad872191.us5.manus.computer")
    OAUTH_PORT = _env_int("OAUTH_PORT", 8787)
    OAUTH_PUBLIC_BASE = _env("OAUTH_PUBLIC_BASE", "https://8787-iwt0n0i784hlf1mom77p3-c93e00d1.sg2.manus.computer")
    AGY_COMMAND = _env("AGY_COMMAND", "/home/ubuntu/.local/bin/agy")
    OPENCODE_COMMAND = _env("OPENCODE_COMMAND", "/home/ubuntu/.opencode/bin/opencode")


# The bot uses Antigravity's and OpenCode's own CLI login flows. No external
# google-oauth.json file is required by this single-file bot.
GOOGLE_OAUTH: dict[str, Any] = {}


# ============================================================================
# LOGGING  (token / key redaction — httpx would otherwise log the bot token)
# ============================================================================


class _Redact(logging.Filter):
    _pats = [
        re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            red = msg
            for p in self._pats:
                red = p.sub("<redacted>", red)
            if red != msg:
                record.msg, record.args = red, ()
        except Exception:
            pass
        return True


def _setup_logging() -> logging.Logger:
    Cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("agent")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = RotatingFileHandler(Cfg.DATA_DIR / "agent.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    for h in (ch, fh):
        h.setFormatter(fmt)
        h.addFilter(_Redact())
        lg.addHandler(h)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return lg


log = _setup_logging()

# Provider credentials configured from the Telegram Settings screen.  They are
# stored outside the source file and are never included in bot replies/logs.
_PROVIDER_FILE = Cfg.DATA_DIR / "providers.json"
_PROVIDER_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {}
try:
    if _PROVIDER_FILE.exists():
        raw = json.loads(_PROVIDER_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # Migrate the old global shape {provider: values} into a protected
            # legacy bucket. New credentials are always stored under user ID.
            if any(isinstance(v, dict) and ("key" in v or "url" in v or "model" in v) for v in raw.values()):
                _PROVIDER_OVERRIDES = {"legacy": {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}}
            else:
                _PROVIDER_OVERRIDES = {
                    str(uid): {str(name): dict(values) for name, values in providers.items() if isinstance(values, dict)}
                    for uid, providers in raw.items() if isinstance(providers, dict)
                }
except (OSError, ValueError, TypeError):
    log.warning("Could not load provider settings; starting with defaults")


def save_provider_override(uid: int, name: str, values: dict[str, str]) -> None:
    _PROVIDER_OVERRIDES.setdefault(str(uid), {})[name] = values
    Cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _PROVIDER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_PROVIDER_OVERRIDES, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(_PROVIDER_FILE)


def delete_provider_override(uid: int, name: str) -> None:
    _PROVIDER_OVERRIDES.setdefault(str(uid), {}).pop(name, None)
    if _PROVIDER_FILE.exists():
        _PROVIDER_FILE.write_text(json.dumps(_PROVIDER_OVERRIDES, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(_PROVIDER_FILE, 0o600)


def provider_override(uid: Optional[int], name: str) -> dict[str, str]:
    if uid is None:
        return {}
    return dict(_PROVIDER_OVERRIDES.get(str(uid), {}).get(name, {}))


MODEL_CATALOG: dict[str, list[tuple[str, bool]]] = {
    "anthropic": [("claude-sonnet-5-5", False), ("claude-opus-5-5", False), ("claude-haiku-4-5", False)],
    "openrouter": [("openrouter/free", True), ("google/gemini-2.5-flash:free", True), ("deepseek/deepseek-r1:free", True),
                   ("qwen/qwen3-coder:free", True), ("meta-llama/llama-4-maverick:free", True)],
    "gemini": [("gemini-2.5-flash", True), ("gemini-2.5-pro", False), ("gemini-2.0-flash", True)],
    "antigravity": [("gemini-3.8-flash", True), ("gemini-3.7-flash", True), ("gemini-3.6-flash", True),
                    ("gemini-3.1-pro-preview", False), ("claude-sonnet-5-5", False)],
    "opencode": [("opencode/gpt-6.1-sol", False), ("opencode/minimax-m2.1", False), ("opencode/gemini-3-pro", False)],
}


def provider_connected(name: str, uid: Optional[int] = None) -> bool:
    if uid is not None and provider_override(uid, name):
        return True
    return bool(provider_config(name, uid=uid)) if "provider_config" in globals() else False

# ============================================================================
# SMALL UTILITIES
# ============================================================================


def esc(s: Any) -> str:
    return html.escape(str(s), quote=False)


# Telegram Premium custom emoji used by the bot UI.  Rich Messages render these
# by ID; classic HTML fallback keeps the visible display character.
PREMIUM_EMOJI = {
    "🪬": "5354878938759867626", "🪽": "5355086127982225076", "🌟": "5357377583228958443",
    "🫀": "5354906903291929674", "🪟": "5355124129852860696", "🐱": "5357186981170291800",
    "🔱": "5355272473728296094", "🔨": "5355158292022731818", "🥇": "5357496669787171532",
    "📔": "5354919938517675490", "🧩": "5354965499530751469", "🧭": "5359832728269268910",
    "👆": "5357153686583815745", "🗝": "5355000147031927044", "👀": "5355267976897537716",
    "💎": "5357324596217422686", "🏆": "5355020243183903114", "🦪": "5355076000449340529",
    "🕓": "5359733999856036575", "🔍": "5354966105121139213", "🍾": "5357585966452223209",
    "🪣": "5357256443676373360", "🧠": "5354848951298204993", "🌻": "5375472692763862117",
    "🚀": "5355258523674522418", "🏹": "5357047300243891102",
    # Remaining UI symbols reuse the closest available Premium artwork while
    # preserving their original display character in fallback messages.
    "❌": "5355158292022731818", "✅": "5357496669787171532",
    "📁": "5355124129852860696", "⚠": "5355158292022731818", "⚙": "5354965499530751469",
    "🌐": "5355124129852860696", "📋": "5354919938517675490", "📂": "5355124129852860696",
    "📖": "5354919938517675490", "📦": "5357256443676373360", "⚡": "5355258523674522418",
    "🗑": "5355158292022731818", "📸": "5359832728269268910", "❓": "5355267976897537716",
    "🔧": "5355158292022731818", "🔒": "5355000147031927044", "🧪": "5354965499530751469",
    "🧾": "5354919938517675490", "📊": "5359832728269268910", "🔐": "5355000147031927044",
    "🔌": "5355272473728296094", "🤖": "5357186981170291800", "✔": "5357496669787171532",
    "☑": "5357496669787171532", "📝": "5354919938517675490",
    "✏": "5355158292022731818", "💻": "5355158292022731818", "📤": "5357047300243891102",
    "💾": "5355000147031927044", "📥": "5357256443676373360", "🔢": "5354919938517675490",
    "📌": "5357153686583815745", "🧹": "5357256443676373360", "✨": "5357377583228958443",
    "🆕": "5357377583228958443", "🖥": "5355124129852860696",
    "👥": "5357186981170291800", "📈": "5359832728269268910", "📢": "5355258523674522418",
}

# ---- Premium artwork supplied by the owner, assigned by role -------------------
_E_BLADE, _E_TOOLS, _E_DEMON, _E_SUN = "5357423891566341596", "5354905928334354683", "5357102516343450821", "5377465656373456180"
_E_ROCK, _E_DRAGON, _E_CORK1, _E_CORK2 = "5375205562977919364", "5357336656485590934", "5354936512796469608", "5357405393142196185"
PREMIUM_EMOJI.update({
    # the artwork's own glyphs (usable directly in any text)
    "🗡": _E_BLADE, "⚒": _E_TOOLS, "👹": _E_DEMON, "🌻": _E_SUN, "🪨": _E_ROCK, "🐉": _E_DRAGON, "🍾": _E_CORK1,
    # stop / danger / delete
    "⏹": _E_BLADE, "■": _E_BLADE, "🗑": _E_BLADE,
    # tools / build / edit
    "🔧": _E_TOOLS, "🛠": _E_TOOLS, "💻": _E_TOOLS, "✏": _E_TOOLS,
    # errors / warnings
    "❌": _E_DEMON, "⚠": _E_DEMON,
    # success
    "✅": _E_SUN, "✔": _E_SUN, "☑": _E_SUN,
    # files / sessions / workspace
    "📁": _E_ROCK, "📂": _E_ROCK, "🖥": _E_ROCK,
    # brand / engine / status
    "🤖": _E_DRAGON, "📡": _E_DRAGON, "🎛": _E_DRAGON,
    # new / highlights
    "🆕": _E_CORK1, "✨": _E_CORK1,
    # time / waiting
    "⏱": _E_CORK2, "⏳": _E_CORK2, "⏸": _E_CORK2,
    # previously unmapped
    "🔎": "5354966105121139213", "📄": "5354919938517675490", "📱": "5357256443676373360", "▶": "5355258523674522418",
})


def premiumize(h: str) -> str:
    """Convert the bot's supported UI emoji to Telegram custom emoji tags."""
    for display, emoji_id in PREMIUM_EMOJI.items():
        tag = f'<tg-emoji emoji-id="{emoji_id}">{display}</tg-emoji>'
        h = h.replace(display + "\ufe0f", tag)
        h = h.replace(display, tag)
    return h


def demote_premium(h: str) -> str:
    """Return the visible glyph when a server rejects a custom emoji ID."""
    return re.sub(r'<tg-emoji\s+emoji-id="\d+">(.*?)</tg-emoji>', r"\1", h, flags=re.S)


def _premium_button(b: JSON) -> JSON:
    """Classic keyboard buttons cannot contain <tg-emoji>; Bot API 9.4 gives them
    `icon_custom_emoji_id` instead. Move the leading emoji of the label into it."""
    if "icon_custom_emoji_id" in b:
        return b
    text = b.get("text") or ""
    for glyph, emoji_id in PREMIUM_EMOJI.items():
        for cand in (glyph + "\ufe0f", glyph):
            if text.startswith(cand):
                rest = text[len(cand):].strip()
                if not rest:
                    return b
                nb = dict(b)
                nb["icon_custom_emoji_id"] = emoji_id
                nb["text"] = rest
                return nb
    return b


def premium_markup(markup: Optional[JSON]) -> Optional[JSON]:
    """Return a copy of an inline keyboard whose leading emoji are Premium icons."""
    rows = (markup or {}).get("inline_keyboard") or []
    if not rows:
        return markup
    return {"inline_keyboard": [[_premium_button(b) for b in row] for row in rows]}


def short(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def trunc(s: str, limit: int, head: float = 0.3) -> str:
    """Keep head and tail (errors usually live at the end of a log)."""
    if len(s) <= limit:
        return s
    h = int(limit * head)
    t = limit - h
    return f"{s[:h]}\n…[{len(s) - limit} chars omitted]…\n{s[-t:]}"


def dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec}s" if sec < 60 else f"{sec // 60}m{sec % 60:02d}s"


def extract_agy_login_url(output: str) -> str:
    """Extract and unwrap agy's Google URL from terminal-formatted output."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    start = re.search(r"https?://(?:accounts\.google\.com|accounts\.googleusercontent\.com)/", clean, re.I)
    if not start:
        return ""
    tail = clean[start.start():]
    # In TTY mode agy emits the URL as an OSC-8 hyperlink.  Stop at the first
    # BEL terminator; otherwise repeated screen redraws can be concatenated
    # and produce invalid URLs with duplicate client_id query parameters.
    osc_end = tail.find("\x07")
    if osc_end >= 0:
        candidate = tail[:osc_end]
    else:
        # Non-TTY output may hard-wrap the URL before its next status line.
        markers = ("Waiting for authentication", "Or, paste the authorization code")
        end = min((tail.find(marker) for marker in markers if tail.find(marker) >= 0), default=-1)
        candidate = tail[:end] if end >= 0 else tail.splitlines()[0]
    candidate = re.sub(r"\s+", "", candidate).rstrip(".,;:)]}>")
    return candidate if re.fullmatch(r"https?://[^\s<>\"']+", candidate, re.I) else ""


def agy_auth_log_status(since: float) -> tuple[str, str]:
    """Read agy's file log because OAuth results are not always printed to the TTY."""
    log_dir = Path.home() / ".gemini" / "antigravity-cli" / "log"
    if not log_dir.exists():
        return "", ""
    for path in sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < since - 5:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            low = text.lower()
            if "account ineligible" in low or "verify your account to continue" in low:
                lines = [ln.strip() for ln in text.splitlines() if "ineligible" in ln.lower() or "verify your account" in ln.lower()]
                return "failure", "\n".join(lines[-3:])
            if "authenticated successfully as" in low or "authentication completed successfully" in low:
                lines = [ln.strip() for ln in text.splitlines() if "authenticated successfully" in ln.lower() or "authentication completed" in ln.lower()]
                return "success", "\n".join(lines[-3:])
    return "", ""


def kfmt(n: int) -> str:
    return str(n) if n < 1000 else f"{n / 1000:.1f}k"


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def detect_image_mime(data: bytes) -> str:
    """Detect supported image formats from their signatures, not user-supplied MIME types."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        raise ToolError("GIF images are not supported yet. Please send the image as JPG, PNG, or WebP.")
    raise ToolError("Unsupported or invalid image. Please send a JPG, PNG, or WebP image.")


def user_content(text: str, images: Optional[list[dict]] = None) -> Any:
    """Build an OpenAI-compatible multimodal user message."""
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text or "Please inspect the attached image."}]
    for image in images:
        mime = image.get("mime_type", "")
        data = image.get("data", "")
        if mime not in ("image/jpeg", "image/png", "image/webp") or not data:
            continue
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return parts


def safe_name(s: str, default: str = "default", maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip(".-_")[:maxlen]
    return s or default


class ToolError(Exception):
    """Raised by tools for model-visible errors."""


JSON = dict[str, Any]

# ============================================================================
# TELEGRAM RENDERING  (Rich HTML <-> classic HTML, Markdown -> Rich HTML)
# ============================================================================

RICH_LIMIT = 30_000  # Rich Messages allow ~32k chars; keep headroom
CLASSIC_LIMIT = 3_900

_CLASSIC_KEEP = r"(?:b|i|u|s|code|pre|a|blockquote|tg-spoiler|tg-emoji)"


def rich_to_classic(h: str) -> str:
    """Degrade Rich HTML to Telegram's classic HTML subset (fallback path)."""
    s = h
    s = re.sub(r"<tg-button-row[^>]*>.*?</tg-button-row>", "", s, flags=re.S)
    s = re.sub(r"<cite>(.*?)</cite>", r"\n<i>\1</i>", s, flags=re.S)
    s = re.sub(r"<tg-thinking>(.*?)</tg-thinking>", r"<i>\1</i>\n", s, flags=re.S)
    s = re.sub(
        r"<details[^>]*>\s*<summary>(.*?)</summary>(.*?)</details>",
        lambda m: f"\n<b>▸ {m.group(1)}</b>\n{m.group(2)}\n",
        s,
        flags=re.S,
    )
    s = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n<b>\1</b>\n", s, flags=re.S)
    s = re.sub(r"<(strong)>(.*?)</\1>", r"<b>\2</b>", s, flags=re.S)
    s = re.sub(r"<(em)>(.*?)</\1>", r"<i>\2</i>", s, flags=re.S)
    s = re.sub(r"<(ins)>(.*?)</\1>", r"<u>\2</u>", s, flags=re.S)
    s = re.sub(r"<(del|strike)>(.*?)</\1>", r"<s>\2</s>", s, flags=re.S)
    s = re.sub(r"<input[^>]*\bchecked\b[^>]*>", "☑ ", s)
    s = re.sub(r"<input[^>]*>", "☐ ", s)
    s = re.sub(r"</?(?:ul|ol)[^>]*>", "", s)
    s = re.sub(r"<li[^>]*>", "• ", s)
    s = s.replace("</li>", "\n")
    s = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " │ ", s)
    s = re.sub(r"</tr>", "\n", s)
    s = re.sub(r"</?(?:table|tr|td|th|thead|tbody)[^>]*>", "", s)
    s = re.sub(r"<hr\s*/?>", "\n──────────\n", s)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<p[^>]*>", "", s)
    s = s.replace("</p>", "\n\n")
    s = re.sub(rf"<(?!/?{_CLASSIC_KEEP}\b)[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


_TAG_RE = re.compile(r"<(/?)([a-zA-Z][\w-]*)([^>]*)>")
_VOID = {"br", "hr", "input", "img"}


def split_html(s: str, limit: int = CLASSIC_LIMIT) -> list[str]:
    """Split HTML on line boundaries keeping tags balanced across chunks."""
    if len(s) <= limit:
        return [s]

    def scan(line: str, stack: list[tuple[str, str]]) -> list[tuple[str, str]]:
        st = list(stack)
        for m in _TAG_RE.finditer(line):
            closing, name = m.group(1), m.group(2).lower()
            if name in _VOID or m.group(0).endswith("/>"):
                continue
            if closing:
                for i in range(len(st) - 1, -1, -1):
                    if st[i][0] == name:
                        del st[i:]
                        break
            else:
                st.append((name, m.group(0)))
        return st

    def closers(st: list[tuple[str, str]]) -> str:
        return "".join(f"</{n}>" for n, _ in reversed(st))

    def openers(st: list[tuple[str, str]]) -> str:
        return "".join(t for _, t in st)

    lines: list[str] = []
    for ln in s.split("\n"):  # hard-split monstrous single lines
        while len(ln) > limit - 300:
            cut = ln.rfind(" ", 0, limit - 300)
            cut = cut if cut > 0 else limit - 300
            lines.append(ln[:cut])
            ln = ln[cut:]
        lines.append(ln)

    chunks: list[str] = []
    cur = ""
    stack: list[tuple[str, str]] = []
    for ln in lines:
        after = scan(ln, stack)
        if cur and len(cur) + len(ln) + 1 + len(closers(after)) > limit:
            chunks.append(cur + closers(stack))
            cur = openers(stack)
        cur = f"{cur}\n{ln}" if cur else ln
        stack = after
    if cur.strip():
        chunks.append(cur + closers(stack))
    return chunks


# ---- Rich buttons (RichMessageButton / RichBlockButtons) & cards ------------
# Bot API 10.3: buttons live INSIDE the message body as <tg-button-row><tg-button …>.
# There is no "RichCard" type in the Bot API; a card is composed from real blocks
# (heading + block quotation with key/value lines + <cite> footer + button row).


def _embeddable(markup: Optional[JSON]) -> bool:
    rows = (markup or {}).get("inline_keyboard") or []
    return bool(rows) and all("callback_data" in b and len(b) <= 3 for r in rows for b in r)


def rich_buttons(markup: Optional[JSON]) -> str:
    """InlineKeyboardMarkup (callback buttons) -> <tg-button-row> blocks, max 8 buttons per row."""
    out = []
    for row in (markup or {}).get("inline_keyboard") or []:
        for i in range(0, len(row), 8):
            cells = []
            for b in row[i:i + 8]:
                st = f' style="{b["style"]}"' if b.get("style") in ("primary", "success", "danger") else ""
                # Rich buttons can contain Rich HTML, so use the same Premium
                # custom-emoji renderer as message bodies. Classic keyboard
                # fallback still uses the plain display character.
                cells.append(
                    f'<tg-button type="callback_data" data="{html.escape(b["callback_data"], quote=True)}"{st}>'
                    f"{premiumize(esc(b['text']))}</tg-button>"
                )
            out.append(f"<tg-button-row>{''.join(cells)}</tg-button-row>")
    return "".join(out)


def card(title: str, icon: str = "", rows: Optional[list[tuple[str, str]]] = None, note: str = "", foot: str = "") -> str:
    """Composed 'card': heading + quotation block of key/value lines (+ note, + cite footer).
    Values and note are already-escaped HTML."""
    h = f"<h4>{icon} {esc(title)}</h4>" if icon else f"<h4>{esc(title)}</h4>"
    lines = [f"<b>{esc(k)}</b> {v}" for k, v in (rows or [])]
    if note:
        lines.append(note)
    if lines:
        cite = f"<cite>{foot}</cite>" if foot else ""
        h += f"<blockquote>{'<br>'.join(lines)}{cite}</blockquote>"
    return h


# ---- Markdown (LLM output) -> Rich HTML -------------------------------------

_LI_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TBL_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


def _inline(s: str) -> str:
    stash: list[str] = []

    def keep(h: str) -> str:
        stash.append(h)
        return f"\x00{len(stash) - 1}\x00"

    s = s.replace("\x00", "")
    s = re.sub(r"`([^`\n]+)`", lambda m: keep(f"<code>{esc(m.group(1))}</code>"), s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: f"<b>{m.group(1) or m.group(2)}</b>", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"(?<![\w_])_(?![\s_])(.+?)(?<![\s_])_(?![\w_])", r"<i>\1</i>", s)
    s = re.sub(r"~~(.+?)~~", r"<s>\1</s>", s)
    s = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2).replace(chr(34), "%22")}">{m.group(1)}</a>',
        s,
    )
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)


def _list_block(lines: list[str], i: int, base: int) -> tuple[str, int]:
    first = _LI_RE.match(lines[i])
    tag = "ol" if first and first.group(2)[0].isdigit() else "ul"
    out = [f"<{tag}>"]
    while i < len(lines):
        m = _LI_RE.match(lines[i])
        if not m:
            break
        ind = len(m.group(1).expandtabs(4))
        if ind < base:
            break
        if ind > base:
            sub, i = _list_block(lines, i, ind)
            if out[-1].endswith("</li>"):
                out[-1] = out[-1][:-5] + sub + "</li>"
            else:
                out.append(f"<li>{sub}</li>")
            continue
        text = m.group(3)
        cb = re.match(r"^\[( |x|X)\]\s+(.*)$", text)
        if cb:
            chk = " checked" if cb.group(1) in "xX" else ""
            item = f'<input type="checkbox"{chk}/>{_inline(cb.group(2))}'
        else:
            item = _inline(text)
        out.append(f"<li>{item}</li>")
        i += 1
    out.append(f"</{tag}>")
    return "".join(out), i


def _row_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def md_to_html(md: str) -> str:
    """Convert (GitHub-flavoured) Markdown from an LLM into Rich Message HTML."""
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)

    def starts_block(idx: int) -> bool:
        ln = lines[idx]
        return bool(
            _FENCE_RE.match(ln)
            or re.match(r"^#{1,6}\s+", ln)
            or _HR_RE.match(ln)
            or _LI_RE.match(ln)
            or ln.lstrip().startswith(">")
            or ("|" in ln and idx + 1 < n and _TBL_SEP.match(lines[idx + 1]))
        )

    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        fm = _FENCE_RE.match(ln)
        if fm:
            fence, lang = fm.group(1), fm.group(2)
            i += 1
            buf: list[str] = []
            while i < n and not re.match(rf"^\s*{re.escape(fence[0])}{{{len(fence)},}}\s*$", lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{esc(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{esc(chr(10).join(buf))}</code></pre>")
            continue
        if _HR_RE.match(ln):
            out.append("<hr/>")
            i += 1
            continue
        hm = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", ln)
        if hm:
            lvl = {1: 2, 2: 3}.get(len(hm.group(1)), 4)
            out.append(f"<h{lvl}>{_inline(hm.group(2))}</h{lvl}>")
            i += 1
            continue
        if "|" in ln and i + 1 < n and _TBL_SEP.match(lines[i + 1]):
            head = _row_cells(ln)
            i += 2
            rows = ["<tr>" + "".join(f"<td><b>{_inline(c)}</b></td>" for c in head) + "</tr>"]
            while i < n and "|" in lines[i] and lines[i].strip():
                cells = _row_cells(lines[i])
                cells = (cells + [""] * len(head))[: len(head)]
                rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("<table>" + "".join(rows) + "</table>")
            continue
        if ln.lstrip().startswith(">"):
            q: list[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                q.append(_inline(re.sub(r"^\s*>\s?", "", lines[i])))
                i += 1
            out.append("<blockquote>" + "<br>".join(q) + "</blockquote>")
            continue
        if _LI_RE.match(ln):
            base = len(_LI_RE.match(ln).group(1).expandtabs(4))  # type: ignore[union-attr]
            blk, i = _list_block(lines, i, base)
            out.append(blk)
            continue
        para = [ln]
        i += 1
        while i < n and lines[i].strip() and not starts_block(i):
            para.append(lines[i])
            i += 1
        out.append("<p>" + "<br>".join(_inline(p.strip()) for p in para) + "</p>")
    return "".join(out)


# ============================================================================
# TELEGRAM CLIENT  (raw HTTP — independent of any bot-framework version, so the
# newest Bot API methods are available the day Telegram ships them)
# ============================================================================


class TGError(Exception):
    def __init__(self, method: str, code: int, desc: str, params: Optional[dict] = None):
        super().__init__(f"{method}: [{code}] {desc}")
        self.method, self.code, self.desc, self.params = method, code, desc or "", params or {}


NOKB: JSON = {"inline_keyboard": []}


def btn(
    text: str,
    data: Optional[str] = None,
    *,
    url: Optional[str] = None,
    style: Optional[str] = None,
    copy: Optional[str] = None,
) -> JSON:
    b: JSON = {"text": text}
    if data is not None:
        b["callback_data"] = data[:64]
    if url:
        b["url"] = url
    if copy:
        b["copy_text"] = {"text": copy}
    if style and Cfg.BUTTON_STYLES:
        b["style"] = style  # "primary" (blue) | "success" (green) | "danger" (red)
    return b


def kb(*rows: list[JSON]) -> JSON:
    return {"inline_keyboard": [list(r) for r in rows if r]}


class TG:
    def __init__(self, token: str):
        self.base = f"{Cfg.TG_API}/bot{token}"
        self.file_base = f"{Cfg.TG_API}/file/bot{token}"
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(75.0, connect=15.0))
        self.rich_ok = True  # temporarily disabled only when a Rich API call is rejected
        self.rich_retry_at = 0.0  # retry Rich after a fallback; do not stay in normal-emoji mode
        self.draft_ok = True
        self.rich_buttons_ok = True  # buttons inside the rich body (RichBlockButtons)
        self._btn_fail = 0
        self.me: JSON = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def call(self, method: str, _retry: bool = True, **params: Any) -> Any:
        body = {k: v for k, v in params.items() if v is not None}
        for attempt in range(4):
            try:
                r = await self.http.post(f"{self.base}/{method}", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == 3:
                    raise TGError(method, 0, f"network error: {type(e).__name__}")
                await asyncio.sleep(1 + attempt)
                continue
            try:
                data = r.json()
            except ValueError:
                data = {"ok": False, "error_code": r.status_code, "description": r.text[:200]}
            if data.get("ok"):
                return data["result"]
            code = int(data.get("error_code", r.status_code))
            desc = data.get("description", "")
            prm = data.get("parameters", {})
            if code == 429 and _retry and attempt < 3:
                await asyncio.sleep(min(int(prm.get("retry_after", 2)), 30) + 0.2)
                continue
            raise TGError(method, code, desc, prm)
        raise TGError(method, 0, "retries exhausted")

    async def upload(self, method: str, field_name: str, path: Path, **params: Any) -> Any:
        data = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            for k, v in params.items()
            if v is not None
        }
        with open(path, "rb") as fh:
            r = await self.http.post(
                f"{self.base}/{method}", data=data, files={field_name: (path.name, fh)}, timeout=120
            )
        j = r.json()
        if not j.get("ok"):
            raise TGError(method, int(j.get("error_code", 0)), j.get("description", ""))
        return j["result"]

    async def download(self, file_id: str, dest: Path) -> None:
        info = await self.call("getFile", file_id=file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with self.http.stream("GET", f"{self.file_base}/{info['file_path']}") as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)

    async def download_bytes(self, file_id: str, max_bytes: int) -> bytes:
        """Download a Telegram attachment into memory with a strict size cap."""
        info = await self.call("getFile", file_id=file_id)
        if int(info.get("file_size") or 0) > max_bytes:
            raise ToolError(f"Image is too large; the limit is {max_bytes // 1_000_000} MB.")
        chunks: list[bytes] = []
        total = 0
        try:
            async with self.http.stream("GET", f"{self.file_base}/{info['file_path']}") as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ToolError(f"Image is too large; the limit is {max_bytes // 1_000_000} MB.")
                    chunks.append(chunk)
        except httpx.HTTPError as e:
            raise ToolError("Telegram could not download that image. Please try sending it again.") from e
        return b"".join(chunks)

    # ---- high level sends ---------------------------------------------------

    def _variants(self, h: str, markup: Optional[JSON]) -> list[tuple[str, Optional[JSON], bool]]:
        """Rich attempts, best first: buttons embedded in the body, then classic keyboard under the message."""
        v: list[tuple[str, Optional[JSON], bool]] = []
        if self.rich_buttons_ok and _embeddable(markup):
            v.append((h + rich_buttons(markup), None, True))
        pm = premium_markup(markup)
        if pm is not markup and pm != markup:
            v.append((h, pm, False))  # keyboard buttons with Premium icon_custom_emoji_id
        v.append((h, markup, False))
        return [x for x in v if len(x[0]) <= RICH_LIMIT]

    def _note_button_result(self, embedded: bool, emb_failed: bool) -> None:
        """Called after a successful rich send. Disable in-body buttons only if they fail while the keyboard variant works."""
        if embedded:
            self._btn_fail = 0
        elif emb_failed:
            self._btn_fail += 1
            if self._btn_fail >= 3 and self.rich_buttons_ok:
                self.rich_buttons_ok = False
                log.warning("In-body rich buttons keep failing — using inline keyboards under the message instead.")

    async def send(
        self,
        chat_id: int,
        h: str,
        *,
        thread: Optional[int] = None,
        markup: Optional[JSON] = None,
        reply_to: Optional[int] = None,
        silent: bool = False,
    ) -> Optional[int]:
        """Send Rich HTML and keep retrying Rich after temporary fallback failures."""
        # Provider replies and normal bot replies must use the same Premium Emoji path.
        # A previous transient Rich failure must not permanently downgrade this bot.
        if not self.rich_ok and time.monotonic() >= self.rich_retry_at:
            self.rich_ok = True
        h = premiumize(h)
        rp = {"message_id": reply_to, "allow_sending_without_reply": True} if reply_to else None
        if self.rich_ok:
            emb_failed = False
            for body, mk, embedded in self._variants(h, markup):
                try:
                    r = await self.call(
                        "sendRichMessage",
                        chat_id=chat_id,
                        message_thread_id=thread,
                        rich_message={"html": body},
                        reply_markup=mk,
                        reply_parameters=rp,
                        disable_notification=True if silent else None,
                    )
                    self._note_button_result(embedded, emb_failed)
                    return r.get("message_id")
                except TGError as e:
                    if e.code in (401, 403):
                        raise
                    if e.code == 404 or "method not found" in e.desc.lower():
                        # Do not permanently downgrade: provider setup and transient API
                        # errors can make one request fail while Rich works afterwards.
                        self.rich_ok = False
                        self.rich_retry_at = time.monotonic() + 15.0
                        log.warning("Rich API unavailable for now — using classic HTML temporarily.")
                        break
                    if "RICH_MESSAGE_EMOJI_INVALID" in e.desc and "<tg-emoji" in h:
                        log.info("Rich server rejected a custom emoji ID; retrying RichMessage with the visible glyph")
                        retry = await self.call(
                            "sendRichMessage", chat_id=chat_id, message_thread_id=thread,
                            rich_message={"html": demote_premium(body)}, reply_markup=mk,
                            reply_parameters=rp, disable_notification=True if silent else None,
                        )
                        return retry.get("message_id")
                    emb_failed = emb_failed or embedded
                    log.warning("sendRichMessage rejected (%s%s)", e.desc[:100], " — retrying without in-body buttons" if embedded else "")
        return await self._send_classic(chat_id, h, thread, markup, rp, silent)

    async def _send_classic(self, chat_id, h, thread, markup, rp, silent) -> Optional[int]:
        chunks = split_html(rich_to_classic(h))
        last: Optional[int] = None
        for idx, ch in enumerate(chunks):
            is_last = idx == len(chunks) - 1
            args = dict(
                chat_id=chat_id,
                message_thread_id=thread,
                reply_markup=markup if is_last else None,
                reply_parameters=rp if idx == 0 else None,
                disable_notification=True if silent else None,
                link_preview_options={"is_disabled": True},
            )
            try:
                if is_last and markup:
                    try:
                        r = await self.call("sendMessage", text=ch, parse_mode="HTML",
                                            **{**args, "reply_markup": premium_markup(markup)})
                    except TGError as e:
                        if e.code in (401, 403):
                            raise
                        r = await self.call("sendMessage", text=ch, parse_mode="HTML", **args)
                else:
                    r = await self.call("sendMessage", text=ch, parse_mode="HTML", **args)
            except TGError as e:
                if e.code in (401, 403):
                    raise
                r = await self.call("sendMessage", text=strip_tags(ch)[:4096], **args)
            last = r.get("message_id")
        return last

    async def edit(self, chat_id: int, mid: int, h: str, markup: Optional[JSON] = None) -> bool:
        if not self.rich_ok and time.monotonic() >= self.rich_retry_at:
            self.rich_ok = True
        h = premiumize(h)
        if self.rich_ok:
            emb_failed = False
            for body, mk, embedded in self._variants(h, markup):
                try:
                    await self.call("editMessageText", chat_id=chat_id, message_id=mid, rich_message={"html": body}, reply_markup=mk)
                    self._note_button_result(embedded, emb_failed)
                    return True
                except TGError as e:
                    if "not modified" in e.desc.lower():
                        return True
                    if e.code == 404 or "method not found" in e.desc.lower():
                        self.rich_ok = False
                        self.rich_retry_at = time.monotonic() + 15.0
                    if "RICH_MESSAGE_EMOJI_INVALID" in e.desc and "<tg-emoji" in h:
                        log.info("Rich server rejected a custom emoji ID; retrying edit with the visible glyph")
                        await self.call(
                            "editMessageText", chat_id=chat_id, message_id=mid,
                            rich_message={"html": demote_premium(body)}, reply_markup=mk,
                        )
                        return True
                    emb_failed = emb_failed or embedded
                    log.debug("rich edit failed: %s", e.desc[:120])
        text = split_html(rich_to_classic(h))[0]
        try:
            for mk in ((premium_markup(markup), markup) if markup and markup.get("inline_keyboard") else (markup,)):
                try:
                    await self.call(
                        "editMessageText", chat_id=chat_id, message_id=mid, text=text, parse_mode="HTML",
                        reply_markup=mk, link_preview_options={"is_disabled": True},
                    )
                    return True
                except TGError as e:
                    if "not modified" in e.desc.lower():
                        return True
                    if mk is markup:
                        raise
                    log.info("Premium button icons rejected on edit (%s); retrying plain", e.desc[:80])
        except TGError as e:
            if "not modified" in e.desc.lower():
                return True
            log.debug("classic edit failed: %s", e.desc[:120])
            return False

    async def delete(self, chat_id: int, mid: int) -> None:
        with contextlib.suppress(TGError):
            await self.call("deleteMessage", chat_id=chat_id, message_id=mid)

    async def answer_cb(self, cb_id: str, text: str = "", alert: bool = False) -> None:
        with contextlib.suppress(TGError):
            await self.call("answerCallbackQuery", callback_query_id=cb_id, text=text[:190] or None, show_alert=alert or None)

    async def react(self, chat_id: int, mid: int, emoji: str = "👀") -> None:
        with contextlib.suppress(TGError):
            await self.call(
                "setMessageReaction", chat_id=chat_id, message_id=mid, reaction=[{"type": "emoji", "emoji": emoji}]
            )

    async def typing(self, chat_id: int, thread: Optional[int] = None) -> None:
        with contextlib.suppress(TGError):
            await self.call("sendChatAction", _retry=False, chat_id=chat_id, action="typing", message_thread_id=thread)

    async def send_file(self, chat_id: int, path: Path, caption: str = "", thread: Optional[int] = None) -> None:
        is_img = path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and path.stat().st_size < 10_000_000
        method, fld = ("sendPhoto", "photo") if is_img else ("sendDocument", "document")
        await self.upload(
            method, fld, path, chat_id=chat_id, caption=caption[:1000] or None, message_thread_id=thread
        )

    # ---- drafts (live progress) -------------------------------------------

    async def draft_rich(self, chat_id: int, draft_id: int, h: str, thread: Optional[int], can_stop: bool = True):
        return await self.call(
            "sendRichMessageDraft",
            _retry=False,
            chat_id=chat_id,
            draft_id=draft_id,
            message_thread_id=thread,
            can_stop=can_stop,
            rich_message={"html": premiumize(h)},
        )

    async def draft_plain(self, chat_id: int, draft_id: int, text: str, thread: Optional[int], can_stop: bool = True):
        return await self.call(
            "sendMessageDraft",
            _retry=False,
            chat_id=chat_id,
            draft_id=draft_id,
            message_thread_id=thread,
            text=text[:4096],
            parse_mode="HTML",
            can_stop=can_stop,
        )


# ============================================================================
# DATABASE  (users, per-project chat history, task log)
# ============================================================================


class DB:
    USER_FIELDS = {"name", "role", "banned", "project", "approval", "provider", "model"}

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(str(path), check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.c.executescript(
                """
                CREATE TABLE IF NOT EXISTS users(
                    uid INTEGER PRIMARY KEY, name TEXT, role TEXT DEFAULT 'user', banned INTEGER DEFAULT 0,
                    project TEXT DEFAULT 'default', approval TEXT, provider TEXT, model TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, uid INTEGER, project TEXT, role TEXT, content TEXT, ts REAL);
                CREATE INDEX IF NOT EXISTS h_idx ON history(uid, project, id);
                CREATE TABLE IF NOT EXISTS tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, uid INTEGER, project TEXT, goal TEXT, status TEXT,
                    started REAL, ended REAL, steps INTEGER DEFAULT 0, cp_start TEXT, cp_end TEXT,
                    tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0);
                """
            )
            self.c.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.c.execute(sql, args)
            rows = cur.fetchall()
            self.c.commit()
            return rows

    def seed_users(self) -> None:
        for uid in Cfg.ADMIN_IDS:
            self._q("INSERT OR IGNORE INTO users(uid,role,created) VALUES(?,?,?)", (uid, "admin", time.time()))
            self._q("UPDATE users SET role='admin' WHERE uid=?", (uid,))
        for uid in Cfg.ALLOWED_USER_IDS:
            self._q("INSERT OR IGNORE INTO users(uid,role,created) VALUES(?,?,?)", (uid, "user", time.time()))

    def user(self, uid: int) -> Optional[dict]:
        r = self._q("SELECT * FROM users WHERE uid=?", (uid,))
        return dict(r[0]) if r else None

    def is_allowed(self, uid: int) -> bool:
        if uid in Cfg.ADMIN_IDS:
            return True
        u = self.user(uid)
        if u and u["banned"]:
            return False
        if Cfg.PUBLIC_ACCESS:
            return True
        return bool(u and not u["banned"])

    def is_admin(self, uid: int) -> bool:
        return uid in Cfg.ADMIN_IDS

    def add_user(self, uid: int, name: str = "") -> None:
        self._q("INSERT OR IGNORE INTO users(uid,name,role,created) VALUES(?,?,?,?)", (uid, name, "user", time.time()))
        self._q("UPDATE users SET banned=0 WHERE uid=?", (uid,))

    def touch_name(self, uid: int, name: str) -> None:
        self._q("UPDATE users SET name=? WHERE uid=? AND (name IS NULL OR name!=?)", (name, uid, name))

    def set(self, uid: int, field_name: str, value: Any) -> None:
        if field_name not in self.USER_FIELDS:
            raise ValueError(field_name)
        self._q(f"UPDATE users SET {field_name}=? WHERE uid=?", (value, uid))

    def users(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM users ORDER BY created")]

    # history -------------------------------------------------------------
    def add_history(self, uid: int, project: str, role: str, content: str) -> None:
        self._q(
            "INSERT INTO history(uid,project,role,content,ts) VALUES(?,?,?,?,?)",
            (uid, project, role, content[:6000], time.time()),
        )

    def history(self, uid: int, project: str, limit: int) -> list[dict]:
        rows = self._q(
            "SELECT role,content FROM history WHERE uid=? AND project=? ORDER BY id DESC LIMIT ?",
            (uid, project, limit * 2),
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_history(self, uid: int, project: str) -> None:
        self._q("DELETE FROM history WHERE uid=? AND project=?", (uid, project))

    # tasks ---------------------------------------------------------------
    def new_task(self, uid: int, project: str, goal: str) -> int:
        with self.lock:
            cur = self.c.execute(
                "INSERT INTO tasks(uid,project,goal,status,started) VALUES(?,?,?,?,?)",
                (uid, project, goal[:2000], "running", time.time()),
            )
            self.c.commit()
            return int(cur.lastrowid)

    def finish_task(self, tid: int, **f: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in f)
        self._q(f"UPDATE tasks SET {cols}, ended=? WHERE id=?", (*f.values(), time.time(), tid))

    def task(self, tid: int) -> Optional[dict]:
        r = self._q("SELECT * FROM tasks WHERE id=?", (tid,))
        return dict(r[0]) if r else None

    def recent_tasks(self, uid: int, project: str, n: int = 6) -> list[dict]:
        return [
            dict(r)
            for r in self._q(
                "SELECT * FROM tasks WHERE uid=? AND project=? ORDER BY id DESC LIMIT ?", (uid, project, n)
            )
        ]

    def stats(self) -> dict:
        t = self._q("SELECT COUNT(*) n, COALESCE(SUM(tokens_in),0) i, COALESCE(SUM(tokens_out),0) o FROM tasks")[0]
        u = self._q("SELECT COUNT(*) n FROM users")[0]
        return {"users": u["n"], "tasks": t["n"], "tokens_in": t["i"], "tokens_out": t["o"]}


# ============================================================================
# WORKSPACE  (per-user, per-project directory; every path is confined to it)
# ============================================================================


@dataclass
class Workspace:
    uid: int
    project: str
    root: Path

    @property
    def agent_dir(self) -> Path:
        return self.root / ".agent"

    @property
    def home(self) -> Path:
        return self.root.parent / ".home"

    def resolve(self, p: str) -> Path:
        p = (p or ".").strip()
        path = Path(p)
        if not path.is_absolute():
            path = self.root / path
        real = Path(os.path.realpath(path))
        root = Path(os.path.realpath(self.root))
        if real != root and root not in real.parents:
            raise ToolError(f"path escapes the project workspace: {p}")
        return real

    def rel(self, p: Path) -> str:
        try:
            return str(Path(os.path.realpath(p)).relative_to(Path(os.path.realpath(self.root)))) or "."
        except ValueError:
            return str(p)


def get_workspace(uid: int, project: str) -> Workspace:
    project = safe_name(project)
    root = Cfg.WORKSPACE_ROOT / str(uid) / project
    root.mkdir(parents=True, exist_ok=True)
    (root / ".agent").mkdir(exist_ok=True)
    (root.parent / ".home").mkdir(exist_ok=True)
    return Workspace(uid, project, root)


def project_kind(root: Path) -> str:
    """Classify a completed project for the action button in its final report."""
    if (root / "package.json").is_file() or (root / "index.html").is_file() or (root / "vite.config.ts").is_file() or (root / "vite.config.js").is_file():
        return "web"
    py_files = list(root.glob("*.py"))
    if py_files or (root / "requirements.txt").is_file() or (root / "pyproject.toml").is_file():
        return "bot"
    return ""


def host_command(root: Path, kind: str) -> tuple[str, int]:
    """Return a conservative default command and port for a one-click host."""
    if kind == "web":
        pkg = root / "package.json"
        if pkg.is_file():
            try:
                scripts = (json.loads(pkg.read_text(encoding="utf-8")).get("scripts") or {})
            except (OSError, ValueError, TypeError):
                scripts = {}
            if "dev" in scripts:
                return "npm run dev -- --host 0.0.0.0 --port 5173", 5173
            if "start" in scripts:
                return "npm start -- --host 0.0.0.0 --port 3000", 3000
        return "python3 -m http.server 8000 --bind 0.0.0.0", 8000
    py = next((p for p in sorted(root.glob("*.py")) if p.name not in {"setup.py", "conftest.py"}), None)
    return (f"python3 {shlex.quote(py.name)}", 0) if py else ("python3 -m pytest -q", 0)


def list_projects(uid: int) -> list[str]:
    base = Cfg.WORKSPACE_ROOT / str(uid)
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))


SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".agent", ".mypy_cache", ".pytest_cache", ".next", "dist", "build", "target", ".idea", ".cache"}


def tree(root: Path, depth: int = 2, cap: int = 120) -> str:
    lines: list[str] = []

    def walk(d: Path, level: int, prefix: str) -> None:
        if len(lines) >= cap:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for e in entries:
            if len(lines) >= cap:
                lines.append(f"{prefix}… (truncated)")
                return
            if e.name in SKIP_DIRS and e.is_dir():
                lines.append(f"{prefix}{e.name}/  (skipped)")
                continue
            if e.is_dir():
                lines.append(f"{prefix}{e.name}/")
                if level < depth:
                    walk(e, level + 1, prefix + "  ")
            else:
                try:
                    sz = e.stat().st_size
                except OSError:
                    sz = 0
                lines.append(f"{prefix}{e.name}  ({sz:,} B)" if sz >= 1024 else f"{prefix}{e.name}")

    walk(root, 1, "")
    return "\n".join(lines) or "(empty)"


# ============================================================================
# SAFETY POLICY  (heuristics: they lower risk, they are not a sandbox)
# ============================================================================

_BLOCK = [
    (r"\brm\s+(?:-\S+\s+)*(?:/|~|/\*|\$HOME|\.\.?)(?:\s|$)", "recursive delete of root/home/parent"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b", "formatting a filesystem"),
    (r"\bdd\b[^|;&]*\bof=/dev/", "raw device write"),
    (r">\s*/dev/(?:sd|nvme|hd|vd)", "raw device write"),
    (r"\b(?:shutdown|reboot|poweroff|halt)\b", "power control"),
    (r"/proc/(?:\d+|self)/environ", "reading process environment"),
    (r"\bchmod\s+-R\s+0?777\s+/(?:\s|$)", "global permission change"),
]
_APPROVE = [
    (r"\bsudo\b", "uses sudo"),
    (r"\brm\s+(?:-\S+\s+)*-\S*[rR]", "recursive delete"),
    (r"\bgit\s+push\b", "pushes to a remote"),
    (r"\bgit\s+(?:reset\s+--hard|clean\s+-\S*f)", "discards local changes"),
    (r"\b(?:curl|wget)\b[^|;]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", "pipes a download into a shell"),
    (r"\b(?:apt|apt-get|yum|dnf|apk|brew|snap)\s+(?:install|remove|purge|upgrade)", "system package change"),
    (r"\bdocker\b|\bpodman\b", "container runtime"),
    (r"\b(?:npm|pnpm|yarn|twine)\s+publish\b|\btwine\s+upload\b", "publishes a package"),
    (r"\b(?:kill|pkill|killall)\b", "kills processes"),
    (r"\b(?:ssh|scp|rsync|sftp)\b", "remote access"),
    (r"\b(?:nc|ncat|netcat|socat)\b", "raw network tool"),
    (r"\b(?:crontab|systemctl|service|launchctl)\b", "system service change"),
    (r"\b(?:chown|mount|umount)\b", "system-level change"),
]
_SAFE_ABS = ("/usr/", "/bin/", "/sbin/", "/lib", "/opt/", "/tmp/", "/dev/null", "/dev/stdout", "/dev/stderr", "/etc/os-release")


def outside_paths(cmd: str, root: Path) -> list[str]:
    try:
        toks = shlex.split(cmd, posix=True)
    except ValueError:
        toks = cmd.split()
    bad: list[str] = []
    root_s = os.path.realpath(root)
    for t in toks:
        t = re.sub(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://\S*", "", t)  # ignore URLs
        if t == ".." or t.endswith("/.."):
            full = os.path.realpath(os.path.join(root_s, t))
            if not (full == root_s or full.startswith(root_s + os.sep)):
                bad.append(t)
            continue
        for cand in re.findall(r"(?:^|[=:,<>])((?:~|/|\.\./)[^\s'\";|&<>:,]*)", t):
            if cand.startswith("~"):
                bad.append(cand)
                continue
            if cand.startswith("../") or "/../" in cand:
                full = os.path.realpath(os.path.join(root_s, cand)) if not cand.startswith("/") else os.path.realpath(cand)
                if not (full == root_s or full.startswith(root_s + os.sep)):
                    bad.append(cand)
                continue
            if cand.startswith("/"):
                full = os.path.realpath(cand)
                if full == root_s or full.startswith(root_s + os.sep) or cand.startswith(_SAFE_ABS):
                    continue
                bad.append(cand)
    return sorted(set(bad))[:4]


def classify_command(cmd: str, root: Path) -> tuple[str, str]:
    """-> ('block'|'approve'|'ok', reason)"""
    for pat, why in _BLOCK:
        if re.search(pat, cmd):
            return "block", why
    for pat, why in _APPROVE:
        if re.search(pat, cmd):
            return "approve", why
    out = outside_paths(cmd, root)
    if out:
        return "approve", "touches paths outside the project: " + ", ".join(out)
    return "ok", ""


# ============================================================================
# SHELL RUNNER
# ============================================================================


def clean_env(ws: Workspace) -> dict[str, str]:
    """Scrubbed environment: the agent's commands never see the bot token or API keys."""
    keep = {"PATH", "LANG", "LC_ALL", "TERM", "TZ", "SHELL", "USER", "LOGNAME"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if not Cfg.SHARE_HOME:
        env["HOME"] = str(ws.home)
    elif "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    env.update(
        CI="1",
        DEBIAN_FRONTEND="noninteractive",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        GIT_TERMINAL_PROMPT="0",
        NO_COLOR="1",
        PYTHONUNBUFFERED="1",
        PAGER="cat",
        GIT_PAGER="cat",
        GIT_AUTHOR_NAME="Coding Agent",
        GIT_AUTHOR_EMAIL="agent@localhost",
        GIT_COMMITTER_NAME="Coding Agent",
        GIT_COMMITTER_EMAIL="agent@localhost",
    )
    return env


def kill_group(proc: asyncio.subprocess.Process, sig: int = signal.SIGTERM) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    kill_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), 2)
    except asyncio.TimeoutError:
        kill_group(proc, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), 2)


async def spawn(cmd: str, ws: Workspace) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_shell(
        cmd,
        cwd=str(ws.root),
        env=clean_env(ws),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        executable=shutil.which("bash") or "/bin/sh",
    )


async def run_shell(
    cmd: str, ws: Workspace, timeout: int, on_output: Optional[Callable[[str], None]] = None
) -> tuple[int, str, bool]:
    proc = await spawn(cmd, ws)
    chunks: list[str] = []
    total = 0

    async def reader() -> None:
        nonlocal total
        assert proc.stdout
        while True:
            data = await proc.stdout.read(8192)
            if not data:
                break
            text = data.decode("utf-8", "replace")
            if total < 1_500_000:
                chunks.append(text)
                total += len(text)
            if on_output:
                on_output(text)

    rd = asyncio.create_task(reader())
    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await _terminate(proc)
    except asyncio.CancelledError:
        await _terminate(proc)
        rd.cancel()
        raise
    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(rd, 5)
    rd.cancel()
    return (proc.returncode if proc.returncode is not None else -1), "".join(chunks), timed_out


@dataclass
class BgProc:
    id: int
    cmd: str
    proc: asyncio.subprocess.Process
    started: float
    buf: deque = field(default_factory=lambda: deque(maxlen=400))
    reader: Optional[asyncio.Task] = None

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None


# ============================================================================
# CHECKPOINTS  (shadow git repo: never touches the project's own .git)
# ============================================================================

_SHADOW_EXCLUDE = ".git/\n.agent/\nnode_modules/\n.venv/\nvenv/\n__pycache__/\n*.pyc\n.next/\n.cache/\n.DS_Store\n"


class Checkpoints:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.gd = ws.agent_dir / "shadow.git"
        self.ok = shutil.which("git") is not None

    async def _git(self, *args: str) -> tuple[int, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.ws.home),
            "GIT_DIR": str(self.gd),
            "GIT_WORK_TREE": str(self.ws.root),
            "GIT_AUTHOR_NAME": "agent",
            "GIT_AUTHOR_EMAIL": "agent@localhost",
            "GIT_COMMITTER_NAME": "agent",
            "GIT_COMMITTER_EMAIL": "agent@localhost",
        }
        p = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(self.ws.root), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await p.communicate()
        return p.returncode or 0, out.decode("utf-8", "replace").strip()

    async def init(self) -> None:
        if not self.ok or (self.gd / "HEAD").exists():
            return
        self.gd.parent.mkdir(parents=True, exist_ok=True)
        await self._git("init", "--quiet")
        (self.gd / "info").mkdir(exist_ok=True)
        (self.gd / "info" / "exclude").write_text(_SHADOW_EXCLUDE)
        await self._git("config", "core.quotepath", "false")

    async def snapshot(self, msg: str) -> Optional[str]:
        if not self.ok:
            return None
        try:
            await self.init()
            await self._git("add", "-A")
            await self._git("commit", "--allow-empty", "--no-verify", "-q", "-m", msg)
            rc, sha = await self._git("rev-parse", "HEAD")
            return sha if rc == 0 else None
        except Exception as e:  # never let checkpoints break a task
            log.warning("checkpoint failed: %s", e)
            return None

    async def diff_stat(self, a: str, b: str) -> str:
        rc, out = await self._git("diff", "--stat", "--no-color", a, b)
        return out if rc == 0 else ""

    async def changed(self, a: str, b: str) -> list[str]:
        rc, out = await self._git("diff", "--name-status", a, b)
        return out.splitlines() if rc == 0 and out else []

    async def restore(self, sha: str) -> str:
        rc, out = await self._git("reset", "--hard", "-q", sha)
        if rc != 0:
            return f"reset failed: {out}"
        await self._git("clean", "-fd", "-q")
        return "ok"


# ============================================================================
# SKILLS  (reusable playbooks the agent can load — and write — itself)
# ============================================================================

BUILTIN_SKILLS: dict[str, tuple[str, str]] = {
    "debug-and-fix": (
        "Systematic procedure for failing tests/crashes",
        "1. Reproduce first: run the failing command and capture the exact error.\n"
        "2. Read the traceback bottom-up; open the referenced file/line.\n"
        "3. Form ONE hypothesis, make the smallest change that tests it.\n"
        "4. Re-run the same command. Repeat; never change several things at once.\n"
        "5. When green, run the full test suite to catch regressions. Add a regression test.",
    ),
    "python-project": (
        "Scaffold a clean Python project",
        "Layout: pyproject.toml, src/<pkg>/__init__.py, tests/test_*.py, README.md, .gitignore.\n"
        "Use a venv: `python3 -m venv .venv && .venv/bin/pip install -e .[dev]`.\n"
        "Run tests with `.venv/bin/python -m pytest -q`. Type hints + docstrings on public functions.",
    ),
    "web-app": (
        "Build and verify a small web app",
        "Prefer plain HTML/CSS/JS or Flask/FastAPI unless told otherwise.\n"
        "Start the server with run_command background=true, poll process_output until it listens,\n"
        "then verify with http_request and take a screenshot (screenshot tool) to show the user.\n"
        "Stop the server (process stop) when finished unless the user wants it running.",
    ),
    "node-project": (
        "Node/TypeScript project conventions",
        "`npm init -y`, add scripts (dev/build/test), use `npm install --no-audit --no-fund`.\n"
        "Verify with `npm test` / `npm run build`. Keep node_modules out of the final report.",
    ),
    "code-review": (
        "Review code changes critically",
        "Read the diff (`git diff` or the touched files). Check: correctness, edge cases, error handling,\n"
        "security (injection, secrets), performance hot spots, tests. Report findings by severity with file:line.",
    ),
}


class Skills:
    def __init__(self, uid: int):
        self.dir = Cfg.DATA_DIR / "skills" / str(uid)

    def all(self) -> dict[str, tuple[str, str, bool]]:
        out = {k: (d, c, True) for k, (d, c) in BUILTIN_SKILLS.items()}
        if self.dir.exists():
            for f in sorted(self.dir.glob("*.md")):
                txt = f.read_text(encoding="utf-8", errors="replace")
                m = re.match(r"^# .*\n> (.*)\n\n(.*)$", txt, re.S)
                out[f.stem] = (m.group(1), m.group(2), False) if m else ("custom skill", txt, False)
        return out

    def save(self, name: str, desc: str, content: str) -> str:
        name = safe_name(name, "skill")
        if name in BUILTIN_SKILLS:
            raise ToolError("cannot overwrite a built-in skill")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{name}.md").write_text(f"# {name}\n> {short(desc, 160)}\n\n{content[:8000]}", encoding="utf-8")
        return name

    def delete(self, name: str) -> bool:
        f = self.dir / f"{safe_name(name)}.md"
        if f.exists():
            f.unlink()
            return True
        return False


# ============================================================================
# LLM CLIENT
#   * Anthropic Messages API (native tool use, prompt caching)
#   * OpenAI-compatible chat/completions (OpenRouter, Gemini, custom/Ollama/…)
#   * Automatic "text tool protocol" fallback for models without tool calling
# ============================================================================


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResult:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    tin: int = 0
    tout: int = 0
    stop: str = ""


class LLMError(Exception):
    def __init__(self, msg: str, status: int = 0, kind: str = ""):
        super().__init__(msg)
        self.status, self.kind = status, kind  # kind: auth | tools | context | rate | server


def provider_config(name: str, model: Optional[str] = None, uid: Optional[int] = None) -> Optional[dict]:
    override = provider_override(uid, name)
    if name in ("anthropic", "openrouter", "gemini"):
        key = override.get("key") or {
            "anthropic": Cfg.ANTHROPIC_API_KEY,
            "openrouter": Cfg.OPENROUTER_API_KEY,
            "gemini": Cfg.GEMINI_API_KEY,
        }[name]
        if key:
            default_model = {
                "anthropic": Cfg.ANTHROPIC_MODEL,
                "openrouter": Cfg.OPENROUTER_MODEL,
                "gemini": Cfg.GEMINI_MODEL,
            }[name]
            return dict(name=name, kind="anthropic" if name == "anthropic" else "openai",
                        url=override.get("url") or {
                            "anthropic": "https://api.anthropic.com/v1/messages",
                            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
                            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                        }[name], key=key,
                        model=model or override.get("model") or default_model)
    if name in ("antigravity", "opencode"):
        # CLI availability on the host is not a user connection. These
        # providers become visible only after this Telegram user completes
        # the matching login flow and receives a user-scoped override.
        if uid is not None and not override:
            return None
        command = override.get("command") or (Cfg.AGY_COMMAND if name == "antigravity" else Cfg.OPENCODE_COMMAND)
        chosen_model = model or override.get("model") or MODEL_CATALOG[name][0][0]
        if override or shutil.which(command):
            return dict(name=name, kind="cli", key="local", command=command, model=chosen_model,
                        auth=override.get("auth", "google" if name == "antigravity" else "openai-chatgpt"),
                        data_dir=override.get("data_dir", ""))
    if name == "anthropic" and Cfg.ANTHROPIC_API_KEY:
        return dict(name=name, kind="anthropic", url="https://api.anthropic.com/v1/messages",
                    key=Cfg.ANTHROPIC_API_KEY, model=model or Cfg.ANTHROPIC_MODEL)
    if name == "openrouter" and Cfg.OPENROUTER_API_KEY:
        return dict(name=name, kind="openai", url="https://openrouter.ai/api/v1/chat/completions",
                    key=Cfg.OPENROUTER_API_KEY, model=model or Cfg.OPENROUTER_MODEL)
    if name == "gemini" and Cfg.GEMINI_API_KEY:
        return dict(name=name, kind="openai",
                    url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    key=Cfg.GEMINI_API_KEY, model=model or Cfg.GEMINI_MODEL)
    if name == "custom":
        base = override.get("url") or Cfg.LLM_BASE_URL
        chosen_model = model or override.get("model") or Cfg.LLM_MODEL
        if base and chosen_model:
            return dict(name=name, kind="openai", url=base.rstrip("/") + "/chat/completions",
                        key=override.get("key") or Cfg.LLM_API_KEY or "none", model=chosen_model)
    return None


def available_providers(uid: Optional[int] = None) -> list[str]:
    return [p for p in ("anthropic", "openrouter", "gemini", "antigravity", "opencode", "custom") if provider_config(p, uid=uid)]


def default_provider(uid: Optional[int] = None) -> Optional[str]:
    av = available_providers(uid)
    if Cfg.LLM_PROVIDER in av:
        return Cfg.LLM_PROVIDER
    return av[0] if av else None


def native_tool_fallback(uid: int, current_name: str) -> Optional[dict]:
    """Pick an already-connected API provider with native tool calling."""
    for name in ("anthropic", "openrouter", "gemini", "custom"):
        if name == current_name:
            continue
        pc = provider_config(name, uid=uid)
        if pc and pc.get("kind") in ("anthropic", "openai"):
            return pc
    return None


def model_catalog(uid: Optional[int] = None) -> dict[str, list[tuple[str, bool]]]:
    """Return visible models, adding each user's custom model without sharing keys."""
    out = {name: list(models) for name, models in MODEL_CATALOG.items()}
    for name in available_providers(uid):
        pc = provider_config(name, uid=uid)
        if not pc:
            continue
        model = pc.get("model")
        if model and all(model != existing for existing, _ in out.setdefault(name, [])):
            out[name].insert(0, (model, False))
    return out


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    for attempt in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            v = json.loads(attempt)
            return v if isinstance(v, dict) else {"value": v}
        except ValueError:
            continue
    return {"__invalid__": s[:300]}


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
CLI_AGENT_TIMEOUT = 1800.0  # seconds a CLI agent (agy / opencode) may work on one task


def _scan_tree(root: Path, cap: int = 200_000) -> dict[str, tuple[int, int]]:
    """Cheap workspace fingerprint: relative path -> (size, mtime_ns)."""
    out: dict[str, tuple[int, int]] = {}
    skip = {".agent", ".git", ".venv", "venv", "node_modules", "__pycache__", ".gradle", ".cache"}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            pth = Path(base) / f
            try:
                st = pth.stat()
            except OSError:
                continue
            out[str(pth.relative_to(root))] = (st.st_size, st.st_mtime_ns)
            if len(out) >= cap:
                return out
    return out


_TEXT_BLOCK = re.compile(r"```(?:tool|json|tool_call)?[ \t]*\n(\{.*?\})[ \t]*\n```", re.S)


class LLM:
    def __init__(self, http: httpx.AsyncClient, pc: dict):
        self.http, self.pc = http, pc
        self.text_mode = False

    @property
    def label(self) -> str:
        return f"{self.pc['name']}:{self.pc['model']}"

    # ---- public --------------------------------------------------------------
    async def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 8192) -> LLMResult:
        last: Optional[Exception] = None
        for attempt in range(4):
            try:
                if self.pc["kind"] == "cli":
                    return await self._cli(system, messages, tools)
                if self.pc["kind"] == "anthropic":
                    return await self._anthropic(system, messages, tools, max_tokens)
                if self.text_mode:
                    return await self._openai_text(system, messages, tools, max_tokens)
                return await self._openai(system, messages, tools, max_tokens)
            except LLMError as e:
                last = e
                if e.kind == "tools" and self.pc["kind"] == "openai" and not self.text_mode:
                    log.warning("model rejects native tools — switching to text tool protocol")
                    self.text_mode = True
                    continue
                if e.kind in ("rate", "server") and attempt < 3:
                    await asyncio.sleep(1.5 * 2**attempt)
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = LLMError(f"network error talking to the LLM: {type(e).__name__}", kind="server")
                if attempt < 3:
                    await asyncio.sleep(2**attempt)
                    continue
                raise last
        raise last or LLMError("LLM call failed")

    @staticmethod
    def _tool_protocol(tools: list[dict]) -> str:
        lines = [
            "## Tool protocol (MANDATORY)",
            "You are the brain of a coding agent. You do NOT have direct access to the user's workspace;",
            "the ONLY way to create/edit files or run commands is by calling the tools below.",
            "Do NOT use your own built-in tools. Do NOT claim a task is done unless you actually called tools",
            "(e.g. write_file) and saw their results. For building anything, call tools first.",
            "To call a tool, reply with a fenced block exactly like this (valid JSON):",
            '```tool\n{"name": "<tool_name>", "arguments": {"arg": "value"}}\n```',
            "You may include several tool blocks in one reply. Tool results arrive in the next message.",
            "Only when the task is truly complete, answer normally WITHOUT any tool block.\n",
            "Available tools:",
        ]
        for t in tools:
            f = t["function"]
            lines.append(f"- {f['name']}: {f['description']}\n  parameters: {json.dumps(f['parameters'])}")
        return "\n".join(lines)

    async def _cli(self, system: str, messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResult:
        """Use the installed user's authenticated CLI without handling credentials here.

        CLI providers return plain text only, so tool calling is done through the
        same fenced ```tool``` text protocol used for non-native-tool models.
        """
        command = self.pc.get("command") or ""
        if not shutil.which(command):
            raise LLMError(f"{self.pc['name']} CLI is not installed. Install `{command}` and run its login flow first.", kind="auth")
        prompt_parts = [system]
        if tools:
            prompt_parts.append(self._tool_protocol(tools))
        for message in messages[-30:]:
            role = message.get("role", "user")
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            if role == "assistant":
                for tc in message.get("tool_calls") or []:
                    blk = json.dumps({"name": tc["function"]["name"], "arguments": _parse_args(tc["function"]["arguments"])})
                    content += f"\n```tool\n{blk}\n```"
                prompt_parts.append(f"[assistant]\n{content.strip() or '(no output)'}")
            elif role == "tool":
                prompt_parts.append(f"[tool_result: {message.get('name', 'tool')}]\n{content}")
            else:
                prompt_parts.append(f"[user]\n{content}")
        if tools:
            prompt_parts.append("[assistant]\n(Respond now. If work remains, emit ```tool blocks.)")
        prompt = "\n\n".join(prompt_parts)
        if self.pc["name"] == "antigravity":
            args = [command, "-p", prompt]
        else:
            args = [command, "run", "--format", "default", "--model", self.pc["model"], prompt]
        try:
            env = os.environ.copy()
            if self.pc.get("name") == "opencode" and self.pc.get("data_dir"):
                Path(self.pc["data_dir"]).mkdir(parents=True, exist_ok=True)
                env["XDG_DATA_HOME"] = self.pc["data_dir"]
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True, env=env,
            )
            out, err = await asyncio.wait_for(proc.communicate(), 240)
        except asyncio.TimeoutError as e:
            with contextlib.suppress(Exception):
                proc.kill()
            raise LLMError(f"{self.pc['name']} CLI timed out", kind="server") from e
        text = out.decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            detail = err.decode("utf-8", "replace").strip()[-500:]
            raise LLMError(f"{self.pc['name']} CLI failed: {detail or text[-500:]}", kind="server")
        calls: list[ToolCall] = []
        if tools:
            for blk in _TEXT_BLOCK.findall(text):
                try:
                    j = json.loads(blk)
                except ValueError:
                    continue
                name = j.get("name") or j.get("tool")
                if name:
                    calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", str(name), _parse_args(j.get("arguments", j.get("args", {})))))
            if calls:
                text = _TEXT_BLOCK.sub("", text).strip()
        return LLMResult(text, calls)

    # ---- CLI agent mode (antigravity / opencode run their OWN agent loop) -------
    @staticmethod
    async def _spawn_collect(argv: list[str], cwd: Path, env: dict, on_line, timeout: float) -> tuple[str, str, int]:
        import signal as _sig

        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env, limit=1 << 20,
        )

        def kill() -> None:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, _sig.SIGKILL)
            with contextlib.suppress(Exception):
                proc.kill()

        out_buf: list[str] = []
        err_buf: list[str] = []

        async def pump(stream, buf: list[str]) -> None:
            size = 0
            while True:
                try:
                    chunk = await stream.readline()
                except ValueError:  # a single absurdly long line
                    chunk = b"(very long line skipped)\n"
                if not chunk:
                    break
                line = chunk.decode("utf-8", "replace")
                if size < 400_000:
                    buf.append(line)
                    size += len(line)
                if on_line:
                    t = _ANSI.sub("", line).strip()
                    if t:
                        with contextlib.suppress(Exception):
                            on_line(t)

        tasks = [asyncio.create_task(pump(proc.stdout, out_buf)), asyncio.create_task(pump(proc.stderr, err_buf))]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, proc.wait()), timeout)
        except asyncio.TimeoutError as e:
            kill()
            raise LLMError(f"CLI agent timed out after {int(timeout // 60)} min", kind="server") from e
        except BaseException:
            kill()
            raise
        finally:
            for t in tasks:
                t.cancel()
        return "".join(out_buf), "".join(err_buf), int(proc.returncode or 0)

    async def cli_agent(self, prompt: str, cwd: Path, on_line=None, timeout: float = CLI_AGENT_TIMEOUT) -> str:
        """Run the installed CLI as a complete autonomous agent with `cwd` as its workspace.

        antigravity (agy) and opencode already have their own file/shell tools, so we let
        them do the work in the session folder instead of forcing a text tool protocol.
        """
        name = self.pc["name"]
        command = self.pc.get("command") or ""
        exe = shutil.which(command)
        if not exe:
            raise LLMError(f"{name} CLI is not installed. Install `{command}` and run its login flow first.", kind="auth")
        if name == "antigravity":
            base = [exe, "-p", prompt]
            # print mode cannot answer permission prompts, so file/shell tools need auto-approval
            variants = [base + ["--dangerously-skip-permissions"], base]
        else:
            variants = [[exe, "run", "--format", "default", "--model", self.pc["model"], prompt]]
        env = os.environ.copy()
        env.update({"NO_COLOR": "1", "CI": "1", "TERM": "dumb"})
        if name == "opencode" and self.pc.get("data_dir"):
            Path(self.pc["data_dir"]).mkdir(parents=True, exist_ok=True)
            env["XDG_DATA_HOME"] = self.pc["data_dir"]
        text = err = ""
        rc = 0
        for i, argv in enumerate(variants):
            out, err, rc = await self._spawn_collect(argv, cwd, env, on_line, timeout)
            text = _ANSI.sub("", out).strip()
            err = _ANSI.sub("", err).strip()
            low = (err + " " + text).lower()
            bad_flag = rc != 0 and any(k in low for k in ("unknown flag", "unknown option", "unrecognized", "not defined", "unexpected argument", "invalid option"))
            if bad_flag and i < len(variants) - 1:
                log.warning("%s CLI rejected a flag, retrying without it: %s", name, (err or text)[-200:])
                continue
            break
        detail = (err or text)[-600:]
        low = (err + " " + text).lower()
        if rc != 0 or not text:
            if any(k in low for k in ("not logged in", "login", "log in", "sign in", "unauthenticated", "unauthorized", "credential", "401", "oauth")):
                raise LLMError(f"{name} CLI login required: {detail}", kind="auth")
            if any(k in low for k in ("quota", "rate limit", "429", "resource exhausted")):
                raise LLMError(f"{name} quota / rate limit: {detail}", kind="rate")
            raise LLMError(f"{name} CLI {'failed (exit %d)' % rc if rc else 'returned no output'}: {detail or '(no output)'}", kind="server")
        return text

    # ---- plumbing ------------------------------------------------------------
    def _classify(self, status: int, msg: str) -> LLMError:
        low = msg.lower()
        kind = ""
        if status in (401, 403):
            kind = "auth"
        elif status == 429:
            kind = "rate"
        elif status >= 500:
            kind = "server"
        elif any(k in low for k in ("context length", "context_length", "too many tokens", "prompt is too long", "maximum context")):
            kind = "context"
        elif status in (400, 404, 422) and any(
            k in low for k in ("tool", "function call", "function_call", "does not support")
        ):
            kind = "tools"
        return LLMError(f"{self.pc['name']} error {status}: {msg[:400]}", status, kind)

    @staticmethod
    def _unwrap(j: Any) -> Any:
        """Gemini's OpenAI-compat endpoint returns errors as a JSON *list*
        (e.g. [{"error": {...}}]); normalise to a dict."""
        while isinstance(j, list):
            j = j[0] if j else {}
        return j

    async def _post(self, body: dict, headers: dict) -> dict:
        r = await self.http.post(self.pc["url"], json=body, headers=headers, timeout=httpx.Timeout(240.0, connect=20.0))
        if r.status_code >= 400:
            try:
                j = self._unwrap(r.json())
                err = j.get("error", j) if isinstance(j, dict) else j
                msg = err.get("message") if isinstance(err, dict) else str(err)
            except (ValueError, AttributeError):
                msg = r.text
            raise self._classify(r.status_code, msg or r.text)
        try:
            data = self._unwrap(r.json())
        except ValueError:
            raise LLMError("provider returned invalid JSON", r.status_code, "server")
        if not isinstance(data, dict):
            raise LLMError(f"provider returned unexpected response: {str(data)[:200]}", r.status_code, "server")
        if "error" in data and "choices" not in data and "content" not in data:
            err = data["error"]
            code = err.get("code", 500) if isinstance(err, dict) else 500
            raise self._classify(int(code) if str(code).isdigit() else 500,
                                 err.get("message", str(err)) if isinstance(err, dict) else str(err))
        return data

    # ---- OpenAI-compatible ---------------------------------------------------
    def _openai_messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            m = dict(m)
            if m["role"] == "tool" and self.pc["name"] != "gemini":
                m.pop("name", None)
            out.append(m)
        return out

    async def _openai(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> LLMResult:
        body: dict = {"model": self.pc["model"], "messages": self._openai_messages(system, messages), "max_tokens": max_tokens}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        headers = {"Authorization": f"Bearer {self.pc['key']}", "Content-Type": "application/json"}
        if self.pc["name"] == "openrouter":
            headers.update({"HTTP-Referer": "https://github.com/", "X-Title": "Telegram Coding Agent"})
        data = await self._post(body, headers)
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(ToolCall(tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", fn.get("name", ""), _parse_args(fn.get("arguments"))))
        u = data.get("usage") or {}
        return LLMResult(text, calls, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), ch.get("finish_reason") or "")

    # ---- text tool protocol (for models without native tools) -----------------
    async def _openai_text(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> LLMResult:
        lines = [
            system,
            "\n## Tool protocol",
            "You can call tools. To call one, reply with a fenced block exactly like this (valid JSON):",
            '```tool\n{"name": "<tool_name>", "arguments": {"arg": "value"}}\n```',
            "You may include several tool blocks in one reply. Tool results arrive in the next message.",
            "When the task is complete, answer normally WITHOUT any tool block.\n",
            "Available tools:",
        ]
        for t in tools:
            f = t["function"]
            lines.append(f"- {f['name']}: {f['description']}\n  parameters: {json.dumps(f['parameters'])}")
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                txt = m.get("content") or ""
                for tc in m.get("tool_calls") or []:
                    blk = json.dumps({"name": tc["function"]["name"], "arguments": _parse_args(tc["function"]["arguments"])})
                    txt += f"\n```tool\n{blk}\n```"
                conv.append({"role": "assistant", "content": txt.strip() or "(no output)"})
            elif m["role"] == "tool":
                conv.append({"role": "user", "content": f"[tool_result: {m.get('name', 'tool')}]\n{m['content']}"})
            else:
                conv.append({"role": "user", "content": m["content"]})
        body = {"model": self.pc["model"], "messages": [{"role": "system", "content": "\n".join(lines)}, *conv], "max_tokens": max_tokens}
        data = await self._post(body, {"Authorization": f"Bearer {self.pc['key']}", "Content-Type": "application/json"})
        ch = (data.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        calls: list[ToolCall] = []
        for blk in _TEXT_BLOCK.findall(text):
            try:
                j = json.loads(blk)
            except ValueError:
                continue
            name = j.get("name") or j.get("tool")
            if name:
                calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", str(name), _parse_args(j.get("arguments", j.get("args", {})))))
        if calls:
            text = _TEXT_BLOCK.sub("", text).strip()
        u = data.get("usage") or {}
        return LLMResult(text, calls, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), ch.get("finish_reason") or "")

    # ---- Anthropic -------------------------------------------------------------
    @staticmethod
    def _to_anthropic(messages: list[dict]) -> list[dict]:
        out: list[dict] = []

        def push(role: str, blocks: list[dict]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            r = m["role"]
            if r == "user":
                content = m.get("content") or ""
                blocks: list[dict] = []
                if isinstance(content, str):
                    blocks.append({"type": "text", "text": content or "(empty)"})
                elif isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            blocks.append({"type": "text", "text": str(part.get("text") or "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url") or {}
                            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                            match = re.fullmatch(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]+={0,2})", url)
                            if match:
                                blocks.append({"type": "image", "source": {
                                    "type": "base64", "media_type": match.group(1), "data": match.group(2),
                                }})
                push("user", blocks or [{"type": "text", "text": "(empty)"}])
            elif r == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": _parse_args(tc["function"]["arguments"])})
                push("assistant", blocks or [{"type": "text", "text": "(no output)"}])
            elif r == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"] or "(no output)"}])
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    async def _anthropic(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> LLMResult:
        body: dict = {
            "model": self.pc["model"],
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": self._to_anthropic(messages),
        }
        if tools:
            body["tools"] = [
                {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
                for t in tools
            ]
        headers = {"x-api-key": self.pc["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = await self._post(body, headers)
        text, calls = "", []
        for b in data.get("content") or []:
            if b.get("type") == "text":
                text += b.get("text", "")
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(b["id"], b["name"], b.get("input") or {}))
        u = data.get("usage") or {}
        tin = int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
        return LLMResult(text, calls, tin, int(u.get("output_tokens", 0)), data.get("stop_reason") or "")


def compact_messages(messages: list[dict], budget: int, keep_last: int = 12) -> list[dict]:
    """Shrink old tool output first, then drop whole old turns, never breaking tool-call pairing."""

    def content_size(content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            # Image payloads are sent separately from text tokens; count each as a
            # modest placeholder so a base64 string does not evict chat history.
            return sum(
                1_000 if p.get("type") == "image_url" else len(str(p.get("text") or ""))
                for p in content if isinstance(p, dict)
            )
        return len(str(content or ""))

    def size(ms: list[dict]) -> int:
        return sum(content_size(m.get("content")) + 60 * len(m.get("tool_calls") or []) for m in ms)

    if size(messages) <= budget:
        return messages
    n = len(messages)
    out = []
    for i, m in enumerate(messages):
        if i < n - keep_last and m["role"] == "tool" and len(m["content"]) > 400:
            m = dict(m)
            m["content"] = m["content"][:300] + f"\n…[{len(m['content']) - 300} chars elided to save context]"
        out.append(m)
    dropped = False
    while size(out) > budget and len(out) > keep_last + 2:
        j = next((k for k in range(1, len(out)) if out[k]["role"] == "assistant"), None)
        if j is None or j >= len(out) - keep_last:
            break
        end = j + 1
        while end < len(out) and out[end]["role"] == "tool":
            end += 1
        del out[j:end]
        dropped = True
    if dropped:
        out.insert(1, {"role": "user", "content": "[Earlier steps of this task were omitted to save context. Re-read files if you need them.]"})
    return out


# ============================================================================
# RUN CONTEXT & LIVE PROGRESS  (sendRichMessageDraft + <tg-thinking> + Stop)
# ============================================================================

PLAN_STATES = ("pending", "in_progress", "done")


class AgentFailure(Exception):
    pass


@dataclass
class Step:
    n: int
    icon: str
    label: str
    started: float
    ended: float = 0.0
    state: str = "run"  # run | ok | err
    note: str = ""


@dataclass
class Run:
    id: int
    uid: int
    chat_id: int
    thread: Optional[int]
    ws: Workspace
    goal: str
    approval: str
    admin: bool
    llm: Optional[LLM]
    draft_id: int = field(default_factory=lambda: random.randint(1, 2**31 - 1))
    started: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None
    progress: Optional["Progress"] = None
    inbox: list[Any] = field(default_factory=list)
    allow_all: bool = False
    plan: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)
    tool_calls: int = 0
    llm_calls: int = 0
    tin: int = 0
    tout: int = 0
    cp_start: Optional[str] = None
    cp_end: Optional[str] = None
    ask_future: Optional[asyncio.Future] = None
    stopped: bool = False
    last_edit_step: int = -1
    last_verify_step: int = -1
    final_status: str = "done"
    images: list[dict] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.chat_id, self.thread)


class Progress:
    """Live status while the agent works.

    Level 0: rich draft  (<tg-thinking> + plan checklist + recent steps)
    Level 1: rich draft  (thinking block only — in case an inner tag is rejected)
    Level 2: classic sendMessageDraft (plain HTML)
    Level 3: a normal message that is edited in place (works everywhere)
    """

    def __init__(self, tg: TG, run: Run):
        self.tg, self.run = tg, run
        # A real message that is edited in place is visible in every client while the task runs;
        # Telegram "drafts" often only show up when the run ends.  Set PROGRESS_MODE=draft to opt back in.
        if Cfg.PROGRESS_MODE == "draft":
            self.level = 0 if (tg.draft_ok and tg.rich_ok and run.chat_id > 0) else (2 if (tg.draft_ok and run.chat_id > 0) else 3)
        else:
            self.level = 3
        self.out_lines: list[str] = []
        self.status_text = "Starting"
        self.waiting = ""
        self.out_tail = ""
        self.cur: Optional[Step] = None
        self.dirty = True
        self.closed = False
        self.last_push = 0.0
        self.next_ok = 0.0
        self.msg_id: Optional[int] = None
        self.task: Optional[asyncio.Task] = None

    # ---- state mutation -------------------------------------------------------
    def set_status(self, s: str) -> None:
        self.status_text, self.dirty = s, True

    def begin(self, icon: str, label: str) -> Step:
        st = Step(len(self.run.steps) + 1, icon, label, time.time())
        self.run.steps.append(st)
        self.cur, self.out_tail, self.dirty = st, "", True
        self.out_lines = []
        return st

    def note(self, icon: str, label: str, ok: bool = True) -> None:
        """Add an already-finished entry to the live list (e.g. a file the agent just wrote)."""
        now = time.time()
        self.run.steps.append(Step(len(self.run.steps) + 1, icon, label, now, now, "ok" if ok else "err"))
        self.dirty = True

    def feed(self, text: str) -> None:
        """Append raw output lines to the live tail shown under the current activity."""
        for l in str(text).splitlines():
            l = _ANSI.sub("", l).strip()
            if l:
                self.out_lines.append(l)
        self.out_lines = self.out_lines[-5:]
        if self.out_lines:
            self.out_tail = self.out_lines[-1]
        self.dirty = True

    def end(self, st: Step, ok: bool, note: str = "") -> None:
        st.ended, st.state, st.note = time.time(), "ok" if ok else "err", note
        if self.cur is st:
            self.cur = None
        self.out_tail, self.dirty = "", True
        self.out_lines = []

    def tail(self, text: str) -> None:
        self.feed(text)

    def touch(self) -> None:
        self.dirty = True

    # ---- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        self.closed = True  # first: late tool results must not resurrect the draft
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        if self.level == 3 and self.msg_id:
            await self.tg.delete(self.run.chat_id, self.msg_id)

    async def _loop(self) -> None:
        try:
            await self._push()
            while not self.closed:
                await asyncio.sleep(0.5)
                now = time.time()
                if not self.closed and ((self.dirty and now - self.last_push >= 1.8) or now - self.last_push >= 4.0):
                    await self._push()
        except asyncio.CancelledError:
            pass

    # ---- rendering ---------------------------------------------------------------
    def _head(self) -> str:
        r, cur = self.run, self.cur
        total = dur(time.time() - r.started)
        if self.waiting:
            return f"⏸ <b>{esc(self.waiting)}</b> · {total}"
        if cur and cur.state == "run":
            h = f"{cur.icon} <b>{esc(short(cur.label, 70))}</b> · {dur(time.time() - cur.started)} · total {total}"
            if self.out_tail:
                h += f"<br><code>{esc(short(self.out_tail, 90))}</code>"
            return h
        return f"🧠 <b>{esc(self.status_text)}</b> · {total}"

    def render_live(self) -> str:
        r, now = self.run, time.time()
        eng = f" · <code>{esc(r.llm.label)}</code>" if r.llm else ""
        steps_done = sum(1 for x in r.steps if x.state != "run")
        parts = [f"<h4>🐉 Agent X working</h4><p>⏱ <b>{dur(now - r.started)}</b> · {steps_done} step(s){eng}</p>"]
        if r.plan:
            done = sum(1 for p in r.plan if p["status"] == "done")
            lis = []
            for p in r.plan:
                txt = esc(p["text"])
                if p["status"] == "in_progress":
                    txt = f"<b>{txt}</b>"
                chk = " checked" if p["status"] == "done" else ""
                lis.append(f'<li><input type="checkbox"{chk}/>{txt}</li>')
            parts.append(f"<h4>📋 Plan · {done}/{len(r.plan)}</h4><ul>{''.join(lis)}</ul>")
        ic = {"ok": "✅", "err": "❌"}
        rows = []
        for x in r.steps[-8:]:
            if x.state == "run":
                rows.append(f"▸ {x.icon} <b>{esc(short(x.label, 60))}</b> · {dur(now - x.started)}")
            else:
                rows.append(f"{ic.get(x.state, '•')} {esc(short(x.label, 60))}")
        if self.waiting:
            rows.append(f"⏸ <b>{esc(self.waiting)}</b>")
        elif not (self.cur and self.cur.state == "run"):
            rows.append(f"🧠 <b>{esc(short(self.status_text, 70))}</b>")
        parts.append("<blockquote>" + "<br>".join(rows) + "</blockquote>")
        if self.out_lines:
            parts.append("<pre>" + esc("\n".join(short(l, 80) for l in self.out_lines[-4:])) + "</pre>")
        return "".join(parts)

    def render(self) -> str:
        if self.level == 3:
            return self.render_live()
        r = self.run
        thinking = self.level in (0, 1, 2)
        head = self._head()
        if self.level == 1:
            return f"<tg-thinking>{esc(strip_tags(head))}</tg-thinking>"
        parts = [f"<tg-thinking>{head}</tg-thinking>" if thinking else f"<p>{head}</p>"]
        if r.plan:
            done = sum(1 for p in r.plan if p["status"] == "done")
            lis = []
            for p in r.plan:
                txt = esc(p["text"])
                if p["status"] == "in_progress":
                    txt = f"<b>{txt}</b>"
                chk = " checked" if p["status"] == "done" else ""
                lis.append(f'<li><input type="checkbox"{chk}/>{txt}</li>')
            parts.append(f"<h4>📋 Plan · {done}/{len(r.plan)}</h4><ul>{''.join(lis)}</ul>")
        recent = [s for s in r.steps if s.state != "run"][-6:]
        if recent:
            icon = {"ok": "✅", "err": "❌"}
            parts.append(
                "<p>"
                + "<br>".join(f"{icon[s.state]} {esc(short(s.label, 60))} <i>{dur(s.ended - s.started) if s.ended - s.started >= 1 else '<1s'}</i>" for s in recent)
                + "</p>"
            )
        return "".join(parts)

    # ---- transport ---------------------------------------------------------------
    async def _push(self) -> None:
        if self.closed:
            return
        now = time.time()
        if now < self.next_ok:
            return
        self.dirty, self.last_push = False, now
        r = self.run
        try:
            if self.level in (0, 1):
                await self.tg.draft_rich(r.chat_id, r.draft_id, self.render(), r.thread)
            elif self.level == 2:
                await self.tg.draft_plain(r.chat_id, r.draft_id, rich_to_classic(self.render()), r.thread)
            else:
                body = self.render()
                stop = kb([btn("⏹ Stop", f"stop:{r.id}", style="danger")])
                if self.msg_id is None:
                    self.msg_id = await self.tg.send(r.chat_id, body, thread=r.thread, markup=stop, silent=True)
                elif not self.closed:
                    await self.tg.edit(r.chat_id, self.msg_id, body, stop)
        except TGError as e:
            if e.code == 429:
                self.next_ok = now + int(e.params.get("retry_after", 3))
            elif e.code in (401, 403):
                log.warning("progress push forbidden: %s", e.desc)
            elif self.level < 3:
                log.warning("draft level %d failed (%s) — degrading", self.level, e.desc[:100])
                self.level += 1
                self.dirty = True
                if self.level == 2 and not self.tg.rich_ok:
                    self.level = 2
            else:
                log.debug("progress edit failed: %s", e.desc[:100])


# ============================================================================
# TOOL SCHEMAS  (what the LLM can do)
# ============================================================================


def _tool(name: str, desc: str, props: dict, req: tuple = ()) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                              "parameters": {"type": "object", "properties": props, "required": list(req)}}}


def _s(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _i(desc: str) -> dict:
    return {"type": "integer", "description": desc}


def _b(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


TOOLS: list[dict] = [
    _tool("update_plan", "Create or update the visible task checklist (3-7 concrete steps). Call it first for any non-trivial task and keep statuses current.",
          {"items": {"type": "array", "description": "Ordered plan items",
                     "items": {"type": "object", "properties": {"text": _s("Step description"),
                                                                 "status": {"type": "string", "enum": list(PLAN_STATES)}},
                               "required": ["text", "status"]}}}, ("items",)),
    _tool("list_dir", "List files/folders as a tree. Use to understand project layout.",
          {"path": _s("Directory relative to project root (default '.')"), "depth": _i("Tree depth 1-4 (default 2)")}),
    _tool("read_file", "Read a text file with line numbers (numbers are display-only, never include them in edits). Large files are paged.",
          {"path": _s("File path"), "start_line": _i("First line (1-based)"), "end_line": _i("Last line (inclusive)")}, ("path",)),
    _tool("write_file", "Create a new file or fully overwrite one. Provide COMPLETE content. For small changes to existing files use edit_file.",
          {"path": _s("File path (parent folders are created)"), "content": _s("Full file content")}, ("path", "content")),
    _tool("edit_file", "Replace an exact string in an existing file. old_str must match exactly once (include enough context) unless replace_all=true.",
          {"path": _s("File path"), "old_str": _s("Exact text to replace"), "new_str": _s("Replacement text"),
           "replace_all": _b("Replace every occurrence")}, ("path", "old_str", "new_str")),
    _tool("search_code", "Regex search across project files (skips node_modules, .git, venvs). Returns file:line: text.",
          {"pattern": _s("Regular expression"), "path": _s("Directory or file (default '.')"),
           "glob": _s("Filename filter like '*.py'"), "ignore_case": _b("Case-insensitive")}, ("pattern",)),
    _tool("run_command", "Run a non-interactive shell command in the project root (tests, builds, installs, git, scripts). Returns exit code + output. "
          "Use background=true for servers/watchers, then use the process tool. Risky commands may require the user's approval.",
          {"command": _s("Shell command"), "timeout": _i("Seconds before kill (default 120)"),
           "background": _b("Start detached and return immediately")}, ("command",)),
    _tool("process", "Manage background processes started with run_command(background=true).",
          {"action": {"type": "string", "enum": ["list", "output", "stop"]}, "id": _i("Process id"), "tail": _i("Lines of output (default 40)")}, ("action",)),
    _tool("http_request", "Make an HTTP request (test an API you built, read docs). Redirects are not followed.",
          {"url": _s("http(s) URL"), "method": _s("GET, POST, PUT, PATCH, DELETE"), "headers": {"type": "object", "description": "Headers"},
           "body": _s("Request body (JSON string or text)"), "timeout": _i("Seconds (default 20)")}, ("url",)),
    _tool("web_search", "Search the web for current information, docs, errors, library versions or news. Returns titles, URLs and snippets. Follow up with web_fetch to read a result page.",
          {"query": _s("Search query (short and specific)"), "max_results": _i("Number of results 1-10 (default 6)")}, ("query",)),
    _tool("web_fetch", "Fetch a web page and return its readable text (HTML is stripped). Use on URLs from web_search or given by the user.",
          {"url": _s("http(s) URL"), "max_chars": _i("Max characters to return (default 8000)")}, ("url",)),
    _tool("screenshot", "Render a web page with a headless browser and send the screenshot to the user (great for showing a finished UI). Needs playwright.",
          {"url": _s("URL, e.g. http://localhost:8000"), "full_page": _b("Capture the full page")}, ("url",)),
    _tool("send_file", "Send a project file to the user in Telegram (zip, report, build artifact, image).",
          {"path": _s("File path"), "caption": _s("Short caption")}, ("path",)),
    _tool("ask_user", "Ask the user ONE focused question when blocked or when a decision is costly to reverse. Offer 2-4 options when possible. Do not ask for things you can decide yourself.",
          {"question": _s("The question"), "options": {"type": "array", "items": {"type": "string"}, "description": "Short answer choices"}}, ("question",)),
    _tool("use_skill", "Load a reusable playbook by name. Call with an empty name to list available skills.",
          {"name": _s("Skill name")}),
    _tool("save_skill", "Save a reusable playbook you learned (procedure, conventions, gotchas) so future tasks can load it.",
          {"name": _s("Short kebab-case name"), "description": _s("One line"), "content": _s("Markdown instructions")}, ("name", "description", "content")),
    _tool("remember", "Store a durable fact about this project (stack, commands, conventions, decisions) in project memory.",
          {"note": _s("One concise fact")}, ("note",)),
]
TOOL_PROPS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"]) for t in TOOLS}
TOOL_REQ = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


def describe(name: str, a: dict) -> tuple[str, str]:
    def p(k: str, n: int = 60) -> str:
        return short(str(a.get(k, "") or ""), n)

    table = {
        "update_plan": ("📋", "Updating plan"),
        "list_dir": ("📂", f"List {p('path') or '.'}"),
        "read_file": ("📖", f"Read {p('path')}"),
        "write_file": ("📝", f"Write {p('path')}"),
        "edit_file": ("✏️", f"Edit {p('path')}"),
        "search_code": ("🔍", f"Search “{p('pattern', 40)}”"),
        "run_command": ("💻", f"$ {p('command', 70)}"),
        "process": ("⚙️", f"Process {p('action')} {p('id')}".strip()),
        "http_request": ("🌐", f"{p('method') or 'GET'} {p('url', 60)}"),
        "web_search": ("🔎", f"Web search “{p('query', 50)}”"),
        "web_fetch": ("📄", f"Fetch {p('url', 60)}"),
        "screenshot": ("📸", f"Screenshot {p('url', 50)}"),
        "send_file": ("📤", f"Send {p('path')}"),
        "ask_user": ("❓", "Asking you"),
        "use_skill": ("🧠", f"Skill {p('name') or '(list)'}"),
        "save_skill": ("🧠", f"Save skill {p('name')}"),
        "remember": ("💾", "Saving project note"),
    }
    return table.get(name, ("🔧", name))


# ============================================================================
# TOOLBOX  (tool implementations — every path goes through Workspace.resolve)
# ============================================================================


def _lint(path: Path, text: str) -> str:
    try:
        if path.suffix == ".py":
            compile(text, str(path), "exec")
        elif path.suffix == ".json":
            json.loads(text)
    except SyntaxError as e:
        return f"\n⚠ SyntaxError at line {e.lineno}: {e.msg}"
    except ValueError as e:
        return f"\n⚠ Invalid JSON: {e}"
    return ""


def _blocked_host(host: str) -> bool:
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(sa[0])
            if ip.is_link_local or str(ip) in ("169.254.169.254", "100.100.100.200"):
                return True
    except (socket.gaierror, ValueError):
        return False
    return False


class ToolBox:
    def __init__(self, bot: "Bot", run: Run):
        self.bot, self.run, self.ws = bot, run, run.ws
        self.step_no = 0

    async def execute(self, tc: ToolCall, step_no: int) -> str:
        run = self.run
        self.step_no = step_no
        run.tool_calls += 1
        icon, label = describe(tc.name, tc.args)
        st = run.progress.begin(icon, label) if run.progress else Step(0, icon, label, time.time())
        fn = getattr(self, f"t_{tc.name}", None)
        ok, res = True, ""
        try:
            if fn is None or tc.name not in TOOL_PROPS:
                raise ToolError(f"unknown tool '{tc.name}'. Available: {', '.join(TOOL_PROPS)}")
            if "__invalid__" in tc.args:
                raise ToolError("tool arguments were not valid JSON — resend the call with valid JSON")
            missing = [k for k in TOOL_REQ[tc.name] if k not in tc.args]
            if missing:
                raise ToolError(f"missing required argument(s): {', '.join(missing)}")
            args = {k: v for k, v in tc.args.items() if k in TOOL_PROPS[tc.name]}
            res = str(await fn(**args))
            ok = not res.startswith(("BLOCKED", "DENIED", "ERROR")) and not re.match(r"exit_code: (?!0\b)", res) and not res.startswith("TIMEOUT")
        except ToolError as e:
            ok, res = False, f"ERROR: {e}"
        except asyncio.CancelledError:
            if run.progress:
                run.progress.end(st, False, "cancelled")
            raise
        except Exception as e:  # tool bugs must not kill the task
            log.exception("tool %s crashed", tc.name)
            ok, res = False, f"ERROR: {type(e).__name__}: {e}"
        if run.progress:
            run.progress.end(st, ok)
        return trunc(res, Cfg.MAX_TOOL_OUTPUT, 0.3)

    # ---- planning / memory / skills ---------------------------------------------
    async def t_update_plan(self, items: list) -> str:
        plan = []
        for it in (items or [])[:12]:
            if isinstance(it, str):
                plan.append({"text": short(it, 90), "status": "pending"})
            elif isinstance(it, dict):
                stt = it.get("status", "pending")
                plan.append({"text": short(str(it.get("text", "")), 90), "status": stt if stt in PLAN_STATES else "pending"})
        self.run.plan = plan
        done = sum(1 for p in plan if p["status"] == "done")
        return f"Plan updated ({done}/{len(plan)} done)."

    async def t_remember(self, note: str) -> str:
        f = self.ws.agent_dir / "memory.md"
        lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        lines.append("- " + short(note, 300))
        f.write_text("\n".join(lines[-80:]) + "\n", encoding="utf-8")
        return "Saved to project memory."

    async def t_use_skill(self, name: str = "") -> str:
        sk = Skills(self.run.uid).all()
        if name in sk:
            return f"# Skill: {name}\n{sk[name][1]}"
        return "Available skills:\n" + "\n".join(f"- {k}: {v[0]}" for k, v in sk.items())

    async def t_save_skill(self, name: str, description: str, content: str) -> str:
        return f"Saved skill '{Skills(self.run.uid).save(name, description, content)}'."

    # ---- filesystem ---------------------------------------------------------------
    async def t_list_dir(self, path: str = ".", depth: int = 2) -> str:
        p = self.ws.resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        return f"{self.ws.rel(p)}/\n" + tree(p, max(1, min(int(depth or 2), 4)))

    async def t_read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"file not found: {path}")
        size = p.stat().st_size
        if size > 2_000_000:
            raise ToolError(f"file too large ({size:,} B) — use search_code or run_command with head/tail/sed")
        data = p.read_bytes()
        if is_binary(data):
            return f"{self.ws.rel(p)}: binary file ({size:,} B)"
        lines = data.decode("utf-8", "replace").split("\n")
        s = max(1, int(start_line or 1))
        e = min(int(end_line) if end_line else s + 399, len(lines), s + 799)
        body = "\n".join(f"{i:>5}│{lines[i - 1]}" for i in range(s, e + 1))
        foot = f"\n[more: continue with start_line={e + 1}]" if e < len(lines) else ""
        return f"{self.ws.rel(p)} · lines {s}-{e} of {len(lines)}\n{body}{foot}"

    def _guard_write(self, p: Path) -> None:
        rel = self.ws.rel(p)
        if rel == ".agent" or rel.startswith((".agent/", ".git/")) or rel == ".git":
            raise ToolError("that location is reserved (checkpoints / git internals)")

    async def t_write_file(self, path: str, content: str) -> str:
        p = self.ws.resolve(path)
        self._guard_write(p)
        if p.is_dir():
            raise ToolError(f"{path} is a directory")
        if len(content) > 1_500_000:
            raise ToolError("content too large (>1.5MB)")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.run.touched.add(self.ws.rel(p))
        self.run.last_edit_step = self.step_no
        return f"{'Overwrote' if existed else 'Created'} {self.ws.rel(p)} ({len(content):,} chars, {content.count(chr(10)) + 1} lines).{_lint(p, content)}"

    async def t_edit_file(self, path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
        p = self.ws.resolve(path)
        self._guard_write(p)
        if not p.is_file():
            raise ToolError(f"file not found: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if old_str == new_str:
            raise ToolError("old_str and new_str are identical")
        n = text.count(old_str)
        if n == 0:
            first = old_str.strip().splitlines()[0] if old_str.strip() else ""
            hint = ""
            if first:
                for i, ln in enumerate(text.splitlines(), 1):
                    if first.strip() and first.strip() in ln:
                        hint = f" Closest match starts near line {i}; re-read the file and copy the text exactly (whitespace matters)."
                        break
            raise ToolError("old_str not found." + (hint or " Re-read the file and copy the text exactly."))
        if n > 1 and not replace_all:
            raise ToolError(f"old_str matches {n} places — add more surrounding context or set replace_all=true")
        line = text[: text.index(old_str)].count("\n") + 1
        new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
        p.write_text(new_text, encoding="utf-8")
        self.run.touched.add(self.ws.rel(p))
        self.run.last_edit_step = self.step_no
        return f"Edited {self.ws.rel(p)}: {n if replace_all else 1} replacement(s), first at line {line}.{_lint(p, new_text)}"

    async def t_search_code(self, pattern: str, path: str = ".", glob: str = "", ignore_case: bool = False) -> str:
        base = self.ws.resolve(path)
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I if ignore_case else 0)
        ws = self.ws

        def work() -> str:
            hits: list[str] = []
            files = [base] if base.is_file() else None
            if files is None:
                files = []
                for dp, dn, fn in os.walk(base):
                    dn[:] = [d for d in dn if d not in SKIP_DIRS]
                    for f in fn:
                        if not glob or fnmatch.fnmatch(f, glob):
                            files.append(Path(dp) / f)
            for f in files:
                try:
                    if f.stat().st_size > 1_000_000:
                        continue
                    data = f.read_bytes()
                except OSError:
                    continue
                if is_binary(data):
                    continue
                for i, ln in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                    if rx.search(ln):
                        hits.append(f"{ws.rel(f)}:{i}: {ln.strip()[:200]}")
                        if len(hits) >= 80:
                            return "\n".join(hits) + "\n[more matches omitted — narrow the search]"
            return "\n".join(hits) or "no matches"

        return await asyncio.to_thread(work)

    # ---- shell ---------------------------------------------------------------------
    async def t_run_command(self, command: str, timeout: Optional[int] = None, background: bool = False) -> str:
        run, cmd = self.run, (command or "").strip()
        if not cmd:
            raise ToolError("empty command")
        verdict, why = classify_command(cmd, self.ws.root)
        if verdict == "block":
            return f"BLOCKED: {why}. This is never allowed; choose a safer approach."
        need = run.approval == "ask" or (verdict == "approve" and run.approval != "auto")
        if need and not run.allow_all:
            decision = await self.bot.request_approval(run, cmd, why or "approval mode is set to ask-every-time")
            if decision == "deny":
                return "DENIED by the user. Do not retry this command; use another approach or ask_user."
            if decision == "all":
                run.allow_all = True
        if background:
            return await self._start_bg(cmd)
        tmo = max(1, min(int(timeout or Cfg.CMD_TIMEOUT), Cfg.CMD_TIMEOUT_MAX))
        prog = run.progress

        def on_out(txt: str) -> None:
            if prog:
                prog.tail(txt)

        rc, out, timed_out = await run_shell(cmd, self.ws, tmo, on_out)
        run.last_verify_step = self.step_no
        out = out.strip()
        head = f"TIMEOUT after {tmo}s (process killed); partial output follows" if timed_out else f"exit_code: {rc}"
        return f"{head}\n{trunc(out, Cfg.MAX_TOOL_OUTPUT - 100, 0.25)}" if out else f"{head} (no output)"

    async def _start_bg(self, cmd: str) -> str:
        procs = self.bot.bg.setdefault(self.run.uid, {})
        for pid in [k for k, v in procs.items() if not v.alive and time.time() - v.started > 3600]:
            procs.pop(pid, None)
        if sum(1 for p in procs.values() if p.alive) >= 5:
            raise ToolError("too many background processes (max 5) — stop one first")
        pid = max(procs, default=0) + 1
        proc = await spawn(cmd, self.ws)
        bp = BgProc(pid, cmd, proc, time.time())

        async def rd() -> None:
            assert proc.stdout
            while True:
                data = await proc.stdout.read(4096)
                if not data:
                    break
                bp.buf.extend(data.decode("utf-8", "replace").splitlines())

        bp.reader = asyncio.create_task(rd())
        procs[pid] = bp
        await asyncio.sleep(2.5)
        self.run.last_verify_step = self.step_no
        out = "\n".join(list(bp.buf)[-25:])
        state = "running" if bp.alive else f"already exited (code {proc.returncode})"
        return f"Background process #{pid} (pid {proc.pid}) {state}.\n--- output so far ---\n{out or '(none yet)'}"

    async def t_process(self, action: str, id: Optional[int] = None, tail: int = 40) -> str:
        procs = self.bot.bg.setdefault(self.run.uid, {})
        if action == "list":
            if not procs:
                return "no background processes"
            return "\n".join(f"#{k} {'running' if v.alive else 'exited ' + str(v.proc.returncode)} · {dur(time.time() - v.started)} · {short(v.cmd, 70)}" for k, v in procs.items())
        bp = procs.get(int(id)) if id is not None else None
        if not bp:
            raise ToolError(f"no such process: {id}")
        if action == "output":
            lines = list(bp.buf)[-max(1, min(int(tail or 40), 200)):]
            return f"#{bp.id} {'running' if bp.alive else 'exited ' + str(bp.proc.returncode)}\n" + ("\n".join(lines) or "(no output)")
        if action == "stop":
            if bp.alive:
                await _terminate(bp.proc)
            return f"#{bp.id} stopped."
        raise ToolError("action must be list, output or stop")

    # ---- network / media ------------------------------------------------------------
    async def t_http_request(self, url: str, method: str = "GET", headers: Optional[dict] = None, body: Optional[str] = None, timeout: int = 20) -> str:
        from urllib.parse import urlparse

        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ToolError("only http(s) URLs are allowed")
        if await asyncio.to_thread(_blocked_host, u.hostname):
            raise ToolError("that address is blocked (link-local / cloud metadata)")
        kw: dict = {}
        if body is not None:
            kw["content"] = body if isinstance(body, str) else json.dumps(body)
            if isinstance(headers, dict) and not any(k.lower() == "content-type" for k in headers):
                headers = {**headers, "Content-Type": "application/json"}
        t0 = time.time()
        r = await self.bot.http.request(
            (method or "GET").upper(), url, headers=headers if isinstance(headers, dict) else None,
            timeout=max(1, min(int(timeout or 20), 120)), follow_redirects=False, **kw,
        )
        self.run.last_verify_step = self.step_no
        text = r.text
        try:
            text = json.dumps(r.json(), indent=2, ensure_ascii=False)
        except ValueError:
            pass
        keep = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "location", "set-cookie", "content-length")}
        return f"HTTP {r.status_code} {r.reason_phrase} · {int((time.time() - t0) * 1000)}ms\n{json.dumps(keep)}\n\n{trunc(text, 4500)}"

    async def t_web_search(self, query: str, max_results: int = 6) -> str:
        query = (query or "").strip()
        if not query:
            raise ToolError("empty query")
        n = max(1, min(int(max_results or 6), 10))
        http = self.bot.http
        results: list[tuple[str, str, str]] = []
        try:
            if Cfg.TAVILY_API_KEY:
                r = await http.post("https://api.tavily.com/search", timeout=25.0,
                                    json={"api_key": Cfg.TAVILY_API_KEY, "query": query, "max_results": n})
                r.raise_for_status()
                for it in r.json().get("results", [])[:n]:
                    results.append((it.get("title", ""), it.get("url", ""), it.get("content", "")))
            elif Cfg.BRAVE_API_KEY:
                r = await http.get("https://api.search.brave.com/res/v1/web/search", timeout=25.0,
                                   params={"q": query, "count": n},
                                   headers={"X-Subscription-Token": Cfg.BRAVE_API_KEY, "Accept": "application/json"})
                r.raise_for_status()
                for it in (r.json().get("web") or {}).get("results", [])[:n]:
                    results.append((it.get("title", ""), it.get("url", ""), re.sub(r"<[^>]+>", "", it.get("description", ""))))
            else:
                # Keyless fallback: DuckDuckGo HTML endpoint
                r = await http.post("https://html.duckduckgo.com/html/", data={"q": query}, timeout=25.0,
                                    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
                                    follow_redirects=True)
                r.raise_for_status()
                page = r.text
                pat = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(?:.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>)?', re.S)
                for href, title, snip in pat.findall(page):
                    href = html.unescape(href)
                    if "uddg=" in href:
                        href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
                    elif href.startswith("//"):
                        href = "https:" + href
                    clean = lambda t: html.unescape(re.sub(r"<[^>]+>", "", t or "")).strip()
                    results.append((clean(title), href, clean(snip)))
                    if len(results) >= n:
                        break
        except httpx.HTTPError as e:
            raise ToolError(f"search failed: {type(e).__name__}: {e}")
        if not results:
            return f"No results for: {query}"
        out = [f"Search results for: {query}"]
        for i, (title, url, snip) in enumerate(results, 1):
            out.append(f"{i}. {title}\n   {url}\n   {short(snip, 300)}")
        self.run.last_verify_step = self.step_no
        return "\n".join(out)

    async def t_web_fetch(self, url: str, max_chars: int = 8000) -> str:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ToolError("only http(s) URLs are allowed")
        if await asyncio.to_thread(_blocked_host, u.hostname):
            raise ToolError("that address is blocked (link-local / cloud metadata)")
        try:
            r = await self.bot.http.get(url, timeout=25.0, follow_redirects=True,
                                        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                                                 "Accept": "text/html,application/json,text/plain,*/*"})
        except httpx.HTTPError as e:
            raise ToolError(f"fetch failed: {type(e).__name__}: {e}")
        final = urlparse(str(r.url)).hostname
        if final and final != u.hostname and await asyncio.to_thread(_blocked_host, final):
            raise ToolError("redirected to a blocked address")
        ctype = r.headers.get("content-type", "")
        text = r.text
        if "html" in ctype or text.lstrip().startswith(("<!", "<html")):
            text = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer|form)[^>]*>.*?</\1>", " ", text)
            text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|pre|section|article)>", "\n", text)
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        limit = max(500, min(int(max_chars or 8000), 20000))
        self.run.last_verify_step = self.step_no
        return f"HTTP {r.status_code} · {url}\n\n{trunc(text, limit)}"

    async def t_screenshot(self, url: str, full_page: bool = False) -> str:
        from urllib.parse import urlparse

        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname or await asyncio.to_thread(_blocked_host, u.hostname):
            raise ToolError("invalid or blocked URL")
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            return "ERROR: playwright is not installed. Run: pip install playwright && playwright install --with-deps chromium"
        out = self.ws.agent_dir / "shots"
        out.mkdir(exist_ok=True)
        path = out / f"shot_{int(time.time())}.png"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            try:
                page = await browser.new_page(viewport={"width": 1280, "height": 800})
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.screenshot(path=str(path), full_page=bool(full_page))
            finally:
                await browser.close()
        self.run.last_verify_step = self.step_no
        await self.bot.tg.send_file(self.run.chat_id, path, f"📸 {url}", self.run.thread)
        return f"Screenshot captured and sent to the user ({path.stat().st_size:,} B)."

    async def t_send_file(self, path: str, caption: str = "") -> str:
        p = self.ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"file not found: {path}")
        if p.stat().st_size > 49_000_000:
            raise ToolError("file exceeds Telegram's 50 MB bot upload limit")
        await self.bot.tg.send_file(self.run.chat_id, p, caption, self.run.thread)
        return f"Sent {self.ws.rel(p)} to the user."

    async def t_ask_user(self, question: str, options: Optional[list] = None) -> str:
        ans = await self.bot.ask_user(self.run, question, [str(o) for o in (options or [])][:8])
        return f"User answered: {ans}" if ans else "The user did not answer in time. Make a sensible assumption, state it, and continue."


# ============================================================================
# AGENT LOOP
# ============================================================================

SYSTEM_PROMPT = """You are an autonomous senior software engineer working inside the user's private project workspace, controlled from Telegram.
You ACT with tools instead of describing what you would do. Today is {today}. Platform: {platform}.

## Task Mode
Every ordinary message received inside a session is a task. Never classify it as CHAT, send it to a separate chat-only loop, or require special task keywords. This includes short requests, questions, research, explanations, translations, and follow-up instructions: handle them in this task run, using the appropriate tools and returning the result directly.
Work across websites, Android applications and APKs, Telegram/other bots, scripts, APIs, automation, documents, and projects in any programming language supported by the available runtimes. Inspect the workspace and installed toolchain first; choose the project's existing stack where possible, install or configure needed tooling when appropriate, then build, test, debug, and provide the resulting source or artifact. For APK requests, check Java/Gradle/Android SDK availability and build an installable artifact when the environment permits. If a required SDK, credential, account, device, or hosting capability is unavailable, complete the parts that are possible and state the specific blocker; never claim an artifact or deployment exists without verifying it.
For an APK request, do not report success unless a non-empty APK containing an Android manifest and DEX code was actually built and sent to the user with send_file. If this cannot be completed, report the verified blocker instead of saying "Done".

## Workflow
1. UNDERSTAND the request. Ask (ask_user) only if blocked or a wrong guess would be costly; otherwise decide, state the assumption, proceed.
2. PLAN: for anything beyond a trivial change call update_plan (3-7 concrete steps) and keep statuses current (one in_progress at a time).
3. EXPLORE before editing: list_dir, search_code, read_file. Never guess file contents or APIs.
4. IMPLEMENT with small targeted edit_file changes; write_file for new files. Deliver complete, runnable code — no placeholders or "TODO: implement".
5. VERIFY: run the tests/linter/build or execute the program and read the real output. If it fails, read the error, fix the ROOT cause, run again. Repeat until it passes or you are genuinely stuck. Never claim success without evidence from a tool result.
6. REPORT: short and factual.

## Rules
- ONE user request = ONE task. Inside it, run as many terminal commands as you need (install, build, test, fix, re-run) — do not stop after a single command and do not ask the user to run commands for you. Chain related commands with && to save steps. Only finish when the whole job is verified.
- Paths are relative to the project root. read_file line numbers are display-only; never put them in edit_file.
- Commands run non-interactively (use -y/--yes flags; no editors, pagers or prompts). Servers/watchers: run_command with background=true, check with process, and verify with http_request/screenshot. Stop servers you started unless the user wants them running.
- Web projects: when the user asks for Preview, detect the project's start command, run it in the background, verify the HTTP endpoint, and report the preview URL/port. Keep it running only when the user explicitly asks for a live preview.
- Bot projects: when the user asks for Sandbox or "1Min Test Host", run the bot/test command in the background, inspect its output for up to 60 seconds, then stop it and report the result. Do not turn this into a permanent host.
- If a command is DENIED or BLOCKED, never retry it; adapt or ask.
- Use web_search for anything current or unknown (library versions, API docs, error messages, news), then web_fetch to read the best result. Cite source URLs in your answer.
- Never print or exfiltrate secrets. Keep credentials in env vars or git-ignored files.
- When a user attaches an image, inspect the provided visual input and use the saved uploads path if useful.
- If the same action fails twice, change approach instead of repeating it.
- Messages starting with "[User update while you were working]" are new instructions: adapt immediately.
- Prefer the standard library and few dependencies. Match the existing code style.
- Save durable project facts with remember; save reusable procedures with save_skill.

## Final answer format
Reply in the SAME LANGUAGE as the user's latest message (code, identifiers and commands stay in English). Markdown, concise:
what you did · how to run/use it (exact commands) · how you verified it (command + result) · caveats or next steps. No filler, no restating the plan.
{skills}{memory}{guide}
## Current project: {project}
Top-level layout:
{tree}
"""


_ACTION_VERB_RE = re.compile(
    r"(?i)\b(create|build|make|write|add|fix|implement|refactor|deploy|install|run|test|debug|edit|update|delete|remove|rename|generate|convert|migrate|compile|package|develop|repair)\b"
)
_ACTION_TARGET_RE = re.compile(
    r"(?i)\b(apk|android|app|application|website|web\s*site|bot|script|code|program|file|project|report|document|api|database|bug|error|test|build|release|package|server)\b"
)
_QUESTION_START_RE = re.compile(r"(?i)^\s*(how|what|why|explain|describe|translate|summari[sz]e|tell\s+me)\b")
_APK_REQUEST_RE = re.compile(r"(?i)\b(apk|android\s+app)\b")
_APK_ACTION_RE = re.compile(r"(?i)\b(create|build|make|write|implement|generate|compile|package|develop)\b")
_ACTION_EVIDENCE_TOOLS = {"write_file", "edit_file", "run_command", "process", "http_request", "screenshot", "send_file"}


def _requires_execution(goal: str) -> bool:
    text = (goal or "").strip()
    if _QUESTION_START_RE.search(text):
        return False
    has_english_action = bool(_ACTION_VERB_RE.search(text) and _ACTION_TARGET_RE.search(text))
    has_burmese_action = bool(re.search(r"ဆောက်|တည်ဆောက်|ဖန်တီး|ရေးပေး|ပြင်ပေး|ထည့်ပေး|ဖျက်ပေး|တပ်ဆင်|စမ်းသပ်|ကုဒ်", text))
    return has_english_action or has_burmese_action


def _is_apk_build_request(goal: str) -> bool:
    text = (goal or "").strip()
    return not _QUESTION_START_RE.search(text) and bool(_APK_REQUEST_RE.search(text) and _APK_ACTION_RE.search(text))


def _find_valid_apk(root: Path) -> Optional[Path]:
    root = root.resolve()
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".agent", ".git", ".venv", "venv", "node_modules"}]
        for name in files:
            if not name.lower().endswith(".apk"):
                continue
            path = Path(base) / name
            try:
                if path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
                    continue
                with zipfile.ZipFile(path) as apk:
                    names = set(apk.namelist())
                if "AndroidManifest.xml" in names and any(re.fullmatch(r"classes(?:\d+)?\.dex", n) for n in names):
                    return path
            except (OSError, zipfile.BadZipFile):
                continue
    return None


class Agent:
    def __init__(self, bot: "Bot", run: Run):
        self.bot, self.run = bot, run

    def _system(self) -> str:
        import datetime
        import platform

        ws = self.run.ws
        mem = ""
        mf = ws.agent_dir / "memory.md"
        if mf.exists():
            mem = "\n## Project memory\n" + mf.read_text(encoding="utf-8")[-3000:]
        guide = ""
        for name in ("AGENTS.md", "AGENT.md", "CLAUDE.md"):
            gf = ws.root / name
            if gf.is_file():
                guide = f"\n## Project instructions ({name})\n" + gf.read_text(encoding="utf-8", errors="replace")[:3000]
                break
        sk = Skills(self.run.uid).all()
        skills = "\n## Skills (load with use_skill)\n" + "; ".join(f"{k} — {v[0]}" for k, v in sk.items()) + "\n"
        return SYSTEM_PROMPT.format(
            today=datetime.date.today().isoformat(),
            platform=f"{platform.system()} {platform.machine()}",
            skills=skills, memory=mem, guide=guide, project=ws.project, tree=tree(ws.root, 2, 60),
        )

    @staticmethod
    def _friendly(e: LLMError, provider: str) -> str:
        if e.kind == "auth":
            if provider in ("antigravity", "opencode"):
                return f"{provider} CLI ကို အကောင့်ဝင်ရာ သို့မဟုတ် ပြင်ဆင်ရာမှာ ပြဿနာရှိနေပါတယ်။ Bot နဲ့ တစ်ဦးချင်းစကားပြောခန်းမှာ /settings ကိုဖွင့်ပြီး Connect my provider ကို နှိပ်ကာ ပြန်ချိတ်ပါ။"
            return f"{provider} ဝန်ဆောင်မှုက API key သို့မဟုတ် သုံးခွင့်ကို လက်မခံပါ (HTTP {e.status})။ Key မှားခြင်း၊ သက်တမ်းကုန်ခြင်း သို့မဟုတ် API သုံးခွင့်/လက်ကျန်မရှိခြင်း ဖြစ်နိုင်ပါတယ်။ Bot နဲ့ တစ်ဦးချင်းစကားပြောခန်းထဲက /settings → Connect my provider ကနေ ပြန်ချိတ်ပါ။ API key ကို ဒီနေရာမှာ မပို့ပါနဲ့။"
        if e.kind == "rate":
            return "The LLM provider is rate-limiting requests. Try again in a minute."
        return str(e)

    _CLI_BRIEF = (
        "You are Agent X, an autonomous senior software engineer working for a user through a Telegram coding-agent bot.\n"
        "Your CURRENT WORKING DIRECTORY is the user's private session workspace. Do ALL work inside it: create and edit files, "
        "run shell commands, install dependencies, build, test and fix problems until the result really works.\n"
        "Rules:\n"
        "- Never ask the user questions. Make sensible assumptions, state them briefly, and finish the job.\n"
        "- Never claim success without evidence: run the code/build/tests and read the output.\n"
        "- Keep every project file inside the current directory. Do not touch files outside it.\n"
        "- For Android / APK requests: check `java -version`, `gradle`, `ANDROID_HOME`/`ANDROID_SDK_ROOT` first. If something is missing, "
        "install it without sudo (e.g. download the official Android command-line tools and use sdkmanager). Build a real, installable "
        "APK (debug build is fine) and keep the final file in the workspace (e.g. ./app-debug.apk). If the toolchain truly cannot be "
        "installed here, say exactly which part is blocked - do not pretend an APK exists.\n"
        "- For websites/apps, produce complete working files (not stubs) and a short README with run instructions.\n"
        "- Final reply: reply in the SAME LANGUAGE as the user's request, concise Markdown: what you built, key files, how to run it, "
        "and any limitation.\n"
    )

    async def _execute_cli(self) -> tuple[str, str]:
        """Antigravity / OpenCode: the CLI is itself a full agent, so run it inside the session workspace."""
        run, bot = self.run, self.bot
        root = Path(run.ws.root)
        root.mkdir(parents=True, exist_ok=True)
        name = run.llm.pc["name"]
        before = _scan_tree(root)
        hist = bot.db.history(run.uid, run.ws.project, 6)
        while hist and hist[0]["role"] != "user":
            hist.pop(0)
        ctx = "\n".join(f"[{h['role']}] {short(str(h['content']), 700)}" for h in hist)
        prompt = self._CLI_BRIEF
        if ctx:
            prompt += "\n## Earlier conversation in this session\n" + ctx + "\n"
        if run.images:
            prompt += "\n(The user attached image(s); they are not available to this CLI - ask for a text description only if essential.)\n"
        prompt += "\n## Task\n" + run.goal + "\n"

        st = run.progress.begin("🛠", f"{name} agent working in session {run.ws.project}")
        run.progress.set_status(f"{name} agent working…")

        def on_line(t: str) -> None:
            run.progress.set_status("Working · " + short(t, 70))
            run.progress.feed(t)

        stop_watch = asyncio.Event()

        async def watch() -> None:
            """Live feed: report every file the CLI creates/changes/deletes, plus its log lines."""
            seen = dict(before)
            started = time.time()
            log_file: Optional[Path] = None
            log_pos = 0
            log_dir = Path.home() / ".gemini" / "antigravity-cli" / "log"
            keep = re.compile(r"(?i)(tool|command|exec|write|edit|read|search|fetch|run_|shell|create|build|install)")
            while not stop_watch.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_watch.wait(), 2.0)
                try:
                    cur = await asyncio.to_thread(_scan_tree, root)
                    ev = [("📝", ("Created " if k not in seen else "Updated ") + k) for k, v in cur.items() if seen.get(k) != v]
                    ev += [("🗑", "Deleted " + k) for k in seen if k not in cur]
                    seen = cur
                    for icon, label in ev[:6]:
                        run.progress.note(icon, label)
                    if len(ev) > 6:
                        run.progress.note("📝", f"…and {len(ev) - 6} more file change(s)")
                    if name == "antigravity" and log_dir.is_dir():  # best-effort: agy's own activity log
                        if log_file is None:
                            fresh = [f for f in log_dir.glob("cli-*.log") if f.stat().st_mtime >= started - 2]
                            if fresh:
                                log_file = max(fresh, key=lambda f: f.stat().st_mtime)
                                log_pos = log_file.stat().st_size if log_file.stat().st_mtime < started else 0
                        if log_file is not None:
                            with log_file.open("rb") as fh:
                                fh.seek(log_pos)
                                chunk = fh.read(20000)
                                log_pos = fh.tell()
                            lines = [l for l in chunk.decode("utf-8", "replace").splitlines() if keep.search(l)]
                            if lines:
                                run.progress.feed("\n".join(lines[-3:]))
                    run.progress.set_status(f"{name} agent working · {dur(time.time() - started)}")
                except Exception as e:  # the watcher must never break the task
                    log.debug("live watcher: %s", e)

        watcher = asyncio.create_task(watch())
        try:
            final = await run.llm.cli_agent(prompt, root, on_line)
        except LLMError as e:
            run.progress.end(st, False, short(str(e), 80))
            raise AgentFailure(self._friendly(e, name) + "\n\n" + short(str(e), 500))
        finally:
            stop_watch.set()
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher
        run.progress.end(st, True)

        after = _scan_tree(root)
        changed = sorted(k for k, v in after.items() if before.get(k) != v)
        deleted = [k for k in before if k not in after]
        run.touched.update(changed)
        run.tool_calls = len(changed) + len(deleted)
        run.last_edit_step = run.last_verify_step = 0

        failure = ""
        if _is_apk_build_request(run.goal):
            apk = _find_valid_apk(root)
            if apk is None:
                failure = "APK မတည်ဆောက်နိုင်ခဲ့ပါ — AndroidManifest + classes.dex ပါတဲ့ APK မတွေ့ပါ။ (Java/Gradle/Android SDK မရှိခြင်း ဖြစ်နိုင်ပါတယ်)\n\n" + short(final, 1500)
            else:
                size = apk.stat().st_size
                if size > 49_000_000:
                    final += f"\n\n📦 APK ကို `{apk.relative_to(root)}` မှာ ထားပါတယ် ({size // 1_000_000} MB — Telegram upload limit ကျော်လို့ /get နဲ့ယူပါ)။"
                else:
                    with contextlib.suppress(Exception):
                        await bot.tg.send_file(run.chat_id, apk, f"📱 {apk.name}", run.thread)
                    run.tool_calls += 1
        elif _requires_execution(run.goal) and not changed and not deleted:
            failure = (f"{name} က workspace ထဲမှာ ဖိုင်တစ်ခုမှ မဖန်တီး/မပြင်ခဲ့ပါ။ ဒါကြောင့် Task ကို မအောင်မြင်ဟု သတ်မှတ်ပါတယ်။\n\n"
                       + short(final, 1500))
        bot.db.add_history(run.uid, run.ws.project, "user", run.goal)
        bot.db.add_history(run.uid, run.ws.project, "assistant", failure or final)
        if failure:
            return "failed", failure
        return "done", final[-9000:] if len(final) > 9000 else final

    async def execute(self) -> tuple[str, str]:
        run, bot = self.run, self.bot
        if run.llm is None:
            raise AgentFailure("No LLM provider is configured. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY in .env and restart.")
        cp = Checkpoints(run.ws)
        run.progress.set_status("Creating checkpoint")
        run.cp_start = await cp.snapshot(f"task {run.id}: start")
        if run.llm.pc.get("kind") == "cli":
            return await self._execute_cli()
        system = self._system()
        messages: list[dict] = bot.db.history(run.uid, run.ws.project, Cfg.HISTORY_TURNS)
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        messages.append({"role": "user", "content": user_content(run.goal, run.images)})
        tb = ToolBox(bot, run)
        nudged = ctx_retry = empty_retry = False
        completion_nudges = 0
        native_fallback_used = False
        successful_tools: set[str] = set()
        recent: deque = deque(maxlen=5)

        for step in range(1, Cfg.MAX_STEPS + 1):
            while run.inbox:
                update = run.inbox.pop(0)
                if isinstance(update, dict):
                    update_text = str(update.get("text") or "")
                    update_images = update.get("images") or []
                else:
                    update_text, update_images = str(update), []
                messages.append({
                    "role": "user",
                    "content": user_content(f"[User update while you were working]\n{update_text}", update_images),
                })
            run.progress.set_status(f"Thinking · step {step}/{Cfg.MAX_STEPS}")
            try:
                res = await run.llm.chat(system, compact_messages(messages, Cfg.CONTEXT_BUDGET_CHARS), TOOLS)
            except LLMError as e:
                if _requires_execution(run.goal) and not native_fallback_used and run.llm.pc.get("kind") == "cli":
                    fallback_pc = native_tool_fallback(run.uid, run.llm.pc.get("name", ""))
                    if fallback_pc:
                        log.info("task #%s switching to native tool provider %s after CLI error", run.id, fallback_pc["name"])
                        run.llm = LLM(run.llm.http, fallback_pc)
                        native_fallback_used = True
                        messages.append({"role": "user", "content": "[provider fallback] The previous provider failed. Continue this same task using the available native tools; do not restart from scratch."})
                        continue
                if e.kind == "context" and not ctx_retry:
                    ctx_retry = True
                    messages = compact_messages(messages, Cfg.CONTEXT_BUDGET_CHARS // 3, keep_last=6)
                    continue
                raise AgentFailure(self._friendly(e, run.llm.pc["name"]))
            run.llm_calls += 1
            run.tin += res.tin
            run.tout += res.tout
            if res.calls and (res.text or "").strip():
                run.progress.note("💬", short(res.text.strip().replace("\n", " "), 70))

            # A new user message can arrive while the model call is in flight.
            # If the model otherwise finished, give that update priority instead
            # of completing the run and silently leaving it in the inbox.
            if not res.calls and run.inbox:
                messages.append({"role": "assistant", "content": res.text or "(no output)"})
                continue

            if res.calls:
                messages.append({
                    "role": "assistant", "content": res.text or None,
                    "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}} for c in res.calls],
                })
                for idx, c in enumerate(res.calls):
                    if idx >= Cfg.MAX_CALLS_PER_TURN:
                        result = f"ERROR: too many tool calls in one turn (max {Cfg.MAX_CALLS_PER_TURN}). Issue fewer calls."
                    else:
                        result = await tb.execute(c, step)
                        sig = (c.name, json.dumps(c.args, sort_keys=True, ensure_ascii=False))
                        failed = result.startswith(("ERROR", "BLOCKED", "DENIED", "TIMEOUT")) or bool(re.match(r"exit_code: (?!0\b)", result))
                        recent.append((sig, failed))
                        if not failed:
                            successful_tools.add(c.name)
                        if failed and sum(1 for s, f in recent if s == sig and f) >= 3:
                            result += "\n\nNOTE: you have repeated this exact failing call 3 times. Stop and try a different approach."
                    messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})
                continue

            final = (res.text or "").strip()
            if not final and not empty_retry:
                empty_retry = True
                messages.append({"role": "user", "content": "[system] Your last reply was empty. Continue the work or give the final answer."})
                continue
            action_required = _requires_execution(run.goal)
            apk_required = _is_apk_build_request(run.goal)
            action_evidence = bool(successful_tools & _ACTION_EVIDENCE_TOOLS)
            apk_path = _find_valid_apk(run.ws.root) if apk_required else None
            apk_delivered = apk_required and apk_path is not None and "send_file" in successful_tools
            completion_missing = (apk_required and not apk_delivered) or (action_required and not apk_required and not action_evidence)
            if completion_missing:
                if action_required and not native_fallback_used and run.llm.pc.get("kind") == "cli":
                    fallback_pc = native_tool_fallback(run.uid, run.llm.pc.get("name", ""))
                    if fallback_pc:
                        log.info("task #%s switching to native tool provider %s after no-action completion", run.id, fallback_pc["name"])
                        run.llm = LLM(run.llm.http, fallback_pc)
                        native_fallback_used = True
                        messages.append({"role": "assistant", "content": final or "(no final answer)"})
                        messages.append({"role": "user", "content": "[provider fallback] The prior provider returned without doing the requested work. Use native function tools now to complete and verify the task; do not claim success until the work is done."})
                        continue
                if completion_nudges < 2:
                    completion_nudges += 1
                    messages.append({"role": "assistant", "content": final or "(no final answer)"})
                    if apk_required:
                        instruction = "This APK request is NOT complete. Verify the Android build toolchain, create and build the project, confirm a valid non-empty APK contains AndroidManifest.xml and classes.dex, then send that APK with send_file. Do not say Done without the artifact. If blocked, use tools to verify the exact blocker and report it."
                    else:
                        instruction = "This request requires an actual action, but no successful action tool has completed it. Use the available tools to do the work and verify the result. Do not claim completion without evidence; if blocked, verify and explain the specific blocker."
                    messages.append({"role": "user", "content": "[completion check] " + instruction})
                    continue
                if apk_required:
                    failure = "APK task not completed: no verified APK artifact was created and sent. Last response: " + (final[:700] or "(empty)")
                else:
                    failure = "Task not completed: no successful action tool was used to do the requested work. Last response: " + (final[:700] or "(empty)")
                bot.db.add_history(run.uid, run.ws.project, "user", run.goal)
                bot.db.add_history(run.uid, run.ws.project, "assistant", failure)
                return "failed", failure
            if run.last_edit_step >= 0 and run.last_verify_step < run.last_edit_step and not nudged:
                nudged = True
                messages.append({"role": "assistant", "content": final or "(done)"})
                messages.append({"role": "user", "content": "[system check] You changed files but have not run anything since your last edit. "
                                 "Verify now (tests / build / run the program), or state precisely why verification is impossible, then give the final answer."})
                continue
            final = final or "Done."
            bot.db.add_history(run.uid, run.ws.project, "user", run.goal)
            bot.db.add_history(run.uid, run.ws.project, "assistant", final)
            return "done", final

        bot.db.add_history(run.uid, run.ws.project, "user", run.goal)
        note = f"I reached the {Cfg.MAX_STEPS}-step limit. Everything done so far is kept — tap **Continue** to carry on."
        bot.db.add_history(run.uid, run.ws.project, "assistant", note)
        return "limit", note


# ============================================================================
# BOT  (update routing, approvals, final report, commands)
# ============================================================================


@dataclass
class Pending:
    run: Run
    fut: asyncio.Future
    kind: str  # ap | ask
    options: list = field(default_factory=list)


def zip_project(ws: Workspace) -> Path:
    out_dir = Cfg.DATA_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ws.project}-{int(time.time())}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(ws.root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS or d in ("dist", "build")]
            for f in fn:
                p = Path(dp) / f
                with contextlib.suppress(OSError):
                    z.write(p, p.relative_to(ws.root))
    return out


EXIT_TEXT = "Exit Session"


class Bot:
    COMMANDS = [
        ("new", "Create a session"), ("sessions", "Your sessions"), ("exit", "Exit current session"),
        ("files", "Browse session files"),
        ("undo", "Revert a task's changes"), ("stop", "Stop the running task"), ("status", "Agent status"),
        ("settings", "Approval mode & provider"), ("models", "Pick a model (free filter)"),
        ("skills", "Reusable playbooks"), ("clear", "Forget chat context"),
        ("help", "How to use"),
    ]

    def __init__(self) -> None:
        self.tg = TG(Cfg.BOT_TOKEN)
        self.http = httpx.AsyncClient(timeout=30.0)
        self.db = DB(Cfg.DATA_DIR / "bot.db")
        self.db.seed_users()
        self.runs: dict[tuple, Run] = {}
        self.pending: dict[str, Pending] = {}
        self.bg: dict[int, dict[int, BgProc]] = {}
        self.sem = asyncio.Semaphore(Cfg.MAX_CONCURRENT_TASKS)
        self.rl: dict[int, deque] = {}
        self.jobs: set[asyncio.Task] = set()
        self.provider_setup: dict[int, str] = {}
        self.hosts: dict[tuple[int, str], int] = {}
        self.agy_read_locks: dict[int, asyncio.Lock] = {}
        self.oauth_states: dict[str, tuple[int, float]] = {}
        self.oauth_server: Optional[asyncio.AbstractServer] = None
        self.oauth_tokens_file = Cfg.DATA_DIR / "google-oauth-tokens.json"
        self.agy_auth: dict[int, asyncio.subprocess.Process] = {}
        self.agy_install_lock = asyncio.Lock()
        self.opencode_auth: dict[int, asyncio.subprocess.Process] = {}
        # Session mode: uids currently "inside" a session (persisted so a restart keeps menus locked)
        self.session_file = Cfg.DATA_DIR / "in_session.json"
        self.in_session: set[int] = set()
        self.session_setup: set[int] = set()   # waiting for a session name
        self.model_cache: dict[tuple[int, str], tuple[float, list[tuple[str, bool]], bool]] = {}
        try:
            self.in_session = {int(x) for x in json.loads(self.session_file.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            pass

    # ---- lifecycle ---------------------------------------------------------------
    async def start(self) -> None:
        self.tg.me = await self.tg.call("getMe")
        with contextlib.suppress(TGError):
            await self.tg.call("deleteWebhook")
        with contextlib.suppress(TGError):
            await self.tg.call(
                "setMyCommands",
                commands=[{"command": c, "description": d} for c, d in self.COMMANDS],
                scope={"type": "all_private_chats"},
            )
        log.info("Started as @%s · providers=%s · admins=%d", self.tg.me.get("username"), available_providers() or "NONE", len(Cfg.ADMIN_IDS))
        if not Cfg.ADMIN_IDS:
            log.warning("ADMIN_IDS is empty — nobody can use the bot. Set ADMIN_IDS in .env")
        if not available_providers():
            log.warning("No LLM key configured — tasks will fail until you set one in .env")

    async def shutdown(self) -> None:
        if self.oauth_server:
            self.oauth_server.close()
            await self.oauth_server.wait_closed()
        for r in list(self.runs.values()):
            if r.task:
                r.task.cancel()
        for procs in self.bg.values():
            for bp in procs.values():
                if bp.alive:
                    kill_group(bp.proc, signal.SIGTERM)
        for proc in self.agy_auth.values():
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        for proc in self.opencode_auth.values():
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        await asyncio.sleep(0.3)
        await self.tg.close()
        await self.http.aclose()

    def _oauth_redirect_uri(self) -> str:
        return Cfg.OAUTH_PUBLIC_BASE.rstrip("/") + "/oauth/google/callback"

    def _google_login_url(self, uid: int) -> str:
        state = secrets.token_urlsafe(32)
        self.oauth_states[state] = (uid, time.time())
        params = {
            "client_id": GOOGLE_OAUTH["client_id"], "redirect_uri": self._oauth_redirect_uri(),
            "response_type": "code", "scope": " ".join(GOOGLE_OAUTH.get("scopes", ["openid", "https://www.googleapis.com/auth/userinfo.email"])),
            "access_type": "offline", "prompt": "consent", "state": state,
        }
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)

    async def _ensure_agy(self, cid: int, thread: Optional[int]) -> bool:
        """Install the official Linux agy CLI when it is missing.

        The installer is user-scoped and puts the binary at Cfg.AGY_COMMAND's
        default location (~/.local/bin/agy), so no sudo/root access is needed.
        """
        if Path(Cfg.AGY_COMMAND).exists() or shutil.which(Cfg.AGY_COMMAND):
            return True
        async with self.agy_install_lock:
            if Path(Cfg.AGY_COMMAND).exists() or shutil.which(Cfg.AGY_COMMAND):
                return True
            await self.reply(cid, thread, "📦 <b>Antigravity CLI မတွေ့ပါ။</b><br/>Official installer နဲ့ <code>agy</code> ကို auto-install လုပ်နေပါတယ်…")
            try:
                download = await self.http.get("https://antigravity.google/cli/install.sh", timeout=45.0)
                download.raise_for_status()
                script = download.text
                if not script.startswith("#!") or "antigravity" not in script.lower():
                    raise RuntimeError("official installer response was not valid")
                proc = await asyncio.create_subprocess_exec(
                    "bash", "-s",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
                out, _ = await asyncio.wait_for(proc.communicate(script.encode("utf-8")), 180)
                if proc.returncode != 0:
                    detail = out.decode("utf-8", "replace")[-700:]
                    raise RuntimeError(detail or f"installer exited with code {proc.returncode}")
                if not Path(Cfg.AGY_COMMAND).exists() and not shutil.which(Cfg.AGY_COMMAND):
                    raise RuntimeError("installer completed but ~/.local/bin/agy was not found")
                await self.reply(cid, thread, "✅ Antigravity CLI <code>agy</code> ကို auto-install ပြီးပါပြီ။ Google login ကို စတင်နေပါတယ်…")
                return True
            except Exception as e:
                log.warning("Antigravity CLI auto-install failed: %s", e)
                await self.reply(cid, thread, f"❌ Antigravity CLI auto-install မအောင်မြင်ပါ။<br/><code>{esc(str(e)[-700:])}</code>")
                return False

    async def cmd_google_login(self, uid: int, cid: int, thread: Optional[int]) -> None:
        """Start Antigravity CLI's own Google auth, not a custom OAuth app flow."""
        if not await self._ensure_agy(cid, thread):
            return
        old = self.agy_auth.get(uid)
        if old and old.returncode is None:
            await self.reply(cid, thread, "⏳ Antigravity login တစ်ခု လုပ်နေပြီးသားပါ။ အပေါ်က URL ကိုဖွင့်ပြီး code ကို ဒီ chat ထဲ ပြန်ပို့ပါ။")
            return
        try:
            # agy's sign-in screen is a TTY application.  With ordinary pipes
            # it prints only "Run agy to log in" after a failed attempt and no
            # remote OAuth URL.  util-linux `script` gives it a real PTY while
            # keeping the child streams available to the bot.
            script_bin = shutil.which("script")
            if script_bin:
                proc = await asyncio.create_subprocess_exec(
                    script_bin, "-qefc", shlex.quote(Cfg.AGY_COMMAND), "/dev/null",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    Cfg.AGY_COMMAND, "-p", "reply only AGENT_X_AUTH_OK",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
        except OSError as e:
            log.warning("could not start Antigravity auth process: %s", e)
            await self.reply(cid, thread, f"❌ Antigravity login စတင်မရပါ။<br/><code>{esc(str(e))}</code>")
            return
        self.agy_auth[uid] = proc
        self.agy_read_locks[uid] = asyncio.Lock()
        self.spawn(self._watch_agy_auth(uid, cid, thread, proc))
        await self.reply(cid, thread,
                         "🔐 <b>Antigravity CLI Google login</b><br/>အောက်က URL ကို browser နဲ့ဖွင့်ပြီး Google account ဖြင့် login လုပ်ပါ။ ပြီးရင် Google ပြတဲ့ authorization code ကို ဒီ chat ထဲ စာသားအဖြစ် ပြန်ပို့ပါ။<br/><i>ဒါက Antigravity CLI ရဲ့ official remote login flow ဖြစ်ပြီး custom website OAuth မဟုတ်ပါ။</i>")

    async def _watch_agy_auth(self, uid: int, cid: int, thread: Optional[int], proc: asyncio.subprocess.Process) -> None:
        url = ""
        output = ""
        oauth_selected = False
        try:
            # Browser/device login output can be delayed while the CLI starts
            # its local auth server.  Twenty seconds was too short on remote
            # hosts, and only reading stderr missed URLs printed on stdout.
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and proc.stdout:
                remaining = max(0.2, deadline - time.monotonic())
                lock = self.agy_read_locks.get(uid)
                if lock:
                    async with lock:
                        line = await asyncio.wait_for(proc.stdout.readline(), remaining)
                else:
                    line = await asyncio.wait_for(proc.stdout.readline(), remaining)
                if not line:
                    break
                # Strip terminal control sequences and punctuation that the
                # CLI may place around a URL (e.g. "Open: <url>").
                output += line.decode("utf-8", "replace")
                if not oauth_selected and re.search(r"Use a Google Cloud project", output, re.I):
                    # The first menu item is Google OAuth.  Wait until the
                    # prompt is actually rendered; sending Enter at process
                    # startup is too early for the TUI and gets discarded.
                    if proc.stdin:
                        proc.stdin.write(b"\r")
                        await proc.stdin.drain()
                    oauth_selected = True
                url = extract_agy_login_url(output)
                if url:
                    break
            if url:
                await self.reply(cid, thread,
                                 "🌐 <b>ဒီ URL ကိုဖွင့်ပါ</b><br/>Google login ပြီး authorization code ရလာရင် code ကို ဒီ chat ထဲ ပို့ပါ။",
                                 kb([btn("Open Antigravity Google Login", url=url)]))
            else:
                await self.reply(cid, thread, "❌ Antigravity CLI login URL မထုတ်နိုင်ပါ။ <code>agy</code> authentication process ကို ပြန်စပါ။")
                with contextlib.suppress(Exception):
                    if proc.returncode is None:
                        proc.kill()
        except Exception as e:
            log.warning("Antigravity auth URL failed: %s", e)
            await self.reply(cid, thread, "❌ Antigravity login စတင်မရပါ။ နောက်တစ်ကြိမ် ပြန်စမ်းပါ။")
            with contextlib.suppress(Exception):
                if proc.returncode is None:
                    proc.kill()
        finally:
            # Do not leave a dead process in the auth map; otherwise the next
            # login attempt can incorrectly be reported as already running.
            if self.agy_auth.get(uid) is proc and not url:
                self.agy_auth.pop(uid, None)
            if self.agy_auth.get(uid) is not proc:
                self.agy_read_locks.pop(uid, None)

    async def _finish_agy_auth(self, uid: int, cid: int, thread: Optional[int], code: str) -> None:
        proc = self.agy_auth.pop(uid, None)
        if not proc or proc.returncode is not None or not proc.stdin:
            await self.reply(cid, thread, "❌ Active Antigravity login မရှိပါ။ Settings မှာ Antigravity Google ကို ပြန်နှိပ်ပါ။")
            return
        code = code.strip()
        if not code or len(code) > 2000 or any(ch.isspace() for ch in code):
            self.agy_auth[uid] = proc
            await self.reply(cid, thread, "❌ Authorization code format မမှန်ပါ။ Google ပြတဲ့ code ကို တစ်ကြောင်းတည်း ပြန်ပို့ပါ။")
            return
        try:
            # The PTY-backed agy TUI accepts carriage return, not a pipe-style
            # newline.  It also stays open after exchanging the code, so
            # communicate() would wait forever; read its result text instead.
            auth_started = time.time()
            proc.stdin.write((code + "\r").encode())
            await proc.stdin.drain()
            output = ""
            success = False
            failure = False
            log_detail = ""
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and proc.stdout:
                try:
                    lock = self.agy_read_locks.get(uid)
                    if lock:
                        async with lock:
                            line = await asyncio.wait_for(proc.stdout.readline(), min(1.0, max(0.2, deadline - time.monotonic())))
                    else:
                        line = await asyncio.wait_for(proc.stdout.readline(), min(1.0, max(0.2, deadline - time.monotonic())))
                except asyncio.TimeoutError:
                    line = b""
                if line:
                    output += re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line.decode("utf-8", "replace"))
                low = output.lower()
                failure = any(marker in low for marker in (
                    "token exchange failed", "invalid_grant", "malformed auth", "authentication failed",
                    "login failed", "sign-in failed",
                ))
                success = any(marker in low for marker in (
                    "authentication successful", "authenticated successfully", "signed in as",
                    "login successful", "successfully signed in",
                ))
                log_status, log_detail = agy_auth_log_status(auth_started)
                if log_status == "failure":
                    failure = True
                elif log_status == "success":
                    success = True
                if success or failure:
                    break
            if failure or (proc.returncode not in (None, 0) and output):
                detail = (log_detail or output[-700:]).strip() or "The authorization code was rejected."
                await self.reply(cid, thread, f"❌ Antigravity CLI login မအောင်မြင်ပါ။<br/><code>{esc(detail)}</code>")
                return
            if not success:
                await self.reply(cid, thread, "❌ Antigravity login response မရပါ။ Code အသစ်နဲ့ ပြန်စမ်းပါ။")
                return
            save_provider_override(uid, "antigravity", {"command": Cfg.AGY_COMMAND, "model": MODEL_CATALOG["antigravity"][0][0], "auth": "google"})
            await self.reply(cid, thread, "✅ Antigravity CLI Google login အောင်မြင်ပါပြီ။ Antigravity provider ကို သင့် account အတွက် ချိတ်ပြီးပါပြီ။")
            await self.cmd_settings(uid, cid, thread)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            await self.reply(cid, thread, "❌ Antigravity login timeout ဖြစ်သွားပါတယ်။ Settings မှာ ပြန်စပါ။")
        finally:
            with contextlib.suppress(Exception):
                if proc.returncode is None:
                    proc.kill()

    async def cmd_opencode_login(self, uid: int, cid: int, thread: Optional[int]) -> None:
        """Start OpenCode's official ChatGPT Pro/Plus device authorization flow."""
        if not Path(Cfg.OPENCODE_COMMAND).exists() and not shutil.which(Cfg.OPENCODE_COMMAND):
            await self.reply(cid, thread, "❌ OpenCode CLI မတွေ့ပါ။ Server မှာ OpenCode install လုပ်ရန်လိုပါတယ်။")
            return
        old = self.opencode_auth.get(uid)
        if old and old.returncode is None:
            await self.reply(cid, thread, "⏳ OpenCode login တစ်ခု လုပ်နေပြီးသားပါ။ အပေါ်က OpenAI URL နဲ့ code ကို သုံးပါ။")
            return
        data_dir = Cfg.DATA_DIR / "opencode-users" / str(uid)
        data_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = str(data_dir)
        proc = await asyncio.create_subprocess_exec(
            Cfg.OPENCODE_COMMAND, "auth", "login", "--provider", "openai", "--method", "ChatGPT Pro/Plus (headless)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        self.opencode_auth[uid] = proc
        self.spawn(self._watch_opencode_auth(uid, cid, thread, proc, data_dir))
        await self.reply(cid, thread, "🔐 <b>OpenCode ChatGPT Plus/Pro login</b><br/>OpenAI ကပြတဲ့ device URL ကိုဖွင့်ပြီး အောက်က code ကို ထည့်ပါ။ ပြီးသွားရင် Bot က အလိုအလျောက် အတည်ပြုပါမယ်။")

    async def _watch_opencode_auth(self, uid: int, cid: int, thread: Optional[int], proc: asyncio.subprocess.Process, data_dir: Path) -> None:
        output = ""
        url = ""
        code = ""
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and proc.stdout:
                line = await asyncio.wait_for(proc.stdout.readline(), max(0.2, deadline - time.monotonic()))
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                output += text + "\n"
                if "https://auth.openai.com/codex/device" in text:
                    url = "https://auth.openai.com/codex/device"
                match = re.search(r"Enter code:\s*([A-Z0-9-]+)", text, re.I)
                if match:
                    code = match.group(1)
                if url and code:
                    break
            if not url or not code:
                await self.reply(cid, thread, "❌ OpenCode device login details မထုတ်နိုင်ပါ။ နောက်တစ်ကြိမ် ပြန်စမ်းပါ။")
                with contextlib.suppress(Exception):
                    proc.kill()
                return
            await self.reply(cid, thread,
                             f"🌐 <b>OpenAI device login</b><br/>1. <a href=\"{url}\">ဒီ URL ကိုဖွင့်ပါ</a><br/>2. Device code ကို ထည့်ပါ: <code>{esc(code)}</code><br/><i>အောင်မြင်တဲ့အထိ ဒီ message ကို မဖျက်ပါနဲ့။</i>")
            out, err = await asyncio.wait_for(proc.communicate(), 180)
            if proc.returncode != 0:
                detail = (err or out).decode("utf-8", "replace")[-500:]
                await self.reply(cid, thread, f"❌ OpenCode ChatGPT login မအောင်မြင်ပါ။<br/><code>{esc(detail)}</code>")
                return
            save_provider_override(uid, "opencode", {
                "command": Cfg.OPENCODE_COMMAND,
                "model": MODEL_CATALOG["opencode"][0][0],
                "auth": "openai-chatgpt",
                "data_dir": str(data_dir),
            })
            await self.reply(cid, thread, "✅ OpenCode ကို ChatGPT Plus/Pro နဲ့ ချိတ်ပြီးပါပြီ။ Credential ကို သင့် Telegram account ရဲ့ သီးသန့် directory ထဲမှာပဲ သိမ်းထားပါတယ်။")
            await self.cmd_settings(uid, cid, thread)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            await self.reply(cid, thread, "❌ OpenCode login timeout ဖြစ်သွားပါတယ်။ Settings မှာ ပြန်စပါ။")
        except Exception as e:
            log.warning("OpenCode auth flow failed: %s", e)
            with contextlib.suppress(Exception):
                proc.kill()
            await self.reply(cid, thread, "❌ OpenCode login စတင်မရပါ။ နောက်တစ်ကြိမ် ပြန်စမ်းပါ။")
        finally:
            self.opencode_auth.pop(uid, None)

    async def _oauth_http(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(reader.read(16384), 20)
            first = request.decode("utf-8", "replace").splitlines()[0].split(" ", 2)
            target = first[1] if len(first) > 1 else "/"
            parsed = urlparse(target)
            if parsed.path != "/oauth/google/callback":
                await self._oauth_response(writer, 404, "Not found")
                return
            q = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
            state = q.get("state", "")
            uid, created = self.oauth_states.pop(state, (0, 0.0))
            if not uid or time.time() - created > 600:
                await self._oauth_response(writer, 400, "This login link has expired. Return to Telegram and request a new one.")
                return
            if q.get("error"):
                await self._oauth_response(writer, 400, "Google login was cancelled. You can close this page.")
                await self.reply(uid, uid, "❌ Google login ကို cancel လုပ်လိုက်ပါတယ်။")
                return
            token = await self.http.post("https://oauth2.googleapis.com/token", data={
                "code": q.get("code", ""), "client_id": GOOGLE_OAUTH["client_id"],
                "client_secret": GOOGLE_OAUTH["client_secret"], "redirect_uri": self._oauth_redirect_uri(),
                "grant_type": "authorization_code",
            })
            token.raise_for_status()
            tokens = token.json()
            info = await self.http.get("https://openidconnect.googleapis.com/v1/userinfo",
                                       headers={"Authorization": "Bearer " + tokens["access_token"]})
            info.raise_for_status()
            profile = info.json()
            self._save_google_token(uid, tokens, profile)
            email = esc(profile.get("email", "Google account"))
            await self._oauth_response(writer, 200, "Google login successful. You can return to Telegram.")
            await self.reply(uid, uid, f"✅ Google account ချိတ်ပြီးပါပြီ။<br/>Account: <code>{email}</code>")
        except Exception as e:
            log.warning("Google OAuth callback failed: %s", e)
            with contextlib.suppress(Exception):
                await self._oauth_response(writer, 500, "Google login failed. Return to Telegram and try again.")
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _oauth_response(self, writer: asyncio.StreamWriter, status: int, body: str) -> None:
        payload = f"<html><body><h2>{html.escape(body)}</h2></body></html>".encode()
        writer.write(f"HTTP/1.1 {status} OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
        await writer.drain()

    def _save_google_token(self, uid: int, tokens: dict, profile: dict) -> None:
        data: dict[str, Any] = {}
        try:
            if self.oauth_tokens_file.exists():
                data = json.loads(self.oauth_tokens_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            data = {}
        data[str(uid)] = {"tokens": tokens, "profile": {"sub": profile.get("sub"), "email": profile.get("email"), "name": profile.get("name")}}
        self.oauth_tokens_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.oauth_tokens_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.oauth_tokens_file)

    def spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self.jobs.add(t)

        def done(task: asyncio.Task) -> None:
            self.jobs.discard(task)
            if not task.cancelled() and task.exception():
                log.error("handler crashed: %r", task.exception(), exc_info=task.exception())

        t.add_done_callback(done)

    async def poll(self) -> None:
        offset = 0
        allowed = ["message", "callback_query", "stopped_message_generation"]
        while True:
            try:
                ups = await self.tg.call("getUpdates", _retry=False, offset=offset, timeout=50, allowed_updates=allowed)
            except TGError as e:
                if e.code == 400 and len(allowed) > 2:
                    allowed = ["message", "callback_query"]
                    log.warning("server rejected stopped_message_generation updates; Stop button will use /stop")
                    continue
                if e.code == 409:
                    log.error("Another instance is polling this bot token (409). Stop it first.")
                    await asyncio.sleep(8)
                else:
                    log.warning("getUpdates failed: %s", e)
                    await asyncio.sleep(3)
                continue
            for u in ups:
                offset = u["update_id"] + 1
                self.spawn(self.handle(u))

    async def handle(self, u: dict) -> None:
        if "message" in u:
            await self.on_message(u["message"])
        elif "callback_query" in u:
            await self.on_callback(u["callback_query"])
        elif "stopped_message_generation" in u:
            await self.on_stop_event(u["stopped_message_generation"])

    # ---- helpers -----------------------------------------------------------------
    def rate_ok(self, uid: int) -> bool:
        q = self.rl.setdefault(uid, deque())
        now = time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= Cfg.RATE_LIMIT_PER_MIN:
            return False
        q.append(now)
        return True

    def reply(self, chat: int, thread: Optional[int], h: str, markup: Optional[JSON] = None):
        return self.tg.send(chat, h, thread=thread, markup=markup)

    def active_project(self, uid: int) -> str:
        u = self.db.user(uid) or {}
        return safe_name(u.get("project") or "default")

    @staticmethod
    def image_attachment(m: dict) -> tuple[Optional[dict], Optional[str]]:
        photos = m.get("photo") or []
        if photos:
            # Telegram lists photo sizes from smaller to larger.
            return max(photos, key=lambda p: (p.get("file_size") or 0, p.get("width", 0) * p.get("height", 0))), None
        doc = m.get("document") or {}
        mime = str(doc.get("mime_type") or "").lower()
        suffix = Path(doc.get("file_name") or "").suffix.lower()
        if mime.startswith("image/") or suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            return doc, doc.get("file_name")
        return None, None

    async def receive_image(self, uid: int, media: dict, filename: Optional[str], mid: int) -> tuple[dict, str]:
        if int(media.get("file_size") or 0) > Cfg.MAX_IMAGE_BYTES:
            raise ToolError(f"Image is too large. The current limit is {Cfg.MAX_IMAGE_BYTES // 1_000_000} MB.")
        raw = await self.tg.download_bytes(media["file_id"], Cfg.MAX_IMAGE_BYTES)
        mime = detect_image_mime(raw)
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]
        stem = Path(filename or f"photo_{mid}").stem
        name = f"{safe_name(stem, f'image_{mid}', 60)}_{mid}{ext}"
        ws = get_workspace(uid, self.active_project(uid))
        dest = ws.root / "uploads" / name

        def save() -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)

        await asyncio.to_thread(save)
        image = {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}
        return image, ws.rel(dest)

    # ---- incoming messages ---------------------------------------------------------
    async def on_message(self, m: dict) -> None:
        chat, frm = m["chat"], m.get("from") or {}
        if chat.get("type") != "private" or frm.get("is_bot"):
            return
        uid, cid = frm["id"], chat["id"]
        thread = m.get("message_thread_id") if m.get("is_topic_message") else None
        if Cfg.PUBLIC_ACCESS and not self.db.user(uid):
            self.db.add_user(uid, frm.get("first_name", "") or frm.get("username", ""))
        if not self.db.is_allowed(uid):
            await self.reply(cid, thread, f"<h4>🔒 Private agent</h4><p>This bot is restricted. Your Telegram ID is <code>{uid}</code> — send it to the owner so they can run <code>/allow {uid}</code>.</p>")
            return
        self.db.touch_name(uid, frm.get("first_name", "") or frm.get("username", ""))
        if not self.rate_ok(uid):
            await self.reply(cid, thread, "⏳ Slow down a little — too many messages per minute.")
            return
        text = (m.get("text") or m.get("caption") or "").strip()
        if uid not in self.in_session and (m.get("photo") or m.get("document")):
            await self.need_session(uid, cid, thread)
            return
        images: list[dict] = []
        image_media, image_name = self.image_attachment(m)
        if image_media:
            try:
                image, saved_path = await self.receive_image(uid, image_media, image_name, int(m.get("message_id") or time.time()))
            except (ToolError, TGError, httpx.HTTPError) as e:
                log.warning("could not receive Telegram image: %s", e)
                await self.reply(cid, thread, f"❌ {esc(e) if isinstance(e, ToolError) else 'Could not download the image. Please try again.'}")
                return
            images.append(image)
            if not text:
                text = "Please inspect the attached image, describe what is visible, read any visible text, and answer the user's question if one is implied."
            text += f"\n\n(The image is also saved in the project at {saved_path}.)"
        elif m.get("document"):
            note = await self.save_upload(uid, m["document"])
            if note and not text:
                run = self.runs.get((cid, thread))
                if run:
                    run.inbox.append(f"The user uploaded a file: {note}")
                await self.reply(cid, thread, f"📥 Saved <code>{esc(note)}</code> in your project. Tell me what to do with it.")
                return
            if note:
                text += f"\n\n(The user uploaded a file: {note})"
        if not text:
            return
        if text.startswith("/"):
            await self.command(uid, cid, thread, text, m)
        else:
            await self.on_text(uid, cid, thread, text, m.get("message_id"), images=images)

    async def save_upload(self, uid: int, doc: dict) -> str:
        if (doc.get("file_size") or 0) > 20_000_000:
            return ""
        ws = get_workspace(uid, self.active_project(uid))
        name = safe_name(doc.get("file_name") or "upload.bin", "upload.bin", 80)
        dest = ws.root / "uploads" / name
        await self.tg.download(doc["file_id"], dest)
        return ws.rel(dest)

    async def check_provider_values(self, name: str, values: dict) -> str:
        """Return an error text when the credentials are definitely wrong; '' when OK or unverifiable."""
        try:
            if name in ("antigravity", "opencode"):
                if not shutil.which(values.get("command", "")):
                    return f"`{values.get('command')}` command ကို server မှာ မတွေ့ပါ (install လုပ်ပါ)။"
                return ""
            key = values.get("key", "")
            if name == "openrouter":
                r = await self.http.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=15.0)
            elif name == "anthropic":
                r = await self.http.get("https://api.anthropic.com/v1/models?limit=1", headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=15.0)
            elif name == "gemini":
                r = await self.http.get("https://generativelanguage.googleapis.com/v1beta/openai/models", headers={"Authorization": f"Bearer {key}"}, timeout=15.0)
            else:
                return ""
            if r.status_code in (401, 403):
                return f"API key မမှန်ပါ (HTTP {r.status_code})။ Key အပြည့်အစုံ ကူးထည့်ထားလား စစ်ပါ။"
        except Exception as e:  # network trouble: do not block saving
            log.info("key check skipped for %s: %s", name, e)
        return ""

    async def finish_provider_setup(self, uid: int, cid: int, thread: Optional[int], text: str, mid: Optional[int] = None) -> None:
        if cid != uid:
            if mid:
                await self.tg.delete(cid, mid)
            await self.reply(cid, thread, "API key ကို ဘော့နဲ့ တစ်ဦးချင်းစကားပြောခန်းမှာပဲ ပို့ပါ။ အဖွဲ့ချတ်ထဲ မပို့ပါနဲ့။")
            return
        name = self.provider_setup.pop(uid, "")
        if mid:
            await self.tg.delete(cid, mid)
        parts = [p.strip() for p in text.split("|")]
        if name == "custom":
            if len(parts) != 3 or not parts[0].startswith(("http://", "https://")) or not parts[1] or not parts[2]:
                self.provider_setup[uid] = name
                await self.reply(cid, thread, "❌ Custom format မမှန်ပါ။<br/><code>BASE_URL | API_KEY | MODEL</code> အတိုင်း ပြန်ပို့ပါ။")
                return
            values = {"url": parts[0].rstrip("/"), "key": parts[1], "model": parts[2]}
        elif name in ("antigravity", "opencode"):
            if len(parts) > 2:
                self.provider_setup[uid] = name
                await self.reply(cid, thread, "❌ Format မမှန်ပါ။ <code>COMMAND | MODEL</code> ပုံစံသုံးပါ။")
                return
            command = parts[0] if parts and parts[0] else ("agy" if name == "antigravity" else "opencode")
            model = parts[1] if len(parts) == 2 and parts[1] else MODEL_CATALOG[name][0][0]
            if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", command):
                self.provider_setup[uid] = name
                await self.reply(cid, thread, "❌ CLI command မမှန်ပါ။")
                return
            values = {"command": command, "model": model, "auth": "google" if name == "antigravity" else "opencode"}
        else:
            if len(parts) not in (1, 2) or not parts[0]:
                self.provider_setup[uid] = name
                await self.reply(cid, thread, "❌ Format မမှန်ပါ။<br/><code>API_KEY | MODEL</code> (သို့) API_KEY တစ်ခုတည်း ပို့ပါ။")
                return
            values = {"key": parts[0]}
            if len(parts) == 2 and parts[1]:
                values["model"] = parts[1]
        bad = await self.check_provider_values(name, values)
        if bad:
            self.provider_setup[uid] = name
            await self.reply(cid, thread, f"❌ <b>{esc(name)}</b> ကို မချိတ်ရသေးပါ — {esc(bad)}<br/>မှန်တဲ့ key ကို ပြန်ပို့ပါ (သို့) /settings ထဲက ပြန်ရွေးပါ။")
            return
        save_provider_override(uid, name, values)
        await self.reply(cid, thread, f"✅ <b>{esc(name)}</b> provider ကို သင့် account အတွက်ပဲ ချိတ်ပြီးပါပြီ။")
        await self.cmd_settings(uid, cid, thread)

    async def on_text(
        self, uid: int, cid: int, thread: Optional[int], text: str, mid: Optional[int], images: Optional[list[dict]] = None
    ) -> None:
        if uid in self.agy_auth:
            await self._finish_agy_auth(uid, cid, thread, text)
            return
        if uid in self.provider_setup:
            if cid != uid:
                if mid:
                    await self.tg.delete(cid, mid)
                await self.reply(cid, thread, "API key ကို ဘော့နဲ့ တစ်ဦးချင်းစကားပြောခန်းမှာပဲ ပို့ပါ။ အဖွဲ့ချတ်ထဲ ပို့ထားတဲ့သော့ကို ပိတ်ပြီး အသစ်ထုတ်ပါ။")
                return
            await self.finish_provider_setup(uid, cid, thread, text, mid)
            return
        if text.strip().strip() == EXIT_TEXT:
            await self.exit_session(uid, cid, thread)
            return
        if uid in self.session_setup and uid not in self.in_session:
            await self.create_session(uid, cid, thread, text)
            return
        if uid not in self.in_session:
            await self.need_session(uid, cid, thread)
            return
        run = self.runs.get((cid, thread))
        if run:
            if run.ask_future and not run.ask_future.done():
                if images:
                    run.inbox.append({"text": text, "images": images})
                run.ask_future.set_result(text)
                return
            run.inbox.append({"text": text, "images": images or []} if images else text)
            if mid:
                await self.tg.react(cid, mid, "👀")
            return
        # Task Mode is the only mode inside a session: execute every ordinary message.
        self.spawn(self.start_run(uid, cid, thread, text, images=images or []))

    # Active-session messages always run as Task Mode jobs; no automatic chat/task classifier.

    # ---- running tasks ---------------------------------------------------------------
    async def start_run(
        self, uid: int, cid: int, thread: Optional[int], goal: str, images: Optional[list[dict]] = None
    ) -> None:
        u = self.db.user(uid) or {}
        ws = get_workspace(uid, u.get("project") or "default")
        approval = u.get("approval") or Cfg.DEFAULT_APPROVAL
        if approval not in ("ask", "smart", "auto") or (approval == "auto" and not self.db.is_admin(uid)):
            approval = "smart"
        pname = u.get("provider") if u.get("provider") and provider_config(u["provider"], uid=uid) else default_provider(uid)
        model = u.get("model") if pname and pname == u.get("provider") else None
        pc = provider_config(pname, model, uid=uid) if pname else None
        tid = self.db.new_task(uid, ws.project, goal)
        run = Run(
            tid, uid, cid, thread, ws, goal, approval, self.db.is_admin(uid), LLM(self.http, pc) if pc else None,
            images=images or [],
        )
        run.progress = Progress(self.tg, run)
        self.runs[run.key] = run
        run.task = asyncio.create_task(self._run_wrapper(run))

    async def _run_wrapper(self, run: Run) -> None:
        status, text = "done", ""
        try:
            if self.sem.locked():
                await self.reply(run.chat_id, run.thread, "⏳ Queued — other tasks are running. I'll start as soon as a slot frees up.")
            async with self.sem:
                run.progress.start()
                status, text = await Agent(self, run).execute()
        except asyncio.CancelledError:
            status, text = "stopped", "Stopped. Anything already changed is kept — use **Undo** to revert it."
        except AgentFailure as e:
            status, text = "failed", str(e)
        except Exception as e:
            log.exception("task %s crashed", run.id)
            status, text = "failed", f"Internal error: `{type(e).__name__}: {e}`"
        run.final_status = status
        try:
            await self._finalize(run, status, text)
        except Exception:
            log.exception("finalize failed")
        finally:
            self.runs.pop(run.key, None)
            for p in [k for k, v in self.pending.items() if v.run is run]:
                f = self.pending.pop(p).fut
                if not f.done():
                    f.cancel()

    async def stop_run(self, run: Run) -> None:
        if run.stopped:
            return
        run.stopped = True
        if run.progress:
            run.progress.closed = True  # first: no more draft updates, whatever the tools still return
        if run.ask_future and not run.ask_future.done():
            run.ask_future.cancel()
        if run.task:
            run.task.cancel()

    async def on_stop_event(self, ev: dict) -> None:
        chat, did = (ev.get("chat") or {}).get("id"), ev.get("draft_id")
        for run in list(self.runs.values()):
            if run.chat_id == chat and run.draft_id == did:  # draft_id match ignores stale presses
                await self.stop_run(run)
                return

    async def _read_host_output(self, bp: BgProc) -> None:
        assert bp.proc.stdout
        while True:
            data = await bp.proc.stdout.read(4096)
            if not data:
                break
            bp.buf.extend(data.decode("utf-8", "replace").splitlines())

    async def start_project_host(self, uid: int, cid: int, thread: Optional[int], kind: str, tid: int) -> None:
        task = self.db.task(tid)
        if not task or task["uid"] != uid:
            await self.reply(cid, thread, "❌ ဒီ task ကို host လုပ်ခွင့်မရှိပါ။")
            return
        ws = get_workspace(uid, task["project"])
        if project_kind(ws.root) != kind:
            await self.reply(cid, thread, "❌ Project type မကိုက်ညီပါ။")
            return
        key = (uid, ws.project)
        old_id = self.hosts.get(key)
        if old_id:
            old = self.bg.get(uid, {}).get(old_id)
            if old and old.alive:
                await self.reply(cid, thread, "✅ ဒီ project က host ဖြစ်နေပြီးသားပါ။")
                if kind == "web":
                    port = host_command(ws.root, kind)[1]
                    url = Cfg.PREVIEW_URL_TEMPLATE.format(port=port)
                    await self.reply(cid, thread, f"🌐 <b>Preview live</b><br><code>{esc(url)}</code>", kb([btn("Open Preview", url=url)]))
                return
        cmd, port = host_command(ws.root, kind)
        pid = max(self.bg.get(uid, {}), default=0) + 1
        proc = await spawn(cmd, ws)
        bp = BgProc(pid, cmd, proc, time.time())
        bp.reader = asyncio.create_task(self._read_host_output(bp))
        self.bg.setdefault(uid, {})[pid] = bp
        self.hosts[key] = pid
        await asyncio.sleep(2.5)
        if not bp.alive:
            out = "\n".join(bp.buf)[-1800:] or "(no output)"
            await self.reply(cid, thread, f"❌ Host မစနိုင်ပါ။<pre>{esc(out)}</pre>")
            return
        if kind == "web":
            url = Cfg.PREVIEW_URL_TEMPLATE.format(port=port)
            await self.reply(cid, thread, f"🌐 <b>Preview hosted</b><br><code>{esc(url)}</code>",
                             kb([btn("Open Preview", url=url)], [btn("Stop Preview", f"hoststop:{uid}:{pid}", style="danger")]))
        else:
            await self.reply(cid, thread, "🧪 <b>Sandbox hosted for 1 minute</b><br>Test process က 60 seconds ပြည့်ရင် အလိုအလျောက်ရပ်ပါမယ်။",
                             kb([btn("Stop Sandbox", f"hoststop:{uid}:{pid}", style="danger")]))
            asyncio.create_task(self._stop_host_later(uid, pid, 60))

    async def _stop_host_later(self, uid: int, pid: int, seconds: int) -> None:
        await asyncio.sleep(seconds)
        bp = self.bg.get(uid, {}).get(pid)
        if bp and bp.alive:
            await _terminate(bp.proc)

    async def stop_project_host(self, uid: int, pid: int) -> None:
        bp = self.bg.get(uid, {}).get(pid)
        if bp and bp.alive:
            await _terminate(bp.proc)
        for key, value in list(self.hosts.items()):
            if key[0] == uid and value == pid:
                self.hosts.pop(key, None)

    # ---- final report ------------------------------------------------------------------
    async def _finalize(self, run: Run, status: str, text: str) -> None:
        if run.progress:
            await run.progress.close()
        cp = Checkpoints(run.ws)
        stat, changed = "", []
        if run.cp_start:
            run.cp_end = await cp.snapshot(f"task {run.id}: end")
            if run.cp_end and run.cp_end != run.cp_start:
                stat = await cp.diff_stat(run.cp_start, run.cp_end)
                changed = await cp.changed(run.cp_start, run.cp_end)
        self.db.finish_task(run.id, status=status, steps=len(run.steps), cp_start=run.cp_start, cp_end=run.cp_end,
                            tokens_in=run.tin, tokens_out=run.tout)
        body = self.compose_final(run, status, text, stat, changed)
        rows: list[list[JSON]] = []
        kind = project_kind(run.ws.root)
        if status == "done" and kind == "web":
            rows.append([btn("🌐 Preview", f"host:preview:{run.id}", style="primary")])
        elif status == "done" and kind == "bot":
            rows.append([btn("🧪 Sandbox · 1Min Host", f"host:sandbox:{run.id}", style="primary")])
        if status in ("limit", "stopped"):
            rows.append([btn("▶️ Continue", f"cont:{run.id}", style="primary")])
        tail = [btn("📦 ZIP", "zip"), btn("📂 Files", "files")]
        rows.append(([btn("↩️ Undo changes", f"undo:{run.id}", style="danger")] if changed else []) + tail)
        await self.tg.send(run.chat_id, body, thread=run.thread, markup=kb(*rows))

    def compose_final(self, run: Run, status: str, text: str, stat: str, changed: list[str]) -> str:
        icon, title = {"done": ("✅", "Task complete"), "stopped": ("■", "Stopped"), "failed": ("❌", "Task failed"),
                       "limit": ("⏱", "Step limit reached")}[status]
        text = text if len(text) <= 16000 else text[:16000] + "\n\n…(answer truncated)"
        parts = [f"<h3>{icon} {title}</h3>", md_to_html(text)]
        if run.plan:
            done = sum(1 for p in run.plan if p["status"] == "done")
            lis = "".join(f'<li><input type="checkbox"{" checked" if p["status"] == "done" else ""}/>{esc(p["text"])}</li>' for p in run.plan)
            parts.append(f"<details><summary>📋 Plan · {done}/{len(run.plan)}</summary><ul>{lis}</ul></details>")
        if changed:
            sym = {"A": "＋", "M": "~", "D": "－"}
            rows = "<br>".join(f"{sym.get(c[:1], '•')} <code>{esc(c.split(chr(9), 1)[-1])}</code>" for c in changed[:40])
            more = f"<br>… and {len(changed) - 40} more" if len(changed) > 40 else ""
            parts.append(f"<details><summary>📁 Changes · {len(changed)} file(s)</summary><p>{rows}{more}</p>"
                         f"<pre>{esc(stat[-1500:])}</pre></details>")
        steps = run.steps[-40:]
        if steps:
            ic = {"ok": "✅", "err": "❌", "run": "⏹"}
            lines = "<br>".join(f"{ic[s.state]} {esc(short(s.label, 80))}" for s in steps)
            parts.append(f"<details><summary>🧾 Activity · {len(run.steps)} step(s)</summary><p>{lines}</p></details>")
        rows = [("⏱ Time", dur(time.time() - run.started)), ("🔧 Actions", str(run.tool_calls)), ("📁 Session", esc(run.ws.project))]
        if run.tin + run.tout:
            rows.append(("🔢 Tokens", f"{kfmt(run.tin)} in / {kfmt(run.tout)} out"))
        if run.cp_start:
            rows.append(("📌 Checkpoint", f"<code>{run.cp_start[:7]}</code>"))
        parts.append("<hr/>" + card("Run summary", "📊", rows, foot=esc(run.llm.label) if run.llm else ""))
        out = "".join(parts)
        if len(out) > RICH_LIMIT - 500:  # drop the bulky sections first
            out = "".join(p for p in parts if not p.startswith("<details><summary>🧾"))
        return out[: RICH_LIMIT - 200] if len(out) > RICH_LIMIT - 200 else out

    # ---- human in the loop ---------------------------------------------------------------
    async def request_approval(self, run: Run, cmd: str, why: str) -> str:
        tok = secrets.token_hex(4)
        fut = asyncio.get_running_loop().create_future()
        self.pending[tok] = Pending(run, fut, "ap")
        body = (card("Approval needed", "🔐", [("Why", esc(why)), ("Session", esc(run.ws.project))], foot="Nothing runs until you choose")
                + f"<pre><code class=\"language-bash\">{esc(trunc(cmd, 1500))}</code></pre>")
        markup = kb([btn("✅ Allow", f"ap:{tok}:y", style="success"), btn("⚡ Allow all (this task)", f"ap:{tok}:a", style="primary")],
                    [btn("Deny", f"ap:{tok}:n", style="danger")])
        mid = await self.tg.send(run.chat_id, body, thread=run.thread, markup=markup)
        run.progress.waiting = "Waiting for your approval"
        decision, label = "deny", "⏱ No answer — denied"
        try:
            d = await asyncio.wait_for(fut, Cfg.APPROVAL_TIMEOUT)
            decision, label = {"y": ("once", "✅ Allowed"), "a": ("all", "⚡ Allowed for this task"), "n": ("deny", "Denied")}[d]
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            if mid:
                await self.tg.edit(run.chat_id, mid, body + "<p>Cancelled</p>", NOKB)
            raise
        finally:
            self.pending.pop(tok, None)
            run.progress.waiting = ""
        if mid:
            await self.tg.edit(run.chat_id, mid, body + f"<p><b>{label}</b></p>", NOKB)
        return decision

    async def ask_user(self, run: Run, question: str, options: list[str]) -> str:
        tok = secrets.token_hex(4)
        fut = asyncio.get_running_loop().create_future()
        run.ask_future = fut
        self.pending[tok] = Pending(run, fut, "ask", options)
        body = f"<h4>❓ Question</h4><aside>{_inline(question)}<cite>Tap an option or just type your answer</cite></aside>"
        rows = [[btn(o[:40], f"ask:{tok}:{i}")] for i, o in enumerate(options)]
        mid = await self.tg.send(run.chat_id, body, thread=run.thread, markup=kb(*rows) if rows else None)
        run.progress.waiting = "Waiting for your answer"
        answer = ""
        try:
            answer = await asyncio.wait_for(fut, 900)
        except asyncio.TimeoutError:
            pass
        finally:
            run.ask_future = None
            self.pending.pop(tok, None)
            run.progress.waiting = ""
        if mid:
            await self.tg.edit(run.chat_id, mid, body + f"<p><b>Answer:</b> {esc(answer or '(none)')}</p>", NOKB)
        return answer

    # ---- callbacks -------------------------------------------------------------------------
    async def on_callback(self, cq: dict) -> None:
        uid, data = cq["from"]["id"], cq.get("data") or ""
        msg = cq.get("message") or {}
        cid = (msg.get("chat") or {}).get("id") or uid
        thread = msg.get("message_thread_id") if msg.get("is_topic_message") else None
        mid = msg.get("message_id")
        if not self.db.is_allowed(uid):
            await self.tg.answer_cb(cq["id"], "Not allowed", True)
            return
        kind, _, rest = data.partition(":")
        if kind in ("cfg", "cfgpick", "cfgdel") and cid != uid:
            await self.tg.answer_cb(cq["id"], "API key ကို bot နဲ့ တစ်ဦးချင်း စကားပြောခန်းမှာပဲ ချိတ်ဆက်ပါ။", True)
            return
        if uid in self.in_session and kind not in self.SESSION_CB:
            await self.tg.answer_cb(cq["id"], "🔒 Session ထဲမှာ ဒီ menu မသုံးနိုင်ပါ။ Exit နှိပ်ပါ။", True)
            return
        await self.tg.answer_cb(cq["id"])

        if kind == "ap":
            tok, _, d = rest.partition(":")
            p = self.pending.get(tok)
            if p and p.run.uid == uid and not p.fut.done():
                p.fut.set_result(d)
        elif kind == "ask":
            tok, _, i = rest.partition(":")
            p = self.pending.get(tok)
            if p and p.run.uid == uid and not p.fut.done() and i.isdigit() and int(i) < len(p.options):
                p.fut.set_result(p.options[int(i)])
        elif kind == "stop":
            run = next((r for r in self.runs.values() if str(r.id) == rest and r.uid == uid), None)
            if run:
                await self.stop_run(run)
        elif kind == "cont":
            if (cid, thread) in self.runs:
                await self.reply(cid, thread, "A task is already running here.")
            else:
                await self.start_run(uid, cid, thread, "Continue the previous task from where you stopped. Re-check the current state of the files first.")
        elif kind == "undo":
            await self.undo_confirm(uid, cid, thread, int(rest) if rest.isdigit() else 0)
        elif kind == "undoy":
            await self.undo_apply(uid, cid, thread, int(rest) if rest.isdigit() else 0, mid)
        elif kind == "redo":
            ws = get_workspace(uid, self.active_project(uid))
            res = await Checkpoints(ws).restore(rest)
            await self.reply(cid, thread, "↪️ Redo applied." if res == "ok" else f"❌ {esc(res)}")
        elif kind == "undon":
            if mid:
                await self.tg.edit(cid, mid, "<p>Undo cancelled.</p>", NOKB)
        elif kind == "zip":
            await self.send_zip(uid, cid, thread)
        elif kind == "files":
            await self.cmd_files(uid, cid, thread)
        elif kind == "newp":
            await self.start_session_name_prompt(uid, cid, thread)
        elif kind == "sescancel":
            self.session_setup.discard(uid)
            if mid:
                await self.tg.edit(cid, mid, "<p>Cancelled.</p>", NOKB)
        elif kind == "sesexit":
            await self.exit_session(uid, cid, thread)
        elif kind == "sesexitf":
            await self.exit_session(uid, cid, thread, force=True)
        elif kind == "proj":
            await self.enter_session(uid, cid, thread, rest)
        elif kind == "mp":
            bits = rest.split(":")
            if len(bits) == 3 and bits[0].isdigit() and bits[2].isdigit() and mid:
                h, mk = await self.models_view(uid, int(bits[0]), bits[1], int(bits[2]))
                await self.tg.edit(cid, mid, h, mk)
        elif kind == "mr":
            bits = rest.split(":")
            if len(bits) == 2 and bits[0].isdigit() and mid:
                h, mk = await self.models_view(uid, int(bits[0]), bits[1], 0, force=True)
                await self.tg.edit(cid, mid, h, mk)
        elif kind == "mm":
            bits = rest.split(":")
            if len(bits) == 2 and bits[0].isdigit() and bits[1].isdigit():
                chosen = await self.pick_model(uid, int(bits[0]), int(bits[1]))
                if chosen is None:
                    await self.tg.answer_cb(cq["id"], "List သက်တမ်းကုန်ပြီ၊ ပြန်ဖွင့်ပါ", True)
                elif mid:
                    h, mk = await self.models_view(uid, int(bits[0]), "a", int(bits[1]) // self.MODELS_PER_PAGE)
                    await self.tg.edit(cid, mid, f"<p>✅ Model → <code>{esc(chosen)}</code></p>" + h, mk)
        elif kind == "host":
            action, _, task_id = rest.partition(":")
            if action in ("preview", "sandbox") and task_id.isdigit():
                await self.start_project_host(uid, cid, thread, "web" if action == "preview" else "bot", int(task_id))
        elif kind == "hoststop":
            bits = rest.split(":")
            if len(bits) == 2 and bits[0].isdigit() and int(bits[0]) == uid and bits[1].isdigit():
                await self.stop_project_host(uid, int(bits[1]))
                await self.reply(cid, thread, "Host ရပ်ပြီးပါပြီ။")
        elif kind == "cfg":
            await self.reply(cid, thread, "🔌 သင့် account အတွက်ပဲ ချိတ်မယ့် provider ကိုရွေးပါ။", kb(
                [btn("Anthropic", "cfgpick:anthropic"), btn("OpenRouter", "cfgpick:openrouter")],
                [btn("Gemini", "cfgpick:gemini"), btn("Custom", "cfgpick:custom")],
                [btn("Antigravity Google", "cfgpick:antigravity"), btn("OpenCode CLI", "cfgpick:opencode")],
            ))
        elif kind == "cfgpick":
            if rest in ("anthropic", "openrouter", "gemini", "custom", "antigravity", "opencode"):
                if rest == "antigravity":
                    await self.cmd_google_login(uid, cid, thread)
                    return
                if rest == "opencode":
                    await self.cmd_opencode_login(uid, cid, thread)
                    return
                self.provider_setup[uid] = rest
                if rest == "custom":
                    prompt = "<b>Custom OpenAI-compatible provider</b><br/><code>BASE_URL | API_KEY | MODEL</code> ပို့ပါ။"
                elif rest == "antigravity":
                    prompt = "<b>Antigravity CLI</b><br/>Google login လုပ်ပြီးသား <code>agy</code> CLI ရဲ့ command နဲ့ model ပို့ပါ။<br/><code>agy | gemini-3.8-flash</code><br/><i>Remote host မှာ browser login မရရင် agy ရဲ့ SSH URL/code flow သုံးပါ။</i>"
                else:
                    prompt = f"<b>{esc(rest)}</b> API key ပို့ပါ။ Model ထည့်လိုပါက <code>API_KEY | MODEL</code> ပုံစံသုံးပါ။"
                await self.reply(cid, thread, prompt + "<br/><i>API key ကို ဘော့နဲ့ တစ်ဦးချင်းစကားပြောခန်းမှာပဲ ပို့ပါ။ သိမ်းပြီးရင် ပို့ထားတဲ့စာကို ဖျက်ဖို့ ကြိုးစားပါမယ်။ သော့ကို Telegram ID တစ်ခုချင်းစီအလိုက် ခွဲသိမ်းထားပါတယ်။</i>")
        elif kind == "cfgdel":
            if rest in ("anthropic", "openrouter", "gemini", "custom", "antigravity", "opencode"):
                delete_provider_override(uid, rest)
                await self.reply(cid, thread, f"🗑️ <b>{esc(rest)}</b> provider setting ဖျက်ပြီးပါပြီ။")
                await self.cmd_settings(uid, cid, thread)
        elif kind == "models":
            if mid:
                h, mk = self.providers_view(uid)
                await self.tg.edit(cid, mid, h, mk)
            else:
                await self.cmd_models(uid, cid, thread)
        elif kind == "set":
            what, _, val = rest.partition(":")
            if what == "mode" and val in ("ask", "smart", "auto"):
                self.db.set(uid, "approval", "smart" if val == "auto" and not self.db.is_admin(uid) else val)
            elif what == "prov" and provider_config(val, uid=uid):
                self.db.set(uid, "provider", val)
                self.db.set(uid, "model", None)
            h, mk = self.settings_view(uid)
            if mid:
                await self.tg.edit(cid, mid, h, mk)
        elif kind == "skdel":
            Skills(uid).delete(rest)
            await self.cmd_skills(uid, cid, thread)
        elif kind == "menu":
            fn = {"home": self.cmd_start, "projects": self.cmd_projects, "settings": self.cmd_settings, "skills": self.cmd_skills,
                  "help": self.cmd_help, "status": self.cmd_status,
                  "preview": self.cmd_preview, "sandbox": self.cmd_sandbox}.get(rest)
            if fn:
                await fn(uid, cid, thread)

    # ---- undo ------------------------------------------------------------------------------------
    async def undo_confirm(self, uid: int, cid: int, thread: Optional[int], tid: int) -> None:
        t = self.db.task(tid)
        if not t or t["uid"] != uid or not t["cp_start"]:
            await self.reply(cid, thread, "❌ No checkpoint for that task.")
            return
        if any(r.uid == uid and r.ws.project == t["project"] for r in self.runs.values()):
            await self.reply(cid, thread, "⚠️ A task is running in this project — stop it before undoing.")
            return
        ws = get_workspace(uid, t["project"])
        cp = Checkpoints(ws)
        now = await cp.snapshot("before undo")
        stat = await cp.diff_stat(t["cp_start"], now) if now else ""
        body = (card(f"Undo task #{tid}?", "↩️", [("Task", esc(short(t['goal'], 120)))],
                     note="The whole project returns to the state <b>before</b> this task (later tasks too).", foot="You can redo right after")
                + f"<pre>{esc(stat[-1500:] or 'no differences')}</pre>")
        await self.reply(cid, thread, body, kb([btn("↩️ Revert now", f"undoy:{tid}", style="danger"), btn("Cancel", "undon")]))

    async def undo_apply(self, uid: int, cid: int, thread: Optional[int], tid: int, mid: Optional[int]) -> None:
        t = self.db.task(tid)
        if not t or t["uid"] != uid or not t["cp_start"]:
            return
        ws = get_workspace(uid, t["project"])
        cp = Checkpoints(ws)
        before = await cp.snapshot("before undo")
        res = await cp.restore(t["cp_start"])
        if mid:
            await self.tg.edit(cid, mid, f"<p>{'✅ Reverted to before task #' + str(tid) if res == 'ok' else '❌ ' + esc(res)}</p>", NOKB)
        if res == "ok" and before:
            await self.reply(cid, thread, "Changed your mind?", kb([btn("↪️ Redo", f"redo:{before[:12]}", style="primary")]))

    # ---- commands ----------------------------------------------------------------------------------
    async def command(self, uid: int, cid: int, thread: Optional[int], text: str, m: dict) -> None:
        head, _, arg = text.partition(" ")
        cmd, arg = head[1:].split("@")[0].lower(), arg.strip()
        admin = self.db.is_admin(uid)
        if cmd == "exit":
            await self.exit_session(uid, cid, thread)
            return
        if uid in self.in_session and cmd not in self.SESSION_CMDS and cmd != "start":
            await self.reply(cid, thread, "🔒 Session ထဲမှာ ဒီ command မသုံးနိုင်ပါ။ ထွက်ရန် /exit",
                             kb([btn("Exit Session", "sesexit", style="danger")]))
            return
        if uid not in self.in_session and cmd in ("files", "get", "zip", "download", "undo", "clear", "preview", "sandbox", "stop") \
                and not (cmd == "stop" and self.runs.get((cid, thread))):
            await self.need_session(uid, cid, thread)
            return
        simple = {"start": self.cmd_start, "sessions": self.cmd_projects, "session": self.cmd_projects, "help": self.cmd_help, "projects": self.cmd_projects, "files": self.cmd_files,
                  "status": self.cmd_status, "settings": self.cmd_settings, "skills": self.cmd_skills, "undo": self.cmd_undo,
                  "preview": self.cmd_preview, "sandbox": self.cmd_sandbox, "models": self.cmd_models,
                  "googlelogin": self.cmd_google_login,
                  "zip": self.send_zip, "download": self.send_zip}
        if cmd in simple:
            await simple[cmd](uid, cid, thread)
        elif cmd == "new":
            await self.cmd_new(uid, cid, thread, arg)
        elif cmd == "use":
            await self.enter_session(uid, cid, thread, arg)
        elif cmd == "stop":
            run = self.runs.get((cid, thread))
            if run:
                await self.stop_run(run)
            else:
                await self.reply(cid, thread, "Nothing is running.")
        elif cmd == "clear":
            self.db.clear_history(uid, self.active_project(uid))
            await self.reply(cid, thread, "🧹 Conversation context cleared (project files untouched).")
        elif cmd == "model":
            await self.cmd_model(uid, cid, thread, arg)
        elif cmd == "get":
            ws = get_workspace(uid, self.active_project(uid))
            try:
                p = ws.resolve(arg)
                if not p.is_file():
                    raise ToolError("not a file")
                await self.tg.send_file(cid, p, thread=thread)
            except ToolError as e:
                await self.reply(cid, thread, f"❌ {esc(e)}")
        elif admin and cmd in ("allow", "ban", "unban", "users", "stats", "broadcast"):
            await self.cmd_admin(cid, thread, cmd, arg)
        else:
            await self.reply(cid, thread, "Unknown command. Just describe what you want built — or see /help.")


    # ======================= SESSION MODE ==========================================
    SESSION_CMDS = {"exit", "stop", "files", "get", "zip", "download", "undo", "clear", "preview", "sandbox"}
    SESSION_CB = {"ap", "ask", "stop", "cont", "undo", "undoy", "undon", "redo", "zip", "files", "host", "hoststop", "sesexit", "sesexitf"}
    SESSION_COMMAND_MENU = [("exit", "Exit session"), ("stop", "Stop the running task"), ("files", "Session files"),
                            ("undo", "Revert a task"), ("clear", "Forget chat context")]

    def _save_session_state(self) -> None:
        with contextlib.suppress(OSError):
            self.session_file.write_text(json.dumps(sorted(self.in_session)), encoding="utf-8")

    async def _set_chat_menu(self, cid: int, inside: bool) -> None:
        """Inside a session the / menu shrinks to session-only commands; outside it is the full menu."""
        with contextlib.suppress(TGError):
            if inside:
                await self.tg.call("setMyCommands", commands=[{"command": c, "description": d} for c, d in self.SESSION_COMMAND_MENU],
                                   scope={"type": "chat", "chat_id": cid})
            else:
                await self.tg.call("deleteMyCommands", scope={"type": "chat", "chat_id": cid})

    def main_menu(self, uid: int) -> tuple[str, JSON]:
        cur = self.active_project(uid)
        pname, pmodel = self._current_model(uid)
        approval = (self.db.user(uid) or {}).get("approval") or Cfg.DEFAULT_APPROVAL
        engine = f"<code>{esc(pname)} / {esc(short(str(pmodel), 30))}</code>" if pname else "<i>မချိတ်ရသေးပါ</i>"
        body = ("<h2>🤖 Agent X</h2>"
                "<p><b>Professional AI Coding Agent</b> — Telegram ထဲကနေ software ကို ရေး၊ run၊ test၊ ship လုပ်ပေးပါတယ်။</p>"
                + card("Workspace", "📡", [("Engine", engine), ("Approval", esc(approval)),
                                          ("Sessions", str(len(list_projects(uid)))), ("Last session", esc(cur))],
                       foot="Session ထဲဝင်ပြီး လိုချင်တာကို ရေးပို့ပါ")
                + card("Capabilities", "🧩", [("Build", "web · APK · bots · scripts · APIs"),
                                              ("Run", "shell · tests · live preview"),
                                              ("Safe", "checkpoints · undo · approvals")])
                + "<aside>ဥပမာ: “Flask todo app ရေးပြီး test run ပေးပါ”</aside>")
        if not pname:
            body += "<p>⚠️ အရင်ဆုံး <b>Settings → Connect my provider</b> နဲ့ model ချိတ်ပါ။</p>"
        return body, kb([btn("🆕 New Session", "newp", style="primary"), btn("📂 Sessions", "menu:projects")],
                        [btn("📔 Models", "models"), btn("🗝 Settings", "menu:settings")],
                        [btn("🧠 Skills", "menu:skills"), btn("📖 Help", "menu:help")])

    async def show_main_menu(self, uid: int, cid: int, thread: Optional[int]) -> None:
        h, mk = self.main_menu(uid)
        await self.reply(cid, thread, h, mk)

    async def need_session(self, uid: int, cid: int, thread: Optional[int]) -> None:
        await self.reply(cid, thread, "🔒 အလုပ်စလုပ်ဖို့ Session တစ်ခု ဖန်တီးပါ (သို့) ရှိပြီးသား Session ထဲဝင်ပါ။",
                         kb([btn("🆕 Create Session", "newp", style="primary"), btn("📂 Sessions", "menu:projects")]))

    async def start_session_name_prompt(self, uid: int, cid: int, thread: Optional[int]) -> None:
        self.session_setup.add(uid)
        await self.reply(cid, thread, "🆕 <b>Create Session</b><br/>Session နာမည်ကို စာသားနဲ့ ပို့ပါ (ဥပမာ <code>my-web</code>)။ နာမည်မပေးချင်ရင် <code>-</code> လို့ပို့ပါ။",
                         kb([btn("❌ Cancel", "sescancel")]))

    async def create_session(self, uid: int, cid: int, thread: Optional[int], raw: str) -> None:
        raw = raw.strip()
        auto = f"session-{int(time.time()) % 100000}"
        name = auto if raw in ("", "-") else safe_name(raw, auto)
        await self.enter_session(uid, cid, thread, name, create=True)

    async def enter_session(self, uid: int, cid: int, thread: Optional[int], name: str, create: bool = False) -> None:
        name = safe_name(name)
        if not create and name not in list_projects(uid):
            await self.reply(cid, thread, f"❌ <code>{esc(name)}</code> ဆိုတဲ့ Session မရှိပါ။")
            return
        if (cid, thread) in self.runs:
            await self.reply(cid, thread, "⚠️ Task တစ်ခု အလုပ်လုပ်နေပါတယ်။ အရင် /stop လုပ်ပါ။")
            return
        get_workspace(uid, name)
        self.session_setup.discard(uid)
        self.db.set(uid, "project", name)
        self.in_session.add(uid)
        self._save_session_state()
        await self._set_chat_menu(cid, True)
        await self.reply(cid, thread,
                         card("Session started" if create else "Session entered", "✅",
                              [("Session", f"<b>{esc(name)}</b>"),
                               ("Engine", f"<code>{esc(str(self._current_model(uid)[0]))} / {esc(short(str(self._current_model(uid)[1]), 28))}</code>"),
                               ("Approval", esc((self.db.user(uid) or {}).get("approval") or Cfg.DEFAULT_APPROVAL))],
                              foot="စာရေးပို့ရုံနဲ့ Task စပါမယ် · ထွက်ရန် Exit")
                         + "<p>ဒီ Session ထဲမှာပဲ အလုပ်လုပ်ပါမယ်။ <b>/stop</b> ရပ်၊ <b>/undo</b> ပြန်ဖျက်၊ <b>/files</b> ဖိုင်ကြည့်ပါ။</p>",
                         kb([btn("Exit Session", "sesexit", style="danger")]))
        for key in ({"text": EXIT_TEXT},):
            try:
                await self.tg.call("sendMessage", chat_id=cid, message_thread_id=thread, text="Session mode — အောက်က Exit ခလုတ်နဲ့ ထွက်နိုင်ပါတယ်။",
                                   reply_markup={"keyboard": [[key]], "resize_keyboard": True, "is_persistent": True})
                break
            except TGError:
                continue

    async def exit_session(self, uid: int, cid: int, thread: Optional[int], force: bool = False) -> None:
        if uid not in self.in_session:
            with contextlib.suppress(TGError):
                await self.tg.call("sendMessage", chat_id=cid, message_thread_id=thread, text="Session ထဲမရှိပါ။", reply_markup={"remove_keyboard": True})
            await self._set_chat_menu(cid, False)
            await self.show_main_menu(uid, cid, thread)
            return
        run = self.runs.get((cid, thread))
        if run and not force:
            await self.reply(cid, thread, "⚠️ Task တစ်ခု အလုပ်လုပ်နေဆဲပါ။ ရပ်ပြီးထွက်မလား?",
                             kb([btn("Stop & Exit", "sesexitf", style="danger"), btn("Stay", "sescancel")]))
            return
        if run:
            await self.stop_run(run)
        self.in_session.discard(uid)
        self._save_session_state()
        await self._set_chat_menu(cid, False)
        with contextlib.suppress(TGError):
            await self.tg.call("sendMessage", chat_id=cid, message_thread_id=thread, text="Session ထဲက ထွက်ပြီးပါပြီ။", reply_markup={"remove_keyboard": True})
        await self.show_main_menu(uid, cid, thread)

    # ======================= MODEL PICKER ==========================================
    MODEL_ORDER = ("anthropic", "openrouter", "gemini", "antigravity", "opencode", "custom")
    MODELS_PER_PAGE = 8

    async def fetch_models(self, uid: int, provider: str, force: bool = False) -> tuple[list[tuple[str, bool]], bool]:
        """Full model list of a connected provider -> ([(model_id, is_free)], came_from_live_api)."""
        key = (uid, provider)
        hit = self.model_cache.get(key)
        if hit and not force and time.time() - hit[0] < 600:
            return hit[1], hit[2]
        pc = provider_config(provider, uid=uid)
        static = list(MODEL_CATALOG.get(provider, []))
        known_free = {m: f for m, f in static}
        found: list[tuple[str, bool]] = []
        live = False
        try:
            if pc and provider == "openrouter":
                r = await self.http.get("https://openrouter.ai/api/v1/models", timeout=25.0)
                r.raise_for_status()
                for m in r.json().get("data", []):
                    mid = str(m.get("id") or "")
                    if not mid:
                        continue
                    pr = m.get("pricing") or {}
                    try:
                        free = float(pr.get("prompt", "x")) == 0 and float(pr.get("completion", "x")) == 0
                    except (TypeError, ValueError):
                        free = False
                    found.append((mid, free or mid.endswith(":free")))
            elif pc and provider == "gemini":
                r = await self.http.get("https://generativelanguage.googleapis.com/v1beta/openai/models",
                                        headers={"Authorization": f"Bearer {pc['key']}"}, timeout=25.0)
                r.raise_for_status()
                for m in r.json().get("data", []):
                    mid = str(m.get("id") or "").removeprefix("models/")
                    if mid.startswith("gemini"):
                        found.append((mid, known_free.get(mid, ("flash" in mid or "lite" in mid) and "pro" not in mid)))
            elif pc and provider == "anthropic":
                r = await self.http.get("https://api.anthropic.com/v1/models?limit=100",
                                        headers={"x-api-key": pc["key"], "anthropic-version": "2023-06-01"}, timeout=25.0)
                r.raise_for_status()
                found = [(str(m["id"]), False) for m in r.json().get("data", []) if m.get("id")]
            elif pc and provider == "custom":
                base = pc["url"].rsplit("/chat/completions", 1)[0]
                r = await self.http.get(base + "/models", headers={"Authorization": f"Bearer {pc['key']}"}, timeout=25.0)
                r.raise_for_status()
                found = [(str(m["id"]), ":free" in str(m["id"]) or str(m["id"]).endswith("-free")) for m in r.json().get("data", []) if m.get("id")]
            elif pc and provider == "opencode":
                env = os.environ.copy()
                if pc.get("data_dir"):
                    env["XDG_DATA_HOME"] = pc["data_dir"]
                proc = await asyncio.create_subprocess_exec(pc["command"], "models", stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.DEVNULL, env=env)
                out, _ = await asyncio.wait_for(proc.communicate(), 30)
                for line in out.decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if re.fullmatch(r"[A-Za-z0-9_.:/-]+/[A-Za-z0-9_.:/-]+", line):
                        found.append((line, known_free.get(line, "free" in line.lower())))
            live = bool(found)
        except Exception as e:  # noqa: BLE001 - any failure -> built-in list
            log.warning("model list fetch failed for %s: %s", provider, e)
            found = []
        if not found:
            found = static
        cur = (pc or {}).get("model")
        if cur and all(cur != m for m, _ in found):
            found.append((cur, known_free.get(cur, False)))
        seen: set[str] = set()
        uniq = [(m, f) for m, f in found if not (m in seen or seen.add(m))]
        uniq.sort(key=lambda x: (not x[1], x[0].lower()))   # free first, then A-Z
        self.model_cache[key] = (time.time(), uniq, live)
        return uniq, live

    def _current_model(self, uid: int) -> tuple[Optional[str], Optional[str]]:
        u = self.db.user(uid) or {}
        pname = u.get("provider") if u.get("provider") and provider_config(u["provider"], uid=uid) else default_provider(uid)
        pc = provider_config(pname, u.get("model") if pname == u.get("provider") else None, uid=uid) if pname else None
        return (pname, pc["model"]) if pc else (None, None)

    def providers_view(self, uid: int) -> tuple[str, JSON]:
        cur_p, cur_m = self._current_model(uid)
        conn = [p for p in self.MODEL_ORDER if provider_connected(p, uid)]
        if not conn:
            return ("<h3>📔 Models</h3><p>ချိတ်ထားတဲ့ Provider မရှိသေးပါ။ အရင် Provider ချိတ်ပါ။</p>",
                    kb([btn("🔱 Connect provider", "cfg", style="primary")]))
        rows = [[btn(("🥇 " if p == cur_p else "") + p, f"mp:{self.MODEL_ORDER.index(p)}:a:0", style="primary" if p == cur_p else None)] for p in conn]
        body = card("Models", "📔", [("Current", f"<code>{esc(str(cur_p))}/{esc(str(cur_m))}</code>")],
                    foot="Provider ကိုနှိပ်ပြီး model အားလုံးကြည့်ပါ") 
        return body, kb(*rows)

    async def models_view(self, uid: int, pi: int, flt: str, page: int, force: bool = False) -> tuple[str, JSON]:
        provider = self.MODEL_ORDER[pi] if 0 <= pi < len(self.MODEL_ORDER) else ""
        if not provider or not provider_connected(provider, uid):
            return self.providers_view(uid)
        models, live = await self.fetch_models(uid, provider, force)
        cur_p, cur_m = self._current_model(uid)
        idx = [i for i, (_, f) in enumerate(models) if flt != "f" or f]
        pages = max(1, -(-len(idx) // self.MODELS_PER_PAGE))
        page = max(0, min(page, pages - 1))
        chunk = idx[page * self.MODELS_PER_PAGE:(page + 1) * self.MODELS_PER_PAGE]
        rows = []
        for i in chunk:
            m, free = models[i]
            on = provider == cur_p and m == cur_m
            rows.append([btn(f"{'✅' if on else ('🆓' if free else '💎')} {m[:44]}", f"mm:{pi}:{i}", style="success" if on else None)])
        if pages > 1:
            nav = []
            if page > 0:
                nav.append(btn("Prev", f"mp:{pi}:{flt}:{page - 1}"))
            nav.append(btn(f"{page + 1}/{pages}", f"mp:{pi}:{flt}:{page}"))
            if page < pages - 1:
                nav.append(btn("Next", f"mp:{pi}:{flt}:{page + 1}"))
            rows.append(nav)
        n_free = sum(1 for _, f in models if f)
        rows.append([btn("📋 All" + (" ✔" if flt != "f" else ""), f"mp:{pi}:a:0"), btn(f"🆓 Free only ({n_free})" + (" ✔" if flt == "f" else ""), f"mp:{pi}:f:0")])
        rows.append([btn("⚡ Refresh", f"mr:{pi}:{flt}"), btn("🔌 Providers", "models")])
        body = card(provider, "📔", [("Models", f"{len(models)} total · {n_free} free" + ("" if live else " · built-in list")),
                                     ("Current", f"<code>{esc(str(cur_m))}</code>" if provider == cur_p else "—")],
                    note="<i>ဒီ filter နဲ့ model မရှိပါ။</i>" if not chunk else "", foot="🆓 free · 💎 paid/plan · ✅ active · နှိပ်ရုံနဲ့ပြောင်းမယ်")
        return body, kb(*rows)

    async def pick_model(self, uid: int, pi: int, idx: int) -> Optional[str]:
        provider = self.MODEL_ORDER[pi] if 0 <= pi < len(self.MODEL_ORDER) else ""
        hit = self.model_cache.get((uid, provider))
        if not provider or not hit or not (0 <= idx < len(hit[1])):
            return None
        model = hit[1][idx][0]
        if not provider_config(provider, model, uid=uid):
            return None
        self.db.set(uid, "provider", provider)
        self.db.set(uid, "model", model)
        return f"{provider}/{model}"

    async def cmd_start(self, uid: int, cid: int, thread: Optional[int]) -> None:
        if uid in self.in_session:
            await self.reply(cid, thread, card("Session", "✅", [("Active", f"<b>{esc(self.active_project(uid))}</b>")], foot="ထွက်ရန် /exit"),
                             kb([btn("Exit Session", "sesexit", style="danger")]))
            return
        await self.show_main_menu(uid, cid, thread)

    async def cmd_preview(self, uid: int, cid: int, thread: Optional[int]) -> None:
        if (cid, thread) in self.runs:
            await self.reply(cid, thread, "⚠️ A task is already running here. Stop it before starting Preview.")
            return
        await self.start_run(uid, cid, thread,
                             "This is a web project Preview request. Detect the project's web stack and start its development server in the background. Verify it with an HTTP request and, if possible, a screenshot. Keep the server running for the user's live Preview and report the exact localhost port/URL and start command. Do not modify application code just to start Preview.")

    async def cmd_sandbox(self, uid: int, cid: int, thread: Optional[int]) -> None:
        if (cid, thread) in self.runs:
            await self.reply(cid, thread, "⚠️ A task is already running here. Stop it before starting Sandbox test.")
            return
        await self.start_run(uid, cid, thread,
                             "This is a bot project Sandbox test request. Detect the bot's test/start command, run it in the background, inspect logs and health for at most 60 seconds, then stop the test process cleanly. Report whether it passed, the relevant output, and any fix needed. This is only a 1-minute test host, not a permanent service.")
        asyncio.create_task(self._sandbox_timeout(cid, thread))

    async def _sandbox_timeout(self, cid: int, thread: Optional[int]) -> None:
        await asyncio.sleep(60)
        run = self.runs.get((cid, thread))
        if run:
            await self.stop_run(run)

    async def cmd_help(self, uid: int, cid: int, thread: Optional[int]) -> None:
        body = ("<h2>📖 Agent X Guide</h2>"
                "<p>Session ထဲဝင်ပြီး လိုချင်တာကို ရိုးရိုးစာနဲ့ ရေးပို့ပါ။ Agent က plan → code → run → test → fix လုပ်ပြီး ရလဒ်ပြန်ပို့ပါတယ်။ Image (JPG/PNG/WebP) ပါ ပို့လို့ရပါတယ်။</p>"
                + card("Session", "📂", [("/new name", "Session အသစ်ဖန်တီး"), ("/sessions", "ရှိပြီးသားထဲဝင်"), ("/exit", "Session ထဲကထွက်")])
                + card("Task control", "🎛", [("/stop", "လက်ရှိ Task ရပ်"), ("/undo", "Task တစ်ခုကို ပြန်ဖျက်"), ("/clear", "Chat context reset")])
                + card("Files & Preview", "📦", [("/files", "ဖိုင်စာရင်း + ZIP"), ("/get path", "ဖိုင်တစ်ခုယူ"),
                                                 ("/preview", "Web project live preview"), ("/sandbox", "၁ မိနစ် test host")])
                + card("Engine", "⚙️", [("/settings", "Approval mode · provider"), ("/models", "Model ရွေး (🆓 filter)"), ("/skills", "Playbooks")])
                + "<details><summary>Approval modes</summary><ul><li><b>ask</b> — command တိုင်း ခွင့်တောင်း</li>"
                "<li><b>smart</b> (default) — အန္တရာယ်ရှိမှ ခွင့်တောင်း</li><li><b>auto</b> — admin သာ၊ မမေးတော့ (blocked command တွေ ဆက်ပိတ်)</li></ul></details>"
                "<details><summary>Providers</summary><p><b>Antigravity / OpenCode</b> — CLI ကိုယ်တိုင် agent အဖြစ် Session folder ထဲမှာ အလုပ်လုပ်ပါတယ်။<br/>"
                "<b>OpenRouter / Anthropic / Gemini / Custom</b> — API key နဲ့ ချိတ်၊ Agent X ရဲ့ tool တွေသုံးပါတယ်။</p></details>")
        await self.reply(cid, thread, body)

    async def cmd_new(self, uid: int, cid: int, thread: Optional[int], arg: str) -> None:
        if arg:
            await self.create_session(uid, cid, thread, arg)
        else:
            await self.start_session_name_prompt(uid, cid, thread)

    async def switch_project(self, uid: int, cid: int, thread: Optional[int], name: str, create: bool = False) -> None:
        # Kept for old buttons/commands: "switching" now means entering a session.
        await self.enter_session(uid, cid, thread, name, create=create)

    async def cmd_projects(self, uid: int, cid: int, thread: Optional[int]) -> None:
        names = list_projects(uid)
        cur = self.active_project(uid)
        btns = [btn("📂 " + n[:24], f"proj:{n}") for n in sorted(names)[:30]]
        rows = [btns[i:i + 2] for i in range(0, len(btns), 2)]
        rows.append([btn("🆕 Create Session", "newp", style="primary"), btn("🧭 Menu", "menu:home")])
        txt = "<p>ဝင်ချင်တဲ့ Session ကိုနှိပ်ပါ။</p>" if names else "<p>Session မရှိသေးပါ။ အသစ်ဖန်တီးပါ။</p>"
        await self.reply(cid, thread, f"<h3>📂 Sessions</h3>{txt}", kb(*rows))

    async def cmd_files(self, uid: int, cid: int, thread: Optional[int]) -> None:
        ws = get_workspace(uid, self.active_project(uid))
        await self.reply(cid, thread, f"<h3>📂 {esc(ws.project)}</h3><pre>{esc(tree(ws.root, 3, 150))}</pre><p><i>/get path — download a file</i></p>",
                         kb([btn("📦 Download ZIP", "zip", style="primary")]))

    async def send_zip(self, uid: int, cid: int, thread: Optional[int]) -> None:
        ws = get_workspace(uid, self.active_project(uid))
        p = await asyncio.to_thread(zip_project, ws)
        if p.stat().st_size > 49_000_000:
            await self.reply(cid, thread, "❌ The project ZIP exceeds Telegram's 50 MB limit. Use /get for single files.")
        else:
            await self.tg.send_file(cid, p, f"📦 {ws.project}", thread)
        with contextlib.suppress(OSError):
            p.unlink()

    async def cmd_undo(self, uid: int, cid: int, thread: Optional[int]) -> None:
        tasks = [t for t in self.db.recent_tasks(uid, self.active_project(uid), 6) if t["cp_start"] and t["status"] != "running"]
        if not tasks:
            await self.reply(cid, thread, "Nothing to undo yet in this session.")
            return
        rows = [[btn(f"↩️ #{t['id']} · {short(t['goal'], 28)}", f"undo:{t['id']}")] for t in tasks]
        await self.reply(cid, thread, "<h3>↩️ Undo</h3><p>Pick the task to roll back <b>before</b>:</p>", kb(*rows))

    def settings_view(self, uid: int) -> tuple[str, JSON]:
        u = self.db.user(uid) or {}
        mode = u.get("approval") or Cfg.DEFAULT_APPROVAL
        if mode == "auto" and not self.db.is_admin(uid):
            mode = "smart"
        pname = u.get("provider") if u.get("provider") and provider_config(u["provider"], uid=uid) else default_provider(uid)
        pc = provider_config(pname, u.get("model") if pname == u.get("provider") else None, uid=uid) if pname else None
        modes = [("ask", "🗝", "Ask"), ("smart", "🧠", "Smart")] + ([("auto", "🚀", "Auto")] if self.db.is_admin(uid) else [])
        # Selected mode shows 🥇 instead of its own icon, so each button has exactly one leading Premium emoji.
        r1 = [btn(f"{'🥇' if mode == k else ico} {name}", f"set:mode:{k}", style="success" if mode == k else None) for k, ico, name in modes]
        r2 = [btn(("🥇 " if p == pname else "") + p, f"set:prov:{p}", style="primary" if p == pname else None) for p in available_providers(uid)]
        all_providers = ("anthropic", "openrouter", "gemini", "antigravity", "opencode", "custom")
        status = ", ".join(f"{p} ✔" if provider_connected(p, uid) else f"{p} —" for p in all_providers)
        body = card("Settings", "🗝", [("Approval", mode), ("Model", f"<code>{esc(pc['name'] + ':' + pc['model']) if pc else 'none configured'}</code>"),
                                       ("Providers", esc(status)), ("Session", esc(self.active_project(uid)))],
                    foot="Provider ချိတ်ရန် အောက်က Connect ကိုနှိပ်ပါ · Model ပြောင်းရန် /model &lt;id&gt;")
        rows = [r1]
        if r2:
            rows.append(r2)
        rows.append([btn("🔱 Connect my provider", "cfg", style="primary"), btn("📔 Models", "models")])
        connected = [p for p in all_providers if provider_connected(p, uid)]
        if connected:
            rows.append([btn(f"🪣 {p}", f"cfgdel:{p}", style="danger") for p in connected[:6]])
        return body, kb(*rows)

    async def cmd_settings(self, uid: int, cid: int, thread: Optional[int]) -> None:
        h, mk = self.settings_view(uid)
        await self.reply(cid, thread, h, mk)

    async def cmd_models(self, uid: int, cid: int, thread: Optional[int]) -> None:
        h, mk = self.providers_view(uid)
        await self.reply(cid, thread, h, mk)

    async def cmd_model(self, uid: int, cid: int, thread: Optional[int], arg: str) -> None:
        if not arg:
            await self.cmd_settings(uid, cid, thread)
            return
        u = self.db.user(uid) or {}
        pname = u.get("provider") if u.get("provider") and provider_config(u["provider"], uid=uid) else default_provider(uid)
        model_name = arg.split("/", 1)[1] if "/" in arg else arg
        if "/" in arg:
            candidate = arg.split("/", 1)[0]
            if provider_config(candidate, uid=uid):
                pname, model_name = candidate, arg.split("/", 1)[1]
        if not pname or not provider_config(pname, model_name, uid=uid):
            await self.reply(cid, thread, "❌ No provider configured for that model.")
            return
        self.db.set(uid, "provider", pname)
        self.db.set(uid, "model", model_name)
        await self.reply(cid, thread, f"✅ Model set to <code>{esc(pname)}/{esc(model_name)}</code>")

    async def cmd_skills(self, uid: int, cid: int, thread: Optional[int]) -> None:
        sk = Skills(uid).all()
        lis = "".join(f"<li><b>{esc(k)}</b> — {esc(v[0])}{'' if v[2] else ' <i>(yours)</i>'}</li>" for k, v in sk.items())
        rows = [[btn(f"🗑 {k}", f"skdel:{k}", style="danger")] for k, v in sk.items() if not v[2]][:8]
        await self.reply(cid, thread, f"<h3>🧠 Skills</h3><p>Playbooks the agent loads on its own. It can also learn new ones.</p><ul>{lis}</ul>", kb(*rows) if rows else None)

    async def cmd_status(self, uid: int, cid: int, thread: Optional[int]) -> None:
        run = self.runs.get((cid, thread))
        procs = self.bg.get(uid, {})
        live = [p for p in procs.values() if p.alive]
        u = self.db.user(uid) or {}
        rows = [("📁 Session", esc(self.active_project(uid))), ("🔐 Approval", u.get("approval") or Cfg.DEFAULT_APPROVAL),
                ("🤖 Providers", esc(", ".join(available_providers(uid)) or "none")),
                ("⚙️ Running tasks", f"{len(self.runs)}/{Cfg.MAX_CONCURRENT_TASKS}")]
        if run:
            rows.append(("🚀 Current task", f"#{run.id} · {dur(time.time() - run.started)} · {len(run.steps)} steps"))
        if live:
            rows.append(("🖥 Background", ", ".join(f"#{p.id} <code>{esc(short(p.cmd, 30))}</code>" for p in live)))
        mk = kb([btn("Stop task", f"stop:{run.id}", style="danger")]) if run else kb([btn("📁 Projects", "menu:projects"), btn("🗝 Settings", "menu:settings")])
        await self.reply(cid, thread, card("Agent status", "📊", rows), mk)

    async def cmd_admin(self, cid: int, thread: Optional[int], cmd: str, arg: str) -> None:
        if cmd == "users":
            rows = "".join(f"<li><code>{u['uid']}</code> {esc(u['name'] or '')} · {u['role']}{' · banned' if u['banned'] else ''}</li>" for u in self.db.users())
            await self.reply(cid, thread, f"<h3>👥 Users</h3><ul>{rows}</ul>")
        elif cmd == "stats":
            s = self.db.stats()
            await self.reply(cid, thread, f"<h3>📈 Stats</h3><p>Users {s['users']} · Tasks {s['tasks']}<br>Tokens in {kfmt(s['tokens_in'])} / out {kfmt(s['tokens_out'])}</p>")
        elif cmd == "broadcast":
            if not arg:
                await self.reply(cid, thread, "Usage: <code>/broadcast text</code>")
                return
            n = 0
            for u in self.db.users():
                if not u["banned"]:
                    with contextlib.suppress(TGError):
                        await self.tg.send(u["uid"], f"<h4>📢 Announcement</h4><p>{esc(arg)}</p>")
                        n += 1
            await self.reply(cid, thread, f"Sent to {n} user(s).")
        else:
            if not arg.lstrip("-").isdigit():
                await self.reply(cid, thread, f"Usage: <code>/{cmd} user_id</code>")
                return
            uid = int(arg)
            if cmd == "allow":
                self.db.add_user(uid)
            elif cmd == "ban" and uid not in Cfg.ADMIN_IDS:
                self.db.add_user(uid)
                self.db.set(uid, "banned", 1)
            elif cmd == "unban":
                self.db.set(uid, "banned", 0)
            await self.reply(cid, thread, f"✅ /{cmd} {uid}")


# ============================================================================
# ENTRY POINT
# ============================================================================


async def amain() -> None:
    if not Cfg.BOT_TOKEN:
        print("BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
        return
    Cfg.WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    bot = Bot()
    await bot.start()
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda: main_task and main_task.cancel())
    try:
        await bot.poll()
    except asyncio.CancelledError:
        pass
    finally:
        await bot.shutdown()
        log.info("Stopped.")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
