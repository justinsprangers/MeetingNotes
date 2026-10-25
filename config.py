"""Settings, stored in settings.json next to the app (gitignored, never committed)."""
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / "settings.json"
MEETINGS_DIR = APP_DIR / "meetings"
PORT = 8806

DEFAULTS = {
    "provider": "claude",                 # "claude" or "ollama"
    "anthropic_api_key": "",              # blank = use the ANTHROPIC_API_KEY environment variable
    "claude_model": "claude-opus-5-5",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "",
    "whisper_model": "auto",              # auto, tiny, base, small, medium, large-v3-turbo, large-v3
    "whisper_device": "auto",             # auto, cpu, cuda
    "language": "auto",                   # "auto" or a code like "en"
    "record_microphone": True,
    "record_system_audio": True,          # Windows only: captures what plays through your speakers
}


def load() -> dict:
    settings = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            settings.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass  # a broken file falls back to defaults rather than crashing the app
    return settings


def save(updates: dict) -> dict:
    settings = load()
    settings.update({k: v for k, v in updates.items() if k in DEFAULTS})
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def public(settings: dict) -> dict:
    """Settings safe to send to the browser: the API key is reduced to a hint."""
    out = dict(settings)
    key = out.pop("anthropic_api_key", "")
    out["api_key_hint"] = f"saved (ends in {key[-4:]})" if key else ""
    return out
