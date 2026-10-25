"""What the app learns from you, across meetings: a glossary of names and terms,
standing preferences, and example outputs you liked.

Stored in memory.json next to the app (gitignored). It's added to the
summarizer's instructions, and the glossary also helps Whisper spell names.
"""
import json
import time
import uuid

from config import APP_DIR

PATH = APP_DIR / "memory.json"
EXAMPLES_IN_PROMPT = 4     # most recent ones; more makes small models slower without helping much
EXAMPLE_CHARS = 2500


def load() -> dict:
    data = {"glossary": "", "preferences": "", "examples": []}
    if PATH.exists():
        try:
            data.update(json.loads(PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return data


def save(data: dict) -> dict:
    PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def update(glossary=None, preferences=None) -> dict:
    data = load()
    if glossary is not None:
        data["glossary"] = glossary
    if preferences is not None:
        data["preferences"] = preferences
    return save(data)


def add_preference(text: str) -> dict:
    data = load()
    prefs = data["preferences"].rstrip()
    data["preferences"] = (prefs + "\n" if prefs else "") + "- " + text.strip()
    return save(data)


def add_example(request: str, output: str) -> dict:
    data = load()
    data["examples"].append({"id": uuid.uuid4().hex[:8], "request": request, "output": output,
                             "saved": time.strftime("%Y-%m-%d")})
    return save(data)


def delete_example(example_id: str) -> dict:
    data = load()
    data["examples"] = [e for e in data["examples"] if e["id"] != example_id]
    return save(data)


def prompt_section(data: dict) -> str:
    """The part of the summarizer's instructions that comes from memory ("" if empty)."""
    parts = []
    if data["glossary"].strip():
        parts.append("<glossary>\nNames and terms that come up in the user's meetings. Use these spellings "
                     f"and meanings, and use them to correct misheard words.\n{data['glossary'].strip()}\n</glossary>")
    if data["preferences"].strip():
        parts.append("<preferences>\nThe user's standing preferences. Follow them unless a request says "
                     f"otherwise.\n{data['preferences'].strip()}\n</preferences>")
    examples = data["examples"][-EXAMPLES_IN_PROMPT:]
    if examples:
        body = "\n".join(
            f"<example>\n<request>{e['request']}</request>\n<output>\n{e['output'][:EXAMPLE_CHARS]}\n</output>\n</example>"
            for e in examples)
        parts.append("<examples>\nOutputs the user liked, from earlier meetings. When a request is similar, match "
                     "their structure, tone, and level of detail. Never reuse their content.\n"
                     f"{body}\n</examples>")
    return "\n\n".join(parts)


def whisper_terms(data: dict) -> str:
    """Glossary as a short phrase list for Whisper's prompt."""
    return data["glossary"].replace("\n", "; ")[:300]
