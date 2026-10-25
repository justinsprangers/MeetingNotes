"""Turns a transcript into whatever the user asks for, using Claude or a local Ollama model."""
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys

import anthropic
import requests

SYSTEM_PROMPT = """You turn meeting transcripts into whatever the user asks for: notes, summaries, action items, updates for people who missed the meeting, lists, or messages formatted for a specific app.

About the transcript:
- It comes from automatic speech recognition, so expect misheard words, missing punctuation, and no speaker names. Use context to work out what was meant.
- Never guess who said something. Name a person only when the transcript shows it: they say their own name, someone addresses them by name right before they answer, or someone says "Name will…". Otherwise write it without a name (for example "Someone suggested…"), and mark owners you can't tell as "[owner unclear]". Only if lines start with "Me:" or "Others:": "Me" is the person who recorded the meeting (their name may be in the glossary) and "Others" is someone else on the call. Never write "Me" as a name or owner in your output. When an unclear word or name matters, flag it briefly (for example "[unclear: probably IPA]") instead of guessing silently.
- Lines start with [minutes:seconds] timestamps from the start of the recording.
- It may arrive in parts while the meeting is still going. Treat later parts as a continuation of the same meeting, and update earlier output when asked.
- Side conversations, small talk, and personal remarks are usually not meeting content. Leave them out unless the user asks for them.
- If someone says something is off the record, or asks for it to be struck or left out, leave it out of everything you write.
- Meeting context from the user (attendees, roles, project names, jargon) is reliable; use it to correct names and terms.

Writing:
- Use only what the transcript supports. Don't invent decisions, owners, or dates. If something was discussed but not decided, say so.
- Follow the requested format, length, and audience exactly. By default, write concise bullet points grouped by topic, with decisions and action items (with owners when named) easy to find.
- No emojis unless the user asks for them.
- When asked for a message to paste into a chat app or email, put it in a single code block so formatting survives copying. Keep Discord messages under 2,000 characters.
- Return only the requested content, with no preamble."""

CLAUDE_EFFORT = "medium"

# Ollama: transcripts that fit the context go in whole. On a real 57-min meeting
# (55k chars, ~16k tokens) gemma4:12b on an 8 GB GPU answered in 69-112 s in one pass,
# vs 453 s two-pass, and the single-pass notes were more complete. Follow-ups start
# in seconds because Ollama reuses the cached transcript prefix. Only transcripts too
# long for MAX_CONTEXT are read in sections first (pass 1: extract the facts), then
# answered from those notes (pass 2). Single passes near 32K tokens are untested.
SINGLE_PASS_CHARS = 80000   # about 23,000 tokens (~80 min); with ANSWER_TOKENS it fits MAX_CONTEXT
SECTION_CHARS = 12000       # about 3,000 tokens, roughly 10-15 minutes
MAX_CONTEXT = 32768         # Ollama's default context is small enough to silently cut off input
KEEP_ALIVE = "2m"           # long enough for a follow-up question, short enough to give the RAM back
PREWARM_KEEP_ALIVE = "15m"  # after a pre-warm, wait longer for the first question
ANSWER_TOKENS = 8192        # context room for the request, follow-ups and answers
EXTRACT_PROMPT = """Below is section {i} of {n} of a meeting transcript. It comes from speech recognition, so expect misheard words; use the glossary and context to correct names and terms.

Write compact notes of everything in this section that could matter later:
- Topics discussed, with the key points and any numbers, names, dates, and reasons given
- Decisions made
- Action items (who, what, and when, if mentioned)
- Open questions, and anything that needs someone's approval
- If something is said to be off the record, write only "[off-the-record part omitted]" for it

The transcript has no speaker names. Name a person only when the text itself shows who spoke (they say their name, are addressed by name, or are named as the owner); never guess from the glossary. Only if lines start with "Me:" or "Others:": "Me" is the person who recorded and "Others" is someone else on the call. Never write "Me" as a name.

Skip small talk and side conversations. Use short bullet points and keep the [mm:ss] timestamp at the start of each item. Don't add anything that isn't in the transcript.
{context}
<transcript_section>
{text}
</transcript_section>"""

RECOMMENDED_MODELS = [
    # (model, download size, minimum GPU memory GB, or minimum system RAM GB)
    # gemma4:12b only partly fits on an 8 GB card (36 of 49 layers), so it's slow there
    # (459 s first answer on a 57-min meeting vs 140 s for qwen3.5:4b), but qwen invented
    # deadlines and got facts wrong on the same meeting, so gemma stays the pick at 8 GB.
    ("gemma4:12b", "7.7 GB", 7.5, 16),   # 7.5: "8 GB" cards report slightly under 8
    ("qwen3.5:4b", "3.3 GB", 0, 0),
]


class SummarizerError(Exception):
    pass


def system_prompt(memory_section: str) -> str:
    return SYSTEM_PROMPT + ("\n\n" + memory_section if memory_section else "")


def _context_block(context):
    return f"<meeting_context>\n{context.strip()}\n</meeting_context>" if context.strip() else ""


def build_user_message(request_text, transcript, sent_transcript, context, sent_context):
    """Only send what the model hasn't seen yet, so the conversation history stays append-only."""
    if not transcript.strip():
        raise SummarizerError("There's no transcript yet. Record, upload audio, or paste a transcript first.")
    parts = []
    if context.strip() and context != sent_context:
        parts.append(_context_block(context))
    if not sent_transcript:
        parts.append(f"<transcript>\n{transcript.strip()}\n</transcript>")
    elif transcript.startswith(sent_transcript):
        delta = transcript[len(sent_transcript):].strip()
        if delta:
            parts.append(f"<transcript_continued>\n{delta}\n</transcript_continued>")
    else:
        parts.append("<transcript_revised>\nThe transcript was edited. This full version replaces the "
                     f"earlier one.\n{transcript.strip()}\n</transcript_revised>")
    parts.append(request_text)
    return "\n\n".join(parts)


def copy_prompt(system, request_text, transcript, context):
    """One self-contained message to paste into an external AI chat (no API cost)."""
    if not transcript.strip():
        raise SummarizerError("There's no transcript yet. Record, upload audio, or paste a transcript first.")
    parts = [system, _context_block(context), f"<transcript>\n{transcript.strip()}\n</transcript>",
             request_text or "Reply with one line confirming you have the transcript, then wait for my request."]
    return "\n\n".join(p for p in parts if p)


# ---- Claude ---------------------------------------------------------------

def claude_stream(settings, system, history, user_message):
    """Yields ("text", str) pieces, then ("raw", content) so the turn can be replayed later."""
    key = settings.get("anthropic_api_key") or None
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    messages = [{"role": t["role"], "content": t.get("raw") or t["text"]} for t in history]
    messages.append({"role": "user", "content": user_message})
    try:
        # fallbacks="default": if a safety filter wrongly declines, the API retries on another model.
        with client.beta.messages.stream(
            model=settings["claude_model"],
            max_tokens=64000,
            system=system,
            messages=messages,
            output_config={"effort": CLAUDE_EFFORT},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as s:
            for text in s.text_stream:
                yield ("text", text)
            final = s.get_final_message()
    except TypeError as e:
        if "authentication" not in str(e).lower():
            raise
        raise SummarizerError("No Anthropic API key. Add one in Settings, switch to Ollama, "
                              "or use Copy for AI to paste into Claude.ai or another AI chat instead.")
    except anthropic.AuthenticationError:
        raise SummarizerError("The Anthropic API key was rejected. Check it in Settings.")
    except anthropic.PermissionDeniedError:
        raise SummarizerError("This API key can't use that model. Check the model name in Settings.")
    except anthropic.NotFoundError:
        raise SummarizerError(f"Model '{settings['claude_model']}' wasn't found. Check it in Settings.")
    except anthropic.RateLimitError:
        raise SummarizerError("Rate limited by Anthropic. Wait a minute and try again.")
    except anthropic.BadRequestError as e:
        raise SummarizerError(f"Anthropic rejected the request: {e.message}")
    except anthropic.APIStatusError as e:
        raise SummarizerError(f"Anthropic API error {e.status_code}. Try again shortly.")
    except anthropic.APIConnectionError:
        raise SummarizerError("Couldn't reach Anthropic. Check your internet connection.")

    if final.stop_reason == "refusal":
        yield ("text", "\n\n[The model declined this request.]")
    elif final.stop_reason == "max_tokens":
        yield ("text", "\n\n[Stopped: the response hit the length limit.]")
    yield ("raw", final.to_dict(mode="json")["content"])


# ---- Ollama ---------------------------------------------------------------

def split_sections(transcript):
    """Split at line breaks into chunks of about SECTION_CHARS."""
    sections, current = [], ""
    for line in transcript.strip().splitlines():
        if current and len(current) + len(line) > SECTION_CHARS:
            sections.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        sections.append(current)
    return sections


def ollama_answer(settings, system, history, request_text, transcript, context, cache, glossary=""):
    """Yields ("status", str) progress notes and ("text", str) pieces.

    history: earlier turns as {"role", "display"}; the transcript is supplied fresh each time.
    cache: dict of section notes, keyed by model + section text, kept on the meeting.
    """
    model = settings["ollama_model"]
    if not model:
        raise SummarizerError("Pick an Ollama model in Settings.")
    if not transcript.strip():
        raise SummarizerError("There's no transcript yet. Record, upload audio, or paste a transcript first.")

    if len(transcript) <= SINGLE_PASS_CHARS:
        material = f"<transcript>\n{transcript.strip()}\n</transcript>"
    else:
        sections = split_sections(transcript)
        notes, used = [], {}
        context_line = "".join(f"\n{label}:\n{text.strip()}\n" for label, text in
                               (("Glossary", glossary), ("Meeting context", context)) if text.strip())
        for i, text in enumerate(sections, 1):
            key = hashlib.sha1(f"{model}\n{context_line}\n{text}".encode()).hexdigest()
            if key not in cache:
                yield ("status", f"Reading the transcript: section {i} of {len(sections)}…")
                prompt = EXTRACT_PROMPT.format(i=i, n=len(sections), context=context_line, text=text)
                cache[key] = _ollama_once(settings, [{"role": "user", "content": prompt}])
            used[key] = cache[key]
            start = re.match(r"\[[\d:]+\]", text)
            notes.append(f"Section {i}{' (from ' + start.group(0) + ')' if start else ''}:\n{cache[key].strip()}")
        cache.clear()
        cache.update(used)  # drop notes for text that has since changed
        material = ("<meeting_notes>\nThe transcript was long, so it was read in sections first. These notes "
                    "cover the whole meeting, in order. Treat them as the transcript.\n\n"
                    + "\n\n".join(notes) + "\n</meeting_notes>")
        yield ("status", "Writing…")

    first = _first_message(context, material)
    turns = [t["display"] for t in history] + [request_text]
    messages = [{"role": "system", "content": system}]
    for i, text in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"{first}\n\n{text}" if i == 0 else text})
    num_ctx = _prefix_context(system, first, sum(len(t) for t in turns))
    yield from (("text", t) for t in _ollama_stream(settings, messages, num_ctx))


def ollama_prewarm(settings, system, transcript, context):
    """Have Ollama read the transcript now, so the first answer starts in seconds.

    Ollama keeps the processed prompt and reuses the matching start of the next request.
    That only works if this sends the same system prompt, the same first message up to
    the end of the transcript, and the same num_ctx (a different one reloads the model).
    """
    if not settings["ollama_model"] or not transcript.strip() or len(transcript) > SINGLE_PASS_CHARS:
        return
    first = _first_message(context, f"<transcript>\n{transcript.strip()}\n</transcript>")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    _ollama_post(settings, {"messages": messages, "keep_alive": PREWARM_KEEP_ALIVE},
                 stream=False, num_ctx=_prefix_context(system, first), num_predict=1)


def _first_message(context, material):
    return "\n\n".join(p for p in (_context_block(context), material) if p)


def _round_context(tokens):
    return int(min(MAX_CONTEXT, max(8192, -(-tokens // 4096) * 4096)))


def _context_size(messages):
    # Big enough for the input plus a long answer (~3.5 characters per token), in 4K steps.
    return _round_context(sum(len(m["content"]) for m in messages) / 3.5 + 4096)


def _prefix_context(system, first, rest_chars=0):
    # Sized from the transcript alone, with fixed room for requests, follow-ups and the
    # answer, so it stays the same from the pre-warm through the first several requests.
    return _round_context((len(system) + len(first)) / 3.5 + max(ANSWER_TOKENS, rest_chars / 3.5 + 4096))


def _ollama_post(settings, body, stream, num_ctx=None, num_predict=None):
    url = settings["ollama_url"].rstrip("/") + "/api/chat"
    options = {"num_ctx": num_ctx or _context_size(body["messages"])}
    if num_predict is not None:
        options["num_predict"] = num_predict
    # Thinking models (gemma4, qwen3.5) otherwise reason at length first: about 3x slower,
    # and the notes were just as good without it in testing.
    # keep_alive: free the model's memory soon after the last request instead of Ollama's 5 min.
    body = {"model": settings["ollama_model"], "stream": stream, "think": False, "keep_alive": KEEP_ALIVE,
            "options": options, **body}
    try:
        r = requests.post(url, json=body, stream=stream, timeout=(5, 900))
        if r.status_code == 400 and "think" in r.text:  # model without a thinking switch
            del body["think"]
            r = requests.post(url, json=body, stream=stream, timeout=(5, 900))
    except requests.RequestException:
        raise SummarizerError(f"Can't reach Ollama at {settings['ollama_url']}. Is it running?")
    if r.status_code != 200:
        raise SummarizerError(f"Ollama error: {r.text[:300]}")
    return r


def _ollama_once(settings, messages):
    data = _ollama_post(settings, {"messages": messages}, stream=False).json()
    if "error" in data:
        raise SummarizerError(f"Ollama error: {data['error']}")
    return data.get("message", {}).get("content", "")


def _ollama_stream(settings, messages, num_ctx=None):
    r = _ollama_post(settings, {"messages": messages}, stream=True, num_ctx=num_ctx)
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise SummarizerError(f"Ollama error: {data['error']}")
        text = data.get("message", {}).get("content", "")
        if text:
            yield text
        if data.get("done"):
            break


def ollama_models(settings):
    try:
        r = requests.get(settings["ollama_url"].rstrip("/") + "/api/tags", timeout=2)
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, ValueError):
        return None  # Ollama isn't running


def ollama_pull(settings, model, on_progress):
    """Download a model through Ollama. on_progress(text) gets status updates."""
    url = settings["ollama_url"].rstrip("/") + "/api/pull"
    try:
        r = requests.post(url, json={"model": model, "stream": True}, stream=True, timeout=(5, 3600))
    except requests.RequestException:
        raise SummarizerError(f"Can't reach Ollama at {settings['ollama_url']}. Is it running?")
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise SummarizerError(f"Ollama error: {data['error']}")
        status = data.get("status", "")
        if data.get("total"):
            status += f" {data.get('completed', 0) / data['total']:.0%}"
        on_progress(status)


# ---- Choosing a model -------------------------------------------------------

def _gpu_memory_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return max(int(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _ram_gb():
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (OSError, ValueError, AttributeError):
        return 0


def recommend_ollama_model():
    gpu, ram = _gpu_memory_gb(), _ram_gb()
    for model, size, min_gpu, min_ram in RECOMMENDED_MODELS:
        if (min_gpu and gpu >= min_gpu) or (min_ram and ram >= min_ram) or not (min_gpu or min_ram):
            break
    hardware = f"{gpu:.0f} GB GPU, {ram:.0f} GB RAM" if gpu else f"no NVIDIA GPU, {ram:.0f} GB RAM"
    return {"model": model, "size": size, "hardware": hardware}
