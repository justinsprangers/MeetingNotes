"""Turns a transcript into whatever the user asks for, using Claude or a local Ollama model."""
import json

import anthropic
import requests

SYSTEM_PROMPT = """You turn meeting transcripts into whatever the user asks for: notes, summaries, action items, updates for people who missed the meeting, lists, or messages formatted for a specific app.

About the transcript:
- It comes from automatic speech recognition, so expect misheard words, missing punctuation, and no speaker names. Use context to work out what was meant. When an unclear word or name matters, flag it briefly (for example "[unclear: probably IPA]") instead of guessing silently.
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
OLLAMA_CONTEXT = 32768  # Ollama's default context is small enough to silently cut off a long transcript


class SummarizerError(Exception):
    pass


def build_user_message(request_text, transcript, sent_transcript, context, sent_context):
    """Only send what the model hasn't seen yet, so the conversation history stays append-only."""
    if not transcript.strip():
        raise SummarizerError("There's no transcript yet. Record, upload audio, or paste a transcript first.")
    parts = []
    if context.strip() and context != sent_context:
        parts.append(f"<meeting_context>\n{context.strip()}\n</meeting_context>")
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


def stream(settings, history, user_message):
    """Yields ("text", str) pieces, then ("raw", content) for Claude so it can be replayed later."""
    if settings["provider"] == "ollama":
        yield from _ollama(settings, history, user_message)
    else:
        yield from _claude(settings, history, user_message)


def _claude(settings, history, user_message):
    key = settings.get("anthropic_api_key") or None
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    messages = [{"role": t["role"], "content": t.get("raw") or t["text"]} for t in history]
    messages.append({"role": "user", "content": user_message})
    try:
        # fallbacks="default": if a safety filter wrongly declines, the API retries on another model.
        with client.beta.messages.stream(
            model=settings["claude_model"],
            max_tokens=64000,
            system=SYSTEM_PROMPT,
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
        raise SummarizerError("No Anthropic API key. Add one in Settings, or switch to Ollama.")
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


def _ollama(settings, history, user_message):
    model = settings["ollama_model"]
    if not model:
        raise SummarizerError("Pick an Ollama model in Settings.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t["role"], "content": t["text"]} for t in history]
    messages.append({"role": "user", "content": user_message})
    url = settings["ollama_url"].rstrip("/") + "/api/chat"
    body = {"model": model, "messages": messages, "stream": True, "options": {"num_ctx": OLLAMA_CONTEXT}}
    try:
        r = requests.post(url, json=body, stream=True, timeout=(5, 600))
    except requests.RequestException:
        raise SummarizerError(f"Can't reach Ollama at {settings['ollama_url']}. Is it running?")
    if r.status_code != 200:
        raise SummarizerError(f"Ollama error: {r.text[:300]}")
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise SummarizerError(f"Ollama error: {data['error']}")
        text = data.get("message", {}).get("content", "")
        if text:
            yield ("text", text)
        if data.get("done"):
            break


def ollama_models(settings):
    try:
        r = requests.get(settings["ollama_url"].rstrip("/") + "/api/tags", timeout=2)
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, ValueError):
        return []
