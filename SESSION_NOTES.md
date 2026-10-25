# Meeting Notes: session notes

## Status
- Record (mic + Windows system audio), upload, or paste → live local Whisper transcript → chat-style requests with presets. Summarize with the Claude API, Ollama, or **Copy for Claude.ai** (the user's main no-cost path; the API is billed separately from a subscription). Meetings are saved and reopenable.
- Memory: glossary, preferences ("Remember" on requests), saved examples ("Save as example" on responses).
- Ollama: two-pass summarizing for long transcripts; a recommended model (gemma4:12b or qwen3.5:4b, by hardware) with a Download button in Settings.
- Verified: file transcription (CPU and GPU), recorder pipeline (simulated input), settings, past meetings, error paths, Copy for Claude.ai, memory, two-pass flow and its caching, and model download (against a mock Ollama).
- **Not verified:** real Claude replies (no API key on the dev machine), real Ollama output quality (Ollama not installed), live system-audio capture (speakers were muted), Mac/Linux.

## Next steps
- First real use: record a short call and try Copy for Claude.ai.
- Install Ollama with gemma4:12b and compare its notes to Claude's on a real transcript; tune `EXTRACT_PROMPT`/`SECTION_CHARS` if needed.
- Push to a public GitHub repo.
- Ideas: speaker labels (mic = "me" vs system = "others"), export notes to .docx.

## Decisions & gotchas
- See `CLAUDE.md` for the append-only history rule, memory snapshot, two-pass design, recorder design, and GPU DLL handling.
- Default Claude model `claude-opus-5-5` at effort `medium`. Ollama `num_ctx` 16384 (two-pass keeps the final input small).
- Whisper auto: large-v3-turbo on CUDA, small on CPU.

## Log
- 2026-10-24: Memory, Ollama two-pass summarizing, model recommendation + download
- 2026-10-24: Copy for Claude.ai button (uses the subscription, no API cost)
- 2026-10-24: v1 built and tested; own git repo (MIT)
