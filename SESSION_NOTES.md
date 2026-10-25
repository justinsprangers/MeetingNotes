# Meeting Notes: session notes

## Status
- v1 built: record (mic + Windows system audio), upload, or paste → live local Whisper transcript → chat-style requests with presets, Claude or Ollama, meetings saved and reopenable.
- Verified: file transcription (CPU small and GPU large-v3-turbo), recorder mixing/resampling/chunking (simulated input), settings, past meetings, every error path (no key, bad key, Ollama missing/unreachable).
- **Not verified:** a real Claude response (no API key on the dev machine; the bad-key test reached the API and got a 401), Ollama responses, live system-audio capture end to end (the dev machine's speakers were muted), Mac/Linux.

## Next steps
- First real use: add an API key, record a short call, check the notes quality.
- Push to a public GitHub repo.
- Ideas: speaker labels (mic = "me" vs system = "others"), export notes to .docx, model picker dropdown.

## Decisions & gotchas
- See `CLAUDE.md` for the append-only history rule, recorder design, and GPU DLL handling.
- Default model `claude-opus-5-5` at effort `medium`. Ollama gets `num_ctx` 32768, because its default would silently cut off long transcripts.
- Whisper auto: large-v3-turbo on CUDA, small on CPU.

## Log
- 2026-10-24: v1 built and tested; own git repo (MIT)
