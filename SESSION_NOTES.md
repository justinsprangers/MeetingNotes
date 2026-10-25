# Meeting Notes: session notes

## Status
- Record (mic + Windows system audio), upload, or paste → live local Whisper transcript → chat-style requests with presets. Summarize with the Claude API, Ollama, or **Copy for Claude.ai** (the user's main no-cost path; the API is billed separately from a subscription). Meetings are saved and reopenable.
- Memory: glossary, preferences ("Remember" on requests), saved examples ("Save as example" on responses).
- Ollama: thinking off, context sized to input, single pass up to ~15 min of transcript, two-pass beyond. Recommended model (gemma4:12b or qwen3.5:4b, by hardware) with a Download button.
- Verified: file transcription (CPU and GPU), recorder pipeline (simulated input), settings, past meetings, error paths, Copy for Claude.ai, memory, two-pass flow/caching (mock), and real gemma4:12b output on the eval meeting.
- **Not verified:** real Claude API replies (no key), live system-audio capture (speakers were muted), meetings longer than 7.5 min on Ollama, Mac/Linux.

## Eval results (2026-10-25, eval/transcript.txt, 7.5-min fictional meeting)
- Claude baseline (written in-session by Opus 5.5 with the same prompt): everything in the answer key.
- gemma4:12b, no thinking, single pass: all 7 action items with owners/dates, all 4 approvals, glossary spellings right, off-the-record excluded. Missed "execution first" in the notes, one open question; one stray preamble line in the update. 86 s notes, 46 s update on the laptop 4060.
- Two-pass on the same meeting: slightly more complete (all 3 open questions), 192 s first answer, 38 s follow-up.
- Thinking on was ~3× slower with no quality gain.

## Next steps
- First real use: record a short call and try Copy for Claude.ai and Ollama.
- Test Ollama on a real 30–60 min meeting; adjust `SINGLE_PASS_CHARS`/`SECTION_CHARS` if needed.
- Push to a public GitHub repo.
- After the user's testing: Windows installer (bundled Python, Start menu shortcut, GitHub Releases), reusing LaserEngraver's build setup. Keep run.bat. The user said to wait until they've tested it.
- Ideas: speaker labels (mic = "me" vs system = "others"), export notes to .docx.

## Decisions & gotchas
- See `CLAUDE.md` for the append-only history rule, memory snapshot, two-pass design, recorder design, and GPU DLL handling.
- Default Claude model `claude-opus-5-5` at effort `medium`. Whisper auto: large-v3-turbo on CUDA, small on CPU.

## Log
- 2026-10-25: Ollama installed; gemma4:12b eval; thinking off, glossary in extraction, single-pass up to ~15 min; eval/ folder
- 2026-10-24: Memory, Ollama two-pass summarizing, model recommendation + download
- 2026-10-24: Copy for Claude.ai button (uses the subscription, no API cost)
- 2026-10-24: v1 built and tested; own git repo (MIT)
