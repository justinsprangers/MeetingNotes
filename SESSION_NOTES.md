# Meeting Notes: session notes

## Status
- Record (mic + Windows system audio), upload, or paste → live local Whisper transcript with voice speaker labels → chat-style requests with multi-select presets. Summarize with the Claude API, Ollama (gemma4:e4b-it-qat, the user's setting since 2026-11-02), or **Copy for AI** (the user's main no-cost path; first copy = everything, then "Copy follow-up" = request + new transcript). Meetings are saved and reopenable.
- Memory: glossary, preferences, saved examples, saved quick prompts (custom preset buttons), remembered voices.
- Ollama: thinking off, single pass up to 80k chars (~80 min, untested near the top), two-pass beyond; pre-warmed after transcription (first text ~4 s on a 57-min meeting, full notes ~2 min); keep_alive 10 min; unloaded whenever Whisper needs the GPU.
- Speaker labels (2026-11-01): 10/25 meeting → 7 voices, Sebastian's lines consistent; a named voice was recognized in a separate recording; the 11/01 recording (poor audio) splits into many small speakers. Live path: 10 min of chunks in 100 s with labels.
- Live-transcript stall (11/01, 83 min, finished 26 min after stop): GPU replay runs 25x real time, so Whisper was on CPU or starved of GPU. Fixed causes: Ollama holding the GPU (reproduced), sticky CPU fallback (GPU retry), and the transcript box freezing while focused (review finding). Now shows "N behind" and logs per-chunk timing to `logs/meetingnotes.log`.
- **Not verified:** real Claude API replies (no key), a real live recording since the stall fixes, live system-audio capture, Mac/Linux.
- Checked 2026-11-02: `transcriber.unload()` frees VRAM (1189 → 101 MiB). `int8_float16` vs float16 on 5 min of 10/25: 17% faster, 93% word agreement, differences only on unclear audio (neither better), kept.
- Ollama `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` tried 2026-11-02 on the 57-min meeting: 36 vs 34 layers on GPU but no speed gain (163 vs 152 s; 8.1 vs 8.7 tok/s). gemma4's KV cache is already small (~0.9 GB); the weights are the bottleneck. Reverted; don't retry. Note: env vars set from a shell only reach Ollama if it's started from a shell that has them.

## Eval results (2026-10-25)
- 7.5-min fictional meeting (`eval/`): gemma4:12b single pass got all 7 action items and 4 approvals; thinking on was ~3× slower with no gain.
- 57-min real meeting: gemma4:12b grounded (one owner misattributed); qwen3.5:4b ~5× faster but invented owners, dates and decisions.
- 2026-11-02, user wanted answers fast (not streamed slowly): writing speed is the bottleneck (12b ~9 tok/s, 13 of 49 layers on CPU). gemma4:12b-it-qat: same 36/49 layers, no gain (removed). **gemma4:e4b-it-qat**: fully on GPU, ~50 tok/s; 57-min notes in 29 s vs 138 s, comparable accuracy (eval: 7/7 actions, 6/6 decisions, 4/4 approvals, no off-record leak; slips: "two five-gallon" → "25-gallon", one "IP"). Now recommended for 6–10 GB GPUs and set in the user's settings.json. Kept streaming: hiding text until done wouldn't shorten the wait.

## Next steps
- User: restart run.bat (it installs sherpa-onnx automatically), record a real meeting, name the speakers, and check the "behind" status and `logs/meetingnotes.log` if it lags.
- Push to a public GitHub repo. Windows installer after the user's testing (reuse LaserEngraver's build setup).

## Decisions & gotchas
- See `CLAUDE.md` for append-only history, memory snapshot, two-pass, pre-warm, speaker labels, GPU sharing, recorder design, GPU DLLs.
- Default Claude model `claude-opus-5-5` at effort `medium`. Whisper auto: large-v3-turbo (int8_float16) on CUDA, small on CPU.

## Log
- 2026-11-02: saved quick prompts ("+ Save quick prompt", stored in memory.json, delete in Memory); browser-tested save, combine, reload, delete
- 2026-11-02: switched Ollama default to gemma4:e4b-it-qat (~5× faster answers); Ollama stays loaded during transcription when both fit on the GPU
- 2026-11-02: verified Whisper unload frees VRAM and int8_float16 accuracy; Ollama flash attention + q8 KV benchmarked, no gain, reverted
- 2026-11-01: Stop button, "reading" status, real token count sizing (78k chars = 22.9k tokens fits), pre-warm on request-box focus (once per transcript), one-click notes after transcription, polling sends transcript only on change (54 KB → 0.7 KB/s), transcript type validation
- 2026-11-01: speaker labels by voice (sherpa-onnx, centered voiceprints, regroup, naming, remembered voices); stall fixes (Ollama unloads for Whisper, GPU retry, catch-up decoding, lag status, log file, focus freeze, worker crash guard, stop race); copy confirm; keep_alive 10 min; run.bat installs new requirements; review session (Sonnet) findings folded in
- 2026-11-01: presets multi-select into one request
- 2026-10-25: Copy for AI + follow-ups; Ollama single pass + pre-warm
- 2026-10-25: Whisper loop guard + filler filter, glossary-as-terms for Whisper, no-guess speaker rule, model comparison
- 2026-10-24: Memory, Ollama two-pass, model recommendation; Copy for Claude.ai; v1 built (MIT)
