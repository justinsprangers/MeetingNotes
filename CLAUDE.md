# CLAUDE.md: Meeting Notes

Local web app: record/upload/paste a meeting → local Whisper transcript → notes in whatever form the user asks for (Claude API or Ollama). User docs are in [README.md](README.md); keep it current when features, settings, or setup change.

**Public repo (MIT).** No personal info, real meeting content, API keys, or `settings.json`/`meetings/` in commits.

## Rules
- Must run without Claude Code: `run.bat`/`run.sh` set up `.venv` on first run (marker `.venv/setup-done`). New dependencies go in `requirements.txt` (GPU-only ones in `requirements-gpu.txt`); tell the user to delete `.venv` to pick them up.
- Server binds to 127.0.0.1 only (the API key lives in settings.json). Port 8806 (`config.PORT`); 8765 is StudyHub's.
- Claude calls follow the claude-api skill: `claude-opus-5-5`, streaming, `output_config.effort` (thinking can't be disabled on Opus 5.5), top-level `cache_control`, `fallbacks="default"` with beta `server-side-fallback-2026-07-01`. No `temperature`/`top_p`.
- **History is append-only.** Opus 5.5 rejects edited history (preserved thinking). `build_user_message` sends only what the model hasn't seen: the full transcript first, then `<transcript_continued>` deltas, or `<transcript_revised>` if the user edited earlier text. Claude turns are stored with their raw content blocks (`raw`) and replayed unchanged.
- A failed request records nothing, so the next one resends the same delta.
- Memory (`memory.py`, `memory.json`): glossary + preferences + saved examples, appended to the system prompt. For Claude the full system prompt is snapshotted into `Meeting.system` on the first successful reply (changing it mid-chat would break append-only history); Reset chat / New meeting picks up new memory. Ollama and Copy for AI use live memory every time.
- Copy for AI: `/api/copy-prompt` sends everything the first time, then `build_user_message` deltas against `Meeting.copied_transcript` (same append-only idea as Claude, for the user's external chat). `restart` or Reset chat clears it.
- Ollama calls send `think: false` (retried without it if a model rejects the field): gemma4 otherwise thinks first and was ~3× slower with no quality gain. `num_ctx` is sized to the input (`_context_size`, 8K–32K).
- Ollama two-pass (`ollama_answer`): transcripts over `SINGLE_PASS_CHARS` are split at line breaks into `SECTION_CHARS` sections; the extraction prompt gets the glossary and context (without it, "IP"/"Mariah" leaked through); each section is extracted once (cached in `Meeting.extractions`, keyed by model + context + section text), then the answer is written from the notes. Ollama rebuilds its messages from `display` text each time; the `text` field still gets the Claude-style delta so switching provider mid-chat works.
- Progress notes stream inside the reply as `\x01note\x02`; `showStreamed` in app.js shows the latest until real text arrives.
- Recommended Ollama models live in `RECOMMENDED_MODELS` (summarizer.py), chosen by GPU memory (nvidia-smi) or RAM. Checked against ollama.com on 2026-10-24; re-check when updating.
- Recording: mic via sounddevice; system audio via PyAudioWPatch WASAPI loopback (Windows only). Sources are aligned by wall clock in `_mix_once` because loopback delivers nothing during silence. Chunks are cut at the quietest 0.3 s in the last 5 s.
- Transcription jobs run one at a time on a single worker thread so the transcript stays in order. `Meeting.whisper_prompt()` passes context + transcript tail as Whisper's `initial_prompt`.
- Loop guard (a real 57-min meeting looped "Adioshi." for 20 min, because the looped tail went back in as the prompt): `condition_on_previous_text=False`, `collapse_repeats`, `_hallucinated` drops segments by compression ratio / no-speech, `add_segment` skips a line identical to the previous one, and `whisper_prompt` drops a low-diversity tail.
- `_FILLER` drops a lone "Thank you."/"Bye" in a chunk's last 1.5 s or stretched over 8+ s (seen 9× in one meeting).
- `memory.whisper_terms` sends only glossary terms (text before ':') as a comma list. Sending "Ben C: Treasurer" lines made Whisper prefix lines with "Ben C.:" when Ben wasn't there.
- Speaker labels: the recorder keeps per-source tracks per chunk; once system audio has been active `CALL_SECONDS`, `app.speaker` labels segments "Others" (system audio active) or "Me" (mic only). Uploads and mic-only recordings get no labels. The system prompt forbids guessing speakers.
- Pre-warm (`app.start_prewarm` → `summarizer.ollama_prewarm`): when transcription finishes (or a meeting is opened) with provider Ollama, it sends system + first message (up to `</transcript>`) with `num_predict` 1 so Ollama caches the prefix; the first answer then starts in ~4 s instead of ~42 s. Must match exactly: same system prompt, same `_first_message`, and same `num_ctx` (`_prefix_context`, sized from the prefix plus fixed `ANSWER_TOKENS`); a different `num_ctx` reloads the model and loses the cache.
- `/api/ask` with Ollama calls `transcriber.unload()` when idle (Whisper large-v3-turbo holds ~2 GB VRAM). Ollama requests send `keep_alive` `KEEP_ALIVE`.
- Never call `Recorder.stop()` while holding `lock`: it flushes a final chunk whose callback takes the lock.

## Gotchas
- `soundcard` crashed (heap corruption in its cffi Windows backend) on the user's machine; that's why it's PyAudioWPatch + sounddevice.
- Windows loopback returns all zeros when the speakers are muted or at 0 volume. The UI warns after ~20 s of silent system audio.
- GPU: ctranslate2 needs cuBLAS/cuDNN DLLs; `transcriber._add_cuda_dlls` adds the pip `nvidia/*/bin` folders. Without them, CUDA fails on first inference, so `_load` runs a 1 s warm-up before accepting a device.
- The Anthropic SDK raises `TypeError` ("Could not resolve authentication method") at request time when no key is set; summarizer maps it to a friendly message.
- The browser pane's screenshots lagged/rendered oddly here; verify with `read_page`/`get_page_text`/JS instead.

## Testing
- Launch entry `meetingnotes` in `C:\Claude\.claude\launch.json` (runs the venv Python with `--no-browser`).
- Test audio: make a WAV with Windows TTS (`System.Speech.Synthesis.SpeechSynthesizer.SetOutputToWaveFile`) and POST it to `/api/upload`. Sound played from Claude's shell doesn't reach the speakers, so live loopback can't be tested end to end from a session; simulate by pushing audio into `_Source` objects and calling `_mix_once`.
- Don't spend the user's API credits on tests without asking.
- Model quality: `eval/run_eval.py <model>` runs a fictional 7.5-min meeting (`eval/transcript.txt`) single- and two-pass; score by hand against `eval/answer_key.md`. Ollama is installed on the dev machine with gemma4:12b (36/49 layers on the 8 GB GPU, ~13 tok/s).
- For plumbing tests without a real model, a mock server (Flask on port 11434 implementing `/api/tags`, `/api/chat` streaming and non-streaming, and `/api/pull`) tests the plumbing; it can't judge output quality.
