# CLAUDE.md: Meeting Notes

Local web app: record/upload/paste a meeting → local Whisper transcript → notes in whatever form the user asks for (Claude API or Ollama). User docs are in [README.md](README.md); keep it current when features, settings, or setup change.

**Public repo (MIT).** No personal info, real meeting content, API keys, or `settings.json`/`meetings/` in commits.

## Rules
- Must run without Claude Code: `run.bat`/`run.sh` set up `.venv` on first run (marker `.venv/setup-done`). New dependencies go in `requirements.txt` (GPU-only ones in `requirements-gpu.txt`); tell the user to delete `.venv` to pick them up.
- Server binds to 127.0.0.1 only (the API key lives in settings.json). Port 8806 (`config.PORT`); 8765 is StudyHub's.
- Claude calls follow the claude-api skill: `claude-opus-5-5`, streaming, `output_config.effort` (thinking can't be disabled on Opus 5.5), top-level `cache_control`, `fallbacks="default"` with beta `server-side-fallback-2026-07-01`. No `temperature`/`top_p`.
- **History is append-only.** Opus 5.5 rejects edited history (preserved thinking). `build_user_message` sends only what the model hasn't seen: the full transcript first, then `<transcript_continued>` deltas, or `<transcript_revised>` if the user edited earlier text. Claude turns are stored with their raw content blocks (`raw`) and replayed unchanged.
- A failed request records nothing, so the next one resends the same delta.
- Recording: mic via sounddevice; system audio via PyAudioWPatch WASAPI loopback (Windows only). Sources are aligned by wall clock in `_mix_once` because loopback delivers nothing during silence. Chunks are cut at the quietest 0.3 s in the last 5 s.
- Transcription jobs run one at a time on a single worker thread so the transcript stays in order. `Meeting.whisper_prompt()` passes context + transcript tail as Whisper's `initial_prompt`.
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
