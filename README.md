# Meeting Notes

Record a meeting (or upload a recording, or paste a transcript), transcribe it on your own computer, then ask for exactly what you need:

- "Organized notes with decisions and action items"
- "A short update for my manager, who missed it: decisions, what needs their approval, what's next"
- "Every peer-mentor training topic we mentioned"
- "Format that for Discord, no emojis"

It runs as a small app in your browser. Nothing is hosted online.

## Features

- **Records your mic and your computer's audio together** (Windows), so in-person meetings and Teams/Zoom/Meet calls both work. The transcript fills in live, about every 30 seconds.
- **Transcribes locally** with [Whisper](https://github.com/SYSTRAN/faster-whisper). Audio never leaves your computer. An NVIDIA GPU makes it much faster but isn't required.
- **You choose what to get out of it:** type any request, or click a preset (meeting notes, update for someone absent, action items, decisions, Discord, email). Follow-ups work like a chat ("shorter", "add that we still need the resin order").
- **Ask mid-meeting.** New transcript text is sent automatically with each request.
- **Off the record:** say so in the meeting and that part is left out of the notes.
- **Learns your style:** a Memory with a glossary (names, jargon), standing preferences, and example outputs you liked. It's used in every meeting.
- **Three ways to summarize:** Claude through the API (best quality, in the app, pay-per-use), Ollama (free, fully offline), or **Copy for AI** (no key: copies everything to paste into Claude.ai, ChatGPT or any AI chat, then copies follow-up requests for the same chat).
- **Everything is saved** per meeting: the audio, `transcript.txt`, and `notes.md` with every response. Reopen any meeting later and keep asking.

## Requirements

- **Windows 10/11.** Mac and Linux work too, but record the microphone only (no system audio); you can still upload a recording of the call.
- **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/). During install, tick **"Add python.exe to PATH"**.
- About 2 GB of free disk space (more with GPU support).
- A summarizer, one of:
  - **Claude API:** a key from [console.anthropic.com](https://console.anthropic.com/). This is pay-as-you-go, billed separately from a Claude Pro/Max subscription.
  - **Claude.ai or another AI chat:** no setup. Use the **Copy for AI** button with your existing account (free or paid).
  - **Ollama:** free, from [ollama.com](https://ollama.com/). Install and start it, then in the app's Settings click **Download** next to the recommended model: `gemma4:12b` (7.7 GB) on computers with an 8 GB+ GPU or 16 GB+ RAM, otherwise `qwen3.5:4b` (3.3 GB).

## Install and run

1. Download this repository: green **Code** button, then **Download ZIP**, and unzip it anywhere. (Or `git clone` it.)
2. **Windows:** double-click `run.bat`.
   **Mac/Linux:** run `./run.sh` in a terminal.
3. The first run takes a few minutes to install. If you have an NVIDIA GPU, it also installs GPU support (about 1 GB). Then your browser opens to the app at http://127.0.0.1:8806.
4. Click **Settings**, choose Claude or Ollama, and paste your API key if you use Claude.

To quit, close the black console window. Run `run.bat` again next time; later launches start in seconds.

The first transcription downloads the Whisper model (about 0.5 GB on CPU, 1.6 GB on GPU), so it can take several minutes. After that it's fast.

## How to use it

1. **Start a transcript.** Click **Record** before the meeting starts, or **Upload audio** (most audio and video files work), or paste a transcript (for example from Teams or Zoom) into the Transcript box.
2. **Add context** (optional but worth it): who's there, their roles, project names, and jargon. Whisper uses it to spell names right, and the summarizer uses it to fix misheard words.
3. **Ask** for what you want, or click a preset. You can ask while still recording.
4. **Refine** with follow-ups. **Copy** on a response copies just the message when the response is a single code block (the Discord and email presets).

**Without an API key:** type your request (or click a preset), then click **Copy for AI**. It copies the instructions, context, transcript and request as one message: paste it (Ctrl+V) into a new chat in Claude.ai, ChatGPT or any AI. The button then becomes **Copy follow-up**: click a preset or type a request and it copies just that (plus any transcript recorded since) to paste into the same chat. **New AI chat** copies everything again for a fresh chat; **Reset chat** also starts over.

## Teaching it (Memory)

The app gets better the more you use it:

- **Glossary:** in **Memory**, list names, roles, and terms, one per line (`Josh: lab manager`). The summarizer uses them to fix misheard words, and Whisper uses them to spell names right.
- **Remember:** click it under a request you'll always want ("no emojis", "keep updates short"). It's added to your standing preferences.
- **Save as example:** click it on a response you liked. Future responses to similar requests copy its structure and tone. The newest 4 examples are used.

Memory is stored in `memory.json` next to the app. Changes apply from the next new chat (**New meeting** or **Reset chat**).

## What to expect from Ollama

Tested on a fictional 7.5-minute meeting (`eval/`) with `gemma4:12b` on a laptop RTX 4060 (8 GB):

- **Quality:** close to Claude. It got every action item, owner, due date, and approval item, used the glossary spellings, and left out the off-the-record remark and small talk. It missed a few softer points that Claude caught (the "execution first" theme in the notes, one parked question) and sometimes misfiled a detail (a tip listed as a decision).
- **Speed:** about 45–90 seconds per answer on that short meeting. On a real 57-minute meeting, about 1–2 minutes for the first answer, and follow-ups start within seconds. The 7.7 GB model doesn't fit in an 8 GB GPU, so part of it runs on the CPU from system RAM.
- **`qwen3.5:4b` is faster but not recommended for notes.** On the same meeting it made up owners, due dates, and decisions nobody made. gemma4:12b stuck to what was said.
- Before an Ollama request, the app unloads Whisper from the GPU to make room (it reloads on the next transcription). Ollama frees the model's memory 2 minutes after the last request.

To check another model, run `.venv\Scripts\python eval\run_eval.py <model>` and compare its output with `eval/answer_key.md`.

## How Ollama handles long meetings

Meetings up to about 90 minutes are read in one pass (on the 57-minute test meeting this was faster and more complete than two passes). Longer transcripts don't fit the model's memory, so the app works in two passes. First it reads the transcript in sections and pulls out the facts (topics, decisions, action items, open questions). Then it answers your request from those notes. You'll see "Reading the transcript: section 2 of 6…" the first time. Section notes are saved with the meeting, so follow-ups are fast, and only new sections are read as the meeting continues.

Use **New meeting** for the next one and **Past meetings** to reopen old ones. The `?` button in the app has a short version of these steps.

## Cost and privacy

- **Audio and transcription stay on your computer.**
- **With the Claude API**, the transcript text is sent to Anthropic for each request. A one-hour meeting is roughly 12,000 tokens, so a request costs about 5–10 cents with the default model (Claude Opus 5.5). Follow-ups cost less because the transcript is cached. You can switch to a cheaper model in Settings (e.g. `claude-sonnet-5-5` or `claude-haiku-5-5`). This is billed separately from any Claude subscription.
- **With Copy for AI**, there's no API cost; it counts toward your normal usage limits in whichever AI chat you paste into, and the transcript goes to that company the same way as any chat.
- **With Ollama**, nothing leaves your computer and it's free, but summaries are noticeably weaker than Claude's.
- Your API key is stored in `settings.json` next to the app, on your computer only. The `.gitignore` keeps it, `memory.json`, and your `meetings/` folder out of git.

**Get consent before recording anyone.** Recording laws vary by place; many require everyone's permission.

## Troubleshooting

| Problem | Fix |
|---|---|
| System audio isn't recorded | Windows records silence when your volume is muted or at 0. Unmute it. The level bars under Record show what's coming in. |
| "Python 3.10 or newer is required" | Install Python from python.org with "Add python.exe to PATH" ticked, then run `run.bat` again. |
| Setup failed partway | Check your internet connection and run `run.bat` again. If it keeps failing, delete the `.venv` folder and retry. |
| Transcription is slow | On CPU, pick a smaller Whisper model in Settings (`base` or `tiny`). With an NVIDIA GPU, make sure Settings > Device shows CUDA (re-run setup by deleting `.venv`). |
| Ollama says it can't connect | Start Ollama, then download a model from Settings (or run `ollama pull gemma4:12b`) and pick it. |
| Ollama is very slow | The model is too big for your GPU. A smaller one like `qwen3.5:4b` is faster but less accurate; for long meetings, Copy for AI is faster and better. |
| Names and terms come out wrong | Put them in Meeting context. |
| The transcript repeats one word over and over | Whisper sometimes loops on silence or noise. The app now drops repeated and low-confidence lines (and stray "Thank you." filler) and stops feeding them back to Whisper. Check the mic level if it keeps happening. |
| Notes name the wrong person | The transcript has no speaker names, so the summarizer only names someone when the words show who it was, and writes "[owner unclear]" otherwise. Say names out loud ("Ben, can you order the motors?"). On calls with system audio on, lines are labeled **Me** (your mic) or **Others** (the call). |
| Port 8806 is in use | Change `PORT` in `config.py`. |

## How it works (for tinkering)

| File | What it does |
|---|---|
| `app.py` | Flask server: meeting state, background transcription queue, API routes |
| `recorder.py` | Captures mic (sounddevice) and system audio (WASAPI loopback via PyAudioWPatch), mixes to 16 kHz mono, writes the WAV, and cuts ~30 s chunks at quiet points |
| `transcriber.py` | faster-whisper; tries the GPU, then falls back to CPU |
| `summarizer.py` | The system prompt, Claude (Anthropic SDK, streaming) and Ollama calls (two-pass for long transcripts), model recommendation |
| `memory.py` | Glossary, preferences, and saved examples (`memory.json`) and how they're added to the prompt |
| `eval/` | A fictional test meeting, its answer key, and a script to check Ollama models against it |
| `config.py` | Settings defaults and storage |
| `static/` | The page: `index.html`, `app.js` (including the preset prompts), `style.css` |

To change what the summarizer does by default, edit `SYSTEM_PROMPT` in `summarizer.py`. To change or add presets, edit `PRESETS` at the top of `static/app.js`.

## License

MIT. See [LICENSE](LICENSE).
