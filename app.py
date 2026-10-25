"""Meeting Notes: a local web app that records or imports a meeting, transcribes
it on this computer, and writes notes in whatever form you ask for.

Start it with run.bat (Windows) or run.sh (Mac/Linux), or `python app.py`.
"""
import json
import queue
import re
import socket
import sys
import threading
import webbrowser
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import config
import memory
import summarizer
from recorder import ACTIVE_LEVEL, RATE, Recorder, system_audio_supported
from transcriber import Transcriber

app = Flask(__name__, static_folder="static", static_url_path="/static")
transcriber = Transcriber()
lock = threading.RLock()
jobs = queue.Queue()


SAVED_FIELDS = ("title", "context", "transcript", "history", "sent_transcript", "sent_context",
                "system", "extractions", "language", "audio_seconds", "copied_transcript", "copied_context")
# Marks a progress note inside the streamed reply; the page shows it and drops it from the text.
STATUS_START, STATUS_END = "\x01", "\x02"
# "[12:34] " plus an optional "Me: "/"Others: " speaker label.
LINE_PREFIX = re.compile(r"^\[[\d:]+\] (?:(?:Me|Others): )?", re.M)


class Meeting:
    """One meeting's transcript and chat, saved in meetings/<id>/."""

    def __init__(self, meeting_id):
        self.id = meeting_id
        self.folder = config.MEETINGS_DIR / meeting_id
        self.title = ""
        self.context = ""
        self.transcript = ""
        self.history = []          # [{"role", "text", "display", "raw"}]
        self.sent_transcript = ""  # what the model has already seen
        self.sent_context = ""
        self.system = ""           # Claude instructions + memory, fixed when the chat starts
        self.extractions = {}      # Ollama pass-1 notes per transcript section
        self.copied_transcript = None  # what "Copy for AI" already put in an external chat
        self.copied_context = None
        self.language = None
        self.audio_seconds = 0.0
        self.recorder = None
        self.pending = 0
        self.error = ""

    @classmethod
    def new(cls):
        return cls(datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    @classmethod
    def load(cls, meeting_id):
        m = cls(meeting_id)
        data = json.loads((m.folder / "meeting.json").read_text(encoding="utf-8"))
        for k in SAVED_FIELDS:
            setattr(m, k, data.get(k, getattr(m, k)))
        return m

    def is_empty(self):
        return not (self.title or self.context or self.transcript or self.history)

    def save(self):
        if self.is_empty() and not self.folder.exists():
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in SAVED_FIELDS}
        (self.folder / "meeting.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        (self.folder / "transcript.txt").write_text(self.transcript, encoding="utf-8")
        notes = [f"# {self.title or self.id}\n"]
        for user, reply in zip(self.history[::2], self.history[1::2]):
            notes.append(f"## {user['display']}\n\n{reply['text']}\n")
        (self.folder / "notes.md").write_text("\n".join(notes), encoding="utf-8")

    def whisper_prompt(self):
        # Whisper only reads the last ~220 tokens of a prompt, so keep it short:
        # glossary and context (names, jargon), then the end of the transcript so far.
        recent = LINE_PREFIX.sub("", self.transcript[-400:]).replace("\n", " ")
        words = recent.lower().split()
        if len(words) >= 10 and len(set(words)) < len(words) / 3:
            recent = ""  # repetitive tail: passing it on would teach Whisper to keep repeating
        terms = memory.whisper_terms(memory.load())
        return f"{terms} {self.context.strip()[:300]} {recent}".strip()

    def add_segment(self, start, text, speaker=None):
        last = self.transcript.rsplit("\n", 1)[-1]
        if LINE_PREFIX.sub("", last).lower() == text.lower():
            return  # the same line again is almost always Whisper repeating itself
        label = f"{speaker}: " if speaker else ""
        line = f"[{int(start // 60):02d}:{int(start % 60):02d}] {label}{text}"
        self.transcript += ("\n" if self.transcript else "") + line


current = Meeting.new()


def speaker(tracks, start, end):
    """On a call, system audio is the other people and the mic is you."""
    if not tracks:
        return None
    a, b = int(start * RATE), max(int(end * RATE), int(start * RATE) + 1)
    level = {name: float(np.sqrt(np.mean(t[a:b] ** 2))) if len(t[a:b]) else 0.0 for name, t in tracks.items()}
    if level.get("System audio", 0) > ACTIVE_LEVEL:
        return "Others"
    if level.get("Microphone", 0) > ACTIVE_LEVEL:
        return "Me"
    return None


prewarm = {"status": "", "meeting": None}


def start_prewarm(meeting):
    """Once a transcript is complete, have Ollama read it in the background."""
    settings = config.load()
    if settings["provider"] != "ollama" or len(meeting.transcript) > summarizer.SINGLE_PASS_CHARS:
        return
    with lock:
        if meeting is not current or meeting.recorder or meeting.pending or not meeting.transcript.strip():
            return
        transcript, context = meeting.transcript, meeting.context
    system = summarizer.system_prompt(memory.prompt_section(memory.load()))

    def run():
        prewarm.update(status="reading", meeting=meeting.id)
        try:
            transcriber.unload()  # Whisper is done; give Ollama the GPU
            summarizer.ollama_prewarm(settings, system, transcript, context)
            prewarm["status"] = "ready"
        except summarizer.SummarizerError:
            prewarm["status"] = ""  # Ollama not running: the first request reports it

    threading.Thread(target=run, daemon=True).start()


def worker():
    """Transcribes queued audio one job at a time so the transcript stays in order."""
    while True:
        meeting, source, offset, is_file, tracks = jobs.get()
        settings = config.load()
        language = None if settings["language"] == "auto" else settings["language"]
        with lock:
            prompt = meeting.whisper_prompt()
        try:
            def on_segment(start, end, text):
                with lock:
                    meeting.add_segment(start, text, speaker(tracks, start - offset, end - offset))
            lang, duration = transcriber.transcribe(
                source, settings, offset, language or meeting.language, on_segment, prompt)
            with lock:
                meeting.language = meeting.language or lang
                if is_file:
                    meeting.audio_seconds += duration
        except Exception as e:
            with lock:
                meeting.error = f"Transcription failed: {e}"
        finally:
            with lock:
                meeting.pending -= 1
                meeting.save()
                done = not (meeting.pending or meeting.recorder or meeting.error)
            if done:
                start_prewarm(meeting)


def error(message, code=400):
    return jsonify({"error": message}), code


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/state")
def state():
    with lock:
        m = current
        return jsonify({
            "id": m.id,
            "title": m.title,
            "context": m.context,
            "transcript": m.transcript,
            "history": [{"role": t["role"], "text": t["display"]} for t in m.history],
            "recording": m.recorder is not None,
            "record_seconds": m.recorder.seconds if m.recorder else 0,
            "levels": m.recorder.levels() if m.recorder else {},
            "pending": m.pending,
            "copied": m.copied_transcript is not None,
            "prewarm": prewarm["status"] if prewarm["meeting"] == m.id else "",
            "error": m.error,
            "whisper": transcriber.info,
            "system_audio_supported": system_audio_supported(),
            "settings": config.public(config.load()),
        })


@app.post("/api/meeting")
def update_meeting():
    data = request.get_json()
    with lock:
        m = current
        if "transcript" in data and (m.recorder or m.pending):
            return error("Wait for recording and transcription to finish before editing.")
        for k in ("title", "context", "transcript"):
            if k in data:
                setattr(m, k, data[k])
        m.error = ""
        m.save()
    return jsonify(ok=True)


@app.post("/api/new")
def new_meeting():
    global current
    with lock:
        if current.recorder:
            return error("Stop recording first.")
        current.save()
        current = Meeting.new()
    return jsonify(ok=True)


@app.get("/api/meetings")
def list_meetings():
    out = []
    if config.MEETINGS_DIR.exists():
        for folder in sorted(config.MEETINGS_DIR.iterdir(), reverse=True):
            f = folder / "meeting.json"
            if f.exists():
                title = json.loads(f.read_text(encoding="utf-8")).get("title", "")
                out.append({"id": folder.name, "title": title})
    return jsonify(out)


@app.post("/api/open")
def open_meeting():
    global current
    with lock:
        if current.recorder or current.pending:
            return error("Wait for recording and transcription to finish first.")
        current.save()
        try:
            current = Meeting.load(secure_filename(request.get_json()["id"]))
        except (OSError, ValueError, KeyError):
            return error("Couldn't open that meeting.")
    start_prewarm(current)
    return jsonify(ok=True)


@app.post("/api/reset-chat")
def reset_chat():
    with lock:
        current.history, current.sent_transcript, current.sent_context, current.system = [], "", "", ""
        current.copied_transcript = current.copied_context = None
        current.save()
    return jsonify(ok=True)


@app.post("/api/record/start")
def record_start():
    with lock:
        m = current
        if m.recorder:
            return error("Already recording.")
        settings = config.load()
        m.folder.mkdir(parents=True, exist_ok=True)
        n = len(list(m.folder.glob("recording-*.wav"))) + 1
        base = m.audio_seconds

        def on_chunk(audio, offset, tracks):
            with lock:
                m.pending += 1
            jobs.put((m, audio, base + offset, False, tracks))

        rec = Recorder(m.folder / f"recording-{n}.wav", on_chunk,
                       use_mic=settings["record_microphone"], use_system=settings["record_system_audio"])
        try:
            rec.start()
        except Exception as e:
            return error(f"Couldn't start recording: {e}")
        m.recorder = rec
        m.error = ""
        m.save()
    return jsonify(ok=True)


@app.post("/api/record/stop")
def record_stop():
    with lock:
        m = current
        rec, m.recorder = m.recorder, None
    if rec:
        rec.stop()  # outside the lock: stopping flushes a final chunk, which takes the lock
        with lock:
            m.audio_seconds += rec.seconds
            m.save()
    return jsonify(ok=True)


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return error("No file selected.")
    with lock:
        m = current
        if m.recorder:
            return error("Stop recording first.")
        m.folder.mkdir(parents=True, exist_ok=True)
        path = m.folder / secure_filename(f.filename)
        f.save(path)
        m.pending += 1
        m.error = ""
        jobs.put((m, str(path), m.audio_seconds, True, None))
    return jsonify(ok=True)


@app.post("/api/ask")
def ask():
    request_text = (request.get_json().get("request") or "").strip()
    if not request_text:
        return error("Type what you want first.")
    settings = config.load()
    with lock:
        m = current
        transcript, context = m.transcript, m.context
        history = list(m.history)
        try:
            # Built for every provider, so the chat can switch to Claude later and stay consistent.
            message = summarizer.build_user_message(
                request_text, transcript, m.sent_transcript, context, m.sent_context)
        except summarizer.SummarizerError as e:
            return error(str(e))
        mem = memory.load()
        live_system = summarizer.system_prompt(memory.prompt_section(mem))
        # Claude: the instructions (with memory) are fixed for the whole chat, because
        # changing them mid-conversation would break its append-only history.
        system = m.system or live_system

    if settings["provider"] == "ollama":
        with lock:
            idle = not (m.recorder or m.pending)
        if idle:
            transcriber.unload()  # Whisper holds ~2 GB of GPU memory that Ollama can use
        source = summarizer.ollama_answer(settings, live_system, history, request_text,
                                          transcript, context, m.extractions, mem["glossary"])
    else:
        source = summarizer.claude_stream(settings, system, history, message)

    def generate():
        parts, raw = [], None
        try:
            for kind, value in source:
                if kind == "text":
                    parts.append(value)
                    yield value
                elif kind == "status":
                    yield f"{STATUS_START}{value}{STATUS_END}"
                else:
                    raw = value
        except summarizer.SummarizerError as e:
            yield f"\n\n[{e}]"
            return  # nothing is recorded, so the next request resends the same transcript
        except Exception as e:
            yield f"\n\n[Unexpected error: {e}]"
            return
        finally:
            with lock:
                m.save()  # keeps Ollama's section notes even if the final answer failed
        with lock:
            if m.history != history:
                return  # chat was reset or changed while this was running
            m.history += [
                {"role": "user", "text": message, "display": request_text, "raw": None},
                {"role": "assistant", "text": "".join(parts), "display": "".join(parts), "raw": raw},
            ]
            m.sent_transcript, m.sent_context = transcript, context
            if settings["provider"] != "ollama" and not m.system:
                m.system = system
            m.save()

    return Response(generate(), mimetype="text/plain; charset=utf-8",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/api/copy-prompt")
def copy_prompt():
    """First copy: everything, for a new external chat. Later copies: just the request
    plus any transcript added since, to paste into that same chat."""
    data = request.get_json()
    request_text = (data.get("request") or "").strip()
    with lock:
        m = current
        if data.get("restart"):
            m.copied_transcript = m.copied_context = None
        try:
            if m.copied_transcript is None:
                system = summarizer.system_prompt(memory.prompt_section(memory.load()))
                text = summarizer.copy_prompt(system, request_text, m.transcript, m.context)
            else:
                if not request_text:
                    return error("Type a request or click a preset first.")
                text = summarizer.build_user_message(
                    request_text, m.transcript, m.copied_transcript, m.context, m.copied_context)
        except summarizer.SummarizerError as e:
            return error(str(e))
        m.copied_transcript, m.copied_context = m.transcript, m.context
        m.save()
    return jsonify(text=text)


@app.get("/api/memory")
def get_memory():
    return jsonify(memory.load())


@app.post("/api/memory")
def save_memory():
    data = request.get_json()
    return jsonify(memory.update(data.get("glossary"), data.get("preferences")))


@app.post("/api/memory/preference")
def add_preference():
    text = (request.get_json().get("text") or "").strip()
    if not text:
        return error("Nothing to remember.")
    return jsonify(memory.add_preference(text))


@app.post("/api/memory/example")
def add_example():
    data = request.get_json()
    return jsonify(memory.add_example(data["request"], data["output"]))


@app.post("/api/memory/example/delete")
def delete_example():
    return jsonify(memory.delete_example(request.get_json()["id"]))


@app.post("/api/settings")
def save_settings():
    data = request.get_json()
    if data.get("clear_api_key"):
        data["anthropic_api_key"] = ""
    elif not data.get("anthropic_api_key"):
        data.pop("anthropic_api_key", None)  # blank field = keep the saved key
    return jsonify(config.public(config.save(data)))


@app.get("/api/ollama")
def ollama_info():
    models = summarizer.ollama_models(config.load())
    return jsonify(running=models is not None, models=models or [],
                   recommended=summarizer.recommend_ollama_model(), pull=pull)


pull = {"model": "", "status": "", "running": False, "error": ""}


@app.post("/api/ollama/pull")
def ollama_pull():
    model = request.get_json().get("model", "")
    if pull["running"]:
        return error("A download is already running.")
    pull.update(model=model, status="Starting download…", running=True, error="")

    def run():
        try:
            summarizer.ollama_pull(config.load(), model, lambda s: pull.update(status=s))
            config.save({"ollama_model": model})
            pull["status"] = "Done. Selected it as your Ollama model."
        except summarizer.SummarizerError as e:
            pull["error"] = str(e)
        finally:
            pull["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)


def already_running():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", config.PORT)) == 0


if __name__ == "__main__":
    url = f"http://127.0.0.1:{config.PORT}"
    if already_running():
        print(f"Meeting Notes is already running. Opening {url}")
        webbrowser.open(url)
        sys.exit()
    threading.Thread(target=worker, daemon=True).start()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    print(f"Meeting Notes is running at {url}\nClose this window to quit.")
    app.run(host="127.0.0.1", port=config.PORT, threaded=True)
