"""Meeting Notes: a local web app that records or imports a meeting, transcribes
it on this computer, and writes notes in whatever form you ask for.

Start it with run.bat (Windows) or run.sh (Mac/Linux), or `python app.py`.
"""
import json
import logging
import queue
import re
import socket
import sys
import threading
import time
import webbrowser
import zlib
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import config
import memory
import speakers
import summarizer
from recorder import ACTIVE_LEVEL, RATE, Recorder, system_audio_supported
from transcriber import Transcriber, decode_audio

app = Flask(__name__, static_folder="static", static_url_path="/static")
transcriber = Transcriber()
tagger = speakers.VoiceTagger()
log = logging.getLogger("meetingnotes")
lock = threading.RLock()
jobs = queue.Queue()


SAVED_FIELDS = ("title", "context", "transcript", "history", "sent_transcript", "sent_context",
                "system", "extractions", "language", "audio_seconds", "copied_transcript", "copied_context")
# Marks a progress note inside the streamed reply; the page shows it and drops it from the text.
STATUS_START, STATUS_END = "\x01", "\x02"
# "[12:34] " plus an optional speaker label ("Speaker 2: ", "Avanish: ", "Me: ").
LINE_PREFIX = re.compile(r"^\[[\d:]+\] (?:[A-Z][\w .'&()-]{0,39}: )?", re.M)
REGROUP_EVERY = 20  # new voiced lines between speaker regroups while recording (~0.1 s each)
SPEAKER_NAME = re.compile(r"^[^\s:\n\[\]][^:\n\[\]]{0,39}$")
_LINE = re.compile(r"^(\[[\d:]+\] )(?:([A-Z][\w .'&()-]{0,39}): )?(.*)$", re.S)


def split_line(line):
    """'[12:34] Avanish: text' -> ('[12:34] ', 'Avanish', 'text'); None if not a transcript line."""
    m = _LINE.match(line)
    return (m.group(1), m.group(2), m.group(3)) if m else None


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
        self.pending_copy = None       # (transcript, context, restart) until the clipboard write is confirmed
        self.prints = []           # (line without its label, voiceprint) per labeled line; voiceprints.npz
        self.regrouped_at = 0      # len(prints) at the last speaker regroup
        self.language = None
        self.audio_seconds = 0.0
        self.recorder = None
        self.stopping = False      # Stop clicked, final chunk not queued yet
        self.transcribed_to = 0.0  # live recording: audio seconds transcribed so far
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
        if (m.folder / "voiceprints.npz").exists():
            z = np.load(m.folder / "voiceprints.npz")
            m.prints = list(zip(z["lines"].tolist(), z["vectors"]))
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
        if self.prints:
            np.savez(self.folder / "voiceprints.npz", lines=[p[0] for p in self.prints],
                     vectors=np.array([p[1] for p in self.prints], np.float32))

    def labeled_prints(self):
        """Transcript lines that have a voiceprint: (raw lines, parsed lines, line indexes,
        voiceprints, current labels). Lines the user has since edited won't match."""
        raw = self.transcript.split("\n")
        parts = [split_line(l) for l in raw]
        where = {}
        for i, p in enumerate(parts):
            if p:
                where.setdefault(p[0] + p[2], i)
        found = [(where[key], v) for key, v in self.prints if key in where]
        idx = [i for i, _ in found]
        X = np.array([v for _, v in found]) if found else np.zeros((0, 256))
        return raw, parts, idx, X, [parts[i][1] for i in idx]

    def regroup_speakers(self):
        """Regroup the whole meeting's voiceprints and relabel lines (see speakers.regroup)."""
        raw, parts, idx, X, labels = self.labeled_prints()
        self.regrouped_at = len(self.prints)
        if len(idx) < speakers.WARMUP_LINES:
            return
        for i, label in zip(idx, speakers.regroup(X, labels, speakers.load_known())):
            tag, _, text = parts[i]
            raw[i] = tag + (f"{label}: " if label else "") + text
        self.transcript = "\n".join(raw)

    def whisper_prompt(self):
        # Whisper only reads the last ~220 tokens of a prompt, so keep it short:
        # glossary and context (names, jargon), then the end of the transcript so far.
        recent = LINE_PREFIX.sub("", self.transcript[-400:]).replace("\n", " ")
        words = recent.lower().split()
        if len(words) >= 10 and len(set(words)) < len(words) / 3:
            recent = ""  # repetitive tail: passing it on would teach Whisper to keep repeating
        terms = memory.whisper_terms(memory.load())
        return f"{terms} {self.context.strip()[:300]} {recent}".strip()

    def add_segment(self, start, text, speaker=None, voiceprint=None):
        last = self.transcript.rsplit("\n", 1)[-1]
        if LINE_PREFIX.sub("", last).lower() == text.lower():
            return  # the same line again is almost always Whisper repeating itself
        tag = f"[{int(start // 60):02d}:{int(start % 60):02d}] "
        label = f"{speaker}: " if speaker else ""
        self.transcript += ("\n" if self.transcript else "") + tag + label + text
        if voiceprint is not None:
            self.prints.append((tag + text, voiceprint))


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


# What Ollama has read in advance: status "", "reading" or "ready"; key = which meeting and text.
prewarm = {"status": "", "meeting": None, "key": None, "at": 0.0}
prewarm_lock = threading.Lock()


def start_prewarm(meeting):
    """Have Ollama read a complete transcript in the background, so answers start in seconds.
    Runs after a recording or upload finishes and when the user clicks into the request box;
    skipped if that exact text is already being read or was read recently."""
    settings = config.load()
    if settings["provider"] != "ollama":
        return
    with lock:
        if (meeting is not current or meeting.recorder or meeting.pending or meeting.stopping
                or not meeting.transcript.strip()):
            return
        transcript, context = meeting.transcript, meeting.context
    system = summarizer.system_prompt(memory.prompt_section(memory.load()))
    if not summarizer.fits_single_pass(system, transcript, context):
        return
    key = (meeting.id, zlib.crc32(f"{settings['ollama_model']}\n{system}\n{context}\n{transcript}".encode()))
    with prewarm_lock:
        fresh = time.time() - prewarm["at"] < 14 * 60  # PREWARM_KEEP_ALIVE is 15 min
        if prewarm["key"] == key and (prewarm["status"] == "reading" or (prewarm["status"] == "ready" and fresh)):
            return
        prewarm.update(status="reading", meeting=meeting.id, key=key)

    def run():
        try:
            transcriber.unload()  # Whisper is done; give Ollama the GPU
            summarizer.ollama_prewarm(settings, system, transcript, context)
            status = "ready"
        except summarizer.SummarizerError:
            status = ""  # Ollama not running: the first request reports it
        with prewarm_lock:
            if prewarm["key"] == key:
                prewarm.update(status=status, at=time.time())

    threading.Thread(target=run, daemon=True).start()


def transcribe_job(meeting, source, offset, is_file, tracks):
    settings = config.load()
    language = None if settings["language"] == "auto" else settings["language"]
    with lock:
        prompt = meeting.whisper_prompt()
        # Live chunks waiting behind this one: decode faster until caught up.
        fast = not is_file and meeting.pending > 1
    # Transcription gets the GPU first. Measured with Whisper loaded: gemma4:e4b leaves 2.15 GB
    # free (keep it), gemma4:12b leaves 0.01 GB (unload it).
    freed = summarizer.ollama_free_gpu(settings, 0.5 if transcriber.on_gpu else summarizer.WHISPER_GPU_GB)
    if freed:
        log.info("Unloaded %s from Ollama to give Whisper the GPU", ", ".join(freed))
        with prewarm_lock:
            prewarm.update(status="", key=None)
    by_voice = settings["speaker_labels"]
    if by_voice and is_file:
        source = decode_audio(source, sampling_rate=speakers.RATE)  # need the samples for voiceprints

    def on_segment(start, end, text):
        label = v = None
        if by_voice:
            a, b = int((start - offset) * speakers.RATE), int((end - offset) * speakers.RATE)
            v = tagger.voiceprint(source[a:b])  # outside the lock: ~70 ms
            if v is not None:
                with lock:
                    _, _, _, X, labels = meeting.labeled_prints()
                label = speakers.assign_live(X, labels, v)
        else:
            label = speaker(tracks, start - offset, end - offset)
        with lock:
            meeting.add_segment(start, text, label, v)

    lang, duration = transcriber.transcribe(
        source, settings, offset, language or meeting.language, on_segment, prompt, fast)
    with lock:
        meeting.language = meeting.language or lang
        if is_file:
            meeting.audio_seconds += duration
        else:
            meeting.transcribed_to = offset + duration
            if len(meeting.prints) - meeting.regrouped_at >= REGROUP_EVERY:
                meeting.regroup_speakers()  # mid-meeting: fix live guesses, apply names


def finish_if_done(meeting):
    """Once a recording or upload is fully transcribed: final speaker regroup, save, pre-warm.
    Call with the lock held; returns True if the caller should start the pre-warm."""
    done = not (meeting.pending or meeting.recorder or meeting.stopping or meeting.error)
    if done and meeting.prints:
        try:
            meeting.regroup_speakers()
        except Exception:
            log.exception("Regrouping speakers failed; keeping the live labels")
    try:
        meeting.save()
    except OSError as e:  # e.g. a sync client or antivirus holding the file
        meeting.error = f"Couldn't save the meeting: {e}"
    return done


def worker():
    """Transcribes queued audio one job at a time so the transcript stays in order.
    Nothing may escape this loop: a dead worker would leave every later chunk queued forever."""
    while True:
        job = jobs.get()
        meeting = job[0]
        try:
            transcribe_job(*job)
        except Exception as e:
            log.exception("Transcription failed")
            with lock:
                meeting.error = f"Transcription failed: {e}"
        finally:
            with lock:
                meeting.pending -= 1
                done = finish_if_done(meeting)
            if done:
                start_prewarm(meeting)


def error(message, code=400):
    return jsonify({"error": message}), code


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/state")
def state():
    """Polled every second. The transcript (~55 KB for an hour) and the speaker list are
    sent only when they changed since the page's last copy (?t=<transcript_key>)."""
    with lock:
        m = current
        key = f"{m.id}:{zlib.crc32(m.transcript.encode())}"
        changed = request.args.get("t") != key
        return jsonify({
            "id": m.id,
            "title": m.title,
            "context": m.context,
            "transcript_key": key,
            **({"transcript": m.transcript, "speakers": speaker_list(m)} if changed else {}),
            "history": [{"role": t["role"], "text": t["display"]} for t in m.history],
            "recording": m.recorder is not None,
            "record_seconds": m.recorder.seconds if m.recorder else 0,
            # How far the live transcript trails the recording (chunks are ~30 s, so ~30-60 s is normal).
            "behind_seconds": max(0, m.audio_seconds + m.recorder.seconds - m.transcribed_to) if m.recorder else 0,
            "levels": m.recorder.levels() if m.recorder else {},
            "pending": m.pending,
            "copied": m.copied_transcript is not None,
            "voice_error": tagger.error,
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
        if "transcript" in data and (m.recorder or m.pending or m.stopping):
            return error("Wait for recording and transcription to finish before editing.")
        for k in ("title", "context", "transcript"):
            if k in data:
                if not isinstance(data[k], str):  # anything else would break every later poll
                    return error(f"{k} must be text.")
                setattr(m, k, data[k])
        m.error = ""
        m.save()
    return jsonify(ok=True)


def speaker_list(m):
    """Speakers in this meeting's transcript, most lines first, with a sample line to recognize them by."""
    if not m.prints:
        return []
    known = speakers.load_known()
    by_label = {}
    for line in m.transcript.splitlines():
        p = split_line(line)
        if p and p[1] and p[1] not in ("Me", "Others"):
            by_label.setdefault(p[1], []).append(line)
    return sorted(({"label": label, "lines": len(lines), "sample": max(lines[:40], key=len),
                    "named": label in known} for label, lines in by_label.items()), key=lambda s: -s["lines"])


@app.post("/api/speaker")
def rename_speaker():
    """Name a speaker ("Speaker 2" -> "Avanish") in this meeting and remember their voice.
    Giving two labels the same name merges them (one person split into two)."""
    data = request.get_json()
    old, new = data.get("from", ""), " ".join((data.get("to") or "").split())
    if not SPEAKER_NAME.match(new):
        return error("Use a short name without colons or brackets.")
    with lock:
        m = current
        if new == old:
            return jsonify(ok=True)
        _, _, _, X, labels = m.labeled_prints()
        if old not in labels:
            return error("That speaker isn't in this meeting.")
        m.transcript = re.sub(rf"^(\[[\d:]+\] ){re.escape(old)}: ", lambda x: f"{x.group(1)}{new}: ",
                              m.transcript, flags=re.M)
        if not new.startswith("Speaker "):
            speakers.remember(new, speakers.voice_of(X, labels, old))
        m.save()
    return jsonify(ok=True)


@app.get("/api/voices")
def list_voices():
    return jsonify(sorted(speakers.load_known()))


@app.post("/api/voices/forget")
def forget_voice():
    speakers.forget(request.get_json().get("name", ""))
    return jsonify(sorted(speakers.load_known()))


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
    return jsonify(ok=True)


@app.post("/api/prewarm")
def request_prewarm():
    """The page calls this when the user clicks into the request box or picks a preset:
    likely about to ask, so have Ollama start reading now (no-op if already read)."""
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
        base = m.transcribed_to = m.audio_seconds

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
        m.stopping = bool(rec)  # until the final chunk is queued, the meeting isn't done
    if rec:
        rec.stop()  # outside the lock: stopping flushes a final chunk, which takes the lock
        with lock:
            m.audio_seconds += rec.seconds
            m.stopping = False
            done = finish_if_done(m)
        if done:
            start_prewarm(m)
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
            source.close()  # Stop in the page drops the connection: stop Ollama/Claude right away
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
        try:
            if m.copied_transcript is None or data.get("restart"):
                system = summarizer.system_prompt(memory.prompt_section(memory.load()))
                text = summarizer.copy_prompt(system, request_text, m.transcript, m.context)
            else:
                if not request_text:
                    return error("Type a request or click a preset first.")
                text = summarizer.build_user_message(
                    request_text, m.transcript, m.copied_transcript, m.context, m.copied_context)
        except summarizer.SummarizerError as e:
            return error(str(e))
        # Counts as copied only once the page confirms the clipboard write worked;
        # otherwise the next click would send a follow-up without the transcript.
        m.pending_copy = (m.transcript, m.context, data.get("restart"))
    return jsonify(text=text)


@app.post("/api/copy-prompt/confirm")
def confirm_copy():
    with lock:
        m = current
        if m.pending_copy:
            m.copied_transcript, m.copied_context, _ = m.pending_copy
            m.pending_copy = None
            m.save()
    return jsonify(ok=True)


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


@app.post("/api/memory/preset")
def add_preset():
    data = request.get_json()
    label = " ".join(str(data.get("label") or "").split())[:40]
    prompt = " ".join(str(data.get("prompt") or "").split())  # one line: presets are "- " lines in the request box
    if not label or not prompt:
        return error("A quick prompt needs a name and what it asks for.")
    return jsonify(memory.add_preset(label, prompt))


@app.post("/api/memory/preset/delete")
def delete_preset():
    return jsonify(memory.delete_preset(request.get_json()["id"]))


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


def setup_logging():
    # logs/meetingnotes.log (gitignored): Whisper device and per-chunk timing, for diagnosing slowdowns.
    folder = config.APP_DIR / "logs"
    folder.mkdir(exist_ok=True)
    handler = RotatingFileHandler(folder / "meetingnotes.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log = logging.getLogger("meetingnotes")
    log.setLevel(logging.INFO)
    log.addHandler(handler)


if __name__ == "__main__":
    setup_logging()
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
