"""Local speech-to-text with faster-whisper. Audio never leaves the computer."""
import gc
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("meetingnotes")


def _add_cuda_dlls():
    # requirements-gpu.txt installs NVIDIA's CUDA libraries as pip packages; on
    # Windows their DLL folders must be added to the search path before loading.
    if sys.platform != "win32":
        return
    for p in sys.path:
        nvidia = Path(p) / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


_add_cuda_dlls()
from faster_whisper import WhisperModel, decode_audio  # noqa: E402,F401  (must come after the DLL setup)

DEFAULT_MODEL = {"cuda": "large-v3-turbo", "cpu": "small"}

# A word or short phrase said 3+ times in a row ("Adioshi. Adioshi. Adioshi.").
_REPEAT = re.compile(r"\b((?:\S+\s*){1,4}?)(?:\1){2,}", re.I)


def collapse_repeats(text):
    """Whisper sometimes loops on one word over silence or noise. Keep one copy."""
    prev = None
    while prev != text:
        prev, text = text, _REPEAT.sub(r"\1", text)
    return text.strip()


# Sign-offs Whisper invents at the end of a chunk or over a long pause (from YouTube-style training data).
_FILLER = re.compile(r"^(thank you( (so|very) much)?|thanks( for watching)?|bye)[.!]?$", re.I)


def _hallucinated(seg, text, duration):
    # Whisper's own warning signs: very repetitive text, or likely silence it filled in anyway.
    if seg.compression_ratio > 2.4 or (seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0):
        return True
    # A lone "Thank you." in the last second of a chunk, or stretched over 8+ s, is filler.
    return bool(_FILLER.match(text)) and (seg.end > duration - 1.5 or seg.end - seg.start > 8)


GPU_RETRY_SECONDS = 120   # in auto mode, how often to try the GPU again after falling back to CPU
# On CPU, leave half the cores free so the computer stays usable while it transcribes.
CPU_THREADS = max(2, (os.cpu_count() or 4) // 2)


class Transcriber:
    def __init__(self):
        self._model = None
        self._key = None
        self._lock = threading.Lock()
        self.device = None
        self._fell_back_at = None  # time auto mode last fell back to CPU
        self.info = "not loaded yet (loads on first use; the first run downloads the model)"

    def _load(self, settings):
        key = (settings["whisper_model"], settings["whisper_device"])
        if self._model is not None and self._key == key:
            # A failed GPU (e.g. right after the laptop wakes from sleep) shouldn't
            # leave the whole meeting on the much slower CPU: try it again now and then.
            if not (self._fell_back_at and time.time() - self._fell_back_at > GPU_RETRY_SECONDS):
                return
        device = settings["whisper_device"]
        attempts = [d for d in ("cuda", "cpu") if device in ("auto", d)]
        errors = []
        for dev in attempts:
            name = DEFAULT_MODEL[dev] if settings["whisper_model"] == "auto" else settings["whisper_model"]
            if self._model is not None and self._key == key and dev == self.device:
                self._fell_back_at = time.time()  # GPU retry failed; keep the loaded CPU model
                self.info = f"{name} on CPU (GPU unavailable, so it's slower)"
                return
            self.info = f"loading {name} on {dev.upper()} (the first time, this downloads it and can take several minutes)"
            try:
                # int8_float16 on the GPU: about 1 GB less memory than float16, leaving room for Ollama.
                model = WhisperModel(name, device=dev, compute_type="int8_float16" if dev == "cuda" else "int8",
                                     cpu_threads=CPU_THREADS)
                # A short warm-up surfaces missing CUDA libraries now instead of mid-meeting.
                list(model.transcribe(np.zeros(16000, np.float32))[0])
            except Exception as e:
                errors.append(f"{dev}: {e}")
                log.warning("Whisper couldn't load %s on %s: %s", name, dev, e)
                continue
            self._model, self._key, self.device = model, key, dev
            self._fell_back_at = time.time() if (device == "auto" and dev == "cpu") else None
            self.info = f"{name} on {dev.upper()}" + (" (GPU unavailable, so it's slower)" if self._fell_back_at else "")
            log.info("Whisper loaded: %s", self.info)
            return
        self.info = "failed to load"
        raise RuntimeError("Could not load Whisper. " + " | ".join(errors))

    def transcribe(self, audio, settings, offset, language, on_segment, prompt=None, fast=False):
        """Transcribe a numpy array (16 kHz mono) or an audio/video file path.

        prompt: text Whisper treats as what came before (meeting context, the
        transcript so far). It helps with names, jargon, and punctuation.
        fast: greedy decoding (about 2x faster, slightly less accurate), used to catch up
        when live chunks are queuing.
        Calls on_segment(start_seconds, end_seconds, text) as segments finish.
        Returns (detected_language, duration_seconds).
        """
        with self._lock:
            self._load(settings)
            t0 = time.time()
            # condition_on_previous_text=False: otherwise one looped segment primes
            # the next one and the loop runs to the end of a long file.
            segments, info = self._model.transcribe(
                audio, language=language, vad_filter=True, beam_size=1 if fast else 5,
                initial_prompt=prompt or None, condition_on_previous_text=False)
            for seg in segments:
                text = collapse_repeats(seg.text)
                if text and not _hallucinated(seg, text, info.duration):
                    on_segment(offset + seg.start, offset + seg.end, text)
            log.info("transcribed %.0f s of audio at %.0f s in %.1f s (%s%s)", info.duration, offset,
                     time.time() - t0, self.info, ", catch-up mode" if fast else "")
            return info.language, info.duration

    def unload(self):
        """Free the GPU/RAM so a local summarizer model has room. Reloads on next use."""
        if not self._lock.acquire(blocking=False):
            return  # mid-transcription; leave it loaded
        try:
            if self._model is not None:
                self._model = self._key = None
                gc.collect()
                self.info = "unloaded to make room for Ollama (reloads on next use)"
        finally:
            self._lock.release()
