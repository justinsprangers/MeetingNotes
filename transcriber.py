"""Local speech-to-text with faster-whisper. Audio never leaves the computer."""
import gc
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np


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
from faster_whisper import WhisperModel  # noqa: E402  (must come after the DLL setup)

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


class Transcriber:
    def __init__(self):
        self._model = None
        self._key = None
        self._lock = threading.Lock()
        self.info = "not loaded yet (loads on first use; the first run downloads the model)"

    def _load(self, settings):
        key = (settings["whisper_model"], settings["whisper_device"])
        if self._model is not None and self._key == key:
            return
        device = settings["whisper_device"]
        attempts = [d for d in ("cuda", "cpu") if device in ("auto", d)]
        errors = []
        for dev in attempts:
            name = DEFAULT_MODEL[dev] if settings["whisper_model"] == "auto" else settings["whisper_model"]
            self.info = f"loading {name} on {dev.upper()} (the first time, this downloads it and can take several minutes)"
            try:
                model = WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8")
                # A short warm-up surfaces missing CUDA libraries now instead of mid-meeting.
                list(model.transcribe(np.zeros(16000, np.float32))[0])
            except Exception as e:
                errors.append(f"{dev}: {e}")
                continue
            self._model, self._key = model, key
            self.info = f"{name} on {dev.upper()}"
            return
        self.info = "failed to load"
        raise RuntimeError("Could not load Whisper. " + " | ".join(errors))

    def transcribe(self, audio, settings, offset, language, on_segment, prompt=None):
        """Transcribe a numpy array (16 kHz mono) or an audio/video file path.

        prompt: text Whisper treats as what came before (meeting context, the
        transcript so far). It helps with names, jargon, and punctuation.
        Calls on_segment(start_seconds, end_seconds, text) as segments finish.
        Returns (detected_language, duration_seconds).
        """
        with self._lock:
            self._load(settings)
            # condition_on_previous_text=False: otherwise one looped segment primes
            # the next one and the loop runs to the end of a long file.
            segments, info = self._model.transcribe(
                audio, language=language, vad_filter=True, beam_size=5, initial_prompt=prompt or None,
                condition_on_previous_text=False)
            for seg in segments:
                text = collapse_repeats(seg.text)
                if text and not _hallucinated(seg, text, info.duration):
                    on_segment(offset + seg.start, offset + seg.end, text)
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
