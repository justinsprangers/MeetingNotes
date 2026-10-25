"""Local speech-to-text with faster-whisper. Audio never leaves the computer."""
import os
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
        Calls on_segment(start_seconds, text) as segments finish.
        Returns (detected_language, duration_seconds).
        """
        with self._lock:
            self._load(settings)
            segments, info = self._model.transcribe(
                audio, language=language, vad_filter=True, beam_size=5, initial_prompt=prompt or None)
            for seg in segments:
                text = seg.text.strip()
                if text:
                    on_segment(offset + seg.start, text)
            return info.language, info.duration
