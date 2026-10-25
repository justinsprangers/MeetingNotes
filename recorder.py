"""Records the microphone and (on Windows) system audio, mixes them into one
WAV file, and hands off ~30-second chunks for live transcription.

The microphone uses sounddevice (all platforms). System audio uses WASAPI
loopback through PyAudioWPatch, which only exists on Windows.
"""
import sys
import threading
import time
import wave

import av
import numpy as np
import sounddevice as sd

try:
    import pyaudiowpatch as pyaudio  # Windows only
except ImportError:
    pyaudio = None

RATE = 16000          # Whisper's sample rate
CHUNK_SECONDS = 30    # how much audio to collect before sending it to Whisper
MIX_INTERVAL = 0.5
ACTIVE_LEVEL = 0.003  # RMS above this counts as sound (loopback is exactly 0 when nothing plays)
CALL_SECONDS = 10     # system audio heard this long = a call, so speaker labels make sense


def system_audio_supported() -> bool:
    return sys.platform == "win32" and pyaudio is not None


class _Source:
    """One input device. Callbacks append raw audio; the mixer drains and resamples it."""

    def __init__(self, name: str, in_rate: int):
        self.name = name
        self.in_rate = in_rate
        self.level = 0.0
        self._buf = []
        self._lock = threading.Lock()
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)

    def push(self, frames: np.ndarray):
        mono = frames.mean(axis=1) if frames.ndim == 2 else frames
        with self._lock:
            self._buf.append(mono.astype(np.float32))

    def drain(self) -> np.ndarray:
        with self._lock:
            raw, self._buf = self._buf, []
        if not raw:
            self.level = 0.0
            return np.zeros(0, np.float32)
        audio = np.concatenate(raw)
        self.level = float(np.sqrt(np.mean(audio ** 2)))
        frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = self.in_rate
        out = [f.to_ndarray().reshape(-1) for f in self._resampler.resample(frame)]
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def _quiet_split(audio: np.ndarray) -> int:
    """Index of the quietest 0.3 s in the last 5 s, so chunks don't cut words in half."""
    win = int(0.3 * RATE)
    start = max(0, len(audio) - 5 * RATE)
    tail = audio[start:]
    if len(tail) <= win:
        return len(audio)
    energy = np.convolve(tail ** 2, np.ones(win), "valid")
    return start + int(np.argmin(energy)) + win // 2


class Recorder:
    def __init__(self, wav_path, on_chunk, use_mic=True, use_system=True):
        """on_chunk(audio, offset_seconds, tracks) is called from a background thread.

        tracks is {"Microphone": audio, "System audio": audio} aligned with the
        chunk when both are recorded and a call has been heard, else None.
        """
        self.wav_path = wav_path
        self.on_chunk = on_chunk
        self.use_mic = use_mic
        self.use_system = use_system and system_audio_supported()
        self.sources = []
        self.seconds = 0.0
        self._streams = []
        self._pa = None
        self._pending = []
        self._pending_len = 0
        self._tracks = {}          # per-source audio for the pending chunk
        self._system_seconds = 0.0
        self._chunk_offset = 0.0
        self._running = False
        self._thread = None
        self._wav = None

    def start(self):
        if not (self.use_mic or self.use_system):
            raise RuntimeError("Turn on the microphone or system audio in Settings.")
        try:
            if self.use_mic:
                self._open_mic()
            if self.use_system:
                self._open_loopback()
        except Exception:
            self._close_streams()
            raise
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(RATE)
        self._running = True
        self._thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._thread.start()

    def _open_mic(self):
        rate = int(sd.query_devices(kind="input")["default_samplerate"])
        src = _Source("Microphone", rate)
        stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                callback=lambda data, *_: src.push(data.copy()))
        stream.start()
        self.sources.append(src)
        self._streams.append(stream)

    def _open_loopback(self):
        self._pa = pyaudio.PyAudio()
        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speaker = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        loop = self._pa.get_wasapi_loopback_analogue_by_dict(speaker)
        channels = loop["maxInputChannels"]
        rate = int(loop["defaultSampleRate"])
        src = _Source("System audio", rate)

        def callback(in_data, *_):
            src.push(np.frombuffer(in_data, np.float32).reshape(-1, channels))
            return (None, pyaudio.paContinue)

        stream = self._pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                               input_device_index=loop["index"], frames_per_buffer=1024,
                               stream_callback=callback)
        self.sources.append(src)
        self._streams.append(stream)

    def _mix_loop(self):
        while self._running:
            time.sleep(MIX_INTERVAL)
            self._mix_once()

    def _mix_once(self):
        # Loopback delivers nothing while the computer is silent, so sources are
        # aligned by wall clock: whatever each one produced this interval is mixed,
        # and shorter ones are padded with silence.
        parts = [s.drain() for s in self.sources]
        n = max((len(p) for p in parts), default=0)
        if n == 0:
            return
        mix = np.zeros(n, np.float32)
        for s, p in zip(self.sources, parts):
            mix[:len(p)] += p
            self._tracks.setdefault(s.name, []).append(np.pad(p, (0, n - len(p))))
            if s.name == "System audio" and s.level > ACTIVE_LEVEL:
                self._system_seconds += n / RATE
        np.clip(mix, -1.0, 1.0, out=mix)
        self._wav.writeframes((mix * 32767).astype(np.int16).tobytes())
        self.seconds += n / RATE
        self._pending.append(mix)
        self._pending_len += n
        if self._pending_len >= CHUNK_SECONDS * RATE:
            audio = np.concatenate(self._pending)
            tracks = {name: np.concatenate(t) for name, t in self._tracks.items()}
            cut = _quiet_split(audio)
            self._emit(audio[:cut], {name: t[:cut] for name, t in tracks.items()})
            rest = audio[cut:]
            self._pending, self._pending_len = [rest], len(rest)
            self._tracks = {name: [t[cut:]] for name, t in tracks.items()}

    def _emit(self, audio: np.ndarray, tracks: dict):
        if len(audio):
            labeled = len(tracks) == 2 and self._system_seconds >= CALL_SECONDS
            self.on_chunk(audio, self._chunk_offset, tracks if labeled else None)
            self._chunk_offset += len(audio) / RATE

    def levels(self) -> dict:
        return {s.name: round(s.level, 4) for s in self.sources}

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._close_streams()
        self._mix_once()
        if self._pending:
            self._emit(np.concatenate(self._pending),
                       {name: np.concatenate(t) for name, t in self._tracks.items()})
            self._pending, self._pending_len, self._tracks = [], 0, {}
        if self._wav:
            self._wav.close()

    def _close_streams(self):
        for s in self._streams:
            try:
                if hasattr(s, "stop_stream"):
                    s.stop_stream()
                else:
                    s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []
        if self._pa:
            self._pa.terminate()
            self._pa = None
