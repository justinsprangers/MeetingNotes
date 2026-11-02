"""Who said it: labels transcript lines by voice, on this computer.

Each line's audio becomes a voiceprint (a 256-number embedding from WeSpeaker's
ResNet34 model via sherpa-onnx, ~70 ms per line on 2 CPU threads). Voiceprints are
centered on the meeting's average first: the room, mic and codec make every voice
in a recording look alike (in one recording any two lines scored 0.82 similar),
and subtracting the average leaves what differs between people. Lines whose
centered voiceprints group together get one label ("Speaker 1", ...). When the user
names a speaker, the voice is saved in voices.json (gitignored) and recognized by
name in later meetings.

Tuned on a real 57-min, ~6-person room meeting: after centering, one person's two
halves of the meeting scored 0.94 against each other and other people at most 0.30.
Lines under 1.5 s are too short to tell people apart and stay unlabeled.
"""
import json
import logging
import threading
import urllib.request

import numpy as np

from config import APP_DIR

log = logging.getLogger("meetingnotes")

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx")
MODEL_PATH = APP_DIR / "models" / "wespeaker_en_voxceleb_resnet34_LM.onnx"
VOICES_PATH = APP_DIR / "voices.json"
RATE = 16000
MIN_SECONDS = 1.5    # shorter lines don't carry enough voice to tell people apart
SAME_GROUP = 0.25    # average centered similarity for two groups of lines to be one person
LIVE_SAME = 0.3      # a single new line joining an existing speaker (noisier than groups)
JOIN_SMALL = 0.2     # lines outside the main groups join the nearest speaker if at least this similar
MIN_LINES = 3        # fewer lines than this (or 2% of the meeting) isn't a speaker of its own
MIN_SHARE = 50       # ...where 2% = 1/50; at 1%, the test meeting split ~7 people into 18 speakers
KNOWN_PERSON = 0.6   # to put a remembered name on a speaker from an earlier meeting
WARMUP_LINES = 8     # before this many lines, the meeting average is too rough to center on


class VoiceTagger:
    def __init__(self):
        self._extractor = None
        self._lock = threading.Lock()
        self.error = ""

    def _load(self):
        if self._extractor is None:
            import sherpa_onnx  # only needed when speaker labels are on
            if not MODEL_PATH.exists():
                log.info("Downloading the voice model (25 MB)")
                MODEL_PATH.parent.mkdir(exist_ok=True)
                tmp = MODEL_PATH.with_suffix(".part")
                urllib.request.urlretrieve(MODEL_URL, tmp)
                tmp.replace(MODEL_PATH)
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL_PATH), num_threads=2))
        return self._extractor

    def voiceprint(self, audio):
        """Unit-length embedding of 16 kHz mono audio, or None if too short or unavailable."""
        if len(audio) < MIN_SECONDS * RATE:
            return None
        with self._lock:
            try:
                ext = self._load()
            except Exception as e:  # no internet for the first download, or sherpa-onnx missing
                self.error = f"Speaker labels are off: couldn't load the voice model ({e})."
                return None
            stream = ext.create_stream()
            stream.accept_waveform(RATE, np.ascontiguousarray(audio, np.float32))
            stream.input_finished()
            v = np.array(ext.compute(stream))
        n = np.linalg.norm(v)
        return v / n if n else None


def _unit_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def center(X):
    """Remove what all lines of a recording share (room, mic, codec)."""
    X = np.asarray(X)
    return _unit_rows(X - X.mean(axis=0)) if len(X) >= WARMUP_LINES else X


def _next_speaker(used):
    n = 1
    while f"Speaker {n}" in used:
        n += 1
    return f"Speaker {n}"


def assign_live(prints, labels, v):
    """Label for a new line as it arrives. prints: raw voiceprints of earlier labeled lines,
    labels: their labels. The periodic regroup() corrects these first guesses."""
    if not len(prints):
        return "Speaker 1"
    C = center(np.vstack([prints, v]))
    new, C = C[-1], C[:-1]
    best, best_sim = None, LIVE_SAME
    for label in set(labels) - {None}:
        sim = float(new @ _unit(C[[l == label for l in labels]].mean(axis=0)))
        if sim >= best_sim:
            best, best_sim = label, sim
    return best or _next_speaker(set(labels))


def _agglomerate(X, threshold):
    """Average-linkage clustering on cosine similarity. Returns a group id per row."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    size = np.ones(len(X))
    group = np.arange(len(X))
    # Best match per row, kept up to date so each merge costs O(n) instead of O(n^2).
    best, arg = S.max(axis=1), S.argmax(axis=1)
    while len(X) > 1:
        i = int(np.argmax(best))
        j = int(arg[i])
        if best[i] < threshold:
            break
        # Lance-Williams update for average linkage: fold group j into group i.
        S[i, :] = (size[i] * S[i, :] + size[j] * S[j, :]) / (size[i] + size[j])
        S[:, i] = S[i, :]
        S[i, i] = -np.inf
        S[j, :] = -np.inf
        S[:, j] = -np.inf
        size[i] += size[j]
        group[group == j] = i
        best[j] = -np.inf
        stale = np.flatnonzero((arg == i) | (arg == j))
        stale = stale[best[stale] > -np.inf]
        best[stale], arg[stale] = S[stale].max(axis=1), S[stale].argmax(axis=1)
        better = S[:, i] > best
        best[better], arg[better] = S[better, i], i
        best[i], arg[i] = S[i].max(), S[i].argmax()
    return group


def regroup(prints, labels, known):
    """Group all of a meeting's voiceprints at once (more accurate than labeling lines as
    they arrive, which lets one big group absorb similar voices) and pick a label per group:
    a name most of its lines already have (given by the user), else a remembered person
    whose voice matches, else "Speaker N" numbered by when they first spoke.

    prints: raw voiceprints, one per labeled line, in order. labels: their current labels.
    Returns the new labels (None = unlabeled).
    """
    C = center(prints)
    group = _agglomerate(C.copy(), SAME_GROUP)
    ids, counts = np.unique(group, return_counts=True)
    big = [g for g, c in zip(ids, counts) if c >= max(MIN_LINES, len(C) // MIN_SHARE)]
    if not big:
        return list(labels)
    cents = {g: _unit(C[group == g].mean(axis=0)) for g in big}
    for k, g in enumerate(group):
        if g not in cents:  # a stray line or two
            sims = {b: float(C[k] @ c) for b, c in cents.items()}
            best = max(sims, key=sims.get)
            group[k] = best if sims[best] >= JOIN_SMALL else -1
    order = sorted(cents, key=lambda g: int(np.argmax(group == g)))  # first appearance
    names, used = {}, set()

    for g in order:  # names the user gave
        mine = [l for l, gg in zip(labels, group) if gg == g and l and not l.startswith("Speaker ")]
        top = max(set(mine), key=mine.count) if mine else None
        names[g] = top if top and top not in used and mine.count(top) * 2 >= np.sum(group == g) else None
        used.add(names[g])
    for g in order:  # remembered people
        if names[g] is None:
            sims = {n: float(cents[g] @ _unit(np.array(e["sum"]))) for n, e in known.items() if n not in used}
            best = max(sims, key=sims.get) if sims else None
            if best and sims[best] >= KNOWN_PERSON:
                names[g] = best
                used.add(best)
    for g in order:
        if names[g] is None:
            names[g] = _next_speaker(used)
            used.add(names[g])
    return [names.get(g) for g in group]


def voice_of(prints, labels, label):
    """Centered average voiceprint of one speaker's lines, for remembering them."""
    C = center(prints)
    mask = [l == label for l in labels]
    return {"sum": _unit(C[mask].mean(axis=0)).tolist(), "n": int(sum(mask))}


def load_known():
    if VOICES_PATH.exists():
        try:
            return json.loads(VOICES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_known(known):
    VOICES_PATH.write_text(json.dumps(known), encoding="utf-8")


def remember(name, entry):
    """Add a named speaker's voice from one meeting to the saved people (weighted by lines)."""
    known = load_known()
    old = known.get(name)
    if old:
        total = old["n"] + entry["n"]
        mixed = (np.array(old["sum"]) * old["n"] + np.array(entry["sum"]) * entry["n"]) / total
        entry = {"sum": _unit(mixed).tolist(), "n": total}
    known[name] = entry
    save_known(known)


def forget(name):
    known = load_known()
    known.pop(name, None)
    save_known(known)
