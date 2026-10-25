"""Check an Ollama model on a fictional test meeting, for tuning the prompts or picking a model.

Usage (Ollama running, model downloaded):
    .venv\\Scripts\\python eval\\run_eval.py gemma4:12b

Writes each output to eval/results/ (gitignored). Score them by hand against
answer_key.md: decisions, action items, approvals, open questions, and that the
off-the-record remark and small talk are left out.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import summarizer  # noqa: E402

transcript = (HERE / "transcript.txt").read_text(encoding="utf-8")
context = "Lab supervisors meeting. Josh is our manager and wasn't here."
glossary = ("Josh: lab manager\nMaria, Devin, Priya, Sam: lab supervisors\nIPA: isopropyl alcohol (Formlabs wash)\n"
            "Formlabs: resin printer company\nGrabCAD Print: printing software")
system = summarizer.system_prompt(
    f"<glossary>\nNames and terms that come up in the user's meetings. Use these spellings and meanings, "
    f"and use them to correct misheard words.\n{glossary}\n</glossary>")
model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b"
settings = {"ollama_url": "http://127.0.0.1:11434", "ollama_model": model}

REQUESTS = {
    "notes": "Write organized meeting notes grouped by topic, with a Decisions section and an Action items section (owner and due date when mentioned).",
    "update": "Write a short, quickly digestible update for someone who missed this meeting: key decisions, items that need their approval or input, and future plans and focus.",
}

out_dir = HERE / "results"
out_dir.mkdir(exist_ok=True)
# The test meeting is short, so "twopass" forces the long-meeting path with small sections.
for mode, single_max, section_chars in (("single", 10 ** 6, 12000), ("twopass", 0, 3000)):
    summarizer.SINGLE_PASS_CHARS, summarizer.SECTION_CHARS = single_max, section_chars
    cache = {}
    for name, request in REQUESTS.items():
        start = time.time()
        parts = []
        for kind, value in summarizer.ollama_answer(settings, system, [], request, transcript, context,
                                                    cache, glossary):
            if kind == "text":
                parts.append(value)
            else:
                print(f"  [{mode}/{name}] {value}", flush=True)
        text = "".join(parts)
        (out_dir / f"{model.replace(':', '_')}_{mode}_{name}.md").write_text(text, encoding="utf-8")
        print(f"{mode}/{name}: {time.time() - start:.0f}s, {len(text)} chars", flush=True)
