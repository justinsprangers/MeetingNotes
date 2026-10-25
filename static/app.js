"use strict";

const PRESETS = [
  ["Meeting notes", "Write organized meeting notes grouped by topic, with a Decisions section and an Action items section (owner and due date when mentioned)."],
  ["Update for someone absent", "Write a short, quickly digestible update for someone who missed this meeting: key decisions, items that need their approval or input, and future plans and focus."],
  ["Action items", "List every action item as a checklist: task, owner, and due date when mentioned."],
  ["Decisions & open questions", "List the decisions that were made. Then, separately, list the open questions that were discussed but not decided."],
  ["Format for Discord", "Format your last response as a Discord message in a single code block, under 2,000 characters, with no emojis."],
  ["Format as email", "Turn your last response into a short, plain email with a subject line, in a single code block."],
];

const $ = (id) => document.getElementById(id);
let meetingId = null;
let historyLength = -1;
let streaming = false;
let saveTimer = null;
let lastState = null;
let silentPolls = 0;

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}

function formatTime(seconds) {
  const s = Math.floor(seconds);
  const pad = (n) => String(n).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`
                   : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

// ---- Rendering ----------------------------------------------------------

function setField(el, value) {
  // Don't clobber what the user is typing, or an edit that hasn't saved yet.
  if (document.activeElement !== el && !saveTimer && el.value !== value) el.value = value;
}

function render(s) {
  lastState = s;
  if (s.id !== meetingId) {
    meetingId = s.id;
    historyLength = -1;
    $("title").value = s.title;
    $("context").value = s.context;
    $("transcript").value = s.transcript;
    loadPastMeetings();
  }
  setField($("title"), s.title);
  setField($("context"), s.context);

  const busy = s.recording || s.pending > 0;
  const t = $("transcript");
  t.readOnly = busy;
  if (busy || document.activeElement !== t) {
    const atBottom = t.scrollTop + t.clientHeight >= t.scrollHeight - 20;
    setField(t, s.transcript);
    if (busy && atBottom) t.scrollTop = t.scrollHeight;
  }

  const rec = $("record");
  rec.textContent = s.recording ? `■ Stop (${formatTime(s.record_seconds)})` : "● Record";
  rec.classList.toggle("recording", s.recording);
  rec.classList.toggle("primary", !s.recording);

  // Windows records silence from muted speakers, which is easy to miss.
  silentPolls = s.recording && s.levels["System audio"] === 0 ? silentPolls + 1 : 0;
  let status = "";
  if (s.pending > 0) status = `Transcribing… (${s.pending} part${s.pending > 1 ? "s" : ""} queued; Whisper: ${s.whisper})`;
  else if (s.recording) status = "Recording. The transcript fills in about every 30 seconds.";
  else if (s.prewarm === "reading") status = "Ollama is reading the transcript so answers start fast (you can ask now; it'll wait for this).";
  if (silentPolls > 20) status += " No system audio so far. Windows records silence when the volume is muted or at 0.";
  $("status").textContent = status;

  $("levels").innerHTML = Object.entries(s.levels).map(([name, level]) =>
    `<span>${name}<span class="meter"><span style="width:${Math.min(100, level * 400)}%"></span></span></span>`).join("");
  showError(s.error);
  renderCopyButtons(s.copied);

  if (!streaming && s.history.length !== historyLength) {
    historyLength = s.history.length;
    renderChat(s.history);
  }
}

function renderChat(history) {
  const chat = $("chat");
  chat.querySelectorAll(".msg").forEach((m) => m.remove());
  chat.querySelector(".empty").hidden = history.length > 0;
  // Saved messages get the "learn from this" buttons.
  history.forEach((m, i) => addMessage(m.role, m.text, i > 0 ? history[i - 1].text : null, true));
  chat.scrollTop = chat.scrollHeight;
}

function toolButton(label, title, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.title = title;
  b.addEventListener("click", () => onClick(b));
  return b;
}

function flash(button, text) {
  const old = button.textContent;
  button.textContent = text;
  setTimeout(() => (button.textContent = old), 1500);
}

function addMessage(role, text, request = null, saved = false) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const body = document.createElement("div");
  body.textContent = text;
  div.append(body);
  const tools = document.createElement("div");
  tools.className = "tools";
  if (role === "assistant") {
    tools.append(toolButton("Copy", "Copy this response", (b) => copyText(body.textContent, b)));
    if (saved && request) {
      tools.append(toolButton("Save as example", "Use this response as a model for similar requests in future meetings",
        async (b) => { await api("/api/memory/example", { request, output: body.textContent }); flash(b, "Saved"); }));
    }
  } else if (saved) {
    tools.append(toolButton("Remember", "Add this request to your standing preferences, for every future meeting",
      async (b) => { await api("/api/memory/preference", { text }); flash(b, "Remembered"); }));
  }
  if (tools.children.length) role === "assistant" ? div.prepend(tools) : div.append(tools);
  $("chat").querySelector(".empty").hidden = true;
  $("chat").append(div);
  return body;
}

async function copyText(text, button) {
  // If the response is one code block (a message to paste somewhere), copy just its contents.
  const blocks = [...text.matchAll(/```[^\n]*\n([\s\S]*?)```/g)];
  const toCopy = blocks.length === 1 ? blocks[0][1].trimEnd() : text;
  await navigator.clipboard.writeText(toCopy);
  flash(button, blocks.length === 1 ? "Copied message" : "Copied");
}

// ---- Polling ------------------------------------------------------------

async function poll() {
  try {
    render(await api("/api/state"));
  } catch (e) {
    $("status").textContent = "Can't reach the app. Is the Meeting Notes window still open?";
  }
  setTimeout(poll, 1000);
}

async function loadPastMeetings() {
  const list = await api("/api/meetings").catch(() => []);
  const sel = $("past");
  sel.innerHTML = '<option value="">Past meetings…</option>' + list.map((m) =>
    `<option value="${m.id}">${m.id.replace(/_(\d\d)(\d\d)\d\d$/, " $1:$2")}${m.title ? " · " + escapeHtml(m.title) : ""}</option>`).join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- Editing ------------------------------------------------------------

function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 600);
}

async function saveNow() {
  clearTimeout(saveTimer);
  saveTimer = null;
  const body = { title: $("title").value, context: $("context").value };
  if (!$("transcript").readOnly) body.transcript = $("transcript").value;
  await api("/api/meeting", body).catch((e) => showError(e.message));
}

["title", "context", "transcript"].forEach((id) => $(id).addEventListener("input", saveSoon));

// ---- Capture ------------------------------------------------------------

$("record").addEventListener("click", async () => {
  const rec = $("record");
  rec.disabled = true;
  try {
    await saveNow();
    await api(lastState && lastState.recording ? "/api/record/stop" : "/api/record/start", {});
  } catch (e) {
    showError(e.message);
  } finally {
    rec.disabled = false;
  }
});

$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  if (!file) return;
  $("status").textContent = `Uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) showError((await res.json()).error);
  $("file").value = "";
});

// ---- Asking -------------------------------------------------------------

async function ask(text) {
  if (streaming || !text.trim()) return;
  await saveNow();
  streaming = true;
  $("send").disabled = true;
  addMessage("user", text);
  const out = addMessage("assistant", "…");
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: text }),
    });
    if (!res.ok) {
      out.textContent = `[${(await res.json()).error}]`;
    } else {
      $("ask").value = "";
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let raw = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const nearBottom = chat.scrollTop + chat.clientHeight >= chat.scrollHeight - 40;
        raw += decoder.decode(value, { stream: true });
        showStreamed(out, raw);
        if (nearBottom) chat.scrollTop = chat.scrollHeight;
      }
    }
  } catch (e) {
    out.textContent += `\n\n[Connection lost: ${e.message}]`;
  } finally {
    // If the server saved this exchange, the history grows and the next poll redraws
    // the chat from it. If it failed, the error stays on screen.
    streaming = false;
    $("send").disabled = false;
  }
}

// Progress notes arrive inside the stream as \x01note\x02 (e.g. "Reading section 2 of 5").
// Show the latest one until the answer itself starts.
function showStreamed(out, raw) {
  const notes = [...raw.matchAll(/\x01([^\x02]*)\x02/g)];
  const text = raw.replace(/\x01[^\x02]*\x02/g, "").replace(/\x01[^\x02]*$/, "");
  if (text.trim() || !notes.length) {
    out.textContent = text || "…";
  } else {
    out.innerHTML = "";
    const note = document.createElement("span");
    note.className = "status-note";
    note.textContent = notes[notes.length - 1][1];
    out.append(note);
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("ask").value);
});
$("ask").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    ask($("ask").value);
  }
});

PRESETS.forEach(([label, prompt]) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.title = prompt;
  b.addEventListener("click", () => {
    $("ask").value = prompt;
    $("ask").focus();
  });
  $("presets").append(b);
});

// No API key needed: paste into any AI chat. The first copy carries the transcript;
// after that, copies carry just the request (plus any new transcript) for the same chat.
function renderCopyButtons(copied) {
  const button = $("copy-ai");
  if (!button.dataset.flashing) {
    button.textContent = copied ? "Copy follow-up" : "Copy for AI";
    button.title = copied
      ? "Copies just your request (and any new transcript) to paste into the same AI chat"
      : "Copies the transcript, context and your request to paste into any AI chat (Claude.ai, ChatGPT, …). No API key or cost.";
  }
  $("copy-new").hidden = !copied;
}

async function copyForAI(restart) {
  const button = $("copy-ai");
  try {
    await saveNow();
    const { text } = await api("/api/copy-prompt", { request: $("ask").value, restart });
    await navigator.clipboard.writeText(text);
    $("ask").value = "";
    button.dataset.flashing = "1";
    button.textContent = "Copied: paste it (Ctrl+V)";
    setTimeout(() => { delete button.dataset.flashing; renderCopyButtons(lastState.copied); }, 3000);
  } catch (e) {
    showError(e.message);
  }
}

$("copy-ai").addEventListener("click", () => copyForAI(false));
$("copy-new").addEventListener("click", () => copyForAI(true));

$("reset-chat").addEventListener("click", async () => {
  if (!confirm("Clear the chat for this meeting? The transcript is kept.")) return;
  await api("/api/reset-chat", {});
  historyLength = -2;
});

// ---- Meetings -----------------------------------------------------------

$("new-meeting").addEventListener("click", async () => {
  try {
    await saveNow();
    await api("/api/new", {});
  } catch (e) {
    showError(e.message);
  }
});

$("past").addEventListener("change", async () => {
  const id = $("past").value;
  if (!id) return;
  try {
    await saveNow();
    await api("/api/open", { id });
  } catch (e) {
    showError(e.message);
  }
  $("past").value = "";
});

// ---- Settings -----------------------------------------------------------

const form = $("settings-form");

async function openSettings() {
  const s = lastState.settings;
  form.provider.value = s.provider;
  form.anthropic_api_key.value = "";
  $("key-hint").textContent = s.api_key_hint || "No key saved.";
  form.claude_model.value = s.claude_model;
  form.ollama_url.value = s.ollama_url;
  form.whisper_model.value = s.whisper_model;
  form.whisper_device.value = s.whisper_device;
  form.language.value = s.language;
  form.record_microphone.checked = s.record_microphone;
  form.record_system_audio.checked = s.record_system_audio;
  form.record_system_audio.disabled = !lastState.system_audio_supported;
  $("system-audio-hint").textContent = lastState.system_audio_supported ? ""
    : "System audio capture is Windows only. On Mac or Linux, record your mic, or upload the meeting's recording afterward.";
  $("whisper-info").textContent = lastState.whisper;

  await refreshOllama(s.ollama_model);
  $("settings").showModal();
}

let recommended = null;

async function refreshOllama(selected) {
  const info = await api("/api/ollama").catch(() => ({ running: false, models: [], pull: {} }));
  const models = info.models;
  const options = models.includes(selected) || !selected ? models : [selected, ...models];
  form.ollama_model.innerHTML = options.map((m) => `<option>${escapeHtml(m)}</option>`).join("");
  form.ollama_model.value = selected || models[0] || "";
  $("ollama-hint").textContent = !info.running
    ? "Ollama isn't running. Install it from ollama.com and start it, then reopen Settings."
    : models.length ? "" : "No models yet. Download the recommended one below.";
  recommended = info.recommended;
  if (recommended) {
    const have = models.includes(recommended.model);
    $("ollama-recommend").textContent = `Recommended for this computer (${recommended.hardware}): ` +
      `${recommended.model}, ${recommended.size}${have ? " (installed)" : ""}`;
    $("ollama-pull").hidden = have;
  }
  $("ollama-pull").disabled = !info.running || info.pull.running;
  const p = info.pull;
  $("ollama-pull-status").textContent = p.error || (p.model ? `${p.model}: ${p.status}` : "");
  if (p.running) setTimeout(() => $("settings").open && refreshOllama(form.ollama_model.value), 1000);
  else if (p.model && p.status.startsWith("Done") && selected !== p.model) form.ollama_model.value = p.model;
}

$("ollama-pull").addEventListener("click", async () => {
  try {
    await api("/api/ollama/pull", { model: recommended.model });
    refreshOllama(recommended.model);
  } catch (e) {
    $("ollama-pull-status").textContent = e.message;
  }
});

// ---- Memory -------------------------------------------------------------

async function openMemory() {
  const mem = await api("/api/memory");
  $("glossary").value = mem.glossary;
  $("preferences").value = mem.preferences;
  renderExamples(mem.examples);
  $("memory").showModal();
}

function renderExamples(examples) {
  const list = $("examples");
  list.innerHTML = examples.length ? "" : '<span class="hint">None saved yet.</span>';
  examples.slice().reverse().forEach((e) => {
    const row = document.createElement("div");
    row.className = "example";
    const label = document.createElement("span");
    label.textContent = `${e.saved}: ${e.request}`;
    label.title = e.output;
    row.append(label, toolButton("Delete", "Forget this example", async () => {
      renderExamples((await api("/api/memory/example/delete", { id: e.id })).examples);
    }));
    list.append(row);
  });
}

$("memory-form").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "save") return;
  await api("/api/memory", { glossary: $("glossary").value, preferences: $("preferences").value })
    .catch((err) => showError(err.message));
});

$("open-memory").addEventListener("click", openMemory);

form.addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "save") return;
  await api("/api/settings", {
    provider: form.provider.value,
    anthropic_api_key: form.anthropic_api_key.value.trim(),
    claude_model: form.claude_model.value.trim(),
    ollama_url: form.ollama_url.value.trim(),
    ollama_model: form.ollama_model.value,
    whisper_model: form.whisper_model.value,
    whisper_device: form.whisper_device.value,
    language: form.language.value.trim() || "auto",
    record_microphone: form.record_microphone.checked,
    record_system_audio: form.record_system_audio.checked,
  }).catch((err) => showError(err.message));
});

$("clear-key").addEventListener("click", async () => {
  await api("/api/settings", { clear_api_key: true });
  $("key-hint").textContent = "No key saved.";
});

$("open-settings").addEventListener("click", openSettings);
$("open-help").addEventListener("click", () => $("help").showModal());

poll();
