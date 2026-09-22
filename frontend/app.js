const API = "/api/chat";

const PROMPTS = {
  en: [
    { label: "Eligibility", text: "Who is eligible for PM POSHAN?" },
    { label: "Documents", text: "What documents are required for Tablet Assistance?" },
    { label: "Benefits", text: "What is the benefit under Medical Checkup Assistance?" },
  ],
  hi: [
    { label: "पात्रता", text: "पीएम पोषण योजना के लिए कौन पात्र है?" },
    { label: "दस्तावेज", text: "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?" },
    { label: "लाभ", text: "चिकित्सा जांच सहायता योजना का लाभ क्या है?" },
  ],
  mr: [
    { label: "पात्रता", text: "पीएम पोषण योजनेसाठी कोण पात्र आहे?" },
    { label: "कागदपत्रे", text: "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?" },
    { label: "लाभ", text: "वैद्यकीय तपासणी सहाय्य योजनेचा लाभ काय आहे?" },
  ],
};

let language = "en";
let busy = false;

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const chatScroll = $("chatScroll");
const welcome = $("welcome");
const input = $("input");
const btnSend = $("btnSend");
const statusDot = $("statusDot");
const statusText = $("statusText");

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  $("themeIcon").className = theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
}

function initTheme() {
  setTheme(localStorage.getItem("theme") || "dark");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderPrompts() {
  const list = PROMPTS[language] || PROMPTS.en;
  $("promptGrid").innerHTML = list
    .map(
      (p) => `
      <button type="button" class="prompt-card" data-text="${escapeHtml(p.text)}">
        <span class="pc-label">${escapeHtml(p.label)}</span>
        ${escapeHtml(p.text)}
      </button>`
    )
    .join("");

  $("promptGrid").querySelectorAll(".prompt-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = btn.getAttribute("data-text");
      input.focus();
      autoGrow();
    });
  });
}

function setBusy(v) {
  busy = v;
  btnSend.disabled = v;
  statusDot.classList.toggle("busy", v);
  statusDot.classList.remove("err");
  statusText.textContent = v ? "Thinking…" : "Ready";
}

function scrollBottom() {
  requestAnimationFrame(() => {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function appendUser(text) {
  welcome.hidden = true;
  const div = document.createElement("div");
  div.className = "row user";
  div.innerHTML = `
    <div class="av"><i class="fa-solid fa-user"></i></div>
    <div class="card">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(div);
  scrollBottom();
}

function appendTyping() {
  const div = document.createElement("div");
  div.className = "row bot";
  div.id = "typingRow";
  div.innerHTML = `
    <div class="av"><i class="fa-solid fa-robot"></i></div>
    <div class="card"><div class="typing"><i></i><i></i><i></i></div></div>
  `;
  messagesEl.appendChild(div);
  scrollBottom();
}

function removeTyping() {
  const t = $("typingRow");
  if (t) t.remove();
}

function appendBot(data) {
  removeTyping();

  const answer = (data.answer || "").trim();

  const cites = (data.citations || [])
    .filter((c) => c && (c.scheme_id || c.section))
    .map(
      (c) => `
      <div class="cite">
        <i class="fa-solid fa-link"></i>
        <span><b>${escapeHtml(c.scheme_id || "—")}</b> · ${escapeHtml(
          c.section || "—"
        )} · ${escapeHtml(c.language || "")}</span>
      </div>`
    )
    .join("");

  const div = document.createElement("div");
  div.className = "row bot";
  div.innerHTML = `
    <div class="av"><i class="fa-solid fa-robot"></i></div>
    <div class="card">
      <div class="answer">${escapeHtml(answer)}</div>
      <div class="stats">
        <span class="stat"><i class="fa-solid fa-clock"></i>${data.latency_ms} ms</span>
        <span class="stat"><i class="fa-solid fa-layer-group"></i>${data.n_retrieved} sources</span>
        <span class="stat"><i class="fa-solid fa-shield"></i>${escapeHtml(
          data.confidence || "—"
        )}</span>
      </div>
      ${
        cites
          ? `<div class="cites"><div class="cites-h">Citations</div>${cites}</div>`
          : ""
      }
    </div>
  `;
  messagesEl.appendChild(div);
  scrollBottom();
}

function appendError(msg) {
  removeTyping();
  statusDot.classList.add("err");
  statusText.textContent = "Error";
  const div = document.createElement("div");
  div.className = "row bot";
  div.innerHTML = `
    <div class="av"><i class="fa-solid fa-triangle-exclamation"></i></div>
    <div class="card">${escapeHtml(msg)}</div>
  `;
  messagesEl.appendChild(div);
  scrollBottom();
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

async function sendMessage() {
  const q = (input.value || "").trim();
  if (!q || busy) return;

  input.value = "";
  autoGrow();
  appendUser(q);
  appendTyping();
  setBusy(true);

  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, language, top_k: 5 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    appendBot(data);
  } catch (e) {
    appendError(e.message || "Request failed");
  } finally {
    setBusy(false);
  }
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
input.addEventListener("input", autoGrow);

$("langSeg").addEventListener("click", (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  language = btn.dataset.lang;
  $("langSeg").querySelectorAll(".seg-btn").forEach((b) => {
    b.classList.toggle("active", b === btn);
  });
  renderPrompts();
});

$("btnTheme").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  setTheme(cur === "dark" ? "light" : "dark");
});

$("btnNew").addEventListener("click", () => {
  messagesEl.innerHTML = "";
  welcome.hidden = false;
  statusText.textContent = "Ready";
  statusDot.classList.remove("err", "busy");
});

initTheme();
renderPrompts();
