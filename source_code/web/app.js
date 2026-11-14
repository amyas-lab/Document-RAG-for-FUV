/* Applied Mathematics Handbook Assistant: static client for the FastAPI backend.
 *
 * Served by nginx in Docker (API proxied under /api) or by FastAPI at /ui/ in local development
 * (API on the same origin, no prefix).
 */
(() => {
  "use strict";

  const API = location.pathname.startsWith("/ui") ? "" : "/api";
  const NOT_FOUND = "I couldn't find this in the Applied Mathematics handbook.";
  const EXAMPLES = [
    ["Minor", "What are the requirements for a minor in Applied Mathematics?"],
    ["Prerequisites", "What are the prerequisites for MATH 305 Stochastic Calculus?"],
    ["Catalogue", "Which courses from other majors are cross-listed with Applied Mathematics?"],
    ["Declaring", "What do I need to do before I can declare the Applied Mathematics major?"],
    ["Cross-listing", "Which MATH courses are cross-listed with Economics?"],
    ["Learning outcomes", "What is MLO 4 of the Applied Mathematics major?"],
  ];

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  };

  // ------------------------------------------------------------------ state
  const state = {
    messages: [],          // {role, content, sources, query, meta, mode, status, error}
    active: -1,            // index of the assistant message shown in the evidence panel
    busy: false,
    docs: [],
    settings: { model: null, mode: "strict", top_k: 5, temperature: 0.1 },
  };

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
      catch { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
  };
  Object.assign(state.settings, store.get("hb.settings", {}));

  // ------------------------------------------------------------------ text rendering
  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // "p. 21", "pp. 9-10", "page 14", "(Source: p. 9, p. 21)" -> clickable page chips
  const CITE = /\b(?:pp?\.|pages?)\s*\d{1,3}(?:\s*(?:,|and|&amp;|–|-)\s*(?:pp?\.\s*|pages?\s*)?\d{1,3})*/gi;
  function linkCitations(html) {
    return html.replace(CITE, (match) => {
      const pages = [...match.matchAll(/\d{1,3}/g)].map((m) => +m[0]).filter((p) => p >= 1 && p <= 200);
      if (!pages.length) return match;
      return pages.map((p) => `<button type="button" class="cite" data-page="${p}" title="Show page ${p}">p.&nbsp;${p}</button>`).join(" ");
    });
  }

  function inline(s) {
    return linkCitations(
      escapeHtml(s)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>"),
    );
  }

  /** Minimal Markdown: paragraphs, bullet and numbered lists, bold, italic, code. */
  function renderMarkdown(text) {
    const out = [];
    let list = null;   // "ul" | "ol"
    let para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^[-*•]\s+(.*)$/);
      const number = line.match(/^\d+[.)]\s+(.*)$/);
      if (bullet || number) {
        flushPara();
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
        out.push(`<li>${inline((bullet || number)[1])}</li>`);
      } else if (!line) {
        flushPara(); closeList();
      } else {
        if (list) closeList();
        para.push(line);
      }
    }
    flushPara(); closeList();
    return out.join("");
  }

  const isRefusal = (t) => t.trim().startsWith(NOT_FOUND.slice(0, 25)) && t.trim().length < NOT_FOUND.length + 40;

  // ------------------------------------------------------------------ API
  async function getJSON(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  /** POST /chat/stream and yield [event, data] pairs from the Server-Sent Events body. */
  async function* streamChat(body, signal) {
    const r = await fetch(`${API}/chat/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    });
    if (!r.ok || !r.body) throw new Error(`${r.status} ${r.statusText}`);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        let event = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) yield [event, JSON.parse(data)];
      }
    }
  }

  // ------------------------------------------------------------------ status
  function setStatus(id, ok) {
    const n = $(id);
    n.classList.toggle("is-ok", ok === true);
    n.classList.toggle("is-bad", ok === false);
  }

  async function refreshHealth() {
    try {
      const h = await getJSON("/health");
      setStatus("#st-ollama", h.ollama);
      setStatus("#st-qdrant", h.qdrant && h.n_chunks > 0);
      $("#st-chunks").textContent = `${h.n_chunks} chunks`;
      $("#kv-collection").textContent = h.collection;
      $("#kv-chunks").textContent = `${h.n_chunks}${h.index_up_to_date ? "" : " (outdated)"}`;
      $("#kv-mode").textContent = `Qdrant (${h.qdrant_mode})`;
      const d = h.defaults || {};
      $("#kv-pipeline").textContent = `${d.parser}, ${d.chunking}, ${d.retrieval}`;
      return h;
    } catch {
      setStatus("#st-ollama", false);
      setStatus("#st-qdrant", false);
      $("#st-chunks").textContent = "API offline";
      return null;
    }
  }

  async function loadModels(defaults) {
    const select = $("#set-model");
    let models = [], fallback = defaults?.model || "qwen2.5:3b";
    try { const m = await getJSON("/models"); models = m.models; fallback = m.default || fallback; } catch { /* offline */ }
    if (!models.length) models = [fallback];
    if (!models.includes(state.settings.model)) state.settings.model = models.includes(fallback) ? fallback : models[0];
    select.innerHTML = models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
    select.value = state.settings.model;
  }

  // ------------------------------------------------------------------ theme
  function applyTheme(choice, persist) {
    if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
    document.querySelectorAll("#theme-group button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.themeChoice === choice)));
    if (persist) store.set("hb.theme", choice);
  }
  applyTheme(store.get("hb.theme", "system"), false);

  // ------------------------------------------------------------------ settings
  function applySettingsToUI() {
    const s = state.settings;
    $("#set-topk").value = s.top_k; $("#out-topk").textContent = s.top_k;
    $("#set-temp").value = s.temperature; $("#out-temp").textContent = (+s.temperature).toFixed(2);
    document.querySelectorAll("#mode-group button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === s.mode)));
    $("#mode-help").textContent = s.mode === "strict"
      ? "Answers only from the handbook and says so when the answer is not there. Recommended."
      : "Baseline prompt: may fill gaps with the model's general knowledge. Kept for comparison.";
    $("#mode-hint").textContent = s.mode === "strict" ? "Strict mode: handbook only" : "Fallback mode: may use general knowledge";
  }
  const saveSettings = () => { store.set("hb.settings", state.settings); applySettingsToUI(); };

  function openDrawer(open) {
    $("#settings").classList.toggle("is-open", open);
    $("#settings").setAttribute("aria-hidden", String(!open));
    $("#scrim").hidden = !open;
    if (open) $("#set-model").focus();
  }

  // ------------------------------------------------------------------ evidence panel
  function showTab(name) {
    const sources = name === "sources";
    $("#tab-sources").classList.toggle("is-active", sources);
    $("#tab-doc").classList.toggle("is-active", !sources);
    $("#tab-sources").setAttribute("aria-selected", String(sources));
    $("#tab-doc").setAttribute("aria-selected", String(!sources));
    $("#pane-sources").hidden = !sources;
    $("#pane-doc").hidden = sources;
  }

  function openEvidence(open) { $("#evidence").classList.toggle("is-open", open); }

  const doc = { name: null, page: 1, pages: 0 };

  function showPage(page) {
    if (!doc.name) { $("#doc-fallback").hidden = false; return; }
    doc.page = Math.max(1, Math.min(page, doc.pages || page));
    const img = $("#page-img");
    $("#doc-fallback").hidden = true;
    img.classList.add("is-loading");
    img.alt = `Handbook page ${doc.page}`;
    img.src = `${API}/documents/${encodeURIComponent(doc.name)}/pages/${doc.page}.png?width=900`;
    $("#pg-input").value = doc.page;
    $("#pg-prev").disabled = doc.page <= 1;
    $("#pg-next").disabled = doc.pages > 0 && doc.page >= doc.pages;
    $("#pg-open").href = `${API}/documents/${encodeURIComponent(doc.name)}#page=${doc.page}`;
    $("#doc-page").textContent = `p. ${doc.page}`;
    $("#page-view").scrollTop = 0;
  }

  function openPage(page, source) {
    const entry = state.docs.find((d) => d.name === source) || state.docs[0];
    if (entry) { doc.name = entry.name; doc.pages = entry.pages; }
    showPage(page);
    showTab("doc");
    openEvidence(true);
  }

  function renderSources() {
    const msg = state.messages[state.active];
    const box = $("#sources");
    const sources = msg?.sources || [];
    $("#src-count").textContent = sources.length;
    $("#query-line").hidden = !msg?.query || msg.query === state.messages[state.active - 1]?.content;
    $("#query-text").textContent = msg?.query || "";
    if (!sources.length) {
      box.innerHTML = `<div class="empty"><svg viewBox="0 0 48 48" aria-hidden="true"><rect x="10" y="6" width="28" height="36" rx="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 16h16M16 22h16M16 28h10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><p>${msg ? "Retrieving passages…" : "The handbook passages used for an answer appear here."}</p></div>`;
      return;
    }
    const max = Math.max(...sources.map((s) => s.score || 0), 1e-9);
    box.replaceChildren(...sources.map((s, i) => {
      const pages = (s.pages && s.pages.length ? s.pages : [s.page]).filter(Boolean);
      const label = pages.length > 1 ? `pp. ${pages[0]}–${pages[pages.length - 1]}` : `p. ${pages[0]}`;
      const body = (s.text || "").replace(/^Section:.*\n+/, "").replace(/^\(Text extracted by OCR[^\n]*\n/, "").replace(/\n{2,}/g, "\n").trim();
      const crumbs = (s.section || "").split(" > ").filter(Boolean).map((c) => `<span>${escapeHtml(c)}</span>`).join("");
      const card = el("article", "source");
      card.dataset.pages = pages.join(",");
      card.innerHTML = `
        <div class="source-head">
          <span class="source-idx">${i + 1}</span>
          <span class="source-page">${label}</span>
          ${s.ocr ? '<span class="badge badge-ocr" title="Read by OCR from an image-only page; may contain recognition errors">OCR</span>' : ""}
          <span class="source-score" title="Relevance (hybrid rank fusion)">${(s.score ?? 0).toFixed(2)}<span class="meter"><i style="width:${Math.round(100 * (s.score || 0) / max)}%"></i></span></span>
        </div>
        ${crumbs ? `<div class="crumbs">${crumbs}</div>` : ""}
        <div class="excerpt">${escapeHtml(body)}</div>
        <div class="source-actions">
          <button type="button" data-act="toggle">Show full passage</button>
          <button type="button" data-act="open">Open page ${pages[0]} ↗</button>
        </div>`;
      const ex = card.querySelector(".excerpt");
      requestAnimationFrame(() => {
        const clamped = ex.scrollHeight > ex.clientHeight + 2;
        ex.classList.toggle("is-clamped", clamped);
        if (!clamped) card.querySelector('[data-act="toggle"]').remove();
      });
      card.addEventListener("click", (e) => {
        const act = e.target.closest("button")?.dataset.act;
        if (act === "toggle") {
          card.classList.toggle("is-open");
          e.target.textContent = card.classList.contains("is-open") ? "Show less" : "Show full passage";
        } else if (act === "open") {
          openPage(pages[0], s.source);
        }
      });
      return card;
    }));
  }

  function highlightPage(page) {
    let hit = null;
    document.querySelectorAll(".source").forEach((c) => {
      const match = c.dataset.pages.split(",").map(Number).includes(page);
      c.classList.toggle("is-hit", match);
      if (match && !hit) hit = c;
    });
    return hit;
  }

  function setActive(index) {
    if (index !== state.active) $("#pane-sources").scrollTop = 0;
    state.active = index;
    document.querySelectorAll(".msg-assistant").forEach((n) => n.classList.toggle("is-active", +n.dataset.index === index));
    renderSources();
  }

  // ------------------------------------------------------------------ thread rendering
  const ICON_INFO = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 9v5M10 6.2v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';

  function renderAssistantBody(msg) {
    if (msg.error) return `<div class="error">⚠ ${escapeHtml(msg.error)}</div>`;
    if (!msg.content) {
      const n = msg.sources?.length;
      const steps = [
        [msg.history ? "Understanding the follow-up question" : "Reading the question", true],
        [n ? `Retrieved ${n} handbook passages` : "Searching the handbook", !!n],
        ["Writing the answer", false],
      ];
      const activeAt = n ? 2 : 1;
      return `<div class="progress">${steps.map(([t, done], i) =>
        `<div class="step ${done || i < activeAt ? "is-done" : i === activeAt ? "is-active" : ""}"><span class="tick"></span>${t}</div>`).join("")}</div>`;
    }
    if (msg.status === "done" && isRefusal(msg.content)) {
      return `<div class="refusal">${ICON_INFO}<div>${escapeHtml(msg.content)}<small>The retrieved passages do not answer this. Try rephrasing, or check the sources on the right.</small></div></div>`;
    }
    let html = renderMarkdown(msg.content);
    if (msg.status === "streaming") html = html.replace(/(<\/(?:p|li)>)(?:<\/[uo]l>)?$/, (m, close) => m.replace(close, `<span class="caret"></span>${close}`));
    return `<div class="answer">${html}</div>`;
  }

  function renderMeta(msg) {
    if (msg.status !== "done" || msg.error) return "";
    const lat = msg.meta?.latency_ms || {};
    const secs = ((lat.retrieval || 0) + (lat.generation || 0)) / 1000;
    return `<div class="msg-meta">
      <button type="button" class="link" data-act="sources">${msg.sources?.length || 0} sources</button>
      <span class="mono">${escapeHtml(msg.meta?.model || "")}</span>
      <span class="mono">${secs.toFixed(1)} s</span>
      ${msg.mode === "fallback" ? '<span class="badge badge-fallback">fallback</span>' : ""}
    </div>`;
  }

  function messageNode(msg, index) {
    if (msg.role === "user") {
      const n = el("div", "msg msg-user");
      n.innerHTML = `<div class="msg-label">Question</div><div class="msg-body">${escapeHtml(msg.content)}</div>`;
      return n;
    }
    const n = el("div", "msg msg-assistant");
    n.dataset.index = index;
    n.innerHTML = `<div class="msg-label">Answer</div><div class="msg-content">${renderAssistantBody(msg)}</div>${renderMeta(msg)}`;
    n.classList.toggle("is-active", index === state.active);
    return n;
  }

  function renderThread() {
    const thread = $("#thread");
    $("#welcome").hidden = state.messages.length > 0;
    thread.querySelectorAll(".msg").forEach((n) => n.remove());
    state.messages.forEach((m, i) => thread.appendChild(messageNode(m, i)));
    thread.scrollTop = thread.scrollHeight;
  }

  let frame = 0;
  function updateMessage(index) {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const node = document.querySelector(`.msg-assistant[data-index="${index}"]`);
      if (!node) return renderThread();
      const thread = $("#thread");
      const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
      node.replaceWith(messageNode(state.messages[index], index));
      if (nearBottom) thread.scrollTop = thread.scrollHeight;
    });
  }

  // ------------------------------------------------------------------ ask
  let controller = null;

  async function ask(question) {
    question = question.trim();
    if (!question || state.busy) return;
    state.busy = true;
    $("#btn-send").disabled = true;

    const history = state.messages
      .filter((m) => !m.error && m.content)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-6);
    state.messages.push({ role: "user", content: question });
    const msg = { role: "assistant", content: "", sources: [], query: "", status: "pending", mode: state.settings.mode, history: history.length > 0 };
    state.messages.push(msg);
    const index = state.messages.length - 1;
    state.active = index;
    $("#pane-sources").scrollTop = 0;
    renderThread();
    renderSources();
    showTab("sources");

    controller = new AbortController();
    try {
      const body = { question, history, ...state.settings };
      for await (const [event, data] of streamChat(body, controller.signal)) {
        if (event === "sources") {
          msg.sources = data.sources; msg.query = data.standalone_query;
          renderSources();
        } else if (event === "token") {
          msg.content += data; msg.status = "streaming";
        } else if (event === "done") {
          msg.content = data.answer; msg.sources = data.sources; msg.query = data.standalone_query;
          msg.meta = data; msg.status = "done";
        } else if (event === "error") {
          throw new Error(data.detail || "The server reported an error.");
        }
        updateMessage(index);
      }
      if (msg.status !== "done") { msg.status = "done"; updateMessage(index); }
    } catch (e) {
      if (e.name !== "AbortError") {
        msg.error = /Failed to fetch|NetworkError/.test(e.message)
          ? "Cannot reach the API. Is the backend running?"
          : e.message;
      }
      msg.status = "done";
      updateMessage(index);
    } finally {
      state.busy = false;
      controller = null;
      $("#btn-send").disabled = !$("#input").value.trim();
      renderSources();
    }
  }

  // ------------------------------------------------------------------ events
  function autoGrow() {
    const t = $("#input");
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 180)}px`;
    $("#btn-send").disabled = state.busy || !t.value.trim();
  }

  function bind() {
    $("#examples").replaceChildren(...EXAMPLES.map(([tag, q]) => {
      const b = el("button", "example", `<span class="tag">${tag}</span><span class="q">${escapeHtml(q)}</span>`);
      b.type = "button";
      b.addEventListener("click", () => ask(q));
      return b;
    }));

    $("#composer").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = $("#input").value;
      $("#input").value = ""; autoGrow();
      ask(q);
    });
    $("#input").addEventListener("input", autoGrow);
    $("#input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
    });

    $("#thread").addEventListener("click", (e) => {
      const cite = e.target.closest(".cite");
      const card = e.target.closest(".msg-assistant");
      if (!card) return;
      setActive(+card.dataset.index);
      if (cite) {
        const page = +cite.dataset.page;
        showTab("sources");
        openEvidence(true);
        const hit = highlightPage(page);
        if (hit) hit.scrollIntoView({ behavior: "smooth", block: "center" });
        else openPage(page);
      } else if (e.target.closest('[data-act="sources"]')) {
        showTab("sources"); openEvidence(true);
      }
    });

    $("#btn-new").addEventListener("click", () => {
      controller?.abort();
      state.messages = []; state.active = -1;
      renderThread(); renderSources(); showTab("sources");
      $("#input").focus();
    });

    $("#tab-sources").addEventListener("click", () => showTab("sources"));
    $("#tab-doc").addEventListener("click", () => {
      if (!$("#page-img").getAttribute("src")) openPage(1);
      else showTab("doc");
    });
    $("#pg-prev").addEventListener("click", () => showPage(doc.page - 1));
    $("#pg-next").addEventListener("click", () => showPage(doc.page + 1));
    $("#pg-input").addEventListener("change", (e) => showPage(+e.target.value || 1));
    $("#page-img").addEventListener("load", (e) => e.target.classList.remove("is-loading"));
    $("#page-img").addEventListener("error", () => { $("#doc-fallback").hidden = false; });
    $("#btn-evidence").addEventListener("click", () => openEvidence(!$("#evidence").classList.contains("is-open")));
    $("#btn-evidence-close").addEventListener("click", () => openEvidence(false));

    $("#btn-settings").addEventListener("click", () => openDrawer(true));
    $("#btn-settings-close").addEventListener("click", () => openDrawer(false));
    $("#scrim").addEventListener("click", () => openDrawer(false));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { openDrawer(false); openEvidence(false); } });

    $("#set-model").addEventListener("change", (e) => { state.settings.model = e.target.value; saveSettings(); });
    $("#set-topk").addEventListener("input", (e) => { state.settings.top_k = +e.target.value; saveSettings(); });
    $("#set-temp").addEventListener("input", (e) => { state.settings.temperature = +e.target.value; saveSettings(); });
    document.querySelectorAll("#mode-group button").forEach((b) =>
      b.addEventListener("click", () => { state.settings.mode = b.dataset.mode; saveSettings(); }));
    document.querySelectorAll("#theme-group button").forEach((b) =>
      b.addEventListener("click", () => applyTheme(b.dataset.themeChoice, true)));

    $("#btn-reindex").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true; btn.textContent = "Indexing…";
      $("#reindex-msg").textContent = "This can take a minute when the index is rebuilt.";
      try {
        const r = await getJSON("/index", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ force: false }) });
        $("#reindex-msg").textContent = r.rebuilt ? `Rebuilt: ${r.chunks} chunks.` : `Already up to date (${r.chunks} chunks).`;
      } catch (err) {
        $("#reindex-msg").textContent = `Indexing failed: ${err.message}`;
      } finally {
        btn.disabled = false; btn.textContent = "Rebuild index";
        refreshHealth();
      }
    });
  }

  // ------------------------------------------------------------------ start
  async function init() {
    bind();
    applySettingsToUI();
    autoGrow();
    const health = await refreshHealth();
    if (health?.defaults && !store.get("hb.settings", null)) {
      Object.assign(state.settings, {
        mode: health.defaults.mode || "strict", top_k: health.defaults.top_k || 5, temperature: health.defaults.temperature ?? 0.1,
      });
      applySettingsToUI();
    }
    await loadModels(health?.defaults);
    try { state.docs = (await getJSON("/documents")).documents; } catch { state.docs = []; }
    if (state.docs[0]) { $("#pg-total").textContent = state.docs[0].pages; $("#pg-input").max = state.docs[0].pages; }
    setInterval(refreshHealth, 20000);
    $("#input").focus();
  }

  init();
})();
