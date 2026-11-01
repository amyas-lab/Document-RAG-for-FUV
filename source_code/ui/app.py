"""Streamlit chat client for the Applied Mathematics handbook assistant.

Pure client: every request goes to the FastAPI backend at API_URL.
Run locally:  streamlit run ui/app.py      (from source_code/, with the API on :8000)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "What do I need to do before I can declare the Applied Mathematics major?",
    "What are the requirements for a minor in Applied Mathematics?",
    "What are the prerequisites for MATH 305 Stochastic Calculus?",
    "Which courses from other majors are cross-listed with Applied Mathematics?",
    "Which MATH courses are cross-listed with Economics?",
    "What is MLO 4 of the Applied Mathematics major?",
]

st.set_page_config(page_title="Applied Math Handbook Assistant", page_icon="📐", layout="wide")


# --------------------------------------------------------------------------- API helpers
@st.cache_data(ttl=10, show_spinner=False)
def get_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_models() -> tuple[list[str], str | None]:
    try:
        data = requests.get(f"{API_URL}/models", timeout=5).json()
        return data.get("models", []), data.get("default")
    except (requests.RequestException, ValueError):
        return [], None


def stream_chat(body: dict) -> Iterator[tuple[str, dict | str]]:
    """Parse the Server-Sent Events from POST /chat/stream."""
    with requests.post(f"{API_URL}/chat/stream", json=body, stream=True, timeout=(5, 600)) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                yield event, json.loads(line[5:].strip())
                event = None


def render_sources(sources: list[dict], key: str) -> None:
    if not sources:
        return
    pages = sorted({p for s in sources for p in (s.get("pages") or [s.get("page")]) if p})
    with st.expander(f"📄 Sources — {len(sources)} excerpts (pages {', '.join(map(str, pages))})"):
        for i, s in enumerate(sources, 1):
            pg = s.get("pages") or [s.get("page")]
            label = f"p. {pg[0]}" if len(pg) == 1 else f"pp. {pg[0]}–{pg[-1]}"
            st.markdown(
                f"**[{i}] {label}** · score `{s.get('score', 0):.3f}`"
                + (" · 🖼️ OCR" if s.get("ocr") else "")
                + (f"  \n<small>{s.get('section')}</small>" if s.get("section") else ""),
                unsafe_allow_html=True,
            )
            st.code(s.get("text", ""), language=None, wrap_lines=True)


# --------------------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources?, meta?}]
if "pending" not in st.session_state:
    st.session_state.pending = None

health = get_health()
models, default_model = get_models()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📐 Handbook Assistant")
    st.caption("B.S. Applied Mathematics · Fulbright University Vietnam")

    if st.button("➕ New chat", width="stretch"):
        st.session_state.messages, st.session_state.pending = [], None
        st.rerun()

    st.subheader("Settings")
    defaults = (health or {}).get("defaults", {})
    model = st.selectbox(
        "LLM model", models or [default_model or "qwen2.5:3b"],
        index=(models.index(default_model) if default_model in models else 0),
    )
    top_k = st.slider("Top-K excerpts", 1, 10, int(defaults.get("top_k", 5)))
    temperature = st.slider("Temperature", 0.0, 1.0, float(defaults.get("temperature", 0.1)), 0.05)
    mode = st.radio(
        "Answer mode", ["strict", "fallback"], horizontal=True,
        help="strict: answer only from the handbook and refuse otherwise (recommended). "
             "fallback: baseline prompt that may use the model's general knowledge.",
    )
    show_query = st.toggle("Show rewritten query", value=False)

    st.subheader("Status")
    if health is None:
        st.error(f"API not reachable at {API_URL}")
    else:
        st.markdown(
            f"{'🟢' if health['ollama'] else '🔴'} Ollama &nbsp; "
            f"{'🟢' if health['qdrant'] else '🔴'} Qdrant ({health.get('qdrant_mode')})"
        )
        st.caption(f"Index: **{health['n_chunks']}** chunks · `{health['collection']}`")
        if not health.get("index_up_to_date"):
            st.warning("Index missing or outdated.")
        if st.button("Rebuild index", width="stretch"):
            with st.spinner("Indexing… (first run embeds every chunk)"):
                try:
                    r = requests.post(f"{API_URL}/index", json={"force": False}, timeout=900).json()
                    st.success(f"{r.get('chunks')} chunks ({'rebuilt' if r.get('rebuilt') else 'up to date'})")
                except requests.RequestException as e:
                    st.error(str(e))
            get_health.clear()

# --------------------------------------------------------------------------- chat
st.header("Ask the Applied Mathematics Major Handbook")

if not st.session_state.messages and not st.session_state.pending:
    st.markdown("Answers come **only** from the handbook, with page citations. Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", width="stretch"):
            st.session_state.pending = q
            st.rerun()

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "📐"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if show_query and m.get("meta", {}).get("standalone_query"):
                st.caption(f"🔎 Search query: {m['meta']['standalone_query']}")
            render_sources(m.get("sources", []), key=f"s{i}")
            if m.get("meta"):
                st.caption(m["meta"]["footer"])

question = st.chat_input("Ask about requirements, courses, prerequisites, MLOs…") or st.session_state.pending
if question:
    st.session_state.pending = None
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📐"):
        placeholder = st.empty()
        placeholder.markdown("_Searching the handbook…_")
        body = {"question": question, "history": history, "model": model,
                "top_k": top_k, "temperature": temperature, "mode": mode}
        text, sources, result, t0 = "", [], {}, time.perf_counter()
        try:
            for event, data in stream_chat(body):
                if event == "sources":
                    sources = data["sources"]
                elif event == "token":
                    text += data
                    placeholder.markdown(text + "▌")
                elif event == "done":
                    result = data
                elif event == "error":
                    raise RuntimeError(data.get("detail"))
            text = result.get("answer", text)
            placeholder.markdown(text)
        except Exception as e:  # noqa: BLE001 - show any backend failure in the chat
            text = f"⚠️ Request failed: {e}"
            placeholder.error(text)

        lat = result.get("latency_ms", {})
        meta = {
            "standalone_query": result.get("standalone_query"),
            "footer": f"⏱️ {time.perf_counter() - t0:.1f}s total · retrieval {lat.get('retrieval', 0)} ms · "
                      f"generation {lat.get('generation', 0) / 1000:.1f}s · {result.get('model', model)} · {mode}",
        } if result else {}
        if show_query and meta.get("standalone_query"):
            st.caption(f"🔎 Search query: {meta['standalone_query']}")
        render_sources(sources, key="live")
        if meta:
            st.caption(meta["footer"])
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources, "meta": meta})
