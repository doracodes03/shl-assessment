import subprocess
import sys
import time
from pathlib import Path
from typing import List

import requests
import streamlit as st

st.set_page_config(page_title="SHL Recommendation Chat", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
API_URL = st.secrets.get("backend_url") or "http://127.0.0.1:8000/chat"
HEALTH_URL = API_URL.replace("/chat", "/health")


def ensure_backend_running() -> bool:
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.ok:
            return True
    except Exception:
        pass

    try:
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        return False

    for _ in range(20):
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            if resp.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "compare_ids" not in st.session_state:
    st.session_state.compare_ids = set()

backend_ready = ensure_backend_running()


def normalize_recommendations(recs: List[dict]) -> List[dict]:
    normalized = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "entity_id": r.get("entity_id") or r.get("id") or r.get("entityId"),
                "name": r.get("name") or r.get("title") or "",
                "url": r.get("url") or r.get("link") or r.get("href"),
                "description": r.get("description") or r.get("desc") or "",
                "duration": r.get("duration") or r.get("duration_raw") or "",
                "keys": r.get("keys") or [],
            }
        )
    return normalized


def send_query(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    payload = {"messages": st.session_state.messages}
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            st.session_state.messages.append({"role": "assistant", "content": f"Backend error: {exc}"})
            st.session_state.recommendations = []
            return

    assistant_text = (data.get("reply") or data.get("assistant") or "").strip()
    if assistant_text:
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    recs = data.get("recommendations") or []
    st.session_state.recommendations = normalize_recommendations(recs)


st.title("SHL Assessment Recommendation")
st.caption("A compact, chat-style assistant for finding SHL assessment recommendations.")

if not backend_ready:
    st.warning("The backend could not be reached automatically. Make sure the FastAPI app is running on port 8000.")

chat_col, info_col = st.columns([2.3, 1.2], gap="large")

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question or describe the hiring need")
    if prompt:
        send_query(prompt)
        st.rerun()

    st.markdown("---")
    st.caption("The full conversation history is sent to the backend on every request.")

with info_col:
    st.header("Recommendations")
    if not st.session_state.recommendations:
        st.info("No recommendations yet. Ask a role-based question and the agent will return grounded SHL cards.")
    else:
        for i, rec in enumerate(st.session_state.recommendations):
            with st.container():
                st.markdown(f"### {rec.get('name')}")
                if rec.get("description"):
                    st.markdown(rec.get("description"))
                meta = []
                if rec.get("duration"):
                    meta.append(f"**Duration:** {rec.get('duration')}")
                if rec.get("keys"):
                    meta.append(f"**Tags:** {', '.join(rec.get('keys'))}")
                if meta:
                    st.markdown(" | ".join(meta))
                if rec.get("url"):
                    st.markdown(f"[Open SHL product page]({rec.get('url')})")
                cid = rec.get("entity_id") or rec.get("name")
                checked = cid in st.session_state.compare_ids
                if st.checkbox("Compare", key=f"cmp_{i}", value=checked):
                    st.session_state.compare_ids.add(cid)
                else:
                    st.session_state.compare_ids.discard(cid)
                st.markdown("---")

    if st.session_state.compare_ids:
        selected = [
            rec for rec in st.session_state.recommendations
            if (rec.get("entity_id") or rec.get("name")) in st.session_state.compare_ids
        ]
        if len(selected) >= 2:
            st.subheader("Comparison")
            lines = ["| Assessment | Duration | Tags | Link |", "| --- | --- | --- | --- |"]
            for rec in selected:
                tags = ", ".join(rec.get("keys") or [])
                link = f"[{rec.get('name')}]({rec.get('url')})" if rec.get("url") else rec.get("name")
                lines.append(f"| {rec.get('name')} | {rec.get('duration') or '-'} | {tags or '-'} | {link} |")
            st.markdown("\n".join(lines))

st.markdown("---")
st.caption(f"Backend URL: {API_URL}")
