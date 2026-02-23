"""
Floating AI Chatbot — session-aware assistant that talks to the FastAPI backend.
All LLM calls happen server-side via POST /api/chat.
"""
import os
import streamlit as st
import requests

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")


def render_chatbot():
    """Render the floating chatbot FAB and modal."""
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    _render_chat_toggle()

    if st.session_state["chat_open"]:
        _render_chat_modal()


def _render_chat_toggle():
    """Render the floating chat button."""
    with st.container(key="chat_fab_btn"):
        icon = "✖" if st.session_state["chat_open"] else "💬"
        if st.button(icon, key="chat_toggle_button", help="Open AI Assistant"):
            st.session_state["chat_open"] = not st.session_state["chat_open"]


@st.fragment
def _render_chat_modal():
    """Render the floating chat interface in a fixed panel."""
    with st.container(key="chat_modal_panel"):
        head_left, head_right = st.columns([5, 1])
        with head_left:
            st.markdown("### 🤖 AI Assistant")
        with head_right:
            if st.button("✖", key="chat_close_button", help="Close chat"):
                st.session_state["chat_open"] = False
                st.rerun()

        st.caption("Ask about your dataset, models, or ML concepts.")

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask about your data or models…", key="floating_chat_input")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            response = _get_chat_response(user_input)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.rerun()


def _get_chat_response(user_message: str) -> str:
    """Call POST /api/chat on the FastAPI backend."""
    session_id = st.session_state.get("session_id")
    if not session_id:
        return "⚠️ No active session. Please upload a dataset first."

    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"session_id": session_id, "message": user_message},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("reply", "No reply received.")
        else:
            detail = resp.json().get("detail", resp.text)
            return f"⚠️ Error: {detail}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach the API server. Make sure `uvicorn main:app` is running on port 8000."
    except Exception as e:
        return f"⚠️ Error: {e}"
