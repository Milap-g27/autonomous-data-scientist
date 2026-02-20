"""
Autonomous Data Scientist Agent — Streamlit app entry point.

Architecture:
  1. Upload    → render_upload_section()
  2. Config    → render_config_section()
  3. Run       → render_run_button()
  4. Dashboard → render_dashboard()
  5. Chatbot   → render_chatbot()  (sidebar)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    layout="wide",
    page_title="Autonomous Data Scientist Agent",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
)

# ── UI modules ──
from ui.styles import inject_global_css, scroll_to_dashboard
from ui.components import (
    render_header,
    render_upload_section,
    render_config_section,
    render_run_button,
    render_dashboard,
)
from ui.chatbot import render_chatbot


def main():
    """Top-level app layout."""
    # 1 — Inject CSS
    inject_global_css()

    # 2 — Gradient Header
    render_header()

    # 3 — Upload
    df = render_upload_section()

    if df is not None:
        # 4 — Config
        config = render_config_section(df)

        # 5 — Run
        render_run_button(df, config)

        # 6 — Dashboard (only when analysis is done)
        if st.session_state.get("analysis_done", False):
            render_dashboard()
            scroll_to_dashboard()

    # 7 — Floating AI Assistant (sidebar chatbot — always available)
    render_chatbot()


if __name__ == "__main__":
    main()
