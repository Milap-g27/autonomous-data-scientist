"""
UI Styles — CSS injection for the Autonomous Data Scientist app.
Animated header, metric cards, tab hover, chatbot button, and auto-scroll.
"""
import streamlit as st


def inject_global_css():
    """Inject all custom CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def scroll_to_dashboard():
    """Inject JS to auto-scroll to the dashboard anchor after run completes."""
    st.markdown(
        '<script>document.getElementById("dashboard-anchor")?.scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True,
    )


_CSS = """
<style>
/* ───────── Animated Gradient Header ───────── */
.gradient-header {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #667eea);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    color: #fff;
}
.gradient-header h1 {
    margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
}
.gradient-header p {
    margin: 0.3rem 0 0; opacity: 0.92; font-size: 1.05rem;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ───────── Metric Cards ───────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.22);
}

/* ───────── Tab hover ───────── */
button[data-baseweb="tab"] {
    transition: all 0.25s ease;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"]:hover {
    filter: brightness(1.15);
    border-bottom: 2px solid #667eea !important;
}

/* ───────── Styled container (cards) ───────── */
.styled-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
    transition: box-shadow 0.25s ease;
}
.styled-card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.18);
}

/* ───────── Dataset info card ───────── */
.dataset-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ───────── Floating Chat Button ───────── */
.chat-fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(102,126,234,0.45);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
}
.chat-fab:hover { transform: scale(1.08); }

/* ───────── Chatbot modal ───────── */
.chat-modal {
    position: fixed;
    bottom: 5.5rem;
    right: 2rem;
    width: 380px;
    max-height: 520px;
    background: var(--background-color, #0e1117);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    z-index: 9998;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.chat-modal-header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 0.8rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 16px 16px 0 0;
}
.chat-modal-body {
    flex: 1;
    overflow-y: auto;
    padding: 0.8rem 1rem;
    max-height: 370px;
}

/* ───────── Streamlit keyed floating chat widgets ───────── */
.st-key-chat_fab_btn {
    position: fixed !important;
    right: 2rem !important;
    bottom: 2rem !important;
    left: auto !important;
    width: 60px !important;
    z-index: 10000;
}

.st-key-chat_fab_btn button {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    border: none !important;
    padding: 0 !important;
    min-height: 0 !important;
    font-size: 1.5rem !important;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.5);
    transition: transform 0.2s, box-shadow 0.2s;
}
.st-key-chat_fab_btn button:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 28px rgba(102,126,234,0.65);
}

.st-key-chat_modal_panel {
    position: fixed;
    right: 1.5rem;
    bottom: 5.8rem;
    width: min(420px, calc(100vw - 2rem));
    max-height: 70vh;
    overflow-y: auto;
    z-index: 9999;
    padding: 0.8rem 0.9rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(14,17,23,0.96);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

.st-key-chat_modal_panel [data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    background: rgba(14,17,23,0.96);
    padding-top: 0.3rem;
}

/* ───────── Upload area ───────── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102,126,234,0.4);
    border-radius: 14px;
    padding: 0.5rem;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(102,126,234,0.8);
}
</style>
"""
