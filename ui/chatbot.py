"""
Floating AI Chatbot — session-aware assistant restricted to dataset/model topics.
Uses ChatGroq (Llama 3.3 70b) for responses, scoped to session context only.
"""
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import settings


_REJECTION = "I can only answer questions related to the current dataset and model analysis."

_SYSTEM_TEMPLATE = """You are an expert data science assistant embedded in the Autonomous Data Scientist application.
You answer questions ONLY about:
- The uploaded dataset (columns, types, distributions)
- Feature engineering and data cleaning steps performed
- Model performance and evaluation metrics
- The AI explanation generated for this analysis
- General machine learning theory and best practices

If the user asks anything unrelated (e.g., weather, politics, coding help outside ML), respond EXACTLY with:
"{rejection}"

Here is the current session context:

### Dataset Summary
{dataset_summary}

### Model Configuration
{model_config}

### Model Results
{model_results}

### AI Explanation
{ai_explanation}

### Data Cleaning Report (Processing Log)
{cleaning_report}

### Feature Engineering Report (Processing Log)
{feature_report}
"""


def _get_system_prompt() -> str:
    """Build system prompt from session state context."""
    dataset_summary = st.session_state.get("dataset_summary", "No dataset uploaded yet.")
    model_config = str(st.session_state.get("model_config", "Not configured yet."))
    model_results = str(st.session_state.get("model_results", "No results yet."))
    ai_explanation = st.session_state.get("ai_explanation", "No explanation generated yet.")
    cleaning_report = st.session_state.get("cleaning_report", "No cleaning report yet.")
    feature_report = st.session_state.get("feature_report", "No feature engineering report yet.")

    return _SYSTEM_TEMPLATE.format(
        rejection=_REJECTION,
        dataset_summary=dataset_summary,
        model_config=model_config,
        model_results=model_results,
        ai_explanation=ai_explanation,
        cleaning_report=cleaning_report,
        feature_report=feature_report,
    )


def render_chatbot():
    """Render the floating chatbot FAB and modal."""
    # Chatbot toggle button (CSS-styled as a floating action button)
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Toggle visibility
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
    """Call ChatGroq with session context to answer the user's question."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return "⚠️ GROQ_API_KEY not configured. Please set it in your `.env` file."

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=1024,
        )

        # Build messages
        messages: list[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=_get_system_prompt()),
        ]

        # Include recent history for context (last 6 exchanges max)
        recent_history = st.session_state["chat_history"][-12:]
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Add the current message
        messages.append(HumanMessage(content=user_message))

        response = llm.invoke(messages)
        return str(response.content)

    except Exception as e:
        return f"⚠️ Error: {e}"
