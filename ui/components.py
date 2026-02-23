"""
UI Components — Streamlit frontend that talks to the FastAPI backend via HTTP.
All ML logic runs on the API; this file only renders UI and makes HTTP calls.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import requests
import base64
import io
import time
import os

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")


# ─────────────────────────────────────────────
# SECTION 1 — Header
# ─────────────────────────────────────────────

def render_header():
    """Render the animated gradient header."""
    st.markdown(
        '<div class="gradient-header">'
        "<h1>🤖 Autonomous Data Scientist Agent</h1>"
        "<p>Upload a dataset — the agent cleans, explores, models, and explains automatically.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# SECTION 2 — Upload
# ─────────────────────────────────────────────

def render_upload_section() -> pd.DataFrame | None:
    """Render file uploader → POST /api/upload → show dataset preview."""
    uploaded_file = st.file_uploader(
        "Drag & drop your CSV dataset here",
        type=["csv"],
        help="Supported format: .csv",
    )
    if uploaded_file is None:
        return None

    # Only call the API once per file (cache by name+size)
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_upload_key") != file_key:
        uploaded_file.seek(0)
        resp = requests.post(
            f"{API_BASE}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
        )
        if resp.status_code != 200:
            st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            return None

        data = resp.json()
        st.session_state["session_id"] = data["session_id"]
        st.session_state["dataset_info"] = data["dataset_info"]
        st.session_state["_upload_key"] = file_key

        # Keep a local copy for preview / predictions form
        uploaded_file.seek(0)
        st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)

    info = st.session_state["dataset_info"]
    df = st.session_state["uploaded_df"]

    # Dataset info badges
    cols = st.columns(4)
    badges = [
        (f"📐 {info['rows']:,} rows", "Rows"),
        (f"📊 {info['columns']} columns", "Columns"),
        (f"🔢 {len(info['numeric_columns'])} numeric", "Numeric"),
        (f"🏷️ {len(info['categorical_columns'])} categorical", "Categorical"),
    ]
    for col, (label, _) in zip(cols, badges):
        col.markdown(f'<span class="dataset-badge">{label}</span>', unsafe_allow_html=True)

    with st.container():
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    return df


# ─────────────────────────────────────────────
# SECTION 3 — Config
# ─────────────────────────────────────────────

def render_config_section(df: pd.DataFrame) -> dict:
    """Render target selection and advanced settings → POST /api/configure."""
    columns = df.columns.tolist()
    target_options = ["None (Clustering)"] + columns

    c1, c2 = st.columns([2, 1])
    with c1:
        target_selection = st.selectbox(
            "🎯 Select Target Column",
            target_options,
            index=1,
            help="Choose the column to predict, or select None for clustering.",
        )
    with c2:
        problem_hint = st.selectbox(
            "📑 Problem Type",
            ["Auto-detect", "Classification", "Regression", "Clustering"],
            index=0,
            help="Auto-detect lets the LLM decide based on the target column.",
        )

    target = None if target_selection == "None (Clustering)" else target_selection
    if target is None:
        problem_hint = "Clustering"

    # Advanced settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
        with adv2:
            test_size = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100
        with adv3:
            scaling = st.toggle("Apply Feature Scaling", value=True)

        feat_selection = st.toggle(
            "Feature Selection (Auto)", value=False,
            help="Let the pipeline auto-select features.",
        )

    config = {
        "target": target,
        "problem_hint": problem_hint,
        "random_seed": int(random_seed),
        "test_size": test_size,
        "scaling": scaling,
        "feature_selection": feat_selection,
    }
    st.session_state["model_config"] = config
    return config


# ─────────────────────────────────────────────
# SECTION 4 — Run
# ─────────────────────────────────────────────

def render_run_button(df: pd.DataFrame, config: dict):
    """Run button → POST /api/configure then POST /api/analyze."""
    is_running = st.session_state.get("is_running", False)
    session_id = st.session_state.get("session_id")

    if not session_id:
        st.warning("No session. Please re-upload your dataset.")
        return

    if st.button(
        "🚀 Run AI Data Scientist",
        use_container_width=True,
        type="primary",
        disabled=is_running,
    ):
        st.session_state["is_running"] = True

        # 1 — Configure
        cfg_payload = {"session_id": session_id, **config}
        cfg_resp = requests.post(f"{API_BASE}/configure", json=cfg_payload)
        if cfg_resp.status_code != 200:
            st.error(f"Config error: {cfg_resp.json().get('detail', cfg_resp.text)}")
            st.session_state["is_running"] = False
            return

        # 2 — Analyze
        with st.spinner("🔄 Agent is working — analyzing, modeling, explaining…"):
            try:
                analyze_resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"session_id": session_id},
                    timeout=600,
                )

                if analyze_resp.status_code != 200:
                    st.error(f"Analysis error: {analyze_resp.json().get('detail', analyze_resp.text)}")
                    st.session_state["is_running"] = False
                    return

                result = analyze_resp.json()

                # Persist to session state
                st.session_state["result"] = result
                st.session_state["target"] = config["target"]
                st.session_state["df"] = df
                st.session_state["training_time"] = result.get("training_time", 0)
                st.session_state["model_results"] = result.get("metrics", {})
                st.session_state["ai_explanation"] = result.get("explanation", "")
                st.session_state["cleaning_report"] = result.get("cleaning_report", "")
                st.session_state["feature_report"] = result.get("feature_report", "")
                st.session_state["analysis_done"] = True

                elapsed = result.get("training_time", 0)
                st.success(f"✅ Analysis complete in **{elapsed}s**!")
            except requests.exceptions.Timeout:
                st.error("Analysis timed out. Try a smaller dataset.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                st.session_state["is_running"] = False


# ─────────────────────────────────────────────
# SECTION 5 — Dashboard
# ─────────────────────────────────────────────

def render_dashboard():
    """Render the full results dashboard from API response stored in session."""
    result = st.session_state["result"]
    target = st.session_state["target"]
    df_original = st.session_state["df"]
    problem_type = result.get("problem_type", "")

    st.markdown('<div id="dashboard-anchor"></div>', unsafe_allow_html=True)
    st.markdown("---")

    # Metric cards
    _render_metric_cards(result, problem_type)

    # Tabs
    tabs = st.tabs([
        "📊 EDA & Insights",
        "📈 Plots & Visualizations",
        "🏆 Model Performance",
        "🔮 Predictions",
        "📝 AI Explanation",
        "⚙️ Processing Log",
    ])

    with tabs[0]:
        _render_eda_tab(result, df_original, problem_type)
    with tabs[1]:
        _render_plots_tab(result, df_original, problem_type)
    with tabs[2]:
        _render_model_tab(result, df_original, target, problem_type)
    with tabs[3]:
        _render_predictions_tab(result, df_original, target, problem_type)
    with tabs[4]:
        _render_explanation_tab(result)
    with tabs[5]:
        _render_log_tab(result)


# ── Metric Cards ────────────────────────────

def _render_metric_cards(result: dict, problem_type: str):
    metrics = result.get("metrics", {})
    best_name = result.get("best_model", "N/A")
    best_metrics = metrics.get(best_name, {})
    elapsed = st.session_state.get("training_time", 0)

    if problem_type == "Clustering":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Algorithm", best_name)
        c2.metric("Silhouette", best_metrics.get("Silhouette Score", "N/A"))
        c3.metric("Calinski-Harabasz", best_metrics.get("Calinski-Harabasz", "N/A"))
        c4.metric("⏱️ Training Time", f"{elapsed}s")
    elif problem_type == "Regression":
        c1, c2, c3, c4 = st.columns(4)
        r2 = best_metrics.get("R2", 0)
        mae = best_metrics.get("MAE", 0)
        c1.metric("Best Model", best_name)
        c2.metric("R² Score", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
        c3.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else mae)
        c4.metric("⏱️ Training Time", f"{elapsed}s")
    else:
        c1, c2, c3, c4 = st.columns(4)
        acc = best_metrics.get("Accuracy", 0)
        f1 = best_metrics.get("F1 Score", 0)
        c1.metric("Best Model", best_name)
        c2.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (int, float)) else acc)
        c3.metric("F1 Score", f"{f1:.4f}" if isinstance(f1, (int, float)) else f1)
        c4.metric("⏱️ Training Time", f"{elapsed}s")

    st.markdown("")


# ── Model Comparison Chart ──────────────────

def _render_model_comparison_chart(metrics: dict, problem_type: str):
    if not metrics:
        return

    df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

    if problem_type == "Clustering":
        score_col = "Silhouette Score"
    elif problem_type == "Regression":
        score_col = "R2"
    else:
        score_col = "F1 Score"
        if score_col not in df_m.columns:
            score_col = "Accuracy"

    if score_col not in df_m.columns:
        return

    df_m = df_m[df_m[score_col].apply(lambda x: isinstance(x, (int, float)))].copy()
    if df_m.empty:
        return

    df_m = df_m.sort_values(score_col, ascending=True)
    fig = px.bar(
        df_m, x=score_col, y="Model", orientation="h",
        title=f"Model Comparison — {score_col}",
        color=score_col, color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=max(350, len(df_m) * 30 + 120),
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis_title="", xaxis_title=score_col, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── EDA Tab ─────────────────────────────────

def _render_eda_tab(result: dict, df_original: pd.DataFrame, problem_type: str):
    st.header("Exploratory Data Analysis")

    eda = result.get("eda_results", {})
    if eda and "description" in eda:
        st.subheader("📋 Summary Statistics")
        summary_df = pd.DataFrame(eda["description"]).T
        st.table(summary_df)

    st.subheader("🔍 Missing Values & Unique Counts")
    info_data = {}
    for col in df_original.columns:
        info_data[col] = {
            "Missing Values": int(df_original[col].isnull().sum()),
            "Missing %": round(df_original[col].isnull().sum() / len(df_original) * 100, 2),
            "Unique Values": int(df_original[col].nunique()),
            "Data Type": str(df_original[col].dtype),
        }
    st.table(pd.DataFrame(info_data).T)


def _render_plots_tab(result: dict, df_original: pd.DataFrame, problem_type: str):
    st.header("Plots & Visualizations")

    # EDA figures from API (base64 images)
    for fig_data in result.get("eda_figures", []):
        heading = fig_data.get("heading", "")
        desc = fig_data.get("description", "")
        img_b64 = fig_data.get("image_base64", "")

        if heading:
            st.subheader(heading)
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(img_bytes, use_container_width=True)
                if desc:
                    st.caption(desc)
            st.markdown("---")

    # Categorical distributions (local — no heavy ML)
    _render_categorical_distributions(df_original)


def _render_categorical_distributions(df_original: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("📊 Distribution of Categorical")
    dist_cols = [c for c in df_original.columns if df_original[c].nunique() <= 5]

    if not dist_cols:
        st.info("No columns with 5 or fewer unique values found.")
        return

    for feat in dist_cols:
        col_data = df_original[feat]
        fig_d, ax_d = plt.subplots(figsize=(6, 4))
        if pd.api.types.is_numeric_dtype(col_data):
            sns.histplot(col_data, kde=False, ax=ax_d, color="#2196F3", discrete=True)
            ax_d.set_title(f"Distribution of {feat}")
            desc_d = f"Bar chart showing the distribution of '{feat}' ({col_data.nunique()} unique values)."
        else:
            sns.countplot(y=col_data, ax=ax_d, color="#2196F3")
            ax_d.set_title(f"Count Plot of {feat}")
            desc_d = f"Count plot showing the frequency of each category in '{feat}' ({col_data.nunique()} unique values)."
        plt.tight_layout()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig_d)
            st.caption(desc_d)
        st.markdown("---")
        plt.close(fig_d)


# ── Model Performance Tab ───────────────────

def _render_model_tab(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    st.header("Model Evaluation")
    metrics = result.get("metrics", {})

    _render_model_comparison_chart(metrics, problem_type)

    if metrics:
        st.subheader("📊 Full Metrics Table")
        st.table(pd.DataFrame(metrics).T)

    st.success(f"**Best Performing Model:** {result.get('best_model', 'N/A')}")


def _render_predictions_tab(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    st.header("Predict with Best Model")
    if problem_type == "Clustering":
        st.info("Prediction is not applicable for clustering tasks.")
        return
    _render_prediction_form(result, df_original, target, problem_type)


def _render_prediction_form(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    best_model_name = result.get("best_model", "")
    session_id = st.session_state.get("session_id")

    if not best_model_name or not session_id:
        st.warning("Model not available for prediction.")
        return

    @st.fragment
    def prediction_fragment():
        st.markdown(
            f'<div class="styled-card">'
            f'<p>Fill in the feature values below and click <strong>Predict</strong> '
            f'to get a prediction from <strong>{best_model_name}</strong>.</p></div>',
            unsafe_allow_html=True,
        )

        original_cols = [c for c in df_original.columns if c != target]

        col_left, col_form, col_right = st.columns([1, 2, 1])

        with col_form:
            with st.form("prediction_form"):
                input_data = {}
                for col_name in original_cols:
                    if df_original[col_name].dtype == "object" or df_original[col_name].dtype.name == "category":
                        unique_vals = sorted(df_original[col_name].dropna().unique().tolist())
                        input_data[col_name] = st.selectbox(f"{col_name}", options=unique_vals)
                    elif df_original[col_name].dtype == "bool":
                        input_data[col_name] = st.selectbox(f"{col_name}", options=[True, False])
                    elif pd.api.types.is_integer_dtype(df_original[col_name]):
                        min_val = int(df_original[col_name].min())
                        max_val = int(df_original[col_name].max())
                        median_val = int(df_original[col_name].median())
                        input_data[col_name] = st.number_input(
                            f"{col_name}", min_value=min_val, max_value=max_val,
                            value=median_val, step=1,
                        )
                    elif pd.api.types.is_float_dtype(df_original[col_name]):
                        min_val = float(df_original[col_name].min())
                        max_val = float(df_original[col_name].max())
                        median_val = float(df_original[col_name].median())
                        input_data[col_name] = st.number_input(
                            f"{col_name}", min_value=min_val, max_value=max_val,
                            value=median_val, step=0.01, format="%.2f",
                        )
                    else:
                        input_data[col_name] = st.number_input(
                            f"{col_name}", value=float(df_original[col_name].median()),
                            step=0.01, format="%.2f",
                        )

                submitted = st.form_submit_button(
                    "🔮 Predict", use_container_width=True, type="primary",
                )

        if submitted:
            try:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    json={"session_id": session_id, "input_data": input_data},
                    timeout=120,
                )
                if resp.status_code != 200:
                    with col_form:
                        st.error(f"Prediction error: {resp.json().get('detail', resp.text)}")
                    return

                pred = resp.json()
                predicted_value = pred["predicted_value"]
                model_used = pred["model_used"]

                with col_form:
                    st.markdown(
                        f'<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); '
                        f'border-radius: 12px; padding: 2rem; text-align: center; '
                        f'border: 1px solid rgba(255,255,255,0.1); margin-top: 1rem;">'
                        f'<p style="color: #aaa; margin-bottom: 0.5rem; font-size: 0.9rem;">Predicted Value</p>'
                        f'<h2 style="color: #4CAF50; margin: 0;">{target}</h2>'
                        f'<h1 style="color: #fff; margin-top: 0.5rem; font-size: 2.5rem;">{predicted_value}</h1>'
                        f'<p style="color: #888; font-size: 0.8rem; margin-top: 1rem;">'
                        f'Model: {model_used}</p></div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                with col_form:
                    st.error(f"Prediction error: {e}")

    prediction_fragment()


# ── AI Explanation Tab ──────────────────────

def _render_explanation_tab(result: dict):
    st.header("AI Explanation")
    explanation = result.get("explanation", "No explanation generated.")
    st.markdown(explanation)


# ── Processing Log Tab ─────────────────────

def _render_log_tab(result: dict):
    st.header("Processing Logs")

    def _to_bullet_list(text: str) -> str:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            return text
        return "\n\n".join(f"- {line}" for line in lines)

    st.subheader("Data Cleaning Report")
    st.markdown(_to_bullet_list(result.get("cleaning_report", "No report")))

    st.subheader("Feature Engineering Report")
    st.markdown(_to_bullet_list(result.get("feature_report", "No report")))
