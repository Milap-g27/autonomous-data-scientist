"""
UI Components — reusable rendering functions for the Autonomous Data Scientist app.
Keeps all visual rendering separated from ML/pipeline logic.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import time


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
    """Render the file uploader and dataset preview. Returns the dataframe or None."""
    uploaded_file = st.file_uploader(
        "Drag & drop your CSV dataset here",
        type=["csv"],
        help="Supported format: .csv",
    )
    if uploaded_file is None:
        return None

    df = pd.read_csv(uploaded_file)
    st.session_state["uploaded_df"] = df

    # Dataset info badges
    cols = st.columns(4)
    badges = [
        (f"📐 {df.shape[0]:,} rows", "Rows"),
        (f"📊 {df.shape[1]} columns", "Columns"),
        (f"🔢 {df.select_dtypes(include='number').shape[1]} numeric", "Numeric"),
        (f"🏷️ {df.select_dtypes(include=['object','category']).shape[1]} categorical", "Categorical"),
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
    """
    Render target selection and advanced settings.
    Returns a config dict consumed by the pipeline.
    """
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

    # Force Clustering when no target
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

        feat_selection = st.toggle("Feature Selection (Auto)", value=False, help="Let the pipeline auto-select features.")

    config = {
        "target": target,
        "problem_hint": problem_hint,
        "random_seed": random_seed,
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
    """
    Render the run button. On click, execute the LangGraph pipeline,
    store outputs in session_state, and set a flag so the dashboard renders.
    """
    from config import settings
    from core.agent_graph import build_graph

    # Disable button while running
    is_running = st.session_state.get("is_running", False)

    if st.button(
        "🚀 Run AI Data Scientist",
        use_container_width=True,
        type="primary",
        disabled=is_running,
    ):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            st.error("GROQ_API_KEY not found. Please check your `.env` file.")
            return

        st.session_state["is_running"] = True

        with st.spinner("🔄 Agent is working — analyzing, modeling, explaining…"):
            start_time = time.time()
            graph = build_graph()
            initial_state = {"df": df, "target": config["target"]}

            try:
                result = graph.invoke(initial_state)  # type: ignore[arg-type]
                elapsed = round(time.time() - start_time, 1)

                # Persist to session_state
                st.session_state["result"] = result
                st.session_state["target"] = config["target"]
                st.session_state["df"] = df
                st.session_state["training_time"] = elapsed
                st.session_state["model_results"] = result.get("metrics", {})
                st.session_state["ai_explanation"] = result.get("explanation", "")
                st.session_state["dataset_summary"] = _build_dataset_summary(df, config["target"])
                st.session_state["cleaning_report"] = result.get("cleaning_report", "")
                st.session_state["feature_report"] = result.get("feature_report", "")
                st.session_state["analysis_done"] = True

                st.success(f"✅ Analysis complete in **{elapsed}s**!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.text(traceback.format_exc())
            finally:
                st.session_state["is_running"] = False


def _build_dataset_summary(df: pd.DataFrame, target) -> str:
    """Create a concise text summary of the dataset for chatbot context."""
    lines = [
        f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Target: {target or 'None (Clustering)'}",
        f"Numeric columns: {df.select_dtypes(include='number').columns.tolist()}",
        f"Categorical columns: {df.select_dtypes(include=['object','category']).columns.tolist()}",
        f"Missing values: {df.isnull().sum().sum()} total",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SECTION 5 — Dashboard
# ─────────────────────────────────────────────

def render_dashboard():
    """Render the full results dashboard — only called when analysis_done is True."""
    result = st.session_state["result"]
    target = st.session_state["target"]
    df_original = st.session_state["df"]
    problem_type = result.get("problem_type", "")

    # Anchor for auto-scroll
    st.markdown('<div id="dashboard-anchor"></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Metric cards row ──
    _render_metric_cards(result, problem_type)

    # ── Tabs ──
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
    """Show top-level KPI cards summarising the best model."""
    metrics = result.get("metrics", {})
    best_name = result.get("model_name", "N/A")
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
        mse = best_metrics.get("MSE", 0)
        mae = best_metrics.get("MAE", 0)
        c1.metric("Best Model", best_name)
        c2.metric("R² Score", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
        c3.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else mae)
        c4.metric("⏱️ Training Time", f"{elapsed}s")
    else:  # Classification
        c1, c2, c3, c4 = st.columns(4)
        acc = best_metrics.get("Accuracy", 0)
        f1 = best_metrics.get("F1 Score", 0)
        prec = best_metrics.get("Precision", 0)
        c1.metric("Best Model", best_name)
        c2.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (int, float)) else acc)
        c3.metric("F1 Score", f"{f1:.4f}" if isinstance(f1, (int, float)) else f1)
        c4.metric("⏱️ Training Time", f"{elapsed}s")

    st.markdown("")  # spacer


# ── Model Comparison Chart (Plotly) ─────────

def _render_model_comparison_chart(metrics: dict, problem_type: str):
    """Plotly bar chart comparing all trained models."""
    if not metrics:
        return

    df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})

    if problem_type == "Clustering":
        score_col = "Silhouette Score"
        if score_col not in df_m.columns:
            return
        # Filter out non-numeric silhouette values
        df_m = df_m[df_m[score_col].apply(lambda x: isinstance(x, (int, float)))].copy()
        if df_m.empty:
            return
        df_m = df_m.sort_values(score_col, ascending=True)
        fig = px.bar(df_m, x=score_col, y="Model", orientation="h", title="Model Comparison — Silhouette Score",
                     color=score_col, color_continuous_scale="Viridis")
    elif problem_type == "Regression":
        score_col = "R2"
        if score_col not in df_m.columns:
            return
        df_m = df_m.sort_values(score_col, ascending=True)
        fig = px.bar(df_m, x=score_col, y="Model", orientation="h", title="Model Comparison — R² Score",
                     color=score_col, color_continuous_scale="Viridis")
    else:
        score_col = "F1 Score"
        if score_col not in df_m.columns:
            score_col = "Accuracy"
        if score_col not in df_m.columns:
            return
        df_m = df_m.sort_values(score_col, ascending=True)
        fig = px.bar(df_m, x=score_col, y="Model", orientation="h", title=f"Model Comparison — {score_col}",
                     color=score_col, color_continuous_scale="Viridis")

    fig.update_layout(
        height=max(350, len(df_m) * 30 + 120),
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis_title="",
        xaxis_title=score_col,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── EDA Tab ─────────────────────────────────

def _render_eda_tab(result: dict, df_original: pd.DataFrame, problem_type: str):
    st.header("Exploratory Data Analysis")

    if "eda_results" in result:
        st.subheader("📋 Summary Statistics")
        summary_df = pd.DataFrame(result["eda_results"]["description"]).T
        st.table(summary_df)

        # Missing / Unique table
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

    # EDA figures
    if "eda_figures" in result:
        for item in result["eda_figures"]:
            fig = item
            desc, heading = "", ""
            if isinstance(item, dict):
                fig = item.get("figure")
                desc = item.get("description", "")
                heading = item.get("heading", "")
            if heading:
                st.subheader(heading)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.pyplot(fig)
                if desc:
                    st.caption(desc)
            st.markdown("---")

    # Feature importance
    _render_feature_importance(result)

    # Categorical distributions
    _render_categorical_distributions(df_original)


def _render_feature_importance(result: dict):
    best_model_name = result.get("model_name", "")
    best_model = result.get("models", {}).get(best_model_name)
    training_X = result.get("X", pd.DataFrame())

    if not best_model or training_X.empty:
        return

    try:
        model_step = best_model
        if hasattr(model_step, "named_steps") and "model" in model_step.named_steps:
            model_step = model_step.named_steps["model"]

        importances = None
        imp_label = "Importance"
        imp_color = "#4CAF50"

        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            coefs = model_step.coef_
            importances = np.abs(coefs).mean(axis=0) if coefs.ndim > 1 else np.abs(coefs)
            imp_label = "|Coefficient|"
            imp_color = "#FF9800"

        if importances is not None:
            st.subheader("📌 Feature Importance")
            feat_df = pd.DataFrame({"Feature": training_X.columns.tolist(), "Importance": importances})
            feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)

            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            ax_imp.barh(feat_df["Feature"], feat_df["Importance"], color=imp_color)
            ax_imp.set_xlabel(imp_label)
            ax_imp.set_title(f"Feature Importance — {best_model_name}")
            plt.tight_layout()

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.pyplot(fig_imp)
            st.markdown("---")
    except Exception as e:
        st.warning(f"Could not generate feature importance: {e}")


def _render_categorical_distributions(df_original: pd.DataFrame):
    st.subheader("📊 Distribution of Categorical")
    dist_cols = [c for c in df_original.columns if df_original[c].nunique() <= 5]

    if not dist_cols:
        st.info("No columns with 5 or fewer unique values found.")
        return

    for feat in dist_cols:
        col_data = df_original[feat]
        fig_d, ax_d = plt.subplots(figsize=(6, 4))
        if pd.api.types.is_numeric_dtype(col_data):
            sns.histplot(col_data, kde=False, ax=ax_d, color="#2196F3", discrete=True)  # type: ignore[arg-type]
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


# ── Model Performance Tab ───────────────────

def _render_model_tab(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    st.header("Model Evaluation")
    metrics = result.get("metrics", {})

    # Comparison chart
    _render_model_comparison_chart(metrics, problem_type)

    # Metrics table
    if metrics:
        st.subheader("📊 Full Metrics Table")
        st.table(pd.DataFrame(metrics).T)

    st.success(f"**Best Performing Model:** {result.get('model_name', 'N/A')}")

    if problem_type == "Clustering":
        _render_cluster_viz(result, df_original)


def _render_predictions_tab(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    st.header("Predict with Best Model")
    if problem_type == "Clustering":
        st.info("Prediction is not applicable for clustering tasks.")
        return
    _render_prediction_form(result, df_original, target, problem_type)


def _render_cluster_viz(result: dict, df_original: pd.DataFrame):
    st.markdown("---")
    st.subheader("🔍 Cluster Assignment Preview")
    best_model_name = result.get("model_name", "")
    best_model = result.get("models", {}).get(best_model_name)
    training_X = result.get("X", pd.DataFrame())

    if not best_model or training_X.empty:
        return

    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(training_X.select_dtypes(include=["number"]))

        if hasattr(best_model, "labels_"):
            labels = best_model.labels_
        elif hasattr(best_model, "predict"):
            labels = best_model.predict(X_scaled)
        else:
            labels = None

        if labels is not None:
            cluster_df = df_original.copy()
            cluster_df["Cluster"] = labels
            st.write(f"**{len(set(labels) - {-1})}** clusters found")
            st.dataframe(cluster_df.head(20), use_container_width=True)

            fig_cl, ax_cl = plt.subplots(figsize=(6, 4))
            pd.Series(labels).value_counts().sort_index().plot(kind="bar", ax=ax_cl, color="#4CAF50")
            ax_cl.set_title("Cluster Sizes")
            ax_cl.set_xlabel("Cluster")
            ax_cl.set_ylabel("Count")
            plt.tight_layout()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.pyplot(fig_cl)
    except Exception as e:
        st.warning(f"Could not visualize clusters: {e}")


def _render_prediction_form(result: dict, df_original: pd.DataFrame, target, problem_type: str):
    best_model_name = result.get("model_name", "")
    best_model = result.get("models", {}).get(best_model_name)
    training_X = result.get("X", pd.DataFrame())

    if not best_model or training_X.empty:
        st.warning("Model or features not available for prediction.")
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
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=median_val,
                            step=1,
                        )
                    elif pd.api.types.is_float_dtype(df_original[col_name]):
                        min_val = float(df_original[col_name].min())
                        max_val = float(df_original[col_name].max())
                        median_val = float(df_original[col_name].median())
                        input_data[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=median_val,
                            step=0.01,
                            format="%.2f",
                        )
                    else:
                        input_data[col_name] = st.number_input(
                            f"{col_name}",
                            value=float(df_original[col_name].median()),
                            step=0.01,
                            format="%.2f",
                        )

                submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

        if submitted:
            try:
                from pipeline.data_cleaning import clean_data
                from pipeline.feature_engineering import perform_feature_engineering

                pred_row = input_data.copy()
                if pd.api.types.is_numeric_dtype(df_original[target]):
                    pred_row[target] = 0
                else:
                    pred_row[target] = df_original[target].mode()[0]

                pred_df = pd.DataFrame([pred_row])
                combined_df = pd.concat([df_original, pred_df], ignore_index=True)

                cleaned_combined, _ = clean_data(combined_df)
                X_combined, _, _ = perform_feature_engineering(cleaned_combined, target, problem_type)

                X_pred = X_combined.iloc[[-1]]
                X_pred = X_pred.reindex(columns=training_X.columns, fill_value=0)

                prediction = best_model.predict(X_pred)
                predicted_value = prediction[0]

                if df_original[target].dtype == "object" or df_original[target].dtype.name == "category":
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    le.fit(df_original[target].astype(str))
                    predicted_value = le.inverse_transform([int(predicted_value)])[0]

                with col_form:
                    st.markdown(
                        f'<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); '
                        f'border-radius: 12px; padding: 2rem; text-align: center; '
                        f'border: 1px solid rgba(255,255,255,0.1); margin-top: 1rem;">' 
                        f'<p style="color: #aaa; margin-bottom: 0.5rem; font-size: 0.9rem;">Predicted Value</p>'
                        f'<h2 style="color: #4CAF50; margin: 0;">{target}</h2>'
                        f'<h1 style="color: #fff; margin-top: 0.5rem; font-size: 2.5rem;">{predicted_value}</h1>'
                        f'<p style="color: #888; font-size: 0.8rem; margin-top: 1rem;">'
                        f'Model: {best_model_name}</p></div>',
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
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return text
        return "\n\n".join(f"- {line}" for line in lines)

    st.subheader("Data Cleaning Report")
    st.markdown(_to_bullet_list(result.get("cleaning_report", "No report")))

    st.subheader("Feature Engineering Report")
    st.markdown(_to_bullet_list(result.get("feature_report", "No report")))

    # Agent graph
    st.subheader("🗺️ Agent Pipeline Graph")
    try:
        from core.agent_graph import build_graph

        graph = build_graph()
        graph_drawable = graph.get_graph()
        try:
            png_bytes = graph_drawable.draw_mermaid_png()
            st.image(png_bytes, caption="Agent Pipeline Graph", width=300)
        except Exception:
            mermaid_str = graph_drawable.draw_mermaid()
            st.code(mermaid_str, language="mermaid")
    except Exception as e:
        st.warning(f"Could not render agent graph: {e}")
