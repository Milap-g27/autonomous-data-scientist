"""
FastAPI routes for the Autonomous Data Scientist Agent.

Endpoints:
  POST /upload          Upload a CSV dataset
  POST /configure       Set analysis configuration
  POST /analyze         Run the full async pipeline
  POST /predict         Predict with the trained best model
  POST /chat            Chat with AI assistant
  GET  /sessions        List active session IDs
  DELETE /sessions/{id} Delete a session
  GET  /health          Health check
"""
import logging
import asyncio
import time
import io
import os
import subprocess
import tempfile
import traceback
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request

from api.schemas import (
    UploadResponse, DatasetInfo,
    ConfigureRequest, ConfigureResponse,
    AnalyzeRequest, AnalyzeResponse, AnalyzeStartResponse, SessionStatusResponse, FigureData,
    PredictRequest, PredictResponse,
    ChatRequest, ChatResponse, PlotDiagnostics,
    HealthResponse,
)
from api.session import store, Session
from api.auth import verify_firebase_token
from api.utils import fig_to_base64, build_dataset_summary, make_json_safe
from config import settings
from limiter import limiter

router = APIRouter()


def _enforce_session_access(session: Session, uid: str) -> None:
    if session.owner_uid is None:
        session.owner_uid = uid
        return
    if session.owner_uid != uid:
        raise HTTPException(status_code=403, detail="Forbidden: session does not belong to this user.")


async def _run_analysis_task(session_id: str) -> None:
    session = await store.get(session_id)
    if not session:
        return

    try:
        from core.agent_graph import build_graph

        session.task_status = "running"
        session.task_error = None
        start_time = time.time()

        graph = build_graph()
        initial_state = {
            "df": session.df,
            "target": session.config.get("target"),
        }

        result = await graph.ainvoke(initial_state)  # type: ignore[arg-type]
        elapsed = round(time.time() - start_time, 1)
        if isinstance(result, dict):
            result["_training_time_sec"] = elapsed
        session.result = result
        session.task_status = "completed"
        session.task_error = None
    except Exception:
        session.task_status = "failed"
        session.task_error = traceback.format_exc()


async def _build_analyze_response(session: Session) -> AnalyzeResponse:
    if not session.result:
        raise HTTPException(status_code=400, detail="No analysis results. Run /analyze first.")

    result = session.result

    figures_out: list[FigureData] = []
    for item in result.get("eda_figures", []):
        if isinstance(item, dict):
            fig = item.get("figure")
            if fig is not None:
                figures_out.append(FigureData(
                    heading=item.get("heading", ""),
                    description=item.get("description", ""),
                    image_base64=await asyncio.to_thread(fig_to_base64, fig),
                ))

    raw_metrics = result.get("metrics", {})
    safe_metrics = make_json_safe(raw_metrics)

    raw_eda = result.get("eda_results", {})
    safe_eda = make_json_safe(raw_eda)

    return AnalyzeResponse(
        session_id=session.id,
        problem_type=result.get("problem_type", ""),
        best_model=result.get("model_name", ""),
        metrics=safe_metrics,
        eda_results=safe_eda,
        eda_figures=figures_out,
        explanation=result.get("explanation", ""),
        cleaning_report=result.get("cleaning_report", ""),
        feature_report=result.get("feature_report", ""),
        training_time=float(result.get("_training_time_sec", 0.0)),
    )


# ── Health ──────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", timestamp=datetime.utcnow())


# ── Upload ──────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload a CSV file, parse it, and return a new session."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    contents = await file.read()
    try:
        df = await asyncio.to_thread(pd.read_csv, io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    session = await store.create()
    session.df = df

    info = DatasetInfo(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        numeric_columns=df.select_dtypes(include="number").columns.tolist(),
        categorical_columns=df.select_dtypes(include=["object", "category"]).columns.tolist(),
        column_names=df.columns.tolist(),
        missing_values=int(df.isnull().sum().sum()),
        preview=df.head(10).to_dict(orient="records"),
    )

    return UploadResponse(session_id=session.id, dataset_info=info)


# ── Configure ──────────────────────────────

@router.post("/configure", response_model=ConfigureResponse)
async def configure(
    req: ConfigureRequest,
    decoded_token: dict = Depends(verify_firebase_token),
):
    """Save analysis configuration to the session."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded in this session.")

    config = req.model_dump(exclude={"session_id"})

    # Force Clustering when no target
    if config["target"] is None:
        config["problem_hint"] = "Clustering"

    session.config = config
    return ConfigureResponse(session_id=session.id, config=config)


# ── Analyze ─────────────────────────────────

@router.post("/analyze", response_model=AnalyzeStartResponse)
@limiter.limit("5/minute")
async def analyze(
    request: Request,
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    decoded_token: dict = Depends(verify_firebase_token),
):
    """Start the async LangGraph pipeline in the background and return immediately."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded in this session.")
    if not session.config or "target" not in session.config:
        raise HTTPException(status_code=400, detail="Configuration not set. Call /configure first.")
    if session.task_status == "running":
        raise HTTPException(status_code=409, detail="Analysis already running for this session.")

    session.task_status = "running"
    session.task_error = None
    background_tasks.add_task(_run_analysis_task, session.id)

    return AnalyzeStartResponse(session_id=session.id, status="started")


# ── Predict ─────────────────────────────────

@router.get("/status/{session_id}", response_model=SessionStatusResponse)
async def get_status(
    session_id: str,
    decoded_token: dict = Depends(verify_firebase_token),
):
    session = await store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)
    return SessionStatusResponse(status=session.task_status, error=session.task_error)


@router.get("/results/{session_id}", response_model=AnalyzeResponse)
async def get_analysis_result(
    session_id: str,
    decoded_token: dict = Depends(verify_firebase_token),
):
    session = await store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)

    if session.task_status == "running":
        raise HTTPException(status_code=409, detail="Analysis is still running.")
    if session.task_status == "failed":
        raise HTTPException(status_code=500, detail=session.task_error or "Analysis failed.")
    if session.task_status != "completed":
        raise HTTPException(status_code=400, detail="Analysis has not completed yet.")

    return await _build_analyze_response(session)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    decoded_token: dict = Depends(verify_firebase_token),
):
    """Run a single prediction using the best trained model."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)

    result = session.result
    if not result:
        raise HTTPException(status_code=400, detail="No analysis results. Run /analyze first.")

    problem_type = result.get("problem_type", "")
    if problem_type == "Clustering":
        raise HTTPException(status_code=400, detail="Prediction not applicable for clustering.")

    best_model_name = result.get("model_name", "")
    best_model = result.get("models", {}).get(best_model_name)
    training_X = result.get("X")

    if best_model is None or training_X is None:
        raise HTTPException(status_code=400, detail="Trained model or features not available.")

    target = session.config.get("target")
    df_original = session.df

    if not target:
        raise HTTPException(status_code=400, detail="No target column configured.")

    try:
        from pipeline.data_cleaning import clean_data
        from pipeline.feature_engineering import perform_feature_engineering

        pred_row = req.input_data.copy()
        # Add a dummy target value so the pipeline shape matches
        if pd.api.types.is_numeric_dtype(df_original[target]):
            pred_row[target] = 0
        else:
            pred_row[target] = df_original[target].mode()[0]

        pred_df = pd.DataFrame([pred_row])
        combined_df = pd.concat([df_original, pred_df], ignore_index=True)

        cleaned_combined, _ = await asyncio.to_thread(clean_data, combined_df)
        X_combined, _, _ = await asyncio.to_thread(
            perform_feature_engineering, cleaned_combined, target, problem_type
        )

        X_pred = X_combined.iloc[[-1]]
        X_pred = X_pred.reindex(columns=training_X.columns, fill_value=0)

        prediction = await asyncio.to_thread(best_model.predict, X_pred)
        predicted_value = prediction[0]

        # Decode label if original target was categorical
        # Use pd.api.types checks to handle both legacy 'object' and pandas 2.x StringDtype
        target_dtype = df_original[target].dtype
        is_categorical_target = (
            pd.api.types.is_string_dtype(df_original[target])
            or isinstance(target_dtype, pd.CategoricalDtype)
        )
        if is_categorical_target:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(df_original[target].astype(str))
            predicted_value = le.inverse_transform([int(predicted_value)])[0]

        return PredictResponse(
            predicted_value=make_json_safe(predicted_value),
            model_used=best_model_name,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


# ── Chat ────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    req: ChatRequest,
    decoded_token: dict = Depends(verify_firebase_token),
):
    """Async AI chatbot scoped to the session's dataset/model context."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)

    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    reply, image_base64, plot_diagnostics = await _get_chat_response(session, req.message, api_key)
    return ChatResponse(reply=reply, image_base64=image_base64, plot_diagnostics=plot_diagnostics)


async def _get_chat_response(session: Session, user_message: str, api_key: str) -> tuple[str, str | None, PlotDiagnostics | None]:
    """Call ChatGroq asynchronously with session context."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    import re
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    _REJECTION = "I can only answer questions related to the current dataset and model analysis."
    _SYSTEM_TEMPLATE = """You are a concise data science assistant embedded in the Autonomous Data Scientist application.

STRICT RULES — FOLLOW EXACTLY:
1. Answer ONLY the user's LATEST message. NEVER repeat, summarize, or re-answer previous questions.
2. ONLY use information from the session context provided below. If the information is not available in the context, say "This information is not available in the current session results." Do NOT guess, fabricate, or derive values that aren't explicitly present.
3. NEVER generate code to retrain models, split data, or re-run analysis. The analysis has already been completed.
4. Be extremely concise. Give direct answers. No unnecessary preamble, formulas, or lengthy explanations.
5. Only generate a <PLOT> block when the user EXPLICITLY asks for a plot, chart, or visualization. NEVER generate <PLOT> for non-visual questions.

PLOTTING RULES (only when user asks for a visualization):
1. ALWAYS handle NaN values before plotting (e.g., `data = df[col].dropna()`).
2. Use `plt.tight_layout()` for multi-subplot figures.
3. Do NOT call `plt.show()`.
4. Use EXACTLY ONE `<PLOT>...</PLOT>` pair per response.
5. You have access to the DataFrame via the global variable `df`. Use ONLY `df` — do NOT import sklearn or retrain models.
6. Do NOT use markdown code blocks inside the tags.

Example plot response:
<PLOT>
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, ax=ax)
plt.tight_layout()
</PLOT>

FORMATTING: Use markdown. Use backticks for column/metric names with NO spaces inside — correct: `focus_score`, WRONG: ` focus_score `.

If the question is COMPLETELY UNRELATED to data science or this dataset, respond EXACTLY with:
"I can only answer questions related to the current dataset and model analysis."

--- SESSION CONTEXT ---

### Problem Type
{problem_type}

### Dataset Summary
{dataset_summary}

### EDA Statistics & Insights
{eda_results}

### Model Configuration
{model_config}

### Model Results
{model_results}

### AI Explanation
{ai_explanation}

### Data Cleaning Report
{cleaning_report}

### Feature Engineering Report
{feature_report}
"""


    dataset_summary = (
        build_dataset_summary(session.df, session.config.get("target"))
        if session.df is not None
        else "No dataset uploaded yet."
    )
    problem_type = session.result.get("problem_type", "Unknown") if session.result else "Not determined yet."
    
    eda_data = session.result.get("eda_results", {}) if session.result else {}
    if eda_data:
        eda_lines = []
        if "columns" in eda_data:
            eda_lines.append(f"Columns used in analysis: {', '.join(f'`{c}`' for c in eda_data['columns'])}")
        if "top_correlated_features" in eda_data:
            eda_lines.append(f"Top features: {', '.join(f'`{c}`' for c in eda_data['top_correlated_features'])}")
        if "description" in eda_data:
            eda_lines.append("\nDescriptive Statistics:")
            try:
                desc_df = pd.DataFrame.from_dict(eda_data["description"])
                eda_lines.append(desc_df.to_string())
            except Exception:
                eda_lines.append(str(eda_data["description"]))
        eda_results = "\n".join(eda_lines)
    else:
        eda_results = "No EDA results available yet."

    model_config = str(session.config) if session.config else "Not configured yet."
    if session.result:
        raw_metrics = session.result.get("metrics", {})
        best_model = session.result.get("model_name", "Unknown")
        # Format all model metrics into a readable string
        lines = [f"Best model: {best_model}"]
        for model_name, scores in raw_metrics.items():
            if isinstance(scores, dict):
                score_str = ", ".join(f"{k}: {v}" for k, v in scores.items())
                lines.append(f"- {model_name}: {score_str}")
            else:
                lines.append(f"- {model_name}: {scores}")
        model_results = "\n".join(lines)
    else:
        model_results = "No results yet."
    ai_explanation = session.result.get("explanation", "No explanation generated yet.") if session.result else "No explanation generated yet."
    cleaning_report = session.result.get("cleaning_report", "No cleaning report yet.") if session.result else "No cleaning report yet."
    feature_report = session.result.get("feature_report", "No feature engineering report yet.") if session.result else "No feature engineering report yet."

    system_prompt = _SYSTEM_TEMPLATE.format(
        problem_type=problem_type,
        dataset_summary=dataset_summary,
        eda_results=eda_results,
        model_config=model_config,
        model_results=model_results,
        ai_explanation=ai_explanation,
        cleaning_report=cleaning_report,
        feature_report=feature_report,
    )

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=1024,
        )

        messages: list = [SystemMessage(content=system_prompt)]

        # Include recent history (last 6 exchanges)
        # Truncate long AI responses to avoid the LLM echoing/repeating old answers
        for msg in session.chat_history[-12:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                # Keep AI history short so the model doesn't re-generate previous replies
                summary = msg["content"]
                if len(summary) > 300:
                    summary = summary[:300] + "…[truncated]"
                messages.append(AIMessage(content=summary))
        messages.append(HumanMessage(content=user_message))

        response = await llm.ainvoke(messages)
        reply = str(response.content)

        # Plot Interception Logic
        image_base64 = None
        plot_diagnostics: PlotDiagnostics | None = None
        
        # New XML-based robust tag extraction
        if "<PLOT>" in reply and "</PLOT>" in reply:
            code_to_run = ""
            try:
                start_idx = reply.find("<PLOT>") + len("<PLOT>")
                end_idx = reply.find("</PLOT>")
                code_to_run = reply[start_idx:end_idx].strip()
                
                # Fallback cleanups in case the LLM still wraps with markdown inside the tags
                if code_to_run.startswith("```python"): code_to_run = code_to_run[9:].strip()
                elif code_to_run.startswith("```"): code_to_run = code_to_run[3:].strip()
                if code_to_run.endswith("```"): code_to_run = code_to_run[:-3].strip()

                sandbox_start = time.perf_counter()
                image_base64 = _run_plot_code_sandboxed(code_to_run, session.df.head(5000).to_json())
                sandbox_duration_ms = int((time.perf_counter() - sandbox_start) * 1000)
                if settings.PLOT_DEBUG:
                    plot_diagnostics = PlotDiagnostics(
                        rows=int(session.df.shape[0]),
                        columns=int(session.df.shape[1]),
                        code_length=len(code_to_run),
                        sandbox_duration_ms=sandbox_duration_ms,
                    )
                if not image_base64:
                    raise RuntimeError("Plot code executed but no image was produced.")
                
                # Remove the raw code block from the user's view
                full_tag_block = reply[reply.find("<PLOT>"):end_idx + len("</PLOT>")]
                reply = reply.replace(full_tag_block, "\nHere is the plot you requested:\n")
            except Exception as e:
                if settings.PLOT_DEBUG and session.df is not None:
                    plot_diagnostics = PlotDiagnostics(
                        rows=int(session.df.shape[0]),
                        columns=int(session.df.shape[1]),
                        code_length=len(code_to_run),
                        sandbox_duration_ms=0,
                        error=str(e),
                    )
                logger.error("Plot execution failed: %s", e, exc_info=True)
                reply += f"\n\nNote: Failed to generate plot: {e}"
        # Fallback for older generations if it still uses # PLOT
        elif "# PLOT" in reply and image_base64 is None:
            match = re.search(r"```(?:python)?\s*# PLOT\s*(.*?)\n```", reply, re.DOTALL)
            if not match:
                match = re.search(r"```(?:python)?\s*\n?(.*?)\n```", reply, re.DOTALL)
                
            if match and "# PLOT" in match.group(0):
                code_to_run = match.group(1).strip()
                if code_to_run.startswith("# PLOT"):
                    code_to_run = code_to_run.replace("# PLOT", "", 1).strip()

                try:
                    sandbox_start = time.perf_counter()
                    image_base64 = _run_plot_code_sandboxed(code_to_run, session.df.head(5000).to_json())
                    sandbox_duration_ms = int((time.perf_counter() - sandbox_start) * 1000)
                    if settings.PLOT_DEBUG:
                        plot_diagnostics = PlotDiagnostics(
                            rows=int(session.df.shape[0]),
                            columns=int(session.df.shape[1]),
                            code_length=len(code_to_run),
                            sandbox_duration_ms=sandbox_duration_ms,
                        )
                    if not image_base64:
                        raise RuntimeError("Plot code executed but no image was produced.")
                    
                    reply = reply.replace(match.group(0), "\nHere is the plot you requested:\n")
                except Exception as e:
                    if settings.PLOT_DEBUG and session.df is not None:
                        plot_diagnostics = PlotDiagnostics(
                            rows=int(session.df.shape[0]),
                            columns=int(session.df.shape[1]),
                            code_length=len(code_to_run),
                            sandbox_duration_ms=0,
                            error=str(e),
                        )
                    logger.error("Plot execution failed: %s", e, exc_info=True)
                    reply += f"\n\nNote: Failed to generate plot: {e}"

        # Persist history
        session.chat_history.append({"role": "user", "content": user_message})
        session.chat_history.append({"role": "assistant", "content": reply})

        return reply, image_base64, plot_diagnostics

    except Exception as e:
        return f"Error: {e}", None, None


# ── Sessions management ────────────────────

@router.get("/sessions")
async def list_sessions(decoded_token: dict = Depends(verify_firebase_token)):
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    ids = await store.list_ids_for_owner(uid)
    return {"sessions": ids}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    decoded_token: dict = Depends(verify_firebase_token),
):
    session = await store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    uid = str(decoded_token.get("uid", ""))
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication failed: missing uid in token.")
    _enforce_session_access(session, uid)
    deleted = await store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


def _run_plot_code_sandboxed(code: str, df_json: str) -> str:
    import base64
    import sys
    encoded_df_json = base64.b64encode(df_json.encode("utf-8")).decode("utf-8")
    PLOT_SCRIPT_TEMPLATE = """
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import sys, os
import tempfile

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except Exception:
    pass

data_json = base64.b64decode("{df_json_b64}").decode("utf-8")
df = pd.read_json(io.StringIO(data_json))

{user_code}

fig = plt.gcf()
if fig and fig.axes:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, format="png", bbox_inches="tight")
    plt.close(fig)
    print(tmp.name)
else:
    print("")
"""
    script_code = PLOT_SCRIPT_TEMPLATE.format(df_json_b64=encoded_df_json, user_code=code)
    
    script_tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    script_tmp.write(script_code)
    script_tmp.close()
    
    png_path = None
    try:
        result = subprocess.run([sys.executable, script_tmp.name], capture_output=True, text=True, timeout=45)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(f"Sandbox error. stderr={stderr[:600]} stdout={stdout[:200]}")

        stdout_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        for line in reversed(stdout_lines):
            if os.path.exists(line):
                png_path = line
                break

        if not png_path:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(
                "Sandbox completed but no image path was returned. "
                f"stderr={stderr[:400]} stdout={stdout[:200]}"
            )

        if png_path and os.path.exists(png_path):
            with open(png_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        raise RuntimeError("Sandbox returned an image path but the file was not found.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Plot generation timed out after 45 seconds.")
    finally:
        if os.path.exists(script_tmp.name):
            try:
                os.remove(script_tmp.name)
            except Exception:
                pass
        if png_path and os.path.exists(png_path):
            try:
                os.remove(png_path)
            except Exception:
                pass

