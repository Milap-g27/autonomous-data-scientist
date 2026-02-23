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
import asyncio
import time
import io
import traceback
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from api.schemas import (
    UploadResponse, DatasetInfo,
    ConfigureRequest, ConfigureResponse,
    AnalyzeRequest, AnalyzeResponse, FigureData,
    PredictRequest, PredictResponse,
    ChatRequest, ChatResponse,
    HealthResponse,
)
from api.session import store, Session
from api.utils import fig_to_base64, build_dataset_summary
from config import settings

router = APIRouter()


# ── Health ──────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", timestamp=datetime.utcnow())


# ── Upload ──────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
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
async def configure(req: ConfigureRequest):
    """Save analysis configuration to the session."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded in this session.")

    config = req.model_dump(exclude={"session_id"})

    # Force Clustering when no target
    if config["target"] is None:
        config["problem_hint"] = "Clustering"

    session.config = config
    return ConfigureResponse(session_id=session.id, config=config)


# ── Analyze ─────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run the full async LangGraph pipeline and return results."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded in this session.")
    if not session.config:
        raise HTTPException(status_code=400, detail="Configuration not set. Call /configure first.")
    if session.task_status == "running":
        raise HTTPException(status_code=409, detail="Analysis already running for this session.")

    session.task_status = "running"
    session.task_error = None

    try:
        from core.agent_graph import build_graph

        start_time = time.time()
        graph = build_graph()
        initial_state = {
            "df": session.df,
            "target": session.config.get("target"),
        }

        result = await graph.ainvoke(initial_state)  # type: ignore[arg-type]
        elapsed = round(time.time() - start_time, 1)

        session.result = result
        session.task_status = "completed"

        # Serialize EDA figures to base64
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

        # Build a JSON-safe metrics dict (convert numpy types)
        raw_metrics = result.get("metrics", {})
        safe_metrics = _make_json_safe(raw_metrics)

        # Build safe eda_results
        raw_eda = result.get("eda_results", {})
        safe_eda = _make_json_safe(raw_eda)

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
            training_time=elapsed,
        )

    except Exception as exc:
        session.task_status = "failed"
        session.task_error = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


# ── Predict ─────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run a single prediction using the best trained model."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

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
        if df_original[target].dtype == "object" or df_original[target].dtype.name == "category":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(df_original[target].astype(str))
            predicted_value = le.inverse_transform([int(predicted_value)])[0]

        return PredictResponse(
            predicted_value=_make_json_safe(predicted_value),
            model_used=best_model_name,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


# ── Chat ────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Async AI chatbot scoped to the session's dataset/model context."""
    session = await store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    reply = await _get_chat_response(session, req.message)
    return ChatResponse(reply=reply)


async def _get_chat_response(session: Session, user_message: str) -> str:
    """Call ChatGroq asynchronously with session context."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    _REJECTION = "I can only answer questions related to the current dataset and model analysis."
    _SYSTEM_TEMPLATE = """You are an expert data science assistant embedded in the Autonomous Data Scientist application.
You answer questions ONLY about:
- The uploaded dataset (columns, types, distributions)
- Feature engineering and data cleaning steps performed
- Model performance and evaluation metrics
- The AI explanation generated for this analysis
- General machine learning theory and best practices

If the user asks anything unrelated, respond EXACTLY with:
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
    model_config = str(session.config) if session.config else "Not configured yet."
    model_results = str(session.result.get("metrics", {})) if session.result else "No results yet."
    ai_explanation = session.result.get("explanation", "No explanation generated yet.") if session.result else "No explanation generated yet."
    cleaning_report = session.result.get("cleaning_report", "No cleaning report yet.") if session.result else "No cleaning report yet."
    feature_report = session.result.get("feature_report", "No feature engineering report yet.") if session.result else "No feature engineering report yet."

    system_prompt = _SYSTEM_TEMPLATE.format(
        rejection=_REJECTION,
        dataset_summary=dataset_summary,
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
        for msg in session.chat_history[-12:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))

        response = await llm.ainvoke(messages)
        reply = str(response.content)

        # Persist history
        session.chat_history.append({"role": "user", "content": user_message})
        session.chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        return f"Error: {e}"


# ── Sessions management ────────────────────

@router.get("/sessions")
async def list_sessions():
    ids = await store.list_ids()
    return {"sessions": ids}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = await store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


# ── Helpers ─────────────────────────────────

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
