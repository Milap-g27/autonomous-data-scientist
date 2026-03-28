"""
Pydantic models for FastAPI request / response serialization.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


# ── Upload ──────────────────────────────────

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    column_names: List[str]
    missing_values: int
    preview: List[Dict[Any, Any]]  # first 10 rows


class UploadResponse(BaseModel):
    session_id: str
    dataset_info: DatasetInfo
    message: str = "Dataset uploaded successfully."


# ── Configure ───────────────────────────────

class ConfigureRequest(BaseModel):
    session_id: str
    target: Optional[str] = None
    problem_hint: str = Field(
        default="Auto-detect",
        description="One of: Auto-detect, Classification, Regression, Clustering",
    )
    random_seed: int = 42
    test_size: float = 0.2
    scaling: bool = True
    feature_selection: bool = False


class ConfigureResponse(BaseModel):
    session_id: str
    config: Dict[str, Any]
    message: str = "Configuration saved."


# ── Analyze ─────────────────────────────────

class AnalyzeRequest(BaseModel):
    session_id: str


class FigureData(BaseModel):
    heading: str = ""
    description: str = ""
    image_base64: str  # PNG encoded as base64


class AnalyzeResponse(BaseModel):
    session_id: str
    problem_type: str
    best_model: str
    metrics: Dict[str, Any]
    eda_results: Dict[str, Any]
    eda_figures: List[FigureData]
    explanation: str
    cleaning_report: str
    feature_report: str
    training_time: float
    message: str = "Analysis complete."


# ── Predict ─────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str
    input_data: Dict[str, Any]


class PredictResponse(BaseModel):
    predicted_value: Any
    model_used: str


# ── Chat ────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class PlotDiagnostics(BaseModel):
    rows: int
    columns: int
    code_length: int
    sandbox_duration_ms: int
    error: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    image_base64: Optional[str] = None
    plot_diagnostics: Optional[PlotDiagnostics] = None


# ── Task / Status ──────────────────────────

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


# ── Health ──────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime
