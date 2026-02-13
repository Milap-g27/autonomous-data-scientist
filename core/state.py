from typing import TypedDict, List, Dict, Any
import pandas as pd
import matplotlib.figure

class AgentState(TypedDict):
    df: pd.DataFrame
    problem_type: str  # 'Regression' or 'Classification'
    target: str
    X: pd.DataFrame
    y: pd.Series
    metrics: Dict[str, Any]
    model_name: str
    eda_results: Dict[str, Any]
    eda_figures: List[matplotlib.figure.Figure]
    explanation: str
    cleaning_report: str
    feature_report: str
    models: Dict[str, Any]
