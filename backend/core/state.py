from typing import TypedDict, List, Dict, Any, Optional

import pandas as pd

class AgentState(TypedDict):
    df: pd.DataFrame
    problem_type: str  # 'Regression', 'Classification', or 'Clustering'
    target: Optional[str]
    X: pd.DataFrame
    y: Optional[pd.Series]
    X_train: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    metrics: Dict[str, Any]
    model_name: str
    best_params: Dict[str, Dict[str, Any]]
    eda_results: Dict[str, Any]
    eda_figures: List[Dict[str, Any]]
    evaluation_results: Dict[str, Any]
    evaluation_figures: List[Dict[str, Any]]
    explanation: str
    cleaning_report: str
    feature_report: str
    models: Dict[str, Any]
