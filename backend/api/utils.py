"""
Utility helpers for the API layer.
"""
import base64
import io
from typing import Any
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure


def fig_to_base64(fig: "matplotlib.figure.Figure") -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_dataset_summary(df, target) -> str:
    """Create a concise text summary of the dataset for chatbot context."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    def fmt_cols(cols):
        return ", ".join(f"`{c}`" for c in cols) if cols else "None"

    lines = [
        f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {fmt_cols(df.columns.tolist())}",
        f"Target: `{target}`" if target else "Target: None (Clustering)",
        f"Numeric columns: {fmt_cols(num_cols)}",
        f"Categorical columns: {fmt_cols(cat_cols)}",
        f"Missing values: {df.isnull().sum().sum()} total",
    ]

    if len(num_cols) > 0:
        lines.append("\n--- Numeric Summary Statistics ---")
        lines.append(df[num_cols].describe().to_string())

    if len(cat_cols) > 0:
        lines.append("\n--- Categorical Summary ---")
        lines.append(df[cat_cols].describe().to_string())

    return "\n".join(lines)


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
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

