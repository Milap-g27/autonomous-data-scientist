"""
Utility helpers for the API layer.
"""
import base64
import io
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — thread-safe
import matplotlib.pyplot as plt
import matplotlib.figure


def fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_dataset_summary(df, target) -> str:
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
