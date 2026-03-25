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

