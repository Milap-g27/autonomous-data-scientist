"""
EDA Module - Pre-model exploratory data analysis.

Guaranteed plots per problem type:

ALL PROBLEMS:
- Missing Values Overview
- Correlation Heatmap
- Distribution Grid (numeric features)
- Outlier Detection (boxplot)

REGRESSION (extra):
- Target Distribution
- Target vs Top Features (scatter + trendline)
- Residual vs Fitted
- Residual Distribution
- Q-Q Plot
- Scale-Location Plot

CLASSIFICATION (extra):
- Class Balance (bar + pie)
- Target vs Numeric Features (boxplot per class)
- Stacked Bar (binary/categorical features x target)
- Pairplot (hue = target)

CLUSTERING (extra):
- PCA Scatter
- Elbow Curve
- Silhouette Score vs K
- t-SNE Scatter

AI planning (Anthropic / Groq) supplements guaranteed plots.
If AI is unavailable, the module still returns deterministic guaranteed plots.
"""

import json
import os
import re
import traceback
import warnings
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import settings


PLOT_PALETTE = "coolwarm"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)


SUPPORTED_PLOT_TYPES = {
    "histogram",
    "kde",
    "boxplot",
    "violinplot",
    "barplot",
    "countplot",
    "scatterplot",
    "lineplot",
    "heatmap",
    "pairplot",
    "piechart",
    "stackedbar",
    "missingvalues",
    "outlierdetection",
    "distribution_grid",
    "target_vs_feature",
    "feature_importance_proxy",
    "class_balance",
    "regression_residual",
    "qq_plot",
    "scale_location",
    "pca_scatter",
    "elbow_curve",
    "silhouette_plot",
    "tsne_scatter",
    "cluster_size_bar",
}


_SYSTEM_PROMPT = """
You are an expert data scientist focused on practical EDA.
Given a dataset profile, return ONLY a JSON array of supplementary plot specs
(do not repeat the guaranteed core plots).

Each item MUST be:
{
  "plot_id": "unique_snake_case_id",
  "plot_type": "supported_type",
  "title": "chart title",
  "description": "what insight this chart gives",
  "columns": ["col1", "col2"],
  "params": {}
}

Supported plot_type values:
histogram, kde, boxplot, violinplot, barplot, countplot, scatterplot, lineplot,
heatmap, pairplot, piechart, stackedbar, missingvalues, outlierdetection,
distribution_grid, target_vs_feature, class_balance, regression_residual,
feature_importance_proxy, pca_scatter, qq_plot, scale_location,
elbow_curve, silhouette_plot, tsne_scatter, cluster_size_bar

Rules:
- Generate 4 to 8 supplementary plots only.
- Use ONLY columns present in the profile.
- Return ONLY raw JSON array (no markdown, no comments).
""".strip()


def _safe_json_array(raw: str) -> list[dict]:
    if not raw:
        return []

    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass

    return []


def _build_profile(df: pd.DataFrame, target: Optional[str], problem_type: str) -> str:
    profile: dict[str, Any] = {
        "problem_type": problem_type,
        "target_column": target,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "n_numeric": int(df.select_dtypes(include="number").shape[1]),
        "n_categorical": int(df.select_dtypes(exclude="number").shape[1]),
        "columns": {},
        "missing_total": int(df.isna().sum().sum()),
    }

    for col in df.columns:
        ser = df[col]
        info: dict[str, Any] = {
            "dtype": str(ser.dtype),
            "nunique": int(ser.nunique(dropna=True)),
            "missing": int(ser.isna().sum()),
            "is_target": bool(col == target),
        }

        if pd.api.types.is_numeric_dtype(ser):
            s = ser.dropna()
            if not s.empty:
                info.update(
                    {
                        "min": round(float(s.min()), 5),
                        "max": round(float(s.max()), 5),
                        "mean": round(float(s.mean()), 5),
                        "std": round(float(s.std()), 5),
                        "skew": round(float(s.skew()), 5),
                    }
                )
        else:
            info["top_values"] = ser.astype(str).value_counts().head(5).to_dict()

        profile["columns"][col] = info

    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:8]
    if len(numeric_cols) > 1:
        try:
            profile["correlation_snippet"] = df[numeric_cols].corr().round(3).to_dict()
        except Exception:
            pass

    return json.dumps(profile, indent=2)


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _top_features(df: pd.DataFrame, target: Optional[str], max_count: int = 5) -> list[str]:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return []

    if target and target in numeric_df.columns:
        try:
            corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
            return [c for c in corr.index.tolist() if c != target][:max_count]
        except Exception:
            pass

    return numeric_df.std().sort_values(ascending=False).head(max_count).index.tolist()


def _sanitize_plan(plan: list[dict], df: pd.DataFrame) -> list[dict]:
    cleaned: list[dict] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(plan):
        ptype = str(item.get("plot_type", "")).strip().lower()
        if ptype not in SUPPORTED_PLOT_TYPES:
            continue

        plot_id = str(item.get("plot_id", f"plot_{i+1}")).strip() or f"plot_{i+1}"
        if plot_id in seen_ids:
            plot_id = f"{plot_id}_{i+1}"
        seen_ids.add(plot_id)

        columns = item.get("columns", [])
        if not isinstance(columns, list):
            columns = []
        columns = _safe_cols(df, [str(c) for c in columns])

        params = item.get("params", {})
        if not isinstance(params, dict):
            params = {}

        cleaned.append(
            {
                "plot_id": plot_id,
                "plot_type": ptype,
                "title": str(item.get("title", ptype)).strip() or ptype,
                "description": str(item.get("description", "")).strip(),
                "columns": columns,
                "params": params,
            }
        )

    return cleaned


def _ask_groq(profile_json: str) -> list[dict]:
    if not settings.GROQ_API_KEY:
        return []

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=2500,
        )
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"Dataset profile:\n{profile_json}"),
            ]
        )
        return _safe_json_array(str(response.content))
    except Exception:
        return []


def _ask_anthropic(profile_json: str) -> list[dict]:
    api_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Dataset profile:\n{profile_json}"}],
        )
        raw = getattr(response.content[0], "text", "") if response.content else ""
        return _safe_json_array(raw)
    except Exception:
        return []


def _guaranteed_plan(df: pd.DataFrame, target: Optional[str], problem_type: str) -> list[dict]:
    """
    Build the guaranteed deterministic plan.
    These plots are always included, regardless of AI availability.
    """
    plan: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    top_num = [c for c in _top_features(df, target, max_count=6) if c in numeric_cols]

    plan.append(
        {
            "plot_id": "missing_values",
            "plot_type": "missingvalues",
            "title": "Missing Values Overview",
            "description": "Highlights columns with missing data and their ratios.",
            "columns": [],
            "params": {},
        }
    )

    if len(numeric_cols) >= 3:
        plan.append(
            {
                "plot_id": "correlation_heatmap",
                "plot_type": "heatmap",
                "title": "Correlation Heatmap",
                "description": "Pairwise linear relationships among numeric features.",
                "columns": [],
                "params": {"method": "pearson"},
            }
        )

    if top_num:
        plan.append(
            {
                "plot_id": "distribution_grid",
                "plot_type": "distribution_grid",
                "title": "Distribution Grid - Numeric Features",
                "description": "Histograms and KDE for top numeric features.",
                "columns": top_num[:6],
                "params": {"max_cols": 6},
            }
        )
        plan.append(
            {
                "plot_id": "outlier_detection",
                "plot_type": "boxplot",
                "title": "Outlier Detection",
                "description": "Boxplots to surface outliers in key numeric features.",
                "columns": top_num[:5],
                "params": {},
            }
        )

    if problem_type == "Regression" and target and target in df.columns:
        plan.append(
            {
                "plot_id": "target_distribution",
                "plot_type": "histogram",
                "title": f"Target Distribution - {target}",
                "description": "Spread and skew of the regression target.",
                "columns": [target],
                "params": {"kde": True},
            }
        )

        for feat in [f for f in top_num if f != target][:3]:
            plan.append(
                {
                    "plot_id": f"target_vs_{feat}",
                    "plot_type": "target_vs_feature",
                    "title": f"{target} vs {feat}",
                    "description": f"Scatter + trendline: how {feat} relates to {target}.",
                    "columns": [feat],
                    "params": {},
                }
            )

        if top_num:
            usable = [f for f in top_num if f != target]
            feat = usable[0] if usable else top_num[0]
            plan.append(
                {
                    "plot_id": "regression_residual",
                    "plot_type": "regression_residual",
                    "title": "Residual Diagnostics (Residual vs Fitted + Distribution)",
                    "description": "Checks randomness and normality of residuals.",
                    "columns": [feat],
                    "params": {},
                }
            )
            plan.append(
                {
                    "plot_id": "qq_plot",
                    "plot_type": "qq_plot",
                    "title": "Q-Q Plot of Residuals",
                    "description": "Checks if residuals follow a normal distribution.",
                    "columns": [feat],
                    "params": {},
                }
            )
            plan.append(
                {
                    "plot_id": "scale_location",
                    "plot_type": "scale_location",
                    "title": "Scale-Location Plot",
                    "description": "Checks homoscedasticity (constant residual variance).",
                    "columns": [feat],
                    "params": {},
                }
            )

    if problem_type == "Classification" and target and target in df.columns:
        plan.append(
            {
                "plot_id": "class_balance",
                "plot_type": "class_balance",
                "title": f"Class Balance - {target}",
                "description": "Frequency and proportion of each target class.",
                "columns": [target],
                "params": {},
            }
        )

        for feat in [f for f in top_num if f != target][:4]:
            plan.append(
                {
                    "plot_id": f"class_vs_{feat}",
                    "plot_type": "target_vs_feature",
                    "title": f"{target} vs {feat}",
                    "description": f"Boxplot: does {feat} differ across {target} classes?",
                    "columns": [feat],
                    "params": {},
                }
            )

        binary_cats = [
            c
            for c in cat_cols
            if c != target and df[c].nunique(dropna=True) <= 5
        ]
        for cat in binary_cats[:3]:
            plan.append(
                {
                    "plot_id": f"stacked_{cat}_vs_target",
                    "plot_type": "stackedbar",
                    "title": f"{cat} x {target}",
                    "description": f"Does {cat} help separate {target}?",
                    "columns": [cat, target],
                    "params": {"normalize": True},
                }
            )

        pairplot_cols = [f for f in top_num if f != target][:4]
        if pairplot_cols:
            plan.append(
                {
                    "plot_id": "pairplot_by_class",
                    "plot_type": "pairplot",
                    "title": f"Pairplot (hue = {target})",
                    "description": "Feature separability between classes at a glance.",
                    "columns": pairplot_cols,
                    "params": {"hue": target, "max_cols": 4},
                }
            )

    if problem_type == "Clustering" and len(numeric_cols) >= 2:
        plan.append(
            {
                "plot_id": "pca_scatter",
                "plot_type": "pca_scatter",
                "title": "PCA 2D Projection",
                "description": "Visual check for natural cluster structure.",
                "columns": [],
                "params": {},
            }
        )
        plan.append(
            {
                "plot_id": "elbow_curve",
                "plot_type": "elbow_curve",
                "title": "Elbow Curve - Optimal K",
                "description": "WCSS vs K to estimate the best cluster count.",
                "columns": [],
                "params": {},
            }
        )
        plan.append(
            {
                "plot_id": "silhouette_plot",
                "plot_type": "silhouette_plot",
                "title": "Silhouette Score vs K",
                "description": "Cohesion and separation quality across K values.",
                "columns": [],
                "params": {},
            }
        )
        plan.append(
            {
                "plot_id": "tsne_scatter",
                "plot_type": "tsne_scatter",
                "title": "t-SNE 2D Projection",
                "description": "Non-linear structure for cluster visualization.",
                "columns": [],
                "params": {},
            }
        )

    return _sanitize_plan(plan, df)


def _supplementary_plan(df: pd.DataFrame, target: Optional[str], problem_type: str) -> list[dict]:
    profile_json = _build_profile(df, target, problem_type)
    ai_plan = _ask_anthropic(profile_json) or _ask_groq(profile_json)
    return _sanitize_plan(ai_plan, df)


def _execute_plot(df: pd.DataFrame, spec: dict, target: Optional[str]) -> Optional[Figure]:
    ptype = spec.get("plot_type", "")
    cols = _safe_cols(df, spec.get("columns", []))
    params = spec.get("params", {}) or {}
    title = spec.get("title", ptype)
    numeric_df = df.select_dtypes(include="number")

    try:
        if ptype == "histogram":
            if not cols:
                return None
            col = cols[0]
            if not pd.api.types.is_numeric_dtype(df[col]):
                return None
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(
                df[col].dropna(),
                bins=params.get("bins", "auto"),
                kde=params.get("kde", True),
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel(col)
            return fig

        if ptype == "kde":
            if not cols:
                return None
            col = cols[0]
            if not pd.api.types.is_numeric_dtype(df[col]):
                return None
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.kdeplot(df[col].dropna(), fill=True, ax=ax)
            ax.set_title(title)
            return fig

        if ptype == "boxplot":
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:6]
            if not num_cols:
                return None
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df[num_cols], orient="h", ax=ax, palette="muted")
            ax.set_title(title)
            return fig

        if ptype == "violinplot":
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return None
            fig, ax = plt.subplots(figsize=(8, 5))
            x_col = cat_cols[0] if cat_cols else None
            if x_col and df[x_col].nunique(dropna=True) <= 15:
                sns.violinplot(
                    data=df,
                    x=x_col,
                    y=num_cols[0],
                    ax=ax,
                    palette="muted",
                    inner="quartile",
                )
            else:
                sns.violinplot(data=df[[num_cols[0]]], ax=ax, palette="muted")
            ax.set_title(title)
            return fig

        if ptype == "barplot":
            cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if not cat_cols or not num_cols:
                return None
            cat, num = cat_cols[0], num_cols[0]
            if df[cat].nunique(dropna=True) > 30:
                return None
            estimator = params.get("estimator", "mean")
            if estimator not in {"mean", "sum", "count"}:
                estimator = "mean"
            agg = df.groupby(cat, dropna=False)[num].agg(estimator).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            agg.plot(kind="bar", ax=ax, color=sns.color_palette("muted"))
            ax.set_title(title)
            ax.set_xlabel(cat)
            ax.set_ylabel(f"{estimator}({num})")
            plt.xticks(rotation=45, ha="right")
            return fig

        if ptype == "countplot":
            if not cols:
                return None
            col = cols[0]
            top_n = int(params.get("top_n", 15))
            top_vals = df[col].astype(str).value_counts().head(top_n).index
            plot_df = df[df[col].astype(str).isin(top_vals)]
            fig, ax = plt.subplots(figsize=(8, 5))
            order = plot_df[col].astype(str).value_counts().index
            sns.countplot(data=plot_df, y=col, order=order, ax=ax, palette="muted")
            ax.set_title(title)
            return fig

        if ptype == "scatterplot":
            if len(cols) < 2:
                return None
            x_col, y_col = cols[0], cols[1]
            if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
                return None
            hue = params.get("hue")
            if hue and hue not in df.columns:
                hue = None
            fig, ax = plt.subplots(figsize=(7, 5))
            sample = df.sample(min(2000, len(df)), random_state=42)
            sns.scatterplot(data=sample, x=x_col, y=y_col, hue=hue, alpha=0.6, ax=ax, palette="muted")
            ax.set_title(title)
            return fig

        if ptype == "lineplot":
            if len(cols) < 2:
                return None
            x_col, y_col = cols[0], cols[1]
            hue = params.get("hue")
            if hue and hue not in df.columns:
                hue = None
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
            ax.set_title(title)
            return fig

        if ptype == "heatmap":
            if numeric_df.shape[1] < 2:
                return None
            method = params.get("method", "pearson")
            if method not in {"pearson", "spearman"}:
                method = "pearson"
            corr = numeric_df.corr(method=method)
            fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.7), max(5, len(corr) * 0.6)))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap=PLOT_PALETTE,
                ax=ax,
                square=True,
                linewidths=0.5,
            )
            ax.set_title(title)
            plt.tight_layout()
            return fig

        if ptype == "pairplot":
            max_cols = min(int(params.get("max_cols", 5)), 6)
            hue = params.get("hue")
            if hue and hue not in df.columns:
                hue = None
            use_cols = _safe_cols(df, cols)[:max_cols]
            if not use_cols:
                use_cols = numeric_df.columns[:max_cols].tolist()
            if not use_cols:
                return None
            pair_cols = use_cols + ([hue] if hue and hue not in use_cols else [])
            plot_df = df[pair_cols].dropna()
            if plot_df.empty:
                return None
            sample = plot_df.sample(min(500, len(plot_df)), random_state=42)
            g = sns.pairplot(sample, hue=hue, palette="muted", diag_kind="kde", plot_kws={"alpha": 0.5})
            g.figure.suptitle(title, y=1.01)
            return g.figure

        if ptype == "piechart":
            if not cols:
                return None
            col = cols[0]
            top_n = int(params.get("top_n", 8))
            vc = df[col].astype(str).value_counts().head(top_n)
            if vc.empty:
                return None
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=sns.color_palette("muted", len(vc)))
            ax.set_title(title)
            return fig

        if ptype == "stackedbar":
            if len(cols) < 2:
                return None
            cat_col, hue_col = cols[0], cols[1]
            if df[cat_col].nunique(dropna=True) > 20 or df[hue_col].nunique(dropna=True) > 10:
                return None
            ct = pd.crosstab(df[cat_col], df[hue_col])
            if params.get("normalize", False):
                ct = ct.div(ct.sum(axis=1), axis=0)
            fig, ax = plt.subplots(figsize=(9, 5))
            ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="white")
            ax.set_title(title)
            plt.xticks(rotation=45, ha="right")
            ax.legend(loc="upper right")
            return fig

        if ptype == "missingvalues":
            missing = df.isna().sum().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(4, len(missing) * 0.35)))
            pct = (missing / max(1, len(df)) * 100).round(2)
            bars = ax.barh(missing.index, pct.values, color=sns.color_palette("muted"))
            ax.set_xlabel("Missing %")
            ax.set_title(title)
            for bar, val in zip(bars, pct.values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
            plt.tight_layout()
            return fig

        if ptype == "outlierdetection":
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:6]
            if not num_cols:
                num_cols = numeric_df.columns[:6].tolist()
            if not num_cols:
                return None
            fig, axes = plt.subplots(1, len(num_cols), figsize=(max(7, len(num_cols) * 2.5), 4), sharey=False)
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                data = df[col].dropna()
                if data.empty:
                    continue
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = data[(data < low) | (data > high)]
                inliers = data[(data >= low) & (data <= high)]
                ax.scatter(range(len(inliers)), inliers, s=5, alpha=0.5, color="steelblue", label="Normal")
                ax.scatter(range(len(outliers)), outliers, s=20, alpha=0.8, color="crimson", label="Outlier")
                ax.set_title(col, fontsize=9)
                ax.legend(fontsize=7)
            fig.suptitle(title, fontsize=11)
            plt.tight_layout()
            return fig

        if ptype == "distribution_grid":
            max_cols = int(params.get("max_cols", 8))
            use_cols = _safe_cols(df, cols)
            if not use_cols:
                use_cols = numeric_df.columns.tolist()
            use_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])][:max_cols]
            if not use_cols:
                return None
            n = len(use_cols)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
            axes = np.array(axes).flatten()
            i = -1
            for i, col in enumerate(use_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
                axes[i].set_title(col, fontsize=9)
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle(title, fontsize=11)
            plt.tight_layout()
            return fig

        if ptype == "target_vs_feature" and target and target in df.columns:
            if not cols:
                return None
            feat = cols[0]
            if feat not in df.columns:
                return None
            fig, ax = plt.subplots(figsize=(7, 5))
            feat_num = pd.api.types.is_numeric_dtype(df[feat])
            target_num = pd.api.types.is_numeric_dtype(df[target])

            if feat_num and target_num:
                sub = df[[feat, target]].dropna()
                if sub.empty:
                    return None
                sample = sub.sample(min(1000, len(sub)), random_state=42)
                sns.scatterplot(data=sample, x=feat, y=target, alpha=0.5, ax=ax)
                coef = np.polyfit(sample[feat].values, sample[target].values, 1)
                xs = np.sort(sample[feat].values)
                ax.plot(xs, np.polyval(coef, xs), color="crimson", linewidth=1.5, linestyle="--")
            elif (not feat_num) and target_num:
                if df[feat].nunique(dropna=True) <= 20:
                    sns.boxplot(data=df, x=feat, y=target, ax=ax, palette="muted")
                    plt.xticks(rotation=45, ha="right")
                else:
                    return None
            elif feat_num and (not target_num):
                if df[target].nunique(dropna=True) <= 10:
                    sns.boxplot(data=df, x=target, y=feat, ax=ax, palette="muted")
                    plt.xticks(rotation=45, ha="right")
                else:
                    return None
            else:
                return None

            ax.set_title(title)
            return fig

        if ptype == "class_balance" and target and target in df.columns:
            vc = df[target].astype(str).value_counts()
            if vc.empty:
                return None
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(vc.index, vc.values, color=sns.color_palette("muted", len(vc)))
            axes[0].set_title("Class Counts")
            axes[0].set_xlabel(target)
            axes[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=sns.color_palette("muted", len(vc)))
            axes[1].set_title("Class Proportions")
            fig.suptitle(title)
            plt.tight_layout()
            return fig

        if ptype == "regression_residual" and target and target in df.columns:
            if not cols:
                return None
            feat = cols[0]
            if feat not in df.columns:
                return None
            if not (pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target])):
                return None
            sub = df[[feat, target]].dropna()
            if sub.empty:
                return None
            coef = np.polyfit(sub[feat], sub[target], 1)
            pred = np.polyval(coef, sub[feat])
            residuals = sub[target].values - pred
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].scatter(pred, residuals, alpha=0.4, s=15, color="steelblue")
            axes[0].axhline(0, color="crimson", linestyle="--")
            axes[0].set_xlabel("Fitted Values")
            axes[0].set_ylabel("Residual")
            axes[0].set_title("Residual vs Fitted")
            sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
            axes[1].set_title("Residual Distribution")
            fig.suptitle(title)
            plt.tight_layout()
            return fig

        if ptype == "qq_plot" and target and target in df.columns:
            if not cols:
                return None
            feat = cols[0]
            if feat not in df.columns:
                return None
            if not (pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target])):
                return None
            sub = df[[feat, target]].dropna()
            if sub.empty:
                return None
            coef = np.polyfit(sub[feat], sub[target], 1)
            residuals = sub[target].values - np.polyval(coef, sub[feat])
            fig, ax = plt.subplots(figsize=(6, 5))
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title(title)
            return fig

        if ptype == "scale_location" and target and target in df.columns:
            if not cols:
                return None
            feat = cols[0]
            if feat not in df.columns:
                return None
            if not (pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target])):
                return None
            sub = df[[feat, target]].dropna()
            if sub.empty:
                return None
            coef = np.polyfit(sub[feat], sub[target], 1)
            pred = np.polyval(coef, sub[feat])
            residuals = sub[target].values - pred
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(pred, np.sqrt(np.abs(residuals)), alpha=0.4, s=15, color="steelblue")
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("sqrt(|Residuals|)")
            ax.set_title(title)
            return fig

        if ptype == "feature_importance_proxy":
            num_df = numeric_df.drop(columns=[target], errors="ignore")
            if num_df.empty:
                return None
            scaled = StandardScaler().fit_transform(num_df.fillna(num_df.median()))
            var = pd.Series(scaled.var(axis=0), index=num_df.columns).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(4, len(var) * 0.35)))
            ax.barh(var.index, var.values, color=sns.color_palette("muted", len(var)))
            ax.set_xlabel("Scaled Variance (proxy importance)")
            ax.set_title(title)
            plt.tight_layout()
            return fig

        if ptype == "pca_scatter":
            num_df = numeric_df.drop(columns=[target], errors="ignore") if target else numeric_df
            if num_df.shape[1] < 2:
                return None
            clean = num_df.fillna(num_df.median())
            x_scaled = StandardScaler().fit_transform(clean)
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(x_scaled)
            ev = pca.explained_variance_ratio_
            hue_col = params.get("hue")
            colors = None
            if hue_col and hue_col in df.columns:
                colors = LabelEncoder().fit_transform(df[hue_col].astype(str))
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, cmap="tab10", alpha=0.6, s=20)
            if colors is not None:
                plt.colorbar(sc, ax=ax, label=hue_col)
            ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)")
            ax.set_title(title)
            return fig

        if ptype == "elbow_curve":
            num_df = numeric_df.drop(columns=[target], errors="ignore") if target else numeric_df
            if num_df.shape[1] < 2:
                return None
            clean = num_df.fillna(num_df.median())
            if len(clean) < 3:
                return None
            scaled = StandardScaler().fit_transform(clean)
            max_k = min(10, len(clean) - 1)
            if max_k < 2:
                return None
            k_range = list(range(2, max_k + 1))
            inertias: list[float] = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                km.fit(scaled)
                inertias.append(float(km.inertia_))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(k_range, inertias, marker="o", color="steelblue")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("WCSS (Inertia)")
            ax.set_title(title)
            return fig

        if ptype == "silhouette_plot":
            num_df = numeric_df.drop(columns=[target], errors="ignore") if target else numeric_df
            if num_df.shape[1] < 2:
                return None
            clean = num_df.fillna(num_df.median())
            if len(clean) < 3:
                return None
            scaled = StandardScaler().fit_transform(clean)
            max_k = min(10, len(clean) - 1)
            if max_k < 2:
                return None
            k_range = list(range(2, max_k + 1))
            scores: list[float] = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = km.fit_predict(scaled)
                if len(np.unique(labels)) < 2:
                    scores.append(0.0)
                    continue
                try:
                    scores.append(float(silhouette_score(scaled, labels)))
                except Exception:
                    scores.append(0.0)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(k_range, scores, marker="s", color="crimson")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title(title)
            return fig

        if ptype == "tsne_scatter":
            num_df = numeric_df.drop(columns=[target], errors="ignore") if target else numeric_df
            if num_df.shape[1] < 2:
                return None
            clean = num_df.fillna(num_df.median())
            if len(clean) < 3:
                return None
            scaled = StandardScaler().fit_transform(clean)
            sample_size = min(500, len(scaled))
            if sample_size < 3:
                return None
            sample_idx = np.random.RandomState(42).choice(len(scaled), sample_size, replace=False)
            perplexity = min(30, sample_size - 1)
            if perplexity < 2:
                return None
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedded = tsne.fit_transform(scaled[sample_idx])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.6, s=20, color="steelblue")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title(title)
            return fig

        if ptype == "cluster_size_bar":
            num_df = numeric_df.drop(columns=[target], errors="ignore") if target else numeric_df
            if num_df.shape[1] < 2:
                return None
            clean = num_df.fillna(num_df.median())
            if len(clean) < 2:
                return None
            scaled = StandardScaler().fit_transform(clean)
            requested_k = int(params.get("k", 3))
            k = min(max(2, requested_k), len(scaled))
            if k < 2:
                return None
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(scaled)
            sizes = pd.Series(labels).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([f"Cluster {i}" for i in sizes.index], sizes.values, color=sns.color_palette("muted", len(sizes)))
            ax.set_ylabel("Count")
            ax.set_title(title)
            return fig

    except Exception:
        warnings.warn(f"[EDA] Failed to render '{title}': {traceback.format_exc()}")
        return None

    return None


def perform_eda(
    df: pd.DataFrame,
    target: Optional[str] = None,
    problem_type: str = "Regression",
) -> tuple[dict, list[dict]]:
    """
    Run EDA and return summary stats plus rendered figures.

    Parameters
    ----------
    df: Input dataframe.
    target: Target column name (None for unsupervised).
    problem_type: Regression, Classification, or Clustering.
    """
    warnings.filterwarnings("ignore")

    core_plan = _guaranteed_plan(df, target, problem_type)
    core_ids = {spec["plot_id"] for spec in core_plan}

    ai_plan = _supplementary_plan(df, target, problem_type)
    ai_plan = [spec for spec in ai_plan if spec["plot_id"] not in core_ids]

    full_plan = core_plan + ai_plan

    figures: list[dict] = []
    for spec in full_plan[:20]:
        fig = _execute_plot(df, spec, target)
        if fig is not None:
            figures.append(
                {
                    "figure": fig,
                    "heading": spec.get("title", spec.get("plot_type", "")),
                    "description": spec.get("description", ""),
                    "plot_id": spec.get("plot_id", ""),
                    "plot_type": spec.get("plot_type", ""),
                }
            )

    results: dict[str, Any] = {
        "description": df.describe().to_dict(),
        "columns": df.columns.tolist(),
        "top_correlated_features": _top_features(df, target, max_count=5),
        "core_plot_plan": core_plan,
        "ai_plot_plan": ai_plan,
        "plot_plan_source": "core_plus_ai" if ai_plan else "deterministic",
        "total_plots": len(figures),
    }

    return results, figures
