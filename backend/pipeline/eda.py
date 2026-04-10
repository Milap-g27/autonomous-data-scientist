"""
AI-guided EDA module with safe fallback.

Behavior:
- Tries to generate a plot plan via LLM (Groq by default, optional Anthropic).
- Validates and executes only supported plot specs.
- Falls back to a deterministic plan if AI planning fails.
- Always returns stable output shape expected by frontend/backend.
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
from sklearn.decomposition import PCA
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
    "class_balance",
    "regression_residual",
    "feature_importance_proxy",
    "pca_scatter",
}


_SYSTEM_PROMPT = """
You are an expert data scientist focused on practical EDA.
Given a dataset profile, return ONLY a JSON array of plot specs.

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
feature_importance_proxy, pca_scatter

Rules:
- Generate 8 to 14 plots.
- Use ONLY columns present in the profile.
- Include heatmap when numeric feature count >= 3.
- Include missingvalues when any missing values exist.
- For Classification include class_balance.
- For Regression include regression_residual.
- For Clustering include pca_scatter.
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
            return []

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
                info["min"] = round(float(s.min()), 5)
                info["max"] = round(float(s.max()), 5)
                info["mean"] = round(float(s.mean()), 5)
                info["std"] = round(float(s.std()), 5)
                info["skew"] = round(float(s.skew()), 5)
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


def _ask_groq_for_plot_plan(profile_json: str) -> list[dict]:
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


def _ask_anthropic_for_plot_plan(profile_json: str) -> list[dict]:
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
        if not response.content:
            return []
        raw = getattr(response.content[0], "text", "")
        return _safe_json_array(raw)
    except Exception:
        return []


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


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

        title = str(item.get("title", ptype)).strip() or ptype
        description = str(item.get("description", "")).strip()
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
                "title": title,
                "description": description,
                "columns": columns,
                "params": params,
            }
        )

    return cleaned


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


def _default_plot_plan(df: pd.DataFrame, target: Optional[str], problem_type: str) -> list[dict]:
    plan: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    has_missing = int(df.isna().sum().sum()) > 0

    top_num = [c for c in _top_features(df, target, max_count=6) if c in numeric_cols]

    if target and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            plan.append(
                {
                    "plot_id": "target_distribution",
                    "plot_type": "histogram",
                    "title": f"Distribution of {target}",
                    "description": "Shows spread and skew of the target variable.",
                    "columns": [target],
                    "params": {"kde": True},
                }
            )
        else:
            plan.append(
                {
                    "plot_id": "class_balance",
                    "plot_type": "class_balance",
                    "title": f"Class Balance for {target}",
                    "description": "Shows target class frequency and proportion.",
                    "columns": [target],
                    "params": {},
                }
            )

    if len(numeric_cols) >= 3:
        plan.append(
            {
                "plot_id": "correlation_heatmap",
                "plot_type": "heatmap",
                "title": "Correlation Heatmap",
                "description": "Shows pairwise linear relationships among numeric features.",
                "columns": [],
                "params": {"method": "pearson"},
            }
        )

    if has_missing:
        plan.append(
            {
                "plot_id": "missing_values",
                "plot_type": "missingvalues",
                "title": "Missing Values Overview",
                "description": "Highlights columns with missing-value ratios.",
                "columns": [],
                "params": {},
            }
        )

    if top_num:
        plan.append(
            {
                "plot_id": "outliers",
                "plot_type": "boxplot",
                "title": "Outlier Detection (Boxplots)",
                "description": "Highlights outliers in important numeric features.",
                "columns": top_num[:4],
                "params": {},
            }
        )
        plan.append(
            {
                "plot_id": "distribution_grid",
                "plot_type": "distribution_grid",
                "title": "Distribution Grid of Top Numeric Features",
                "description": "Shows histograms/KDE for high-signal numeric features.",
                "columns": top_num[:6],
                "params": {"max_cols": 6},
            }
        )

    if problem_type == "Clustering" and len(numeric_cols) >= 2:
        plan.append(
            {
                "plot_id": "pca_scatter",
                "plot_type": "pca_scatter",
                "title": "PCA Projection",
                "description": "2D PCA projection for cluster structure inspection.",
                "columns": [],
                "params": {},
            }
        )

    if problem_type in {"Regression", "Classification"} and target:
        for feat in top_num[:3]:
            if feat == target:
                continue
            plan.append(
                {
                    "plot_id": f"target_vs_{feat}",
                    "plot_type": "target_vs_feature",
                    "title": f"{target} vs {feat}",
                    "description": "Compares target behavior against an important feature.",
                    "columns": [feat],
                    "params": {},
                }
            )

        if problem_type == "Regression" and top_num:
            plan.append(
                {
                    "plot_id": "regression_residual",
                    "plot_type": "regression_residual",
                    "title": "Residual Diagnostics",
                    "description": "Checks residual spread and distribution for regression fit.",
                    "columns": [top_num[0]],
                    "params": {},
                }
            )

    if cat_cols:
        plan.append(
            {
                "plot_id": "top_category_counts",
                "plot_type": "countplot",
                "title": f"Top Categories in {cat_cols[0]}",
                "description": "Shows most frequent categories in a key categorical feature.",
                "columns": [cat_cols[0]],
                "params": {"top_n": 12},
            }
        )

    return _sanitize_plan(plan, df)


def _execute_plot(df: pd.DataFrame, spec: dict, target: Optional[str]) -> Optional[plt.Figure]:
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
                sns.violinplot(data=df, x=x_col, y=num_cols[0], ax=ax, palette="muted", inner="quartile")
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
            fig, ax = plt.subplots(
                figsize=(max(6, len(corr) * 0.7), max(5, len(corr) * 0.6))
            )
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
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if missing.empty:
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
            method = params.get("method", "zscore")
            fig, axes = plt.subplots(1, len(num_cols), figsize=(max(7, len(num_cols) * 2.5), 4), sharey=False)
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                data = df[col].dropna()
                if data.empty:
                    continue
                if method == "zscore":
                    std = data.std()
                    if std == 0 or np.isnan(std):
                        inliers, outliers = data, data.iloc[0:0]
                    else:
                        z = np.abs((data - data.mean()) / std)
                        outliers = data[z > 3]
                        inliers = data[z <= 3]
                else:
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
            max_c = int(params.get("max_cols", 8))
            use_cols = _safe_cols(df, cols)
            if not use_cols:
                use_cols = numeric_df.columns.tolist()
            use_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])][:max_c]
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
                    sns.kdeplot(data=df, x=feat, hue=target, ax=ax, fill=True, alpha=0.4)
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
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Residual")
            axes[0].set_title("Residual vs Fitted")
            sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
            axes[1].set_title("Residual Distribution")
            fig.suptitle(title)
            plt.tight_layout()
            return fig

        if ptype == "feature_importance_proxy":
            num_df = numeric_df.drop(columns=[target], errors="ignore")
            if num_df.empty:
                return None
            scaler = StandardScaler()
            scaled = scaler.fit_transform(num_df.fillna(num_df.median()))
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
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(clean)
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(x_scaled)
            ev = pca.explained_variance_ratio_
            hue_col = params.get("hue")
            colors = None
            if hue_col and hue_col in df.columns:
                le = LabelEncoder()
                colors = le.fit_transform(df[hue_col].astype(str))
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, cmap="tab10", alpha=0.6, s=20)
            if colors is not None:
                plt.colorbar(sc, ax=ax, label=hue_col)
            ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)")
            ax.set_title(title)
            return fig

    except Exception:
        warnings.warn(f"[EDA] Failed to render plot '{title}': {traceback.format_exc()}")
        return None

    return None


def perform_eda(
    df: pd.DataFrame,
    target: Optional[str] = None,
    problem_type: str = "Regression",
) -> tuple[dict, list[dict]]:
    """
    Run EDA and return summary + rendered figures.

    Returns
    -------
    results : dict
        Includes description, columns, top_correlated_features, and plot plan metadata.
    figures : list[dict]
        Each item has figure, heading, description, plot_id, plot_type.
    """
    warnings.filterwarnings("ignore")

    profile_json = _build_profile(df, target, problem_type)

    plan_source = "deterministic"
    plan = _ask_anthropic_for_plot_plan(profile_json)
    if plan:
        plan_source = "anthropic"
    else:
        groq_plan = _ask_groq_for_plot_plan(profile_json)
        if groq_plan:
            plan = groq_plan
            plan_source = "groq"

    plan = _sanitize_plan(plan, df)
    if not plan:
        plan = _default_plot_plan(df, target, problem_type)
        plan_source = "deterministic"

    figures: list[dict] = []
    for spec in plan[:18]:
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

    if not figures and plan_source != "deterministic":
        fallback_plan = _default_plot_plan(df, target, problem_type)
        for spec in fallback_plan:
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
        plan = fallback_plan
        plan_source = "deterministic"

    results: dict[str, Any] = {
        "description": df.describe().to_dict(),
        "columns": df.columns.tolist(),
        "top_correlated_features": _top_features(df, target, max_count=5),
        "ai_plot_plan": plan,
        "plot_plan_source": plan_source,
    }

    return results, figures
