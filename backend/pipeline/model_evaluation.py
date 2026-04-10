from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize


sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)


def _is_supported_param_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_supported_param_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.ndim <= 1 and value.size <= 50
    return False


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize_param_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_param_value(item) for item in value]
    return value


def _param_values_equal(left: Any, right: Any) -> bool:
    try:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        return bool(left == right)
    except Exception:
        return False


def extract_model_best_params(models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract a compact, JSON-safe parameter map for each trained model.

    For tuned models, this usually captures non-default hyperparameters selected by search.
    If non-default extraction is empty, falls back to serializable current parameters.
    """
    params_by_model: Dict[str, Dict[str, Any]] = {}

    for model_name, fitted in models.items():
        estimator = fitted
        if hasattr(fitted, "named_steps") and "model" in fitted.named_steps:
            estimator = fitted.named_steps["model"]

        if not hasattr(estimator, "get_params"):
            params_by_model[model_name] = {}
            continue

        try:
            current_params = estimator.get_params(deep=False)
        except Exception:
            params_by_model[model_name] = {}
            continue

        defaults: Dict[str, Any] = {}
        try:
            defaults = estimator.__class__().get_params(deep=False)
        except Exception:
            defaults = {}

        selected: Dict[str, Any] = {}
        for key in sorted(current_params.keys()):
            value = current_params[key]
            if not _is_supported_param_value(value):
                continue

            if key in defaults and _param_values_equal(value, defaults[key]):
                continue

            selected[key] = _normalize_param_value(value)

        if not selected:
            for key in sorted(current_params.keys()):
                value = current_params[key]
                if _is_supported_param_value(value):
                    selected[key] = _normalize_param_value(value)

        if len(selected) > 30:
            selected = dict(list(selected.items())[:30])

        params_by_model[model_name] = selected

    return params_by_model


def _cluster_labels(model: Any, X_scaled: np.ndarray) -> Optional[np.ndarray]:
    try:
        if hasattr(model, "labels_"):
            labels = np.asarray(model.labels_)
            if labels.shape[0] == X_scaled.shape[0]:
                return labels
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X_scaled))
    except Exception:
        return None
    return None


def _scaled_numeric_matrix(model: Any, X_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    numeric_df = X_df.select_dtypes(include=["number"], exclude=["bool"])
    scaler = getattr(model, "_scaler", None)
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
    else:
        scaled = scaler.transform(numeric_df)
    return numeric_df, np.asarray(scaled)


def evaluate_models(models: Dict[str, Any], X_test, y_test, problem_type: str) -> tuple[Dict[str, Any], str]:
    """
    Evaluate trained models on holdout data.

    For clustering, X_test is expected to be the full feature matrix.
    Returns per-model metrics and the selected best model name.
    """
    metrics: Dict[str, Any] = {}
    best_score = -float("inf")
    best_model_name = ""

    if problem_type == "Clustering":
        X_full = X_test
        if X_full is None or len(X_full) < 2:
            return metrics, best_model_name

        for name, model in models.items():
            try:
                numeric_df, X_scaled = _scaled_numeric_matrix(model, X_full)
                if numeric_df.shape[1] < 1 or X_scaled.shape[0] < 2:
                    continue

                labels = _cluster_labels(model, X_scaled)
                if labels is None:
                    continue

                n_clusters = len(set(labels.tolist()) - {-1})
                n_noise = int(np.sum(labels == -1))

                if n_clusters < 2:
                    metrics[name] = {
                        "Clusters": n_clusters,
                        "Noise Points": n_noise,
                        "Silhouette Score": "N/A (< 2 clusters)",
                        "Calinski-Harabasz": "N/A",
                        "Davies-Bouldin": "N/A",
                    }
                    continue

                mask = labels != -1
                if int(mask.sum()) < 2 or len(set(labels[mask].tolist())) < 2:
                    continue

                sil = float(silhouette_score(X_scaled[mask], labels[mask]))
                ch = float(calinski_harabasz_score(X_scaled[mask], labels[mask]))
                db = float(davies_bouldin_score(X_scaled[mask], labels[mask]))

                metrics[name] = {
                    "Clusters": n_clusters,
                    "Noise Points": n_noise,
                    "Silhouette Score": round(sil, 4),
                    "Calinski-Harabasz": round(ch, 2),
                    "Davies-Bouldin": round(db, 4),
                }

                if sil > best_score:
                    best_score = sil
                    best_model_name = name
            except Exception:
                continue

        if not best_model_name and models:
            best_model_name = list(models.keys())[0]

        return metrics, best_model_name

    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
        except Exception:
            continue

        if problem_type == "Regression":
            mse = float(mean_squared_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))

            metrics[name] = {
                "MSE": mse,
                "R2": r2,
                "MAE": mae,
            }

            if r2 > best_score:
                best_score = r2
                best_model_name = name
        else:
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            metrics[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

            if f1 > best_score:
                best_score = f1
                best_model_name = name

    return metrics, best_model_name


def _figure_item(
    fig: Figure,
    plot_id: str,
    plot_type: str,
    heading: str,
    description: str,
) -> Dict[str, Any]:
    return {
        "figure": fig,
        "plot_id": plot_id,
        "plot_type": plot_type,
        "heading": heading,
        "description": description,
    }


def _safe_score_matrix(model: Any, X) -> tuple[Optional[np.ndarray], str]:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X)), "predict_proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X)), "decision_function"
        except Exception:
            pass
    return None, ""


def _learning_curve_figure(
    model: Any,
    X,
    y,
    scoring: str,
    title: str,
    is_classification: bool,
) -> Optional[Figure]:
    if X is None or y is None:
        return None

    n_rows = len(X)
    if n_rows < 12:
        return None

    try:
        if is_classification:
            min_class_count = int(pd.Series(y).value_counts().min())
            if min_class_count < 2:
                return None
            cv_folds = max(2, min(5, min_class_count))
        else:
            cv_folds = max(2, min(5, n_rows // 10 if n_rows >= 10 else 2))

        if cv_folds < 2:
            return None

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=clone(model),
            X=X,
            y=y,
            cv=cv_folds,
            n_jobs=-1,
            scoring=scoring,
            train_sizes=np.linspace(0.2, 1.0, 5),
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train")
        ax.plot(train_sizes, val_scores.mean(axis=1), marker="s", label="Validation")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel(scoring)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _regression_post_training(
    model: Any,
    X_test,
    y_test,
    X_train,
    y_train,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    summary: Dict[str, Any] = {}
    figures: list[Dict[str, Any]] = []

    if model is None or X_test is None or y_test is None or len(X_test) == 0:
        return summary, figures

    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    residuals = y_true - y_pred

    summary.update(
        {
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
            "test_mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
            "test_r2": round(float(r2_score(y_true, y_pred)), 6),
            "residual_mean": round(float(np.mean(residuals)), 6),
            "residual_std": round(float(np.std(residuals)), 6),
        }
    )

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(y_pred, residuals, alpha=0.5, s=18, color="steelblue")
    ax1.axhline(0.0, color="crimson", linestyle="--")
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    figures.append(
        _figure_item(
            fig1,
            "post_residual_vs_fitted",
            "regression_residual",
            "Post-Training Residuals vs Fitted",
            "Uses best-model predictions to check residual spread and bias.",
        )
    )

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Residual Q-Q Plot")
    figures.append(
        _figure_item(
            fig2,
            "post_residual_qq",
            "qq_plot",
            "Post-Training Q-Q Plot",
            "Checks normality of residuals from the selected trained model.",
        )
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.scatter(y_true, y_pred, alpha=0.5, s=18, color="teal")
    min_v = float(np.min([np.min(y_true), np.min(y_pred)]))
    max_v = float(np.max([np.max(y_true), np.max(y_pred)]))
    ax3.plot([min_v, max_v], [min_v, max_v], "--", color="crimson")
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    ax3.set_title("Actual vs Predicted")
    figures.append(
        _figure_item(
            fig3,
            "post_actual_vs_predicted",
            "prediction_scatter",
            "Post-Training Actual vs Predicted",
            "Compares model predictions against true targets on holdout data.",
        )
    )

    X_lc = X_train if X_train is not None else X_test
    y_lc = y_train if y_train is not None else y_test
    lc_fig = _learning_curve_figure(
        model=model,
        X=X_lc,
        y=y_lc,
        scoring="r2",
        title="Learning Curve (R2)",
        is_classification=False,
    )
    if lc_fig is not None:
        figures.append(
            _figure_item(
                lc_fig,
                "post_learning_curve",
                "learning_curve",
                "Post-Training Learning Curve",
                "Shows train and validation R2 as sample size increases.",
            )
        )

    return summary, figures


def _classification_post_training(
    model: Any,
    X_test,
    y_test,
    X_train,
    y_train,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    summary: Dict[str, Any] = {}
    figures: list[Dict[str, Any]] = []

    if model is None or X_test is None or y_test is None or len(X_test) == 0:
        return summary, figures

    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    labels = np.unique(np.concatenate([y_true, y_pred]))

    summary.update(
        {
            "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "test_precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
            "test_recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
            "test_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
            "classes": [str(c) for c in labels.tolist()],
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(c) for c in labels],
        yticklabels=[str(c) for c in labels],
        ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    figures.append(
        _figure_item(
            fig1,
            "post_confusion_matrix",
            "confusion_matrix",
            "Post-Training Confusion Matrix",
            "Shows class-wise prediction performance of the selected model.",
        )
    )

    scores, score_source = _safe_score_matrix(model, X_test)
    if scores is not None:
        classes_model = np.asarray(getattr(model, "classes_", labels))
        n_classes = len(labels)

        if n_classes == 2:
            positive_label = labels[-1]

            if scores.ndim == 1:
                positive_scores = scores
            else:
                idx = np.where(classes_model == positive_label)[0]
                if idx.size == 0:
                    col = min(1, scores.shape[1] - 1)
                else:
                    col = int(idx[0])
                positive_scores = scores[:, col]

            y_bin = (y_true == positive_label).astype(int)

            fpr, tpr, _ = roc_curve(y_bin, positive_scores)
            roc_auc = float(auc(fpr, tpr))
            summary["roc_auc"] = round(roc_auc, 6)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(fpr, tpr, color="steelblue", label=f"AUC={roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], "--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend(loc="lower right")
            figures.append(
                _figure_item(
                    fig2,
                    "post_roc_curve",
                    "roc_curve",
                    "Post-Training ROC Curve",
                    "Binary ROC based on holdout predictions.",
                )
            )

            precision, recall, _ = precision_recall_curve(y_bin, positive_scores)
            ap = float(average_precision_score(y_bin, positive_scores))
            summary["average_precision"] = round(ap, 6)

            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.plot(recall, precision, color="crimson", label=f"AP={ap:.3f}")
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve")
            ax3.legend(loc="lower left")
            figures.append(
                _figure_item(
                    fig3,
                    "post_pr_curve",
                    "precision_recall_curve",
                    "Post-Training Precision-Recall",
                    "Binary PR curve using holdout prediction scores.",
                )
            )

            if score_source == "predict_proba":
                try:
                    prob_true, prob_pred = calibration_curve(
                        y_bin,
                        np.clip(positive_scores, 0.0, 1.0),
                        n_bins=10,
                        strategy="quantile",
                    )
                    fig4, ax4 = plt.subplots(figsize=(7, 4))
                    ax4.plot(prob_pred, prob_true, marker="o", color="teal")
                    ax4.plot([0, 1], [0, 1], "--", color="gray")
                    ax4.set_xlabel("Mean Predicted Probability")
                    ax4.set_ylabel("Observed Positive Frequency")
                    ax4.set_title("Calibration Curve")
                    figures.append(
                        _figure_item(
                            fig4,
                            "post_calibration_curve",
                            "calibration_curve",
                            "Post-Training Calibration Curve",
                            "Checks if predicted probabilities are well calibrated.",
                        )
                    )
                except Exception:
                    pass

        elif scores.ndim == 2 and scores.shape[1] >= n_classes:
            aligned_scores = np.zeros((scores.shape[0], n_classes), dtype=float)
            for index, cls in enumerate(labels):
                idx = np.where(classes_model == cls)[0]
                if idx.size > 0:
                    aligned_scores[:, index] = scores[:, int(idx[0])]

            y_bin = label_binarize(y_true, classes=labels)
            if y_bin.shape[1] == aligned_scores.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), aligned_scores.ravel())
                roc_auc = float(auc(fpr, tpr))
                summary["roc_auc_micro"] = round(roc_auc, 6)

                fig5, ax5 = plt.subplots(figsize=(7, 4))
                ax5.plot(fpr, tpr, color="steelblue", label=f"Micro AUC={roc_auc:.3f}")
                ax5.plot([0, 1], [0, 1], "--", color="gray")
                ax5.set_xlabel("False Positive Rate")
                ax5.set_ylabel("True Positive Rate")
                ax5.set_title("Multiclass ROC (Micro)")
                ax5.legend(loc="lower right")
                figures.append(
                    _figure_item(
                        fig5,
                        "post_multiclass_roc",
                        "roc_curve",
                        "Post-Training Multiclass ROC",
                        "Micro-averaged ROC curve across all classes.",
                    )
                )

                precision, recall, _ = precision_recall_curve(y_bin.ravel(), aligned_scores.ravel())
                ap_micro = float(average_precision_score(y_bin, aligned_scores, average="micro"))
                summary["average_precision_micro"] = round(ap_micro, 6)

                fig6, ax6 = plt.subplots(figsize=(7, 4))
                ax6.plot(recall, precision, color="crimson", label=f"Micro AP={ap_micro:.3f}")
                ax6.set_xlabel("Recall")
                ax6.set_ylabel("Precision")
                ax6.set_title("Multiclass PR (Micro)")
                ax6.legend(loc="lower left")
                figures.append(
                    _figure_item(
                        fig6,
                        "post_multiclass_pr",
                        "precision_recall_curve",
                        "Post-Training Multiclass PR",
                        "Micro-averaged precision-recall curve across all classes.",
                    )
                )

    X_lc = X_train if X_train is not None else X_test
    y_lc = y_train if y_train is not None else y_test
    lc_fig = _learning_curve_figure(
        model=model,
        X=X_lc,
        y=y_lc,
        scoring="f1_weighted",
        title="Learning Curve (F1 Weighted)",
        is_classification=True,
    )
    if lc_fig is not None:
        figures.append(
            _figure_item(
                lc_fig,
                "post_learning_curve",
                "learning_curve",
                "Post-Training Learning Curve",
                "Shows train and validation F1 as sample size increases.",
            )
        )

    return summary, figures


def _clustering_post_training(model: Any, X_full) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    summary: Dict[str, Any] = {}
    figures: list[Dict[str, Any]] = []

    if model is None or X_full is None or len(X_full) < 2:
        return summary, figures

    numeric_df = X_full.select_dtypes(include=["number"], exclude=["bool"])
    if numeric_df.shape[1] < 2:
        return summary, figures

    numeric_df, X_scaled = _scaled_numeric_matrix(model, X_full)
    labels = _cluster_labels(model, X_scaled)
    if labels is None:
        return summary, figures

    label_series = pd.Series(labels)
    unique_non_noise = sorted([int(x) for x in set(label_series.tolist()) if int(x) != -1])
    n_clusters = len(unique_non_noise)
    n_noise = int((label_series == -1).sum())

    summary.update(
        {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": {str(k): int(v) for k, v in label_series.value_counts().sort_index().to_dict().items()},
        }
    )

    if n_clusters >= 2:
        mask = labels != -1
        if int(mask.sum()) > 2 and len(set(labels[mask].tolist())) >= 2:
            sil = float(silhouette_score(X_scaled[mask], labels[mask]))
            summary["silhouette_score"] = round(sil, 6)
            try:
                sil_samples = silhouette_samples(X_scaled[mask], labels[mask])
                fig0, ax0 = plt.subplots(figsize=(7, 4))
                sns.histplot(sil_samples, kde=True, ax=ax0, color="steelblue")
                ax0.set_xlabel("Silhouette value")
                ax0.set_title("Silhouette Distribution")
                figures.append(
                    _figure_item(
                        fig0,
                        "post_silhouette_distribution",
                        "silhouette_distribution",
                        "Post-Training Silhouette Distribution",
                        "Distribution of silhouette values across non-noise points.",
                    )
                )
            except Exception:
                pass

    size_counts = label_series.value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar([f"Cluster {idx}" for idx in size_counts.index], size_counts.values, color=sns.color_palette("muted", len(size_counts)))
    ax1.set_ylabel("Count")
    ax1.set_title("Cluster Size Distribution")
    plt.xticks(rotation=30, ha="right")
    figures.append(
        _figure_item(
            fig1,
            "post_cluster_size_bar",
            "cluster_size_bar",
            "Post-Training Cluster Sizes",
            "Counts of samples per discovered cluster label.",
        )
    )

    if hasattr(model, "cluster_centers_"):
        try:
            centers = np.asarray(model.cluster_centers_)
            if centers.ndim == 2 and centers.shape[1] == numeric_df.shape[1]:
                fig2, ax2 = plt.subplots(figsize=(max(7, numeric_df.shape[1] * 0.45), max(4, centers.shape[0] * 0.4)))
                sns.heatmap(
                    centers,
                    cmap="coolwarm",
                    center=0,
                    xticklabels=numeric_df.columns.tolist(),
                    yticklabels=[f"Cluster {i}" for i in range(centers.shape[0])],
                    ax=ax2,
                )
                ax2.set_title("Cluster Centroid Heatmap")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                figures.append(
                    _figure_item(
                        fig2,
                        "post_centroid_heatmap",
                        "centroid_heatmap",
                        "Post-Training Centroid Heatmap",
                        "Feature profile of each fitted cluster centroid.",
                    )
                )
        except Exception:
            pass

    if hasattr(model, "children_") and len(X_scaled) <= 600:
        try:
            z = linkage(X_scaled, method="ward")
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            dendrogram(z, truncate_mode="level", p=5, ax=ax3, no_labels=True)
            ax3.set_title("Hierarchical Dendrogram")
            ax3.set_xlabel("Samples / merged groups")
            ax3.set_ylabel("Distance")
            plt.tight_layout()
            figures.append(
                _figure_item(
                    fig3,
                    "post_dendrogram",
                    "dendrogram",
                    "Post-Training Dendrogram",
                    "Hierarchical linkage structure for fitted clusters.",
                )
            )
        except Exception:
            pass

    return summary, figures


def generate_post_training_evaluation(
    model: Any,
    problem_type: str,
    X_test=None,
    y_test=None,
    X_train=None,
    y_train=None,
    X_full=None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """
    Build post-training diagnostics that require fitted model outputs.

    Returns
    -------
    evaluation_results: dict
        Scalar diagnostics and metadata.
    evaluation_figures: list[dict]
        Figure payloads with the same shape as EDA figure items.
    """
    if model is None:
        return {"note": "No fitted model available for post-training evaluation."}, []

    try:
        if problem_type == "Regression":
            return _regression_post_training(model, X_test, y_test, X_train, y_train)
        if problem_type == "Classification":
            return _classification_post_training(model, X_test, y_test, X_train, y_train)
        if problem_type == "Clustering":
            return _clustering_post_training(model, X_full)
    except Exception as exc:
        return {"error": str(exc)}, []

    return {"note": f"Unsupported problem_type: {problem_type}"}, []
