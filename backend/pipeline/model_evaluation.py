from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
from typing import Dict, Any


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


def evaluate_models(models: Dict[str, Any], X_test, y_test, problem_type: str) -> tuple[Dict[str, Any], str]:
    """
    Evaluates trained models on test data.
    For clustering, evaluates on full X (no train/test split needed).
    Returns a dictionary of metrics and the name of the best model.
    """
    metrics = {}
    best_score = -float('inf')
    best_model_name = ""
    
    if problem_type == "Clustering":
        # For clustering, X_test is actually the full X, y_test is None
        X_full = X_test
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full.select_dtypes(include=['number']))
        
        for name, model in models.items():
            try:
                # Get cluster labels
                if hasattr(model, 'labels_'):
                    labels = model.labels_
                elif hasattr(model, 'predict'):
                    labels = model.predict(X_scaled)
                else:
                    continue
                
                n_clusters = len(set(labels) - {-1})  # exclude noise label -1
                n_noise = list(labels).count(-1)
                
                if n_clusters < 2:
                    metrics[name] = {
                        "Clusters": n_clusters,
                        "Noise Points": n_noise,
                        "Silhouette Score": "N/A (< 2 clusters)",
                        "Calinski-Harabasz": "N/A",
                        "Davies-Bouldin": "N/A"
                    }
                    continue
                
                # Filter out noise points for scoring
                mask = np.array(labels) != -1
                if mask.sum() < 2:
                    continue
                
                sil = silhouette_score(X_scaled[mask], np.array(labels)[mask])
                ch = calinski_harabasz_score(X_scaled[mask], np.array(labels)[mask])
                db = davies_bouldin_score(X_scaled[mask], np.array(labels)[mask])
                
                metrics[name] = {
                    "Clusters": n_clusters,
                    "Noise Points": n_noise,
                    "Silhouette Score": round(sil, 4),
                    "Calinski-Harabasz": round(ch, 2),
                    "Davies-Bouldin": round(db, 4)
                }
                
                # Best = highest silhouette score
                if sil > best_score:
                    best_score = sil
                    best_model_name = name
            except Exception:
                pass
        
        if not best_model_name and models:
            best_model_name = list(models.keys())[0]
        
        return metrics, best_model_name
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if problem_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics[name] = {
                "MSE": mse,
                "R2": r2,
                "MAE": mae
            }
            
            # Select best based on R2
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                
        else: # Classification
            accuracy = accuracy_score(y_test, y_pred)
            # Use weighted average for multiclass, binary for binary
            # We'll use 'weighted' to be safe for both
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            
            # Select best based on F1 Score
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                
    return metrics, best_model_name
