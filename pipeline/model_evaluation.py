from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from typing import Dict, Any

def evaluate_models(models: Dict[str, Any], X_test, y_test, problem_type: str) -> tuple[Dict[str, Any], str]:
    """
    Evaluates trained models on test data.
    Returns a dictionary of metrics and the name of the best model.
    """
    metrics = {}
    best_score = -float('inf')
    best_model_name = ""
    
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
