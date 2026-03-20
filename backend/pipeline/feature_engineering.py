import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import Optional

def perform_feature_engineering(df: pd.DataFrame, target: Optional[str] = None, problem_type: str = "Classification") -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Separates target, creates additional features (interactions/polynomials) if applicable.
    When target is None (clustering), all columns become features and y is None.
    Returns X (features), y (target or None), and a report.
    """
    report = []
    
    # 1. Separate Target (or use all columns for clustering)
    if target is not None:
        y = df[target]
        X = df.drop(columns=[target])
        report.append(f"Separated target variable `{target}` from features.")
    else:
        y = None
        X = df.copy()
        report.append("No target variable (clustering mode). Using all columns as features.")
    
    # 2. Identify numerical and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2.5 One-Hot Encoding for Categorical Columns
    if categorical_cols:
        # report.append(f"Found categorical columns: {categorical_cols}")
        # One-hot encode with drop_first=True to avoid dummy variable trap
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        report.append(f"Applied One-Hot Encoding to: {', '.join(f'`{c}`' for c in categorical_cols)}. Total features now: {X.shape[1]}")
    
    # Re-identify numeric columns after encoding (all should be numeric now ideally, but good to be safe)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # 3. Create Interaction Features for Top Correlated Columns (if enough columns)
    if len(numeric_cols) >= 2 and y is not None:
        # Calculate correlations with target to find best features
        # Note: y might need to be numeric for correlation, if not we skip
        if pd.api.types.is_numeric_dtype(y):
            correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            top_2 = correlations.head(2).index.tolist()
            
            if len(top_2) == 2:
                col1, col2 = top_2
                new_col_name = f"{col1}_x_{col2}"
                X[new_col_name] = X[col1] * X[col2]
                report.append(f"Created interaction feature: `{new_col_name}`")
    
    # 4. (Optional) Polynomial Features
    #  For this agent, we'll keep it simple to avoid overfitting or memory issues with large datasets
    #  But we could add degree=2 for top feature
    if len(numeric_cols) > 0 and y is not None and pd.api.types.is_numeric_dtype(y):
        try:
            # Re-calc correlations as X might have changed
            correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            if not correlations.empty:
                top_feature = correlations.idxmax()
                X[f"{top_feature}_squared"] = X[top_feature] ** 2
                report.append(f"Created polynomial feature: `{top_feature}_squared`")
        except Exception as e:
            report.append(f"Skipped polynomial feature due to error: {str(e)}")

    return X, y, "\n".join(report)
