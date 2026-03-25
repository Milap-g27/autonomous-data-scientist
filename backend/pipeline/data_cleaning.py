import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Cleans the dataframe by handling missing values, duplicates, and encoding categoricals.
    Returns the cleaned dataframe and a textual report of changes.
    """
    report = []

    # 0. Normalise pandas 2.x extension dtypes to numpy-compatible equivalents.
    #    pd.read_csv with pandas >= 2.0 may return StringDtype / BooleanDtype /
    #    Int64Dtype columns that numpy (and np.issubdtype) cannot interpret.
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        # Convert nullable integer / float extension types → standard numpy dtype
        if hasattr(dtype, 'numpy_dtype'):
            try:
                df[col] = df[col].astype(dtype.numpy_dtype)
            except (TypeError, ValueError):
                df[col] = df[col].astype(object)
        # Convert StringDtype → object
        elif isinstance(dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
        # Convert BooleanDtype → bool
        elif isinstance(dtype, pd.BooleanDtype):
            try:
                df[col] = df[col].astype(bool)
            except (TypeError, ValueError):
                df[col] = df[col].astype(object)

    # 1. Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        report.append(f"Dropped {dropped_rows} duplicate rows.")
    
    # 2. Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_count = df[col].isnull().sum()
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill numerical with median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                report.append(f"Filled {missing_count} missing values in `{col}` with median ({median_val}).")
            else:
                # Fill categorical with mode
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    report.append(f"Filled {missing_count} missing values in `{col}` with mode ('{mode_val}').")
                else:
                    # Fallback if mode is empty (all NaNs)
                    df[col] = df[col].fillna("Unknown")
                    report.append(f"Filled {missing_count} missing values in `{col}` with 'Unknown'.")

    # 3. Encode categorical variables
    # We use Label Encoding for simplicity to ensure all data is numeric for models
    # Ideally we'd use OneHot for low cardinality, but for a generic agent, Label is safer to avoid explosion
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # strict conversion to string to avoid mixed type issues
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        report.append(f"Encoded categorical column `{col}` using LabelEncoder.")

    if not report:
        report.append("No cleaning was necessary.")
        
    return df, "\n".join(report)
