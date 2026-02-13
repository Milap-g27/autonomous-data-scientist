import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

def perform_eda(df: pd.DataFrame, target: str) -> tuple[dict, list]:
    """
    Performs EDA: Generates stats and creates visualizations (histograms, boxplots, scatter, heatmap).
    Returns a dict of summary stats and a list of {"figure": fig, "description": str}.
    """
    results = {}
    figures = []
    
    # Filter out irrelevant columns (IDs, Names, high cardinality)
    cols_to_use = []
    total_rows = len(df)
    
    for col in df.columns:
        if col == target:
            cols_to_use.append(col)
            continue
            
        # 1. Check for IDs / Unique Identifiers
        # If values are unique (or very close to unique > 95%) and it's not a float/numeric feature we want
        is_unique = df[col].nunique() >= total_rows
        is_almost_unique = df[col].nunique() > 0.95 * total_rows
        
        # 2. Check Semantic Name patterns
        col_lower = col.lower()
        is_id_name = any(x in col_lower for x in ['id', 'uuid', 'guid', 'pk', 'index'])
        is_pii_name = any(x in col_lower for x in ['name', 'address', 'phone', 'email'])
        
        if df[col].dtype == 'object':
            # Object Filtering
            if is_unique or (is_almost_unique and is_id_name):
                continue # Skip IDs
                
            if is_pii_name and df[col].nunique() > 20: 
                continue # Skip potential PII with many values
                
            if df[col].nunique() > 50:
                continue # Skip high cardinality categories
                
            cols_to_use.append(col)
            
        else:
            # Numeric Filtering
            if is_id_name and is_unique:
                continue # Skip numeric IDs (e.g. PassengerId)
            
            # Skip if it looks like an index (monotonic increasing 0,1,2,3...)
            # Heuristic: perfectly correlated with index? 
            # Simple check: if it equals the index (or index+1)
            # Avoiding complex checks for now, just sticking to ID name + uniqueness
            
            cols_to_use.append(col)
            
    df_vis = df[cols_to_use]
    
    # 1. basic stats (Computed ONLY on relevant columns)
    results['description'] = df_vis.describe().to_dict()
    results['columns'] = df_vis.columns.tolist()
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 2. Target Distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4)) # Reduced size from (8, 5)
    if pd.api.types.is_numeric_dtype(df[target]):
        sns.histplot(df[target], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of Target: {target}")
        desc1 = f"Histogram showing the distribution of the target variable '{target}'."
    else:
        sns.countplot(y=df[target], ax=ax1)
        ax1.set_title(f"Count Plot of Target: {target}")
        desc1 = f"Count plot showing the frequency of each class in the target variable '{target}'."
    figures.append({"figure": fig1, "description": desc1})
    
    # 3. Correlation Heatmap (numerical only)
    numeric_df = df_vis.select_dtypes(include=['number'])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        fig2, ax2 = plt.subplots(figsize=(7, 5)) # Reduced from (10, 8)
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
        ax2.set_title("Correlation Heatmap")
        figures.append({"figure": fig2, "description": "Heatmap displaying the correlation coefficients between numerical features."})
        
        # Identify top correlated features with target
        if target in corr.columns:
            target_corr = corr[target].abs().sort_values(ascending=False)
            top_features = target_corr[1:6].index.tolist() # Top 5 excluding target itself
            results['top_correlated_features'] = top_features
        else:
            top_features = numeric_df.columns[:5].tolist()
    else:
        top_features = []
        
    # 4. Scatter Plots for Top Correlated Features vs Target
    # Limit to top 3 to avoid overcrowding
    for feature in top_features[:3]:
        fig, ax = plt.subplots(figsize=(6, 4)) # Reduced from (8, 5)
        if pd.api.types.is_numeric_dtype(df[target]):
            sns.scatterplot(data=df_vis, x=feature, y=target, ax=ax)
            ax.set_title(f"{feature} vs {target}")
            desc = f"Scatter plot showing the relationship between '{feature}' and the target '{target}'."
        else:
            sns.boxplot(data=df_vis, x=target, y=feature, ax=ax)
            ax.set_title(f"{feature} by {target}")
            desc = f"Boxplot showing the distribution of '{feature}' for each class of '{target}'."
        figures.append({"figure": fig, "description": desc})
        
    # 5. Boxplots for Numerical Features to check outliers
    if top_features:
        fig, ax = plt.subplots(figsize=(7, 4)) # Reduced from (10, 6)
        # Melding for boxplot requires scaling if ranges differ wildly, 
        # so let's just do individual boxplots for top 3 features
        data_to_plot = df_vis[top_features[:3]]
        # Standardize for visualization if needed, but raw is often better for outlier detection
        sns.boxplot(data=data_to_plot, orient="h", ax=ax)
        ax.set_title("Boxplots of Top Features")
        figures.append({"figure": fig, "description": f"Boxplots of top features {top_features[:3]} to visualize distributions and potential outliers."})

    return results, figures
