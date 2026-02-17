# Autonomous Data Scientist Agent 🤖

A production-grade AI agent that autonomously cleans data, performs EDA, engineers features, trains models, and explains results using **LangGraph**, **Streamlit**, and **ChatGroq**. Supports **Regression**, **Classification**, and **Clustering** workflows.

## Features

- **Auto-Cleaning**: Handles missing values (median/mode fill), duplicates, and categorical encoding (LabelEncoder). Reports each step with column names in backticks.
  
- **Advanced EDA**:
  - Summary statistics with missing values count, unique values count, and data types table.
  - Target distribution plot, correlation heatmap, and outlier detection boxplots — each with headings and descriptions.
  - Feature importance chart (tree-based `feature_importances_` & coefficient-based `coef_` models).
  - Distribution plots for low-cardinality features, displayed individually.
    
- **Comprehensive Modeling**:
  - **Regression** (14 models): Linear, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble.
  - **Classification** (13 models): Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble.
  - **Clustering** (6 models): KMeans (auto-k via silhouette), Agglomerative, DBSCAN, MeanShift, Birch, Gaussian Mixture.
  - Hyperparameter tuning via `RandomizedSearchCV` with dynamic CV folds (safe for small classes).
  - Each model wrapped in `try/except` for robustness — a single model failure never crashes the pipeline.
  
- **Interactive Prediction** (Supervised): Fill in a form with original dataset column names and types (int/float/categorical with min/max ranges) — input is processed through the same cleaning & feature engineering pipeline, and the best model predicts the output with reverse-mapped labels.

- **Cluster Visualization** (Clustering): Preview cluster assignments, cluster count, and cluster size distribution bar chart.
  
- **LLM Reasoning**: Uses Llama 3.3 70B via Groq to infer problem type (or auto-detect Clustering when no target) and explain results in plain English. Column names are wrapped in backticks for clarity.
  
- **Deterministic Workflow**: Built on LangGraph state machine with Streamlit `session_state` persistence and `@st.fragment` for isolated form reruns.

## Architecture

The agent follows a linear graph:

```
Understand Data → Clean Data → EDA → Feature Engineering → Train Models → Evaluate Models → Explain Results
```

- **Supervised** (Regression/Classification): LLM detects problem type → full pipeline with train/test split → metrics (R², MSE, MAE for regression; Accuracy, Precision, Recall, F1 for classification).
- **Clustering**: Auto-detected when target is "None" → no train/test split → models fitted on full data → evaluated with Silhouette Score, Calinski-Harabasz, and Davies-Bouldin indices.

## Tabs

| Tab | Contents |
|-----|----------|
| **📊 EDA & Insights** | Summary stats, missing/unique counts table, target distribution, correlation heatmap, boxplots, feature importance, categorical distributions |
| **🏆 Model Performance** | Metrics table for all models, best model highlight, interactive prediction form (supervised) or cluster assignment preview + size chart (clustering) |
| **📝 AI Explanation** | LLM-generated plain-English explanation with backtick-wrapped column names |
| **⚙️ Processing Log** | Data cleaning report and feature engineering report displayed as bullet-point lists with backtick column names, plus agent pipeline graph |

## Setup

1. **Clone the repository**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac
   ```
3. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
4. **Get Groq API Key**:
   - Sign up at [console.groq.com](https://console.groq.com) to get a free API key.
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run main.py
   ```
   Or explicitly with the venv Python:
   ```bash
   .venv\Scripts\python.exe -m streamlit run main.py
   ```
2. **Upload a CSV file** (e.g., Titanic, Iris, Housing prices, Personality dataset).
3. **Select the target column** you want to predict, or choose **"None (Clustering)"** for unsupervised clustering.
4. **Click "🚀 Run AI Data Scientist"** and wait for the analysis to complete.
5. **Explore results** across the four tabs.
6. **Make predictions** (supervised) — fill in feature values in the Model Performance tab and click "🔮 Predict".

## Project Structure

```
├── main.py                          # Streamlit frontend (tabs, prediction form, cluster viz)
├── config.py                        # App settings (API keys via pydantic-settings)
├── .env                             # Environment variables (GROQ_API_KEY)
├── requirements.txt                 # Python dependencies
├── run.bat                          # One-click launcher (Windows)
├── core/
│   ├── agent_graph.py               # LangGraph workflow definition (clustering-aware)
│   └── state.py                     # Agent state (TypedDict, optional target/y)
├── llm/
│   ├── understand_data.py           # LLM node: infer problem type (or auto-Clustering)
│   └── explain_results.py           # LLM node: explain results (supervised & clustering prompts)
├── pipeline/
│   ├── data_cleaning.py             # Cleaning: missing values, duplicates, encoding
│   ├── eda.py                       # EDA: stats, plots, correlations (target-optional)
│   ├── feature_engineering.py       # Feature engineering: OHE, interactions, polynomials
│   ├── model_training.py            # 14 regression / 13 classification / 6 clustering models
│   └── model_evaluation.py          # Evaluation: R²/F1 (supervised), silhouette (clustering)
```

## Models

### Regression (14)
Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Classification (13)
Logistic Regression (tuned), SVM, KNN, Naive Bayes, Decision Tree, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Clustering (6)
KMeans (auto-k), Agglomerative, DBSCAN, MeanShift, Birch, Gaussian Mixture

## Requirements

See `requirements.txt` for the full list. Key libraries: `langgraph`, `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `langchain-groq`.
