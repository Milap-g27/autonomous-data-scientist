# Autonomous Data Scientist Agent 🤖

An end-to-end AI agent that takes a raw CSV file and autonomously cleans it, explores it, engineers features, trains and evaluates multiple ML models, and explains the results in plain English — all without writing a single line of code.

Built with **LangGraph** (agent orchestration), **Streamlit** (UI), **ChatGroq / Llama 3.3 70B** (LLM reasoning), and a full **scikit-learn / XGBoost / LightGBM / CatBoost** model suite. Supports **Regression**, **Classification**, and **Clustering** out of the box.

---


## What It Does

You upload a CSV. The agent does everything else:

1. **Understands** the data — an LLM reads column names, types, and a sample to infer whether the problem is regression, classification, or clustering.
2. **Cleans** the data — fills missing values (median for numeric, mode for categorical), drops duplicates, encodes categoricals with LabelEncoder.
3. **Explores** the data — computes summary statistics, counts missing values, plots distributions, correlation heatmaps, and boxplots for outlier detection.
4. **Engineers features** — applies one-hot encoding, polynomial features, and interaction terms where appropriate.
5. **Trains models** — runs a full suite of algorithms with `RandomizedSearchCV` hyperparameter tuning:
   - **Regression (14 models):** Linear, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP, Voting Ensemble
   - **Classification (13 models):** Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP, Voting Ensemble
   - **Clustering (6 models):** KMeans (auto-k via silhouette), Agglomerative, DBSCAN, MeanShift, Birch, Gaussian Mixture
6. **Evaluates** models — picks the best by R² (regression), F1 (classification), or Silhouette Score (clustering).
7. **Explains** — the LLM writes a plain-English report covering model performance, key metrics, and data insights.

---

## Agent Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LangGraph State Machine                             │
│                                                                             │
│  Understand Data  →  Clean Data  →  EDA  →  Feature Engineering            │
│                                                                             │
│  → Train Models  →  Evaluate Models  →  Explain Results  →  [ END ]        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each step is a **LangGraph node** that reads from and writes to a shared `AgentState` TypedDict. The graph is compiled once and invoked per run — results are persisted in Streamlit `session_state` so the UI never re-runs the pipeline on rerender.

**Supervised path (Regression / Classification):**
- LLM detects problem type from data context
- Train/test split (configurable, default 80/20)
- Models trained on train set, evaluated on held-out test set
- Best model selected and exposed for live prediction

**Clustering path:**
- Auto-triggered when no target column is selected
- No train/test split — all data used for fitting
- Evaluated with Silhouette Score, Calinski-Harabasz, and Davies-Bouldin indices
- Best cluster assignment previewed in the UI

---

## UI Overview

The results dashboard is split into **6 tabs**:

| Tab | What You'll Find |
|-----|-----------------|
| **📊 EDA & Insights** | Summary statistics table (rows = columns, cols = stats), missing values & unique counts table with data types |
| **📈 Plots & Visualizations** | Target distribution, correlation heatmap, outlier boxplots, feature importance chart, low-cardinality distribution plots |
| **🏆 Model Performance** | Plotly bar chart comparing all trained models, full metrics table, best model highlighted |
| **🔮 Predictions** | Centered vertical form pre-filled with median values — submit to get a live prediction from the best model with a styled result card |
| **📝 AI Explanation** | LLM-generated narrative covering model choice, metric interpretation, and data insights |
| **⚙️ Processing Log** | Step-by-step data cleaning and feature engineering reports, plus the LangGraph agent pipeline diagram |

A **floating AI chatbot** (💬 button) is available on every page — it is scoped to the current session and can answer questions about the dataset, models, metrics, and explanations.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autonomous-data-scientist.git
cd autonomous-data-scientist
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Sign up for a free key at [console.groq.com](https://console.groq.com), then create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

---

## Usage

```bash
streamlit run main.py
```

Then in the browser:

1. **Upload** a CSV dataset (Titanic, Iris, Housing prices, or any tabular dataset).
2. **Select a target column** to predict — or choose **"None (Clustering)"** for unsupervised analysis.
3. **Configure** problem type (or leave on Auto-detect), test split size, random seed, and feature scaling in Advanced Settings.
4. **Click 🚀 Run AI Data Scientist** and wait ~30–60 seconds.
5. **Explore** all six result tabs.
6. **Predict** — go to the Predictions tab, adjust feature values, and click 🔮 Predict to get an instant inference from the best model.
7. **Ask questions** — use the 💬 chatbot to query anything about the analysis.

---

## Project Structure

```
autonomous-data-scientist/
│
├── main.py                     # App entry point — layout, page config, section orchestration
├── config.py                   # Pydantic-settings config (loads GROQ_API_KEY from .env)
├── requirements.txt            # All Python dependencies
├── .env                        # Your API keys (not committed to git)
│
├── core/
│   ├── agent_graph.py          # LangGraph graph definition — nodes wired into a linear pipeline
│   └── state.py                # AgentState TypedDict — shared state passed between all nodes
│
├── llm/
│   ├── understand_data.py      # LLM node: infer problem type from column names + sample data
│   └── explain_results.py      # LLM node: generate plain-English explanation of results
│
├── pipeline/
│   ├── data_cleaning.py        # Missing value imputation, deduplication, label encoding
│   ├── eda.py                  # Summary stats, distribution plots, heatmaps, boxplots
│   ├── feature_engineering.py  # OHE, polynomial features, interaction terms
│   ├── model_training.py       # 14 regression / 13 classification / 6 clustering models
│   └── model_evaluation.py     # Metric computation and best-model selection
│
└── ui/
    ├── components.py           # All dashboard rendering functions (tabs, charts, forms)
    ├── chatbot.py              # Floating chatbot — session-scoped LLM assistant
    └── styles.py               # Global CSS injection and scroll helpers
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM | Llama 3.3 70B via ChatGroq |
| ML Models | scikit-learn, XGBoost, LightGBM, CatBoost |
| Visualizations | Matplotlib, Seaborn, Plotly |
| Config | pydantic-settings + python-dotenv |

## Models

### Regression (14)
Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Classification (13)
Logistic Regression (tuned), SVM, KNN, Naive Bayes, Decision Tree, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Clustering (6)
KMeans (auto-k), Agglomerative, DBSCAN, MeanShift, Birch, Gaussian Mixture

## Requirements

See `requirements.txt` for the full list. Key libraries: `langgraph`, `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `langchain-groq`.
