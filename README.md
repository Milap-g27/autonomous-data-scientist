# Autonomous Data Scientist Agent 🤖

A production-grade AI agent that autonomously cleans data, performs EDA, engineers features, trains models, and explains results using **LangGraph**, **Streamlit**, and **ChatGroq**.

## Features

- **Auto-Cleaning**: Handles missing values, duplicates, and categorical encoding (LabelEncoder).
- **Advanced EDA**:
  - Summary statistics with missing values count, unique values count, and data types.
  - Target distribution plot, correlation heatmap, and outlier detection boxplots — each with headings and descriptions.
  - Feature importance chart (tree-based & coefficient-based models).
  - Distribution plots for low-cardinality features, displayed individually.
- **Smart Modeling**: Trains Linear/Logistic Regression, Random Forests, and Ensemble Voting models with hyperparameter tuning.
- **Interactive Prediction**: Fill in a form with original dataset column names and types (int/float/categorical) — input is processed through the same cleaning & feature engineering pipeline, and the best model predicts the output with reverse-mapped labels.
- **LLM Reasoning**: Uses Llama 3.3 via Groq to infer problem type and explain results in plain English. Column names are wrapped in backticks for clarity.
- **Deterministic Workflow**: Built on LangGraph state machine with session state persistence.

## Architecture

The agent follows a linear graph:

```
Understand Data → Clean Data → EDA → Feature Engineering → Train Models → Evaluate Models → Explain Results
```

## Tabs

| Tab | Contents |
|-----|----------|
| **📊 EDA & Insights** | Summary stats, missing/unique counts, target distribution, correlation heatmap, boxplots, feature importance, categorical distributions |
| **🏆 Model Performance** | Metrics table, best model highlight, interactive prediction form |
| **📝 AI Explanation** | LLM-generated plain-English explanation of results |
| **⚙️ Processing Log** | Data cleaning report, feature engineering report |

## Setup

1. **Clone the repository**
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac
   ```
3. **Install dependencies**:
   ```bash
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
3. **Select the target column** you want to predict.
4. **Click "🚀 Run AI Data Scientist"** and wait for the analysis to complete.
5. **Explore results** across the four tabs.
6. **Make predictions** — fill in feature values in the Model Performance tab and click "🔮 Predict".

## Project Structure

```
├── main.py                          # Streamlit frontend
├── config.py                        # App settings (API keys via pydantic-settings)
├── .env                             # Environment variables (GROQ_API_KEY)
├── requirements.txt                 # Python dependencies
├── core/
│   ├── agent_graph.py               # LangGraph workflow definition
│   └── state.py                     # Agent state (TypedDict)
├── llm/
│   ├── understand_data.py           # LLM node: infer problem type
│   └── explain_results.py           # LLM node: explain results
├── pipeline/
│   ├── data_cleaning.py             # Cleaning: missing values, duplicates, encoding
│   ├── eda.py                       # EDA: stats, plots, correlations
│   ├── feature_engineering.py       # Feature engineering: OHE, interactions, polynomials
│   ├── model_training.py            # Model training: LR, RF, Ensemble with tuning
│   └── model_evaluation.py          # Model evaluation: metrics, best model selection
```

## Requirements

See `requirements.txt` for the full list. Key libraries: `langgraph`, `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `langchain-groq`.
