# Autonomous Data Scientist Agent 🤖

An end-to-end AI agent that takes a raw CSV file and autonomously cleans it, explores it, engineers features, trains and evaluates multiple ML models, and explains the results in plain English — all without writing a single line of code.

Built with **LangGraph** (agent orchestration), **React & Tailwind CSS** (UI), **ChatGroq / Llama 3.3 70B** (LLM reasoning), and a full **scikit-learn / XGBoost / LightGBM / CatBoost** model suite. Supports **Regression**, **Classification**, and **Clustering** out of the box.

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

Each step is a **LangGraph node** that reads from and writes to a shared `AgentState` TypedDict. The graph is compiled once and invoked per run via the FastAPI backend.

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
| **🔮 Predictions** | Form pre-filled with median values — submit to get a live prediction from the best model with a styled result card |
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

### 2. Add your Groq API key

Sign up for a free key at [console.groq.com](https://console.groq.com), then create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Start the Backend (FastAPI)

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

### 4. Start the Frontend (React + Vite)

```bash
# Open a new terminal
cd frontend
npm install
npm run dev
```

---

## Usage (Local)

1. Open `http://localhost:5173` in your browser.
2. **Upload** a CSV dataset (Titanic, Iris, Housing prices, or any tabular dataset).
3. **Select a target column** to predict — or choose **"None (Clustering)"** for unsupervised analysis.
4. **Configure** problem type (or leave on Auto-detect), test split size, random seed, and feature scaling in Advanced Settings.
5. **Click 🚀 Run AI Data Scientist** and wait ~30–60 seconds.
6. **Explore** all six result tabs.
7. **Predict** — go to the Predictions tab, adjust feature values, and click 🔮 Predict to get an instant inference from the best model.
8. **Ask questions** — use the 💬 chatbot to query anything about the analysis.

---

## Deployment

The app is deployed as **two services**:

| Service | Platform | What it runs |
|---------|----------|--------------|
| **Backend API** | [Render](https://render.com) (Docker) | FastAPI + ML pipeline (`main:app`) |
| **Frontend UI** | [Vercel](https://vercel.com) or [Netlify](https://netlify.com) | React static SPA |

### Deploy the Backend on Render

1. Push your repo to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New +** → **Blueprint** or **Web Service**.
3. Connect your GitHub repo and point the root directory to `backend/`.
4. Set environment variables in the Render dashboard:
   - `GROQ_API_KEY` — your Groq API key
   - `CORS_ORIGINS` — comma-separated allowed origins (e.g. `https://your-app.vercel.app,https://your-custom-domain.com`)
   - `FIREBASE_SERVICE_ACCOUNT_JSON` — full JSON string for Firebase Admin service account (recommended in production)
   - `FIREBASE_SERVICE_ACCOUNT_PATH` — leave empty in production unless you mount a file
   - `OPENAI_API_KEY` — optional, if you switch LLM providers
5. Deploy. Your API will be live at `https://<service-name>.onrender.com`.

### Deploy the Frontend on Vercel

1. Go to [Vercel](https://vercel.com) and sign in with GitHub.
2. Click **Add New** → **Project** and select your repo.
3. Configure the Root Directory to `frontend`.
4. Add Environment Variables:
   ```
   VITE_API_BASE = "https://<your-render-service>.onrender.com/api"
   VITE_FIREBASE_API_KEY = "..."
   VITE_FIREBASE_AUTH_DOMAIN = "<your-project>.firebaseapp.com"
   VITE_FIREBASE_PROJECT_ID = "<your-project>"
   VITE_FIREBASE_STORAGE_BUCKET = "<your-project>.appspot.com"
   VITE_FIREBASE_MESSAGING_SENDER_ID = "..."
   VITE_FIREBASE_APP_ID = "..."
   VITE_FIREBASE_MEASUREMENT_ID = "..."  # optional
   ```
5. Click **Deploy**. Your UI will be live at `https://your-app.vercel.app`.

6. In Firebase Console → Authentication → Settings → Authorized domains, add:
   - `your-app.vercel.app`
   - your custom frontend domain (if any)

### Environment Variables Summary

| Variable | Where | Value |
|----------|-------|-------|
| `GROQ_API_KEY` | Render (backend) | Your Groq API key |
| `CORS_ORIGINS` | Render (backend) | `https://your-app.vercel.app,https://your-custom-domain.com` |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Render (backend) | Full service-account JSON string |
| `VITE_API_BASE` | Vercel (frontend) | `https://<render-service>.onrender.com/api` |
| `VITE_FIREBASE_API_KEY` | Vercel (frontend) | Firebase web app API key |
| `VITE_FIREBASE_AUTH_DOMAIN` | Vercel (frontend) | `<your-project>.firebaseapp.com` |
| `VITE_FIREBASE_PROJECT_ID` | Vercel (frontend) | Firebase project id |
| `VITE_FIREBASE_STORAGE_BUCKET` | Vercel (frontend) | `<your-project>.appspot.com` |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Vercel (frontend) | Firebase sender id |
| `VITE_FIREBASE_APP_ID` | Vercel (frontend) | Firebase app id |

---

## Project Structure

```
autonomous-data-scientist/
│
├── backend/                    # FastAPI Backend
│   ├── main.py                 # FastAPI entry point — all API endpoints
│   ├── config.py               # Pydantic-settings config
│   ├── requirements.txt        # All Python dependencies
│   ├── Dockerfile              # Docker image for Render deployment
│   ├── core/                   # LangGraph graph definition & state
│   ├── llm/                    # LLM nodes for context & explanation
│   ├── pipeline/               # Data cleaning, EDA, FE, ML training & evaluation
│   └── api/                    # Route handlers, schemas, utilities
│
└── frontend/                   # React + Vite Frontend
    ├── index.html              # HTML entry point
    ├── package.json            # Node dependencies
    ├── tailwind.config.js      # Tailwind CSS styling config
    ├── vite.config.ts          # Vite build configuration
    └── src/
        ├── App.tsx             # Main routing component
        ├── pages/              # Page views (Home, Configure, Results)
        ├── components/         # Reusable UI components & Chatbot
        ├── features/           # Tab views and specific logic modules
        └── services/           # API interaction functions
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | React, Vite, Tailwind CSS, Framer Motion |
| Agent Orchestration | LangGraph |
| LLM | Llama 3.3 70B via ChatGroq |
| ML Models | scikit-learn, XGBoost, LightGBM, CatBoost |
| Backend | FastAPI |
| Config | pydantic-settings + python-dotenv |

## Models

### Regression (14)
Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Classification (13)
Logistic Regression (tuned), SVM, KNN, Naive Bayes, Decision Tree, Random Forest (tuned), Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP Neural Network, Voting Ensemble

### Clustering (6)
KMeans (auto-k), Agglomerative, DBSCAN, MeanShift, Birch, Gaussian Mixture
