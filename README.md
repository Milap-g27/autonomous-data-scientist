# Autonomous Data Scientist Agent 🤖

A production-grade AI agent that autonomously cleans data, performs EDA, engineers features, trains models, and explains results using **LangGraph**, **Streamlit**, and **ChatGroq**.

## Features
- **Auto-Cleaning**: Handles missing values, duplicates, and categorical encoding.
- **Advanced EDA**: Generates histograms, heatmaps, and correlation analysis.
- **Smart Modeling**: Trains Linear/Logistic Regression, Random Forests, and Ensemble Voting models.
- **LLM Reasoning**: Uses Llama 3 via Groq to infer problem type and explain results in plain English.
- **Deterministic Workflow**: Built on LangGraph state machine.

## Architecture
The agent follows a linear graph:
`Understand Data` -> `Clean Data` -> `EDA` -> `Feature Engineering` -> `Train Models` -> `Evaluate Models` -> `Explain Results`

## Setup

1. **Clone the repository**
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Get Groq API Key**:
   - Sign up for free at [console.groq.com](https://console.groq.com) to get an API key.

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
2. **Upload a CSV File** (e.g., Titanic, Iris, Housing prices).
3. **Enter your Groq API Key** in the sidebar.
4. **Select the Target Column** you want to predict.
5. **Click "Run AI Data Scientist"**.

## Project Structure
- `app.py`: Streamlit frontend.
- `agent_graph.py`: Main LangGraph workflow definition.
- `state.py`: Agent state definition.
- `agents/`: LLM-powered reasoning nodes (`understand_data`, `explain_results`).
- `tools/`: Pure Python functions for computation (`cleaning`, `eda`, `training`, `evaluation`).

## Requirements
See `requirements.txt` for full list. Key libraries: `langgraph`, `streamlit`, `pandas`, `scikit-learn`, `langchain-groq`.
