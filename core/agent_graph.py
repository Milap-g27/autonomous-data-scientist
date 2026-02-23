import asyncio
from langgraph.graph import StateGraph, END , START
from core.state import AgentState

# Import nodes
from llm.understand_data import understand_data_node
from llm.explain_results import explain_results_node
from pipeline.data_cleaning import clean_data
from pipeline.eda import perform_eda
from pipeline.feature_engineering import perform_feature_engineering
from pipeline.model_training import train_models
from pipeline.model_evaluation import evaluate_models
from sklearn.model_selection import train_test_split

# Wrapper functions to match StateGraph node signature (state -> update)
# All wrappers are async; CPU-bound pipeline calls use asyncio.to_thread

async def clean_data_node(state: AgentState) -> dict:
    df = state['df']
    cleaned_df, report = await asyncio.to_thread(clean_data, df)
    return {"df": cleaned_df, "cleaning_report": report}

async def eda_node(state: AgentState) -> dict:
    df = state['df']
    target = state.get('target')
    metrics, figures = await asyncio.to_thread(perform_eda, df, target)
    return {"eda_results": metrics, "eda_figures": figures}

async def feature_engineering_node(state: AgentState) -> dict:
    df = state['df']
    target = state.get('target')
    problem_type = state['problem_type']
    X, y, report = await asyncio.to_thread(perform_feature_engineering, df, target, problem_type)
    return {"X": X, "y": y, "feature_report": report}

async def train_models_node(state: AgentState) -> dict:
    X = state['X']
    y = state.get('y')
    problem_type = state['problem_type']
    
    if problem_type == "Clustering":
        # No train/test split for clustering — use all data
        models = await asyncio.to_thread(train_models, X, None, problem_type)
        return {"models": models}
    
    # Split data here for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = await asyncio.to_thread(train_models, X_train, y_train, problem_type)
    
    return {"models": models}

async def evaluate_models_node(state: AgentState) -> dict:
    X = state['X']
    y = state.get('y')
    models = state['models']
    problem_type = state['problem_type']
    
    if problem_type == "Clustering":
        # Evaluate on full data, no y needed
        metrics, best_model_name = await asyncio.to_thread(evaluate_models, models, X, None, problem_type)
        return {"metrics": metrics, "model_name": best_model_name}
    
    # Re-split to get the same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    metrics, best_model_name = await asyncio.to_thread(evaluate_models, models, X_test, y_test, problem_type)
    
    return {"metrics": metrics, "model_name": best_model_name}

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("understand_data", understand_data_node)# type: ignore
    workflow.add_node("clean_data", clean_data_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("train_models", train_models_node)
    workflow.add_node("evaluate_models", evaluate_models_node)
    workflow.add_node("explain_results", explain_results_node)# type: ignore
    
    # Add edges (Linear flow)
    workflow.set_entry_point("understand_data")
    workflow.add_edge("understand_data", "clean_data")
    workflow.add_edge("clean_data", "eda")
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "train_models")
    workflow.add_edge("train_models", "evaluate_models")
    workflow.add_edge("evaluate_models", "explain_results")
    workflow.add_edge("explain_results", END)
    
    return workflow.compile()
