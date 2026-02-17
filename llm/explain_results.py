from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from config import settings

def explain_results_node(state: dict) -> dict:
    """
    Generates a natural language explanation of the results using the LLM.
    """
    # Initialize LLM here to ensure API key is captured from environment
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", 
        temperature=0.7
    )

    metrics = state['metrics']
    best_model = state['model_name']
    problem_type = state['problem_type']
    target = state.get('target')
    
    # Extract key EDA insights if available
    eda_summary = ""
    if 'eda_results' in state and 'top_correlated_features' in state['eda_results']:
        eda_summary = f"Top correlated features: {state['eda_results']['top_correlated_features']}"
    
    if problem_type == "Clustering":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Scientist. Explain the results of unsupervised clustering to a non-technical stakeholder. IMPORTANT: Always wrap column names, feature names, and variable names in backticks (` `) when mentioning them."),
            ("user", """
            Problem Type: Clustering (Unsupervised)
            
            Best Clustering Algorithm: {best_model}
            
            Clustering Metrics:
            {metrics}
            
            Key Data Insights:
            {eda_summary}
            
            Provide a clear, concise explanation of:
            1. Which clustering algorithm performed best and why (based on silhouette score and other metrics).
            2. What the metrics mean in plain English (silhouette score, Calinski-Harabasz, Davies-Bouldin).
            3. How many clusters were found and what that might mean for the data.
            4. Any recommendations for further analysis.
            
            IMPORTANT: When referring to any column name or feature name, always wrap it in backticks like `column_name`.
            
            Keep it professional but accessible.
            """)
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Scientist. Explain the results of the model training to a non-technical stakeholder. IMPORTANT: Always wrap column names, feature names, and variable names in backticks (` `) when mentioning them."),
            ("user", """
            Problem Type: {problem_type}
            Target Variable: `{target}`
            
            Best Performing Model: {best_model}
            
            Model Metrics:
            {metrics}
            
            Key Data Insights:
            {eda_summary}
            
            Provide a clear, concise explanation of:
            1. Which model was best and why (based on metrics).
            2. What the metrics mean in plain English (e.g. valid accuracy or error).
            3. Any interesting patterns from the data insights.
            
            IMPORTANT: When referring to any column name or feature name, always wrap it in backticks like `column_name`.
            
            Keep it professional but accessible.
            """)
        ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "problem_type": problem_type,
            "target": target or "N/A",
            "best_model": best_model,
            "metrics": metrics,
            "eda_summary": eda_summary
        })
        explanation = response.content
    except Exception as e:
        explanation = f"Error generating explanation: {e}"
    
    return {"explanation": explanation}
