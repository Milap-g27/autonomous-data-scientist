import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import settings

def understand_data_node(state: dict) -> dict:
    """
    Analyzes the dataframe to determine the problem type (Regression, Classification, or Clustering).
    """
    df = state['df']
    target = state.get('target')
    
    # If no target is specified, it's a clustering problem
    if target is None:
        return {"problem_type": "Clustering"}
    
    # Initialize LLM here to ensure API key is captured from environment
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )
    
    # Create a summary of the data for the LLM
    # If the df is huge, we only take head and info-like details
    data_head = df.head().to_markdown()
    target_info = df[target].describe().to_markdown() if target in df.columns else "Target not found in DF"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Data Scientist. Your task is to identify if the problem is 'Regression' or 'Classification' based on the target variable."),
        ("user", "Here is the dataset head:\n{data_head}\n\nTarget Variable Stats:\n{target_info}\n\nTarget Column Name: {target}\n\nDetermine the problem type. Return ONLY one word: 'Regression' or 'Classification'.")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "data_head": data_head,
            "target_info": target_info,
            "target": target
        })
        problem_type = response.content.strip().replace("'", "").replace('"', "")
        
        # Fallback validation
        if "regression" in problem_type.lower():
            problem_type = "Regression"
        elif "classification" in problem_type.lower():
            problem_type = "Classification"
        else:
            # Default fallback if LLM hallucinates
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                problem_type = "Regression"
            else:
                problem_type = "Classification"
                
    except Exception as e:
        print(f"Error in understand_data: {e}")
        # Heuristic fallback
        if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
            problem_type = "Regression"
        else:
            problem_type = "Classification"
    
    return {"problem_type": problem_type}
