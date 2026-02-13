import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

# Import config and core logic
from config import settings
from core.agent_graph import build_graph

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page config
st.set_page_config(layout="wide", page_title="Autonomous Data Scientist Agent")

st.title("🤖 Autonomous Data Scientist Agent")
st.markdown("""
This agent autonomously:
1. **Understands** your data
2. **Cleans** & Preprocesses it
3. **performs EDA** with visualizations
4. **Engineers Features**
5. **Trains & Evaluates** multiple models
6. **Explains** the results in plain English
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        
        # Column Selection
        columns = df.columns.tolist()
        target = st.selectbox("Select Target Column (the value you want to predict)", columns)
        
        if st.button("🚀 Run AI Data Scientist"):
            api_key = settings.GROQ_API_KEY
            if not api_key:
                st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
            else:
                with st.spinner("Initializing Agent..."):
                    graph = build_graph()
                
                # Progress Container
                status_container = st.empty()
                status_container.info("Agent is working... (Check your terminal for detailed logs)")
                
                # Initial State
                initial_state = {"df": df, "target": target}
                
                try:
                    # Run the graph
                    result = graph.invoke(initial_state)
                    
                    status_container.success("Analysis Complete!")
                    
                    # Create Tabs for Results
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Insights", "🏆 Model Performance", "📝 AI Explanation", "⚙️ Processing Log"])
                    
                    with tab1:
                        st.header("Exploratory Data Analysis")
                        
                        # Display EDA Stats
                        if 'eda_results' in result:
                            st.write("**Summary Statistics:**")
                            st.json(result['eda_results']['description'])
                        
                        # Display Figures
                        if 'eda_figures' in result:
                            for idx, item in enumerate(result['eda_figures']):
                                fig = item
                                desc = ""
                                if isinstance(item, dict):
                                    fig = item.get('figure')
                                    desc = item.get('description', '')
                                
                                c1, c2, c3 = st.columns([1, 2, 1])
                                with c2:
                                    st.pyplot(fig)
                                    if desc:
                                        st.caption(desc)
                                st.markdown("---")
                                
                    with tab2:
                        st.header("Model Evaluation")
                        if 'metrics' in result:
                            # Convert metrics dict to dataframe for nice display
                            metrics_df = pd.DataFrame(result['metrics']).T
                            st.table(metrics_df)
                            
                        st.success(f"**Best Performing Model:** {result.get('model_name', 'N/A')}")
                        
                    with tab3:
                        st.header("AI Explanation")
                        st.markdown(result.get('explanation', "No explanation generated."))
                        
                    with tab4:
                        st.header("Processing Logs")
                        st.subheader("Data Cleaning Report")
                        st.text(result.get('cleaning_report', 'No report'))
                        
                        st.subheader("Feature Engineering Report")
                        st.text(result.get('feature_report', 'No report'))
                        
                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")
                    import traceback
                    st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"Error reading file: {e}")
