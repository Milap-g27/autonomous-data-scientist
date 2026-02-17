import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
        target_options = ["None (Clustering)"] + columns
        target_selection = st.selectbox(
            "Select Target Column (the value you want to predict, or None for clustering)",
            target_options,
            index=1,  # Default to first actual column
        )
        target = None if target_selection == "None (Clustering)" else target_selection
        
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
                    
                    # Store results in session state so they persist across reruns
                    st.session_state['result'] = result
                    st.session_state['target'] = target
                    st.session_state['df'] = df
                    
                    status_container.success("Analysis Complete!")
                    
                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        
        # Display results from session state (persists across reruns)
        if 'result' in st.session_state:
            result = st.session_state['result']
            target = st.session_state['target']
            df_original = st.session_state['df']
            
            # Create Tabs for Results
            tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Insights", "🏆 Model Performance", "📝 AI Explanation", "⚙️ Processing Log"])
            
            with tab1:
                st.header("Exploratory Data Analysis")
                
                # Display EDA Stats
                if 'eda_results' in result:
                    st.subheader("📋 Summary Statistics")
                    st.json(result['eda_results']['description'])
                    
                    # Missing Values & Unique Counts
                    st.subheader("🔍 Missing Values & Unique Counts")
                    info_data = {}
                    for col in df_original.columns:
                        info_data[col] = {
                            'Missing Values': int(df_original[col].isnull().sum()),
                            'Missing %': round(df_original[col].isnull().sum() / len(df_original) * 100, 2),
                            'Unique Values': int(df_original[col].nunique()),
                            'Data Type': str(df_original[col].dtype)
                        }
                    info_df = pd.DataFrame(info_data).T
                    st.table(info_df)
                
                # Display Figures
                if 'eda_figures' in result:
                    for idx, item in enumerate(result['eda_figures']):
                        fig = item
                        desc = ""
                        heading = ""
                        if isinstance(item, dict):
                            fig = item.get('figure')
                            desc = item.get('description', '')
                            heading = item.get('heading', '')
                        
                        if heading:
                            st.subheader(heading)
                        c1, c2, c3 = st.columns([1, 2, 1])
                        with c2:
                            st.pyplot(fig)
                            if desc:
                                st.caption(desc)
                        st.markdown("---")
                
                # --- Feature Importance ---
                best_model_name = result.get('model_name', '')
                best_model = result.get('models', {}).get(best_model_name)
                training_X = result.get('X', pd.DataFrame())
                
                if best_model and not training_X.empty:
                    try:
                        model_step = best_model
                        if hasattr(model_step, 'named_steps') and 'model' in model_step.named_steps:
                            model_step = model_step.named_steps['model']
                        
                        importances = None
                        imp_label = 'Importance'
                        imp_color = '#4CAF50'
                        
                        if hasattr(model_step, 'feature_importances_'):
                            importances = model_step.feature_importances_
                        elif hasattr(model_step, 'coef_'):
                            coefs = model_step.coef_
                            importances = np.abs(coefs).mean(axis=0) if coefs.ndim > 1 else np.abs(coefs)
                            imp_label = '|Coefficient|'
                            imp_color = '#FF9800'
                        
                        if importances is not None:
                            st.subheader("📌 Feature Importance")
                            feature_names = training_X.columns.tolist()
                            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                            feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True).tail(15)
                            
                            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                            ax_imp.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=imp_color)
                            ax_imp.set_xlabel(imp_label)
                            ax_imp.set_title(f'Feature Importance — {best_model_name}')
                            plt.tight_layout()
                            
                            c1, c2, c3 = st.columns([1, 2, 1])
                            with c2:
                                st.pyplot(fig_imp)
                            st.markdown("---")
                    except Exception as e:
                        st.warning(f"Could not generate feature importance: {e}")
                
                # --- Distribution of columns with <= 5 unique values ---
                st.subheader("📊 Distribution of Categorical")
                dist_cols = [c for c in df_original.columns if df_original[c].nunique() <= 5]
                
                if dist_cols:
                    for feat in dist_cols:
                        col_data = df_original[feat]
                        fig_d, ax_d = plt.subplots(figsize=(6, 4))
                        if pd.api.types.is_numeric_dtype(col_data):
                            sns.histplot(col_data, kde=False, ax=ax_d, color='#2196F3', discrete=True)
                            ax_d.set_title(f'Distribution of {feat}')
                            desc_d = f"Bar chart showing the distribution of '{feat}' ({col_data.nunique()} unique values)."
                        else:
                            sns.countplot(y=col_data, ax=ax_d, color='#2196F3')
                            ax_d.set_title(f'Count Plot of {feat}')
                            desc_d = f"Count plot showing the frequency of each category in '{feat}' ({col_data.nunique()} unique values)."
                        plt.tight_layout()
                        
                        c1, c2, c3 = st.columns([1, 2, 1])
                        with c2:
                            st.pyplot(fig_d)
                            st.caption(desc_d)
                        st.markdown("---")
                else:
                    st.info("No columns with 5 or fewer unique values found.")
                            
            with tab2:
                st.header("Model Evaluation")
                problem_type = result.get('problem_type', '')
                
                if 'metrics' in result:
                    # Convert metrics dict to dataframe for nice display
                    metrics_df = pd.DataFrame(result['metrics']).T
                    st.table(metrics_df)
                    
                st.success(f"**Best Performing Model:** {result.get('model_name', 'N/A')}")
                
                if problem_type == "Clustering":
                    # --- Cluster Visualization ---
                    st.markdown("---")
                    st.subheader("🔍 Cluster Assignment Preview")
                    best_model_name = result.get('model_name', '')
                    best_model = result.get('models', {}).get(best_model_name)
                    training_X = result.get('X', pd.DataFrame())
                    
                    if best_model and not training_X.empty:
                        try:
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(training_X.select_dtypes(include=['number']))
                            
                            if hasattr(best_model, 'labels_'):
                                labels = best_model.labels_
                            elif hasattr(best_model, 'predict'):
                                labels = best_model.predict(X_scaled)
                            else:
                                labels = None
                            
                            if labels is not None:
                                cluster_df = df_original.copy()
                                cluster_df['Cluster'] = labels
                                st.write(f"**{len(set(labels) - {-1})}** clusters found")
                                st.dataframe(cluster_df.head(20))
                                
                                # Cluster size distribution
                                fig_cl, ax_cl = plt.subplots(figsize=(6, 4))
                                pd.Series(labels).value_counts().sort_index().plot(kind='bar', ax=ax_cl, color='#4CAF50')
                                ax_cl.set_title("Cluster Sizes")
                                ax_cl.set_xlabel("Cluster")
                                ax_cl.set_ylabel("Count")
                                plt.tight_layout()
                                c1, c2, c3 = st.columns([1, 2, 1])
                                with c2:
                                    st.pyplot(fig_cl)
                        except Exception as e:
                            st.warning(f"Could not visualize clusters: {e}")
                else:
                    # --- Prediction Section (supervised only) ---
                    st.markdown("---")
                    st.subheader("🔮 Predict with Best Model")
                    best_model_name = result.get('model_name', '')
                    best_model = result.get('models', {}).get(best_model_name)
                    training_X = result.get('X', pd.DataFrame())
                    
                    if best_model and not training_X.empty:
                        
                        @st.fragment
                        def prediction_fragment():
                            st.info(f"Fill in the values below (same columns as your uploaded dataset) and click **Predict** to get a prediction from **{best_model_name}**.")
                            
                            # Show original dataset columns (excluding target)
                            original_cols = [c for c in df_original.columns if c != target]
                            
                            with st.form("prediction_form"):
                                input_data = {}
                                cols_layout = st.columns(3)
                                for i, col_name in enumerate(original_cols):
                                    with cols_layout[i % 3]:
                                        if df_original[col_name].dtype == 'object' or df_original[col_name].dtype.name == 'category':
                                            unique_vals = sorted(df_original[col_name].dropna().unique().tolist())
                                            input_data[col_name] = st.selectbox(f"`{col_name}`", options=unique_vals)
                                        elif df_original[col_name].dtype == 'bool':
                                            input_data[col_name] = st.selectbox(f"`{col_name}`", options=[True, False])
                                        elif pd.api.types.is_integer_dtype(df_original[col_name]):
                                            min_val = int(df_original[col_name].min())
                                            max_val = int(df_original[col_name].max())
                                            median_val = int(df_original[col_name].median())
                                            input_data[col_name] = st.number_input(
                                                f"`{col_name}`", 
                                                min_value=min_val, max_value=max_val,
                                                value=median_val, step=1
                                            )
                                        elif pd.api.types.is_float_dtype(df_original[col_name]):
                                            min_val = float(df_original[col_name].min())
                                            max_val = float(df_original[col_name].max())
                                            median_val = float(df_original[col_name].median())
                                            input_data[col_name] = st.number_input(
                                                f"`{col_name}`", 
                                                min_value=min_val, max_value=max_val,
                                                value=median_val, step=0.01, format="%.2f"
                                            )
                                        else:
                                            input_data[col_name] = st.number_input(f"`{col_name}`", value=float(df_original[col_name].median()), step=0.01, format="%.2f")
                                
                                submitted = st.form_submit_button("🔮 Predict")
                            
                            if submitted:
                                try:
                                    from pipeline.data_cleaning import clean_data
                                    from pipeline.feature_engineering import perform_feature_engineering
                                    
                                    # Build prediction row with a dummy target value
                                    pred_row = input_data.copy()
                                    if pd.api.types.is_numeric_dtype(df_original[target]):
                                        pred_row[target] = 0
                                    else:
                                        pred_row[target] = df_original[target].mode()[0]
                                    
                                    # Combine with original data so pipeline transformations are consistent
                                    pred_df = pd.DataFrame([pred_row])
                                    combined_df = pd.concat([df_original, pred_df], ignore_index=True)
                                    
                                    # Run through same pipeline
                                    cleaned_combined, _ = clean_data(combined_df)
                                    X_combined, _, _ = perform_feature_engineering(cleaned_combined, target, problem_type)
                                    
                                    # Extract last row (prediction input) and align to training columns
                                    X_pred = X_combined.iloc[[-1]]
                                    X_pred = X_pred.reindex(columns=training_X.columns, fill_value=0)
                                    
                                    prediction = best_model.predict(X_pred)
                                    predicted_value = prediction[0]
                                    
                                    # Reverse-map encoded prediction back to original label
                                    if df_original[target].dtype == 'object' or df_original[target].dtype.name == 'category':
                                        from sklearn.preprocessing import LabelEncoder
                                        le = LabelEncoder()
                                        le.fit(df_original[target].astype(str))
                                        predicted_value = le.inverse_transform([int(predicted_value)])[0]
                                    
                                    st.success(f"**Predicted `{target}`: {predicted_value}**")
                                except Exception as e:
                                    st.error(f"Prediction error: {e}")
                        
                        prediction_fragment()
                    else:
                        st.warning("Model or features not available for prediction.")
                
            with tab3:
                st.header("AI Explanation")
                st.markdown(result.get('explanation', "No explanation generated."))
                
            with tab4:
                st.header("Processing Logs")
                
                def _to_bullet_list(text: str) -> str:
                    """Convert newline-separated report lines to bullet point list."""
                    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
                    if not lines:
                        return text
                    return '\n\n'.join(f"- {line}" for line in lines)
                
                st.subheader("Data Cleaning Report")
                st.markdown(_to_bullet_list(result.get('cleaning_report', 'No report')))
                
                st.subheader("Feature Engineering Report")
                st.markdown(_to_bullet_list(result.get('feature_report', 'No report')))
                
                # --- Agent Graph Visualization ---
                st.subheader("🗺️ Agent Pipeline Graph")
                try:
                    graph = build_graph()
                    graph_drawable = graph.get_graph()
                    try:
                        png_bytes = graph_drawable.draw_mermaid_png()
                        st.image(png_bytes, caption="Agent Pipeline Graph", width=300)
                    except Exception:
                        mermaid_str = graph_drawable.draw_mermaid()
                        st.code(mermaid_str, language="mermaid")
                except Exception as e:
                    st.warning(f"Could not render agent graph: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
