from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from typing import Dict, Any
import numpy as np

def train_models(X, y, problem_type: str) -> Dict[str, Any]:
    """
    Trains baseline, random forest, and ensemble models based on problem type.
    Returns a dictionary of trained models.
    """
    models = {}
    
    # Create preprocessor for selective scaling
    # Scale only numerical features (float/int), pass through binary/categorical (e.g. from OHE)
    # Exclude bool explicitly by restricting to float/int types.
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), lambda df: df.select_dtypes(include=['number'], exclude=['bool']).columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    if problem_type == "Regression":
        # 1. Linear Regression (Baseline)
        lr = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
        lr.fit(X, y)
        models['Linear Regression'] = lr
        
        # 2. Random Forest Regressor - Tune this!
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        rf_param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
        
        rf_search = RandomizedSearchCV(
            rf_pipeline, 
            rf_param_grid, 
            n_iter=5, # Keep it low for speed in this agent
            cv=3, 
            scoring='r2', 
            n_jobs=-1,
            random_state=42
        )
        rf_search.fit(X, y)
        models['Random Forest'] = rf_search.best_estimator_
        
        # 3. Voting Regressor (Ensemble)
        # We'll use the best RF and a standard LR
        
        lr_pipe = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
        # Use the best RF we found
        # best_rf = rf_search.best_estimator_.named_steps['model']
        # The tuned estimator is a pipeline. We can use it directly or extract the model step.
        # But VotingRegressor expects estimators. We can just use the tuned pipeline!
        
        ensemble = VotingRegressor(estimators=[
            ('lr', lr_pipe), 
            ('rf', models['Random Forest']) # The tuned pipeline
        ])
        ensemble.fit(X, y)
        models['Voting Ensemble'] = ensemble
        
    else: # Classification
        # 1. Logistic Regression (Baseline)
        log_reg = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000))])
        
        # Simple tuning for LR
        lr_param_grid = {
            'model__C': [0.1, 1.0, 10.0]
        }
        
        lr_search = RandomizedSearchCV(
            log_reg,
            lr_param_grid,
            n_iter=3,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        lr_search.fit(X, y)
        models['Logistic Regression'] = lr_search.best_estimator_
        
        # 2. Random Forest Classifier
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        rf_param_grid = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
        
        rf_search = RandomizedSearchCV(
            rf_pipeline, 
            rf_param_grid, 
            n_iter=5, 
            cv=3, 
            scoring='accuracy', # or f1_macro
            n_jobs=-1,
            random_state=42
        )
        rf_search.fit(X, y)
        models['Random Forest'] = rf_search.best_estimator_
        
        # 3. Voting Classifier (Soft Voting)
        ensemble = VotingClassifier(estimators=[
            ('lr_best', models['Logistic Regression']), 
            ('rf_best', models['Random Forest'])
        ], voting='soft')
        ensemble.fit(X, y)
        models['Voting Ensemble'] = ensemble
        
    return models
