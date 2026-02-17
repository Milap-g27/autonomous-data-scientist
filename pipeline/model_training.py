from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from typing import Dict, Any
import numpy as np
import warnings

# Optional imports for LightGBM and CatBoost
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def train_models(X, y, problem_type: str) -> Dict[str, Any]:
    """
    Trains multiple ML models based on problem type.
    Regression: Linear, Ridge, Lasso, ElasticNet, SVR, KNN, Random Forest, 
                Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP, Voting Ensemble.
    Classification: Logistic, SVM, KNN, Naive Bayes, Decision Tree, Random Forest,
                    Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, MLP, Voting Ensemble.
    Clustering: KMeans, DBSCAN, Agglomerative, MeanShift, Birch, GaussianMixture.
    Returns a dictionary of trained models.
    """
    models = {}
    warnings.filterwarnings("ignore")
    
    # Create preprocessor for selective scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), lambda df: df.select_dtypes(include=['number'], exclude=['bool']).columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    def _make_pipeline(model):
        return Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    if problem_type == "Regression":
        # Determine safe CV folds
        cv_folds = max(2, min(3, len(X)))

        # 1. Linear Regression (Baseline)
        try:
            lr = _make_pipeline(LinearRegression())
            lr.fit(X, y)
            models['Linear Regression'] = lr
        except Exception:
            pass
        
        # 2. Ridge Regression
        try:
            ridge = _make_pipeline(Ridge(alpha=1.0))
            ridge.fit(X, y)
            models['Ridge Regression'] = ridge
        except Exception:
            pass
        
        # 3. Lasso Regression
        try:
            lasso = _make_pipeline(Lasso(alpha=0.1, max_iter=5000))
            lasso.fit(X, y)
            models['Lasso Regression'] = lasso
        except Exception:
            pass
        
        # 4. ElasticNet
        try:
            enet = _make_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))
            enet.fit(X, y)
            models['ElasticNet'] = enet
        except Exception:
            pass
        
        # 5. SVR (Support Vector Regression)
        try:
            svr = _make_pipeline(SVR(kernel='rbf', C=1.0))
            svr.fit(X, y)
            models['SVR'] = svr
        except Exception:
            pass
        
        # 6. KNN Regressor
        try:
            knn = _make_pipeline(KNeighborsRegressor(n_neighbors=5))
            knn.fit(X, y)
            models['KNN Regressor'] = knn
        except Exception:
            pass

        # 7. Random Forest Regressor (with tuning)
        try:
            rf_pipeline = _make_pipeline(RandomForestRegressor(random_state=42))
            rf_param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
            rf_search = RandomizedSearchCV(
                rf_pipeline, rf_param_grid,
                n_iter=5, cv=cv_folds, scoring='r2', n_jobs=-1, random_state=42
            )
            rf_search.fit(X, y)
            models['Random Forest'] = rf_search.best_estimator_
        except Exception:
            pass
        
        # 8. Gradient Boosting Regressor
        try:
            gb = _make_pipeline(GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
            gb.fit(X, y)
            models['Gradient Boosting'] = gb
        except Exception:
            pass
        
        # 9. AdaBoost Regressor
        try:
            ada = _make_pipeline(AdaBoostRegressor(n_estimators=100, random_state=42))
            ada.fit(X, y)
            models['AdaBoost'] = ada
        except Exception:
            pass
        
        # 10. XGBoost Regressor
        if HAS_XGBOOST:
            try:
                xgb = _make_pipeline(XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0))
                xgb.fit(X, y)
                models['XGBoost'] = xgb
            except Exception:
                pass
        
        # 11. LightGBM Regressor
        if HAS_LIGHTGBM:
            try:
                lgbm = _make_pipeline(LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1))
                lgbm.fit(X, y)
                models['LightGBM'] = lgbm
            except Exception:
                pass
        
        # 12. CatBoost Regressor
        if HAS_CATBOOST:
            try:
                cb = _make_pipeline(CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=0))
                cb.fit(X, y)
                models['CatBoost'] = cb
            except Exception:
                pass
        
        # 13. MLP Regressor (Neural Network)
        try:
            mlp = _make_pipeline(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
            mlp.fit(X, y)
            models['Neural Network (MLP)'] = mlp
        except Exception:
            pass
        
        # 14. Voting Ensemble (top models)
        try:
            estimators = []
            if 'Linear Regression' in models:
                estimators.append(('lr', models['Linear Regression']))
            if 'Random Forest' in models:
                estimators.append(('rf', models['Random Forest']))
            if 'Gradient Boosting' in models:
                estimators.append(('gb', models['Gradient Boosting']))
            if len(estimators) >= 2:
                ensemble = VotingRegressor(estimators=estimators)
                ensemble.fit(X, y)
                models['Voting Ensemble'] = ensemble
        except Exception:
            pass
        
    elif problem_type == "Classification":
        # Determine safe number of CV folds based on smallest class
        min_class_count = int(y.value_counts().min())
        cv_folds = max(2, min(3, min_class_count))

        # 1. Logistic Regression (with tuning)
        try:
            log_reg = _make_pipeline(LogisticRegression(max_iter=1000))
            if min_class_count >= 2:
                lr_param_grid = {'model__C': [0.1, 1.0, 10.0]}
                lr_search = RandomizedSearchCV(
                    log_reg, lr_param_grid,
                    n_iter=3, cv=cv_folds, n_jobs=-1, random_state=42
                )
                lr_search.fit(X, y)
                models['Logistic Regression'] = lr_search.best_estimator_
            else:
                log_reg.fit(X, y)
                models['Logistic Regression'] = log_reg
        except Exception:
            pass
        
        # 2. SVM (Support Vector Classifier)
        try:
            svm = _make_pipeline(SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
            svm.fit(X, y)
            models['SVM'] = svm
        except Exception:
            pass
        
        # 3. KNN Classifier
        try:
            knn = _make_pipeline(KNeighborsClassifier(n_neighbors=5))
            knn.fit(X, y)
            models['KNN Classifier'] = knn
        except Exception:
            pass
        
        # 4. Naive Bayes
        try:
            nb = _make_pipeline(GaussianNB())
            nb.fit(X, y)
            models['Naive Bayes'] = nb
        except Exception:
            pass
        
        # 5. Decision Tree Classifier
        try:
            dt = _make_pipeline(DecisionTreeClassifier(max_depth=10, random_state=42))
            dt.fit(X, y)
            models['Decision Tree'] = dt
        except Exception:
            pass
        
        # 6. Random Forest Classifier (with tuning)
        try:
            rf_pipeline = _make_pipeline(RandomForestClassifier(random_state=42))
            rf_param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
            rf_search = RandomizedSearchCV(
                rf_pipeline, rf_param_grid,
                n_iter=5, cv=cv_folds, scoring='accuracy', n_jobs=-1, random_state=42
            )
            rf_search.fit(X, y)
            models['Random Forest'] = rf_search.best_estimator_
        except Exception:
            pass
        
        # 7. Gradient Boosting Classifier
        try:
            gb = _make_pipeline(GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
            gb.fit(X, y)
            models['Gradient Boosting'] = gb
        except Exception:
            pass
        
        # 8. AdaBoost Classifier
        try:
            ada = _make_pipeline(AdaBoostClassifier(n_estimators=100, random_state=42))
            ada.fit(X, y)
            models['AdaBoost'] = ada
        except Exception:
            pass
        
        # 9. XGBoost Classifier
        if HAS_XGBOOST:
            try:
                xgb = _make_pipeline(XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'))
                xgb.fit(X, y)
                models['XGBoost'] = xgb
            except Exception:
                pass
        
        # 10. LightGBM Classifier
        if HAS_LIGHTGBM:
            try:
                lgbm = _make_pipeline(LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1))
                lgbm.fit(X, y)
                models['LightGBM'] = lgbm
            except Exception:
                pass
        
        # 11. CatBoost Classifier
        if HAS_CATBOOST:
            try:
                cb = _make_pipeline(CatBoostClassifier(iterations=100, depth=5, random_seed=42, verbose=0))
                cb.fit(X, y)
                models['CatBoost'] = cb
            except Exception:
                pass
        
        # 12. MLP Classifier (Neural Network)
        try:
            mlp = _make_pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
            mlp.fit(X, y)
            models['Neural Network (MLP)'] = mlp
        except Exception:
            pass
        
        # 13. Voting Classifier (Soft Voting - top models)
        try:
            estimators = []
            if 'Logistic Regression' in models:
                estimators.append(('lr', models['Logistic Regression']))
            if 'Random Forest' in models:
                estimators.append(('rf', models['Random Forest']))
            if 'Gradient Boosting' in models:
                estimators.append(('gb', models['Gradient Boosting']))
            if len(estimators) >= 2:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                ensemble.fit(X, y)
                models['Voting Ensemble'] = ensemble
        except Exception:
            pass

    elif problem_type == "Clustering":
        # Scale data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))

        # Determine optimal k using silhouette score
        from sklearn.metrics import silhouette_score
        best_k = 3
        best_sil = -1
        for k in range(2, min(11, len(X))):
            try:
                km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_temp = km_temp.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels_temp)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            except Exception:
                pass

        # 1. KMeans
        try:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            km.fit(X_scaled)
            km._scaler = scaler  # attach scaler for prediction
            km._best_k = best_k
            models['KMeans'] = km
        except Exception:
            pass

        # 2. Agglomerative Clustering
        try:
            agg = AgglomerativeClustering(n_clusters=best_k)
            agg.fit(X_scaled)
            agg._scaler = scaler
            models['Agglomerative'] = agg
        except Exception:
            pass

        # 3. DBSCAN
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(X_scaled)
            dbscan._scaler = scaler
            models['DBSCAN'] = dbscan
        except Exception:
            pass

        # 4. MeanShift
        try:
            ms = MeanShift()
            ms.fit(X_scaled)
            ms._scaler = scaler
            models['MeanShift'] = ms
        except Exception:
            pass

        # 5. Birch
        try:
            birch = Birch(n_clusters=best_k)
            birch.fit(X_scaled)
            birch._scaler = scaler
            models['Birch'] = birch
        except Exception:
            pass

        # 6. Gaussian Mixture
        try:
            gmm = GaussianMixture(n_components=best_k, random_state=42)
            gmm.fit(X_scaled)
            gmm._scaler = scaler
            models['Gaussian Mixture'] = gmm
        except Exception:
            pass
        
    return models
