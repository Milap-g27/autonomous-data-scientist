from typing import Any, Dict
import warnings

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional imports for gradient boosting libraries
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


DEFAULT_RANDOM_STATE = 42


def _make_preprocessor() -> ColumnTransformer:
    """Scale only numeric columns and passthrough all others."""
    return ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                lambda df: df.select_dtypes(include=["number"], exclude=["bool"]).columns,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", _make_preprocessor()), ("model", model)])


def _param_space_size(param_distributions: Any) -> int:
    """Estimate total candidates so n_iter does not exceed search space."""
    if isinstance(param_distributions, list):
        total = 0
        for item in param_distributions:
            total += _param_space_size(item)
        return max(1, total)

    if not isinstance(param_distributions, dict):
        return 1

    total = 1
    for values in param_distributions.values():
        try:
            total *= max(1, len(values))
        except TypeError:
            return max(1, total)
    return max(1, total)


def _fit_default(pipeline: Pipeline, X, y):
    pipeline.fit(X, y)
    return pipeline


def _tune(
    pipeline: Pipeline,
    param_distributions: Any,
    X,
    y,
    cv_folds: int,
    scoring: str,
    n_iter: int = 20,
):
    """
    Run RandomizedSearchCV and return the best estimator.
    If search fails (small data, invalid combos, etc.), fit defaults instead.
    """
    effective_n_iter = max(1, min(n_iter, _param_space_size(param_distributions)))

    try:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=effective_n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=DEFAULT_RANDOM_STATE,
            error_score="raise",
        )
        search.fit(X, y)
        return search.best_estimator_
    except Exception:
        return _fit_default(pipeline, X, y)


def _max_knn_neighbors(n_rows: int, cv_folds: int) -> int:
    if n_rows <= 2:
        return 1
    smallest_train_fold = max(1, n_rows - int(np.ceil(n_rows / max(2, cv_folds))))
    return max(1, min(30, smallest_train_fold))


def _train_regression(X, y) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    n_rows = len(X)
    cv_folds = max(2, min(5, n_rows // 10 if n_rows >= 10 else 2))
    scoring = "r2"

    try:
        baseline = _make_pipeline(LinearRegression())
        baseline.fit(X, y)
        models["Linear Regression"] = baseline
    except Exception:
        pass

    try:
        models["Ridge Regression"] = _tune(
            _make_pipeline(Ridge()),
            {"model__alpha": np.logspace(-4, 4, 60)},
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    try:
        models["Lasso Regression"] = _tune(
            _make_pipeline(Lasso(max_iter=10000)),
            {"model__alpha": np.logspace(-5, 1, 50)},
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    try:
        models["ElasticNet"] = _tune(
            _make_pipeline(ElasticNet(max_iter=10000)),
            {
                "model__alpha": np.logspace(-5, 1, 40),
                "model__l1_ratio": np.linspace(0.1, 0.9, 9),
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=24,
        )
    except Exception:
        pass

    try:
        models["SVR"] = _tune(
            _make_pipeline(SVR()),
            {
                "model__kernel": ["rbf", "poly", "sigmoid"],
                "model__C": np.logspace(-2, 3, 18),
                "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 0, 8)),
                "model__epsilon": [0.01, 0.1, 0.2, 0.5],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        neighbor_upper = _max_knn_neighbors(n_rows, cv_folds)
        models["KNN Regressor"] = _tune(
            _make_pipeline(KNeighborsRegressor()),
            {
                "model__n_neighbors": list(range(1, neighbor_upper + 1)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 40],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    try:
        models["Decision Tree"] = _tune(
            _make_pipeline(DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=24,
        )
    except Exception:
        pass

    try:
        models["Random Forest"] = _tune(
            _make_pipeline(RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5, None],
                "model__bootstrap": [True, False],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=30,
        )
    except Exception:
        pass

    try:
        models["Gradient Boosting"] = _tune(
            _make_pipeline(GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 5, 7],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__max_features": ["sqrt", "log2", None],
                "model__min_samples_split": [2, 5, 10],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        models["AdaBoost"] = _tune(
            _make_pipeline(AdaBoostRegressor(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [50, 100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                "model__loss": ["linear", "square", "exponential"],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    if HAS_XGBOOST:
        try:
            models["XGBoost"] = _tune(
                _make_pipeline(XGBRegressor(random_state=DEFAULT_RANDOM_STATE, verbosity=0)),
                {
                    "model__n_estimators": [100, 200, 300, 500],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "model__max_depth": [3, 5, 7, 9],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0, 0.01, 0.1, 1],
                    "model__reg_lambda": [0.5, 1, 2, 5],
                    "model__min_child_weight": [1, 3, 5],
                    "model__gamma": [0, 0.1, 0.3],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=36,
            )
        except Exception:
            pass

    if HAS_LIGHTGBM:
        try:
            models["LightGBM"] = _tune(
                _make_pipeline(LGBMRegressor(random_state=DEFAULT_RANDOM_STATE, verbose=-1)),
                {
                    "model__n_estimators": [100, 200, 300, 500],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__max_depth": [-1, 5, 10, 15],
                    "model__num_leaves": [31, 63, 127],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0, 0.1, 1],
                    "model__reg_lambda": [0, 0.1, 1],
                    "model__min_child_samples": [10, 20, 50],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=36,
            )
        except Exception:
            pass

    if HAS_CATBOOST:
        try:
            models["CatBoost"] = _tune(
                _make_pipeline(CatBoostRegressor(random_seed=DEFAULT_RANDOM_STATE, verbose=0)),
                {
                    "model__iterations": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__depth": [4, 6, 8, 10],
                    "model__l2_leaf_reg": [1, 3, 5, 10],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bylevel": [0.6, 0.8, 1.0],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=28,
            )
        except Exception:
            pass

    try:
        models["Neural Network (MLP)"] = _tune(
            _make_pipeline(MLPRegressor(random_state=DEFAULT_RANDOM_STATE, early_stopping=True)),
            {
                "model__hidden_layer_sizes": [
                    (64,),
                    (128,),
                    (64, 32),
                    (128, 64),
                    (128, 64, 32),
                    (256, 128, 64),
                ],
                "model__activation": ["relu", "tanh"],
                "model__alpha": np.logspace(-5, -1, 20),
                "model__learning_rate_init": [0.001, 0.005, 0.01],
                "model__max_iter": [500, 1000],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        candidates = [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
        ]
        estimators = [
            (f"reg_{index}", models[name])
            for index, name in enumerate(candidates)
            if name in models
        ]
        if len(estimators) >= 2:
            ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X, y)
            models["Voting Ensemble"] = ensemble
    except Exception:
        pass

    return models


def _min_class_count(y) -> int:
    try:
        return int(y.value_counts().min())
    except Exception:
        _, counts = np.unique(np.asarray(y), return_counts=True)
        return int(counts.min())


def _train_classification(X, y) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    n_rows = len(X)
    min_class_count = _min_class_count(y)
    cv_folds = max(2, min(5, min_class_count))
    scoring = "accuracy"

    try:
        logistic_param_space = [
            {
                "model__C": np.logspace(-3, 3, 30),
                "model__penalty": ["l1", "l2"],
                "model__solver": ["saga"],
            },
            {
                "model__C": np.logspace(-3, 3, 30),
                "model__penalty": ["elasticnet"],
                "model__solver": ["saga"],
                "model__l1_ratio": [0.2, 0.4, 0.6, 0.8],
            },
        ]
        models["Logistic Regression"] = _tune(
            _make_pipeline(LogisticRegression(max_iter=5000, random_state=DEFAULT_RANDOM_STATE)),
            logistic_param_space,
            X,
            y,
            cv_folds,
            scoring,
            n_iter=24,
        )
    except Exception:
        pass

    try:
        models["SVM"] = _tune(
            _make_pipeline(SVC(probability=True, random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__kernel": ["rbf", "poly", "sigmoid"],
                "model__C": np.logspace(-2, 3, 18),
                "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 0, 8)),
                "model__degree": [2, 3, 4],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        neighbor_upper = _max_knn_neighbors(n_rows, cv_folds)
        models["KNN Classifier"] = _tune(
            _make_pipeline(KNeighborsClassifier()),
            {
                "model__n_neighbors": list(range(1, neighbor_upper + 1)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 40],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    try:
        models["Naive Bayes"] = _tune(
            _make_pipeline(GaussianNB()),
            {"model__var_smoothing": np.logspace(-12, -1, 30)},
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    try:
        models["Decision Tree"] = _tune(
            _make_pipeline(DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__max_features": ["sqrt", "log2", None],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=24,
        )
    except Exception:
        pass

    try:
        models["Random Forest"] = _tune(
            _make_pipeline(RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__bootstrap": [True, False],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=30,
        )
    except Exception:
        pass

    try:
        models["Gradient Boosting"] = _tune(
            _make_pipeline(GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 5, 7],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__max_features": ["sqrt", "log2", None],
                "model__min_samples_split": [2, 5, 10],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        models["AdaBoost"] = _tune(
            _make_pipeline(AdaBoostClassifier(random_state=DEFAULT_RANDOM_STATE)),
            {
                "model__n_estimators": [50, 100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                "model__algorithm": ["SAMME"],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=20,
        )
    except Exception:
        pass

    if HAS_XGBOOST:
        try:
            models["XGBoost"] = _tune(
                _make_pipeline(
                    XGBClassifier(
                        random_state=DEFAULT_RANDOM_STATE,
                        verbosity=0,
                        eval_metric="logloss",
                    )
                ),
                {
                    "model__n_estimators": [100, 200, 300, 500],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "model__max_depth": [3, 5, 7, 9],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0, 0.01, 0.1, 1],
                    "model__reg_lambda": [0.5, 1, 2, 5],
                    "model__min_child_weight": [1, 3, 5],
                    "model__gamma": [0, 0.1, 0.3],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=36,
            )
        except Exception:
            pass

    if HAS_LIGHTGBM:
        try:
            models["LightGBM"] = _tune(
                _make_pipeline(LGBMClassifier(random_state=DEFAULT_RANDOM_STATE, verbose=-1)),
                {
                    "model__n_estimators": [100, 200, 300, 500],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__max_depth": [-1, 5, 10, 15],
                    "model__num_leaves": [31, 63, 127],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0, 0.1, 1],
                    "model__reg_lambda": [0, 0.1, 1],
                    "model__min_child_samples": [10, 20, 50],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=36,
            )
        except Exception:
            pass

    if HAS_CATBOOST:
        try:
            models["CatBoost"] = _tune(
                _make_pipeline(CatBoostClassifier(random_seed=DEFAULT_RANDOM_STATE, verbose=0)),
                {
                    "model__iterations": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "model__depth": [4, 6, 8, 10],
                    "model__l2_leaf_reg": [1, 3, 5, 10],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bylevel": [0.6, 0.8, 1.0],
                },
                X,
                y,
                cv_folds,
                scoring,
                n_iter=28,
            )
        except Exception:
            pass

    try:
        models["Neural Network (MLP)"] = _tune(
            _make_pipeline(MLPClassifier(random_state=DEFAULT_RANDOM_STATE, early_stopping=True)),
            {
                "model__hidden_layer_sizes": [
                    (64,),
                    (128,),
                    (64, 32),
                    (128, 64),
                    (128, 64, 32),
                    (256, 128, 64),
                ],
                "model__activation": ["relu", "tanh"],
                "model__alpha": np.logspace(-5, -1, 20),
                "model__learning_rate_init": [0.001, 0.005, 0.01],
                "model__max_iter": [500, 1000],
            },
            X,
            y,
            cv_folds,
            scoring,
            n_iter=28,
        )
    except Exception:
        pass

    try:
        candidates = [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
        ]
        estimators = [
            (f"cls_{index}", models[name])
            for index, name in enumerate(candidates)
            if name in models
        ]
        if len(estimators) >= 2:
            ensemble = VotingClassifier(estimators=estimators, voting="soft")
            ensemble.fit(X, y)
            models["Voting Ensemble"] = ensemble
    except Exception:
        pass

    return models


def _silhouette_for_labels(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = set(labels)
    cluster_count = len(unique_labels - {-1})
    if cluster_count < 2:
        return -1.0

    if -1 in unique_labels:
        mask = labels != -1
        if int(mask.sum()) < 2 or len(set(labels[mask])) < 2:
            return -1.0
        return float(silhouette_score(X_scaled[mask], labels[mask]))

    if len(unique_labels) >= len(labels):
        return -1.0

    return float(silhouette_score(X_scaled, labels))


def _best_k_by_silhouette(X_scaled: np.ndarray) -> int:
    upper = min(10, len(X_scaled) - 1)
    if upper < 2:
        return 2

    best_k = 2
    best_score = -1.0
    for k in range(2, upper + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=10).fit_predict(
                X_scaled
            )
            score = _silhouette_for_labels(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            pass
    return best_k


def _attach_scaler(model: Any, scaler: StandardScaler) -> Any:
    model._scaler = scaler
    return model


def _train_clustering(X) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    numeric_X = X.select_dtypes(include=["number"], exclude=["bool"])
    if numeric_X.shape[1] == 0 or len(numeric_X) < 2:
        return models

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_X)
    upper_k = min(10, len(X_scaled) - 1)
    best_k = _best_k_by_silhouette(X_scaled)

    try:
        best_model = None
        best_score = -1.0
        for k in range(2, upper_k + 1):
            for n_init in [10, 20, 30]:
                try:
                    km = KMeans(n_clusters=k, n_init=n_init, random_state=DEFAULT_RANDOM_STATE)
                    labels = km.fit_predict(X_scaled)
                    score = _silhouette_for_labels(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = km
                except Exception:
                    pass
        if best_model is not None:
            models["KMeans"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    try:
        best_model = None
        best_score = -1.0
        for linkage in ["ward", "complete", "average", "single"]:
            for k in range(2, upper_k + 1):
                try:
                    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                    labels = agg.fit_predict(X_scaled)
                    score = _silhouette_for_labels(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = agg
                except Exception:
                    pass
        if best_model is not None:
            best_model.fit(X_scaled)
            models["Agglomerative"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    try:
        best_model = None
        best_score = -1.0
        for eps in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            for min_samples in [3, 5, 8, 10, 15]:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    score = _silhouette_for_labels(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = dbscan
                except Exception:
                    pass
        if best_model is not None:
            models["DBSCAN"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    try:
        from sklearn.cluster import estimate_bandwidth

        best_model = None
        best_score = -1.0
        base_bw = estimate_bandwidth(X_scaled, quantile=0.2)
        if base_bw > 0:
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
                try:
                    bandwidth = base_bw * factor
                    if bandwidth <= 0:
                        continue
                    ms_model = MeanShift(bandwidth=bandwidth)
                    labels = ms_model.fit_predict(X_scaled)
                    score = _silhouette_for_labels(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = ms_model
                except Exception:
                    pass

        if best_model is None:
            default_ms = MeanShift()
            default_ms.fit(X_scaled)
            models["MeanShift"] = _attach_scaler(default_ms, scaler)
        else:
            models["MeanShift"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    try:
        best_model = None
        best_score = -1.0
        for threshold in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for branching_factor in [30, 50, 100]:
                try:
                    birch = Birch(
                        n_clusters=best_k,
                        threshold=threshold,
                        branching_factor=branching_factor,
                    )
                    labels = birch.fit_predict(X_scaled)
                    score = _silhouette_for_labels(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_model = birch
                except Exception:
                    pass
        if best_model is not None:
            best_model.fit(X_scaled)
            models["Birch"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    try:
        best_model = None
        best_bic = np.inf
        for k in range(2, upper_k + 1):
            for covariance_type in ["full", "tied", "diag", "spherical"]:
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=covariance_type,
                        random_state=DEFAULT_RANDOM_STATE,
                        n_init=3,
                    )
                    gmm.fit(X_scaled)
                    bic = gmm.bic(X_scaled)
                    if bic < best_bic:
                        best_bic = bic
                        best_model = gmm
                except Exception:
                    pass
        if best_model is not None:
            models["Gaussian Mixture"] = _attach_scaler(best_model, scaler)
    except Exception:
        pass

    return models


def train_models(X, y, problem_type: str) -> Dict[str, Any]:
    """
    Train and tune multiple models based on problem type.

    problem_type must be one of: Regression, Classification, Clustering.
    """
    warnings.filterwarnings("ignore")

    if problem_type == "Regression":
        return _train_regression(X, y)
    if problem_type == "Classification":
        return _train_classification(X, y)
    if problem_type == "Clustering":
        return _train_clustering(X)

    raise ValueError(
        f"Unknown problem_type: '{problem_type}'. "
        "Choose 'Regression', 'Classification', or 'Clustering'."
    )