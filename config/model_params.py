"""
BTC Prediction Project - Model Hyperparameter Grids
"""

# XGBoost hyperparameter space for Bayesian optimization
XGBOOST_PARAMS = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0, 10),
    'reg_lambda': (0, 10),
    'scale_pos_weight': (1, 10)  # Handle class imbalance
}

# Random Forest hyperparameter space for Bayesian optimization
RANDOM_FOREST_PARAMS = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Logistic Regression hyperparameter grid for GridSearchCV
LOGISTIC_REGRESSION_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 2000, 5000]
}

# Meta-LR hyperparameter grid for Level 1 model
META_LR_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None]
}

# Bayesian optimization settings
BAYESIAN_OPTIMIZATION = {'n_iter': 50, 'random_state': 42, 'n_jobs': -1}

# GridSearchCV settings
GRID_SEARCH = {'cv': 5, 'scoring': 'f1', 'n_jobs': -1, 'verbose': 1}

# Feature pruning settings
FEATURE_PRUNING = {
    'importance_threshold': 0.001,  # Remove features with importance < 0.001
    'max_features': 200,  # Maximum number of features to keep
    'min_features': 50  # Minimum number of features to keep
}

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'
]

# Primary metric for model selection
PRIMARY_METRIC = 'f1'
