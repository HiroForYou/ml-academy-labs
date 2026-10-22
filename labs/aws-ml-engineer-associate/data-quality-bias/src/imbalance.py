"""Compara estrategias para el desbalance de clases con XGBoost.

Compares class-imbalance strategies with XGBoost. The test split is never resampled.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

FEATURES = ["age", "income", "loan_amount", "credit_history_years", "debt_ratio", "income_missing"]
TARGET = "default"


def _metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def _xgb(**params) -> XGBClassifier:
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="auc", random_state=42, **params)


def compare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Devuelve métricas por estrategia y sus predicciones sobre test.

    Returns metrics per strategy and its test predictions (for confusion matrices).
    """
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]
    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())

    # SMOTE solo sobre el entrenamiento. SMOTE on the training split only.
    x_smote, y_smote = SMOTE(random_state=42).fit_resample(x_train, y_train)

    models = {
        "always 'no default'": (DummyClassifier(strategy="most_frequent"), x_train, y_train),
        "xgboost baseline": (_xgb(), x_train, y_train),
        # scale_pos_weight ≈ negativos / positivos. negatives / positives.
        f"scale_pos_weight={negatives / positives:.1f}": (_xgb(scale_pos_weight=negatives / positives), x_train, y_train),
        "SMOTE (train only)": (_xgb(), x_smote, y_smote),
    }

    rows, predictions = {}, {}
    for name, (model, x_fit, y_fit) in models.items():
        model.fit(x_fit, y_fit)
        predictions[name] = model.predict(x_test)
        rows[name] = _metrics(y_test, predictions[name], model.predict_proba(x_test)[:, 1])

    return pd.DataFrame(rows).T.round(3), predictions
