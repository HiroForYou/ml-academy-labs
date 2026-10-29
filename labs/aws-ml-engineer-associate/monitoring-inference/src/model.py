"""Modelo de fraude y calidad semanal con retraso de etiquetas. Fraud model and weekly quality with label delay."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

NUMERIC = ["amount", "customer_age", "account_age_days"]
CATEGORICAL = ["channel"]


def train(ref: pd.DataFrame):
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p)), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=1000))
    model.fit(ref[NUMERIC + CATEGORICAL], ref["fraud"])
    # Umbral: alerta al 5 % de transacciones con mayor riesgo en la referencia. Threshold: flag the riskiest 5% in reference.
    model.threshold_ = float(np.quantile(model.predict_proba(ref[NUMERIC + CATEGORICAL])[:, 1], 0.95))
    return model


def weekly_quality(model, production: pd.DataFrame, label_delay: int) -> pd.DataFrame:
    """Calidad real de cada lote y la semana en que se conoce (cuando llegan las etiquetas).

    True quality of each batch and the week it becomes known (when the labels arrive).
    """
    rows = []
    for week, batch in production.groupby("week"):
        scores = model.predict_proba(batch[NUMERIC + CATEGORICAL])[:, 1]
        rows.append(
            {
                "week": week,
                "flagged_rate": (scores >= model.threshold_).mean(),  # se conoce al instante / known immediately
                "fraud_rate": batch["fraud"].mean(),
                "auc": roc_auc_score(batch["fraud"], scores),
                "recall": recall_score(batch["fraud"], scores >= model.threshold_),
                "known_at_week": week + label_delay,
            }
        )
    return pd.DataFrame(rows)
