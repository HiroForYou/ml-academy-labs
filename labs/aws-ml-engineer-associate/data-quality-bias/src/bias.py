"""Métricas de sesgo pre-training (mismas definiciones que SageMaker Clarify).

Pre-training bias metrics (same definitions as SageMaker Clarify).
  CI  = (n_a - n_d) / (n_a + n_d)   -> is facet d underrepresented?
  DPL = q_a - q_d                   -> does facet a get more positive labels?
"""

import pandas as pd


def class_imbalance(facet_d: pd.Series) -> float:
    n_d = int(facet_d.sum())
    n_a = len(facet_d) - n_d
    return (n_a - n_d) / (n_a + n_d)


def difference_in_proportions_of_labels(labels: pd.Series, facet_d: pd.Series) -> float:
    q_a = labels[~facet_d].mean()
    q_d = labels[facet_d].mean()
    return float(q_a - q_d)


def report(df: pd.DataFrame, facet_d: pd.Series, label: str) -> dict:
    return {
        "n_a": int((~facet_d).sum()),
        "n_d": int(facet_d.sum()),
        "q_a": float(df.loc[~facet_d, label].mean()),
        "q_d": float(df.loc[facet_d, label].mean()),
        "CI": class_imbalance(facet_d),
        "DPL": difference_in_proportions_of_labels(df[label], facet_d),
    }
