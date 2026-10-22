"""Limpieza sin data leakage: las estadísticas se calculan solo con el conjunto de entrenamiento.

Cleaning without data leakage: statistics are computed on the training split only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default"


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y registros inválidos. Drops duplicates and invalid records."""
    df = df.drop_duplicates(subset="application_id")
    return df[df["age"].between(18, 100)].reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """División estratificada y mezclada. Stratified, shuffled split."""
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], shuffle=True, random_state=seed)


def fit_cleaning(train: pd.DataFrame) -> dict:
    """Aprende la mediana y los límites IQR de income. Learns income median and IQR bounds."""
    income = train["income"].dropna()
    q1, q3 = income.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {"median": income.median(), "low": q1 - 1.5 * iqr, "high": q3 + 1.5 * iqr}


def apply_cleaning(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    df["income_missing"] = df["income"].isna().astype(int)
    df["income"] = df["income"].fillna(stats["median"]).clip(stats["low"], stats["high"])
    return df
