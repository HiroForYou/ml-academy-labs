"""Genera un dataset sintético de préstamos con problemas de calidad, sesgo y desbalance.

Generates a synthetic loan dataset with quality, bias and imbalance problems.
Same seed -> same data, so results are reproducible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10_000


def generate(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Faceta d (mayores de 60) subrepresentada: ~14 %. Facet d (over 60) is underrepresented.
    older = rng.random(n_rows) < 0.14
    age = np.where(older, rng.integers(61, 85, n_rows), rng.integers(18, 61, n_rows))

    income = rng.lognormal(mean=10.6, sigma=0.5, size=n_rows).round(0)
    credit_years = np.clip(age - 18 - rng.integers(0, 10, n_rows), 0, None)
    debt_ratio = rng.beta(2, 5, n_rows).round(3)
    loan_amount = (income * rng.uniform(0.1, 0.6, n_rows)).round(0)

    # Probabilidad de impago: depende de deuda, ingreso e historial; la faceta d tiene más impagos.
    # Default probability: driven by debt, income and history; facet d defaults more often.
    logit = -3.4 + 4.0 * debt_ratio - 0.6 * np.log(income / 40_000) - 0.03 * credit_years + 1.6 * older
    default = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{i:06d}" for i in range(n_rows)],
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_history_years": credit_years,
            "debt_ratio": debt_ratio,
            "default": default,
        }
    )

    # Faltantes en income, más frecuentes entre impagos. Missing income, more common among defaults.
    missing = rng.random(n_rows) < np.where(df["default"] == 1, 0.25, 0.08)
    df.loc[missing, "income"] = np.nan

    # Outliers de captura (ingresos x50). Data-entry outliers (income x50).
    outliers = rng.choice(n_rows, 40, replace=False)
    df.loc[outliers, "income"] = df.loc[outliers, "income"] * 50

    # Edades inválidas. Invalid ages.
    df.loc[rng.choice(n_rows, 15, replace=False), "age"] = 150

    # Solicitudes duplicadas. Duplicate applications.
    duplicates = df.sample(150, random_state=seed)
    return pd.concat([df, duplicates]).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "loans_raw.csv"
    out.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(out, index=False)
    print(f"{len(data)} rows -> {out}")
