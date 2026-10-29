"""Genera un dataset de referencia y lotes semanales de producción con drift controlado.

Generates a reference dataset and weekly production batches with controlled drift.
Same seed -> same data, so results are reproducible.

Escenario / scenario (card-payment fraud model):
- amount: drift gradual desde la semana 1 / gradual drift from week 1.
- channel: cambio brusco en la semana 6 (campaña de la app) / sudden shift in week 6 (app campaign).
- customer_age, account_age_days: estables / stable.
- Semana 11: 8 % de customer_age faltante (error del pipeline) / week 11: 8% missing customer_age (pipeline bug).
- Semana 9: cambia la relación entre variables y fraude (concept drift) sin cambiar las distribuciones.
  Week 9: the feature-fraud relationship changes (concept drift) without changing the distributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_REFERENCE = 10_000
N_BATCH = 2_000
WEEKS = 14
CONCEPT_WEEK = 9
MISSING_WEEK = 11

NUMERIC = ["amount", "customer_age", "account_age_days"]
CATEGORICAL = ["channel"]
CHANNELS = ["web", "app", "pos"]


def _features(rng, n: int, amount_shift: float = 0.0, channel_p=(0.50, 0.35, 0.15)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": rng.lognormal(3.8 + amount_shift, 0.9, n).round(2),
            "customer_age": np.clip(rng.normal(41, 12, n), 18, 90).round(0),
            "account_age_days": rng.gamma(1.5, 300, n).round(0),
            "channel": rng.choice(CHANNELS, n, p=channel_p),
        }
    )


def _fraud(rng, df: pd.DataFrame, concept: str) -> np.ndarray:
    z_amount = (np.log(df["amount"]) - 3.8) / 0.9
    z_account = (np.log1p(df["account_age_days"]) - 5.7) / 1.0
    if concept == "v1":
        # Fraude en importes altos, web y cuentas nuevas. Fraud on high amounts, web, and new accounts.
        logit = -4.0 + 1.2 * z_amount + 1.0 * (df["channel"] == "web") - 0.9 * z_account
    else:
        # Nuevo patrón: fraude en la app; importe y antigüedad casi no influyen.
        # New pattern: app fraud; amount and account age barely matter.
        logit = -3.7 + 0.3 * z_amount + 1.2 * (df["channel"] == "app") - 0.2 * z_account
    return (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)


def reference(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _features(rng, N_REFERENCE)
    df["fraud"] = _fraud(rng, df, "v1")
    return df


def production(seed: int = SEED) -> pd.DataFrame:
    """Lotes semanales concatenados con la columna week. Weekly batches concatenated, with a week column."""
    rng = np.random.default_rng(seed + 1)
    batches = []
    for week in range(1, WEEKS + 1):
        channel_p = (0.50, 0.35, 0.15) if week < 6 else (0.28, 0.60, 0.12)
        df = _features(rng, N_BATCH, amount_shift=0.04 * week, channel_p=channel_p)
        df["fraud"] = _fraud(rng, df, "v1" if week < CONCEPT_WEEK else "v2")
        if week == MISSING_WEEK:
            df.loc[rng.random(N_BATCH) < 0.08, "customer_age"] = np.nan
        df.insert(0, "week", week)
        batches.append(df)
    return pd.concat(batches, ignore_index=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    reference().to_csv(out / "reference.csv", index=False)
    production().to_csv(out / "production.csv", index=False)
    print(f"reference + production -> {out}")
