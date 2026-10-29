"""Línea base, PSI, Kolmogorov-Smirnov y restricciones de calidad por lote.

Baseline, PSI, Kolmogorov-Smirnov, and data-quality constraints per batch.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Umbrales de PSI: convención habitual en la industria (riesgo crediticio), no una regla de AWS. Ajústalos a tu caso.
# PSI thresholds: a common industry convention (credit risk), not an AWS rule. Tune them for your case.
PSI_WARNING = 0.1
PSI_ALERT = 0.2
MISSING_TOLERANCE = 0.02  # faltantes por encima de la línea base / missing rate above baseline
EPS = 1e-4  # evita log(0) en bins vacíos / avoids log(0) in empty bins


def baseline(ref: pd.DataFrame, numeric: list[str], categorical: list[str], bins: int = 10) -> dict:
    """Estadísticas y bins de la referencia (como statistics.json + constraints.json).

    Reference statistics and bins (like statistics.json + constraints.json).
    """
    base = {"numeric": {}, "categorical": {}}
    for col in numeric:
        s = ref[col]
        edges = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, bins + 1)[1:-1]))
        base["numeric"][col] = {
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p50": s.median(),
            "max": s.max(),
            "missing": s.isna().mean(),
            "edges": edges,
            "proportions": _bin_props(s, edges),
        }
    for col in categorical:
        props = ref[col].value_counts(normalize=True)
        base["categorical"][col] = {"proportions": props, "missing": ref[col].isna().mean()}
    return base


def _bin_props(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, s.dropna(), side="right"), minlength=len(edges) + 1)
    return counts / counts.sum()


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = Σ (actual − expected) · ln(actual / expected)."""
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warning" if value >= PSI_WARNING else "ok"


def drift_report(base: dict, ref: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """Una fila por variable: PSI, estadístico y p-valor de KS, estado y violaciones de calidad.

    One row per feature: PSI, KS statistic and p-value, status, and quality violations.
    """
    rows = []
    for col, b in base["numeric"].items():
        cur = batch[col]
        ks = stats.ks_2samp(ref[col].dropna(), cur.dropna())
        rows.append(
            {
                "feature": col,
                "psi": psi(b["proportions"], _bin_props(cur, b["edges"])),
                "ks_stat": ks.statistic,
                "ks_pvalue": ks.pvalue,
                "mean": cur.mean(),
                "missing": cur.isna().mean(),
                "violations": _violations(cur.isna().mean(), b["missing"]),
            }
        )
    for col, b in base["categorical"].items():
        cur = batch[col].value_counts(normalize=True)
        cats = b["proportions"].index.union(cur.index)
        unseen = sorted(set(cur.index) - set(b["proportions"].index))
        violations = _violations(batch[col].isna().mean(), b["missing"])
        if unseen:
            violations.append(f"unseen categories: {unseen}")
        rows.append(
            {
                "feature": col,
                "psi": psi(b["proportions"].reindex(cats, fill_value=0).values, cur.reindex(cats, fill_value=0).values),
                "ks_stat": np.nan,  # KS solo aplica a variables continuas / KS only applies to continuous features
                "ks_pvalue": np.nan,
                "mean": np.nan,
                "missing": batch[col].isna().mean(),
                "violations": violations,
            }
        )
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(status)
    return out


def _violations(missing: float, base_missing: float) -> list[str]:
    if missing > base_missing + MISSING_TOLERANCE:
        return [f"completeness: {missing:.1%} missing (baseline {base_missing:.1%})"]
    return []


def monitor(base: dict, ref: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Informe de drift para cada semana. Drift report for every week."""
    reports = [drift_report(base, ref, batch).assign(week=week) for week, batch in production.groupby("week")]
    return pd.concat(reports, ignore_index=True)
