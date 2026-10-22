"""Gráficos del laboratorio, con la paleta de ML Academy. Lab charts, using the ML Academy palette.

Every function takes lang="es" | "en" and returns the matplotlib Figure (rendered inline in notebooks).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

INK = "#111114"
INK_SOFT = "#3d3d45"
MUTED = "#6a6a74"
SURFACE = "#f3f3f5"
ACCENT = "#b3132b"
OK = "#1f7a4d"
GRAY = "#c9c9cf"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "font.size": 10,
    }
)

LABELS = {
    "es": {
        "no_default": "Sin impago (0)",
        "default": "Impago (1)",
        "rows": "Filas",
        "class_title": "Distribución de clases",
        "trap": "Predecir siempre «sin impago»\nya da {acc:.0%} de accuracy",
        "missing_title": "Income faltante por clase",
        "missing_rate": "% de filas sin income",
        "income": "income",
        "before": "Antes (crudo)",
        "after": "Después (imputado y recortado)",
        "iqr": "Límites IQR (entrenamiento)",
        "income_title": "Outliers de income",
        "facet_a": "a: ≤ 60 años",
        "facet_d": "d: > 60 años",
        "facet_rows": "Filas por faceta",
        "facet_rate": "Tasa de impago por faceta",
        "rate": "Proporción de etiquetas positivas",
        "metrics_title": "Métricas en test (sin remuestrear)",
        "score": "Valor",
        "cm_title": "Matriz de confusión: {name}",
        "predicted": "Predicho",
        "actual": "Real",
    },
    "en": {
        "no_default": "No default (0)",
        "default": "Default (1)",
        "rows": "Rows",
        "class_title": "Class distribution",
        "trap": "Always predicting “no default”\nalready gives {acc:.0%} accuracy",
        "missing_title": "Missing income by class",
        "missing_rate": "% of rows without income",
        "income": "income",
        "before": "Before (raw)",
        "after": "After (imputed and clipped)",
        "iqr": "IQR bounds (training)",
        "income_title": "Income outliers",
        "facet_a": "a: ≤ 60 years",
        "facet_d": "d: > 60 years",
        "facet_rows": "Rows per facet",
        "facet_rate": "Default rate per facet",
        "rate": "Share of positive labels",
        "metrics_title": "Test metrics (not resampled)",
        "score": "Value",
        "cm_title": "Confusion matrix: {name}",
        "predicted": "Predicted",
        "actual": "Actual",
    },
}


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(length=3, labelsize=9)


def _bar_labels(ax, bars, fmt: str) -> None:
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )


def class_distribution(df: pd.DataFrame, lang: str = "es"):
    t = LABELS[lang]
    counts = df["default"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bars = ax.bar([t["no_default"], t["default"]], counts.values, color=[GRAY, ACCENT], width=0.55, zorder=3)
    _bar_labels(ax, bars, "{:,.0f}")
    ax.annotate(
        t["trap"].format(acc=counts[0] / counts.sum()),
        xy=(0, counts[0] * 0.55),
        xytext=(0.62, counts[0] * 0.62),
        fontsize=9,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ax.set_ylabel(t["rows"])
    ax.set_title(t["class_title"], loc="left")
    clean_axes(ax)
    fig.tight_layout()
    return fig


def missing_by_class(df: pd.DataFrame, lang: str = "es"):
    t = LABELS[lang]
    rate = df.groupby("default")["income"].apply(lambda s: s.isna().mean())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar([t["no_default"], t["default"]], rate.values, color=[GRAY, ACCENT], width=0.55, zorder=3)
    _bar_labels(ax, bars, "{:.1%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel(t["missing_rate"])
    ax.set_title(t["missing_title"], loc="left")
    clean_axes(ax)
    fig.tight_layout()
    return fig


def income_before_after(raw: pd.DataFrame, cleaned: pd.DataFrame, stats: dict, lang: str = "es"):
    t = LABELS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, data, title, color in [
        (axes[0], raw["income"].dropna(), t["before"], GRAY),
        (axes[1], cleaned["income"], t["after"], OK),
    ]:
        ax.hist(data, bins=60, color=color, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel(t["income"])
        ax.axvline(stats["high"], color=ACCENT, lw=1.2, ls=(0, (4, 2)), label=t["iqr"])
        ax.set_yscale("log")
        clean_axes(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(t["income_title"], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def facets(df: pd.DataFrame, facet_d: pd.Series, report: dict, lang: str = "es"):
    t = LABELS[lang]
    names = [t["facet_a"], t["facet_d"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    bars = axes[0].bar(names, [report["n_a"], report["n_d"]], color=[GRAY, ACCENT], width=0.55, zorder=3)
    _bar_labels(axes[0], bars, "{:,.0f}")
    axes[0].set_title(f"{t['facet_rows']} · CI = {report['CI']:+.2f}", loc="left")
    axes[0].set_ylabel(t["rows"])

    bars = axes[1].bar(names, [report["q_a"], report["q_d"]], color=[GRAY, ACCENT], width=0.55, zorder=3)
    _bar_labels(axes[1], bars, "{:.1%}")
    axes[1].set_title(f"{t['facet_rate']} · DPL = {report['DPL']:+.3f}", loc="left")
    axes[1].set_ylabel(t["rate"])
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    for ax in axes:
        clean_axes(ax)
    fig.tight_layout()
    return fig


def metrics_comparison(metrics: pd.DataFrame, lang: str = "es"):
    t = LABELS[lang]
    cols = ["accuracy", "balanced_acc", "recall", "f1", "roc_auc"]
    colors = [GRAY, MUTED, INK_SOFT, OK, ACCENT]
    x = np.arange(len(cols))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (name, row) in enumerate(metrics[cols].iterrows()):
        ax.bar(x + i * width - 0.4 + width / 2, row.values, width=width, color=colors[i % len(colors)], label=name, zorder=3)
    ax.set_xticks(x, cols)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(t["score"])
    ax.set_title(t["metrics_title"], loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axes(ax)
    fig.tight_layout()
    return fig


def confusion_matrices(y_true, predictions: dict, names: list[str], lang: str = "es"):
    t = LABELS[lang]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        cm = confusion_matrix(y_true, predictions[name], labels=[0, 1])
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max() * 1.6)
        for (i, j), v in np.ndenumerate(cm):
            dark_cell = v > cm.max() * 0.5
            ax.text(
                j, i, f"{v:,}", ha="center", va="center", fontsize=11, color="white" if dark_cell else ACCENT if i == 1 else INK
            )
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel(t["predicted"])
        ax.set_ylabel(t["actual"])
        ax.set_title(t["cm_title"].format(name=name), loc="left", fontsize=10)
    fig.tight_layout()
    return fig
