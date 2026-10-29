"""Gráficos del laboratorio, con la paleta de ML Academy. Lab charts, using the ML Academy palette.

Every function takes lang="es" | "en" and returns the matplotlib Figure (rendered inline in notebooks).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        "recall": "Recall@k",
        "precision": "Precision@k",
        "leak": "Preguntas con un chunk de otro tenant",
        "filter": "con filtro de tenant",
        "no_filter": "sin filtro",
        "metrics_title": "Recuperación por estrategia de chunking",
        "outcomes_title": "Resultado de extremo a extremo (top-3)",
        "questions": "Preguntas",
        "correct": "Correcta",
        "wrong_tenant": "Dato de otro tenant",
        "retrieval_miss": "Fallo de recuperación",
        "generation_miss": "Fallo de generación",
    },
    "en": {
        "recall": "Recall@k",
        "precision": "Precision@k",
        "leak": "Questions with another tenant's chunk",
        "filter": "with tenant filter",
        "no_filter": "no filter",
        "metrics_title": "Retrieval by chunking strategy",
        "outcomes_title": "End-to-end outcome (top-3)",
        "questions": "Questions",
        "correct": "Correct",
        "wrong_tenant": "Other tenant's data",
        "retrieval_miss": "Retrieval miss",
        "generation_miss": "Generation miss",
    },
}

STRATEGY_COLORS = {"fixed-30": INK_SOFT, "paragraph": ACCENT}
OUTCOME_COLORS = {"correct": OK, "wrong_tenant": ACCENT, "retrieval_miss": MUTED, "generation_miss": GRAY}


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(length=3, labelsize=9)


def metrics_vs_k(summaries: dict[tuple[str, bool], pd.DataFrame], lang: str = "es"):
    """summaries: {(strategy, use_filter): evaluation.summary(...)}."""
    t = LABELS[lang]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True)
    for (strategy, use_filter), s in summaries.items():
        style = dict(
            color=STRATEGY_COLORS.get(strategy, MUTED),
            ls="-" if use_filter else (0, (4, 2)),
            marker="o",
            ms=3.5,
            lw=1.6,
            label=f"{strategy} · {t['filter'] if use_filter else t['no_filter']}",
        )
        for ax, col in zip(axes, ["recall", "precision", "leak"], strict=True):
            ax.plot(s.index, s[col], **style)
    for ax, col in zip(axes, ["recall", "precision", "leak"], strict=True):
        ax.set_title(t[col], loc="left", fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, 6))
        ax.yaxis.set_major_formatter(lambda v, _, c=col: f"{v:.0%}" if c == "leak" else f"{v:.1f}")
        clean_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(t["metrics_title"], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def outcomes(results: dict[str, pd.DataFrame], lang: str = "es"):
    """results: {row label: evaluation.end_to_end(...)}; stacked bars of outcomes."""
    t = LABELS[lang]
    names = list(results)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.4))
    left = np.zeros(len(names))
    for outcome, color in OUTCOME_COLORS.items():
        counts = np.array([(results[n]["outcome"] == outcome).sum() for n in names])
        ax.barh(names, counts, left=left, color=color, label=t[outcome], height=0.6, zorder=3)
        for y, (c, x) in enumerate(zip(counts, left, strict=True)):
            if c:
                ax.text(x + c / 2, y, str(c), ha="center", va="center", fontsize=9, color="white" if color != GRAY else INK)
        left += counts
    ax.invert_yaxis()
    ax.set_xlabel(t["questions"])
    ax.set_title(t["outcomes_title"], loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    clean_axes(ax)
    fig.tight_layout()
    return fig
