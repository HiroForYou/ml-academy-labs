"""Gráficos del laboratorio, con la paleta de ML Academy. Lab charts, using the ML Academy palette.

Every function takes lang="es" | "en" and returns the matplotlib Figure (rendered inline in notebooks).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift import PSI_ALERT, PSI_WARNING

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
        "psi_title": "PSI semanal por variable",
        "week": "Semana",
        "warning": "aviso (0,1)",
        "alert": "alerta (0,2)",
        "density": "Densidad",
        "share": "Proporción",
        "reference": "Referencia",
        "week_n": "Semana {w}",
        "amount_title": "amount: referencia vs. semana {w}",
        "channel_title": "channel: referencia vs. semana {w}",
        "delay_psi": "PSI máximo del lote (se conoce al instante)",
        "delay_auc": "AUC del modelo",
        "auc_true": "AUC real del lote (aún desconocida)",
        "auc_known": "AUC conocida (llegan las etiquetas, +{d} semanas)",
        "concept": "cambia el patrón de fraude",
        "detected": "primera AUC baja conocida",
    },
    "en": {
        "psi_title": "Weekly PSI per feature",
        "week": "Week",
        "warning": "warning (0.1)",
        "alert": "alert (0.2)",
        "density": "Density",
        "share": "Share",
        "reference": "Reference",
        "week_n": "Week {w}",
        "amount_title": "amount: reference vs. week {w}",
        "channel_title": "channel: reference vs. week {w}",
        "delay_psi": "Batch max PSI (known immediately)",
        "delay_auc": "Model AUC",
        "auc_true": "True batch AUC (not known yet)",
        "auc_known": "Known AUC (labels arrive, +{d} weeks)",
        "concept": "fraud pattern changes",
        "detected": "first low AUC known",
    },
}

FEATURE_COLORS = {"amount": ACCENT, "channel": INK, "customer_age": GRAY, "account_age_days": MUTED}


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(length=3, labelsize=9)


def _thresholds(ax, t: dict) -> None:
    ax.axhline(PSI_WARNING, color=MUTED, lw=0.9, ls=(0, (4, 2)), label=t["warning"])
    ax.axhline(PSI_ALERT, color=ACCENT, lw=0.9, ls=(0, (4, 2)), label=t["alert"])


def psi_over_time(report: pd.DataFrame, lang: str = "es"):
    t = LABELS[lang]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for feature, g in report.groupby("feature"):
        ax.plot(g["week"], g["psi"], marker="o", ms=3.5, lw=1.6, color=FEATURE_COLORS.get(feature, MUTED), label=feature)
    _thresholds(ax, t)
    ax.set_xticks(sorted(report["week"].unique()))
    ax.set_xlabel(t["week"])
    ax.set_ylabel("PSI")
    ax.set_title(t["psi_title"], loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    clean_axes(ax)
    fig.tight_layout()
    return fig


def overlays(ref: pd.DataFrame, production: pd.DataFrame, amount_week: int, channel_week: int, lang: str = "es"):
    t = LABELS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    cur = production.loc[production["week"] == amount_week, "amount"]
    bins = np.logspace(np.log10(1), np.log10(max(ref["amount"].max(), cur.max())), 45)
    axes[0].hist(ref["amount"], bins=bins, density=True, color=GRAY, label=t["reference"], zorder=2)
    axes[0].hist(
        cur, bins=bins, density=True, histtype="step", lw=1.6, color=ACCENT, label=t["week_n"].format(w=amount_week), zorder=3
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("amount (log)")
    axes[0].set_ylabel(t["density"])
    axes[0].set_title(t["amount_title"].format(w=amount_week), loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5)

    ref_p = ref["channel"].value_counts(normalize=True)
    cur_p = production.loc[production["week"] == channel_week, "channel"].value_counts(normalize=True)
    cats = ref_p.index
    x = np.arange(len(cats))
    axes[1].bar(x - 0.2, ref_p[cats], width=0.4, color=GRAY, label=t["reference"], zorder=3)
    axes[1].bar(
        x + 0.2, cur_p.reindex(cats, fill_value=0), width=0.4, color=ACCENT, label=t["week_n"].format(w=channel_week), zorder=3
    )
    axes[1].set_xticks(x, cats)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_ylabel(t["share"])
    axes[1].set_title(t["channel_title"].format(w=channel_week), loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5)

    for ax in axes:
        clean_axes(ax)
    fig.tight_layout()
    return fig


def label_delay(report: pd.DataFrame, quality: pd.DataFrame, concept_week: int, delay: int, lang: str = "es"):
    t = LABELS[lang]
    weeks = sorted(report["week"].unique())
    last = max(weeks)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})

    max_psi = report.groupby("week")["psi"].max()
    axes[0].bar(
        max_psi.index,
        max_psi.values,
        color=[ACCENT if v >= PSI_ALERT else MUTED if v >= PSI_WARNING else GRAY for v in max_psi],
        zorder=3,
    )
    _thresholds(axes[0], t)
    axes[0].set_ylabel("PSI")
    axes[0].set_title(t["delay_psi"], loc="left", fontsize=10)

    known = quality[quality["known_at_week"] <= last]
    axes[1].plot(quality["week"], quality["auc"], color=GRAY, lw=1.4, ls=(0, (4, 2)), marker="o", ms=3, label=t["auc_true"])
    axes[1].plot(
        known["known_at_week"], known["auc"], color=INK, lw=1.8, marker="o", ms=3.5, label=t["auc_known"].format(d=delay)
    )
    first_low = known.loc[known["week"] >= concept_week, "known_at_week"].min()
    for ax in axes:
        ax.axvline(concept_week, color=ACCENT, lw=1, alpha=0.7)
    axes[1].axvline(first_low, color=INK, lw=1, alpha=0.5)
    axes[1].text(concept_week - 0.1, 1.0, t["concept"], color=ACCENT, fontsize=8.5, ha="right", va="top")
    axes[1].text(first_low - 0.1, 0.94, t["detected"], color=INK, fontsize=8.5, ha="right", va="top")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_ylabel("AUC")
    axes[1].set_xlabel(t["week"])
    axes[1].set_xticks(weeks)
    axes[1].set_title(t["delay_auc"], loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    for ax in axes:
        clean_axes(ax)
    fig.tight_layout()
    return fig
