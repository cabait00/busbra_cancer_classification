"""Shared drawing of the study comparison figure (validation ROC-AUC per group with every seed, and the paired differences per seed) of the 4A, 4B and 4E comparisons.

Only stored table values are drawn. Importing this module draws nothing and selects the non-interactive Agg backend (through `style`).
"""
from __future__ import annotations

import sys

import numpy as np
from style import INK, INK_2, SEED_STYLE, SURFACE, style  # selects the Agg backend

import matplotlib.pyplot as plt  # noqa: E402


def auc_values(paired, pairs) -> tuple:
    """(rows of the ROC-AUC differences, {(group, seed): validation ROC-AUC}) from the stored paired table; value_a / value_b are the two groups of each pair. Stops if a pair does not hold the five seeds."""
    auc_pairs = paired[paired["metric"] == "roc_auc"]
    if sorted(auc_pairs["seed"].unique()) != sorted(SEED_STYLE) or any(len(auc_pairs[auc_pairs["pair"] == p]) != 5 for p, _ in pairs):
        sys.exit("The stored tables do not hold five seeds for each of the two paired comparisons; no figure drawn.")
    values = {}
    for pair, _ in pairs:
        for row in auc_pairs[auc_pairs["pair"] == pair].itertuples():
            values[(row.model_a, int(row.seed))] = row.value_a
            values[(row.model_b, int(row.seed))] = row.value_b
    return auc_pairs, values


def draw_auc_and_pairs(summary, paired, paired_summary, groups, pairs, *, title_left: str, ylabel_pairs: str, suptitle: str, height: float = 6.4, headroom: float = 0.11, tight_top: float = 0.95):
    """Left: validation ROC-AUC of the groups [(pipeline, label)], every single run with its own colour and marker, mean and sample SD (ddof=1). Right: the paired ROC-AUC differences of pairs [(pair name, label)]."""
    auc_pairs, values = auc_values(paired, pairs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, height), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
    style(ax1)
    auc = summary[summary["metric"] == "roc_auc"].set_index("pipeline")
    for k, (name, _) in enumerate(groups):
        for offset, seed in zip(np.linspace(-0.25, 0.25, 5), sorted(SEED_STYLE)):
            colour, marker = SEED_STYLE[seed]
            ax1.scatter(k + offset, values[(name, seed)], s=70, color=colour, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        mean, sd = auc.loc[name, "mean"], auc.loc[name, "sd_ddof1"]
        ax1.hlines(mean, k - 0.35, k + 0.35, color=INK, linewidth=2.2, zorder=3)
        ax1.vlines(k + 0.40, mean - sd, mean + sd, color=INK_2, linewidth=1.6, zorder=3)
        ax1.hlines([mean - sd, mean + sd], k + 0.37, k + 0.43, color=INK_2, linewidth=1.6, zorder=3)
        top = max(max(values[(name, s)] for s in SEED_STYLE), mean + sd)
        ax1.text(k, top + 0.008, f"mean {mean:.3f}\nSD {sd:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_xticks(range(len(groups)), [label for _, label in groups])
    ax1.set_ylabel("validation ROC-AUC (385 images, 213 cases)", color=INK_2)
    all_values = list(values.values())
    ax1.set_ylim(min(all_values) - 0.03, max(all_values) + 0.075)
    ax1.set_xlim(-0.6, len(groups) - 0.4)
    ax1.set_title(title_left, fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker=SEED_STYLE[s][1], color=SEED_STYLE[s][0], linestyle="none", markersize=8, label=f"seed {s}") for s in sorted(SEED_STYLE)]
    handles += [plt.Line2D([], [], color=INK, linewidth=2.2, label="mean of five runs"), plt.Line2D([], [], color=INK_2, linewidth=1.6, marker="_", markersize=9, label="± sample SD (ddof=1)")]
    ax1.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)

    style(ax2)
    for k, (pair, _) in enumerate(pairs):
        part = auc_pairs[auc_pairs["pair"] == pair].sort_values("seed")
        stats = paired_summary[(paired_summary["pair"] == pair) & (paired_summary["metric"] == "roc_auc")].iloc[0]
        for offset, row in zip(np.linspace(-0.3, 0.3, 5), part.itertuples()):
            colour, marker = SEED_STYLE[int(row.seed)]
            ax2.scatter(k + offset, row.difference_b_minus_a, s=80, color=colour, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=4)
            ax2.text(k + offset, row.difference_b_minus_a + (0.012 if row.difference_b_minus_a >= 0 else -0.012), f"{row.difference_b_minus_a:+.3f}", ha="center",
                     va="bottom" if row.difference_b_minus_a >= 0 else "top", fontsize=8, color=INK, rotation=90, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8}, zorder=5)
        ax2.hlines(stats["mean_difference"], k - 0.4, k + 0.4, color=INK_2, linestyle="--", linewidth=1.6, zorder=3)
        ax2.text(k, 0.98, f"mean {stats['mean_difference']:+.4f}\nSD of the differences {stats['sd_ddof1']:.4f}\npositive/negative/equal: {int(stats['n_positive'])}/{int(stats['n_negative'])}/{int(stats['n_tie'])}",
                 ha="center", va="top", fontsize=9, color=INK, transform=ax2.get_xaxis_transform(), bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2.0})
    ax2.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax2.set_xticks(range(len(pairs)), [label for _, label in pairs])
    lowest, highest = auc_pairs["difference_b_minus_a"].min(), auc_pairs["difference_b_minus_a"].max()
    ax2.set_ylim(min(lowest, 0.0) - 0.06, max(highest, 0.0) + headroom)  # headroom above for the printed mean and SD
    ax2.set_xlim(-0.65, len(pairs) - 0.35)
    ax2.set_ylabel(ylabel_pairs, color=INK_2)
    ax2.set_title("Paired differences per seed (descriptive)", fontsize=10, color=INK, loc="left")
    ax2.legend(handles=[plt.Line2D([], [], color=INK_2, linestyle="--", linewidth=1.6, label="mean of the five paired differences"), plt.Line2D([], [], color=INK, linewidth=1.2, label="zero: no difference")],
               frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.suptitle(suptitle, fontsize=9.5, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, tight_top))
    return fig
