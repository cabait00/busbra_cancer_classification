# %% Figures of the project documentation docs/BUSBRA_Projektdokumentation.md (CPU, no TensorFlow, no inference): drawn only from the stored result tables of the finished studies, no image data
#
# Reads the aggregate tables of the final test evaluation, the robustness study and the adversarial add-on, and the learning curves of the 4B and 4E runs. Writes six German
# figures to reports/figures/ (new files with the prefix bericht_; the earlier figures stay unchanged):
#   bericht_pipeline_uebersicht.png   scheme of the three input conditions (without ultrasound data)
#   bericht_test_auc.png              image ROC-AUC of the seven groups on the test: single seeds, seed mean and sample SD
#   bericht_haupt_vergleiche.png      the three main comparisons: point estimate and 95 % percentile interval (image and case level)
#   bericht_betriebspunkte.png        sensitivity reached and benign false alarms at the frozen image thresholds for the targets 0.90 and 0.95
#   bericht_lernkurven.png            loss curves of the fine-tuning (full image 4B and expert ROI 4E), five seeds, validation loss with best epoch
#   bericht_fgsm.png                  FGSM stress test on the test subset (full image and automatic ROI) and the validation add-on
# The figures describe fixed models, fixed thresholds and a fixed sample; they are no ranking, no significance statement and no clinical statement.
#
# Usage (CPU only, from the project root BUSBRA):  .venv/bin/python plots/plot_final_report_figures.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "plots")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from busbra import config, results  # noqa: E402
from style import COLOR, GROUPS  # noqa: E402  (Okabe-Ito colours: full image blue, expert ROI green, automatic ROI orange)

TABLES, FIGURES = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
SEEDS = tuple(config.SEEDS)
LABEL = {"dummy": "Dummy\n(Klassenprior)", "logreg_l2": "Logistische\nRegression\n64 px", "small_cnn": "Kleines CNN\n(3F)", "full_frozen": "Vollbild\nfrozen", "full_finetuned": "Vollbild\nfeinjustiert",
         "expert_roi_finetuned": "Experten-ROI\n(manuelle Maske)", "auto_roi_finetuned": "Automatische\nROI"}
COMPARISONS = [("expert_minus_full", "Experten-ROI minus Vollbild", "#009E73"), ("auto_minus_expert", "Automatische ROI minus Experten-ROI", "#D55E00"), ("auto_minus_full", "Automatische ROI minus Vollbild", "#0072B2")]
SHORT_LABEL = {"logreg_l2": "Logistische\nRegression", "small_cnn": "Kleines\nCNN (3F)", "full_frozen": "Vollbild\nfrozen", "full_finetuned": "Vollbild\nfeinjustiert", "expert_roi_finetuned": "Experten-\nROI", "auto_roi_finetuned": "Automat.\nROI"}
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def de(x: float, nd: int = 3, sign: bool = False) -> str:
    text = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return text.replace("-", "−").replace(".", ",")


SOURCES = ["test_evaluation_metrics", "test_evaluation_operating_points", "test_evaluation_bootstrap_summary", "robustness_test_metrics", "robustness_test_summary_metrics", "adversarial_roi_summary"]
HISTORIES = {"Vollbild (4B, Feinjustierung 1e-4)": ("augmentation160", "mobilenetv2_higher_lr_160_aug", "#0072B2"),
             "Experten-ROI (4E, Feinjustierung 1e-4)": ("expert_roi", "mobilenetv2_higher_lr_160_roi", "#009E73")}


def load_tables() -> dict:
    return {name: results.table(name) for name in SOURCES}


def load_histories() -> dict:
    """The loss histories of the ten fine-tuning runs (4B full image and 4E expert ROI, five seeds each)."""
    return {(label, seed): pd.read_csv(TABLES / folder / f"seed_{seed}" / pipeline / "history.csv") for label, (folder, pipeline, _) in HISTORIES.items() for seed in SEEDS}


def save(fig, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# %% Abbildung 1: Schema der drei Eingabebedingungen
def figure_pipelines() -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    columns = [(1, 20, "Eingabe"), (24, 22, "Lokalisation"), (49, 22, "Zuschnitt und Aufbereitung"), (74, 15, "Klassifikator"), (91, 8.5, "Ausgabe")]
    for x, w, title in columns:
        ax.text(x + w / 2, 64, title, ha="center", va="center", fontsize=10, color=MUTED, fontweight="bold")
    lanes = [
        ("#0072B2", "Vollbild (4B)", 46, [("Ultraschallbild", None), ("keine", "kein Zuschnitt"), ("ganzes Bild auf 160 × 160\n(Seitenverhältnis erhalten,\nRand schwarz)", None), ("MobileNetV2,\nfeinjustiert", None)]),
        ("#009E73", "Experten-ROI (4E)", 26, [("Ultraschallbild\n+ manuelle Läsionsmaske", None), ("Box um die Maske\n(braucht Expertenarbeit)", None), ("Box + 20 % je Seite,\nunmaskiertes Original,\nauf 160 × 160", None), ("MobileNetV2,\nauf Zuschnitten\nfeinjustiert", None)]),
        ("#D55E00", "Automatische ROI (4F-B)", 6, [("Ultraschallbild", None), ("U-Net (256 × 256),\nSchwelle 0,5, größte\nKomponente", None), ("Box + 20 % je Seite;\nleere Maske: Vollbild;\nauf 160 × 160", None), ("derselbe 4E-\nKlassifikator\n(nicht neu trainiert)", None)]),
    ]
    for color, title, y, cells in lanes:
        ax.text(0.2, y + 14.4, title, ha="left", va="center", fontsize=11, color=color, fontweight="bold")
        for (x, w, _), (text, sub) in zip(columns[:4], cells):
            ax.add_patch(FancyBboxPatch((x, y), w, 11, boxstyle="round,pad=0.25,rounding_size=1.2", linewidth=1.4, edgecolor=color, facecolor="white"))
            ax.text(x + w / 2, y + 5.5, text, ha="center", va="center", fontsize=8.6)
        x, w, _ = columns[4]
        ax.add_patch(FancyBboxPatch((x, y), w, 11, boxstyle="round,pad=0.25,rounding_size=1.2", linewidth=1.4, edgecolor=color, facecolor="#f4f4f4"))
        ax.text(x + w / 2, y + 5.5, "Score\n„bösartig“", ha="center", va="center", fontsize=8.6)
        for (x0, w0, _), (x1, _, _) in zip(columns[:4], columns[1:5]):
            ax.annotate("", xy=(x1 - 0.3, y + 5.5), xytext=(x0 + w0 + 0.3, y + 5.5), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.2})
    ax.text(50, 0.4, "Schema der drei Eingabebedingungen; Klassifikationsziel in allen Zweigen: Pathology (gutartig = 0, bösartig = 1). Keine Ultraschalldaten dargestellt.", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return save(fig, "bericht_pipeline_uebersicht.png")


# %% Abbildung 2: Bild-ROC-AUC der sieben Gruppen
def figure_auc(metrics: pd.DataFrame) -> Path:
    image = metrics[metrics["level"] == "image"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rng = np.random.default_rng(0)  # nur für die waagerechte Streuung der Punkte, ohne Bedeutung für Werte
    for i, group in enumerate(GROUPS):
        values = image[image["group_id"] == group]["roc_auc"].to_numpy(float)
        ax.scatter(i + rng.uniform(-0.18, 0.18, values.size), values, s=34, color=COLOR[group], alpha=0.85, zorder=3, edgecolor="white", linewidth=0.5)
        mean = values.mean()
        sd = values.std(ddof=1) if values.size > 1 else 0.0
        ax.errorbar(i + 0.33, mean, yerr=sd if values.size > 1 else None, fmt="D", color="black", capsize=4, markersize=6, zorder=4)
        ax.text(i + 0.33, mean + (sd if values.size > 1 else 0) + 0.012, de(mean), ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.text(len(GROUPS) - 0.45, 0.503, "Zufallsniveau 0,5", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(GROUPS)), [LABEL[g] for g in GROUPS], fontsize=9)
    ax.set_ylabel("ROC-AUC (bösartig = positive Klasse)")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", color=GRID)
    ax.set_title("Testdaten (376 Bilder, 212 Fälle): Bild-ROC-AUC je Gruppe", loc="left", fontsize=11, fontweight="bold")
    ax.text(0, -0.30, "Punkte: einzelne Trainings-Seeds (Dummy und logistische Regression: je ein Modell); Raute: Seed-Mittel ± Stichproben-SD. Keine Pooling-AUC, kein Ensemble.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    return save(fig, "bericht_test_auc.png")


# %% Abbildung 3: die drei Hauptvergleiche
def figure_comparisons(boot: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    rows, y = [], 0.0
    for level, metric in (("Bildebene (376 Bilder)", "auc_image"), ("Case-Ebene (212 Fälle)", "auc_case")):
        for key, label, color in COMPARISONS:
            r = boot[(boot["comparison"] == key) & (boot["metric"] == metric)].iloc[0]
            rows.append((y, level, label, color, r))
            y -= 1
        y -= 0.7
    for yy, level, label, color, r in rows:
        ax.errorbar(r["point_estimate_seed_mean_difference"], yy, xerr=[[r["point_estimate_seed_mean_difference"] - r["ci95_low"]], [r["ci95_high"] - r["point_estimate_seed_mean_difference"]]], fmt="o", color=color, capsize=4, markersize=7, linewidth=1.8)
        ax.text(0.092, yy, f"{de(r['point_estimate_seed_mean_difference'], 3, True)}  [{de(r['ci95_low'], 3, True)}; {de(r['ci95_high'], 3, True)}]", va="center", ha="left", fontsize=9)
    ax.set_yticks([r[0] for r in rows], [r[2] for r in rows], fontsize=9)
    for yy, level, *_ in rows[::3]:
        ax.text(-0.0515, yy + 0.75, level, fontsize=9.5, fontweight="bold", va="bottom", ha="left")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_xlim(-0.055, 0.145)
    ax.set_xticks(np.arange(-0.04, 0.0801, 0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 2, True).replace("+0,00", "0,00")))
    ax.set_ylim(rows[-1][0] - 0.8, rows[0][0] + 1.5)
    ax.set_xlabel("Differenz der ROC-AUC (zweite Bedingung minus erste), Seed-Mittel der gepaarten Differenzen")
    ax.grid(axis="x", color=GRID)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Drei Hauptvergleiche mit 95-%-Perzentilintervall (Cluster-Bootstrap über Fälle, 2.000 gepaarte Ziehungen)", loc="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.04, "Das Intervall beschreibt die Stichprobenunsicherheit der Testfälle bei festen Modellen; es umfasst weder neues Training noch eine externe Klinik.", fontsize=8.5, color=MUTED)
    return save(fig, "bericht_haupt_vergleiche.png")


# %% Abbildung 4: Betriebspunkte
def figure_operating_points(points: pd.DataFrame) -> Path:
    groups = [g for g in GROUPS if g != "dummy"]
    image = points[points["level"] == "image"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    rng = np.random.default_rng(0)
    for col, (point, target) in enumerate((("target_0.90", 0.90), ("target_0.95", 0.95))):
        for row, (column, ylabel) in enumerate((("sensitivity", "erreichte Sensitivität auf dem Test"), ("fp", "benigne Fehlalarme (von 254 gutartigen Bildern)"))):
            ax = axes[row, col]
            for i, group in enumerate(groups):
                values = image[(image["group_id"] == group) & (image["point"] == point)][column].to_numpy(float)
                ax.scatter(i + rng.uniform(-0.16, 0.16, values.size), values, s=30, color=COLOR[group], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
                if values.size > 1:
                    ax.errorbar(i + 0.32, values.mean(), yerr=values.std(ddof=1), fmt="D", color="black", capsize=3.5, markersize=5.5, zorder=4)
                else:
                    ax.scatter([i + 0.32], values, marker="D", color="black", s=30, zorder=4)
            if row == 0:
                ax.axhline(target, color="#777777", linestyle="--", linewidth=1)
                ax.text(-0.55, target, f"Validation-Ziel {de(target, 2)}", ha="left", va="bottom", fontsize=8.5, color=MUTED)
                ax.set_title(f"Schwelle für das Ziel {de(target, 2)} (auf Validation gewählt)", fontsize=10.5, loc="left", fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlim(-0.6, len(groups) - 0.2)
            ax.grid(axis="y", color=GRID)
    for ax in axes[1]:
        ax.set_xticks(range(len(groups)), [SHORT_LABEL[g] for g in groups], fontsize=8.8)
    fig.suptitle("Eingefrorene Bildschwellen auf den Testdaten (376 Bilder)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.005, "Punkte: Seeds (logistische Regression: ein Modell); Raute: Seed-Mittel ± SD. Die Ziele bezeichnen die Auswahl auf Validation; die auf dem Test erreichte Sensitivität weicht ab und ist oben ausgewiesen.\n"
             "Fehlalarme an einem festen Punkt sind kein Vergleich bei garantiert gleicher Sensitivität. Dummy (alle Bilder positiv: Sensitivität 1, 254 Fehlalarme) ist nicht dargestellt.", fontsize=8.3, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return save(fig, "bericht_betriebspunkte.png")


# %% Abbildung 5: Lernkurven
def figure_learning_curves(histories: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, (label, (_, _, color)) in zip(axes, HISTORIES.items()):
        for seed in SEEDS:
            h = histories[(label, seed)]
            ax.plot(h["epoch"], h["loss"], color=color, alpha=0.45, linestyle="--", linewidth=1.1)
            ax.plot(h["epoch"], h["val_loss"], color=color, linewidth=1.5)
            best = h[h["is_best_epoch"].astype(str) == "True"]
            ax.scatter(best["epoch"], best["val_loss"], color=color, edgecolor="black", s=36, zorder=4)
        ax.set_title(label, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlabel("Epoche der Feinjustierung")
        ax.grid(color=GRID)
    axes[0].set_ylabel("Binary Crossentropy")
    handles = [plt.Line2D([], [], color="#555555", linestyle="--", alpha=0.6, label="Training (Loss)"), plt.Line2D([], [], color="#555555", label="Validation (Loss, bestimmt den Checkpoint)"),
               plt.Line2D([], [], color="#555555", marker="o", markeredgecolor="black", linestyle="", label="beste Epoche")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8.6, frameon=False)
    fig.suptitle("Lernkurven der Feinjustierung, fünf Seeds je Panel (ohne die vorangehende Phase mit eingefrorenem Backbone)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "bericht_lernkurven.png")


# %% Abbildung 6: FGSM auf dem Test und Zusatzexperiment 5B-R auf der Validation
def figure_fgsm(robust: pd.DataFrame, summary: pd.DataFrame, adv: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    epsilons = [0, 1, 2, 4]
    ax = axes[0]
    for condition, label, color, marker in (("full", "Vollbild (4B, feinjustiert)", "#0072B2", "o"), ("auto", "Automatische ROI (4E-Klassifikator)", "#D55E00", "s")):
        part = robust[(robust["condition"] == condition) & (robust["level"] == "image")]
        for seed in SEEDS:
            s = part[part["seed"] == seed].sort_values("epsilon_255")
            ax.plot(s["epsilon_255"], s["roc_auc"], color=color, alpha=0.3, linewidth=1)
        m = summary[(summary["condition"] == condition) & (summary["level"] == "image") & (summary["metric"] == "roc_auc")].sort_values("epsilon_255")
        ax.errorbar(m["epsilon_255"], m["mean"], yerr=m["sd_ddof1"], marker=marker, color=color, linewidth=2.2, capsize=3.5, label=label)
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.text(4.05, 0.51, "Zufallsniveau", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xticks(epsilons, ["0", "1/255", "2/255", "4/255"])
    ax.set_xlabel("FGSM-Stärke ε (Eingabeskala des Klassifikators, [0, 1])")
    ax.set_ylabel("Bild-ROC-AUC")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(color=GRID)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)
    ax.set_title("Test: einstufiger digitaler Angriff", fontsize=10.5, loc="left", fontweight="bold")
    ax = axes[1]
    styles = (("original", "Ausgangsmodell (4E)", "#999999", "o"), ("clean_continuation", "Fortsetzung mit sauberen Bildern", "#56B4E9", "s"), ("fgsm_continuation", "Fortsetzung mit FGSM-Beispielen", "#CC79A7", "^"))
    for state, label, color, marker in styles:
        m = adv[(adv["state"] == state) & (adv["condition"] == "expert") & (adv["level"] == "image") & (adv["metric"] == "roc_auc")].sort_values("epsilon_255")
        ax.errorbar(m["epsilon_255"], m["mean"], yerr=m["sd_ddof1"], marker=marker, color=color, linewidth=2, capsize=3.5, label=label)
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(epsilons, ["0", "1/255", "2/255", "4/255"])
    ax.set_xlabel("FGSM-Stärke ε (Eingabeskala des Klassifikators, [0, 1])")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(color=GRID)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)
    ax.set_title("Validation (Zusatzexperiment 5B-R): Experten-ROI-Klassifikatoren", fontsize=10.5, loc="left", fontweight="bold")
    fig.suptitle("Seed-Mittel ± Stichproben-SD (fünf Seeds); links dünne Linien: einzelne Seeds. Untersucht wird nur der Klassifikator, nicht das U-Net.", x=0.01, ha="left", fontsize=9.5, color=MUTED, y=-0.01)
    fig.tight_layout()
    return save(fig, "bericht_fgsm.png")


# %% Main
FIGURES.mkdir(parents=True, exist_ok=True)
tables = load_tables()
histories = load_histories()
made = [figure_pipelines(), figure_auc(tables["test_evaluation_metrics"]), figure_comparisons(tables["test_evaluation_bootstrap_summary"]), figure_operating_points(tables["test_evaluation_operating_points"]),
        figure_learning_curves(histories), figure_fgsm(tables["robustness_test_metrics"], tables["robustness_test_summary_metrics"], tables["adversarial_roi_summary"])]
print("written:", *[p.name for p in made], sep="\n  ")
sys.exit(0)
