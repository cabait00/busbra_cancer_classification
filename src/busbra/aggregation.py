"""Aggregation of the per-run results of a classification study (pure pandas / numpy, no TensorFlow).

Statistics are over training runs (seeds) on the same holdout, not over independent patient cohorts. Sample standard deviation
uses ddof=1; undefined values are never replaced by zero. Paired differences are model B minus model A within the same seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from busbra import config

METRIC_COLUMNS = ["roc_auc", "average_precision", "sensitivity", "specificity", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp"]
TIE_TOLERANCE = 1e-12


def read_run_metrics(variant: config.Variant, seed: int) -> pd.DataFrame:
    """Metrics of one run in the common format. The early single runs (steps 3A-3E) stored no seed column; their documented seed is config.HISTORICAL_SEED."""
    table = pd.read_csv(config.run_paths(variant, seed)["metrics"])
    if "seed" not in table.columns:
        if variant.seed_mode != "historical" or seed != config.HISTORICAL_SEED:
            raise ValueError(f"{variant.name}: the stored metrics have no seed column and only the historical seed {config.HISTORICAL_SEED} is documented for them, not {seed}")
        table.insert(1, "seed", seed)
    return table


def collect_runs(names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Metrics of all runs (train and validation rows) in one table with the columns seed and pipeline."""
    frames = []
    for seed in seeds:  # the order of the training jobs: seeds outer, variants inner
        for name in names:
            variant = config.VARIANTS[name]
            table = read_run_metrics(variant, seed)
            table.insert(2, "pipeline", name)
            frames.append(table.drop(columns=["model"]))
    return pd.concat(frames, ignore_index=True)


def validate_run_table(runs: pd.DataFrame, seeds, pipelines, subset: str = "validation") -> list:
    """Problems that must prevent a complete-success report: missing or duplicated seed/pipeline combinations, unexpected seeds or
    pipelines, non-finite ROC-AUC/AP, differing image counts or thresholds."""
    problems = []
    part = runs[runs["subset"] == subset]
    for name in pipelines:
        rows = part[part["pipeline"] == name]
        found = sorted(rows["seed"].astype(int).tolist())
        if len(found) != len(set(found)):
            problems.append(f"{name}: duplicated seed combination {sorted({s for s in found if found.count(s) > 1})}")
        missing = [s for s in seeds if s not in found]
        if missing:
            problems.append(f"{name}: missing seeds {missing}")
        extra = [s for s in set(found) if s not in seeds]
        if extra:
            problems.append(f"{name}: unexpected seeds {sorted(extra)}")
        for metric in ("roc_auc", "average_precision"):
            values = pd.to_numeric(rows[metric], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                problems.append(f"{name}: non-finite {metric}")
    unexpected = sorted(set(part["pipeline"]) - set(pipelines))
    if unexpected:
        problems.append(f"unexpected pipelines {unexpected}")
    for column in ("n_images", "n_cases", "threshold"):
        if column in part and part[column].nunique() > 1:
            problems.append(f"{column} differs between runs: {sorted(part[column].unique().tolist())}")
    return problems


def summarize(runs: pd.DataFrame, pipelines, metrics=METRIC_COLUMNS, subset: str = "validation") -> pd.DataFrame:
    """Per pipeline and metric: number of defined values, mean, sample SD (ddof=1), minimum, maximum over the training runs.

    These are statistics over training runs on the same holdout, not over independent patient cohorts. Undefined values (for example
    a precision without any predicted malignant image) are not counted and never replaced by zero; n_defined shows how many exist."""
    rows = []
    for name in pipelines:
        part = runs[(runs["pipeline"] == name) & (runs["subset"] == subset)]
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(dtype=float)
            rows.append({"pipeline": name, "metric": metric, "n_runs": int(len(part)), "n_defined": int(len(values)),
                         "mean": float(values.mean()) if len(values) else None,
                         "sd_ddof1": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None,
                         "undefined_note": "" if len(values) == len(part) else "some runs have an undefined value (zero denominator); not replaced by zero"})
    return pd.DataFrame(rows)


def paired_differences(runs: pd.DataFrame, pipelines, metrics=("roc_auc", "average_precision"), subset: str = "validation") -> pd.DataFrame:
    """Model B minus model A within the same seed for every pair (A before B in the pipeline order): ten pairs, five seeds each."""
    part = runs[runs["subset"] == subset]
    rows = []
    for i, name_a in enumerate(pipelines):
        for name_b in pipelines[i + 1:]:
            for metric in metrics:
                a = part[part["pipeline"] == name_a].set_index("seed")[metric].astype(float)
                b = part[part["pipeline"] == name_b].set_index("seed")[metric].astype(float)
                if sorted(a.index) != sorted(b.index) or a.index.duplicated().any() or b.index.duplicated().any():
                    raise ValueError(f"pair {name_a} / {name_b}: the seeds do not match one to one")
                for seed in a.index:
                    rows.append({"pair": f"{name_b} minus {name_a}", "model_a": name_a, "model_b": name_b, "metric": metric, "seed": int(seed),
                                 "value_a": float(a[seed]), "value_b": float(b[seed]), "difference_b_minus_a": float(b[seed] - a[seed])})
    return pd.DataFrame(rows)


def summarize_pairs(differences: pd.DataFrame, tolerance: float = TIE_TOLERANCE) -> pd.DataFrame:
    """Mean and sample SD of the paired differences themselves (not the difference of the two model SDs), their signs and the five values."""
    rows = []
    for (pair, metric), part in differences.groupby(["pair", "metric"], sort=False):
        d = part.sort_values("seed")["difference_b_minus_a"].to_numpy(dtype=float)
        rows.append({"pair": pair, "model_a": part["model_a"].iloc[0], "model_b": part["model_b"].iloc[0], "metric": metric, "n_seeds": int(len(d)),
                     "mean_difference": float(d.mean()), "sd_ddof1": float(d.std(ddof=1)) if len(d) > 1 else None, "min": float(d.min()), "max": float(d.max()),
                     "n_positive": int((d > tolerance).sum()), "n_negative": int((d < -tolerance).sum()), "n_tie": int((np.abs(d) <= tolerance).sum()),
                     "seeds": ";".join(str(int(s)) for s in sorted(part["seed"])), "differences_by_seed": ";".join(f"{v:.10f}" for v in d)})
    return pd.DataFrame(rows)


def paired_between(table_a: pd.DataFrame, table_b: pd.DataFrame, pairs: list[tuple[str, str, str]], seeds: list[int], metrics, subset: str = "validation") -> pd.DataFrame:
    """Differences between pipelines of two run tables within each seed (replaces the separate comparisons of the 160/250/augmentation/ROI studies).

    pairs: (label, pipeline in table_a, pipeline in table_b); the difference is b minus a. Every pipeline must provide exactly the given seeds once.
    """
    a_part, b_part = table_a[table_a["subset"] == subset], table_b[table_b["subset"] == subset]

    def values(part, name, metric):
        series = part[part["pipeline"] == name].set_index("seed")[metric].astype(float)
        if series.index.duplicated().any() or sorted(series.index) != sorted(seeds):
            raise ValueError(f"{name}: seeds are not the planned seeds exactly once")
        return series

    rows = []
    for label, name_a, name_b in pairs:
        for metric in metrics:
            va, vb = values(a_part, name_a, metric), values(b_part, name_b, metric)
            for seed in seeds:
                rows.append({"pair": label, "model_a": name_a, "model_b": name_b, "metric": metric, "seed": seed, "value_a": float(va[seed]), "value_b": float(vb[seed]),
                             "difference_b_minus_a": float(vb[seed] - va[seed])})
    return pd.DataFrame(rows)


def write_new_table(table: pd.DataFrame, path: Path) -> None:
    """Write a result table; an existing file is never overwritten."""
    if Path(path).exists():
        raise FileExistsError(f"{path} already exists; nothing was changed")
    table.to_csv(path, index=False, lineterminator="\n")


# ---- study summaries with their reference groups --------------------------------------------------------------------------------------------------------
BASELINE_NOTE = "fixed reference from the accepted baseline run (not re-trained, no seed spread)"


def group_description(preset: str, name: str) -> dict:
    """The descriptive columns of one group in the summary of a study (what distinguishes the groups of that study)."""
    variant = config.VARIANTS[name]
    step = config.PRESET_STEPS[config.VARIANTS[name].namespace]
    source = f"step {step}" if variant.namespace == preset else f"step {step} (read, not re-run)"
    if preset == "resolution250":
        return {"resolution_px": variant.size, "source": source}
    if preset == "augmentation160":
        return {"augmentation": ("flip only" if variant.augmentation == "flip" else "flip + rotation + zoom") + (f" ({variant.size} px input)" if variant.size != 160 else ""), "source": source}
    if preset == "expert_roi":
        return {"input": "full image" if variant.data == "full" else "expert-mask ROI (needs a manual mask)", "source": source}
    return {}


def study_groups(preset: str) -> list[str]:
    """The groups of a study summary: its own variants and the reference variants, in the alphabetical order of the stored summaries (the multi-seed study keeps its pipeline order)."""
    names = list(config.PRESETS[preset]) + [n for _, group in config.PRESET_REFERENCES[preset] for n in group]
    context = lambda n: preset == "augmentation160" and config.VARIANTS[n].namespace == "resolution250"  # noqa: E731  the 250 px runs of the earlier study stand last
    return names if preset == "multiseed" else sorted(names, key=lambda n: (context(n), n))


def collect_study_runs(preset: str, seeds: list[int]) -> pd.DataFrame:
    """Runs of the study and of its reference groups in one table (the reference runs are the stored ones of the earlier studies)."""
    return collect_runs(study_groups(preset), seeds)


def baseline_rows(baseline_metrics: pd.DataFrame, historical: dict | None = None, subset: str = "validation") -> pd.DataFrame:
    """Summary rows of the two baselines (one fixed run each, no seed spread) in the layout of `summarize`."""
    rows = []
    for name, part in baseline_metrics[baseline_metrics["subset"] == subset].groupby("model", sort=False):
        for metric in METRIC_COLUMNS:
            value = pd.to_numeric(part[metric], errors="coerce").iloc[0]
            defined = bool(np.isfinite(value))
            rows.append({"pipeline": name, "metric": metric, "n_runs": 1, "n_defined": int(defined), "mean": float(value) if defined else None, "sd_ddof1": None,
                         "min": float(value) if defined else None, "max": float(value) if defined else None, "undefined_note": BASELINE_NOTE,
                         "historical_single_seed_value": float(value) if defined else None})
    return pd.DataFrame(rows)


def study_summary(preset: str, seeds: list[int], baseline_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validation summary of a study including its reference groups (and, for the multi-seed study, the early single-run values and the two baselines)."""
    groups = study_groups(preset)
    summary = summarize(collect_study_runs(preset, seeds), groups)
    for column in reversed(list(group_description(preset, groups[0]))):
        summary.insert(1, column, [group_description(preset, n)[column] for n in summary["pipeline"]])
    if preset == "multiseed":
        single = {name: collect_runs([single_name], config.PRESET_SEEDS["single_runs"]) for name, single_name in config.SINGLE_RUN_OF.items()}
        historical = []
        for r in summary.itertuples():
            run = single[r.pipeline]
            historical.append(float(run[(run["subset"] == "validation")][r.metric].iloc[0]) if r.metric in ("roc_auc", "average_precision") else None)
        summary["historical_single_seed_value"] = historical
        if baseline_metrics is not None:
            summary = pd.concat([summary, baseline_rows(baseline_metrics)], ignore_index=True)
    return summary
