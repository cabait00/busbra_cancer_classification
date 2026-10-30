# %% [03] Classification studies: small CNN and MobileNetV2 variants (train and validation only)
#
# One entry point for all neural classification experiments. The variants and their settings are listed in src/busbra/config.py
# (VARIANTS, PRESETS); the training itself is src/busbra/training.py. Presets:
#   single_runs     steps 3A-3E, seed 42 (small CNN, frozen MobileNetV2, limited / extended fine-tuning, fine-tuning with 1e-4)
#   multiseed       step 3F, five pipelines over seeds 7, 21, 42, 123, 2026
#   resolution250   step 4A, 250 px input
#   augmentation160 step 4B, flip + rotation + zoom
#   expert_roi      step 4E, expert-mask ROI crop (needs a manual lesion mask when applied; not a pure image-to-score pipeline)
#
# Actions (exactly one is chosen explicitly; the default only reads stored results and starts nothing):
#   --show             list which runs exist, print the validation results and compare the recomputed study tables with the stored ones
#   --train            train the selected variants and seeds on the GPU: one fresh Python process per variant and seed, started by busbra.launcher (it sets PYTHONHASHSEED
#                      equal to the seed for study presets, as the worker processes of the earlier studies had); refuses the whole plan if an output of a run already exists
#   --train-one NAME   direct entry for ONE variant and ONE seed in this process (what the launcher starts); study variants need PYTHONHASHSEED=<seed> in the environment
#   --aggregate        write the study tables (run metrics, summary with its reference groups, paired differences; for expert_roi also threshold curves and operating points) that do not
#                      exist yet; nothing is written if one exists
#   --prepare          (expert_roi) measure the expert-mask geometry of the training and validation pairs ONCE for all seeds and variants and write it with its frequency table; the ROI
#                      training and the adversarial add-on read this geometry (busbra.data.read_expert_geometry) and never measure masks themselves; refuses if the tables exist
#
# Valid seeds: the five study seeds 7, 21, 42, 123, 2026 for the study presets; the single runs of steps 3A-3E exist for seed 42 only. --variants selects a subset of a preset.
#
# Examples (from the project folder):
#   .venv/bin/python analysis/03_classification.py --preset single_runs                      # the early seed-42 runs, including the small CNN (src/busbra/models.py)
#   .venv/bin/python analysis/03_classification.py --preset multiseed                        # five seeds, with reference groups and the stored-table comparison
#   .venv/bin/python analysis/03_classification.py --preset expert_roi --prepare              # once: expert geometry of train and validation (needs the extracted files)
#   .venv/bin/python analysis/03_classification.py --preset expert_roi --train --seeds 7 21  # GPU: one process per variant and seed, seed 7 first; checks the plan and the geometry first
#   PYTHONHASHSEED=7 .venv/bin/python analysis/03_classification.py --preset expert_roi --train-one mobilenetv2_frozen_160_roi --seeds 7
#
# The small CNN (builder build_small_cnn in src/busbra/models.py, shared trainer src/busbra/training.py) replaces the former single-file script. Do not mix up the early
# seed-42 run (preset single_runs, variant small_cnn) with the five-seed results of the final comparison (preset multiseed, variant small_cnn_160).
#
# The test subset is never loaded here. Fine-tuning variants start from the best checkpoint of their source variant for the same seed, so a preset is trained in the order of PRESETS.
# Existing results are never overwritten.

# %% Imports
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from busbra import aggregation, config, data, results, roi_study  # noqa: E402

FULL_METRICS = ("roc_auc", "average_precision", "sensitivity", "specificity", "precision", "f1", "accuracy")
# Cross-study comparisons of the later studies: preset -> (reference preset, metrics, [(label, pipeline a, pipeline b)]); the difference is b minus a within the same seed
V = config.VARIANTS
CROSS = {
    "resolution250": ("multiseed", ("roc_auc", "average_precision", "sensitivity", "specificity"),
                      [("mobilenetv2_frozen_250 minus mobilenetv2_frozen_160", "mobilenetv2_frozen_160", "mobilenetv2_frozen_250"),
                       ("mobilenetv2_higher_lr_250 minus mobilenetv2_higher_lr_160", "mobilenetv2_higher_lr_160", "mobilenetv2_higher_lr_250"),
                       ("mobilenetv2_higher_lr_250 minus mobilenetv2_frozen_250 (250 px)", "mobilenetv2_frozen_250", "mobilenetv2_higher_lr_250"),
                       ("mobilenetv2_higher_lr_160 minus mobilenetv2_frozen_160 (160 px)", "mobilenetv2_frozen_160", "mobilenetv2_higher_lr_160")]),
    "augmentation160": ("multiseed", FULL_METRICS,
                        [("mobilenetv2_frozen_160_aug minus mobilenetv2_frozen_160", "mobilenetv2_frozen_160", "mobilenetv2_frozen_160_aug"),
                         ("mobilenetv2_higher_lr_160_aug minus mobilenetv2_higher_lr_160", "mobilenetv2_higher_lr_160", "mobilenetv2_higher_lr_160_aug"),
                         ("mobilenetv2_higher_lr_160_aug minus mobilenetv2_frozen_160_aug (with augmentation)", "mobilenetv2_frozen_160_aug", "mobilenetv2_higher_lr_160_aug"),
                         ("mobilenetv2_higher_lr_160 minus mobilenetv2_frozen_160 (3F, flip only)", "mobilenetv2_frozen_160", "mobilenetv2_higher_lr_160")]),
    "expert_roi": ("augmentation160", FULL_METRICS + ("tn", "fp", "fn", "tp"),
                   [("mobilenetv2_frozen_160_roi minus mobilenetv2_frozen_160_aug", "mobilenetv2_frozen_160_aug", "mobilenetv2_frozen_160_roi"),
                    ("mobilenetv2_higher_lr_160_roi minus mobilenetv2_higher_lr_160_aug", "mobilenetv2_higher_lr_160_aug", "mobilenetv2_higher_lr_160_roi"),
                    ("mobilenetv2_higher_lr_160_roi minus mobilenetv2_frozen_160_roi (expert ROI)", "mobilenetv2_frozen_160_roi", "mobilenetv2_higher_lr_160_roi"),
                    ("mobilenetv2_higher_lr_160_aug minus mobilenetv2_frozen_160_aug (full image, 4B)", "mobilenetv2_frozen_160_aug", "mobilenetv2_higher_lr_160_aug")]),
}


# %% Actions
def existing(variant: config.Variant, seed: int) -> bool:
    return config.run_paths(variant, seed)["metrics"].is_file()


def cross_table(preset: str, seeds: list[int]) -> pd.DataFrame | None:
    """Paired differences of a later study against its reference study and within itself (one generic function for the earlier separate implementations)."""
    if preset not in CROSS:
        return None
    reference_preset, metrics, pairs = CROSS[preset]
    both = pd.concat([aggregation.collect_runs(config.PRESETS[reference_preset], seeds), aggregation.collect_runs(config.PRESETS[preset], seeds)], ignore_index=True)
    return aggregation.paired_between(both, both, pairs, seeds, metrics)


def study_tables(preset: str, seeds: list[int]) -> dict[str, pd.DataFrame]:
    """All tables of a study, recomputed in memory from the stored per-run metrics: run metrics of the own variants, summary with reference groups, paired differences and their summary."""
    names = config.PRESETS[preset]
    runs = aggregation.collect_runs(names, seeds)
    problems = aggregation.validate_run_table(aggregation.collect_study_runs(preset, seeds), seeds, aggregation.study_groups(preset))
    if problems:
        raise ValueError("incomplete or inconsistent runs: " + "; ".join(problems))
    baselines = pd.read_csv(config.TABLES_DIR / "baseline_metrics.csv") if preset == "multiseed" else None
    tables = {"run_metrics": runs, "summary_validation": aggregation.study_summary(preset, seeds, baselines)}
    cross = cross_table(preset, seeds)
    if cross is None and len(names) > 1:
        cross = aggregation.paired_differences(runs, names)
    if cross is not None:
        tables["paired_differences"], tables["paired_summary"] = cross, aggregation.summarize_pairs(cross)
    if preset == "expert_roi":  # threshold curves and operating points of the four groups (full image and expert ROI, frozen and fine-tuned) from the stored validation scores
        sets = roi_study.expert_score_sets(seeds)
        roi_study.check_score_sets(sets, data.subset_frame(data.read_manifest(), "validation"), seeds)
        tables.update(roi_study.expert_operating_tables(sets))
    return tables


def compare_with_stored(preset: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Per table: identical to the stored one? The run metrics of the earlier studies hold more technical columns, so only the shared columns are compared; the descriptive `source` text may be worded differently."""
    lines = []
    for key, frame in tables.items():
        path = config.TABLES_DIR / f"{preset}_{key}.csv"
        if not path.is_file():
            lines.append(f"{path.name}: not stored")
            continue
        stored = pd.read_csv(path, nrows=0).columns
        if key == "run_metrics":
            shared = [c for c in frame.columns if c in stored and c != "run_id"]
            full = pd.read_csv(path)[shared]
            same = len(frame) == len(full) and results.same_as_stored(frame[shared], path, ignore=[c for c in stored if c not in shared])
        else:
            same = results.same_as_stored(frame, path, ignore=("source",))
        lines.append(f"{path.name}: {'identical' if same else 'DIFFERENT'}")
    return lines


def show_single_runs() -> None:
    runs = aggregation.collect_runs(config.PRESETS["single_runs"], config.seeds_for("single_runs"))
    columns = ["pipeline", "roc_auc", "average_precision", "sensitivity", "specificity", "precision", "f1", "accuracy", "best_epoch"]
    for subset in ("validation", "train"):
        print(f"\n{subset} metrics at threshold 0.5 (seed {config.HISTORICAL_SEED}):")
        print(runs[runs["subset"] == subset][columns].round(4).to_string(index=False))
    print("\nThese are the early single runs of steps 3A-3E (small CNN first). The five-seed results of the final comparison are the preset multiseed (small_cnn_160).")


def show(preset: str, seeds: list[int]) -> None:
    names = config.PRESETS[preset]
    print(f"preset {preset} (step {config.PRESET_STEPS[preset]}): {len(names)} variants x {len(seeds)} seeds")
    for name in names:
        done = [s for s in seeds if existing(V[name], s)]
        print(f"  {name:34s} results for seeds {done or 'none'}")
    if not all(existing(V[n], s) for n in names for s in seeds):
        print("  (the summary needs the results of all listed runs)")
        return
    if preset == "single_runs":
        show_single_runs()
        return
    tables = study_tables(preset, seeds)
    summary = tables["summary_validation"]
    print(summary[summary["metric"].isin(["roc_auc", "average_precision"])][["pipeline", "metric", "n_runs", "mean", "sd_ddof1", "min", "max"]].to_string(index=False))
    if sorted(seeds) != sorted(config.seeds_for(preset)):
        print(f"  PARTIAL VIEW of the seeds {sorted(seeds)}: not comparable with the stored tables of all {len(config.seeds_for(preset))} study seeds; no equality is checked, nothing can be written from it")
        return
    print("  tables recomputed from the stored run results vs. stored tables:")
    for line in compare_with_stored(preset, tables):
        print("   ", line)


def train(preset: str, seeds: list[int], variants: list[str] | None) -> None:
    from busbra import launcher  # no TensorFlow here: the children import it
    plan = launcher.run_plan(preset, Path(__file__).resolve(), seeds, variants)
    print(f"finished {len(plan)} runs")


def train_one(preset: str, name: str, seeds: list[int]) -> None:
    """Direct entry: one variant, one seed, this process."""
    if name not in config.PRESETS[preset]:
        raise SystemExit(f"{name} does not belong to preset {preset}: {config.PRESETS[preset]}")
    if len(seeds) != 1:
        raise SystemExit("--train-one needs exactly one seed")
    from busbra import training  # imports TensorFlow only when training is requested
    training.train_variant(V[name], seeds[0])


def write_tables(preset: str, seeds: list[int]) -> None:
    """Write the study tables that do not exist yet. All targets are checked first; if one exists nothing is written (the stored tables of the earlier studies stay as they are)."""
    if sorted(seeds) != sorted(config.seeds_for(preset)):
        raise ValueError(f"the study tables aggregate all study seeds {config.seeds_for(preset)}, not {sorted(seeds)}; nothing was read or written")
    targets = {key: config.TABLES_DIR / f"{preset}_{key}.csv" for key in table_keys(preset)}
    clash = [p.name for p in targets.values() if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was written")
    tables = study_tables(preset, seeds)
    for key, frame in tables.items():
        aggregation.write_new_table(frame, targets[key])
    print(f"wrote {len(tables)} tables of {preset} to {config.TABLES_DIR}")


def table_keys(preset: str) -> list[str]:
    """The tables an aggregation of the preset writes (known before any input is read, so that all targets can be checked first)."""
    keys = ["run_metrics", "summary_validation"]
    if len(config.PRESETS[preset]) > 1 or preset in CROSS:
        keys += ["paired_differences", "paired_summary"]
    return keys + (list(roi_study.EXPERT_OPERATING_FILES) if preset == "expert_roi" else [])


def prepare() -> None:
    """Measure the expert-mask geometry of the training and validation pairs once and write it with its frequency table. Explicit action; the only place that opens these masks for the studies.
    Both targets are checked first; if one exists nothing is measured. No test image or mask is read."""
    targets = [data.expert_geometry_path(), data.expert_frequencies_path()]
    clash = [p.name for p in targets if p.exists()]
    if clash:
        raise FileExistsError(f"already present: {clash}; nothing was measured and nothing was written")
    manifest = data.read_manifest()
    data.check_split(manifest)
    geometry = data.measure_expert_geometry(config.PROJECT_ROOT, manifest)
    data.validate_expert_geometry(geometry, manifest)
    geometry.to_csv(targets[0], index=False, lineterminator="\n")
    roi_study.geometry_frequencies(geometry).to_csv(targets[1], index=False, lineterminator="\n")
    print(f"wrote the expert geometry of {len(geometry)} pairs and its frequency table to {config.TABLES_DIR}")


# %% Entry point
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--preset", choices=sorted(config.PRESETS), default="multiseed")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the seeds of the preset (single runs: 42 only)")
    parser.add_argument("--variants", nargs="+", help="subset of the variants of the preset (--train)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--show", action="store_true", help="read stored results only (default)")
    action.add_argument("--train", action="store_true", help="train on the GPU, one process per variant and seed")
    action.add_argument("--train-one", metavar="NAME", help="train one variant for one seed in this process (GPU)")
    action.add_argument("--aggregate", action="store_true", help="write the study tables")
    action.add_argument("--prepare", action="store_true", help="measure the expert geometry of train and validation once (preset expert_roi)")
    args = parser.parse_args(argv)
    allowed = config.seeds_for(args.preset)
    seeds = args.seeds or allowed
    if any(s not in allowed for s in seeds) or len(set(seeds)) != len(seeds):
        parser.error(f"preset {args.preset} is defined for the seeds {allowed} (each once), not {seeds}")
    if args.aggregate and args.preset == "single_runs":
        parser.error("single_runs has one seed (42) and no study tables; its stored comparison tables are listed in docs/; use the default view or the preset multiseed")
    if args.prepare and args.preset != "expert_roi":
        parser.error("--prepare belongs to the preset expert_roi (the geometry of the expert-mask ROI is prepared once for all seeds and variants)")
    if args.aggregate and sorted(seeds) != sorted(allowed):
        parser.error(f"the study tables aggregate all study seeds {allowed}; the seeds {sorted(seeds)} give a partial view only (default action), not a table to write")
    if args.prepare and args.seeds:
        parser.error("--prepare measures the geometry of train and validation once for all seeds; it takes no --seeds")
    if args.variants and not args.train:
        parser.error("--variants selects the variants of a --train plan; the other actions always use the whole preset (use --train-one for a single variant)")
    if args.variants and any(v not in config.PRESETS[args.preset] for v in args.variants):
        parser.error(f"--variants must be from {config.PRESETS[args.preset]}")
    if args.train:
        train(args.preset, seeds, args.variants)
    elif args.train_one:
        train_one(args.preset, args.train_one, seeds)
    elif args.aggregate:
        write_tables(args.preset, seeds)
    elif args.prepare:
        prepare()
    else:
        show(args.preset, seeds)


if __name__ == "__main__":
    main(sys.argv[1:] if "__file__" in globals() else [])
