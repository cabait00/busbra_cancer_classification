"""Starts explicitly requested classification training runs: one fresh Python process per variant and seed.

The trainer itself is `busbra.training.train_variant`; this module only plans the runs, checks that no output exists, and starts the child processes with the
environment the study runs need (PYTHONHASHSEED equal to the seed for study variants, none for the historical single runs, no foreign CUDA library path).
Nothing here imports TensorFlow, loads images or models, or writes results. A failing child stops the plan; there is no fallback to other parameters.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from busbra import config, data


def plan_runs(preset: str, seeds: list[int] | None = None, variants: list[str] | None = None) -> list[tuple[str, int]]:
    """(variant, seed) in the documented order: seeds outer in the given order, variants inner in the order of config.PRESETS. Invalid names or seeds are errors."""
    names = config.PRESETS[preset]
    chosen = variants if variants else names
    unknown = [v for v in chosen if v not in names]
    if unknown:
        raise ValueError(f"variants {unknown} do not belong to preset {preset}; valid: {names}")
    seeds = seeds if seeds else config.seeds_for(preset)
    allowed = config.seeds_for(preset)
    if any(s not in allowed for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"preset {preset} is defined for the seeds {allowed} (each once), not {seeds}")
    return [(name, seed) for seed in seeds for name in names if name in chosen]


def check_plan(plan: list[tuple[str, int]], outputs=config.run_outputs) -> None:
    """Refuses the whole plan before any process starts: an existing output of a planned run, or a fine-tuning run whose source checkpoint neither exists nor is planned earlier."""
    problems, planned = [], set()
    for name, seed in plan:
        variant = config.VARIANTS[name]
        present = [path for path in outputs(variant, seed) if Path(path).exists()]
        if present:
            problems.append(f"{name} / seed {seed}: {len(present)} of its {len(outputs(variant, seed))} output files already exist (for example {present[0]})")
        if variant.start == "checkpoint":
            source = config.VARIANTS[variant.source]
            if (variant.source, seed) not in planned and not Path(config.run_paths(source, seed)["checkpoint"]).exists():
                problems.append(f"{name} / seed {seed}: the checkpoint of {variant.source} does not exist and is not part of this plan")
        planned.add((name, seed))
    if problems:
        raise FileExistsError("; ".join(problems) + ". Nothing was started and nothing was changed.")


def check_inputs(plan: list[tuple[str, int]], geometry_reader=None) -> None:
    """Inputs the planned runs need before any process starts (no TensorFlow): runs on expert-mask ROI inputs need the stored, validated expert geometry. The geometry is shared by all
    runs and seeds; it is read here, never measured and never written by a training run."""
    if any(config.VARIANTS[name].data == "roi" for name, _ in plan):
        (geometry_reader or data.read_expert_geometry)()


def child_environment(variant: config.Variant, seed: int, base: dict | None = None) -> dict:
    """Environment of one training process (PYTHONHASHSEED must be set before Python starts, so it cannot be set by the trainer itself)."""
    env = dict(os.environ if base is None else base)
    env.pop("LD_LIBRARY_PATH", None)  # TensorFlow steps run without a foreign CUDA library path
    if variant.seed_mode == "study":
        env["PYTHONHASHSEED"] = str(seed)
    else:
        env.pop("PYTHONHASHSEED", None)  # the historical single runs had no hash seed
    return env


def child_command(script: Path, preset: str, name: str, seed: int) -> list[str]:
    return [sys.executable, str(script), "--preset", preset, "--train-one", name, "--seeds", str(seed)]


def run_plan(preset: str, script: Path, seeds: list[int] | None = None, variants: list[str] | None = None, starter=subprocess.run, base_env: dict | None = None,
             outputs=config.run_outputs, geometry_reader=None) -> list[tuple[str, int]]:
    """Check the plan (outputs, source checkpoints, shared inputs), then start one child process per run, one after the other. `starter` is subprocess.run (replaceable for tests)."""
    plan = plan_runs(preset, seeds, variants)
    check_plan(plan, outputs)
    check_inputs(plan, geometry_reader)
    for name, seed in plan:
        result = starter(child_command(script, preset, name, seed), env=child_environment(config.VARIANTS[name], seed, base_env))
        if result.returncode != 0:
            raise RuntimeError(f"{name} / seed {seed} failed with exit code {result.returncode}; the remaining runs were not started")
    return plan
