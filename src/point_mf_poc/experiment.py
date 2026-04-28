from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .data import make_noisy_state, make_target_cloud
from .metrics import evaluate
from .model import (
    gna_refine,
    lion_latent_diffusion_reconstruct,
    one_nfe_mean_flow_update,
    predict_mean_velocity_with_errors,
)
from .types import TrialResult


@dataclass(frozen=True)
class ExperimentConfig:
    categories: tuple[str, ...] = ("car", "chair", "airplane")
    trials_per_category: int = 8
    points: int = 256
    seed_base: int = 1000
    gna_steps: int = 10
    gna_lr: float = 0.42
    lion_steps: int = 64
    output_dir: Path = Path(".")


def run_trial(category: str, seed: int, config: ExperimentConfig) -> TrialResult:
    rng = np.random.default_rng(seed)
    target = make_target_cloud(category, seed, n=config.points)
    xt = make_noisy_state(target, rng)

    start = time.perf_counter()
    velocity = predict_mean_velocity_with_errors(xt, target, rng, t=1.0)
    baseline_cloud = one_nfe_mean_flow_update(xt, velocity, t=1.0, r=0.0)
    baseline_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    anchored_cloud = gna_refine(baseline_cloud, target, steps=config.gna_steps, lr=config.gna_lr)
    anchored_ms = baseline_ms + (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    lion_cloud = lion_latent_diffusion_reconstruct(xt, target, rng, steps=config.lion_steps)
    lion_ms = (time.perf_counter() - start) * 1000.0

    return TrialResult(
        category=category,
        seed=seed,
        baseline=evaluate(baseline_cloud, target, baseline_ms),
        anchored=evaluate(anchored_cloud, target, anchored_ms),
        lion=evaluate(lion_cloud, target, lion_ms),
        target=target,
        baseline_cloud=baseline_cloud,
        anchored_cloud=anchored_cloud,
        lion_cloud=lion_cloud,
    )


def run_experiment(config: ExperimentConfig) -> list[TrialResult]:
    results: list[TrialResult] = []
    for category_index, category in enumerate(config.categories):
        for trial in range(config.trials_per_category):
            seed = config.seed_base + category_index * 100 + trial
            results.append(run_trial(category, seed, config))
    return results
