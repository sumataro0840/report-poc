from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Cloud = NDArray[np.float64]


@dataclass(frozen=True)
class Metrics:
    chamfer: float
    emd: float
    fscore: float
    outlier_rate: float
    ms: float


@dataclass(frozen=True)
class TrialResult:
    category: str
    seed: int
    baseline: Metrics
    anchored: Metrics
    lion: Metrics
    target: Cloud
    baseline_cloud: Cloud
    anchored_cloud: Cloud
    lion_cloud: Cloud


@dataclass(frozen=True)
class StabilityCheck:
    label: str
    passed: bool
    actual: str


@dataclass(frozen=True)
class StabilitySummary:
    passed: bool
    checks: list[StabilityCheck]
