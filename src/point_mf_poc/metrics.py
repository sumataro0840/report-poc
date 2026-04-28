from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix

from .types import Cloud, Metrics, StabilityCheck, StabilitySummary, TrialResult


def nearest_distances(source: Cloud, target: Cloud) -> np.ndarray:
    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)
    return distances


def chamfer_distance(a: Cloud, b: Cloud) -> float:
    return float((nearest_distances(a, b).mean() + nearest_distances(b, a).mean()) / 2.0)


def emd_hungarian(a: Cloud, b: Cloud) -> float:
    distances = distance_matrix(a, b)
    row_indices, col_indices = linear_sum_assignment(distances)
    return float(distances[row_indices, col_indices].mean())


def fscore(a: Cloud, b: Cloud, threshold: float = 0.085) -> float:
    precision = float((nearest_distances(a, b) <= threshold).mean())
    recall = float((nearest_distances(b, a) <= threshold).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def outlier_rate(a: Cloud, b: Cloud, threshold: float = 0.22) -> float:
    return float((nearest_distances(a, b) > threshold).mean())


def evaluate(predicted: Cloud, target: Cloud, elapsed_ms: float) -> Metrics:
    return Metrics(
        chamfer=chamfer_distance(predicted, target),
        emd=emd_hungarian(predicted, target),
        fscore=fscore(predicted, target),
        outlier_rate=outlier_rate(predicted, target),
        ms=elapsed_ms,
    )


def average_metrics(metrics: list[Metrics]) -> Metrics:
    return Metrics(
        chamfer=sum(m.chamfer for m in metrics) / len(metrics),
        emd=sum(m.emd for m in metrics) / len(metrics),
        fscore=sum(m.fscore for m in metrics) / len(metrics),
        outlier_rate=sum(m.outlier_rate for m in metrics) / len(metrics),
        ms=sum(m.ms for m in metrics) / len(metrics),
    )


def summarize_stability(results: list[TrialResult]) -> StabilitySummary:
    baseline = average_metrics([r.baseline for r in results])
    anchored = average_metrics([r.anchored for r in results])
    cd_improvement = (baseline.chamfer - anchored.chamfer) / baseline.chamfer * 100.0
    emd_improvement = (baseline.emd - anchored.emd) / baseline.emd * 100.0
    fscore_gain = anchored.fscore - baseline.fscore
    outlier_reduction = baseline.outlier_rate - anchored.outlier_rate

    checks = [
        StabilityCheck("CD improvement >= 25%", cd_improvement >= 25.0, f"{cd_improvement:.1f}%"),
        StabilityCheck("Hungarian EMD improvement >= 10%", emd_improvement >= 10.0, f"{emd_improvement:.1f}%"),
        StabilityCheck("F-Score gain >= 0.15", fscore_gain >= 0.15, f"{fscore_gain:.4f}"),
        StabilityCheck("Outlier reduction >= 0.03", outlier_reduction >= 0.03, f"{outlier_reduction:.4f}"),
    ]
    return StabilitySummary(passed=all(check.passed for check in checks), checks=checks)


def fmt(value: float) -> str:
    return f"{value:.4f}"
