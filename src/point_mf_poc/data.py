from __future__ import annotations

import numpy as np

from .types import Cloud


def sample_box_surface(
    rng: np.random.Generator,
    count: int,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> Cloud:
    faces = rng.integers(0, 6, size=count)
    points = rng.uniform(-0.5, 0.5, size=(count, 3)) * np.asarray(size, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0

    points[faces == 0, 0] = -half[0]
    points[faces == 1, 0] = half[0]
    points[faces == 2, 1] = -half[1]
    points[faces == 3, 1] = half[1]
    points[faces == 4, 2] = -half[2]
    points[faces == 5, 2] = half[2]

    return points + np.asarray(center, dtype=float)


def sample_cylinder(
    rng: np.random.Generator,
    count: int,
    center: tuple[float, float, float],
    radius: float,
    width: float,
    axis: str = "x",
) -> Cloud:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    circle = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    span = rng.uniform(-width / 2.0, width / 2.0, size=count)
    points = np.zeros((count, 3), dtype=float)

    if axis == "x":
        points[:, 0] = span
        points[:, 1:] = circle
    elif axis == "y":
        points[:, 1] = span
        points[:, 0] = circle[:, 0]
        points[:, 2] = circle[:, 1]
    else:
        points[:, 2] = span
        points[:, :2] = circle

    return points + np.asarray(center, dtype=float)


def _fit_count(points: list[Cloud], n: int, rng: np.random.Generator) -> Cloud:
    cloud = np.concatenate(points, axis=0)
    if len(cloud) < n:
        repeats = rng.choice(len(cloud), size=n - len(cloud), replace=True)
        cloud = np.concatenate([cloud, cloud[repeats]], axis=0)
    return cloud[:n]


def make_target_cloud(category: str, seed: int, n: int = 256) -> Cloud:
    """Make a synthetic target point cloud for a category."""

    rng = np.random.default_rng(seed)
    parts: list[Cloud]

    if category == "car":
        parts = [
            sample_box_surface(rng, n // 2, (0.0, 0.0, 0.06), (1.25, 0.52, 0.28)),
            sample_box_surface(rng, n // 5, (-0.08, 0.0, 0.31), (0.62, 0.45, 0.23)),
            sample_cylinder(rng, n // 8, (-0.42, -0.29, -0.12), 0.13, 0.08, "y"),
            sample_cylinder(rng, n // 8, (0.42, -0.29, -0.12), 0.13, 0.08, "y"),
            sample_cylinder(rng, n // 8, (-0.42, 0.29, -0.12), 0.13, 0.08, "y"),
            sample_cylinder(rng, n // 8, (0.42, 0.29, -0.12), 0.13, 0.08, "y"),
        ]
    elif category == "chair":
        legs = [
            sample_box_surface(rng, n // 16, (x, y, -0.44), (0.08, 0.08, 0.82))
            for x in (-0.28, 0.28)
            for y in (-0.22, 0.22)
        ]
        parts = [
            sample_box_surface(rng, n // 4, (0.0, 0.0, 0.0), (0.72, 0.64, 0.12)),
            sample_box_surface(rng, n // 4, (0.0, 0.28, 0.48), (0.72, 0.10, 0.86)),
            *legs,
        ]
    elif category == "airplane":
        parts = [
            sample_cylinder(rng, n // 3, (0.0, 0.0, 0.0), 0.10, 1.35, "x"),
            sample_box_surface(rng, n // 3, (0.0, 0.0, 0.0), (0.32, 1.35, 0.045)),
            sample_box_surface(rng, n // 6, (-0.55, 0.0, 0.12), (0.28, 0.42, 0.04)),
            sample_box_surface(rng, n // 6, (0.68, 0.0, 0.0), (0.24, 0.16, 0.16)),
        ]
    else:
        raise ValueError(f"unknown category: {category}")

    cloud = _fit_count(parts, n, rng)
    return cloud + rng.normal(0.0, 0.01, size=cloud.shape)


def make_noisy_state(target: Cloud, rng: np.random.Generator, t: float = 1.0) -> Cloud:
    return target + rng.normal(0.0, 0.55 * t, size=target.shape)
