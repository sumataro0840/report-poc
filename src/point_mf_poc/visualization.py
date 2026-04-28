from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib-cache").resolve()))

import matplotlib.pyplot as plt

from .types import TrialResult


def save_cloud_comparison(result: TrialResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 4.6), dpi=150)

    views = [
        ("MeanFlow", result.baseline_cloud, "#b75d31"),
        ("MeanFlow + GNA", result.anchored_cloud, "#006d77"),
        ("LION-style", result.lion_cloud, "#5b5ea6"),
    ]
    for index, (title, cloud, color) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(result.target[:, 0], result.target[:, 1], result.target[:, 2], s=7, c="#aab6c4", alpha=0.32)
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=8, c=color, alpha=0.88)
        ax.set_title(f"{result.category}: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=20, azim=-55)
        ax.set_box_aspect((1.4, 1.0, 0.9))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
