from __future__ import annotations

from pathlib import Path

from src.point_mf_poc.experiment import ExperimentConfig, run_experiment
from src.point_mf_poc.reports import make_text_report, write_reports
from src.point_mf_poc.visualization import save_cloud_comparison


def main() -> None:
    output_dir = Path(".")
    artifacts_dir = Path("artifacts")
    figure_path = artifacts_dir / "cloud_comparison.png"

    config = ExperimentConfig(
        categories=("car", "chair", "airplane"),
        trials_per_category=8,
        points=256,
        output_dir=output_dir,
    )
    results = run_experiment(config)

    save_cloud_comparison(results[0], figure_path)
    write_reports(results, output_dir, figure_path)

    print(make_text_report(results))
    print("\n出力: report.txt, report.html, metrics.csv, artifacts/cloud_comparison.png")


if __name__ == "__main__":
    main()
