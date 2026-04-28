from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from .metrics import average_metrics, fmt, summarize_stability
from .types import TrialResult


def make_text_report(results: list[TrialResult]) -> str:
    baseline = average_metrics([r.baseline for r in results])
    anchored = average_metrics([r.anchored for r in results])
    lion = average_metrics([r.lion for r in results])
    stability = summarize_stability(results)
    cd_improvement = (baseline.chamfer - anchored.chamfer) / baseline.chamfer * 100.0
    emd_improvement = (baseline.emd - anchored.emd) / baseline.emd * 100.0

    lines = [
        "Point-MF / Geometry Noise Anchor PoC",
        "=" * 80,
        "目的: 260310baba-y_4.pdf の主張を、合成点群の1-NFE更新で小さく検証する。",
        "",
        "平均結果",
        "Method       | CD↓    | Hungarian EMD↓ | F-Score↑ | Outlier↓ | ms/sample",
        "-------------+--------+----------------+----------+----------+----------",
        f"MeanFlow     | {fmt(baseline.chamfer)} | {fmt(baseline.emd)}         | {fmt(baseline.fscore)}   | {fmt(baseline.outlier_rate)}   | {fmt(baseline.ms)}",
        f"MeanFlow+GNA | {fmt(anchored.chamfer)} | {fmt(anchored.emd)}         | {fmt(anchored.fscore)}   | {fmt(anchored.outlier_rate)}   | {fmt(anchored.ms)}",
        f"LION-style   | {fmt(lion.chamfer)} | {fmt(lion.emd)}         | {fmt(lion.fscore)}   | {fmt(lion.outlier_rate)}   | {fmt(lion.ms)}",
        "",
        f"CD改善率: {cd_improvement:.1f}%",
        f"Hungarian EMD改善率: {emd_improvement:.1f}%",
        "",
        f"安定化判定: {'PASS' if stability.passed else 'FAIL'}",
    ]
    for check in stability.checks:
        lines.append(f"- {'OK' if check.passed else 'NG'}: {check.label} (actual: {check.actual})")

    lines.extend(["", "カテゴリ別"])
    for category in sorted({r.category for r in results}):
        category_results = [r for r in results if r.category == category]
        b = average_metrics([r.baseline for r in category_results])
        a = average_metrics([r.anchored for r in category_results])
        l = average_metrics([r.lion for r in category_results])
        lines.append(
            f"- {category}: CD MeanFlow {fmt(b.chamfer)}, GNA {fmt(a.chamfer)}, LION {fmt(l.chamfer)}; "
            f"F-Score MeanFlow {fmt(b.fscore)}, GNA {fmt(a.fscore)}, LION {fmt(l.fscore)}"
        )

    lines.extend(
        [
            "",
            "解釈:",
            "- 速度場だけの1-step更新では、点の取り違えや外れ点が残る。",
            "- GNA相当の集合距離アンカーを入れると、x0側が表面へ寄り、CD/EMD/F-Scoreが改善する。",
            "- LION-style はLIONの階層VAE/潜在点拡散を模した反復型ベースラインで、反復回数ぶん遅いが潜在空間で段階的に復元する。",
            "- これは論文の「点群空間の大きな区間ジャンプを、データ空間の幾何拘束で安定化する」主張に対応する。",
        ]
    )
    return "\n".join(lines)


def make_html_report(results: list[TrialResult], figure_path: Path) -> str:
    baseline = average_metrics([r.baseline for r in results])
    anchored = average_metrics([r.anchored for r in results])
    lion = average_metrics([r.lion for r in results])
    stability = summarize_stability(results)
    verdict = "PASS" if stability.passed else "FAIL"
    stability_items = "".join(
        f"<li>{'OK' if check.passed else 'NG'}: {escape(check.label)} (actual: {escape(check.actual)})</li>"
        for check in stability.checks
    )
    rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{method}</td><td>{fmt(metrics.chamfer)}</td><td>{fmt(metrics.emd)}</td>"
                f"<td>{fmt(metrics.fscore)}</td><td>{fmt(metrics.outlier_rate)}</td><td>{fmt(metrics.ms)}</td>"
                "</tr>"
            )
            for method, metrics in [("MeanFlow", baseline), ("MeanFlow + GNA", anchored), ("LION-style", lion)]
        ]
    )

    category_rows = []
    for category in sorted({r.category for r in results}):
        category_results = [r for r in results if r.category == category]
        b = average_metrics([r.baseline for r in category_results])
        a = average_metrics([r.anchored for r in category_results])
        l = average_metrics([r.lion for r in category_results])
        category_rows.append(
            "<tr>"
            f"<td>{escape(category)}</td>"
            f"<td>{fmt(b.chamfer)} / {fmt(a.chamfer)} / {fmt(l.chamfer)}</td>"
            f"<td>{fmt(b.fscore)} / {fmt(a.fscore)} / {fmt(l.fscore)}</td>"
            f"<td>{fmt(b.ms)} / {fmt(a.ms)} / {fmt(l.ms)}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Point-MF / GNA PoC</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; line-height: 1.7; }}
    header {{ padding: 42px max(22px, calc((100vw - 1060px) / 2)) 30px; border-bottom: 1px solid #d8dee8; background: #f8fbfc; }}
    main {{ max-width: 1060px; margin: 0 auto; padding: 30px 22px 56px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.14; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; padding-bottom: 8px; color: #006d77; border-bottom: 2px solid #d8dee8; letter-spacing: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.95rem; }}
    th, td {{ border: 1px solid #d8dee8; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fb; }}
    code {{ background: #f6f8fb; padding: 1px 5px; border-radius: 4px; }}
    .meta {{ color: #667085; display: flex; flex-wrap: wrap; gap: 8px 16px; }}
    .figure {{ width: 100%; max-width: 980px; border: 1px solid #d8dee8; }}
    @media (max-width: 720px) {{ table {{ display: block; overflow-x: auto; white-space: nowrap; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Point-MF / Geometry Noise Anchor PoC</h1>
    <div class="meta">
      <span>対象: 260310baba-y_4.pdf</span>
      <span>実験: 合成点群 {len(results)} trials</span>
      <span>依存: NumPy / SciPy / Matplotlib</span>
    </div>
  </header>
  <main>
    <section>
      <h2>何を確認しているか</h2>
      <p>点群空間で <code>x0 = xt - (t-r)u</code> の1-NFE更新を行うとき、速度場の誤差が外れ点や密度偏りとして出る状況を作り、<code>x0</code> を集合距離で拘束するGNA相当の補正が効くかを確認します。さらに、LIONの階層VAE/潜在点拡散を模した反復型ベースラインと速度・品質を比較します。</p>
    </section>
    <section>
      <h2>平均結果</h2>
      <p><strong>安定化判定: {verdict}</strong></p>
      <ul>{stability_items}</ul>
      <table>
        <thead><tr><th>Method</th><th>CD↓</th><th>Hungarian EMD↓</th><th>F-Score↑</th><th>Outlier↓</th><th>ms/sample</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    <section>
      <h2>カテゴリ別</h2>
      <table>
        <thead><tr><th>Category</th><th>CD<br>MeanFlow / GNA / LION</th><th>F-Score<br>MeanFlow / GNA / LION</th><th>ms/sample<br>MeanFlow / GNA / LION</th></tr></thead>
        <tbody>{''.join(category_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>可視化サンプル</h2>
      <p>灰色が正解点群、色付きが生成点群です。GNAは1-step出力を表面近傍へ寄せ、LION-styleは潜在空間で反復的に復元します。</p>
      <img class="figure" src="{escape(figure_path.as_posix())}" alt="MeanFlow, MeanFlow + GNA, and LION-style point cloud comparison">
    </section>
  </main>
</body>
</html>"""


def write_csv(results: list[TrialResult], filename: Path) -> None:
    with filename.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "seed", "method", "chamfer", "emd", "fscore", "outlier_rate", "ms"])
        for result in results:
            for method, metrics in [
                ("mean_flow", result.baseline),
                ("mean_flow_gna", result.anchored),
                ("lion_style", result.lion),
            ]:
                writer.writerow(
                    [
                        result.category,
                        result.seed,
                        method,
                        fmt(metrics.chamfer),
                        fmt(metrics.emd),
                        fmt(metrics.fscore),
                        fmt(metrics.outlier_rate),
                        fmt(metrics.ms),
                    ]
                )


def write_reports(results: list[TrialResult], output_dir: Path, figure_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(make_text_report(results), encoding="utf-8")
    (output_dir / "report.html").write_text(make_html_report(results, figure_path), encoding="utf-8")
    write_csv(results, output_dir / "metrics.csv")
