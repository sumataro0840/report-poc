# Point-MF / GNA / LION-style PoC

`260310baba-y_4.pdf` の Mean Flow + Geometry Noise Anchor と、`LION.pdf` の latent point diffusion の考え方を、小さな合成点群実験で比較するPoCです。

このリポジトリは論文のフル再実装ではありません。目的は、点群空間での大きな `1-NFE` 更新に対して、GNA相当の幾何アンカーやLION-styleの反復的な潜在点拡散が、速度と復元品質にどう効くかを確認することです。

## 比較する方式

- `MeanFlow`: 平均速度場による1-step更新
- `MeanFlow + GNA`: `MeanFlow` 出力を点群集合距離で補正
- `LION-style`: LIONの階層VAE / point-structured latent diffusion を単純化した反復型ベースライン

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行

```bash
python3 main.py
```

実行後に以下が生成されます。

- `report.txt`: 実験サマリ
- `report.html`: 表と点群比較図つきHTML
- `metrics.csv`: trialごとのメトリクス
- `artifacts/cloud_comparison.png`: 3方式の点群比較図

詳しい手順と判定基準は [RUN.md](RUN.md) を参照してください。

## 主なコード構成

```text
main.py                          実行入口
requirements.txt                 依存ライブラリ
src/point_mf_poc/data.py          合成点群生成
src/point_mf_poc/model.py         Mean Flow / GNA / LION-style
src/point_mf_poc/metrics.py       CD / EMD / F-Score / 安定化判定
src/point_mf_poc/experiment.py    実験実行
src/point_mf_poc/reports.py       txt/html/csv出力
src/point_mf_poc/visualization.py 点群比較PNG出力
```

## 評価指標

- `CD`: Chamfer Distance。小さいほどよい
- `Hungarian EMD`: 線形割当による点群対応距離。小さいほどよい
- `F-Score`: 近傍しきい値内のPrecision/Recall。大きいほどよい
- `Outlier`: 正解点群から離れた生成点の割合。小さいほどよい
- `ms/sample`: 1サンプルあたりの実行時間

## 制限

- DINOv3、DiT、JVP、APML/Sinkhornは実装していません。
- LIONのVAE/DDMを学習しているわけではありません。
- 画像入力ではなく、`car` / `chair` / `airplane` の合成点群を使います。
- GNAは学習損失ではなく、予測点群への直接補正として再現しています。

## 次の拡張候補

- PyTorchで小型Transformerを学習してGNAあり/なしを比較する
- 小型VAE + diffusionを実装してLION-styleを学習ベースに近づける
- 実データまたはShapeNet subsetで同じ比較を行う

## 参考文献
-https://arxiv.org/pdf/2210.06978
-https://mm.cs.uec.ac.jp/pub/conf25/260310baba-y_4.pdf

