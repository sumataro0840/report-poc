# Point-MF / GNA PoC 実行手順

`260310baba-y_4.pdf` のMean Flow/GNAの主張と、ルートに置いた `LION.pdf` の潜在点拡散の考え方を、小さな合成点群実験で比較します。

比較する方式は次の3つです。

- `MeanFlow`: 1-step平均速度場更新
- `MeanFlow + GNA`: 1-step更新後に `x0` 側の集合距離アンカーで補正
- `LION-style`: LIONの階層VAE/latent point diffusionを模した反復型ベースライン

このPoCは論文モデルそのものではなく、各手法の性質を切り出して速度と復元品質を比較するための実験です。

## セットアップ

Python 3.10 以上を使います。依存ライブラリは `requirements.txt` から入れます。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

使っている主なライブラリ:

- `numpy`: 合成点群生成、Mean Flow更新、GNA補正
- `scipy`: 最近傍探索、Chamfer Distance、Hungarian EMD
- `matplotlib`: 点群比較図のPNG出力

## 実行

```bash
source .venv/bin/activate
python3 main.py
```

実行後、次のファイルが更新されます。

- `report.txt`: コンソールと同じ実験サマリ
- `report.html`: 表と点群PNG可視化つきの確認用HTML
- `metrics.csv`: trialごとのメトリクス
- `artifacts/cloud_comparison.png`: `MeanFlow` / `MeanFlow + GNA` / `LION-style` の点群比較図

## 結果の見方

まず `report.txt` か `report.html` の `安定化判定` を確認します。

判定は次の4条件をすべて満たすと `PASS` です。

- `CD improvement >= 25%`
- `Hungarian EMD improvement >= 10%`
- `F-Score gain >= 0.15`
- `Outlier reduction >= 0.03`

指標の意味:

- `CD`: Chamfer Distance。小さいほど正解点群に近い
- `Hungarian EMD`: SciPyの線形割当で計算する対応距離。小さいほど点群全体の対応がよい
- `F-Score`: しきい値内に入った点のPrecision/Recall。大きいほどよい
- `Outlier`: 正解点群から離れた生成点の割合。小さいほど安定

典型的には `MeanFlow + GNA` で `CD`、`Hungarian EMD`、`Outlier` が下がり、`F-Score` が上がればGNAで安定化できています。

`LION-style` は反復型なので、`ms/sample` が `MeanFlow` より大きくなります。品質は `CD`、`Hungarian EMD`、`F-Score` を見て比較します。

## 実験内容

### MeanFlow

速度場だけで次の1ステップ更新を行います。

```text
x0 = xt - (t - r) * u
```

### MeanFlow + GNA

`MeanFlow` の出力 `x0` を正解点群への最近傍集合距離で補正します。論文のGNAを学習しているわけではなく、GNAの補助損失が学習中に促す方向を、実験で直接かけている簡易版です。

### LION-style

`LION.pdf` は、点群を階層VAEの潜在空間へ写像し、global shape latent と point-structured latent に対してdenoising diffusion modelを学習する手法です。

このPoCでは以下で近似しています。

1. 正解点群を局所近傍で平滑化し、`latent point` とみなす
2. ノイズから複数ステップでlatent pointへ反復デノイズする
3. decoder相当として高周波の形状差分を一部戻す

そのため `LION-style` は `MeanFlow` より遅い一方、反復デノイズにより点群全体の対応が良くなる傾向があります。

## コード構成

```text
main.py                         実行入口
requirements.txt                依存ライブラリ
src/point_mf_poc/data.py         合成点群生成
src/point_mf_poc/model.py        Mean Flow更新、GNA相当補正、LION-style反復復元
src/point_mf_poc/metrics.py      CD / EMD / F-Score / 安定化判定
src/point_mf_poc/experiment.py   trial実行
src/point_mf_poc/reports.py      txt/html/csv出力
src/point_mf_poc/visualization.py 点群比較PNG出力
```

## 生成物を消して再実行したい場合

```bash
rm -f report.txt report.html metrics.csv artifacts/cloud_comparison.png
python3 main.py
```

## 制限

- DINOv3、DiT、JVP、APML/Sinkhornの本実装ではありません。
- LIONのVAE/DDMを学習しているわけではありません。
- 画像入力ではなく、`car` / `chair` / `airplane` の合成カテゴリを単一画像条件の代用にしています。
- GNAは学習損失ではなく、予測点群への直接補正として再現しています。
- LION-styleは、LIONの設計思想を比較用に単純化した反復型ベースラインです。

次の段階では、PyTorchで小型Transformerや小型VAE/DDMを学習し、GNAあり/なし、LION-style latent diffusionあり/なしの学習曲線と推論点群を比較すると、論文実装に近い検証になります。
