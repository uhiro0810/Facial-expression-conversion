# 顔画像の表情変換（AU 条件）README — OpenFace 出力フォルダ構成対応版

本リポジトリは、**OpenFace の AU 出力**をそのまま使って、入力顔画像の**同一性を保ったまま表情のみを編集**する実験コード一式です。以下の 4 フォルダ構成を前提に、学習・推論・評価まで最短経路で実行できるようにしています。

```
📁 Train              # 学習用の元画像（jpg/png）
📁 Test               # 評価/推論用の元画像
📁 OpenFace_AU_Train  # Train に対応する OpenFace の出力（*.csv）
📁 OpenFace_AU_Test   # Test に対応する OpenFace の出力（*.csv）
```

> **前提**：OpenFace の CSV は標準列（`frame, timestamp, AU01_r, AU02_r, ...` 等）を想定。利用する AU は `_r`（回帰）系を基本とし、必要に応じて `_c`（分類）を併用します。

---

## 1. 特徴（サマリ）
- **AU 条件制御**：AU 強度を連続値で指定し、微細な表情コントロール
- **同一性保持**：ArcFace 類似度で人物性を維持（ID 損失）
- **軽量学習**：Stable Diffusion (SD1.5) + ControlNet に **LoRA(q/k/v)** を注入
- **自動評価**：AU 一致度 / LPIPS / ArcFace 類似度を計測

---

## 2. セットアップ
### 依存関係（例）
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers peft accelerate
pip install facenet-pytorch lpips opencv-python albumentations pandas
```
> GPU 推奨。`Pipelines loaded with dtype=torch.float16 cannot run with cpu` エラー時は `--mixed_precision fp16` を外すか、`cuda` を使用してください。

---

## 3. データ準備
### 3.1 アノテーション CSV の生成
OpenFace 出力（CSV）と画像ファイル名を**ベース名一致**（拡張子を除いたファイル名）で対応付け、学習/評価用のアノテーションを作ります。

```bash
python scripts/make_annotations_from_openface.py \
  --img_dir Train \
  --openface_dir OpenFace_AU_Train \
  --out_csv data/annotations_train.csv

python scripts/make_annotations_from_openface.py \
  --img_dir Test \
  --openface_dir OpenFace_AU_Test \
  --out_csv data/annotations_test.csv
```

**スクリプトの仕様（想定）**
- 対応付け：`<name>.jpg` ↔ `<name>.csv`（または `<name>_of.csv` など、前方一致）
- AU 列：`[AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r]`
- 正規化：`clip(value, 0, 5) / 5.0` をデフォルト（必要に応じて変更可）
- 出力列：`image_path, AU01, AU02, ..., AU45`

> **注**：動画→連番画像の場合は、フレーム毎に 1 行。静止画で OpenFace が 1 行だけ出す場合も同様に処理します。

### 3.2 前処理（任意）
- 顔検出/アライメント（RetinaFace + 5 点ランドマーク）
- 解像度の統一（例：`256×256`）
- 本 README では **既に整列済み**の前提で進めます（必要なら `scripts/preprocess.py` を用意してください）。

---

## 4. 学習（LoRA）
```bash
accelerate launch scripts/train.py \
  --config configs/train_sd15_lora.yaml \
  --train_csv data/annotations_train.csv \
  --img_root . \
  --au_keys AU01 AU02 AU04 AU06 AU12 AU15 AU25 AU26 AU45 \
  --lora_rank 16 --batch_size 8 --max_steps 20000 \
  --lr 1e-4 --mixed_precision fp16 \
  --lambda_id 0.5 --lambda_au 1.0 --lambda_lpips 0.2
```
**ハイパーパラメータのヒント**
- **人物性が崩れる**：`--lambda_id` を上げる、LoRA Rank を下げる、学習率を下げる
- **表情が弱い**：`--lambda_au` を上げる、AU 埋め込み層の容量を増やす
- **ノイズ化**：学習ステップ増加、EMA 導入、正規化の見直し

---

## 5. 推論（img2img / ControlNet）
### 5.1 img2img
```bash
python scripts/infer_img2img.py \
  --input Test/sample.jpg \
  --output outputs/sample_out.jpg \
  --au "AU04:0.6,AU12:0.7,AU25:0.6" \
  --strength 0.35 --guidance_scale 7.5 \
  --lora_path models/lora_unet.safetensors
```

### 5.2 ControlNet（任意の構図制約）
```bash
python scripts/infer_controlnet.py \
  --input Test/sample.jpg \
  --cond conds/canny.png \
  --au "AU06:0.5,AU12:0.8" \
  --strength 0.30 --guidance_scale 8.0 \
  --controlnet_path models/controlnet.safetensors
```

---

## 6. 評価（Test + OpenFace_AU_Test）
```bash
python scripts/evaluate.py \
  --refs Test \
  --gens outputs \
  --metrics au,arcface,lpips \
  --au_csv data/annotations_test.csv \
  --report results/metrics.json
```
**計算指標**
- **AU 一致度**：`||f(I′) − a||`（OpenFace 互換 AU 推定器 `f`）
- **ArcFace 類似度**（↑）/ **LPIPS**（↓）

---

## 7. フォルダと命名のルール
- `Train/xxx.jpg` に対し、`OpenFace_AU_Train/xxx.csv` が 1:1 で存在すること
- 拡張子のみ違う場合はベース名で照合
- 連番（例：`video_000123.jpg`）は **完全一致**を推奨

---

## 8. 典型的なエラーと対処
- **マッチしない画像が多い**：ファイル名の空白/全角・半角/大文字小文字を統一
- **AU 列が欠ける**：不足列は 0 で補完するか、`--au_keys` から外す
- **CPU で fp16 例外**：`--mixed_precision` を外す or GPU を使用
- **発散/別人化**：`--lambda_id`↑、学習率↓、LoRA Rank↓、ID バランスの良いデータを追加

---

## 9. 結果テンプレート
| 指標 | 値 |
|---|---|
| AU 一致度 (↓) | 0.18 |
| ArcFace 類似度 (↑) | 0.62 |
| LPIPS (↓) | 0.21 |

> 目標に応じて `λ_id` と `λ_au` のトレードオフを調整してください。

---

## 10. 倫理・ライセンス
- 顔画像と生成物の扱いは**同意・プライバシー・公平性**に配慮
- なりすまし等の誤用を禁止（研究目的に限定）
- ライセンス例：MIT（必要に応じて変更）

---

## 付録 A. アノテーション CSV の例
```csv
image_path,AU01,AU02,AU04,AU06,AU12,AU15,AU25,AU26,AU45
Train/0001.jpg,0.02,0.01,0.00,0.35,0.71,0.06,0.58,0.12,0.00
Train/0002.jpg,0.00,0.00,0.10,0.22,0.15,0.00,0.20,0.00,0.00
...
```

## 付録 B. 参考スクリプト（ダミー仕様）
> `scripts/make_annotations_from_openface.py` の擬似仕様（実装の目安）
```python
# 1) openface_dir の *.csv を列挙
# 2) 対応する画像を img_dir から探索（ベース名一致）
# 3) AU*_r 列を抽出し [0,5]→/5.0 で正規化
# 4) image_path と AU ベクトルを CSV に書き出し
```

---

