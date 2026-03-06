# AU 条件付き顔表情変換

AffectNet データセットと OpenFace の Action Unit (AU) 出力を用いて、顔画像の**同一性を保ったまま表情のみを編集**する実験プロジェクトです。Google Colab 上で実行することを前提としています。

## ディレクトリ構成

```
workspace/
├── code_202312944_uno/          # Jupyter Notebook（実験コード一式）
│   ├── image_rename.ipynb       # 1. 画像ファイルの連番リネーム
│   ├── AU.ipynb                 # 2. OpenFace による AU 抽出 & Parquet 変換
│   ├── makePairs.ipynb          # 3. 画像パス ↔ AU ベクトルのペア CSV 生成
│   ├── distill.ipynb            # 4. ResNet18 による AU 推定モデルの蒸留学習
│   ├── ControlNet.ipynb         # 5. SD2.1 + ControlNet + LoRA による表情編集学習・推論
│   └── Img2Img.ipynb            # 6. SD1.5 + LoRA による Img2Img 表情編集学習・推論・FID 評価
├── OpenFace_AU_Train/           # Train 画像に対応する OpenFace AU 出力
│   ├── anger/
│   │   ├── anger.csv            #   AU 値（frame 単位）
│   │   └── anger_of_details.txt #   OpenFace 実行時のメタ情報
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── OpenFace_AU_Test/            # Test 画像に対応する OpenFace AU 出力（同構成）
├── train_pairs.csv              # 学習用ペア（11,183 件）
├── test_pairs.csv               # 評価用ペア（8,733 件）
└── README.md
```

> **注**: 画像データ（`Train/`, `Test/`）は Google Drive 上に配置されており、本リポジトリには含まれません。

## 実験パイプライン

### Step 1. 画像リネーム（`image_rename.ipynb`）

AffectNet の各表情フォルダ内の画像を `image_000001.jpg` 形式に連番リネームします。

- 対象: `Train/` および `Test/` 配下の 8 表情カテゴリ
- Train: 約 16,108 枚 / Test: 約 14,518 枚

### Step 2. OpenFace AU 抽出（`AU.ipynb`）

Google Colab 上で OpenFace をビルドし、`FeatureExtraction` で各表情フォルダの画像から AU を一括抽出します。

- dlib（CPU）+ OpenFace をソースからビルド
- 8 表情クラスごとに AU CSV を出力
- CSV → Parquet 変換で Train / Test それぞれ集約

**AU CSV の列構成**（OpenFace 標準出力）:
```
frame, face_id, timestamp, confidence, success,
AU01_r, AU02_r, AU04_r, ..., AU45_r,   # 回帰値（17列）
AU01_c, AU02_c, AU04_c, ..., AU45_c    # 分類値（18列）
```

### Step 3. ペア CSV 生成（`makePairs.ipynb`）

画像パスと対応する AU ベクトル（35 次元 = 17 回帰 + 18 分類）をペアにした CSV を生成します。

- 出力: `train_pairs.csv`（11,183 件）, `test_pairs.csv`（8,733 件）
- 各行の形式: `image_path, au_vector`（au_vector は JSON リスト）

### Step 4. AU 推定モデルの蒸留（`distill.ipynb`）

OpenFace の AU 出力を教師信号として、軽量な ResNet18 ベースの AU 推定ネットワークを学習します。学習ループ内の AU 損失計算に使用します。

- モデル: ResNet18 → Linear(512, 35) → Sigmoid
- 損失: L1 (MAE)
- 10 エポック学習、最終 MAE: 0.2273
- 出力: `resnet18_au.pth`

### Step 5. ControlNet による表情編集（`ControlNet.ipynb`）

Stable Diffusion 2.1 + ControlNet (MediaPipe Face) + LoRA で AU 条件付き表情編集モデルを学習・推論します。

- **ベースモデル**: `stabilityai/stable-diffusion-2-1-base`
- **ControlNet**: `CrucibleAI/ControlNetMediaPipeFace`
- **LoRA**: rank=16, target=`to_k, to_q, to_v, to_out.0`
- **AU 条件入力**: AU ベクトル → Linear(35, cross_attn_dim) → CLIP 空プロンプトと連結
- **損失関数**: `0.1*L_noise + 0.5*L_id + 1.0*L_au + 0.1*L_perc`
  - L_noise: 拡散ノイズ予測 MSE
  - L_id: ArcFace (VGGFace2) によるコサイン類似度損失
  - L_au: 蒸留 AU ネットによる AU 一致度 (L1)
  - L_perc: LPIPS (VGG) 知覚的損失
- 10,000 ステップ学習、バッチサイズ 32、画像解像度 96px

### Step 6. Img2Img による表情編集（`Img2Img.ipynb`）

Stable Diffusion 1.5 + LoRA で Img2Img ベースの表情編集を行います。

- **ベースモデル**: `runwayml/stable-diffusion-v1-5`
- **LoRA**: rank=8, target=`to_q, to_k, to_v, to_out.0`
- **AU 条件入力**: AU ベクトル → MLP(35→512→768) → UNet の cross-attention へ
- **損失関数**: 拡散ノイズ予測 MSE + Classifier-Free Guidance (drop=10%)
- 勾配累積 4 ステップ、画像解像度 256px
- **推論**: DDIM スケジューラ、strength=0.6、guidance_scale=7.5
- **評価**: FID (Frechet Inception Distance)、テスト 200 サンプル

## データ仕様

### 表情カテゴリ（8 クラス）

| カテゴリ | 英名 | Train 枚数 | Test 枚数 |
|---------|------|-----------|----------|
| 怒り | anger | 1,500 | 1,718 |
| 軽蔑 | contempt | 1,559 | 1,312 |
| 嫌悪 | disgust | 1,229 | 1,248 |
| 恐怖 | fear | 1,512 | 1,664 |
| 喜び | happy | 2,340 | 2,704 |
| 中立 | neutral | 2,758 | 2,368 |
| 悲しみ | sad | 3,091 | 1,584 |
| 驚き | surprise | 2,119 | 1,920 |

### AU ベクトル（35 次元）

前半 17 次元が回帰値（`_r`、0〜5 の連続値）、後半 18 次元が分類値（`_c`、0 or 1）です。

| # | AU 列 | 意味 |
|---|-------|------|
| 1 | AU01_r | Inner Brow Raiser |
| 2 | AU02_r | Outer Brow Raiser |
| 3 | AU04_r | Brow Lowerer |
| 4 | AU05_r | Upper Lid Raiser |
| 5 | AU06_r | Cheek Raiser |
| 6 | AU07_r | Lid Tightener |
| 7 | AU09_r | Nose Wrinkler |
| 8 | AU10_r | Upper Lip Raiser |
| 9 | AU12_r | Lip Corner Puller |
| 10 | AU14_r | Dimpler |
| 11 | AU15_r | Lip Corner Depressor |
| 12 | AU17_r | Chin Raiser |
| 13 | AU20_r | Lip Stretcher |
| 14 | AU23_r | Lip Tightener |
| 15 | AU25_r | Lips Part |
| 16 | AU26_r | Jaw Drop |
| 17 | AU45_r | Blink |
| 18-35 | AU*_c | 上記 + AU28_c の分類値 |

## 実行環境

- **プラットフォーム**: Google Colab (GPU ランタイム)
- **Python**: 3.11
- **主要ライブラリ**:
  - PyTorch 2.3.0 (CUDA 12.1)
  - diffusers 0.32.0 / 0.28.0+
  - transformers 4.40.0
  - peft 0.10.0
  - accelerate
  - facenet-pytorch (ArcFace)
  - lpips (知覚的損失)
  - mediapipe 0.10.9 (ControlNet 用ランドマーク検出)
  - OpenFace (AU 抽出、Colab 上でビルド)

## 実行手順（概要）

```
1. image_rename.ipynb   → 画像を連番にリネーム
2. AU.ipynb             → OpenFace で AU 抽出 → Parquet 化
3. makePairs.ipynb      → (image_path, au_vector) ペア CSV 生成
4. distill.ipynb        → ResNet18 AU 推定モデルを蒸留学習
5. ControlNet.ipynb     → SD2.1 + ControlNet + LoRA で表情編集学習・推論
   or
6. Img2Img.ipynb        → SD1.5 + LoRA で Img2Img 表情編集学習・推論・FID 評価
```

> 各 Notebook は Google Drive マウント (`/content/drive/MyDrive/AffectNet/`) を前提としています。パスは環境に応じて修正してください。

## 倫理・注意事項

- 顔画像データおよび生成物の取り扱いには**同意・プライバシー・公平性**に十分配慮してください
- なりすまし等の悪用は禁止です（研究目的に限定）
- AffectNet データセットの利用条件に従ってください
