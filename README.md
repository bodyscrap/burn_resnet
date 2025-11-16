# ResNet50 Training and Fine-tuning with Burn

このプロジェクトは、Rust製機械学習ライブラリ[Burn](https://github.com/tracel-ai/burn)を使用したResNet50の実装です。2D-OFDB-1kデータセットでの事前学習と、CIFAR-10でのファインチューニングをサポートしています。

## プロジェクト構成

### ソースファイル
- `src/model.rs` - ResNet50アーキテクチャの実装（Bottleneckブロック、転移学習用のヘッド置き換え機能）
- `src/dataset.rs` - ImageFolder形式のデータローダー（PNG/PPM/JPG/BMP対応）
- `src/main.rs` - 2D-OFDB-1kでの事前学習スクリプト
- `src/finetune.rs` - CIFAR-10でのファインチューニングスクリプト
- `src/convert_cifar10.rs` - CIFAR-10バイナリをImageFolder形式に変換

### バイナリ
- `train` - 2D-OFDB-1kでの事前学習
- `finetune` - CIFAR-10でのファインチューニング（スクラッチまたは事前学習済み）
- `convert_cifar10` - CIFAR-10データセット変換ツール

## データセット構造

### 2D-OFDB-1k（事前学習用）
```
2D-OFDB-1k/
  00000/
    *.ppm
  00001/
    *.ppm
  ...
  00999/
    *.ppm
```
各フォルダが1つのクラス（1000クラス）を表し、フォルダ内にPPM形式の画像が含まれます。

### CIFAR-10（ファインチューニング用）
```
cifar-10-images/
  airplane/
    *.png
  automobile/
    *.png
  bird/
    *.png
  ...
  truck/
    *.png
```
ImageFolder形式（10クラス、各5000枚の訓練画像）。`convert_cifar10`で自動生成されます。

## セットアップ

### 1. ビルド
```bash
cargo build --release
```

### 2. CIFAR-10データセット変換
CIFAR-10バイナリデータセットを[公式サイト](https://www.cs.toronto.edu/~kriz/cifar.html)からダウンロードし、`cifar-10-batches-bin/`に展開後、以下を実行:

```bash
cargo run --release --bin convert_cifar10
```

これにより`cifar-10-images/`ディレクトリに50,000枚のPNG画像が生成されます。

## 使用方法

### 事前学習（2D-OFDB-1k）

```bash
cargo run --release --bin train
```

**設定:**
- 画像サイズ: 224×224
- バッチサイズ: 16
- エポック数: 100
- 学習率: 0.001
- クラス数: 1000
- チェックポイント: `checkpoints/model_epoch_*.mpk`（10エポックごと）

**メモリ最適化:**
- バッチごとに画像を読み込み、処理完了後即座に解放
- 24GB GPUでの動作を確認

### ファインチューニング（CIFAR-10）

#### スクラッチから学習
```bash
cargo run --release --bin finetune
```

#### 事前学習済みモデルから学習
```bash
cargo run --release --bin finetune checkpoints/model_epoch_100.mpk
```

**設定:**
- 画像サイズ: 224×224
- バッチサイズ: 16
- エポック数: 20
- 学習率: 0.0001（事前学習済み）
- クラス数: 10
- 評価データ: 1000サンプル

**出力:**
- `cifar10_finetuning_scratch.csv` - スクラッチ学習の結果
- `cifar10_finetuning_pretrained.csv` - 事前学習済みモデルの結果

**CSV形式:**
```csv
epoch,train_loss,train_accuracy,test_loss,test_accuracy
1,1.387062,49.9286,1.082862,61.8000
...
```

### 両方の実験を一括実行

```powershell
.\run_both_experiments.ps1
```

このスクリプトは以下を順次実行します:
1. スクラッチからのCIFAR-10学習
2. 事前学習済みモデルでのCIFAR-10ファインチューニング
3. 結果の比較表示

## メモリ最適化の実装

このプロジェクトは、24GB GPUでResNet50を訓練するために以下のメモリ最適化を実装しています:

### 訓練時
- バッチごとの画像読み込み（全データを一度にメモリに展開しない）
- 各バッチ処理後の自動メモリ解放
- 厳密なスコープ管理によるテンソルの早期解放

### 評価時
- **重要**: 訓練モデル（Autodiff）を明示的にdropしてGPUメモリを解放
- 素のバックエンド（勾配トラッキングなし）で評価モデルをロード
- 評価完了後、訓練モデルを再ロード

この実装により、訓練と評価で異なるモデルインスタンスを使用し、GPU OOMエラーを回避しています。

## 実験結果

### スクラッチ学習（20エポック）
- 最終訓練精度: 97.92%
- 最終テスト精度: 97.80%

### 事前学習済みモデル（20エポック）
- 実行中...

## 技術仕様

- **フレームワーク**: Burn 0.16
- **バックエンド**: WGPU（DirectX12/Vulkan対応）
- **最適化**: Adam (epsilon=1e-8)
- **損失関数**: CrossEntropy
- **正規化**: ImageNet標準（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
- **データ拡張**: リサイズのみ（224×224）

## 依存関係

- Rust 1.70以上
- Burn 0.16（train、wgpu、dataset機能）
- burn-wgpu 0.16
- image 0.25
- walkdir 2.5
- rand 0.8

詳細は`Cargo.toml`を参照してください。

## トラブルシューティング

### GPU OOMエラー
- バッチサイズを削減（16 → 8）
- 評価サンプル数を削減（1000サンプル程度）

### コンパイルエラー
- `cargo clean`を実行後、再ビルド
- Rustツールチェーンを最新版に更新

## ライセンス

MIT License
