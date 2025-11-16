# ResNet50 Fine-tuning on CIFAR-10

OFDBまたはImageNetで事前学習したResNet50をCIFAR-10でfine-tuningして比較するプロジェクトです。

## 必要なデータ

### 1. CIFAR-10データセット

CIFAR-10のバイナリ版をダウンロード:
```
https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

展開後、以下の構造にする:
```
burn_resnet/
  cifar-10-batches-bin/
    data_batch_1.bin
    data_batch_2.bin
    data_batch_3.bin
    data_batch_4.bin
    data_batch_5.bin
    test_batch.bin
```

### 2. 事前学習済みモデル

- **OFDBモデル**: `checkpoints/model_epoch_XXX.mpk`
- **ImageNetモデル**: Burnには事前学習済みモデルが含まれていないため、OFDBモデルと同様に事前学習が必要です

## 使用方法

### 1. OFDB事前学習モデルでfine-tuning

```bash
cargo run --release --bin finetune checkpoints/model_epoch_500.mpk
```

### 2. ランダム初期化（事前学習なし）

```bash
cargo run --release --bin finetune
```

### 3. 比較実験

#### ステップ1: OFDBで事前学習
```bash
cargo run --release --bin train
```

#### ステップ2: OFDB事前学習モデルでfine-tuning
```bash
cargo run --release --bin finetune checkpoints/model_epoch_500.mpk
```

#### ステップ3: ランダム初期化モデルでfine-tuning
```bash
cargo run --release --bin finetune
```

## パラメータ

`src/finetune.rs`で以下を調整可能:

- `image_size`: 224 (CIFAR-10の32x32を224x224にリサイズ)
- `batch_size`: 32
- `num_epochs`: 50
- `learning_rate`: 0.0001

## 出力

### 学習中の出力例:
```
=== Epoch 1/50 ===
Epoch 1, Batch 0: Loss = 2.3012, Accuracy = 10.25%
Epoch 1, Batch 10: Loss = 2.1045, Accuracy = 15.62%
...
Epoch 1 - Train Loss: 2.0123, Train Accuracy: 20.45%
Epoch 1 - Test Loss: 1.9234, Test Accuracy: 22.31%
```

### チェックポイント

`cifar10_checkpoints/`フォルダに10エポックごとに保存されます。

## 評価指標

両方のモデルで以下を比較:
- **訓練精度**: 各エポックの訓練データでの精度
- **テスト精度**: 各エポックのテストデータでの精度
- **収束速度**: 高精度に到達するまでのエポック数
- **最終精度**: 最後のエポックでのテスト精度

## 期待される結果

一般的に、事前学習済みモデルは:
- より速く収束する
- より高い最終精度を達成する
- より少ないエポックで良い結果が得られる

ランダム初期化モデルと比較することで、OFDBでの事前学習の効果を確認できます。
