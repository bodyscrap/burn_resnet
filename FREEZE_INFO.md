# Backboneのフリーズについて

## 現在の実装

現在の`finetune.rs`では、**全ての層が学習可能**（freezeされていません）になっています。

## Burn 0.16の制限

Burn 0.16では、PyTorchの`requires_grad=False`のような層単位のfreezeがネイティブにサポートされていません。

## 推奨される代替アプローチ

### 1. 低い学習率を使用（現在の実装）

事前学習済みモデルでは、小さい学習率（0.0001）を使用することで、backboneの変化を抑制します。

```rust
let learning_rate = 0.0001; // 事前学習モデル用
// let learning_rate = 0.001; // ランダム初期化用
```

### 2. 段階的な学習

**Phase 1: ヘッドのみ学習** (最初の10エポック)
- 学習率: 0.0001

**Phase 2: 全体をfine-tuning** (残りのエポック)
- 学習率: 0.00001

### 3. より高度な実装（将来のburn対応後）

Burnが層単位の学習率制御をサポートした場合:

```rust
// 疑似コード（未実装）
let optimizer = AdamConfig::new()
    .with_param_group("backbone", 0.00001)  // backboneは低学習率
    .with_param_group("head", 0.001)        // headは高学習率
    .init();
```

## 実験的な確認

事前学習モデルと比較することで、backboneの効果を確認できます:

1. **ランダム初期化** (backbone無し)
   ```bash
   cargo run --release --bin finetune
   ```

2. **OFDB事前学習モデル** (backboneあり)
   ```bash
   cargo run --release --bin finetune checkpoints/model_epoch_500.mpk
   ```

事前学習モデルの方が:
- より速く収束
- より高い精度を達成

これにより、backboneの特徴が活用されていることが確認できます。

## まとめ

現在の実装では全層が学習可能ですが、**低学習率**を使用することで事前学習された特徴を保持しながらfine-tuningを行います。これは一般的なfine-tuning戦略の一つです。
