mod dataset;
mod model;

use burn::{
    data::dataset::Dataset,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, Optimizer},
    record::CompactRecorder,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};
use burn_wgpu::{Wgpu, WgpuDevice};
use dataset::{ImageFolderDataset, ImageItem};
use model::ResNet;
use rand::seq::SliceRandom;
use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;

type MyBackend = Wgpu;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

struct Batcher {
    device: WgpuDevice,
}

impl Batcher {
    fn new(device: WgpuDevice) -> Self {
        Self { device }
    }
    
    // 素のバックエンド用（評価時）
    fn batch_no_grad(&self, items: Vec<ImageItem>) -> (Tensor<MyBackend, 4>, Tensor<MyBackend, 1, Int>) {
        let batch_size = items.len();
        let image_size = 224;
        
        let mut images_data = Vec::with_capacity(batch_size * 3 * image_size * image_size);
        let mut labels_data = Vec::with_capacity(batch_size);
        
        for item in items {
            images_data.extend(item.image);
            labels_data.push(item.label as i32);
        }
        
        let images = Tensor::<MyBackend, 4>::from_data(
            TensorData::new(images_data, [batch_size, 3, image_size, image_size]),
            &self.device,
        );
        
        let labels = Tensor::<MyBackend, 1, Int>::from_data(
            TensorData::new(labels_data, [batch_size]),
            &self.device,
        );
        
        (images, labels)
    }

    fn batch(&self, items: Vec<ImageItem>) -> (Tensor<MyAutodiffBackend, 4>, Tensor<MyAutodiffBackend, 1, Int>) {
        let batch_size = items.len();
        let image_size = (items[0].image.len() as f32 / 3.0).sqrt() as usize;
        
        let mut images_data = Vec::new();
        let mut labels_data = Vec::new();
        
        for item in items {
            images_data.extend(item.image);
            labels_data.push(item.label as i64);
        }
        
        let images = Tensor::<MyAutodiffBackend, 4>::from_data(
            TensorData::new(images_data, [batch_size, 3, image_size, image_size]),
            &self.device,
        );
        
        let labels = Tensor::<MyAutodiffBackend, 1, Int>::from_data(
            TensorData::new(labels_data, [batch_size]),
            &self.device,
        );
        
        (images, labels)
    }
    
    fn batch_valid(&self, items: Vec<ImageItem>) -> (Tensor<MyBackend, 4>, Tensor<MyBackend, 1, Int>) {
        let batch_size = items.len();
        let image_size = (items[0].image.len() as f32 / 3.0).sqrt() as usize;
        
        let mut images_data = Vec::new();
        let mut labels_data = Vec::new();
        
        for item in items {
            images_data.extend(item.image);
            labels_data.push(item.label as i64);
        }
        
        let images = Tensor::<MyBackend, 4>::from_data(
            TensorData::new(images_data, [batch_size, 3, image_size, image_size]),
            &self.device,
        );
        
        let labels = Tensor::<MyBackend, 1, Int>::from_data(
            TensorData::new(labels_data, [batch_size]),
            &self.device,
        );
        
        (images, labels)
    }
}

fn train_epoch(
    model: ResNet<MyAutodiffBackend>,
    dataset: &ImageFolderDataset,
    indices: &[usize],
    batcher: &Batcher,
    optimizer: &mut impl Optimizer<ResNet<MyAutodiffBackend>, MyAutodiffBackend>,
    epoch: usize,
    learning_rate: f64,
    batch_size: usize,
) -> (ResNet<MyAutodiffBackend>, f32, f32) {
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_samples = 0;
    let mut num_batches = 0;
    let mut current_model = model;
    
    let dataset_len = indices.len();
    let expected_batches = (dataset_len + batch_size - 1) / batch_size;
    
    if num_batches == 0 {
        println!("Dataset size: {}, Batch size: {}, Expected batches: {}", 
                 dataset_len, batch_size, expected_batches);
        println!("Loading first batch...");
    }
    
    for batch_start in (0..dataset_len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset_len);
        
        // バッチごとに画像を読み込む（スコープ内で処理）
        let (images, labels, batch_len) = {
            let mut batch_items = Vec::new();
            for i in batch_start..batch_end {
                let idx = indices[i];
                if let Some(item) = dataset.get(idx) {
                    batch_items.push(item);
                }
            }
            
            if num_batches == 0 {
                println!("First batch items loaded: {}", batch_items.len());
            }
            
            if batch_items.is_empty() {
                continue;
            }
            
            if num_batches == 0 {
                println!("Creating tensors for first batch...");
            }
            let (imgs, lbls) = batcher.batch(batch_items);
            let len = lbls.dims()[0];
            if num_batches == 0 {
                println!("Tensors created, batch size: {}", len);
            }
            (imgs, lbls, len)
        }; // batch_itemsはここで解放
        
        // Forward pass, loss計算, backward passをスコープ内で実行
        let (loss_value, correct) = {
            if num_batches == 0 {
                println!("Running forward pass...");
            }
            let logits = current_model.forward(images);
            if num_batches == 0 {
                println!("Forward pass complete.");
            }
            
            let predictions = logits.clone().argmax(1).squeeze(1);
            let correct_count = predictions.equal(labels.clone()).int().sum().into_scalar().elem::<i32>();
            
            let device = logits.device();
            let loss = CrossEntropyLossConfig::new()
                .init(&device)
                .forward(logits, labels);
            
            let loss_val = loss.clone().into_scalar().elem::<f32>();
            
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &current_model);
            current_model = optimizer.step(learning_rate, current_model, grads);
            
            (loss_val, correct_count)
        }; // logits, predictions, loss, gradsはここで解放
        
        total_correct += correct as usize;
        total_samples += batch_len;
        total_loss += loss_value;
        num_batches += 1;
        
        if num_batches % 10 == 0 {
            let batch_accuracy = (correct as f32 / batch_len as f32) * 100.0;
            println!(
                "Epoch {}, Batch {}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch, num_batches, loss_value, batch_accuracy
            );
        }
    }
    
    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f32
    } else {
        0.0
    };
    
    let avg_accuracy = if total_samples > 0 {
        (total_correct as f32 / total_samples as f32) * 100.0
    } else {
        0.0
    };
    
    (current_model, avg_loss, avg_accuracy)
}

fn evaluate_with_indices_no_grad(
    model: &ResNet<MyBackend>,
    dataset: &ImageFolderDataset,
    indices: &[usize],
    batcher: &Batcher,
    batch_size: usize,
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_samples = 0;
    let mut num_batches = 0;
    
    let dataset_len = indices.len();
    
    for batch_start in (0..dataset_len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset_len);
        
        if num_batches % 10 == 0 && num_batches > 0 {
            println!("Evaluation batch {}/{}", num_batches, (dataset_len + batch_size - 1) / batch_size);
        }
        
        // バッチデータを読み込む
        let mut batch_items = Vec::new();
        for i in batch_start..batch_end {
            let idx = indices[i];
            if let Some(item) = dataset.get(idx) {
                batch_items.push(item);
            }
        }
        
        if batch_items.is_empty() {
            continue;
        }
        
        let (images, labels) = batcher.batch_no_grad(batch_items);
        let batch_len = labels.dims()[0];
        
        // Forward pass（素のバックエンド）
        let logits = model.forward(images);
        
        // 精度の計算
        let predictions = logits.clone().argmax(1).squeeze(1);
        let correct = predictions.equal(labels.clone()).int().sum().into_scalar();
        total_correct += correct as usize;
        total_samples += batch_len;
        
        // Loss計算
        let device = logits.device();
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, labels);
        
        let loss_value = loss.into_scalar();
        total_loss += loss_value;
        num_batches += 1;
    }
    
    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f32
    } else {
        0.0
    };
    
    let avg_accuracy = if total_samples > 0 {
        (total_correct as f32 / total_samples as f32) * 100.0
    } else {
        0.0
    };
    
    (avg_loss, avg_accuracy)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // コマンドライン引数から事前学習モデルのパスを取得
    let pretrained_path = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };
    
    // Configuration
    let image_size = 224;
    let batch_size = 16;
    let num_epochs = 20;
    // 学習率の設定
    // - 事前学習モデルの場合: 小さい学習率 (0.0001) を推奨
    // - ランダム初期化の場合: やや大きい学習率 (0.001) を推奨
    let learning_rate = 0.0001;
    let cifar10_classes = 10;
    let pretrain_classes = 1000;
    
    let device = WgpuDevice::default();
    println!("Using GPU device: {:?}", device);
    
    // CIFAR-10データセットの読み込み（ImageFolder形式）
    println!("Loading CIFAR-10 dataset...");
    let train_dataset = ImageFolderDataset::new("cifar-10-images", image_size);
    let train_len = train_dataset.len();
    
    println!("Train size: {}", train_len);
    
    // テストデータは訓練データと混在しているため、後で分割するか
    // または別途test用のフォルダを作成する必要があります
    // 今回は訓練データの一部を評価に使用
    let test_len = 1000.min(train_len); // 1000サンプルに制限（メモリ節約）
    
    // モデルの作成とヘッドの置き換え
    let mut model = if let Some(ref path) = pretrained_path {
        println!("Loading pretrained model from: {}", path);
        let recorder = CompactRecorder::new();
        // 一時的に大きなモデルをロードするため、スコープで管理
        let loaded_model = {
            match ResNet::<MyAutodiffBackend>::resnet50(pretrain_classes, &device)
                .load_file(path, &recorder, &device)
            {
                Ok(loaded_model) => {
                    println!("Successfully loaded pretrained model");
                    Some(loaded_model)
                }
                Err(e) => {
                    eprintln!("Failed to load pretrained model: {:?}", e);
                    println!("Starting with randomly initialized model");
                    None
                }
            }
        };
        
        if let Some(loaded_model) = loaded_model {
            loaded_model.replace_head(cifar10_classes, &device)
        } else {
            ResNet::<MyAutodiffBackend>::resnet50(cifar10_classes, &device)
        }
    } else {
        println!("No pretrained model specified, starting from scratch");
        ResNet::<MyAutodiffBackend>::resnet50(cifar10_classes, &device)
    };
    
    // オプティマイザーの作成
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .init();
    
    // Batcherの作成
    let batcher = Batcher::new(device.clone());
    
    // チェックポイントディレクトリ
    let checkpoint_dir = "cifar10_checkpoints";
    fs::create_dir_all(checkpoint_dir).expect("Failed to create checkpoint directory");
    
    // CSV出力ファイルの準備
    let csv_filename = if pretrained_path.is_some() {
        "cifar10_finetuning_pretrained.csv"
    } else {
        "cifar10_finetuning_scratch.csv"
    };
    let mut csv_file = File::create(csv_filename).expect("Failed to create CSV file");
    writeln!(csv_file, "epoch,train_loss,train_accuracy,test_loss,test_accuracy").expect("Failed to write CSV header");
    println!("Logging results to: {}", csv_filename);
    
    println!("Starting fine-tuning with learning rate: {}", learning_rate);
    
    for epoch in 1..=num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch, num_epochs);
        
        println!("Shuffling dataset...");
        // データセット全体をシャッフル
        let mut all_indices: Vec<usize> = (0..train_len).collect();
        let mut rng = rand::thread_rng();
        all_indices.shuffle(&mut rng);
        println!("Shuffle complete.");
        
        // 最初の80%を訓練、残り20%をテストに使用
        let train_split = train_len - test_len;
        let train_indices = &all_indices[..train_split];
        let test_indices = &all_indices[train_split..];
        
        println!("Starting training epoch...");
        // 訓練（バッチごとに画像を読み込む）
        let (updated_model, train_loss, train_acc) = train_epoch(
            model,
            &train_dataset,
            train_indices,
            &batcher,
            &mut optimizer,
            epoch,
            learning_rate,
            batch_size,
        );
        
        model = updated_model;
        
        println!("Epoch {} - Train Loss: {:.4}, Train Accuracy: {:.2}%", epoch, train_loss, train_acc);
        
        // 評価前にGPUメモリを解放するため、モデルを保存
        println!("Preparing model for evaluation...");
        let recorder = CompactRecorder::new();
        let temp_path = format!("{}/temp_eval_epoch_{}.mpk", checkpoint_dir, epoch);
        model.clone().save_file(&temp_path, &recorder).expect("Failed to save temp model");
        
        // 訓練モデルを明示的にdropしてGPUメモリ解放
        drop(model);
        println!("Training model dropped, loading evaluation model...");
        
        // 素のバックエンドモデルをロードして評価
        let (test_loss, test_acc) = {
            let eval_model = ResNet::<MyBackend>::resnet50(cifar10_classes, &device)
                .load_file(&temp_path, &recorder, &device)
                .expect("Failed to load eval model");
            
            println!("Evaluating on test set...");
            let result = evaluate_with_indices_no_grad(&eval_model, &train_dataset, test_indices, &batcher, batch_size);
            result
        };
        
        // 評価完了後、次のエポックのために訓練モデルを再ロード
        println!("Reloading training model...");
        model = ResNet::<MyAutodiffBackend>::resnet50(cifar10_classes, &device)
            .load_file(&temp_path, &recorder, &device)
            .expect("Failed to reload training model");
        
        let _ = fs::remove_file(&temp_path); // 一時ファイルを削除
        println!("Epoch {} - Test Loss: {:.4}, Test Accuracy: {:.2}%", epoch, test_loss, test_acc);
        
        // CSV出力
        writeln!(csv_file, "{},{:.6},{:.4},{:.6},{:.4}", epoch, train_loss, train_acc, test_loss, test_acc).expect("Failed to write to CSV");
        csv_file.flush().expect("Failed to flush CSV file");
        
        // チェックポイント保存
        if epoch % 10 == 0 {
            let checkpoint_path = format!("{}/model_epoch_{}.mpk", checkpoint_dir, epoch);
            let recorder = CompactRecorder::new();
            if let Err(e) = model.clone().save_file(&checkpoint_path, &recorder) {
                eprintln!("Failed to save checkpoint: {:?}", e);
            } else {
                println!("Checkpoint saved: {}", checkpoint_path);
            }
        }
    }
    
    println!("\nFine-tuning completed!");
}
