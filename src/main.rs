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
use std::fs;
use std::path::Path;

type MyBackend = Wgpu;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

struct Batcher {
    device: WgpuDevice,
}

impl Batcher {
    fn new(device: WgpuDevice) -> Self {
        Self { device }
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
    
    for batch_start in (0..dataset_len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset_len);
        let mut batch_items = Vec::new();
        
        // バッチごとに画像を読み込む
        for i in batch_start..batch_end {
            let idx = indices[i];
            if let Some(item) = dataset.get(idx) {
                batch_items.push(item);
            }
        }
        
        if batch_items.is_empty() {
            continue;
        }
        
        let (images, labels) = batcher.batch(batch_items);
        let batch_len = labels.dims()[0];
        let logits = current_model.forward(images);
        
        // 精度の計算
        let predictions = logits.clone().argmax(1).squeeze(1);
        let correct = predictions.equal(labels.clone()).int().sum().into_scalar().elem::<i32>();
        total_correct += correct as usize;
        total_samples += batch_len;
        
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), labels.clone());
        
        let loss_value = loss.clone().into_scalar().elem::<f32>();
        total_loss += loss_value;
        num_batches += 1;
        
        let grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(grads, &current_model);
        current_model = optimizer.step(learning_rate, current_model, grads);
        
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

fn find_latest_checkpoint(checkpoint_dir: &str) -> Option<(String, usize)> {
    let path = Path::new(checkpoint_dir);
    if !path.exists() {
        return None;
    }
    
    let mut checkpoints = Vec::new();
    
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.starts_with("model_epoch_") && filename.ends_with(".mpk") {
                    // model_epoch_XX.mpk から XX を抽出
                    if let Some(epoch_str) = filename
                        .strip_prefix("model_epoch_")
                        .and_then(|s| s.strip_suffix(".mpk"))
                    {
                        if let Ok(epoch_num) = epoch_str.parse::<usize>() {
                            checkpoints.push((entry.path().to_string_lossy().to_string(), epoch_num));
                        }
                    }
                }
            }
        }
    }
    
    checkpoints.sort_by_key(|(_, epoch)| *epoch);
    checkpoints.last().cloned()
}

fn save_checkpoint(
    model: &ResNet<MyAutodiffBackend>,
    epoch: usize,
    checkpoint_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_path = format!("{}/model_epoch_{}.mpk", checkpoint_dir, epoch);
    
    // モデルをバイナリ形式で保存
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(checkpoint_path.as_str(), &recorder)?;
    
    println!("Checkpoint saved: epoch {} at {}", epoch, checkpoint_path);
    Ok(())
}

fn load_checkpoint(
    checkpoint_path: &str,
    device: &WgpuDevice,
    num_classes: usize,
) -> Result<ResNet<MyAutodiffBackend>, Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    let model = ResNet::<MyAutodiffBackend>::resnet50(num_classes, device)
        .load_file(checkpoint_path, &recorder, device)?;
    
    println!("Model loaded from: {}", checkpoint_path);
    Ok(model)
}

fn main() {
    // Configuration
    let image_size = 224;
    let batch_size = 16;
    let num_epochs = 500;
    let num_classes = 1000;
    let learning_rate = 0.0001; // 学習率を明示的に指定
    let checkpoint_dir = "checkpoints";
    let checkpoint_interval = 10; // エポックごとにチェックポイントを保存
    
    let device = WgpuDevice::default();
    println!("Using GPU device: {:?}", device);
    
    // チェックポイントディレクトリを作成
    fs::create_dir_all(checkpoint_dir).expect("Failed to create checkpoint directory");
    
    println!("Loading dataset...");
    let dataset = ImageFolderDataset::new("2D-OFDB-1k", image_size);
    let dataset_len = dataset.len();
    
    println!("Dataset size: {}", dataset_len);
    
    // チェックポイントから再開するかチェック
    let latest_checkpoint = find_latest_checkpoint(checkpoint_dir);
    let (mut model, start_epoch) = if let Some((checkpoint_path, epoch_num)) = latest_checkpoint {
        println!("Found checkpoint: {}, resuming from epoch {}", checkpoint_path, epoch_num + 1);
        match load_checkpoint(&checkpoint_path, &device, num_classes) {
            Ok(loaded_model) => {
                println!("Successfully loaded model from checkpoint");
                (loaded_model, epoch_num + 1)
            }
            Err(e) => {
                eprintln!("Failed to load checkpoint: {:?}", e);
                println!("Starting with a new model instead");
                let model = ResNet::<MyAutodiffBackend>::resnet50(num_classes, &device);
                (model, 1)
            }
        }
    } else {
        println!("No checkpoint found, starting from scratch");
        let model = ResNet::<MyAutodiffBackend>::resnet50(num_classes, &device);
        (model, 1)
    };
    
    // Create optimizer
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .init();
    
    // Create batcher
    let batcher = Batcher::new(device.clone());
    
    println!("Starting training with learning rate: {}", learning_rate);
    for epoch in start_epoch..=num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch, num_epochs);
        
        // データのインデックスをシャッフル
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        // 訓練（バッチごとに画像を読み込む）
        let (updated_model, avg_loss, avg_accuracy) = train_epoch(
            model,
            &dataset,
            &indices,
            &batcher,
            &mut optimizer,
            epoch,
            learning_rate,
            batch_size,
        );
        
        model = updated_model;
        
        println!("Epoch {} - Average Loss: {:.4}, Average Accuracy: {:.2}%", epoch, avg_loss, avg_accuracy);
        
        // Save checkpoint every checkpoint_interval epochs
        if epoch % checkpoint_interval == 0 {
            if let Err(e) = save_checkpoint(&model, epoch, checkpoint_dir) {
                eprintln!("Failed to save checkpoint at epoch {}: {:?}", epoch, e);
            }
        }
    }
    
    println!("\nTraining completed!");
}
