use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{Int, Tensor, TensorData},
    data::dataloader::Dataset,
};
use burn_wgpu::{Wgpu, WgpuDevice};

mod model;
mod dataset;

use model::ResNet;
use dataset::ImageFolderDataset;

type MyBackend = Wgpu;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

struct Batcher {
    device: WgpuDevice,
}

impl Batcher {
    fn new(device: WgpuDevice) -> Self {
        Self { device }
    }
    
    fn batch(&self, items: Vec<dataset::ImageItem>) -> (Tensor<MyBackend, 4>, Tensor<MyBackend, 1, Int>) {
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
}

fn evaluate_only(
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
    println!("Starting evaluation on {} samples with batch size {}", dataset_len, batch_size);
    
    for batch_start in (0..dataset_len).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset_len);
        
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
        
        let (images, labels) = batcher.batch(batch_items);
        let batch_len = labels.dims()[0];
        
        // Forward pass
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
        
        if num_batches % 100 == 0 {
            let batch_accuracy = (correct as f32 / batch_len as f32) * 100.0;
            println!("Batch {}: Loss = {:.4}, Accuracy = {:.2}", num_batches, loss_value, batch_accuracy);
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
    
    (avg_loss, avg_accuracy)
}

fn main() {
    let image_size = 224;
    let batch_size = 16;
    let cifar10_classes = 10;
    
    let device = WgpuDevice::default();
    println!("Using GPU device: {:?}", device);
    
    // CIFAR-10データセットの読み込み
    println!("Loading CIFAR-10 dataset...");
    let train_dataset = ImageFolderDataset::new("cifar-10-images", image_size);
    let train_len = train_dataset.len();
    
    println!("Dataset loaded: {} samples", train_len);
    
    // テストデータのインデックス（最初の1000サンプルのみ使用）
    let test_len = 1000.min(train_len);
    let test_indices: Vec<usize> = (0..test_len).collect();
    
    // モデルの作成（スクラッチ、評価専用なのでAutodiffなし）
    println!("Creating model...");
    let model = ResNet::<MyBackend>::resnet50(cifar10_classes, &device);
    println!("Model created.");
    
    // Batcherの作成
    let batcher = Batcher::new(device.clone());
    
    // 評価実行
    println!("Starting evaluation...");
    let (test_loss, test_acc) = evaluate_only(&model, &train_dataset, &test_indices, &batcher, batch_size);
    println!("Evaluation complete!");
    println!("Test Loss: {:.4}, Test Accuracy: {:.2}%", test_loss, test_acc);
}
