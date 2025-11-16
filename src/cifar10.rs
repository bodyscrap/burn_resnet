use burn::data::dataset::Dataset;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Cifar10Item {
    pub image: Vec<f32>,
    pub label: usize,
}

pub struct Cifar10Dataset {
    // ファイルの内容を一度メモリに読み込んでキャッシュ
    file_data: Vec<Vec<u8>>,
    indices: Vec<(usize, usize)>, // (file_index, offset_in_file)
    labels: Vec<usize>,
    image_size: usize,
}

impl Cifar10Dataset {
    pub fn train<P: AsRef<Path>>(root: P, image_size: usize) -> Self {
        let mut file_data = Vec::new();
        let mut indices = Vec::new();
        let mut labels = Vec::new();
        
        println!("Loading CIFAR-10 training data into memory...");
        
        // CIFAR-10のtrainバッチファイルをメモリに読み込む
        for i in 1..=5 {
            let path = root.as_ref().join(format!("data_batch_{}.bin", i));
            if path.exists() {
                if let Ok(mut file) = File::open(&path) {
                    let mut buffer = Vec::new();
                    if file.read_to_end(&mut buffer).is_ok() {
                        let batch_labels = extract_labels_from_buffer(&buffer);
                        let file_index = file_data.len();
                        file_data.push(buffer);
                        
                        for (offset, label) in batch_labels.into_iter().enumerate() {
                            indices.push((file_index, offset));
                            labels.push(label);
                        }
                    }
                }
            }
        }
        
        println!("Loaded {} training images from CIFAR-10", labels.len());
        
        Self { file_data, indices, labels, image_size }
    }
    
    pub fn test<P: AsRef<Path>>(root: P, image_size: usize) -> Self {
        let mut file_data = Vec::new();
        let mut indices = Vec::new();
        let mut labels = Vec::new();
        
        println!("Loading CIFAR-10 test data into memory...");
        
        // CIFAR-10のtestバッチファイルをメモリに読み込む
        let path = root.as_ref().join("test_batch.bin");
        if path.exists() {
            if let Ok(mut file) = File::open(&path) {
                let mut buffer = Vec::new();
                if file.read_to_end(&mut buffer).is_ok() {
                    let batch_labels = extract_labels_from_buffer(&buffer);
                    file_data.push(buffer);
                    
                    for (offset, label) in batch_labels.into_iter().enumerate() {
                        indices.push((0, offset));
                        labels.push(label);
                    }
                }
            }
        }
        
        println!("Loaded {} test images from CIFAR-10", labels.len());
        
        Self { file_data, indices, labels, image_size }
    }
    
    fn preprocess_image(&self, raw_image: &[u8]) -> Vec<f32> {
        // CIFAR-10画像は32x32x3のRGB画像
        // バイナリ形式: [R0, R1, ..., R1023, G0, G1, ..., G1023, B0, B1, ..., B1023]
        
        let mut pixels = Vec::with_capacity(3 * self.image_size * self.image_size);
        
        // ImageNet標準の正規化
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        
        // 32x32から指定サイズにリサイズ (簡易版: 最近傍補間)
        let scale = 32.0 / self.image_size as f32;
        
        for c in 0..3 {
            for y in 0..self.image_size {
                for x in 0..self.image_size {
                    let src_x = (x as f32 * scale) as usize;
                    let src_y = (y as f32 * scale) as usize;
                    let src_idx = c * 1024 + src_y * 32 + src_x;
                    
                    let pixel = raw_image[src_idx];
                    let normalized = (pixel as f32 / 255.0 - mean[c]) / std[c];
                    pixels.push(normalized);
                }
            }
        }
        
        pixels
    }
}

impl Dataset<Cifar10Item> for Cifar10Dataset {
    fn get(&self, index: usize) -> Option<Cifar10Item> {
        if index >= self.labels.len() {
            return None;
        }
        
        let (file_index, offset) = self.indices[index];
        let label = self.labels[index];
        
        // メモリ内のデータから該当する画像データを取得
        let raw_image = extract_image_from_buffer(&self.file_data[file_index], offset)?;
        let image = self.preprocess_image(&raw_image);
        
        Some(Cifar10Item {
            image,
            label,
        })
    }
    
    fn len(&self) -> usize {
        self.labels.len()
    }
}

// バッファからラベルを抽出
fn extract_labels_from_buffer(buffer: &[u8]) -> Vec<usize> {
    let mut labels = Vec::new();
    const ENTRY_SIZE: usize = 3073;
    
    for chunk in buffer.chunks_exact(ENTRY_SIZE) {
        labels.push(chunk[0] as usize);
    }
    
    labels
}

// バッファから特定の画像データを抽出
fn extract_image_from_buffer(buffer: &[u8], offset: usize) -> Option<Vec<u8>> {
    const ENTRY_SIZE: usize = 3073;
    let start = offset * ENTRY_SIZE + 1; // +1でラベルをスキップ
    let end = start + 3072; // 32x32x3 = 3072バイト
    
    if end <= buffer.len() {
        Some(buffer[start..end].to_vec())
    } else {
        None
    }
}
