use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use image::{ImageBuffer, Rgb};

fn main() {
    let cifar_dir = "cifar-10-batches-bin";
    let output_dir = "cifar-10-images";
    
    // クラス名
    let class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];
    
    println!("Converting CIFAR-10 to image folder format...");
    
    // 出力ディレクトリ作成
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    
    // 各クラスのディレクトリ作成
    for class_name in &class_names {
        let class_dir = Path::new(&output_dir).join(class_name);
        fs::create_dir_all(&class_dir).expect("Failed to create class directory");
    }
    
    // 訓練データの変換
    println!("Converting training data...");
    let mut train_counts = vec![0usize; 10];
    for i in 1..=5 {
        let path = Path::new(cifar_dir).join(format!("data_batch_{}.bin", i));
        if path.exists() {
            convert_batch(&path, &output_dir, &class_names, &mut train_counts);
            println!("  Converted data_batch_{}.bin", i);
        }
    }
    
    // テストデータの変換
    println!("Converting test data...");
    let mut test_counts = vec![0usize; 10];
    let test_path = Path::new(cifar_dir).join("test_batch.bin");
    if test_path.exists() {
        convert_batch(&test_path, &output_dir, &class_names, &mut test_counts);
        println!("  Converted test_batch.bin");
    }
    
    println!("\nConversion complete!");
    println!("Training images per class:");
    for (i, count) in train_counts.iter().enumerate() {
        println!("  {}: {}", class_names[i], count);
    }
    
    println!("Total training images: {}", train_counts.iter().sum::<usize>());
}

fn convert_batch(
    path: &Path,
    output_dir: &str,
    class_names: &[&str],
    counts: &mut [usize],
) {
    let mut file = File::open(path).expect("Failed to open batch file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read batch file");
    
    const ENTRY_SIZE: usize = 3073; // 1 byte label + 3072 bytes image
    
    for chunk in buffer.chunks_exact(ENTRY_SIZE) {
        let label = chunk[0] as usize;
        let image_data = &chunk[1..];
        
        // PNG形式で保存
        let class_dir = Path::new(output_dir).join(class_names[label]);
        let image_path = class_dir.join(format!("img_{:05}.png", counts[label]));
        
        save_as_png(&image_path, image_data);
        counts[label] += 1;
    }
}

fn save_as_png(path: &Path, raw_image: &[u8]) {
    // CIFAR-10のデータ形式: [R0..R1023, G0..G1023, B0..B1023]
    // ImageBufferの形式: [R0,G0,B0, R1,G1,B1, ...]
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(32, 32);
    
    for y in 0u32..32u32 {
        for x in 0u32..32u32 {
            let idx = (y * 32 + x) as usize;
            let r = raw_image[idx];
            let g = raw_image[idx + 1024];
            let b = raw_image[idx + 2048];
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    
    img.save(path).expect("Failed to save image");
}
