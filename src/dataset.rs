use burn::data::dataset::Dataset;
use image::imageops::FilterType;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct ImageItem {
    pub image: Vec<f32>,
    pub label: usize,
}

pub struct ImageFolderDataset {
    items: Vec<(PathBuf, usize)>,
    image_size: usize,
}

impl ImageFolderDataset {
    pub fn new<P: AsRef<Path>>(root: P, image_size: usize) -> Self {
        let mut class_folders = Vec::new();
        
        // Collect all class folders
        for entry in std::fs::read_dir(&root).expect("Failed to read directory") {
            if let Ok(entry) = entry {
                if entry.path().is_dir() {
                    class_folders.push(entry.path());
                }
            }
        }
        
        // Sort folders to ensure consistent class ordering
        class_folders.sort();
        
        let mut items = Vec::new();
        
        // Iterate through each class folder
        for (class_idx, class_folder) in class_folders.iter().enumerate() {
            // Find all image files in the folder
            for entry in WalkDir::new(class_folder)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext = ext.to_string_lossy().to_lowercase();
                        if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "ppm" {
                            items.push((path.to_path_buf(), class_idx));
                        }
                    }
                }
            }
        }
        
        println!("Found {} images in {} classes", items.len(), class_folders.len());
        
        Self { items, image_size }
    }
    
    fn load_and_preprocess_image(&self, path: &Path) -> Vec<f32> {
        let img = image::open(path)
            .expect(&format!("Failed to open image: {:?}", path));
        
        // Resize to target size
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            FilterType::Lanczos3,
        );
        
        // Convert to RGB
        let img = img.to_rgb8();
        
        // Normalize to [0, 1] and then apply ImageNet normalization
        let mut pixels = Vec::with_capacity(3 * self.image_size * self.image_size);
        
        // ImageNet mean and std
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        
        // Convert from HWC to CHW format and normalize
        for c in 0..3 {
            for y in 0..self.image_size {
                for x in 0..self.image_size {
                    let pixel = img.get_pixel(x as u32, y as u32)[c];
                    let normalized = (pixel as f32 / 255.0 - mean[c]) / std[c];
                    pixels.push(normalized);
                }
            }
        }
        
        pixels
    }
}

impl Dataset<ImageItem> for ImageFolderDataset {
    fn get(&self, index: usize) -> Option<ImageItem> {
        if index >= self.items.len() {
            return None;
        }
        
        let (path, label) = &self.items[index];
        let image = self.load_and_preprocess_image(path);
        
        Some(ImageItem {
            image,
            label: *label,
        })
    }
    
    fn len(&self) -> usize {
        self.items.len()
    }
}
