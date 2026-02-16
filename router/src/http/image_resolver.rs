use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use image::imageops::FilterType;
use ndarray::{Array, Dim};
use reqwest::Client;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ImageResolutionError {
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),
    #[error("Failed to download image: {0}")]
    DownloadError(String),
    #[error("Failed to process image: {0}")]
    ProcessingError(#[from] image::ImageError),
    #[error("Base64 decoding error: {0}")]
    Base64Error(#[from] base64::DecodeError),
}

#[derive(Clone)]
pub struct ImageResolver {
    client: Client,
}

impl ImageResolver {
    pub fn new() -> Self {
        let client = Client::new();
        
        Self {
            client,
        }
    }

    pub async fn resolve_image(&self, image_source: &str) -> Result<Array<f32, Dim<[usize; 4]>>, ImageResolutionError> {
        if image_source.starts_with("base64:") {
            self.resolve_base64_image(&image_source[7..]).await
        } else if image_source.starts_with("url:") {
            self.resolve_url_image(&image_source[4..]).await
        } else {
            Err(ImageResolutionError::InvalidFormat(
                "Image source must start with 'base64:' or 'url:'".to_string()
            ))
        }
    }

    async fn resolve_base64_image(&self, base64_data: &str) -> Result<Array<f32, Dim<[usize; 4]>>, ImageResolutionError> {
        let decoded_data = general_purpose::STANDARD.decode(base64_data)?;
        let image = image::load_from_memory(&decoded_data)?;
        
        Ok(self.process_image(image)?)
    }

    async fn resolve_url_image(&self, url: &str) -> Result<Array<f32, Dim<[usize; 4]>>, ImageResolutionError> {
        let response = self.client.get(url).send().await
            .map_err(|e| ImageResolutionError::DownloadError(e.to_string()))?;
        
        let image_data = response.bytes().await
            .map_err(|e| ImageResolutionError::DownloadError(e.to_string()))?;
        
        let image = image::load_from_memory(&image_data)?;
        
        Ok(self.process_image(image)?)
    }

    fn process_image(&self, image: image::DynamicImage) -> Result<Array<f32, Dim<[usize; 4]>>, ImageResolutionError> {
        // Resize to standard size (e.g., 224x224 for CLIP models)
        let resized = image.resize_to_fill(224, 224, FilterType::Lanczos3);
        
        // Convert to RGB if not already
        let rgb_image = resized.to_rgb8();
        
        // Convert to float array and normalize to [0, 1]
        let (width, height) = rgb_image.dimensions();
        let mut array = Array::zeros((1, 3, height as usize, width as usize));
        
        for (c, y, x) in ndarray::indices((3, height as usize, width as usize)) {
            let pixel = rgb_image.get_pixel(x as u32, y as u32);
            let value = pixel[c] as f32 / 255.0;
            array[[0, c, y, x]] = value;
        }
        
        Ok(array)
    }
}

impl Default for ImageResolver {
    fn default() -> Self {
        Self::new()
    }
}