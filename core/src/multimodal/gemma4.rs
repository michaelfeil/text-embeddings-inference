//! Rune/Gemma4 image inputs: bounded fetching, aspect-ratio resize and HWC patches.
use super::{
    image::{resize_rgb, QwenImageProcessor},
    media::{MediaBudget, MediaResolver},
    MultimodalConfig,
};
use crate::TextEmbeddingsError;
use image::ImageDecoder;
use std::{path::Path, sync::Arc};
use text_embeddings_backend::ImagePatches;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

type Result<T> = std::result::Result<T, TextEmbeddingsError>;
fn invalid(value: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(value.into())
}

pub struct PreparedGemmaImages {
    pub images: Vec<Arc<ImagePatches>>,
    pub memory: Arc<OwnedSemaphorePermit>,
}

#[derive(Clone)]
pub struct Gemma4ImageProcessor {
    resolver: MediaResolver,
    budget: MediaBudget,
    workers: Arc<Semaphore>,
    config: MultimodalConfig,
    max_soft_tokens: usize,
}

impl Gemma4ImageProcessor {
    pub fn load(root: &Path, config: MultimodalConfig) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("processor_config.json"))
                .map_err(|e| invalid(e.to_string()))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let image = &value["image_processor"];
        let max_soft_tokens = image["max_soft_tokens"].as_u64().unwrap_or(0) as usize;
        if image["patch_size"] != 16
            || image["pooling_kernel_size"] != 3
            || ![70, 140, 280, 560, 1120].contains(&max_soft_tokens)
            || image["resample"] != 3
            || image["do_resize"] != true
            || image["do_rescale"] != true
            || image["do_normalize"] != false
            || image["do_convert_rgb"] != true
            || image["rescale_factor"].as_f64() != Some(1.0 / 255.0)
            || config.workers == 0
            || config.workers > 32
            || config.max_images == 0
            || config.max_images > 32
            || config.max_decoded_pixels == 0
            || config.max_decoded_pixels > 16_777_216
        {
            return Err(invalid("Unsupported Gemma4 image processor configuration"));
        }
        let budget = MediaBudget::new(config.memory_budget_bytes)?;
        let resolver = MediaResolver::new(
            config.allowed_image_hosts.clone(),
            budget.clone(),
            config.max_image_bytes,
            config.download_concurrency,
            config.timeout,
        )?;
        Ok(Self {
            resolver,
            budget,
            workers: Arc::new(Semaphore::new(config.workers)),
            config,
            max_soft_tokens,
        })
    }

    pub async fn prepare(&self, sources: Vec<String>) -> Result<PreparedGemmaImages> {
        if sources.is_empty() || sources.len() > self.config.max_images {
            return Err(invalid("Invalid number of images"));
        }
        tokio::time::timeout(self.config.timeout, self.prepare_inner(sources))
            .await
            .map_err(|_| invalid("Image preprocessing timed out"))?
    }

    async fn prepare_inner(&self, sources: Vec<String>) -> Result<PreparedGemmaImages> {
        let mut resolved = Vec::with_capacity(sources.len());
        for source in sources {
            resolved.push(self.resolver.resolve(source).await?);
        }
        let worker = self.workers.clone().try_acquire_owned()?;
        let processor = self.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let _worker = worker;
            let result = (|| {
                let mut plans = Vec::new();
                let mut memory_bytes = 0usize;
                for image in &resolved {
                    if tx.is_closed() {
                        return Err(invalid("Image preprocessing canceled"));
                    }
                    let mut decoder = QwenImageProcessor::decoder(
                        &image.bytes,
                        processor.config.max_decoded_pixels,
                    )
                    .map_err(invalid)?;
                    let (mut w, mut h) = decoder.dimensions();
                    if matches!(
                        decoder
                            .orientation()
                            .map_err(|_| invalid("Invalid image orientation"))?,
                        image::metadata::Orientation::Rotate90
                            | image::metadata::Orientation::Rotate270
                            | image::metadata::Orientation::Rotate90FlipH
                            | image::metadata::Orientation::Rotate270FlipH
                    ) {
                        std::mem::swap(&mut w, &mut h);
                    }
                    let (rw, rh) = dimensions(w, h, processor.max_soft_tokens)?;
                    // Decoded/rotated input, RGB conversion, horizontal/output buffers,
                    // patch floats and a backend upload copy. Metadata uses encoded reservation.
                    memory_bytes += decoder.total_bytes() as usize * 2
                        + w as usize * h as usize * 3
                        + rw as usize * h as usize * 3
                        + rw as usize * rh as usize * (3 + 3 * 4 * 2)
                        + (w as usize + h as usize + rw as usize + rh as usize) * 128;
                    plans.push((rw, rh));
                }
                let memory = Arc::new(processor.budget.reserve(memory_bytes)?);
                let mut images = Vec::new();
                for (image, (w, h)) in resolved.into_iter().zip(plans) {
                    if tx.is_closed() {
                        return Err(invalid("Image preprocessing canceled"));
                    }
                    let rgb = resize_rgb(&image.bytes, w, h, processor.config.max_decoded_pixels)
                        .map_err(invalid)?;
                    let (gh, gw) = (h as usize / 16, w as usize / 16);
                    let mut pixels = Vec::with_capacity(rgb.len());
                    for y in 0..gh {
                        for x in 0..gw {
                            for py in 0..16 {
                                for px in 0..16 {
                                    for c in 0..3 {
                                        pixels.push(
                                            rgb[((y * 16 + py) * w as usize + x * 16 + px) * 3 + c]
                                                as f32
                                                * (1.0f32 / 255.0),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    images.push(Arc::new(ImagePatches {
                        pixels,
                        grid_thw: [1, gh, gw],
                        patch_dim: 768,
                        merge_size: 3,
                    }));
                }
                Ok(PreparedGemmaImages { images, memory })
            })();
            let _ = tx.send(result);
        });
        rx.await
            .map_err(|_| invalid("Image processing worker failed"))?
    }
}

fn dimensions(w: u32, h: u32, max_soft_tokens: usize) -> Result<(u32, u32)> {
    if w == 0 || h == 0 {
        return Err(invalid("Invalid image dimensions"));
    }
    let scale = ((max_soft_tokens * 9 * 256) as f64 / (f64::from(w) * f64::from(h))).sqrt();
    let mut rw = (f64::from(w) * scale / 48.0).floor() as u32 * 48;
    let mut rh = (f64::from(h) * scale / 48.0).floor() as u32 * 48;
    if rh == 0 {
        rh = 48;
        rw = ((w / h) * 48).min(max_soft_tokens as u32 * 48);
    }
    if rw == 0 {
        rw = 48;
        rh = ((h / w) * 48).min(max_soft_tokens as u32 * 48);
    }
    Ok((rw, rh))
}
