//! Vision feature injection for Rune's packed Gemma4 decision path.
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma4::{
    config::Gemma4VisionConfig, multimodal_embedding::MultimodalEmbedder, vision::VisionTower,
};
use text_embeddings_backend_core::Batch;

pub(super) struct Gemma4Vision {
    tower: VisionTower,
    projection: MultimodalEmbedder,
}

impl Gemma4Vision {
    pub(super) fn load(
        vb: VarBuilder,
        config: &serde_json::Value,
        hidden_size: usize,
    ) -> Result<Self> {
        if config["use_clipped_linears"].as_bool().unwrap_or(false) {
            candle::bail!("Clipped Gemma4 vision projections are unsupported");
        }
        let config: Gemma4VisionConfig =
            serde_json::from_value(config.clone()).map_err(candle::Error::wrap)?;
        if config.patch_size != 16 || config.pooling_kernel_size != 3 {
            candle::bail!("Unsupported Gemma4 vision patch layout");
        }
        // Transformers wraps even unclipped vision projections in `linear`.
        let vision_vb = vb.clone().rename_f(|name| {
            if name.starts_with("model.vision_tower.encoder.") && name.ends_with("_proj.weight") {
                format!("{}.linear.weight", name.trim_end_matches(".weight"))
            } else {
                name.to_owned()
            }
        });
        Ok(Self {
            tower: VisionTower::new(&config, vision_vb.pp("model.vision_tower"))?,
            projection: MultimodalEmbedder::new(
                config.hidden_size,
                hidden_size,
                config.rms_norm_eps,
                vb.pp("model.embed_vision"),
            )?,
        })
    }

    pub(super) fn inject(
        &self,
        batch: &Batch,
        embeddings: &mut Tensor,
    ) -> Result<Vec<(usize, usize, usize)>> {
        if batch.multimodal.len() != batch.len() {
            candle::bail!("Invalid Gemma4 image batch metadata");
        }
        let mut spans = Vec::new();
        let mut features_by_image = std::collections::HashMap::new();
        for (index, media) in batch.multimodal.iter().enumerate() {
            let Some(media) = media else { continue };
            let offset = batch.cumulative_seq_lengths[index] as usize;
            let length = (batch.cumulative_seq_lengths[index + 1]
                - batch.cumulative_seq_lengths[index]) as usize;
            let mut previous_end = 0;
            for (start, image) in &media.images {
                let [t, h, w] = image.grid_thw;
                if t != 1
                    || h == 0
                    || w == 0
                    || h % 3 != 0
                    || w % 3 != 0
                    || image.merge_size != 3
                    || image.patch_dim != 768
                    || image.pixels.len() != h * w * 768
                    || *start < previous_end
                    || start + image.token_count() > length
                {
                    candle::bail!("Invalid Gemma4 image patches");
                }
                let features =
                    if let Some(features) = features_by_image.get(&std::sync::Arc::as_ptr(image)) {
                        Tensor::clone(features)
                    } else {
                        let pv = Tensor::from_slice(
                            &image.pixels,
                            (1, h, w, 16, 16, 3),
                            embeddings.device(),
                        )?
                        .permute((0, 5, 1, 3, 2, 4))?
                        .contiguous()?
                        .reshape((1, 3, h * 16, w * 16))?;
                        let features = self
                            .projection
                            .forward(&self.tower.forward(&[pv])?)?
                            .squeeze(0)?;
                        if features.dim(0)? != image.token_count() {
                            candle::bail!("Gemma4 vision token count mismatch");
                        }
                        features_by_image.insert(std::sync::Arc::as_ptr(image), features.clone());
                        features
                    };
                *embeddings = embeddings.slice_scatter0(&features, offset + start)?;
                spans.push((offset + start, image.token_count(), offset));
                previous_end = start + image.token_count();
            }
        }
        Ok(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use std::{collections::BTreeSet, path::PathBuf};

    #[test]
    #[ignore = "requires Rune checkpoint, Transformers pixel/features fixtures, and CUDA"]
    fn rune_vision_matches_reference() -> anyhow::Result<()> {
        let root = PathBuf::from(std::env::var("RUNE_CHECKPOINT_DIR")?);
        let fixture = PathBuf::from(std::env::var("RUNE_IMAGE_FIXTURE_DIR")?);
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json"))?)?;
        let paths: BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| root.join(v.as_str().unwrap()))
            .collect();
        let device = Device::new_cuda(0)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &paths.into_iter().collect::<Vec<_>>(),
                DType::BF16,
                &device,
            )?
        };
        let vision = Gemma4Vision::load(
            vb,
            &config["vision_config"],
            config["text_config"]["hidden_size"].as_u64().unwrap() as usize,
        )?;
        for name in ["red", "green", "blue"] {
            let input = candle::safetensors::load(
                fixture.join(format!("{name}.inputs.safetensors")),
                &device,
            )?;
            let positions = input["image_position_ids"]
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?;
            let w = positions.chunks_exact(2).map(|p| p[0]).max().unwrap() as usize + 1;
            let h = positions.chunks_exact(2).map(|p| p[1]).max().unwrap() as usize + 1;
            let pixels = input["pixel_values"]
                .narrow(1, 0, h * w)?
                .reshape((1, h, w, 16, 16, 3))?
                .permute((0, 5, 1, 3, 2, 4))?
                .contiguous()?
                .reshape((1, 3, h * 16, w * 16))?;
            let actual = vision
                .projection
                .forward(&vision.tower.forward(&[pixels])?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let expected = candle::safetensors::load(
                fixture.join(format!("{name}.outputs.safetensors")),
                &device,
            )?
            .remove("features")
            .unwrap();
            let a = actual.flatten_all()?.to_vec1::<f32>()?;
            let b = expected.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a.len(), b.len());
            let dot: f64 = a
                .iter()
                .zip(&b)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norm = |x: &[f32]| x.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(&a) * norm(&b));
            println!("{name}: vision feature cosine {cosine}");
            assert!(cosine > 0.995, "{name}: {cosine}");
        }
        Ok(())
    }
}
