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
    dtype: candle::DType,
}

impl Gemma4Vision {
    pub(super) fn load(
        vb: VarBuilder,
        config: &serde_json::Value,
        hidden_size: usize,
    ) -> Result<Self> {
        let config: Gemma4VisionConfig =
            serde_json::from_value(config.clone()).map_err(candle::Error::wrap)?;
        if config.patch_size != 16 || config.pooling_kernel_size != 3 {
            candle::bail!("Unsupported Gemma4 vision patch layout");
        }
        Ok(Self {
            tower: VisionTower::new(&config, vb.pp("model.vision_tower"))?,
            projection: MultimodalEmbedder::new(
                config.hidden_size,
                hidden_size,
                config.rms_norm_eps,
                vb.pp("model.embed_vision"),
            )?,
            dtype: vb.dtype(),
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
        let mut pixels = Vec::new();
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
                let pv =
                    Tensor::from_slice(&image.pixels, (1, h, w, 16, 16, 3), embeddings.device())?
                        .permute((0, 5, 1, 3, 2, 4))?
                        .contiguous()?
                        .reshape((1, 3, h * 16, w * 16))?
                        .to_dtype(self.dtype)?;
                pixels.push(pv);
                spans.push((offset + start, image.token_count(), offset));
                previous_end = start + image.token_count();
            }
        }
        let features = self
            .projection
            .forward(&self.tower.forward(&pixels)?)?
            .squeeze(0)?;
        let mut feature_offset = 0;
        for &(start, length, _) in &spans {
            *embeddings =
                embeddings.slice_scatter0(&features.narrow(0, feature_offset, length)?, start)?;
            feature_offset += length;
        }
        if feature_offset != features.dim(0)? {
            candle::bail!("Gemma4 vision token count mismatch");
        }
        Ok(spans)
    }
}
