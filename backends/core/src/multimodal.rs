//! Prepared image tensors aligned to one token sequence.
use std::fmt;
use tokio::sync::OwnedSemaphorePermit;

#[derive(Clone)]
pub struct ImagePatches {
    pub pixels: Vec<f32>,
    pub grid_thw: [usize; 3],
    pub patch_dim: usize,
    pub merge_size: usize,
}

impl ImagePatches {
    pub fn token_count(&self) -> usize {
        self.grid_thw.iter().product::<usize>() / (self.merge_size * self.merge_size)
    }
}

impl fmt::Debug for ImagePatches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImagePatches")
            .field("grid_thw", &self.grid_thw)
            .field("patch_dim", &self.patch_dim)
            .field("merge_size", &self.merge_size)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct MultimodalEncoding {
    /// (First image token, patches), in message/content order.
    pub images: Vec<(usize, ImagePatches)>,
    pub position_ids: [Vec<u32>; 3],
    /// Keeps the processor's memory reservation alive until the last batch consumer drops it.
    pub memory: Option<OwnedSemaphorePermit>,
}
