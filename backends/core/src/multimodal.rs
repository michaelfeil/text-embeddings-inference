//! Prepared image tensors aligned to one token sequence.
use std::{fmt, sync::Arc};
use tokio::sync::OwnedSemaphorePermit;

#[derive(Clone)]
pub struct ImagePatches {
    pub pixels: Vec<f32>,
    pub grid_thw: [usize; 3],
    pub patch_dim: usize,
    pub merge_size: usize,
}

impl ImagePatches {
    fn same_content(&self, other: &Self) -> bool {
        self.grid_thw == other.grid_thw
            && self.patch_dim == other.patch_dim
            && self.merge_size == other.merge_size
            && self.pixels.len() == other.pixels.len()
            && self
                .pixels
                .iter()
                .zip(&other.pixels)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

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
    pub images: Vec<(usize, Arc<ImagePatches>)>,
    pub position_ids: [Vec<u32>; 3],
    /// Keeps the processor's memory reservation alive until the last batch consumer drops it.
    pub memory: Option<Arc<OwnedSemaphorePermit>>,
}

impl MultimodalEncoding {
    /// Token-only folding is safe for images only when every sequence has the
    /// same prepared pixel bits/layout and context through the final image block.
    /// Equal placeholder IDs alone are insufficient.
    pub fn allows_radix(media: &[Option<Arc<Self>>], ids: &[u32], cumulative: &[u32]) -> bool {
        let Some(first) = media.iter().flatten().find(|m| !m.images.is_empty()) else {
            return true;
        };
        if cumulative.len() != media.len() + 1 {
            return false;
        }
        let end = first
            .images
            .iter()
            .map(|(start, image)| start + image.token_count())
            .max()
            .unwrap();
        if first
            .position_ids
            .iter()
            .any(|axis| axis.get(..end).is_none())
        {
            return false;
        }
        let mut prefix: Option<&[u32]> = None;
        for (row, item) in media.iter().enumerate() {
            let Some(item) = item else { return false };
            if item.images.len() != first.images.len()
                || !item
                    .images
                    .iter()
                    .zip(&first.images)
                    .all(|((a, x), (b, y))| a == b && (Arc::ptr_eq(x, y) || x.same_content(y)))
                || (0..3).any(|axis| {
                    item.position_ids[axis].get(..end) != first.position_ids[axis].get(..end)
                })
            {
                return false;
            }
            let start = cumulative[row] as usize;
            if start + end > cumulative[row + 1] as usize {
                return false;
            }
            let Some(current) = ids.get(start..start + end) else {
                return false;
            };
            if prefix.is_some_and(|prefix| prefix != current) {
                return false;
            }
            prefix = Some(current);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radix_requires_identical_image_content_and_complete_prefix() {
        let image = Arc::new(ImagePatches {
            pixels: vec![0.; 12],
            grid_thw: [1, 2, 2],
            patch_dim: 3,
            merge_size: 1,
        });
        let media = |image| {
            Some(Arc::new(MultimodalEncoding {
                images: vec![(1, image)],
                position_ids: std::array::from_fn(|_| (0..6).collect()),
                memory: None,
            }))
        };
        let first = media(image.clone());
        let ids = [1, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 3];
        assert!(MultimodalEncoding::allows_radix(
            &[first.clone(), media(image.clone())],
            &ids,
            &[0, 6, 12]
        ));
        assert!(MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new((*image).clone()))],
            &ids,
            &[0, 6, 12]
        ));
        let mut different = (*image).clone();
        different.pixels[11] = 1.;
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new(different))],
            &ids,
            &[0, 6, 12]
        ));
        let mut different = (*image).clone();
        different.grid_thw = [1, 1, 4];
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(Arc::new(different))],
            &ids,
            &[0, 6, 12]
        ));
        let mut changed = ids;
        changed[6] = 2;
        assert!(!MultimodalEncoding::allows_radix(
            &[first.clone(), media(image)],
            &changed,
            &[0, 6, 12]
        ));
        assert!(!MultimodalEncoding::allows_radix(
            &[first, None],
            &ids,
            &[0, 6, 12]
        ));
    }
}
