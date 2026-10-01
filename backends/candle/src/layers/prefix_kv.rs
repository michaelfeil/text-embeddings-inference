//! Single immutable causal prefix. Scratch is reused under the model's forward lock.
#[cfg(feature = "cuda")]
#[path = "prefix_kv_cuda.rs"]
mod cuda;
use candle::{
    backend::BackendStorage, CpuStorage, DType, Device, InplaceOp2, Layout, Result, Tensor,
};
use std::collections::HashMap;
use text_embeddings_backend_core::Batch;

// A bounded, contiguous destination view; Candle handles strided QKV sources.
struct CopyInto;
impl InplaceOp2 for CopyInto {
    fn name(&self) -> &'static str {
        "prefix-kv-copy"
    }
    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        src: &CpuStorage,
        sl: &Layout,
    ) -> Result<()> {
        src.copy_strided_src(dst, dl.start_offset(), sl)
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        dst: &mut candle::CudaStorage,
        dl: &Layout,
        src: &candle::CudaStorage,
        sl: &Layout,
    ) -> Result<()> {
        src.copy_strided_src(dst, dl.start_offset(), sl)
    }
}
fn copy_rows(dst: &Tensor, start: usize, src: &Tensor) -> Result<()> {
    let view = dst.narrow(0, start, src.dim(0)?)?;
    if !view.is_contiguous() || view.dims() != src.dims() {
        candle::bail!("prefix cache copy requires matching contiguous destination");
    }
    view.inplace_op2(src, &CopyInto)
}

pub(crate) struct PrefixKvCache {
    capacity: usize,
    ids: Vec<u32>,
    positions: Vec<u32>,
    layers: Vec<(Tensor, Tensor)>,
    outputs: Option<Tensor>,
    scratch: Option<(Tensor, Tensor)>,
}
pub(crate) struct PrefixPlan {
    pub suffix: Batch,
    pub kv_offsets: Tensor,
    pub max_k: usize,
    offsets: Vec<u32>,
    hits: Vec<usize>,
    fill: usize,
    #[cfg(feature = "cuda")]
    rows: Tensor,
}
impl PrefixKvCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ids: vec![],
            positions: vec![],
            layers: vec![],
            outputs: None,
            scratch: None,
        }
    }

    pub fn plan(
        &mut self,
        batch: &Batch,
        device: &Device,
        dtype: DType,
        layers: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Option<PrefixPlan>> {
        // TEI's startup warmup/health probes contain only zero IDs. Never pin them.
        if batch.is_empty()
            || batch.input_ids.iter().all(|&id| id == 0)
            || batch.multimodal.iter().any(Option::is_some)
        {
            return Ok(None);
        }
        let plan = self.make_plan(batch, device)?;
        if plan.fill == 0 && plan.hits.iter().all(|&n| n == 0) {
            return Ok(None);
        }
        if self.layers.is_empty() {
            self.layers = (0..layers)
                .map(|_| {
                    let shape = (self.capacity, heads, dim);
                    Ok((
                        Tensor::zeros(shape, dtype, device)?,
                        Tensor::zeros(shape, dtype, device)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
        }
        let tokens = batch.input_ids.len();
        if self
            .scratch
            .as_ref()
            .is_none_or(|(k, _)| k.dims()[0] < tokens)
        {
            let shape = (tokens.next_power_of_two(), heads, dim);
            self.scratch = Some((
                Tensor::zeros(shape, dtype, device)?,
                Tensor::zeros(shape, dtype, device)?,
            ));
        }
        tracing::debug!(
            cached_tokens = self.ids.len(),
            reused_tokens = plan.hits.iter().sum::<usize>(),
            suffix_tokens = plan.suffix.input_ids.len(),
            "Qwen3 prefix KV cache"
        );
        Ok(Some(plan))
    }

    fn make_plan(&self, batch: &Batch, device: &Device) -> Result<PrefixPlan> {
        let mut suffix = batch.clone();
        suffix.input_ids.clear();
        suffix.position_ids.clear();
        suffix.token_type_ids.clear();
        suffix.cumulative_seq_lengths = vec![0];
        suffix.max_length = 0;
        suffix.compact_input_ids = None;
        suffix.compact_position_ids = None;
        suffix.scatter_unfold = None;
        suffix.fold_gather = None;
        let mut hits = Vec::with_capacity(batch.len());
        let mut original_rows = Vec::new();
        for w in batch.cumulative_seq_lengths.windows(2) {
            let (start, end) = (w[0] as usize, w[1] as usize);
            if start >= end {
                candle::bail!("prefix cache requires nonempty sequences");
            }
            let hit = batch.input_ids[start..end]
                .iter()
                .zip(&batch.position_ids[start..end])
                .zip(self.ids.iter().zip(&self.positions))
                .take_while(|(a, b)| a == b)
                .count()
                .min(end - start - 1);
            // Recompute the last token even on an exact hit: no zero-length Q sequences.
            hits.push(hit);
            suffix
                .input_ids
                .extend_from_slice(&batch.input_ids[start + hit..end]);
            suffix
                .position_ids
                .extend_from_slice(&batch.position_ids[start + hit..end]);
            suffix
                .token_type_ids
                .extend_from_slice(&batch.token_type_ids[start + hit..end]);
            original_rows.extend(start + hit..end);
            suffix
                .cumulative_seq_lengths
                .push(suffix.input_ids.len() as u32);
            suffix.max_length = suffix.max_length.max((end - start - hit) as u32);
        }
        // Retain the existing RadixMLP equivalence classes for the remaining suffix.
        if let Some(scatter) = &batch.scatter_unfold {
            let mut remap = HashMap::new();
            let mut ids = Vec::new();
            let mut pos = Vec::new();
            let mut unfold = Vec::new();
            let mut fold = Vec::new();
            for (row, &original) in original_rows.iter().enumerate() {
                let next = remap.len() as u32;
                let compact = *remap.entry(scatter[original]).or_insert_with(|| {
                    ids.push(batch.input_ids[original]);
                    pos.push(batch.position_ids[original]);
                    fold.push(row as u32);
                    next
                });
                unfold.push(compact);
            }
            suffix.compact_input_ids = Some(ids);
            suffix.compact_position_ids = Some(pos);
            suffix.scatter_unfold = Some(unfold);
            suffix.fold_gather = Some(fold);
        }
        #[cfg(feature = "cuda")]
        let rows = {
            let mut rows = Vec::with_capacity(batch.input_ids.len());
            for (i, &hit) in hits.iter().enumerate() {
                rows.extend((0..hit).map(|r| r as u32));
                rows.extend(
                    (suffix.cumulative_seq_lengths[i]..suffix.cumulative_seq_lengths[i + 1])
                        .map(|r| r + self.capacity as u32),
                );
            }
            Tensor::new(rows.as_slice(), device)?
        };
        Ok(PrefixPlan {
            #[cfg(feature = "cuda")]
            rows,
            suffix,
            hits,
            offsets: batch.cumulative_seq_lengths.clone(),
            kv_offsets: Tensor::new(batch.cumulative_seq_lengths.as_slice(), device)?,
            max_k: batch.max_length as usize,
            fill: if self.ids.is_empty() {
                self.capacity.min(batch.cumulative_seq_lengths[1] as usize)
            } else {
                0
            },
        })
    }

    pub fn assemble(
        &self,
        layer: usize,
        plan: &PrefixPlan,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (saved_k, saved_v) = &self.layers[layer];
        if plan.fill > 0 {
            copy_rows(saved_k, 0, &k.narrow(0, 0, plan.fill)?)?;
            copy_rows(saved_v, 0, &v.narrow(0, 0, plan.fill)?)?;
            return Ok((k.clone(), v.clone()));
        }
        let (scratch_k, scratch_v) = self.scratch.as_ref().unwrap();
        #[cfg(feature = "cuda")]
        if k.device().is_cuda() && matches!(k.dtype(), DType::F16 | DType::BF16) {
            let total = *plan.offsets.last().unwrap() as usize;
            let out_k = scratch_k.narrow(0, 0, total)?;
            let out_v = scratch_v.narrow(0, 0, total)?;
            let op = cuda::Assemble(plan.rows.clone());
            out_k.inplace_op3(saved_k, k, &op)?;
            out_v.inplace_op3(saved_v, v, &op)?;
            return Ok((out_k, out_v));
        }
        for (i, &hit) in plan.hits.iter().enumerate() {
            let start = plan.offsets[i] as usize;
            let qstart = plan.suffix.cumulative_seq_lengths[i] as usize;
            let qlen = (plan.suffix.cumulative_seq_lengths[i + 1] as usize) - qstart;
            if hit > 0 {
                copy_rows(scratch_k, start, &saved_k.narrow(0, 0, hit)?)?;
                copy_rows(scratch_v, start, &saved_v.narrow(0, 0, hit)?)?;
            }
            copy_rows(scratch_k, start + hit, &k.narrow(0, qstart, qlen)?)?;
            copy_rows(scratch_v, start + hit, &v.narrow(0, qstart, qlen)?)?;
        }
        let total = *plan.offsets.last().unwrap() as usize;
        Ok((
            scratch_k.narrow(0, 0, total)?,
            scratch_v.narrow(0, 0, total)?,
        ))
    }

    pub fn finish(&mut self, plan: &PrefixPlan, batch: &Batch, output: Tensor) -> Result<Tensor> {
        if plan.fill > 0 {
            let shape = (self.capacity, output.dim(1)?);
            let cached = Tensor::zeros(shape, output.dtype(), output.device())?;
            copy_rows(&cached, 0, &output.narrow(0, 0, plan.fill)?)?;
            self.outputs = Some(cached);
            // Publish validity only after every layer and the final norm have succeeded.
            self.ids = batch.input_ids[..plan.fill].to_vec();
            self.positions = batch.position_ids[..plan.fill].to_vec();
            return Ok(output);
        }
        // Returned tensors own their storage: later forwards cannot mutate an earlier result.
        let full = Tensor::zeros(
            (batch.input_ids.len(), output.dim(1)?),
            output.dtype(),
            output.device(),
        )?;
        for (i, &hit) in plan.hits.iter().enumerate() {
            let start = plan.offsets[i] as usize;
            let qstart = plan.suffix.cumulative_seq_lengths[i] as usize;
            let qlen = plan.suffix.cumulative_seq_lengths[i + 1] as usize - qstart;
            if hit > 0 {
                copy_rows(
                    &full,
                    start,
                    &self.outputs.as_ref().unwrap().narrow(0, 0, hit)?,
                )?;
            }
            copy_rows(&full, start + hit, &output.narrow(0, qstart, qlen)?)?;
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_survive_hits_and_position_misses() -> Result<()> {
        let batch = Batch {
            multimodal: vec![],
            input_ids: vec![3, 4, 5],
            token_type_ids: vec![0; 3],
            position_ids: vec![0, 1, 2],
            cumulative_seq_lengths: vec![0, 3],
            max_length: 3,
            pooled_indices: vec![0],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let mut cache = PrefixKvCache::new(4);
        let plan = cache
            .plan(&batch, &Device::Cpu, DType::F32, 1, 1, 2)?
            .unwrap();
        assert_eq!(plan.fill, 3);
        let kv = Tensor::ones((3, 1, 2), DType::F32, &Device::Cpu)?;
        cache.assemble(0, &plan, &kv, &kv)?;
        cache.finish(
            &plan,
            &batch,
            Tensor::ones((3, 2), DType::F32, &Device::Cpu)?,
        )?;
        let ids = (
            cache.layers[0].0.id(),
            cache.scratch.as_ref().unwrap().0.id(),
        );
        let plan = cache
            .plan(&batch, &Device::Cpu, DType::F32, 1, 1, 2)?
            .unwrap();
        assert_eq!(plan.hits, vec![2]);
        assert_eq!(plan.suffix.input_ids, vec![5]);
        assert_eq!(
            ids,
            (
                cache.layers[0].0.id(),
                cache.scratch.as_ref().unwrap().0.id()
            )
        );
        let mut moved = batch.clone();
        moved.position_ids = vec![4, 5, 6];
        assert!(cache
            .plan(&moved, &Device::Cpu, DType::F32, 1, 1, 2)?
            .is_none());
        assert_eq!(cache.ids, batch.input_ids);
        Ok(())
    }
}
