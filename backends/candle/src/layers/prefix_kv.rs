//! Bounded dynamic causal prefixes, serialized by the model forward lock.
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

use super::prefix_index::{CacheMode, PrefixIndex, Token};
#[cfg(feature = "prefix-cache-paged")]
use super::prefix_index::{PageLayout, PAGE_SIZE};
pub(crate) struct PrefixKvCache {
    index: PrefixIndex,
    valid: bool,
    layers: Vec<(Tensor, Tensor)>,
    outputs: Option<Tensor>,
    scratch: Option<(Tensor, Tensor)>,
    #[cfg(feature = "prefix-cache-paged")]
    pool: Option<(Tensor, Tensor)>,
}
pub(crate) struct PrefixPlan {
    pub suffix: Batch,
    pub kv_offsets: Tensor,
    pub max_k: usize,
    next: PrefixIndex,
    rows: Vec<usize>,
    writes: Vec<(usize, usize)>,
    #[cfg(feature = "cuda")]
    row_tensor: Tensor,
    #[cfg(feature = "cuda")]
    write_tensor: Tensor,
    hits: usize,
    #[cfg(feature = "prefix-cache-paged")]
    capture_maps: (Tensor, Tensor),
    #[cfg(feature = "prefix-cache-paged")]
    paged: Option<(
        candle_flash_attn_v4::Seqlens,
        Vec<candle_flash_attn_v4::PagedKv>,
        Tensor,
    )>,
}
impl PrefixKvCache {
    #[cfg(test)]
    pub fn new(capacity: usize) -> Self {
        Self::with_mode(capacity, capacity, CacheMode::Sequence)
    }
    pub(super) fn with_mode(capacity: usize, max_prefix: usize, mode: CacheMode) -> Self {
        Self {
            index: PrefixIndex::new(mode, capacity, max_prefix),
            valid: true,
            layers: vec![],
            outputs: None,
            scratch: None,
            #[cfg(feature = "prefix-cache-paged")]
            pool: None,
        }
    }

    pub fn configured(capacity: usize, max_prefix: usize, paged: bool) -> Result<Self> {
        if capacity == 0 || capacity > i32::MAX as usize || max_prefix == 0 {
            candle::bail!(
                "prefix cache requires a positive int32 token budget and positive prefix limit"
            );
        }
        if paged {
            #[cfg(not(feature = "prefix-cache-paged"))]
            candle::bail!("paged prefix caching requires the prefix-cache-paged build feature");
            #[cfg(feature = "prefix-cache-paged")]
            {
                if capacity < PAGE_SIZE {
                    candle::bail!("paged cache requires at least 64 tokens");
                }
                return Ok(Self::with_mode(capacity, max_prefix, CacheMode::Paged));
            }
        }
        Ok(Self::with_mode(capacity, max_prefix, CacheMode::Sequence))
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
        if batch.is_empty()
            || batch.input_ids.iter().all(|&id| id == 0)
            || batch.multimodal.iter().any(Option::is_some)
        {
            return Ok(None);
        }
        // A failed forward may have overwritten resident slots. Discard their
        // metadata before another request can reuse partially updated state.
        if !self.valid {
            self.index.clear();
        }
        let capacity = self.index.capacity();
        if capacity == 0 {
            candle::bail!("prefix cache capacity is empty")
        }
        let sequences: Vec<Vec<Token>> = batch
            .cumulative_seq_lengths
            .windows(2)
            .map(|w| {
                batch.input_ids[w[0] as usize..w[1] as usize]
                    .iter()
                    .copied()
                    .zip(
                        batch.position_ids[w[0] as usize..w[1] as usize]
                            .iter()
                            .copied(),
                    )
                    .collect()
            })
            .collect();
        if sequences.iter().any(Vec::is_empty) {
            candle::bail!("prefix cache requires nonempty sequences")
        }
        if capacity
            .checked_add(batch.input_ids.len())
            .is_none_or(|n| n > u32::MAX as usize)
        {
            candle::bail!("prefix row map exceeds uint32 indexing");
        }
        let index_plan = self.index.plan(&sequences);
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
        let mut original_rows = vec![];
        let mut rows = vec![];
        for (i, w) in batch.cumulative_seq_lengths.windows(2).enumerate() {
            let (start, end) = (w[0] as usize, w[1] as usize);
            let hit = index_plan.hits[i].len();
            rows.extend_from_slice(&index_plan.hits[i]);
            rows.extend(
                (suffix.input_ids.len()..suffix.input_ids.len() + end - start - hit)
                    .map(|i| capacity + i),
            );
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
        if let Some(scatter) = &batch.scatter_unfold {
            let mut remap = HashMap::new();
            let (mut ids, mut pos, mut unfold, mut fold) = (vec![], vec![], vec![], vec![]);
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
        #[cfg(feature = "prefix-cache-paged")]
        let paged = if self.index.mode() == CacheMode::Paged {
            if !device.is_cuda() || !matches!(dtype, DType::F16 | DType::BF16) {
                candle::bail!("paged prefix cache requires CUDA FP16 or BF16");
            }
            let max_rows = capacity.checked_mul(layers).and_then(|n| {
                batch
                    .input_ids
                    .len()
                    .checked_add(sequences.len() * PAGE_SIZE)
                    .and_then(|t| n.checked_add(t.next_power_of_two()))
            });
            if max_rows.is_none_or(|n| n > i32::MAX as usize) {
                candle::bail!("paged prefix arena exceeds int32 indexing");
            }
            let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
            let layout = PageLayout::new(&index_plan.hits, &lengths, capacity, layers);
            let persistent = capacity * layers;
            let needed = persistent + layout.transient_rows;
            if self.pool.as_ref().is_none_or(|(k, _)| k.dims()[0] < needed) {
                let shape = (
                    persistent + layout.transient_rows.next_power_of_two(),
                    heads,
                    dim,
                );
                let k = Tensor::zeros(shape, dtype, device)?;
                let v = Tensor::zeros(shape, dtype, device)?;
                if let Some((old_k, old_v)) = &self.pool {
                    copy_rows(&k, 0, &old_k.narrow(0, 0, persistent)?)?;
                    copy_rows(&v, 0, &old_v.narrow(0, 0, persistent)?)?;
                }
                self.layers = (0..layers)
                    .map(|i| {
                        Ok((
                            k.narrow(0, i * capacity, capacity)?,
                            v.narrow(0, i * capacity, capacity)?,
                        ))
                    })
                    .collect::<Result<_>>()?;
                self.pool = Some((k, v));
            }
            let pages = self.pool.as_ref().unwrap().0.dim(0)? / PAGE_SIZE;
            let lengths: Vec<u32> = lengths.iter().map(|&n| n as u32).collect();
            let metadata = (0..layers)
                .map(|i| {
                    candle_flash_attn_v4::PagedKv::new(
                        &lengths,
                        &layout.for_layer(&index_plan.hits, capacity, i),
                        pages,
                        device,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let writes: Vec<u32> = layout
                .suffix_writes
                .iter()
                .flat_map(|&(d, s)| [d as u32, s as u32])
                .collect();
            Some((
                candle_flash_attn_v4::Seqlens::new(&suffix.cumulative_seq_lengths, device)?,
                metadata,
                Tensor::new(writes.as_slice(), device)?,
            ))
        } else {
            None
        };
        if self.layers.is_empty() {
            self.layers = (0..layers)
                .map(|_| {
                    let shape = (capacity, heads, dim);
                    Ok((
                        Tensor::zeros(shape, dtype, device)?,
                        Tensor::zeros(shape, dtype, device)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
        }
        // Paged attention needs staging only for newly admitted rows. Sequence
        // attention needs a packed view of the entire batch.
        let total = if self.index.mode() == CacheMode::Paged {
            index_plan.writes.len().max(1)
        } else {
            batch.input_ids.len()
        };
        if self
            .scratch
            .as_ref()
            .is_none_or(|(k, _)| k.dims()[0] < total)
        {
            let shape = (total.next_power_of_two(), heads, dim);
            self.scratch = Some((
                Tensor::zeros(shape, dtype, device)?,
                Tensor::zeros(shape, dtype, device)?,
            ));
        }
        let hits = index_plan.hits.iter().map(Vec::len).sum();
        tracing::debug!(mode=?self.index.mode(),resident_tokens=self.index.resident_tokens(),reused_tokens=hits,evictions=index_plan.evictions,"Qwen3 dynamic prefix cache");
        let plan = PrefixPlan {
            #[cfg(feature = "prefix-cache-paged")]
            capture_maps: (
                Tensor::new(
                    index_plan
                        .writes
                        .iter()
                        .map(|&(_, source)| rows[source] as u32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    device,
                )?,
                Tensor::new(
                    index_plan
                        .writes
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &(dest, _))| [dest as u32, i as u32])
                        .collect::<Vec<_>>()
                        .as_slice(),
                    device,
                )?,
            ),
            #[cfg(feature = "prefix-cache-paged")]
            paged,
            #[cfg(feature = "cuda")]
            row_tensor: Tensor::new(
                rows.iter()
                    .map(|&n| n as u32)
                    .collect::<Vec<_>>()
                    .as_slice(),
                device,
            )?,
            #[cfg(feature = "cuda")]
            write_tensor: Tensor::new(
                index_plan
                    .writes
                    .iter()
                    .flat_map(|&(d, s)| [d as u32, s as u32])
                    .collect::<Vec<_>>()
                    .as_slice(),
                device,
            )?,
            suffix,
            next: index_plan.next,
            rows,
            writes: index_plan.writes,
            hits,
            kv_offsets: Tensor::new(batch.cumulative_seq_lengths.as_slice(), device)?,
            max_k: batch.max_length as usize,
        };
        self.valid = false;
        Ok(Some(plan))
    }

    fn gather(
        &self,
        dst: &Tensor,
        saved: &Tensor,
        suffix: &Tensor,
        plan: &PrefixPlan,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if dst.device().is_cuda() && matches!(dst.dtype(), DType::F16 | DType::BF16) {
            return dst.inplace_op3(saved, suffix, &cuda::Assemble(plan.row_tensor.clone()));
        }
        let capacity = self.index.capacity();
        for (out, &row) in plan.rows.iter().enumerate() {
            let (source, row) = if row < capacity {
                (saved, row)
            } else {
                (suffix, row - capacity)
            };
            copy_rows(dst, out, &source.narrow(0, row, 1)?)?;
        }
        Ok(())
    }
    fn store(&self, dst: &Tensor, source: &Tensor, plan: &PrefixPlan) -> Result<()> {
        if plan.writes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if dst.device().is_cuda() && matches!(dst.dtype(), DType::F16 | DType::BF16) {
            return dst.inplace_op2(source, &cuda::Scatter(plan.write_tensor.clone()));
        }
        for &(dest, src) in &plan.writes {
            copy_rows(dst, dest, &source.narrow(0, src, 1)?)?;
        }
        Ok(())
    }

    #[cfg(feature = "prefix-cache-paged")]
    pub fn paged_attention(
        &self,
        layer: usize,
        plan: &PrefixPlan,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        let Some((lengths, metadata, writes)) = &plan.paged else {
            return Ok(None);
        };
        let (pool_k, pool_v) = self.pool.as_ref().unwrap();
        pool_k.inplace_op2(k, &cuda::Scatter(writes.clone()))?;
        pool_v.inplace_op2(v, &cuda::Scatter(writes.clone()))?;
        let (rows, heads, dim) = pool_k.dims3()?;
        let shape = (rows / PAGE_SIZE, PAGE_SIZE, heads, dim);
        candle_flash_attn_v4::flash_attn_paged(
            q,
            &pool_k.reshape(shape)?,
            &pool_v.reshape(shape)?,
            lengths,
            &metadata[layer],
            scale,
        )
        .map(Some)
    }

    #[cfg(feature = "prefix-cache-paged")]
    pub fn capture_paged(
        &self,
        layer: usize,
        plan: &PrefixPlan,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if plan.writes.is_empty() {
            return Ok(());
        }
        let (scratch_k, scratch_v) = self.scratch.as_ref().unwrap();
        let (saved_k, saved_v) = &self.layers[layer];
        // Snapshot admitted rows before recycling resident slots. Source rows
        // can still refer to an old page needed by another admission.
        let staged_k = scratch_k.narrow(0, 0, plan.writes.len())?;
        let staged_v = scratch_v.narrow(0, 0, plan.writes.len())?;
        staged_k.inplace_op3(saved_k, k, &cuda::Assemble(plan.capture_maps.0.clone()))?;
        staged_v.inplace_op3(saved_v, v, &cuda::Assemble(plan.capture_maps.0.clone()))?;
        saved_k.inplace_op2(&staged_k, &cuda::Scatter(plan.capture_maps.1.clone()))?;
        saved_v.inplace_op2(&staged_v, &cuda::Scatter(plan.capture_maps.1.clone()))
    }

    pub fn assemble(
        &self,
        layer: usize,
        plan: &PrefixPlan,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if plan.hits == 0 {
            return Ok((k.clone(), v.clone()));
        }
        let (saved_k, saved_v) = &self.layers[layer];
        let (scratch_k, scratch_v) = self.scratch.as_ref().unwrap();
        let total = plan.rows.len();
        let out_k = scratch_k.narrow(0, 0, total)?;
        let out_v = scratch_v.narrow(0, 0, total)?;
        self.gather(&out_k, saved_k, k, plan)?;
        self.gather(&out_v, saved_v, v, plan)?;
        Ok((out_k, out_v))
    }

    // Called after attention has consumed the old slots, on the same device stream.
    pub fn capture(&self, layer: usize, plan: &PrefixPlan, k: &Tensor, v: &Tensor) -> Result<()> {
        self.store(&self.layers[layer].0, k, plan)?;
        self.store(&self.layers[layer].1, v, plan)
    }

    pub fn finish(&mut self, plan: &PrefixPlan, batch: &Batch, output: Tensor) -> Result<Tensor> {
        let full = if plan.hits == 0 {
            output
        } else {
            let full = Tensor::zeros(
                (batch.input_ids.len(), output.dim(1)?),
                output.dtype(),
                output.device(),
            )?;
            self.gather(&full, self.outputs.as_ref().unwrap(), &output, plan)?;
            full
        };
        if self.outputs.is_none() {
            self.outputs = Some(Tensor::zeros(
                (self.index.capacity(), full.dim(1)?),
                full.dtype(),
                full.device(),
            )?);
        }
        self.store(self.outputs.as_ref().unwrap(), &full, plan)?;
        self.index = plan.next.clone();
        self.valid = true;
        Ok(full)
    }
}

#[cfg(all(test, feature = "prefix-cache-paged"))]
#[path = "prefix_kv_paged_tests.rs"]
mod paged_tests;
