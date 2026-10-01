use candle::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg},
        CudaDType,
    },
    CpuStorage, CudaStorage, DType, InplaceOp3, Layout, Result, Storage, Tensor,
};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/prefix_kv_ptx.rs"));
}

pub(super) struct Assemble(pub Tensor);
impl Assemble {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        dst: &mut CudaStorage,
        dl: &Layout,
        saved: &CudaStorage,
        sl: &Layout,
        suffix: &CudaStorage,
        nl: &Layout,
    ) -> Result<()> {
        let device = dst.device().clone();
        let tokens = dl.dims()[0];
        let width = dl.shape().elem_count() / tokens;
        let mut dst = dst
            .as_cuda_slice_mut::<T>()?
            .slice_mut(dl.start_offset()..dl.start_offset() + tokens * width);
        let saved = saved.as_cuda_slice::<T>()?.slice(sl.start_offset()..);
        let suffix = suffix.as_cuda_slice::<T>()?.slice(nl.start_offset()..);
        let (rows, rl) = self.0.storage_and_layout();
        let Storage::Cuda(rows) = &*rows else {
            candle::bail!("prefix row map must be on CUDA")
        };
        let rows = rows
            .as_cuda_slice::<u32>()?
            .slice(rl.start_offset()..rl.start_offset() + tokens);
        let function =
            device.get_or_load_custom_func("prefix_kv_copy", "tei-prefix-kv", ptx::PREFIX_KV)?;
        let saved_rows = sl.dims()[0] as u32;
        let width = width as u32;
        let saved_stride = sl.stride()[0] as u64;
        let suffix_stride = nl.stride()[0] as u64;
        let mut builder = function.builder();
        builder
            .arg(&mut dst)
            .arg(&saved)
            .arg(&suffix)
            .arg(&rows)
            .arg(&saved_rows)
            .arg(&width)
            .arg(&saved_stride)
            .arg(&suffix_stride);
        // PrefixPlan bounds every source row; destination is contiguous, source
        // inner dimensions are contiguous, and buffers are tracked on one stream.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok(())
    }
}
impl InplaceOp3 for Assemble {
    fn name(&self) -> &'static str {
        "prefix-kv-assemble"
    }
    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<()> {
        candle::bail!("CUDA prefix assembly called on CPU")
    }
    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dl: &Layout,
        saved: &CudaStorage,
        sl: &Layout,
        suffix: &CudaStorage,
        nl: &Layout,
    ) -> Result<()> {
        match dst.dtype() {
            DType::F16 => self.launch::<half::f16>(dst, dl, saved, sl, suffix, nl),
            DType::BF16 => self.launch::<half::bf16>(dst, dl, saved, sl, suffix, nl),
            _ => candle::bail!("prefix assembly requires FP16/BF16"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn prefix_copy_preserves_bits_and_strided_sources() -> Result<()> {
        let device = candle::Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            let saved =
                Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 1, 2), &device)?.to_dtype(dtype)?;
            let packed = Tensor::from_vec(
                (0..24).map(|n| n as f32).collect::<Vec<_>>(),
                (4, 3, 2),
                &device,
            )?
            .to_dtype(dtype)?;
            let suffix = packed.narrow(1, 1, 1)?;
            let out = Tensor::zeros((6, 1, 2), dtype, &device)?;
            let rows = Tensor::new(&[0u32, 2, 1, 5], &device)?;
            out.narrow(0, 1, 4)?
                .inplace_op3(&saved, &suffix, &Assemble(rows))?;
            assert_eq!(
                out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                vec![0., 0., 1., 2., 2., 3., 3., 4., 20., 21., 0., 0.]
            );
        }
        Ok(())
    }
}

pub(super) struct Scatter(pub Tensor);
impl Scatter {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        dst: &mut CudaStorage,
        dl: &Layout,
        src: &CudaStorage,
        sl: &Layout,
    ) -> Result<()> {
        let device = dst.device().clone();
        let width = (dl.shape().elem_count() / dl.dims()[0]) as u32;
        let mut dst = dst.as_cuda_slice_mut::<T>()?.slice_mut(dl.start_offset()..);
        let src = src.as_cuda_slice::<T>()?.slice(sl.start_offset()..);
        let (pairs, pl) = self.0.storage_and_layout();
        let Storage::Cuda(pairs) = &*pairs else {
            candle::bail!("CUDA scatter requires device indices")
        };
        let pairs = pairs
            .as_cuda_slice::<u32>()?
            .slice(pl.start_offset()..pl.start_offset() + self.0.elem_count());
        let ds = dl.stride()[0] as u64;
        let ss = sl.stride()[0] as u64;
        let function =
            device.get_or_load_custom_func("prefix_kv_scatter", "tei-prefix-kv", ptx::PREFIX_KV)?;
        let mut builder = function.builder();
        builder
            .arg(&mut dst)
            .arg(&src)
            .arg(&pairs)
            .arg(&width)
            .arg(&ds)
            .arg(&ss);
        // The host index plan bounds every pair and guarantees unique destinations.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((self.0.elem_count() / 2) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok(())
    }
}
impl candle::InplaceOp2 for Scatter {
    fn name(&self) -> &'static str {
        "prefix-kv-scatter"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle::bail!("CUDA scatter called on CPU")
    }
    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dl: &Layout,
        src: &CudaStorage,
        sl: &Layout,
    ) -> Result<()> {
        match dst.dtype() {
            DType::F16 => self.launch::<half::f16>(dst, dl, src, sl),
            DType::BF16 => self.launch::<half::bf16>(dst, dl, src, sl),
            _ => candle::bail!("prefix scatter requires half values"),
        }
    }
}
