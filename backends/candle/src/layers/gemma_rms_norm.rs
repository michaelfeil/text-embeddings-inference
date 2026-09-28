//! BF16 Gemma RMSNorm with FP32 scale and accumulation, without cast buffers.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/gemma_norm_ptx.rs"));
}

pub(crate) fn forward(x: &Tensor, scale: &Tensor, epsilon: f32) -> Result<Tensor> {
    let width = x.dim(candle::D::Minus1)?;
    if x.dtype() != DType::BF16
        || scale.dtype() != DType::F32
        || !x.device().is_cuda()
        || !scale.device().same_device(x.device())
        || !x.is_contiguous()
        || !scale.is_contiguous()
        || scale.dims() != [width]
        || width == 0
        || width > 8192
        || x.elem_count() / width > i32::MAX as usize
    {
        candle::bail!("Gemma RMSNorm requires contiguous CUDA BF16 activations and FP32 scale");
    }
    if x.elem_count() == 0 {
        return Ok(x.clone());
    }
    x.apply_op2_no_bwd(scale, &Norm { epsilon })
}
struct Norm {
    epsilon: f32,
}
impl CustomOp2 for Norm {
    fn name(&self) -> &'static str {
        "gemma-rms-norm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma RMSNorm requires CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        scale: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = x.device();
        let n = xl.shape().elem_count();
        let width = *xl.dims().last().unwrap() as u32;
        let x = x
            .as_cuda_slice::<half::bf16>()?
            .slice(xl.start_offset()..xl.start_offset() + n);
        let scale = scale
            .as_cuda_slice::<f32>()?
            .slice(sl.start_offset()..sl.start_offset() + width as usize);
        // The grid writes every output element exactly once.
        let mut out = unsafe { device.alloc::<half::bf16>(n)? };
        let kernel = device.get_or_load_custom_func(
            "gemma_rms_norm_bf16",
            "tei-gemma-rms-norm",
            ptx::GEMMA_RMS_NORM,
        )?;
        let mut launch = kernel.builder();
        launch
            .arg(&x)
            .arg(&scale)
            .arg(&mut out)
            .arg(&width)
            .arg(&self.epsilon);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((n / width as usize) as u32, 1, 1),
                block_dim: (if width <= 128 { 32 } else { 256 }, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xl.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, D};
    #[test]
    #[ignore = "requires CUDA; validates and times the fused Gemma normalization"]
    fn reference_and_timing() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for width in [32, 256, 768, 1024] {
            for rows in [1, 128, 2048] {
                let x = Tensor::randn(0f32, 2f32, (rows, width), &device)?.to_dtype(DType::BF16)?;
                let scale = (Tensor::randn(0f32, 0.2f32, width, &device)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?
                    + 1.)?;
                let old = || {
                    candle_layer_norm::rms_norm(&x.to_dtype(DType::F32)?, &scale, None, 1e-6)?
                        .to_dtype(DType::BF16)
                };
                let actual = forward(&x, &scale, 1e-6)?.to_dtype(DType::F32)?;
                let expected = old()?.to_dtype(DType::F32)?;
                let error = (&actual - &expected)?
                    .abs()?
                    .broadcast_div(&(&expected.abs()? + 0.01)?)?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 0.008,
                    "rows={rows}, width={width}, relative error={error}"
                );
                // Also compare against independent unfused FP32 normalization.
                let xf = x.to_dtype(DType::F32)?;
                let reference = xf
                    .broadcast_div(&(xf.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?
                    .broadcast_mul(&scale)?
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?;
                let error = (&actual - reference)?
                    .abs()?
                    .broadcast_div(&(&actual.abs()? + 0.01)?)?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(error < 0.008, "independent relative error={error}");
                let mut times = Vec::new();
                for fused in [false, true] {
                    for _ in 0..10 {
                        let _ = if fused {
                            forward(&x, &scale, 1e-6)?
                        } else {
                            old()?
                        };
                    }
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..200 {
                        let _ = if fused {
                            forward(&x, &scale, 1e-6)?
                        } else {
                            old()?
                        };
                    }
                    device.synchronize()?;
                    times.push(start.elapsed().as_secs_f64() * 1e6 / 200.);
                }
                println!(
                    "rows={rows} width={width} old_us={:.2} fused_us={:.2} speedup={:.2}",
                    times[0],
                    times[1],
                    times[0] / times[1]
                );
            }
        }
        Ok(())
    }
}
