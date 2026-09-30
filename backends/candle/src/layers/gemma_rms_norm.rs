//! BF16 Gemma RMSNorm with FP32 scale and accumulation, without cast buffers.
//! Packed Q/K views may have gaps between tokens; the result is contiguous.
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
        || !supported_layout(x.layout())
        || !scale.is_contiguous()
        || scale.dims() != [width]
        || width == 0
        || width > 8192
        || x.elem_count() / width > i32::MAX as usize
    {
        candle::bail!("Gemma RMSNorm requires row-contiguous CUDA BF16 activations and FP32 scale");
    }
    if x.elem_count() == 0 {
        return Ok(x.clone());
    }
    x.apply_op2_no_bwd(scale, &Norm { epsilon })
}
// Packed Q/K views have contiguous heads within each token and a gap between tokens.
fn supported_layout(layout: &Layout) -> bool {
    if layout.is_contiguous() {
        return true;
    }
    let dims = layout.dims();
    let strides = layout.stride();
    (dims.len() == 2 && strides[1] == 1)
        || (dims.len() == 3 && strides[2] == 1 && strides[1] == dims[2])
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
        let heads = if xl.dims().len() == 3 {
            xl.dims()[1]
        } else {
            1
        } as u32;
        let token_stride = if xl.is_contiguous() {
            width as usize * heads as usize
        } else {
            xl.stride()[0]
        } as u64;
        let x = x.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
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
            .arg(&heads)
            .arg(&token_stride)
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
    #[ignore = "requires CUDA"]
    fn strided_matches_contiguous() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for tokens in [1, 7, 511] {
            for (heads, width) in [(1, 32), (3, 256), (4, 768)] {
                let packed = Tensor::randn(0f32, 2f32, (tokens, (heads + 2) * width), &device)?
                    .to_dtype(DType::BF16)?;
                let scale = Tensor::randn(1f32, 0.2f32, width, &device)?;
                let view = packed
                    .reshape((tokens, heads + 2, width))?
                    .narrow(1, 1, heads)?;
                let expected = forward(&view.contiguous()?, &scale, 1e-6)?;
                let actual = forward(&view, &scale, 1e-6)?;
                assert_eq!(
                    actual.flatten_all()?.to_vec1::<half::bf16>()?,
                    expected.flatten_all()?.to_vec1::<half::bf16>()?
                );
                let view2 = packed.narrow(1, width, width)?;
                assert_eq!(
                    forward(&view2, &scale, 1e-6)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?,
                    forward(&view2.contiguous()?, &scale, 1e-6)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA; validates and times the fused Gemma normalization"]
    fn reference_and_timing() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for width in [32, 256, 768, 1024, 2816] {
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
