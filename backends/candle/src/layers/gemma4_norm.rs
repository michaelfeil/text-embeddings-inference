//! Fused Gemma4 normalization preserving Candle's reduction tree and FP32 rounding.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor, D};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/gemma4_norm_ptx.rs"));
}

pub(crate) fn fused(input: &Tensor, weight: Option<&Tensor>, epsilon: f64) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if width < 32
        || width > i32::MAX as usize
        || input.elem_count() == 0
        || input.elem_count() / width > i32::MAX as usize
        || input.dtype() != DType::BF16
    {
        candle::bail!("invalid Gemma4 fused normalization shape or dtype");
    }
    if let Some(weight) = weight {
        if weight.dtype() != DType::BF16 || weight.dims() != [width] {
            candle::bail!("invalid Gemma4 fused normalization weight");
        }
    }
    for tensor in [input, weight.unwrap_or(input)] {
        if !tensor.is_contiguous() || !tensor.device().same_device(input.device()) {
            candle::bail!(
                "Gemma4 fused normalization requires contiguous tensors on one CUDA device"
            );
        }
    }
    input.apply_op2_no_bwd(
        weight.unwrap_or(input),
        &Fused {
            epsilon: epsilon as f32,
            weighted: weight.is_some(),
        },
    )
}
struct Fused {
    epsilon: f32,
    weighted: bool,
}
impl CustomOp2 for Fused {
    fn name(&self) -> &'static str {
        "gemma4-norm-fused"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma4 fused normalization kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let n = il.shape().elem_count();
        let width = *il.dims().last().unwrap() as u32;
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(il.start_offset()..il.start_offset() + n);
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let weight = weight.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        // Each row CTA writes all its output elements; launch tracks lifetimes.
        let mut output = unsafe { device.alloc::<half::bf16>(n)? };
        let function = device.get_or_load_custom_func(
            "gemma4_norm_fused_bf16",
            "gemma4-norm-fused",
            ptx::GEMMA4_NORM,
        )?;
        let scale = (1.0_f64 / f64::from(width)) as f32;
        let weighted = i32::from(self.weighted);
        let mut builder = function.builder();
        builder
            .arg(&input)
            .arg(&weight)
            .arg(&mut output)
            .arg(&width)
            .arg(&scale)
            .arg(&self.epsilon)
            .arg(&weighted);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((n / width as usize) as u32, 1, 1),
                block_dim: (width.min(1024).next_power_of_two(), 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            il.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn fused_preserves_candle_reduction_and_rounding() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for width in [256, 512, 1536, 2560, 2816] {
            let rows = 1 + 65536_usize.div_ceil(width);
            for random in [false, true] {
                let mut seed = 719_u32;
                let values = (0..rows * width)
                    .map(|i| {
                        if random {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            half::bf16::from_f32(
                                (seed as f32 / u32::MAX as f32 - 0.5)
                                    * 2.0_f32.powi((i / width % 25) as i32 - 12),
                            )
                        } else {
                            half::bf16::from_bits(i as u16)
                        }
                    })
                    .collect::<Vec<_>>();
                let input =
                    Tensor::from_vec(values, (rows, width), &device)?.narrow(0, 1, rows - 1)?;
                let weight = Tensor::from_vec(
                    (0..width + 1)
                        .map(|i| half::bf16::from_f32((i as f32 % 19.0 - 9.0) / 4.0))
                        .collect::<Vec<_>>(),
                    width + 1,
                    &device,
                )?
                .narrow(0, 1, width)?;
                for weighted in [false, true] {
                    for epsilon in [1e-6, 1e-12] {
                        let states = input.to_dtype(DType::F32)?;
                        let variance = states.sqr()?.mean_keepdim(D::Minus1)?;
                        let mut expected = states.broadcast_div(&(variance + epsilon)?.sqrt()?)?;
                        if weighted {
                            expected = expected.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
                        }
                        let expected = expected
                            .to_dtype(DType::BF16)?
                            .flatten_all()?
                            .to_vec1::<half::bf16>()?;
                        let actual = fused(&input, weighted.then_some(&weight), epsilon)?
                            .flatten_all()?
                            .to_vec1::<half::bf16>()?;
                        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                            assert!((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(), "width={width} random={random} weighted={weighted} epsilon={epsilon} element={i}: {a:?} != {b:?}");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
