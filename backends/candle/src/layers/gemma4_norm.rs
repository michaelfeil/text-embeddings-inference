//! Gemma4 normalization with the original variance reduction and FP32 rounding.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp3, DType, Layout, Result, Shape, Tensor, D,
};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/gemma4_norm_ptx.rs"));
}

pub(crate) fn square(input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16
        || !input.is_contiguous()
        || input.elem_count() == 0
        || input.elem_count().div_ceil(256) > i32::MAX as usize
    {
        candle::bail!("Gemma4 square requires a nonempty contiguous BF16 tensor");
    }
    input.apply_op1_no_bwd(&Square)
}
struct Square;
impl CustomOp1 for Square {
    fn name(&self) -> &'static str {
        "gemma4-square"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma4 square kernel requires CUDA")
    }
    fn cuda_fwd(&self, input: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let n = layout.shape().elem_count();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(layout.start_offset()..layout.start_offset() + n);
        // Every output element is written; LaunchBuilder tracks buffer lifetimes.
        let mut output = unsafe { device.alloc::<f32>(n)? };
        let function = device.get_or_load_custom_func(
            "gemma4_square_bf16_f32",
            "gemma4-square",
            ptx::GEMMA4_NORM,
        )?;
        let count = n as u64;
        let mut builder = function.builder();
        builder.arg(&input).arg(&mut output).arg(&count);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            layout.shape().clone(),
        ))
    }
}

pub(crate) fn finish(
    input: &Tensor,
    variance: &Tensor,
    weight: Option<&Tensor>,
    epsilon: f64,
) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    if width == 0
        || width > u32::MAX as usize
        || input.elem_count() == 0
        || input.elem_count().div_ceil(256) > i32::MAX as usize
        || input.dtype() != DType::BF16
        || variance.dtype() != DType::F32
        || variance.elem_count() != input.elem_count() / width
    {
        candle::bail!("invalid Gemma4 normalization shape or dtype");
    }
    if let Some(weight) = weight {
        if weight.dtype() != DType::BF16 || weight.dims() != [width] {
            candle::bail!("invalid Gemma4 normalization weight");
        }
    }
    for tensor in [input, variance, weight.unwrap_or(input)] {
        if !tensor.is_contiguous() || !tensor.device().same_device(input.device()) {
            candle::bail!("Gemma4 normalization requires contiguous tensors on one CUDA device");
        }
    }
    input.apply_op3_no_bwd(
        variance,
        weight.unwrap_or(input),
        &Finish {
            epsilon: epsilon as f32,
            weighted: weight.is_some(),
        },
    )
}

struct Finish {
    epsilon: f32,
    weighted: bool,
}
impl CustomOp3 for Finish {
    fn name(&self) -> &'static str {
        "gemma4-norm-finish"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Gemma4 normalization kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        variance: &CudaStorage,
        vl: &Layout,
        weight: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let n = il.shape().elem_count();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(il.start_offset()..il.start_offset() + n);
        let variance = variance.as_cuda_slice::<f32>()?;
        let variance =
            variance.slice(vl.start_offset()..vl.start_offset() + vl.shape().elem_count());
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let weight = weight.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        // The kernel writes every output element and the launch tracks buffer lifetimes.
        let mut output = unsafe { device.alloc::<half::bf16>(n)? };
        let function = device.get_or_load_custom_func(
            "gemma4_norm_finish_bf16",
            "gemma4-norm-finish",
            ptx::GEMMA4_NORM,
        )?;
        let mut builder = function.builder();
        let count = n as u64;
        let width = *il.dims().last().unwrap() as u32;
        let weighted = i32::from(self.weighted);
        builder
            .arg(&input)
            .arg(&variance)
            .arg(&weight)
            .arg(&mut output)
            .arg(&count)
            .arg(&width)
            .arg(&self.epsilon)
            .arg(&weighted);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
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
    fn square_preserves_all_bf16_values() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values = (0..=65536)
            .map(|i| half::bf16::from_bits(i as u16))
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, 65537, &device)?.narrow(0, 1, 65536)?;
        let expected = input.to_dtype(DType::F32)?.sqr()?.to_vec1::<f32>()?;
        let actual = square(&input)?.to_vec1::<f32>()?;
        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "element={i}: {a:?} != {b:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn finish_preserves_candle_rounding() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for width in [256, 512, 2816] {
            let rows = 1 + 65536_usize.div_ceil(width);
            let values = (0..rows * width)
                .map(|i| half::bf16::from_bits(i as u16))
                .collect::<Vec<_>>();
            // Exercise a contiguous view with a nonzero storage offset.
            let input = Tensor::from_vec(values, (rows, width), &device)?.narrow(0, 1, rows - 1)?;
            let variance = Tensor::from_vec(
                (0..rows - 1)
                    .map(|i| [0.0_f32, 1e-35, 1e-12, 1.0, 10000.0, f32::INFINITY][i % 6])
                    .collect::<Vec<_>>(),
                (rows - 1, 1),
                &device,
            )?;
            let weight = Tensor::from_vec(
                (0..width)
                    .map(|i| (i as f32 % 19.0 - 9.0) / 4.0)
                    .collect::<Vec<_>>(),
                width,
                &device,
            )?
            .to_dtype(DType::BF16)?;
            for epsilon in [1e-6, 1e-12] {
                for weighted in [false, true] {
                    let mut expected = input
                        .to_dtype(DType::F32)?
                        .broadcast_div(&(&variance + epsilon)?.sqrt()?)?;
                    if weighted {
                        expected = expected.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
                    }
                    let expected = expected
                        .to_dtype(DType::BF16)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?;
                    let actual = finish(&input, &variance, weighted.then_some(&weight), epsilon)?
                        .flatten_all()?
                        .to_vec1::<half::bf16>()?;
                    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                        assert!((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(), "width={width} weighted={weighted} epsilon={epsilon} element={i}: {a:?} != {b:?}");
                    }
                }
            }
        }
        Ok(())
    }
}
