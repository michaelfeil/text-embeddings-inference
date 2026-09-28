//! Token-major Gemma4 rotary, preserving Candle's BF16 product rounding.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/gemma4_rope_ptx.rs"));
}

pub(crate) fn forward(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, tokens, heads, dim) = input.dims4()?;
    if batch != 1
        || tokens == 0
        || heads == 0
        || dim == 0
        || dim % 2 != 0
        || input.elem_count().div_ceil(256) > i32::MAX as usize
        || cos.dims() != [1, 1, tokens, dim]
        || sin.dims() != cos.dims()
    {
        candle::bail!("invalid Gemma4 rotary shape");
    }
    for tensor in [input, cos, sin] {
        if tensor.dtype() != DType::BF16
            || !tensor.is_contiguous()
            || !tensor.device().same_device(input.device())
        {
            candle::bail!("Gemma4 rotary requires contiguous BF16 tensors on one CUDA device");
        }
    }
    input.apply_op3_no_bwd(cos, sin, &Rotary)
}
struct Rotary;
impl CustomOp3 for Rotary {
    fn name(&self) -> &'static str {
        "gemma4-rope"
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
        candle::bail!("Gemma4 rotary kernel requires CUDA")
    }
    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        il: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let device = input.device();
        let n = il.shape().elem_count();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(il.start_offset()..il.start_offset() + n);
        let cos = cos.as_cuda_slice::<half::bf16>()?;
        let cos = cos.slice(cl.start_offset()..cl.start_offset() + cl.shape().elem_count());
        let sin = sin.as_cuda_slice::<half::bf16>()?;
        let sin = sin.slice(sl.start_offset()..sl.start_offset() + sl.shape().elem_count());
        // Every output element is written; LaunchBuilder tracks buffer lifetimes.
        let mut output = unsafe { device.alloc::<half::bf16>(n)? };
        let function =
            device.get_or_load_custom_func("gemma4_rope_bf16", "gemma4-rope", ptx::GEMMA4_ROPE)?;
        let count = n as u64;
        let heads = il.dims()[2] as u64;
        let dim = il.dims()[3] as u64;
        let mut builder = function.builder();
        builder
            .arg(&input)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&count)
            .arg(&heads)
            .arg(&dim);
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
    fn rotary_preserves_bf16_products_and_offsets() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for (tokens, heads, dim) in [(1, 1, 256), (7, 8, 256), (19, 16, 512)] {
            // Narrowing exercises nonzero storage offsets for all three inputs.
            let input = Tensor::from_vec(
                (0..(tokens + 1) * heads * dim)
                    .map(|i| half::bf16::from_bits((i * 7919) as u16))
                    .collect::<Vec<_>>(),
                (tokens + 1, heads, dim),
                &device,
            )?
            .narrow(0, 1, tokens)?
            .unsqueeze(0)?;
            let angles = Tensor::from_vec(
                (0..(tokens + 1) * dim)
                    .map(|i| {
                        // Gemma4 global layers rotate only a quarter of each half.
                        if (i % dim) % (dim / 2) >= dim / 8 {
                            0.0_f32
                        } else {
                            i as f32 / 37.0
                        }
                    })
                    .collect::<Vec<_>>(),
                (tokens + 1, dim),
                &device,
            )?;
            let cos = angles
                .cos()?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, tokens)?
                .reshape((1, 1, tokens, dim))?;
            let sin = angles
                .sin()?
                .to_dtype(DType::BF16)?
                .narrow(0, 1, tokens)?
                .reshape((1, 1, tokens, dim))?;
            let expected = crate::layers::apply_rotary(&input.transpose(1, 2)?, &cos, &sin, dim)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<half::bf16>()?;
            let actual = forward(&input, &cos, &sin)?
                .flatten_all()?
                .to_vec1::<half::bf16>()?;
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                    "tokens={tokens} heads={heads} dim={dim} element={i}: {a:?} != {b:?}"
                );
            }
        }
        Ok(())
    }
}
