//! Fused Qwen3 head normalization and NeoX RoPE, with existing half rounding.
use super::RMSNorm;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use candle::cuda_backend::CudaDType;
use candle::{
    CpuStorage, CudaStorage, CustomOp2, DType, Device, Layout, Result, Shape, Storage, Tensor,
};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/qk_ptx.rs"));
}

pub(crate) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    qn: &RMSNorm,
    kn: &RMSNorm,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    if q.rank() != 3 || k.rank() != 3 {
        return Ok(None);
    }
    let (tokens, qheads, width) = q.dims3()?;
    let (ktokens, kheads, kwidth) = k.dims3()?;
    if width != 128
        || kwidth != 128
        || tokens != ktokens
        || tokens == 0
        || qheads == 0
        || kheads == 0
        || !matches!(q.device(), Device::Cuda(_))
        || !matches!(q.dtype(), DType::F16 | DType::BF16)
        || qn.epsilon.to_bits() != kn.epsilon.to_bits()
        || tokens
            .checked_mul(qheads + kheads)
            .is_none_or(|n| n > i32::MAX as usize / 128)
        || qn.weight.dims() != [128]
        || kn.weight.dims() != [128]
        || cos.dims() != [tokens, 64]
        || sin.dims() != [tokens, 64]
    {
        return Ok(None);
    }
    for tensor in [q, k, &qn.weight, &kn.weight, cos, sin] {
        if tensor.dtype() != q.dtype()
            || !tensor.device().same_device(q.device())
            || !tensor.is_contiguous()
            || !tensor.layout().start_offset().is_multiple_of(8)
        {
            return Ok(None);
        }
    }
    let packed = q.apply_op2_no_bwd(
        k,
        &Fused {
            qw: qn.weight.clone(),
            kw: kn.weight.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            epsilon: qn.epsilon,
        },
    )?;
    let nq = q.elem_count();
    Ok(Some((
        packed.narrow(0, 0, nq)?.reshape(q.shape())?,
        packed.narrow(0, nq, k.elem_count())?.reshape(k.shape())?,
    )))
}

struct Fused {
    qw: Tensor,
    kw: Tensor,
    cos: Tensor,
    sin: Tensor,
    epsilon: f32,
}
impl Fused {
    fn launch<T: CudaDType + DeviceRepr>(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        name: &str,
    ) -> Result<(CudaStorage, Shape)> {
        let (tokens, qheads, _) = ql.shape().dims3()?;
        let (_, kheads, _) = kl.shape().dims3()?;
        let device = q.device();
        let q = q
            .as_cuda_slice::<T>()?
            .slice(ql.start_offset()..ql.start_offset() + ql.shape().elem_count());
        let k = k
            .as_cuda_slice::<T>()?
            .slice(kl.start_offset()..kl.start_offset() + kl.shape().elem_count());
        let (qw, qwl) = self.qw.storage_and_layout();
        let (kw, kwl) = self.kw.storage_and_layout();
        let (cos, cl) = self.cos.storage_and_layout();
        let (sin, sl) = self.sin.storage_and_layout();
        let (Storage::Cuda(qw), Storage::Cuda(kw), Storage::Cuda(cos), Storage::Cuda(sin)) =
            (&*qw, &*kw, &*cos, &*sin)
        else {
            candle::bail!("Q/K fusion requires CUDA tensors")
        };
        let qw = qw
            .as_cuda_slice::<T>()?
            .slice(qwl.start_offset()..qwl.start_offset() + 128);
        let kw = kw
            .as_cuda_slice::<T>()?
            .slice(kwl.start_offset()..kwl.start_offset() + 128);
        let cos = cos
            .as_cuda_slice::<T>()?
            .slice(cl.start_offset()..cl.start_offset() + tokens * 64);
        let sin = sin
            .as_cuda_slice::<T>()?
            .slice(sl.start_offset()..sl.start_offset() + tokens * 64);
        let rows = tokens * (qheads + kheads);
        // Every head writes 128 disjoint output elements.
        let mut output = unsafe { device.alloc::<T>(rows * 128)? };
        let function =
            device.get_or_load_custom_func(name, "tei-qk-norm-rope", ptx::QK_NORM_ROPE)?;
        let mut builder = function.builder();
        let tokens = tokens as i32;
        let qheads = qheads as i32;
        let kheads = kheads as i32;
        builder
            .arg(&q)
            .arg(&k)
            .arg(&qw)
            .arg(&kw)
            .arg(&cos)
            .arg(&sin)
            .arg(&mut output)
            .arg(&tokens)
            .arg(&qheads)
            .arg(&kheads)
            .arg(&self.epsilon);
        // Shape/dtype/alignment checks above bound every read/write. The launch builder
        // tracks all tensor buffers on their device stream, including the norm weights.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows.div_ceil(8).min(4096) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(candle::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (rows * 128,).into(),
        ))
    }
}
impl CustomOp2 for Fused {
    fn name(&self) -> &'static str {
        "qk-norm-rope"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Q/K fusion requires CUDA")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.launch::<half::f16>(q, ql, k, kl, "qk_norm_rope_f16"),
            DType::BF16 => self.launch::<half::bf16>(q, ql, k, kl, "qk_norm_rope_bf16"),
            dtype => candle::bail!("unsupported Q/K fusion dtype {dtype:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn tensor(shape: &[usize], seed: u32, dtype: DType, device: &Device) -> Result<Tensor> {
        let values: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| {
                let mut x = (i as u32).wrapping_add(seed);
                x ^= x >> 16;
                x = x.wrapping_mul(0x7feb352d);
                x ^= x >> 15;
                x = x.wrapping_mul(0x846ca68b);
                x ^= x >> 16;
                ((x % 10001) as i32 - 5000) as f32 / 1000.
            })
            .collect();
        Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    #[test]
    fn fused_qk_matches_separate_norm_and_rope() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for tokens in [1, 7, 257] {
                let q = tensor(&[tokens, 3, 128], 91, dtype, &device)?;
                let k = tensor(&[tokens, 1, 128], 31, dtype, &device)?;
                let cos = tensor(&[tokens, 64], 81, dtype, &device)?;
                let sin = tensor(&[tokens, 64], 51, dtype, &device)?;
                let qweight = tensor(&[128], 73, dtype, &device)?;
                let kweight = tensor(&[128], 111, dtype, &device)?;
                for epsilon in [1e-6, 1e-5] {
                    let qnorm = RMSNorm::load(
                        VarBuilder::from_tensors(
                            HashMap::from([("weight".into(), qweight.clone())]),
                            dtype,
                            &device,
                        ),
                        128,
                        epsilon,
                    )?;
                    let knorm = RMSNorm::load(
                        VarBuilder::from_tensors(
                            HashMap::from([("weight".into(), kweight.clone())]),
                            dtype,
                            &device,
                        ),
                        128,
                        epsilon,
                    )?;
                    let (expected_q, _) = qnorm.forward(&q, None)?;
                    let (expected_k, _) = knorm.forward(&k, None)?;
                    candle_rotary::apply_rotary_inplace(
                        &expected_q,
                        &expected_k,
                        &cos,
                        &sin,
                        true,
                    )?;
                    let (actual_q, actual_k) = try_forward(&q, &k, &qnorm, &knorm, &cos, &sin)?
                        .expect("supported shape must fuse");
                    for (actual, expected) in [(actual_q, expected_q), (actual_k, expected_k)] {
                        let actual = actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let expected = expected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "dtype={dtype:?}, tokens={tokens}, epsilon={epsilon}, index={i}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
