use super::Linear;
use candle::{Result, Tensor};

/// Only MLP projections opt into quantization. Attention and output heads retain
/// the existing precision and implementation.
pub(crate) enum MlpLinear {
    Dense(Linear),
    #[cfg(feature = "experimental-fp8")]
    Fp8(candle_cublaslt::fp8::Fp8Linear),
}

#[cfg(feature = "experimental-fp8")]
thread_local! {
    // Model weights can move from the loader to a backend worker. cuBLASLt
    // descriptors and mutable workspace stay on the executing thread instead.
    // One executor per thread bounds workspace usage independently of layer count.
    static FP8_EXECUTOR: std::cell::RefCell<Option<(candle::Device, candle_cublaslt::fp8::Fp8Matmul)>> = const { std::cell::RefCell::new(None) };
}

impl MlpLinear {
    pub(crate) fn new(weight: Tensor, enable_fp8_dynamic: bool) -> Result<Self> {
        if enable_fp8_dynamic {
            #[cfg(feature = "experimental-fp8")]
            return Ok(Self::Fp8(candle_cublaslt::fp8::Fp8Linear::new(&weight)?));
            #[cfg(not(feature = "experimental-fp8"))]
            candle::bail!("Dynamic FP8 requires an experimental-fp8 build");
        }
        Ok(Self::Dense(Linear::new(weight, None, None)))
    }

    pub(crate) fn forward_gated(&self, x: &Tensor, act: &super::HiddenAct) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(&super::gated_activation(x, Some(act))?),
            #[cfg(feature = "experimental-fp8")]
            Self::Fp8(linear) => {
                if matches!(act, super::HiddenAct::Silu) {
                    if let Some(y) =
                        with_fp8_executor(x, |executor| linear.forward_packed_swiglu(x, executor))?
                    {
                        return Ok(y);
                    }
                }
                let activated = super::gated_activation(x, Some(act))?;
                with_fp8_executor(&activated, |executor| linear.forward(&activated, executor))
            }
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            #[cfg(feature = "experimental-fp8")]
            Self::Fp8(linear) => with_fp8_executor(x, |executor| linear.forward(x, executor)),
        }
    }
}

#[cfg(feature = "experimental-fp8")]
fn with_fp8_executor<T>(
    x: &Tensor,
    f: impl FnOnce(&mut candle_cublaslt::fp8::Fp8Matmul) -> Result<T>,
) -> Result<T> {
    FP8_EXECUTOR.with(|slot| {
        let mut slot = slot.try_borrow_mut().map_err(candle::Error::wrap)?;
        if slot
            .as_ref()
            .is_none_or(|(device, _)| !device.same_device(x.device()))
        {
            *slot = Some((
                x.device().clone(),
                candle_cublaslt::fp8::Fp8Matmul::new(x.device())?,
            ));
        }
        let (_, executor) = slot.as_mut().expect("executor initialized above");
        f(executor)
    })
}

#[cfg(all(test, feature = "experimental-fp8"))]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    #[ignore = "requires a CUDA device"]
    fn disabled_fp8_preserves_dense_outputs_without_executor() -> Result<()> {
        // A fresh thread isolates the lazy executor from other GPU tests.
        std::thread::spawn(|| -> Result<()> {
            let device = Device::new_cuda(0)?;
            let weight = Tensor::arange(0f32, 256f32, &device)?
                .reshape((16, 16))?
                .affine(0.001, -0.1)?
                .to_dtype(DType::F16)?;
            let dense = Linear::new(weight.clone(), None, None);
            let disabled = MlpLinear::new(weight, false)?;
            assert!(matches!(&disabled, MlpLinear::Dense(_)));
            let x = Tensor::arange(0f32, 64f32, &device)?
                .reshape((4, 16))?
                .affine(0.01, -0.2)?
                .to_dtype(DType::F16)?;
            let values = |t: Tensor| t.to_dtype(DType::F32)?.to_vec2::<f32>();
            assert_eq!(values(disabled.forward(&x)?)?, values(dense.forward(&x)?)?);
            let packed = Tensor::cat(&[&x, &x], 1)?;
            let act = super::super::HiddenAct::Silu;
            assert_eq!(
                values(disabled.forward_gated(&packed, &act)?)?,
                values(dense.forward(&super::super::gated_activation(&packed, Some(&act))?)?)?
            );
            FP8_EXECUTOR.with(|slot| assert!(slot.borrow().is_none()));
            Ok(())
        })
        .join()
        .expect("disabled FP8 test thread panicked")
    }
}
