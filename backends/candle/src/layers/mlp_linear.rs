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

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            #[cfg(feature = "experimental-fp8")]
            Self::Fp8(linear) => FP8_EXECUTOR.with(|slot| {
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
                linear.forward(x, executor)
            }),
        }
    }
}
