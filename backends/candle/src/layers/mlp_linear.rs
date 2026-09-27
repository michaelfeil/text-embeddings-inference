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
        #[cfg(feature = "experimental-fp8")]
        if matches!(act, super::HiddenAct::Silu) {
            // Lab-only control: default path remains the evaluated unfused recipe.
            static FUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *FUSE.get_or_init(|| std::env::var("TEI_PERF_FP8_SWIGLU").as_deref() == Ok("1")) {
                if let Self::Fp8(linear) = self {
                    if let Some(y) =
                        with_fp8_executor(x, |executor| linear.forward_packed_swiglu(x, executor))?
                    {
                        return Ok(y);
                    }
                }
            }
        }
        self.forward(&super::gated_activation(x, Some(act))?)
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
