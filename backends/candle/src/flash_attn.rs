use candle::Tensor;
thread_local! {
    static COMPUTE_CAPS: std::cell::RefCell<std::collections::HashMap<
        candle::cuda_backend::DeviceId, usize
    >> = std::cell::RefCell::new(std::collections::HashMap::new());
}

fn runtime_compute_cap(device: &candle::Device) -> candle::Result<usize> {
    let candle::Device::Cuda(cuda) = device else {
        candle::bail!("Flash attention requires a CUDA tensor");
    };
    COMPUTE_CAPS.with(|caps| {
        let mut caps = caps.borrow_mut();
        if let Some(&cap) = caps.get(&cuda.id()) {
            return Ok(cap);
        }
        use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
        };
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let major = context.attribute(MAJOR).map_err(candle::Error::wrap)?;
        let minor = context.attribute(MINOR).map_err(candle::Error::wrap)?;
        let cap = (major * 10 + minor) as usize;
        caps.insert(cuda.id(), cap);
        Ok(cap)
    })
}

#[allow(clippy::too_many_arguments, unused)]
pub(crate) fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    let runtime_compute_cap = runtime_compute_cap(q.device())?;

    if runtime_compute_cap == 75 {
        if alibi_slopes.is_some() {
            candle::bail!("Flash attention v1 does not support alibi");
        }
        if window_size_left.is_some() | window_size_right.is_some() {
            candle::bail!("Flash attention v1 does not support attention windowing");
        }

        #[cfg(feature = "flash-attn-v1")]
        {
            use candle_flash_attn_v1::flash_attn_varlen;
            return flash_attn_varlen(
                q,
                k,
                v,
                seqlens_q,
                seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
            );
        }
        #[cfg(not(feature = "flash-attn-v1"))]
        candle::bail!("Flash attention v1 is not installed. Use `flash-attn-v1` feature.")
    } else if (80..90).contains(&runtime_compute_cap)
        || runtime_compute_cap == 90
        || runtime_compute_cap == 120
        || runtime_compute_cap == 100
    {
        #[cfg(feature = "flash-attn")]
        {
            use candle_flash_attn::{flash_attn_varlen_alibi_windowed, flash_attn_varlen_windowed};

            let window_size_right = if causal {
                Some(0)
            } else if window_size_right.is_some() {
                window_size_right
            } else {
                None
            };

            let attention = if let Some(alibi_slopes) = alibi_slopes {
                flash_attn_varlen_alibi_windowed(
                    q,
                    k,
                    v,
                    alibi_slopes,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                )
            } else {
                flash_attn_varlen_windowed(
                    q,
                    k,
                    v,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                )
            };

            return attention;
        }
        #[cfg(not(feature = "flash-attn"))]
        candle::bail!("Flash attention is not installed. Use `flash-attn` feature.")
    }
    candle::bail!(
        "GPU with CUDA capability {} is not supported",
        runtime_compute_cap
    );
}
