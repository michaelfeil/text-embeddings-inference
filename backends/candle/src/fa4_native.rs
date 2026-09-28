//! Shared FA4 wrapper integration. Boundaries are validated once per model batch.
use candle::{DType, Result, Tensor};
use candle_flash_attn_v4::{AttentionConfig, Mask, Seqlens};
use std::cell::{Cell, RefCell};

thread_local! {
    static LOGGED: Cell<u8> = const { Cell::new(0) };
    static BATCH: RefCell<Option<(Tensor, Seqlens)>> = const { RefCell::new(None) };
}

// FA4 is experimental and must be selected explicitly.
pub(crate) fn enabled() -> Result<bool> {
    parse_backend(std::env::var("ATTN_BACKEND"))
}

fn parse_backend(value: std::result::Result<String, std::env::VarError>) -> Result<bool> {
    match value.as_deref() {
        Ok("fa4") => Ok(true),
        Ok("fa2") | Err(std::env::VarError::NotPresent) => Ok(false),
        _ => candle::bail!("ATTN_BACKEND must be fa2 or fa4"),
    }
}

pub(crate) struct BatchGuard {
    // None denotes an inactive scope; Some(None) an active scope with no parent.
    previous: Option<Option<(Tensor, Seqlens)>>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for BatchGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            BATCH.with(|batch| *batch.borrow_mut() = previous);
        }
    }
}

pub(crate) fn prepare_batch(offsets: &Tensor, host_offsets: &[u32]) -> Result<BatchGuard> {
    if !enabled()?
        || !offsets.device().is_cuda()
        || crate::flash_attn::runtime_compute_cap(offsets.device())?
            != candle_flash_attn_v4::compiled_compute_capability() as usize
    {
        return Ok(BatchGuard {
            previous: None,
            _thread: std::marker::PhantomData,
        });
    }
    let lengths = Seqlens::new(host_offsets, offsets.device())?;
    let previous = BATCH.with(|batch| batch.replace(Some((offsets.clone(), lengths))));
    Ok(BatchGuard {
        previous: Some(previous),
        _thread: std::marker::PhantomData,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offsets_q: &Tensor,
    offsets_k: &Tensor,
    scale: f32,
    causal: bool,
    left: Option<usize>,
    right: Option<usize>,
) -> Result<Option<Tensor>> {
    let (_, h, d) = q.dims3()?;
    let (_, hk, kd) = k.dims3()?;
    if offsets_q.id() != offsets_k.id() || d != kd || !matches!(q.dtype(), DType::F16 | DType::BF16)
    {
        return Ok(None);
    }
    if v.dims() != k.dims() || h == 0 || hk == 0 {
        return Ok(None);
    }
    for t in [q, k, v] {
        if t.dtype() != q.dtype()
            || !t.device().same_device(q.device())
            || t.stride()[2] != 1
            || t.stride()[1] != t.dims()[2]
            || t.stride()[0] < t.dims()[1] * t.dims()[2]
            || !t.stride()[0].is_multiple_of(8)
            || !t.layout().start_offset().is_multiple_of(8)
        {
            return Ok(None);
        }
    }
    let mask = match (d, h == hk, causal, left, right) {
        (64, true, false, None, None) => Mask::Global,
        (64, true, false, Some(left), Some(right)) => Mask::Window { left, right },
        (128, _, true, None, None) if hk > 0 && h == 4 * hk => Mask::Causal,
        _ => return Ok(None),
    };
    BATCH.with(|batch| {
        let batch = batch.borrow();
        let Some((offsets, lengths)) = batch.as_ref() else {
            return Ok(None);
        };
        if offsets.id() != offsets_q.id() {
            candle::bail!("FA4 batch boundary registration mismatch");
        }
        let output = candle_flash_attn_v4::flash_attn_varlen_with_config(
            q,
            k,
            v,
            lengths,
            AttentionConfig {
                mask,
                softmax_scale: Some(scale),
            },
        )?;
        let bit = match mask {
            Mask::Global => 1,
            Mask::Causal => 2,
            _ => 4,
        };
        LOGGED.with(|logged| {
            if logged.get() & bit == 0 {
                tracing::info!(
                    ?mask,
                    heads = h,
                    kv_heads = hk,
                    dim = d,
                    "Shared FA4 wrapper active"
                );
                logged.set(logged.get() | bit);
            }
        });
        Ok(Some(output))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn backend_selection_requires_explicit_opt_in() {
        assert!(!parse_backend(Err(std::env::VarError::NotPresent)).unwrap());
        assert!(!parse_backend(Ok("fa2".into())).unwrap());
        assert!(parse_backend(Ok("fa4".into())).unwrap());
        for invalid in ["auto", "", "FA4", "fa3"] {
            assert!(parse_backend(Ok(invalid.into())).is_err());
        }
        assert!(parse_backend(Err(std::env::VarError::NotUnicode("invalid".into()))).is_err());
    }

    #[test]
    #[ignore = "requires SM90, FA4 native bundle, and ATTN_BACKEND=fa4"]
    fn ragged_metadata_scopes_restore_on_return_and_error() -> Result<()> {
        let device = Device::new_cuda(0)?;
        assert!(enabled()?);
        let offsets = Tensor::new(&[0u32, 1, 5], &device)?;
        let qk = Tensor::zeros((5, 1, 64), DType::F16, &device)?;
        let values: Vec<f32> = [9., 1., 2., 3., 4.]
            .into_iter()
            .flat_map(|v| [v; 64])
            .collect();
        let v = Tensor::from_vec(values, (5, 1, 64), &device)?.to_dtype(DType::F16)?;
        let run = || try_forward(&qk, &qk, &v, &offsets, &offsets, 0.125, false, None, None);
        assert!(run()?.is_none());
        {
            let _outer = prepare_batch(&offsets, &[0, 1, 5])?;
            let expected = run()?
                .expect("explicit opt-in must select FA4")
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(expected[..64].iter().all(|v| (*v - 9.).abs() < 0.001));
            assert!(expected[64..].iter().all(|v| (*v - 2.5).abs() < 0.001));
            assert!(prepare_batch(&offsets, &[0, 4, 3]).is_err());
            assert!(run()?.is_some());
            {
                let inner_offsets = Tensor::new(&[0u32, 5], &device)?;
                let _inner = prepare_batch(&inner_offsets, &[0, 5])?;
                assert!(run().is_err(), "stale sequence boundaries must not be used");
            }
            let restored = run()?
                .unwrap()
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(expected, restored);
        }
        assert!(
            run()?.is_none(),
            "a completed batch must not leak its metadata"
        );
        Ok(())
    }
}
