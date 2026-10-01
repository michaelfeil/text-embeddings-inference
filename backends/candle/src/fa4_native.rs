//! Shared FA4 wrapper integration. Boundaries are validated once per model batch.
use candle::{DType, Result, Tensor};
use candle_flash_attn_v4::{AttentionConfig, Mask, Seqlens};
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

thread_local! {
    static LOGGED: Cell<u8> = const { Cell::new(0) };
    static BATCH: RefCell<Option<BatchLengths>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Auto,
    Fa2,
    Fa4,
}

// Process configuration is immutable after the first attention batch. Cache the
// environment lookup; tensor layouts and batch boundaries are still checked below.
fn backend() -> Result<Backend> {
    static BACKEND: OnceLock<std::result::Result<Backend, String>> = OnceLock::new();
    match BACKEND
        .get_or_init(|| parse_backend(std::env::var("ATTN_BACKEND")).map_err(|e| e.to_string()))
    {
        Ok(backend) => Ok(*backend),
        Err(message) => candle::bail!("{message}"),
    }
}
pub(crate) fn enabled() -> Result<bool> {
    Ok(backend()? != Backend::Fa2)
}

fn parse_backend(value: std::result::Result<String, std::env::VarError>) -> Result<Backend> {
    match value.as_deref() {
        Ok("auto") | Err(std::env::VarError::NotPresent) => Ok(Backend::Auto),
        Ok("fa4") => Ok(Backend::Fa4),
        Ok("fa2") => Ok(Backend::Fa2),
        _ => candle::bail!("ATTN_BACKEND must be auto, fa2 or fa4"),
    }
}

// The global d128 GQA2 family is model-qualified only in BF16 (Voyage).
// Explicit FA4 retains the existing FP16 opt-in; neither backend cures model
// FP16 overflow, so Voyage deployments should use BF16.
fn dtype_qualified(backend: Backend, dtype: DType, dim: usize, causal: bool) -> bool {
    !(backend == Backend::Auto && dtype == DType::F16 && dim == 128 && !causal)
}

struct BatchLengths {
    query: (Tensor, Seqlens),
    kv: Option<(Tensor, Seqlens)>,
}

pub(crate) struct BatchGuard {
    // None denotes an inactive scope; Some(None) an active scope with no parent.
    previous: Option<Option<BatchLengths>>,
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
    prepare_cross_batch(offsets, host_offsets, None)
}

pub(crate) fn prepare_cross_batch(
    offsets: &Tensor,
    host_offsets: &[u32],
    kv: Option<(&Tensor, &[u32])>,
) -> Result<BatchGuard> {
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
    let kv = kv
        .map(|(offsets, host)| -> Result<_> {
            Ok((offsets.clone(), Seqlens::new(host, offsets.device())?))
        })
        .transpose()?;
    let previous = BATCH.with(|batch| {
        batch.replace(Some(BatchLengths {
            query: (offsets.clone(), lengths),
            kv,
        }))
    });
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
    let backend = backend()?;
    if backend == Backend::Fa2 {
        return Ok(None);
    }
    let (_, h, d) = q.dims3()?;
    let (_, hk, kd) = k.dims3()?;
    if d != kd || !matches!(q.dtype(), DType::F16 | DType::BF16) {
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
    let mask = dtype_qualified(backend, q.dtype(), d, causal)
        .then(|| supported_mask(d, h, hk, causal, left, right))
        .flatten();
    let Some(mask) = mask else {
        LOGGED.with(|logged| {
            if logged.get() & 8 == 0 {
                tracing::info!(
                    heads = h,
                    kv_heads = hk,
                    dim = d,
                    causal,
                    ?left,
                    ?right,
                    dtype = ?q.dtype(), "FA4 shape/dtype not qualified; using FA2"
                );
                logged.set(logged.get() | 8);
            }
        });
        return Ok(None);
    };
    BATCH.with(|batch| {
        let batch = batch.borrow();
        let Some(lengths) = batch.as_ref() else {
            return Ok(None);
        };
        if lengths.kv.is_none() && offsets_q.id() != offsets_k.id() {
            return Ok(None);
        }
        let (registered_q, q_lengths) = &lengths.query;
        let (registered_kv, kv_lengths) = lengths.kv.as_ref().unwrap_or(&lengths.query);
        if registered_q.id() != offsets_q.id() || registered_kv.id() != offsets_k.id() {
            candle::bail!("FA4 batch boundary registration mismatch");
        }
        let output = candle_flash_attn_v4::flash_attn_varlen_cross(
            q,
            k,
            v,
            q_lengths,
            kv_lengths,
            AttentionConfig {
                mask,
                softmax_scale: Some(scale),
            },
        )?;
        let bit = if offsets_q.id() != offsets_k.id() {
            16
        } else {
            match mask {
                Mask::Global => 1,
                Mask::Causal => 2,
                _ => 4,
            }
        };
        LOGGED.with(|logged| {
            if logged.get() & bit == 0 {
                tracing::info!(
                    ?mask,
                    heads = h,
                    kv_heads = hk,
                    dim = d,
                    query_tokens = q.dims()[0],
                    kv_tokens = k.dims()[0],
                    "Shared FA4 wrapper active"
                );
                logged.set(logged.get() | bit);
            }
        });
        Ok(Some(output))
    })
}

// Four cheap shape/mask comparisons. Do not cache tensor layouts or sequence
// boundaries: those can change between batches even within the same layer.
fn supported_mask(
    d: usize,
    h: usize,
    hk: usize,
    causal: bool,
    left: Option<usize>,
    right: Option<usize>,
) -> Option<Mask> {
    if h == 0 || hk == 0 {
        return None;
    }
    match (d, h == hk, causal, left, right) {
        (64, true, false, None, None) => Some(Mask::Global),
        (64, true, false, Some(left), Some(right)) => Some(Mask::Window { left, right }),
        (128, _, true, None, None)
            if hk.checked_mul(4) == Some(h) || hk.checked_mul(2) == Some(h) =>
        {
            Some(Mask::Causal)
        }
        (128, _, false, None, None) if hk.checked_mul(2) == Some(h) => Some(Mask::Global),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn backend_selection_defaults_to_validated_auto() {
        assert_eq!(
            parse_backend(Err(std::env::VarError::NotPresent)).unwrap(),
            Backend::Auto
        );
        assert_eq!(parse_backend(Ok("fa2".into())).unwrap(), Backend::Fa2);
        assert_eq!(parse_backend(Ok("fa4".into())).unwrap(), Backend::Fa4);
        assert_eq!(parse_backend(Ok("auto".into())).unwrap(), Backend::Auto);
        for invalid in ["", "FA4", "fa3"] {
            assert!(parse_backend(Ok(invalid.into())).is_err());
        }
        assert!(parse_backend(Err(std::env::VarError::NotUnicode("invalid".into()))).is_err());
    }

    #[test]
    fn auto_keeps_unqualified_fp16_global_gqa_off_fa4() {
        assert!(!dtype_qualified(Backend::Auto, DType::F16, 128, false));
        assert!(dtype_qualified(Backend::Auto, DType::BF16, 128, false));
        assert!(dtype_qualified(Backend::Fa4, DType::F16, 128, false));
        assert!(dtype_qualified(Backend::Auto, DType::F16, 128, true));
        assert!(dtype_qualified(Backend::Auto, DType::F16, 64, false));
    }

    #[test]
    fn supported_shapes_preserve_mask_semantics() {
        assert!(matches!(
            supported_mask(64, 12, 12, false, None, None),
            Some(Mask::Global)
        ));
        assert!(matches!(
            supported_mask(64, 12, 12, false, Some(64), Some(64)),
            Some(Mask::Window {
                left: 64,
                right: 64
            })
        ));
        assert!(matches!(
            supported_mask(128, 16, 4, true, None, None),
            Some(Mask::Causal)
        ));
        assert!(matches!(
            supported_mask(128, 16, 8, false, None, None),
            Some(Mask::Global)
        ));
        assert!(matches!(
            supported_mask(128, 16, 8, true, None, None),
            Some(Mask::Causal)
        ));
        for shape in [
            (256, 3, 1, false, None, None),
            (128, 16, 16, true, None, None),
            (128, 16, 4, false, None, None),
            (64, 12, 12, true, None, None),
            (64, 12, 12, false, Some(64), None),
            (128, 16, 4, true, Some(64), Some(0)),
            (64, 0, 0, false, None, None),
        ] {
            assert!(supported_mask(shape.0, shape.1, shape.2, shape.3, shape.4, shape.5).is_none());
        }
    }

    #[test]
    #[ignore = "requires SM90, FA4 native bundle, and ATTN_BACKEND=fa4"]
    fn cached_prefix_uses_bottom_right_causal_mask() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let q_offsets = Tensor::new(&[0u32, 1, 3], &device)?;
        let kv_offsets = Tensor::new(&[0u32, 3, 7], &device)?;
        let q = Tensor::zeros((3, 16, 128), DType::F16, &device)?;
        let k = Tensor::zeros((7, 8, 128), DType::F16, &device)?;
        let values: Vec<f32> = [1., 2., 3., 10., 20., 30., 40.]
            .into_iter()
            .flat_map(|v| [v; 8 * 128])
            .collect();
        let v = Tensor::from_vec(values, (7, 8, 128), &device)?.to_dtype(DType::F16)?;
        let _guard = prepare_cross_batch(&q_offsets, &[0, 1, 3], Some((&kv_offsets, &[0, 3, 7])))?;
        let output = try_forward(&q, &k, &v, &q_offsets, &kv_offsets, 0.125, true, None, None)?
            .expect("cached-prefix attention must select FA4")
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (row, expected) in output.chunks_exact(16 * 128).zip([2., 20., 25.]) {
            assert!(row.iter().all(|v| (*v - expected).abs() < 0.02));
        }
        Ok(())
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
