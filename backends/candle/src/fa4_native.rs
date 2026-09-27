//! Shared FA4 wrapper integration. Boundaries are validated once per model batch.
use candle::{DType, Result, Tensor};
use candle_flash_attn_v4::{AttentionConfig, Mask, Seqlens};
use std::cell::{Cell, RefCell};

thread_local! {
    static LOGGED: Cell<u8> = const { Cell::new(0) };
    static BATCH: RefCell<Option<(Tensor, Seqlens)>> = const { RefCell::new(None) };
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
    if std::env::var("TEI_PERF_FA4").as_deref() != Ok("1") {
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
