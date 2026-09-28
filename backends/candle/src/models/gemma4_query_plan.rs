//! Query compaction for causal attention over a prefix-folded batch.
//!
//! Each compact token needs only the attention result at its canonical expanded
//! occurrence. Keep the suffix containing those occurrences in each sequence,
//! and retain the original full KV sequence. Flash attention's bottom-right
//! causal alignment then addresses exactly the same keys. Rounding the suffix
//! start down to a query-tile boundary retains the original FA2 tiling too.
//! K/V remain expanded: this changes query work, not KV sharing or positions.
#[cfg(feature = "flash-attn")]
use candle::{Device, Result, Tensor};
#[cfg(feature = "flash-attn")]
use text_embeddings_backend_core::Batch;

#[cfg(feature = "flash-attn")]
pub(super) struct QueryPlan {
    pub queries: Tensor,
    pub output: Tensor,
    pub cu_queries: Tensor,
    pub max_queries: usize,
}

#[cfg(feature = "flash-attn")]
impl QueryPlan {
    pub fn new(batch: &Batch, device: &Device) -> Result<Option<Self>> {
        let (Some(ids), Some(positions), Some(scatter), Some(fold)) = (
            &batch.compact_input_ids,
            &batch.compact_position_ids,
            &batch.scatter_unfold,
            &batch.fold_gather,
        ) else {
            return Ok(None);
        };
        // Match CompactUnfoldTensors: a partial set of compact metadata uses
        // the original expanded input and cannot use a compact query plan.
        if ids.len() != positions.len()
            || ids.len() != fold.len()
            || scatter.len() != batch.input_ids.len()
        {
            return Ok(None);
        }
        let Some(RawPlan {
            queries,
            output,
            cu_queries,
            max_queries,
        }) = plan(scatter, fold, &batch.cumulative_seq_lengths)
        else {
            return Ok(None);
        };
        tracing::debug!(
            expanded_queries = scatter.len(),
            retained_queries = queries.len(),
            "Compacting Gemma4 causal attention queries"
        );
        Ok(Some(Self {
            queries: Tensor::new(queries.as_slice(), device)?,
            output: Tensor::new(output.as_slice(), device)?,
            cu_queries: Tensor::new(cu_queries.as_slice(), device)?,
            max_queries,
        }))
    }
}

struct RawPlan {
    queries: Vec<u32>,
    output: Vec<u32>,
    cu_queries: Vec<u32>,
    max_queries: usize,
}

fn plan(scatter: &[u32], fold: &[u32], cu: &[u32]) -> Option<RawPlan> {
    if cu.first() != Some(&0) || cu.last().copied()? as usize != scatter.len() || fold.is_empty() {
        return None;
    }
    if scatter.iter().any(|&i| i as usize >= fold.len()) {
        return None;
    }
    let mut canonical = vec![None; scatter.len()];
    for (compact, &expanded) in fold.iter().enumerate() {
        let e = expanded as usize;
        if e >= scatter.len() || scatter[e] as usize != compact || canonical[e].is_some() {
            return None;
        }
        canonical[e] = Some(compact);
    }
    let mut queries = Vec::new();
    let mut output = vec![u32::MAX; fold.len()];
    let mut cu_queries = vec![0];
    let mut max_queries = 0;
    for seq in cu.windows(2) {
        let (start, end) = (seq[0] as usize, seq[1] as usize);
        if start >= end || end > scatter.len() {
            return None;
        }
        let first = canonical[start..end]
            .iter()
            .position(Option::is_some)
            .unwrap_or(end - start - 1);
        // Preserve the original absolute query-tile alignment (FA2 M=64/128).
        // This retains some duplicate queries, but avoids shifting reduction
        // boundaries and keeps every sequence nonempty, including duplicates.
        let q_start = start + first / 128 * 128;
        let base = queries.len();
        for (index, item) in canonical[q_start..end].iter().enumerate() {
            if let Some(compact) = item {
                output[*compact] = u32::try_from(base + index).ok()?;
            }
        }
        queries.extend_from_slice(&scatter[q_start..end]);
        cu_queries.push(u32::try_from(queries.len()).ok()?);
        max_queries = max_queries.max(end - q_start);
    }
    if queries.len() >= scatter.len() || output.contains(&u32::MAX) {
        return None;
    }
    Some(RawPlan {
        queries,
        output,
        cu_queries,
        max_queries,
    })
}

#[cfg(test)]
mod tests {
    use super::{plan, RawPlan};

    #[test]
    fn query_plan_retains_canonical_rows_and_causal_positions() {
        for prefix in [0, 1, 127, 128, 129, 260, 3900] {
            for duplicate in [false, true] {
                let mut scatter = Vec::new();
                let mut fold = Vec::new();
                let mut cu = vec![0];
                let mut next = prefix;
                for (seq, tail) in [3, 7, 10].into_iter().enumerate() {
                    for token in 0..prefix {
                        if seq == 0 {
                            fold.push(scatter.len() as u32);
                        }
                        scatter.push(token as u32);
                    }
                    for _ in 0..tail {
                        fold.push(scatter.len() as u32);
                        scatter.push(next as u32);
                        next += 1;
                    }
                    cu.push(scatter.len() as u32);
                }
                if duplicate {
                    scatter.extend_from_within(0..cu[1] as usize);
                    cu.push(scatter.len() as u32);
                }
                let Some(RawPlan {
                    queries,
                    output,
                    cu_queries: qcu,
                    max_queries: maxq,
                }) = plan(&scatter, &fold, &cu)
                else {
                    assert!(prefix < 128);
                    continue;
                };
                assert!(queries.len() < scatter.len());
                assert_eq!(output.len(), fold.len());
                assert_eq!(qcu.len(), cu.len());
                assert_eq!(
                    maxq,
                    qcu.windows(2)
                        .map(|x| (x[1] - x[0]) as usize)
                        .max()
                        .unwrap()
                );
                for (compact, &q) in output.iter().enumerate() {
                    assert_eq!(queries[q as usize] as usize, compact);
                    let seq = qcu.windows(2).position(|x| x[0] <= q && q < x[1]).unwrap();
                    let skipped = (cu[seq + 1] - cu[seq]) - (qcu[seq + 1] - qcu[seq]);
                    assert_eq!(skipped % 128, 0);
                    // Bottom-right causal alignment addresses the same KV prefix.
                    assert_eq!(cu[seq] + skipped + q - qcu[seq], fold[compact]);
                }
            }
        }
    }

    #[test]
    fn malformed_query_maps_fall_back() {
        assert!(plan(&[0], &[1], &[0, 1]).is_none());
        assert!(plan(&[0], &[0], &[1, 1]).is_none());
        assert!(plan(&[0], &[0], &[0, 2]).is_none());
        assert!(plan(&[0, 1], &[0, 0], &[0, 2]).is_none());
        assert!(plan(&[0, 2], &[0, 1], &[0, 2]).is_none());
    }
}
