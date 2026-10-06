use super::*;

fn batch(sequences: &[Vec<u32>]) -> Batch {
    let mut b = Batch {
        multimodal: vec![],
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    for s in sequences {
        b.input_ids.extend(s);
        b.token_type_ids.extend(vec![0; s.len()]);
        b.position_ids.extend(0..s.len() as u32);
        b.cumulative_seq_lengths.push(b.input_ids.len() as u32);
        b.max_length = b.max_length.max(s.len() as u32);
    }
    b
}

#[test]
fn paged_admission_eviction_and_failure_match_packed() -> Result<()> {
    use candle_flash_attn_v4::{flash_attn_varlen_cross, AttentionConfig, Mask, Seqlens};
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::F16, DType::BF16] {
        let mut cache = PrefixKvCache::configured(256, 192, true)?;
        let a: Vec<u32> = (1..=151).collect();
        let mut b = a.clone();
        b[140] = 900;
        let c: Vec<u32> = (300..=500).collect();
        let calls = vec![
            vec![a.clone()],
            vec![b.clone(), a.clone()],
            vec![c.clone()],
            vec![a.clone(), c.clone()],
            vec![b.clone()],
            vec![a.clone()],
        ];
        let mut retained = vec![];
        for (step, sequences) in calls.iter().enumerate() {
            let batch = batch(sequences);
            let plan = cache.plan(&batch, &device, dtype, 2, 2, 128)?.unwrap();
            if step == 1 {
                assert_eq!(
                    plan.hits, 256,
                    "both sequences must reuse their shared pages"
                );
            }
            if step == 4 {
                assert_eq!(
                    plan.hits, 0,
                    "failed forward must invalidate resident metadata"
                );
            }
            let all_lengths = Seqlens::new(&batch.cumulative_seq_lengths, &device)?;
            let query_lengths = Seqlens::new(&plan.suffix.cumulative_seq_lengths, &device)?;
            let make = |ids: &[u32], heads: usize, shift: f32| -> Result<Tensor> {
                let values: Vec<f32> = ids
                    .iter()
                    .flat_map(|&id| {
                        (0..heads * 128).map(move |i| {
                            ((id as f32 * 0.17 + i as f32 * 0.03 + shift).sin()) * 0.3
                        })
                    })
                    .collect();
                Tensor::from_vec(values, (ids.len(), heads, 128), &device)?.to_dtype(dtype)
            };
            for layer in 0..2 {
                let q = make(&plan.suffix.input_ids, 4, 0.)?;
                let k = make(&plan.suffix.input_ids, 2, layer as f32)?;
                let v = make(&plan.suffix.input_ids, 2, layer as f32 + 1.)?;
                let full_k = make(&batch.input_ids, 2, layer as f32)?;
                let full_v = make(&batch.input_ids, 2, layer as f32 + 1.)?;
                let actual = cache
                    .paged_attention(layer, &plan, &q, &k, &v, 1. / 128f32.sqrt())?
                    .unwrap();
                let expected = flash_attn_varlen_cross(
                    &q,
                    &full_k,
                    &full_v,
                    &query_lengths,
                    &all_lengths,
                    AttentionConfig {
                        mask: Mask::Causal,
                        softmax_scale: None,
                    },
                )?;
                let error = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 0.003,
                    "{dtype:?} call {step} layer {layer}: {error}"
                );
                cache.capture_paged(layer, &plan, &k, &v)?;
            }
            // Simulate a failed forward after it has already overwritten KV.
            if step == 3 {
                continue;
            }
            let output =
                make(&plan.suffix.input_ids, 1, 3.)?.reshape((plan.suffix.input_ids.len(), 128))?;
            let full = cache.finish(&plan, &batch, output)?;
            let expected = make(&batch.input_ids, 1, 3.)?.reshape((batch.input_ids.len(), 128))?;
            retained.push((full, expected));
        }
        for (actual, expected) in retained {
            assert_eq!(
                (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?,
                0.
            );
        }
    }
    Ok(())
}
