//! Opt-in full-encoder timing harness; see docs/modernbert-performance.md.
use super::{ModernBertConfig, ModernBertModel};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use std::{path::PathBuf, time::Instant};
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[test]
#[ignore = "requires a local checkpoint and an idle CUDA GPU"]
#[cfg(feature = "cuda")]
fn modernbert_encoder_latency() -> anyhow::Result<()> {
    let checkpoint = PathBuf::from(std::env::var("MODERNBERT_BENCH_CHECKPOINT")?);
    let output = PathBuf::from(std::env::var("MODERNBERT_BENCH_OUTPUT")?);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("encoder/config.json"))?)?;
    // Recent Transformers serializes these under rope_parameters. Normalize to
    // the older ModernBERT configuration read by main, preserving their values.
    for (field, kind) in [
        ("global_rope_theta", "full_attention"),
        ("local_rope_theta", "sliding_attention"),
    ] {
        if config.get(field).is_none() {
            config[field] = config["rope_parameters"][kind]["rope_theta"].clone();
        }
    }
    let config: ModernBertConfig = serde_json::from_value(config)?;
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[checkpoint.join("model.safetensors")],
            dtype,
            &device,
        )?
    };
    let model = ModernBertModel::load(vb.pp("encoder"), &config, ModelType::Embedding(Pool::Cls))?;
    let mut results = Vec::new();
    let cases: Vec<_> = [128usize, 512, 1024]
        .into_iter()
        .flat_map(|tokens| {
            [1usize, 8, 32]
                .into_iter()
                .map(move |batch| (tokens, batch, false))
        })
        .chain([(1024, 8, true)])
        .collect();
    for (tokens, count, mixed_lengths) in cases {
        let make_batch = || {
            let lengths: Vec<usize> = (0..count)
                .map(|i| {
                    if mixed_lengths && i % 2 == 0 {
                        tokens / 2
                    } else {
                        tokens
                    }
                })
                .collect();
            let mut offsets = vec![0u32];
            for &length in &lengths {
                offsets.push(offsets.last().unwrap() + length as u32);
            }
            Batch {
                input_ids: lengths
                    .iter()
                    .flat_map(|&length| (0..length).map(|i| 100 + i as u32 % 28))
                    .collect(),
                token_type_ids: vec![0; lengths.iter().sum()],
                position_ids: lengths
                    .iter()
                    .flat_map(|&length| 0..length as u32)
                    .collect(),
                cumulative_seq_lengths: offsets,
                max_length: tokens as u32,
                pooled_indices: (0..count as u32).collect(),
                raw_indices: vec![],
            }
        };
        for _ in 0..3 {
            let _ = model.forward(make_batch())?;
            device.synchronize()?;
        }
        let mut samples = Vec::new();
        for _ in 0..20 {
            let batch = make_batch();
            device.synchronize()?;
            let start = Instant::now();
            let result = model.forward(batch)?;
            device.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1000.);
            assert!(result.0.is_some());
        }
        samples.sort_by(f64::total_cmp);
        let row = serde_json::json!({"max_tokens":tokens,"batch_size":count,"mixed_lengths":mixed_lengths,
            "p50_ms":(samples[9]+samples[10])/2.,"p95_ms":samples[18],"samples_ms":samples});
        eprintln!("{row}");
        results.push(row);
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope":"dense ModernBERT encoder forward including mask construction and CLS pooling; excludes tokenization, queue, HTTP, and decision head",
                "dtype":"bfloat16","warmups":3,"samples":20,"concurrency":1,
                "input":"deterministic synthetic token IDs; alternating half/full lengths in mixed case",
                "layers":config.num_hidden_layers,"hidden_size":config.hidden_size,"results":results
            }))?,
        )?;
    }
    Ok(())
}
