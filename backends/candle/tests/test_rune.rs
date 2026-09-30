#![cfg(feature = "flash-attn")]

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, DecisionInput, ModelType};

fn batch(rows: &[Vec<u32>]) -> Batch {
    let mut batch = Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        fold_gather: None,
        scatter_unfold: None,
        tokens: vec![],
        offsets: vec![],
    };
    for row in rows {
        batch.input_ids.extend(row);
        batch.token_type_ids.extend(vec![0; row.len()]);
        batch.position_ids.extend(0..row.len() as u32);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(row.len() as u32);
    }
    batch
}

#[test]
#[ignore = "requires Rune checkpoint, independent reference fixture and CUDA"]
fn rune_selected_logits_batch_and_radix_match_reference() -> Result<()> {
    let root = PathBuf::from(std::env::var("RUNE_CHECKPOINT_DIR")?);
    let fixture: Value = serde_json::from_slice(&std::fs::read(std::env::var("RUNE_FIXTURE")?)?)?;
    let reference: Value =
        serde_json::from_slice(&std::fs::read(std::env::var("RUNE_REFERENCE")?)?)?;
    let rows: Vec<Vec<u32>> = fixture["sequences"]
        .as_array()
        .context("sequences")?
        .iter()
        .map(|s| serde_json::from_value(s["ids"].clone()))
        .collect::<Result<_, _>>()?;
    let inputs: Vec<DecisionInput> = fixture["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            Ok(DecisionInput::OptionTokens {
                token_ids: serde_json::from_value(s["option_token_ids"].clone())?,
            })
        })
        .collect::<Result<_>>()?;
    let backend = CandleBackend::new(&root, "bfloat16".into(), ModelType::Decision, None, 0)?;
    assert!(backend.supports_radix_mlp());
    let plain = backend.decide(batch(&rows), inputs.clone())?;
    let prefix = (0..rows.iter().map(Vec::len).min().unwrap())
        .take_while(|&i| rows.iter().all(|r| r[i] == rows[0][i]))
        .count();
    assert!(prefix > 0);
    let mut folded = batch(&rows);
    let mut gather = Vec::new();
    let mut scatter = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let start = folded.cumulative_seq_lengths[i];
        for position in 0..row.len() {
            if i > 0 && position < prefix {
                scatter.push(position as u32);
            } else {
                scatter.push(gather.len() as u32);
                gather.push(start + position as u32);
            }
        }
    }
    folded.compact_input_ids = Some(
        gather
            .iter()
            .map(|&i| folded.input_ids[i as usize])
            .collect(),
    );
    folded.compact_position_ids = Some(
        gather
            .iter()
            .map(|&i| folded.position_ids[i as usize])
            .collect(),
    );
    folded.fold_gather = Some(gather);
    folded.scatter_unfold = Some(scatter);
    let compact = backend.decide(folded, inputs.clone())?;
    let compare = |actual: &[f32], expected: &[f32], context: &str| {
        assert_eq!(actual.len(), expected.len());
        // BF16 MoE routing and attention differ from Transformers SDPA, and
        // changing the token batch changes GEMM rounding. Bound the scores as
        // well as the distribution, and require identical fixture decisions.
        let probabilities = |logits: &[f32]| {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f64> = logits
                .iter()
                .map(|&x| (x as f64 - max as f64).exp())
                .collect();
            let sum: f64 = p.iter().sum();
            p.iter_mut().for_each(|x| *x /= sum);
            p
        };
        let a = probabilities(actual);
        let b = probabilities(expected);
        let mode = |p: &[f64]| {
            p.iter()
                .enumerate()
                .fold(0, |best, (i, &p_i)| if p_i > p[best] { i } else { best })
        };
        assert_eq!(mode(&a), mode(&b), "{context}: decision changed");
        for ((&x, &y), (&p, &q)) in actual.iter().zip(expected).zip(a.iter().zip(&b)) {
            assert!((x - y).abs() <= 1.0, "{context}: logit {x} vs {y}");
            assert!((p - q).abs() <= 0.03, "{context}: probability {p} vs {q}");
        }
    };
    for (i, row) in rows.iter().enumerate() {
        let single = backend.decide(batch(std::slice::from_ref(row)), vec![inputs[i].clone()])?;
        let expected: Vec<f32> =
            serde_json::from_value(reference["sequences"][i]["logits"].clone())?;
        println!(
            "{} reference={expected:?} single={:?} batch={:?} radix={:?}",
            fixture["sequences"][i]["id"], single[0].logits, plain[i].logits, compact[i].logits
        );
        for (name, output) in [
            ("single", &single[0]),
            ("batch", &plain[i]),
            ("radix", &compact[i]),
        ] {
            compare(&output.logits, &expected, &format!("{name} row {i}"));
        }
        assert_eq!(compact[i].logits, plain[i].logits, "Radix row {i}");
    }
    Ok(())
}
