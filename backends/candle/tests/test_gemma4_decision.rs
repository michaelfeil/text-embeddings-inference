#![cfg(feature = "flash-attn")]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, DecisionPromptStyle, ModelType, Pool};
use tokenizers::Tokenizer;

fn option_batch(prompt: &[u32], options: &[Vec<u32>], compact: bool) -> Batch {
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
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    for option in options {
        let sequence = prompt.iter().chain(option);
        let length = (prompt.len() + option.len()) as u32;
        batch.input_ids.extend(sequence);
        batch.position_ids.extend(0..length);
        batch
            .token_type_ids
            .extend(std::iter::repeat_n(0, length as usize));
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(length);
    }
    if compact {
        let mut prefixes = HashMap::new();
        let mut ids = Vec::new();
        let mut positions = Vec::new();
        let mut scatter = Vec::new();
        let mut fold = Vec::new();
        for bounds in batch.cumulative_seq_lengths.windows(2) {
            let start = bounds[0] as usize;
            for pos in start..bounds[1] as usize {
                let prefix = batch.input_ids[start..=pos].to_vec();
                let next = ids.len() as u32;
                let row = *prefixes.entry(prefix).or_insert_with(|| {
                    ids.push(batch.input_ids[pos]);
                    positions.push(batch.position_ids[pos]);
                    fold.push(pos as u32);
                    next
                });
                scatter.push(row);
            }
        }
        batch.compact_input_ids = Some(ids);
        batch.compact_position_ids = Some(positions);
        batch.scatter_unfold = Some(scatter);
        batch.fold_gather = Some(fold);
    }
    batch
}

#[test]
#[ignore = "requires a local dense Gemma4 checkpoint and CUDA"]
fn gemma4_decision_scores_agree_with_unfolded_sequences() -> Result<()> {
    let root =
        PathBuf::from(std::env::var("GEMMA4_MODEL_ROOT").context("GEMMA4_MODEL_ROOT is not set")?);
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let backend = CandleBackend::new(
        &root,
        "bfloat16".into(),
        ModelType::Embedding(Pool::LastToken),
        None,
        0,
    )?;
    assert!(!backend.is_padded());
    assert!(backend.supports_decision_scoring());
    assert_eq!(
        backend.decision_prompt_style(),
        Some(DecisionPromptStyle::Gemma4)
    );
    let prompt =
        "<bos><|turn>user\nReply with a JSON object answering yes or no.<turn|>\n<|turn>model\n";
    let prompt = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .get_ids()
        .to_vec();
    assert_eq!(prompt[0], 2); // <bos>
    assert!(prompt.contains(&105)); // <|turn>
    assert!(prompt.contains(&106)); // <turn|>
    let options: Vec<Vec<u32>> = [r#"{"answer":"yes"}"#, r#"{"answer":"no"}"#]
        .into_iter()
        .map(|option| tokenizer.encode(option, false).unwrap().get_ids().to_vec())
        .collect();
    let mut embedding_batch = option_batch(&prompt, &options, false);
    embedding_batch.pooled_indices = vec![0, 1];
    assert_eq!(backend.embed(embedding_batch)?.len(), 2);
    let lengths = vec![prompt.len(); options.len()];
    let folded = backend.score_options(option_batch(&prompt, &options, true), &lengths)?;
    let unfolded = backend.score_options(option_batch(&prompt, &options, false), &lengths)?;
    assert_eq!(folded.len(), options.len());
    for (a, b) in folded.iter().zip(unfolded) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() < 0.5, "folded {a} versus unfolded {b}");
    }
    Ok(())
}
