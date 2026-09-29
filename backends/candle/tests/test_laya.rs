//! Set LAYA_CHECKPOINT_DIR to run this test against a local copy of the
//! published convaiinnovations/laya-typed-decisions checkpoint.
use anyhow::Result;
use candle::{DType, Device};
use text_embeddings_backend_candle::LayaModel;
use tokenizers::Tokenizer;

#[test]
fn laya_checkpoint_scores_all_markers() -> Result<()> {
    let Ok(path) = std::env::var("LAYA_CHECKPOINT_DIR") else {
        return Ok(());
    };
    let path = std::path::Path::new(&path);
    let (model, config) = LayaModel::from_model_dir(path, DType::F32, &Device::Cpu)?;
    let tokenizer = Tokenizer::from_file(path.join("tokenizer/tokenizer.json"))
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let cls = tokenizer.token_to_id("[CLS]").unwrap();
    let sep = tokenizer.token_to_id("[SEP]").unwrap();
    let mask = tokenizer.token_to_id("[MASK]").unwrap();
    let mut ids = vec![cls];
    ids.extend(
        tokenizer
            .encode("choice question: Which team?", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    let first = ids.len();
    ids.push(mask);
    ids.extend(
        tokenizer
            .encode(" billing", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    let second = ids.len();
    ids.push(mask);
    ids.extend(
        tokenizer
            .encode(" support", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    ids.extend(
        tokenizer
            .encode("I was billed twice", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    let result = model.forward(&ids, &[first, second], 0)?;
    assert_eq!(result.logits.len(), 2);
    assert!(result.logits.iter().all(|x| x.is_finite()));
    assert!((0.0..=1.0).contains(&result.action_probability));
    // Same token IDs and marker positions as the upstream Laya/PyTorch fixture.
    let reference = [2.1987615_f32, 0.16098782];
    for (actual, expected) in result.logits.iter().zip(reference) {
        assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
    }
    assert!(config.temperature_for(0, 2).is_finite());
    Ok(())
}
