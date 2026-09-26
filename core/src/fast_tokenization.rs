//! BPE ID encoding for embeddings. HF remains responsible for request-specific
//! truncation and post-processing, and for endpoints requiring source alignment.
use tokenizers::{Encoding, Tokenizer};

pub(crate) struct FastTokenizer {
    tokenizer: fastokens::Tokenizer,
    pool: rayon::ThreadPool,
}

pub(crate) fn load(tokenizer: &Tokenizer, workers: usize) -> Option<FastTokenizer> {
    if !matches!(
        tokenizer.get_model(),
        tokenizers::models::ModelWrapper::BPE(_)
    ) {
        return None;
    }
    // Serialize the configured tokenizer, including any server-added processors.
    // Unsupported models/normalizers simply retain the existing implementation.
    let result = serde_json::to_value(tokenizer)
        .map_err(|e| e.to_string())
        .and_then(|json| {
            if json["model"]["dropout"].as_f64().is_some_and(|p| p > 0.0) {
                return Err("BPE dropout requires the Hugging Face tokenizer".to_string());
            }
            let tokenizer = fastokens::Tokenizer::from_json(json).map_err(|e| e.to_string())?;
            // Fastokens parallelizes splits internally. Bound it to the configured
            // worker budget instead of initializing Rayon's host-wide global pool.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers.max(1))
                .build()
                .map_err(|e| e.to_string())?;
            Ok(FastTokenizer { tokenizer, pool })
        });
    match result {
        Ok(tokenizer) => {
            tracing::info!("Using fastokens BPE encoding for embedding text inputs");
            Some(tokenizer)
        }
        Err(err) => {
            tracing::debug!("Using Hugging Face tokenizer: {err}");
            None
        }
    }
}

pub(crate) fn encode(
    fast: &FastTokenizer,
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> tokenizers::Result<Encoding> {
    // Do not truncate final IDs: HF reserves space for special tokens before
    // truncating content, and applies configured type IDs and post-processors.
    let ids = fast.pool.install(|| fast.tokenizer.encode(text))?;
    let len = ids.len();
    let raw = Encoding::new(
        ids,
        vec![0; len],
        vec![String::new(); len],
        vec![None; len],
        vec![(0, 0); len],
        vec![0; len],
        vec![1; len],
        Vec::new(),
        Default::default(),
    );
    // Alignment placeholders only support HF's Encoding transformations. The
    // caller discards these fields; NER and /tokenize never enter this path.
    tokenizer.post_process(raw, None, add_special_tokens)
}

#[cfg(test)]
pub(crate) fn test_tokenizer() -> Tokenizer {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let vocab: serde_json::Map<String, serde_json::Value> = alphabet
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), serde_json::json!(i)))
        .collect();
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":true},
        "post_processor": null, "decoder": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":true},
        "model": {"type":"BPE", "vocab":vocab, "merges":[], "dropout":null,
            "unk_token":null, "continuing_subword_prefix":null, "end_of_word_suffix":null,
            "fuse_unk":false, "byte_fallback":false, "ignore_merges":false}
    });
    Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, TruncationDirection, TruncationParams};

    #[test]
    fn preserves_postprocessing_and_truncation() {
        let mut tokenizer = test_tokenizer();
        tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ]);
        let fast = load(&tokenizer, 2).unwrap();
        // A processor installed after fastokens initialization must still apply.
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<s>:2 $A:3 </s>:2")
                .unwrap()
                .special_tokens(vec![("<s>", 256), ("</s>", 257)])
                .build()
                .unwrap(),
        ));
        for direction in [TruncationDirection::Left, TruncationDirection::Right] {
            for limit in [2, 3, 8, 128] {
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: limit,
                        direction,
                        ..Default::default()
                    }))
                    .unwrap();
                for text in ["hello world", " Café 東京 👩🏽‍💻", "<s> hello </s>", ""] {
                    for special in [true, false] {
                        let expected = tokenizer.encode(text, special).unwrap();
                        let actual = encode(&fast, &tokenizer, text, special).unwrap();
                        assert_eq!(
                            actual.get_ids(),
                            expected.get_ids(),
                            "{text:?}, {direction:?}, {limit}"
                        );
                        assert_eq!(actual.get_type_ids(), expected.get_type_ids());
                        assert_eq!(actual.get_attention_mask(), expected.get_attention_mask());
                        assert_eq!(
                            actual.get_special_tokens_mask(),
                            expected.get_special_tokens_mask()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unsupported_models_and_dropout_use_hf() {
        let wordpiece = Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default());
        assert!(load(&wordpiece, 1).is_none());
        let mut json = serde_json::to_value(test_tokenizer()).unwrap();
        json["model"]["dropout"] = serde_json::json!(0.1);
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(load(&tokenizer, 1).is_none());
    }
}
