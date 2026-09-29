/// Payload tokenization logic
use crate::TextEmbeddingsError;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;
pub use tokenizers::Encoding as RawEncoding;
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::oneshot;
use tracing::{instrument, Span};

static MAX_CHAR_MULTIPLIER: usize = 250;

/// Validation
#[derive(Debug, Clone)]
pub struct Tokenization {
    /// Channel to communicate with the background tokenization task
    sender: async_channel::Sender<TokenizerRequest>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SimpleToken {
    pub id: u32,
    pub text: String,
    pub special: bool,
    pub start: Option<usize>,
    pub stop: Option<usize>,
}

impl Tokenization {
    pub fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
    ) -> Self {
        tracing::info!("Starting {workers} tokenization workers");

        // Create channel
        let (sender, receiver) = async_channel::bounded(workers * 4);

        // Spawn a background thread that creates all workers
        // since tokenizer.clone() require 0.2s.
        std::thread::spawn(move || {
            let fast_tokenizer = crate::fast_tokenization::load(&tokenizer, workers).map(Arc::new);
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let fast_tokenizer = fast_tokenizer.clone();
                let receiver_clone = receiver.clone();
                let default_prompt_clone = default_prompt.clone();
                let prompts_clone = prompts.clone();
                std::thread::spawn(move || {
                    tokenizer_worker(
                        tokenizer_clone,
                        fast_tokenizer,
                        max_input_length,
                        position_offset,
                        default_prompt_clone,
                        prompts_clone,
                        receiver_clone,
                    )
                });
            }
        });

        Self { sender }
    }

    #[instrument(skip_all)]
    pub async fn encode(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        self.encode_request(inputs, truncate, truncation_direction, prompt_name, false)
            .await
    }

    /// Encode embedding inputs without requesting token text or source offsets.
    #[instrument(skip_all)]
    pub async fn encode_embedding(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        self.encode_request(inputs, truncate, truncation_direction, prompt_name, true)
            .await
    }

    async fn encode_request(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        embedding: bool,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
        // Check if inputs is empty
        if inputs.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`inputs` cannot be empty".to_string(),
            ));
        }

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send(TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                embedding,
                response_sender,
                Span::current(),
            ))
            .await
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }

    #[instrument(skip_all)]
    pub async fn tokenize(
        &self,
        inputs: EncodingInput,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
        // Check if inputs is empty
        if inputs.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`inputs` cannot be empty".to_string(),
            ));
        }

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send(TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_sender,
                Span::current(),
            ))
            .await
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }

    #[instrument(skip_all)]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        // Check if input_ids is empty
        if ids.is_empty() {
            return Err(TextEmbeddingsError::Empty(
                "`input_ids` cannot be empty".to_string(),
            ));
        }

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send(TokenizerRequest::Decode(
                ids,
                skip_special_tokens,
                response_sender,
                Span::current(),
            ))
            .await
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }
}

/// Start tokenization workers
#[allow(clippy::too_many_arguments)]
fn tokenizer_worker(
    mut tokenizer: Tokenizer,
    fast_tokenizer: Option<Arc<crate::fast_tokenization::FastTokenizer>>,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompts: Option<HashMap<String, String>>,
    receiver: async_channel::Receiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Ok(request) = receiver.recv_blocking() {
        match request {
            TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                embedding,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(encode_input(
                            inputs,
                            truncate,
                            truncation_direction,
                            max_input_length,
                            position_offset,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                            if embedding {
                                fast_tokenizer.as_deref()
                            } else {
                                None
                            },
                        ));
                    }
                })
            }
            TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(tokenize_input(
                            inputs,
                            add_special_tokens,
                            max_input_length,
                            None,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                            None,
                        ));
                    }
                })
            }
            TokenizerRequest::Decode(ids, skip_special_tokens, response_tx, parent_span) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ =
                            response_tx.send(decode_ids(ids, skip_special_tokens, &mut tokenizer));
                    }
                })
            }
        }
    }
}

fn decode_ids(
    ids: Vec<u32>,
    skip_special_tokens: bool,
    tokenizer: &mut Tokenizer,
) -> Result<String, TextEmbeddingsError> {
    Ok(tokenizer
        .with_truncation(None)?
        .decode(&ids, skip_special_tokens)?)
}

fn prepare_pre_prompt(
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
) -> Result<Option<String>, TextEmbeddingsError> {
    let pre_prompt = if let Some(prompt_name) = prompt_name.as_ref() {
        match prompts {
            None => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but no prompts were found in the Sentence Transformers configuration")));
            }
            Some(prompts) if !prompts.contains_key(prompt_name) => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}", prompts.keys())));
            }
            Some(prompts) => prompts.get(prompt_name).cloned(),
        }
    } else {
        default_prompt
    };
    Ok(pre_prompt)
}

#[allow(clippy::too_many_arguments)]
fn tokenize_input(
    mut inputs: EncodingInput,
    add_special_tokens: bool,
    max_input_length: usize,
    truncate_params: Option<TruncationParams>,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
    fast_tokenizer: Option<&crate::fast_tokenization::FastTokenizer>,
) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
    let pre_prompt = prepare_pre_prompt(default_prompt, prompt_name, prompts)?;

    let input_chars = inputs.count_chars();
    let limit = max_input_length * MAX_CHAR_MULTIPLIER;
    if input_chars > limit {
        if truncate_params.is_none() {
            return Err(TextEmbeddingsError::Validation(format!(
                "`inputs` must have less than {limit} characters. Given: {input_chars}"
            )));
        }
        inputs.apply_limit(limit);
    }

    let encoding = match inputs {
        // encode input
        EncodingInput::Single(s) => {
            let s = if let Some(mut pre_prompt) = pre_prompt {
                pre_prompt.push_str(&s);
                pre_prompt
            } else {
                s
            };

            tokenizer.with_truncation(truncate_params)?;
            let encoding = match fast_tokenizer {
                Some(fast) => {
                    match crate::fast_tokenization::encode(fast, tokenizer, &s, add_special_tokens)
                    {
                        Ok(encoding) => encoding,
                        Err(err) => {
                            tracing::debug!(
                                "Fast BPE encoding failed; using Hugging Face tokenizer: {err}"
                            );
                            tokenizer.encode::<&str>(&s, add_special_tokens)?
                        }
                    }
                }
                None => tokenizer.encode::<&str>(&s, add_special_tokens)?,
            };

            (Some(s), encoding)
        }
        EncodingInput::Dual(s1, s2) => {
            if pre_prompt.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be set with dual inputs".to_string(),
                ));
            }

            (
                None,
                tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<(String, String)>((s1, s2), add_special_tokens)?,
            )
        }
        // input is encoded -> convert to tokenizers Encoding
        EncodingInput::Ids(ids) => {
            if let Some(mut pre_prompt) = pre_prompt {
                let text = tokenizer.decode(&ids, true)?;
                pre_prompt.push_str(&text);

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&pre_prompt, true)?;

                (Some(pre_prompt), encoding)
            } else {
                let text = tokenizer.decode(&ids, false)?;

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&text, false)?;

                (Some(text), encoding)
            }
        }
    };
    Ok(encoding)
}

/// Get input length and optionally truncate it
#[allow(clippy::too_many_arguments)]
fn encode_input(
    inputs: EncodingInput,
    truncate: bool,
    truncation_direction: TruncationDirection,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
    fast_tokenizer: Option<&crate::fast_tokenization::FastTokenizer>,
) -> Result<ValidEncoding, TextEmbeddingsError> {
    // Default truncation params
    let truncate_params = truncate.then_some(TruncationParams {
        direction: truncation_direction,
        max_length: max_input_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });

    let (_, encoding) = tokenize_input(
        inputs,
        true,
        max_input_length,
        truncate_params,
        default_prompt,
        prompt_name,
        prompts,
        tokenizer,
        fast_tokenizer,
    )?;
    let seq_len = encoding.len();

    if seq_len > max_input_length {
        return Err(TextEmbeddingsError::Validation(format!(
            "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
        )));
    }
    let histogram = metrics::histogram!("te_request_input_length");
    histogram.record(seq_len as f64);
    Ok(ValidEncoding {
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
        tokens: if fast_tokenizer.is_some() {
            Vec::new()
        } else {
            encoding.get_tokens().to_vec()
        },
        offsets: if fast_tokenizer.is_some() {
            Vec::new()
        } else {
            encoding.get_offsets().to_vec()
        },
    })
}

#[derive(Debug)]
pub struct ValidEncoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub enum EncodingInput {
    Single(String),
    Dual(String, String),
    Ids(Vec<u32>),
}

impl EncodingInput {
    fn is_empty(&self) -> bool {
        match self {
            EncodingInput::Single(s) => s.is_empty(),
            EncodingInput::Dual(s1, s2) => s1.is_empty() && s2.is_empty(),
            EncodingInput::Ids(v) => v.is_empty(),
        }
    }

    fn count_chars(&self) -> usize {
        match self {
            EncodingInput::Single(s) => s.chars().count(),
            EncodingInput::Dual(s1, s2) => s1.chars().count() + s2.chars().count(),
            EncodingInput::Ids(v) => v.len(),
        }
    }

    fn apply_limit(&mut self, limit: usize) {
        let truncate_string = |s: &mut String, limit: usize| {
            if s.is_char_boundary(limit) {
                s.truncate(limit)
            }
        };

        match self {
            EncodingInput::Single(s) => {
                truncate_string(s, limit);
            }
            EncodingInput::Dual(s1, s2) => {
                truncate_string(s1, limit / 2);
                truncate_string(s2, limit / 2);
            }
            EncodingInput::Ids(_) => {}
        }
    }
}

impl From<String> for EncodingInput {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<(String, String)> for EncodingInput {
    fn from(value: (String, String)) -> Self {
        Self::Dual(value.0, value.1)
    }
}

enum TokenizerRequest {
    Encode(
        EncodingInput,
        bool,
        TruncationDirection,
        Option<String>,
        bool, // Embedding request: token strings and source offsets are not consumed.
        oneshot::Sender<Result<ValidEncoding, TextEmbeddingsError>>,
        Span,
    ),
    Tokenize(
        EncodingInput,
        bool,
        Option<String>,
        oneshot::Sender<Result<(Option<String>, RawEncoding), TextEmbeddingsError>>,
        Span,
    ),
    Decode(
        Vec<u32>,
        bool,
        oneshot::Sender<Result<String, TextEmbeddingsError>>,
        Span,
    ),
}

pub fn into_tokens(encoding: tokenizers::Encoding, input: &str) -> Vec<SimpleToken> {
    encoding
        .get_ids()
        .iter()
        .zip(encoding.get_offsets())
        .zip(encoding.get_special_tokens_mask())
        .zip(encoding.get_tokens())
        .map(|(((&id, &(start, stop)), special), token)| {
            let special = *special == 1;
            match special {
                true => SimpleToken {
                    id,
                    text: token.clone(),
                    special,
                    start: None,
                    stop: None,
                },
                false => {
                    let text: Vec<u8> = input.bytes().skip(start).take(stop - start).collect();
                    let text: String = String::from_utf8_lossy(&text).to_string();
                    SimpleToken {
                        id,
                        text,
                        special,
                        start: Some(start),
                        stop: Some(stop),
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::sync::ApiBuilder;

    #[test]
    fn tokenizer() {
        let api = ApiBuilder::from_env().build().unwrap();
        let filename = api
            .model("BAAI/bge-m3".to_string())
            .get("tokenizer.json")
            .unwrap();
        let string = "这是一个文本向量化的测试句子";
        let tokenizer = Tokenizer::from_file(filename).unwrap();

        let encoded = tokenizer.encode(string, true).unwrap();
        assert_eq!(
            encoded.get_offsets(),
            vec![
                (0, 0),
                (0, 3),
                (0, 12),
                (12, 18),
                (18, 21),
                (21, 24),
                (24, 30),
                (30, 36),
                (36, 39),
                (39, 42),
                (0, 0)
            ]
        );

        let tokens = into_tokens(encoded, string);
        assert_eq!(
            tokens,
            vec![
                SimpleToken {
                    id: 0,
                    text: "<s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                },
                SimpleToken {
                    id: 6,
                    text: "这".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(3)
                },
                SimpleToken {
                    id: 100013,
                    text: "这是一个".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(12)
                },
                SimpleToken {
                    id: 189061,
                    text: "文本".to_string(),
                    special: false,
                    start: Some(12),
                    stop: Some(18)
                },
                SimpleToken {
                    id: 2110,
                    text: "向".to_string(),
                    special: false,
                    start: Some(18),
                    stop: Some(21)
                },
                SimpleToken {
                    id: 3272,
                    text: "量".to_string(),
                    special: false,
                    start: Some(21),
                    stop: Some(24)
                },
                SimpleToken {
                    id: 41904,
                    text: "化的".to_string(),
                    special: false,
                    start: Some(24),
                    stop: Some(30)
                },
                SimpleToken {
                    id: 49125,
                    text: "测试".to_string(),
                    special: false,
                    start: Some(30),
                    stop: Some(36)
                },
                SimpleToken {
                    id: 27683,
                    text: "句".to_string(),
                    special: false,
                    start: Some(36),
                    stop: Some(39)
                },
                SimpleToken {
                    id: 1344,
                    text: "子".to_string(),
                    special: false,
                    start: Some(39),
                    stop: Some(42)
                },
                SimpleToken {
                    id: 2,
                    text: "</s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                }
            ]
        );
    }
}

#[cfg(test)]
mod fast_embedding_tests {
    use super::*;

    #[test]
    fn workers_preserve_prompts_truncation_and_metadata_endpoints() {
        let mut hf = crate::fast_tokenization::test_tokenizer();
        hf.add_special_tokens(&[
            tokenizers::AddedToken::from("<s>", true),
            tokenizers::AddedToken::from("</s>", true),
        ]);
        hf.with_post_processor(Some(
            tokenizers::processors::template::TemplateProcessing::builder()
                .try_single("<s>:0 $A:0 </s>:0")
                .unwrap()
                .try_pair("<s>:0 $A:0 </s>:0 $B:1 </s>:1")
                .unwrap()
                .special_tokens(vec![("<s>", 256), ("</s>", 257)])
                .build()
                .unwrap(),
        ));
        let raw_ids = hf.encode("hello", false).unwrap().get_ids().to_vec();
        let tokenizer = Tokenization::new(
            2,
            hf,
            32,
            2,
            Some("default: ".into()),
            Some(HashMap::from([("query".into(), "query: ".into())])),
        );
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                for prompt in [None, Some("query".to_owned())] {
                    for input in [
                        EncodingInput::Single("hello 東京 👩🏽‍💻 <s> longer text".into()),
                        EncodingInput::Ids(raw_ids.clone()),
                    ] {
                        // Clone via the variants: EncodingInput intentionally has no Clone impl.
                        let duplicate = match &input {
                            EncodingInput::Single(s) => EncodingInput::Single(s.clone()),
                            EncodingInput::Ids(ids) => EncodingInput::Ids(ids.clone()),
                            _ => unreachable!(),
                        };
                        let actual = tokenizer
                            .encode_embedding(input, true, direction, prompt.clone())
                            .await
                            .unwrap();
                        let expected = tokenizer
                            .encode(duplicate, true, direction, prompt.clone())
                            .await
                            .unwrap();
                        assert_eq!(actual.input_ids, expected.input_ids);
                        assert_eq!(actual.token_type_ids, expected.token_type_ids);
                        assert_eq!(actual.position_ids, expected.position_ids);
                        assert!(!expected.offsets.is_empty());
                        let decoded = tokenizer
                            .decode(expected.input_ids.clone(), false)
                            .await
                            .unwrap();
                        assert!(!decoded.is_empty());
                    }
                }
            }
            // The metadata endpoint must keep actual strings and source offsets.
            let (_, tokens) = tokenizer
                .tokenize(EncodingInput::Single("hi".into()), true, None)
                .await
                .unwrap();
            assert!(tokens.get_offsets().iter().any(|&(start, end)| end > start));
            assert!(tokens.get_tokens().iter().all(|token| !token.is_empty()));
            assert!(tokenizer
                .encode_embedding(
                    EncodingInput::Single("hello".into()),
                    true,
                    TruncationDirection::Right,
                    Some("missing".into())
                )
                .await
                .is_err());
            assert!(tokenizer
                .encode_embedding(
                    EncodingInput::Single("x".repeat(100)),
                    false,
                    TruncationDirection::Right,
                    None
                )
                .await
                .is_err());
        });
    }

    #[test]
    fn concurrent_requests_preserve_fast_and_fallback_ids() {
        let tokenizer = Tokenization::new(
            4,
            crate::fast_tokenization::test_tokenizer(),
            256,
            0,
            None,
            None,
        );
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let mut tasks = tokio::task::JoinSet::new();
            for i in 0..32 {
                let tokenizer = tokenizer.clone();
                tasks.spawn(async move {
                    let pair = (format!("query {i}"), "東京 passage".to_string());
                    let expected = tokenizer
                        .encode(pair.clone().into(), true, TruncationDirection::Right, None)
                        .await
                        .unwrap();
                    let actual = tokenizer
                        .encode_embedding(pair.into(), true, TruncationDirection::Right, None)
                        .await
                        .unwrap();
                    assert_eq!(actual.input_ids, expected.input_ids);
                    assert_eq!(actual.token_type_ids, expected.token_type_ids);
                    let text = format!("request {i} 東京 👩🏽‍💻");
                    let expected = tokenizer
                        .encode(text.clone().into(), true, TruncationDirection::Left, None)
                        .await
                        .unwrap();
                    let actual = tokenizer
                        .encode_embedding(text.into(), true, TruncationDirection::Left, None)
                        .await
                        .unwrap();
                    assert_eq!(actual.input_ids, expected.input_ids);
                    assert_eq!(actual.token_type_ids, expected.token_type_ids);
                });
            }
            while let Some(result) = tasks.join_next().await {
                result.unwrap();
            }
        });
    }
}
