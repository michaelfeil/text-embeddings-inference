/// Payload tokenization logic
use crate::TextEmbeddingsError;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer;
pub use tokenizers::Encoding as RawEncoding;
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::oneshot;
use tracing::{instrument, Span};

#[cfg(feature = "multimodal")]
use base64::engine::general_purpose;

static MAX_CHAR_MULTIPLIER: usize = 250;

// Image preprocessing function
#[cfg(feature = "multimodal")]
fn preprocess_image(image_source: &str) -> Result<(Vec<u8>, Vec<Vec<f32>>), String> {
    // This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Parse the image source (base64 or URL)
    // 2. Download/decode the image
    // 3. Resize to standard dimensions (e.g., 224x224)
    // 4. Normalize pixel values
    // 5. Convert to tensor format
    
    if image_source.starts_with("base64:") {
        // Decode base64 image
        let base64_data = &image_source[7..];
        let decoded_data = general_purpose::STANDARD
            .decode(base64_data)
            .map_err(|e| format!("Base64 decoding error: {}", e))?;
        
        // Create dummy tensor data (3 channels, 224x224)
        let tensor_data = vec![vec![0.0; 224 * 224]; 3]; // RGB channels
        
        Ok((decoded_data, tensor_data))
    } else if image_source.starts_with("url:") {
        // For URL images, we'd download and process them
        // For now, return dummy data
        let dummy_bytes = vec![0u8; 1024];
        let tensor_data = vec![vec![0.0; 224 * 224]; 3]; // RGB channels
        
        Ok((dummy_bytes, tensor_data))
    } else {
        Err("Invalid image source format".to_string())
    }
}

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
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let receiver_clone = receiver.clone();
                let default_prompt_clone = default_prompt.clone();
                let prompts_clone = prompts.clone();
                std::thread::spawn(move || {
                    tokenizer_worker(
                        tokenizer_clone,
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
fn tokenizer_worker(
    mut tokenizer: Tokenizer,
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

            let encoding = tokenizer
                .with_truncation(truncate_params)?
                .encode::<&str>(&s, add_special_tokens)?;

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
        // Multi-modal inputs
        EncodingInput::MultiModal { text, image: _ } => {
            if pre_prompt.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be set with multi-modal inputs".to_string(),
                ));
            }

            // For multi-modal, we only tokenize the text part
            let text_to_encode = text.unwrap_or_else(|| "".to_string());
            
            let encoding = if text_to_encode.is_empty() {
                // If no text, create a minimal encoding for image-only input
                tokenizer.encode("", add_special_tokens)?
            } else {
                tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&text_to_encode, add_special_tokens)?
            };

            (Some(text_to_encode), encoding)
        }
        EncodingInput::ImageOnly(_) => {
            if pre_prompt.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be set with image-only inputs".to_string(),
                ));
            }

            // For image-only, create a minimal encoding
            let encoding = tokenizer.encode("", add_special_tokens)?;
            (None, encoding)
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
) -> Result<ValidEncoding, TextEmbeddingsError> {
    // Default truncation params
    let truncate_params = truncate.then_some(TruncationParams {
        direction: truncation_direction,
        max_length: max_input_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });

    let (_text, encoding) = tokenize_input(
        inputs.clone(),
        true,
        max_input_length,
        truncate_params,
        default_prompt,
        prompt_name,
        prompts,
        tokenizer,
    )?;
    let seq_len = encoding.len();

    if seq_len > max_input_length {
        return Err(TextEmbeddingsError::Validation(format!(
            "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
        )));
    }
    let histogram = metrics::histogram!("te_request_input_length");
    histogram.record(seq_len as f64);

    // Create multi-modal option and preprocess images if needed
    #[cfg(feature = "multimodal")]
    let (multimodal, pixel_values, image_tensors) = match inputs {
        EncodingInput::MultiModal { text: input_text, image } => {
            // For now, we'll prioritize text over image
            // In a real implementation, you might want to handle this differently
            if let Some(text) = input_text {
                (Some(text_embeddings_backend::MultiModalOption::Text(text)), None, None)
            } else {
                // Preprocess image
                let (preprocessed_bytes, tensor_data) = preprocess_image(&image)
                    .map_err(|e| TextEmbeddingsError::Validation(e))?;
                (Some(text_embeddings_backend::MultiModalOption::ImageUrl(image)), Some(preprocessed_bytes), Some(tensor_data))
            }
        }
        EncodingInput::ImageOnly(image) => {
            // Preprocess image
            let (preprocessed_bytes, tensor_data) = preprocess_image(&image)
                .map_err(|e| TextEmbeddingsError::Validation(e))?;
            (Some(text_embeddings_backend::MultiModalOption::ImageUrl(image)), Some(preprocessed_bytes), Some(tensor_data))
        }
        _ => (None, None, None),
    };
    
    #[cfg(not(feature = "multimodal"))]
    let (multimodal, pixel_values, image_tensors) = (None, None, None);

    Ok(ValidEncoding {
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
        tokens: encoding.get_tokens().to_vec(),
        offsets: encoding.get_offsets().to_vec(),
        multimodal,
        pixel_values,
        image_tensors,
    })
}

#[derive(Debug)]
pub struct ValidEncoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
    pub multimodal: Option<text_embeddings_backend::MultiModalOption>,
    pub pixel_values: Option<Vec<u8>>,
    pub image_tensors: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub enum EncodingInput {
    Single(String),
    Dual(String, String),
    Ids(Vec<u32>),
    // New multi-modal variants
    MultiModal {
        text: Option<String>,
        image: String, // base64: or url: prefixed
    },
    ImageOnly(String), // base64: or url: prefixed
}

impl EncodingInput {
    fn is_empty(&self) -> bool {
        match self {
            EncodingInput::Single(s) => s.is_empty(),
            EncodingInput::Dual(s1, s2) => s1.is_empty() && s2.is_empty(),
            EncodingInput::Ids(v) => v.is_empty(),
            EncodingInput::MultiModal { text, image } => {
                text.as_ref().map_or(true, |t| t.is_empty()) && image.is_empty()
            }
            EncodingInput::ImageOnly(image) => image.is_empty(),
        }
    }

    fn count_chars(&self) -> usize {
        match self {
            EncodingInput::Single(s) => s.chars().count(),
            EncodingInput::Dual(s1, s2) => s1.chars().count() + s2.chars().count(),
            EncodingInput::Ids(v) => v.len(),
            EncodingInput::MultiModal { text, image } => {
                text.as_ref().map_or(0, |t| t.chars().count()) + image.chars().count()
            }
            EncodingInput::ImageOnly(image) => image.chars().count(),
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
            EncodingInput::MultiModal { text, image: _ } => {
                if let Some(text) = text {
                    truncate_string(text, limit / 2);
                }
                // Image URLs/base64 typically don't get truncated
            }
            EncodingInput::ImageOnly(_) => {
                // Image URLs/base64 typically don't get truncated
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
