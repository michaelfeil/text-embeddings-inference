use crate::ErrorType;
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;
use std::fmt::Formatter;
use text_embeddings_core::tokenization::EncodingInput;
use utoipa::openapi::{RefOr, Schema};
use utoipa::ToSchema;

fn default_ignore_labels() -> Vec<String> {
    vec!["O".to_string()]
}

use crate::http::ner::AggregationStrategy;

#[derive(Debug)]
pub(crate) enum Sequence {
    Single(String),
    Pair(String, String),
    Ids(Vec<u32>),
}

impl Sequence {
    pub(crate) fn count_chars(&self) -> usize {
        match self {
            Sequence::Single(s) => s.chars().count(),
            Sequence::Pair(s1, s2) => s1.chars().count() + s2.chars().count(),
            Sequence::Ids(_) => 0,
        }
    }
}

impl From<Sequence> for EncodingInput {
    fn from(value: Sequence) -> Self {
        match value {
            Sequence::Single(s) => Self::Single(s),
            Sequence::Pair(s1, s2) => Self::Dual(s1, s2),
            Sequence::Ids(ids) => Self::Ids(ids),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PredictInput {
    Single(Sequence),
    Batch(Vec<Sequence>),
}

impl<'de> Deserialize<'de> for PredictInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Internal {
            Single(String),
            Multiple(Vec<String>),
            Id(u32),
            Ids(Vec<u32>),
        }

        struct PredictInputVisitor;

        impl<'de> Visitor<'de> for PredictInputVisitor {
            type Value = PredictInput;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "a string, \
                    a pair of strings [string, string] \
                    a batch of mixed strings and pairs [[string], [string, string], ...], \
                    a final token-ID sequence [integer, ...] or a batch [[integer, ...], ...]",
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(PredictInput::Single(Sequence::Single(v.to_string())))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let sequence_from_vec = |mut value: Vec<String>| {
                    // Validate that value is correct
                    match value.len() {
                        1 => Ok(Sequence::Single(value.pop().unwrap())),
                        2 => {
                            // Second element is last
                            let second = value.pop().unwrap();
                            let first = value.pop().unwrap();
                            Ok(Sequence::Pair(first, second))
                        }
                        // Sequence can only be a single string or a pair of strings
                        _ => Err(de::Error::invalid_length(value.len(), &self)),
                    }
                };

                // Get first element
                // This will determine if input is a batch or not
                let s = match seq
                    .next_element::<Internal>()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?
                {
                    // Input is not a batch
                    // Return early
                    Internal::Single(value) => {
                        // Option get second element
                        let second = seq.next_element()?;

                        if seq.next_element::<String>()?.is_some() {
                            // Error as we do not accept > 2 elements
                            return Err(de::Error::invalid_length(3, &self));
                        }

                        if let Some(second) = second {
                            // Second element exists
                            // This is a pair
                            return Ok(PredictInput::Single(Sequence::Pair(value, second)));
                        } else {
                            // Second element does not exist
                            return Ok(PredictInput::Single(Sequence::Single(value)));
                        }
                    }
                    // Input is a batch
                    Internal::Multiple(value) => sequence_from_vec(value),
                    Internal::Id(id) => {
                        let mut ids = vec![id];
                        while let Some(id) = seq.next_element::<u32>()? {
                            ids.push(id);
                        }
                        return Ok(PredictInput::Single(Sequence::Ids(ids)));
                    }
                    Internal::Ids(ids) => {
                        let mut batch = vec![Sequence::Ids(ids)];
                        while let Some(ids) = seq.next_element::<Vec<u32>>()? {
                            batch.push(Sequence::Ids(ids));
                        }
                        return Ok(PredictInput::Batch(batch));
                    }
                }?;

                let mut batch = Vec::with_capacity(32);
                // Push first sequence
                batch.push(s);

                // Iterate on all sequences
                while let Some(value) = seq.next_element::<Vec<String>>()? {
                    // Validate sequence
                    let s = sequence_from_vec(value)?;
                    // Push to batch
                    batch.push(s);
                }
                Ok(PredictInput::Batch(batch))
            }
        }

        deserializer.deserialize_any(PredictInputVisitor)
    }
}

impl<'__s> ToSchema<'__s> for PredictInput {
    fn schema() -> (&'__s str, RefOr<Schema>) {
        let token_ids = utoipa::openapi::ArrayBuilder::new()
            .items(
                utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::SchemaType::Integer)
                    .minimum(Some(0.0))
                    .maximum(Some(u32::MAX as f64)),
            )
            .min_items(Some(1))
            .description(Some(
                "A final, unpadded token-ID sequence; segment IDs are zero",
            ))
            .build();
        (
            "PredictInput",
            utoipa::openapi::OneOfBuilder::new()
                .item(
                    utoipa::openapi::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::SchemaType::String)
                        .description(Some("A single string")),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::ObjectBuilder::new()
                                .schema_type(utoipa::openapi::SchemaType::String),
                        )
                        .description(Some("A pair of strings"))
                        .min_items(Some(2))
                        .max_items(Some(2)),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::OneOfBuilder::new()
                                .item(
                                    utoipa::openapi::ArrayBuilder::new()
                                        .items(
                                            utoipa::openapi::ObjectBuilder::new()
                                                .schema_type(utoipa::openapi::SchemaType::String),
                                        )
                                        .description(Some("A single string"))
                                        .min_items(Some(1))
                                        .max_items(Some(1)),
                                )
                                .item(
                                    utoipa::openapi::ArrayBuilder::new()
                                        .items(
                                            utoipa::openapi::ObjectBuilder::new()
                                                .schema_type(utoipa::openapi::SchemaType::String),
                                        )
                                        .description(Some("A pair of strings"))
                                        .min_items(Some(2))
                                        .max_items(Some(2)),
                                ),
                        )
                        .description(Some("A batch")),
                )
                .item(token_ids.clone())
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(token_ids)
                        .min_items(Some(1))
                        .description(Some("A batch of independent final token-ID sequences")),
                )
                .description(Some(
                    "Model input: a string, a string pair, a batch of single strings and pairs, \
                    or final token IDs and batches of final token IDs.",
                ))
                .example(Some(json!("What is Deep Learning?")))
                .into(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, ToSchema, Eq, Default)]
pub(crate) enum TruncationDirection {
    #[serde(alias = "left", alias = "Left")]
    Left,
    #[serde(alias = "right", alias = "Right")]
    #[default]
    Right,
}

impl From<TruncationDirection> for tokenizers::TruncationDirection {
    fn from(value: TruncationDirection) -> Self {
        match value {
            TruncationDirection::Left => Self::Left,
            TruncationDirection::Right => Self::Right,
        }
    }
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictRequest {
    pub inputs: PredictInput,
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictTokensRequest {
    pub inputs: Vec<String>,
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
    #[serde(default)]
    #[schema(default = "none", example = "none")]
    pub aggregation_strategy: AggregationStrategy,
    #[serde(default = "default_ignore_labels")]
    pub ignore_labels: Vec<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Prediction {
    #[schema(example = "0.5")]
    pub score: f32,
    #[schema(example = "admiration")]
    pub label: String,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum PredictResponse {
    Single(Vec<Prediction>),
    Batch(Vec<Vec<Prediction>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct RerankRequest {
    #[schema(example = "What is Deep Learning?")]
    pub query: String,
    #[schema(example = json!(["Deep Learning is ..."]))]
    pub texts: Vec<String>,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub return_text: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Rank {
    #[schema(example = "0")]
    pub index: usize,
    #[schema(nullable = true, example = "Deep Learning is ...", default = "null")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[schema(example = "1.0")]
    pub score: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct RerankResponse(pub Vec<Rank>);

pub(crate) use text_embeddings_core::input::EmbeddingInput;

/// Internal batching shape. A conversation remains a single model input.
pub(crate) enum InputBatch {
    Single(EncodingInput),
    Batch(Vec<EncodingInput>),
}

impl From<EmbeddingInput> for InputBatch {
    fn from(value: EmbeddingInput) -> Self {
        match value {
            EmbeddingInput::Text(text) => Self::Single(EncodingInput::Single(text)),
            EmbeddingInput::TextBatch(texts) => {
                Self::Batch(texts.into_iter().map(EncodingInput::Single).collect())
            }
            EmbeddingInput::Messages(messages) => Self::Single(EncodingInput::Messages(messages)),
            EmbeddingInput::TokenIds(ids) => Self::Single(EncodingInput::Ids(ids)),
            EmbeddingInput::TokenIdsBatch(batch) => {
                Self::Batch(batch.into_iter().map(EncodingInput::Ids).collect())
            }
        }
    }
}

#[derive(Deserialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct OpenAICompatRequest {
    pub input: EmbeddingInput,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub model: Option<String>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub user: Option<String>,
    #[schema(default = "float", example = "float")]
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatEmbedding {
    #[schema(example = "embedding")]
    pub object: &'static str,
    #[schema(example = json!([0.0, 1.0, 2.0]))]
    pub embedding: Embedding,
    #[schema(example = "0")]
    pub index: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatUsage {
    #[schema(example = "512")]
    pub prompt_tokens: usize,
    #[schema(example = "512")]
    pub total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatResponse {
    #[schema(example = "list")]
    pub object: &'static str,
    pub data: Vec<OpenAICompatEmbedding>,
    #[schema(example = "thenlper/gte-base")]
    pub model: String,
    pub usage: OpenAICompatUsage,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityInput {
    /// The string that you wish to compare the other strings with. This can be a phrase, sentence,
    /// or longer passage, depending on the model being used.
    #[schema(example = "What is Deep Learning?")]
    pub source_sentence: String,
    /// A list of strings which will be compared against the source_sentence.
    #[schema(example = json!(["What is Machine Learning?"]))]
    pub sentences: Vec<String>,
}

#[derive(Deserialize, ToSchema, Default)]
pub(crate) struct SimilarityParameters {
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityRequest {
    pub inputs: SimilarityInput,
    /// Additional inference parameters for Sentence Similarity
    #[schema(default = "null", example = "null", nullable = true)]
    pub parameters: Option<SimilarityParameters>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([0.0, 1.0, 0.5]))]
pub(crate) struct SimilarityResponse(pub Vec<f32>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    pub inputs: EmbeddingInput,

    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,

    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,

    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,

    #[serde(default = "default_normalize")]
    #[schema(default = "true", example = "true")]
    pub normalize: bool,

    /// The number of dimensions that the output embeddings should have. If not set, the original
    /// shape of the representation will be returned instead.
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
}

fn default_normalize() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[0.0, 1.0, 2.0]]))]
pub(crate) struct EmbedResponse(pub Vec<Vec<f32>>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedSparseRequest {
    pub inputs: EmbeddingInput,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct SparseValue {
    pub index: usize,
    pub value: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct EmbedSparseResponse(pub Vec<Vec<SparseValue>>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedAllRequest {
    pub inputs: EmbeddingInput,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[[0.0, 1.0, 2.0]]]))]
pub(crate) struct EmbedAllResponse(pub Vec<Vec<Vec<f32>>>);

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatErrorResponse {
    pub message: String,
    pub code: u16,
    #[serde(rename(serialize = "type"))]
    pub error_type: ErrorType,
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum TokenizeInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct TokenizeRequest {
    pub inputs: TokenizeInput,
    #[serde(default = "default_add_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub add_special_tokens: bool,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

fn default_add_special_tokens() -> bool {
    true
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SimpleToken {
    #[schema(example = 0)]
    pub id: u32,
    #[schema(example = "test")]
    pub text: String,
    #[schema(example = "false")]
    pub special: bool,
    #[schema(example = 0)]
    pub start: Option<usize>,
    #[schema(example = 2)]
    pub stop: Option<usize>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[{"id": 0, "text": "test", "special": false, "start": 0, "stop": 2}]]))]
pub(crate) struct TokenizeResponse(pub Vec<Vec<SimpleToken>>);

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum InputIds {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct DecodeRequest {
    pub ids: InputIds,
    #[serde(default = "default_skip_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub skip_special_tokens: bool,
}

fn default_skip_special_tokens() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!(["test"]))]
pub(crate) struct DecodeResponse(pub Vec<String>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct VertexRequest {
    pub instances: Vec<serde_json::Value>,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum VertexPrediction {
    Embed(EmbedResponse),
    EmbedSparse(EmbedSparseResponse),
    Predict(PredictResponse),
    Rerank(RerankResponse),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct VertexResponse {
    pub predictions: Vec<VertexPrediction>,
}

#[derive(Serialize, ToSchema, Clone)]
pub(crate) struct TokenPrediction {
    #[schema(example = "Hello")]
    pub token: String,
    #[schema(example = 0)]
    pub token_id: u32,
    #[schema(example = 0)]
    pub start: Option<usize>,
    #[schema(example = 5)]
    pub end: Option<usize>,
    #[schema(example = json!({"O": 9.41, "B-MISC": -1.15, "I-MISC": -0.85}))]
    pub results: std::collections::HashMap<String, f32>,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum TokenPredictResponse {
    Batch(Vec<Vec<TokenPrediction>>),
}

#[cfg(test)]
mod embedding_input_tests {
    use super::*;

    #[test]
    fn conversation_is_one_input_and_texts_remain_a_batch() {
        let value = json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"}
        ]);
        let req: EmbedRequest = serde_json::from_value(json!({"inputs": value})).unwrap();
        let InputBatch::Single(EncodingInput::Messages(messages)) = InputBatch::from(req.inputs)
        else {
            panic!("a conversation must be one input");
        };
        assert_eq!(serde_json::to_value(messages).unwrap(), value);
        let req: OpenAICompatRequest =
            serde_json::from_value(json!({"input": ["hello", "world"]})).unwrap();
        let InputBatch::Batch(inputs) = InputBatch::from(req.input) else {
            panic!("independent texts must stay a batch");
        };
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], EncodingInput::Single(text) if text == "hello"));
        assert!(matches!(&inputs[1], EncodingInput::Single(text) if text == "world"));
    }

    #[test]
    fn token_ids_are_one_input_or_an_ordered_batch_on_both_endpoints() {
        let req: EmbedRequest = serde_json::from_value(json!({"inputs": [101, 42, 102]})).unwrap();
        assert!(matches!(InputBatch::from(req.inputs),
            InputBatch::Single(EncodingInput::Ids(ids)) if ids == vec![101, 42, 102]));
        let req: OpenAICompatRequest =
            serde_json::from_value(json!({"input": [[101, 42, 102], [101, 43, 102]]})).unwrap();
        let InputBatch::Batch(inputs) = InputBatch::from(req.input) else {
            panic!("token sequences must remain independent inputs");
        };
        assert_eq!(inputs.len(), 2);
        assert!(matches!(&inputs[0], EncodingInput::Ids(ids) if ids == &vec![101, 42, 102]));
        assert!(matches!(&inputs[1], EncodingInput::Ids(ids) if ids == &vec![101, 43, 102]));
    }

    #[test]
    fn prediction_inputs_preserve_text_pairs_and_accept_final_ids() {
        let parse = |value| serde_json::from_value::<PredictInput>(value).unwrap();
        assert!(matches!(
            parse(json!("hello")),
            PredictInput::Single(Sequence::Single(_))
        ));
        assert!(matches!(parse(json!(["query", "document"])),
            PredictInput::Single(Sequence::Pair(query, document))
                if query == "query" && document == "document"));
        assert!(matches!(parse(json!([["hello"], ["query", "document"]])),
            PredictInput::Batch(batch) if batch.len() == 2));
        assert!(matches!(parse(json!([101, 42, 102])),
            PredictInput::Single(Sequence::Ids(ids)) if ids == vec![101, 42, 102]));
        let PredictInput::Batch(batch) = parse(json!([[101, 42], [102]])) else {
            panic!("token sequences must remain independent inputs");
        };
        assert!(matches!(&batch[0], Sequence::Ids(ids) if ids == &vec![101, 42]));
        assert!(matches!(&batch[1], Sequence::Ids(ids) if ids == &vec![102]));
        for invalid in [
            json!([]),
            json!([[]]),
            json!([-1]),
            json!([1.5]),
            json!([4294967296_u64]),
            json!([1, "text"]),
            json!([[1], ["text"]]),
            json!(["query", "document", "extra"]),
        ] {
            assert!(serde_json::from_value::<PredictInput>(invalid).is_err());
        }
        let schema = serde_json::to_value(PredictInput::schema().1).unwrap();
        assert_eq!(schema["oneOf"][3]["items"]["type"], "integer");
        assert_eq!(schema["oneOf"][4]["items"]["items"]["type"], "integer");
    }

    #[test]
    fn embedding_schemas_reference_the_shared_contract() {
        for (schema, field) in [
            (EmbedRequest::schema().1, "inputs"),
            (EmbedAllRequest::schema().1, "inputs"),
            (EmbedSparseRequest::schema().1, "inputs"),
            (OpenAICompatRequest::schema().1, "input"),
        ] {
            let value = serde_json::to_value(schema).unwrap();
            assert_eq!(
                value["properties"][field]["$ref"],
                "#/components/schemas/EmbeddingInput"
            );
        }
    }
}
