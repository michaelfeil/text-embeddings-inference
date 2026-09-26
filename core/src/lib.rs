pub mod backend_pool;
pub mod download;
mod fast_tokenization;
pub mod infer;
pub mod queue;
pub mod tokenization;

use text_embeddings_backend::BackendError;
use thiserror::Error;
use tokio::sync::TryAcquireError;

#[derive(Error, Debug)]
pub enum TextEmbeddingsError {
    #[error("tokenizer error {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Input validation error: {0}")]
    Validation(String),
    #[error("Input validation error: {0}")]
    Empty(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
}

pub(crate) fn log_response_dropped_after(stage: &'static str) {
    metrics::counter!("te_request_canceled", "stage" => stage).increment(1);
}
