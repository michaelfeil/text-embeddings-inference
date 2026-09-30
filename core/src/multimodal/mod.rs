//! Model-owned multimodal preparation before token budgeting and inference batching.
//!
//! The processor is a library entry point; HTTP/queue/backend wiring is a separate step.
mod image;
mod media;
mod positions;
mod qwen3_vl;

pub use qwen3_vl::{MultimodalConfig, PreparedMultimodal, Qwen3VlProcessor};
