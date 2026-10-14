//! Keeps transport input admission alive while background jobs own request data.
use std::{future::Future, sync::Arc};

pub(crate) type InputMemoryOwner = Arc<dyn Send + Sync>;
tokio::task_local! { static INPUT_MEMORY: InputMemoryOwner; }

/// Scope transport admission so queued Core jobs can retain the same owner.
pub async fn with_input_memory<T: Send + Sync + 'static, F: Future>(
    owner: Arc<T>,
    future: F,
) -> F::Output {
    let owner: InputMemoryOwner = owner;
    INPUT_MEMORY.scope(owner, future).await
}

pub(crate) fn current() -> Option<InputMemoryOwner> {
    INPUT_MEMORY.try_with(Arc::clone).ok()
}
