use crate::infer::InferResult;
use crate::log_response_dropped_after;
use crate::tokenization::ValidEncoding;
use std::cmp::max;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use text_embeddings_backend::{BackendError, Batch};
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

fn early_dispatch_tokens(value: Option<&str>) -> Option<usize> {
    match value.map(str::trim) {
        None => Some(5_000),
        Some("") => None,
        Some(value) => {
            let tokens = value
                .parse::<usize>()
                .expect("TEI_EARLY_DISPATCH_TOKENS must be a nonnegative integer or empty");
            (tokens != 0).then_some(tokens)
        }
    }
}

fn should_cut(current: usize, next: usize, target: Option<usize>) -> bool {
    current > 0 && target.is_some_and(|target| current + next > target)
}

fn target_for_backlog(
    queued_tokens: usize,
    target: Option<usize>,
    replicas: usize,
) -> Option<usize> {
    target.filter(|_| queued_tokens < 20_000usize.saturating_mul(replicas.max(1)))
}

/// Queue entry
#[derive(Debug)]
pub struct Entry {
    /// Payload
    pub encoding: ValidEncoding,
    /// Entry metadata
    pub metadata: Metadata,
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    /// Shared HTTP-request identity and count of inputs still tokenizing.
    pub(crate) client_batch: Option<Arc<AtomicUsize>>,
    /// InferResponse sender to communicate between the Infer struct and the batching_task
    pub(crate) response_tx: oneshot::Sender<Result<InferResult, BackendError>>,
    /// Tokenization duration
    pub(crate) tokenization: Duration,
    /// Instant when this entry was queued
    pub(crate) queue_time: Instant,
    /// Number of tokens in the prompt
    pub(crate) prompt_tokens: usize,
    /// Pooled embedding
    pub(crate) pooling: bool,
    pub(crate) token_classification: bool,
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::Sender<QueueCommand>,
    closed: Arc<AtomicBool>,
}

impl Queue {
    pub fn new(
        padded_model: bool,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
    ) -> Self {
        Self::with_replicas(
            padded_model,
            max_batch_tokens,
            max_batch_requests,
            radix_mlp_threshold,
            max_concurrent_requests,
            1,
        )
    }

    pub fn with_replicas(
        padded_model: bool,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
        replicas: usize,
    ) -> Self {
        // Validate on the caller's thread before spawning the detached worker.
        let configured_target = match std::env::var("TEI_EARLY_DISPATCH_TOKENS") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(err) => panic!("Invalid TEI_EARLY_DISPATCH_TOKENS: {err}"),
        };
        let early_dispatch_tokens = if replicas > 1 || configured_target.is_some() {
            early_dispatch_tokens(configured_target.as_deref())
        } else {
            None
        };
        // Create channels
        // https://github.com/huggingface/text-embeddings-inference/pull/726, b10 fix.
        let (queue_sender, queue_receiver) = mpsc::channel(2 * max_concurrent_requests);

        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        // Launch background queue task
        std::thread::spawn(move || {
            queue_blocking_task(
                padded_model,
                max_batch_tokens,
                max_batch_requests,
                radix_mlp_threshold,
                max_concurrent_requests,
                queue_receiver,
                early_dispatch_tokens,
                replicas,
                worker_closed,
            )
        });

        Self {
            queue_sender,
            closed,
        }
    }

    /// Reject future appends and settle queued requests. Even if the channel is
    /// full, the worker observes the closed flag on its next command.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _ = self.queue_sender.try_send(QueueCommand::Close);
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub async fn append(&self, entry: Entry) {
        if self.closed.load(Ordering::SeqCst) {
            let _ = entry
                .metadata
                .response_tx
                .send(Err(BackendError::Unhealthy));
            return;
        }
        if let Err(err) = self
            .queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .await
        {
            if let QueueCommand::Append(entry, _) = err.0 {
                let _ = entry
                    .metadata
                    .response_tx
                    .send(Err(BackendError::Unhealthy));
            }
        }
    }

    /// Get the next batch from the queue
    #[instrument(skip(self))]
    pub async fn next_batch(&self) -> Option<NextBatch> {
        let (response_sender, response_receiver) = oneshot::channel();

        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                response_sender,
                span: Span::current(),
            })
            .await
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect(
            "Queue background task dropped the sender without sending a new batch. This is a bug.",
        )
    }
}

// Background task responsible of the queue state
#[allow(clippy::too_many_arguments)]
fn queue_blocking_task(
    padded_model: bool,
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    radix_mlp_threshold: f32,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
    early_dispatch_tokens: Option<usize>,
    replicas: usize,
    closed: Arc<AtomicBool>,
) {
    let capacity = max_batch_requests.unwrap_or(max_concurrent_requests);
    tracing::info!(
        ?early_dispatch_tokens,
        replicas,
        full_batch_backlog_tokens = 20_000usize.saturating_mul(replicas.max(1)),
        "Early dispatch token target"
    );
    let radix_mlp_pad: Option<usize> = std::env::var("RADIX_MLP_PAD")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut entries: VecDeque<Entry> = VecDeque::with_capacity(max_concurrent_requests);

    tracing::info!("Using {} queue {}", "custom Baseten", "implementation.");

    while let Some(cmd) = queue_receiver.blocking_recv() {
        if closed.load(Ordering::SeqCst) {
            for entry in entries.drain(..) {
                let _ = entry
                    .metadata
                    .response_tx
                    .send(Err(BackendError::Unhealthy));
            }
            metrics::gauge!("te_queue_size").set(0.0);
            match cmd {
                QueueCommand::Append(entry, _) => {
                    let _ = entry
                        .metadata
                        .response_tx
                        .send(Err(BackendError::Unhealthy));
                }
                QueueCommand::NextBatch {
                    response_sender, ..
                } => {
                    let _ = response_sender.send(None);
                }
                QueueCommand::Close => {}
            }
            continue;
        }
        match cmd {
            QueueCommand::Close => {}
            QueueCommand::Append(entry, span) => {
                let _span = span.entered();
                entries.push_back(*entry);
                let gauge = metrics::gauge!("te_queue_size");
                gauge.increment(1.0);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();
                // Snapshot live queued work once, including the batch being selected.
                // Draining a busy queue must not enable the cut midway through this batch.
                let queued_tokens = entries
                    .iter()
                    .filter(|entry| !entry.metadata.response_tx.is_closed())
                    .map(|entry| entry.encoding.input_ids.len())
                    .sum();
                let target = target_for_backlog(queued_tokens, early_dispatch_tokens, replicas);

                let mut input_ids = Vec::with_capacity(max_batch_tokens);
                let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
                let mut position_ids = Vec::with_capacity(max_batch_tokens);
                let mut tokens = Vec::with_capacity(max_batch_tokens);
                let mut offsets = Vec::with_capacity(max_batch_tokens);

                let mut pooled_indices = Vec::with_capacity(capacity);
                let mut raw_indices = Vec::with_capacity(capacity);
                let mut metadata: Vec<Metadata> = Vec::with_capacity(capacity);
                let mut cu_seq_lengths = Vec::with_capacity(capacity);
                cu_seq_lengths.push(0);

                let mut current_tokens = 0;
                let mut max_length = 0;

                let mut entry_index = 0;

                while let Some(entry) = entries.pop_front() {
                    // Filter entries where the response receiver was dropped (== entries where the request
                    // was dropped by the client)
                    if entry.metadata.response_tx.is_closed() {
                        log_response_dropped_after("after tokenization / before batching");
                        continue;
                    }

                    let entry_tokens = entry.encoding.input_ids.len();

                    // Soft limit: always allow one input, even if it exceeds the target.
                    let unfinished_client = || {
                        metadata.iter().any(|selected| {
                            selected.client_batch.as_ref().is_some_and(|client| {
                                client.load(Ordering::SeqCst) != 0
                                    || entry
                                        .metadata
                                        .client_batch
                                        .as_ref()
                                        .is_some_and(|next| Arc::ptr_eq(client, next))
                                    || entries.iter().any(|pending| {
                                        !pending.metadata.response_tx.is_closed()
                                            && pending
                                                .metadata
                                                .client_batch
                                                .as_ref()
                                                .is_some_and(|other| Arc::ptr_eq(client, other))
                                    })
                            })
                        })
                    };
                    if should_cut(current_tokens, entry_tokens, target) && !unfinished_client() {
                        entries.push_front(entry);
                        break;
                    }

                    let total_tokens = if padded_model {
                        (max(max_length, entry_tokens as u32) * (metadata.len() + 1) as u32)
                            as usize
                    } else {
                        current_tokens + entry_tokens
                    };

                    if total_tokens > max_batch_tokens {
                        debug_assert!(
                            entry_tokens <= max_batch_tokens,
                            "Single entry exceeds max batch tokens, leading to stall."
                        );
                        entries.push_front(entry);
                        break;
                    }

                    match entry.metadata.pooling {
                        true => pooled_indices.push(entry_index),
                        false => raw_indices.push(entry_index),
                    }

                    max_length = max(max_length, entry_tokens as u32);

                    input_ids.extend(entry.encoding.input_ids);
                    token_type_ids.extend(entry.encoding.token_type_ids);
                    position_ids.extend(entry.encoding.position_ids);
                    tokens.extend(entry.encoding.tokens);
                    offsets.extend(entry.encoding.offsets);

                    current_tokens += entry_tokens;
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(current_tokens as u32);

                    entry_index += 1;

                    if Some(metadata.len()) == max_batch_requests {
                        break;
                    }
                }

                // Compute RadixMLP compact representation with BOTH mappings
                let (compact_input_ids, compact_position_ids, scatter_unfold, fold_gather) =
                    if radix_mlp_threshold > 1e-6 && !input_ids.is_empty() {
                        let (compact_ids, compact_pos, scatter, fold) =
                            radix_mlp::compute_fold_and_scatter(
                                &input_ids,
                                &position_ids,
                                &cu_seq_lengths,
                                radix_mlp_pad,
                            );

                        // Only use if we achieved meaningful compression
                        let compression_ratio = compact_ids.len() as f32 / input_ids.len() as f32;
                        tracing::info!(
                            "RadixMLP compression ratio: {:.2} ({} -> {})",
                            compression_ratio,
                            input_ids.len(),
                            compact_ids.len()
                        );
                        metrics::histogram!("te_radix_mlp_compression_ratio")
                            .record(compression_ratio as f64);
                        if radix_mlp_threshold >= 1.0 || compression_ratio < radix_mlp_threshold {
                            (
                                Some(compact_ids),
                                Some(compact_pos),
                                Some(scatter),
                                Some(fold),
                            )
                        } else {
                            (None, None, None, None)
                        }
                    } else {
                        (None, None, None, None)
                    };

                let batch_size = metadata.len();
                let next_batch = if metadata.is_empty() {
                    None
                } else {
                    Some((
                        metadata,
                        Batch {
                            input_ids,
                            token_type_ids,
                            position_ids,
                            cumulative_seq_lengths: cu_seq_lengths,
                            max_length,
                            pooled_indices,
                            raw_indices,
                            compact_input_ids,
                            compact_position_ids,
                            scatter_unfold,
                            fold_gather,
                            tokens,
                            offsets,
                        },
                    ))
                };

                if let Err(Some((metadata, _))) = response_sender.send(next_batch) {
                    for entry in metadata {
                        let _ = entry.response_tx.send(Err(BackendError::Inference(
                            "Batch pipeline exited before dispatch".into(),
                        )));
                    }
                }

                let histogram = metrics::histogram!("te_batch_next_size");
                histogram.record(batch_size as f64);
                let histogram = metrics::histogram!("te_batch_next_tokens");
                histogram.record(current_tokens as f64);
                let gauge = metrics::gauge!("te_queue_size");
                gauge.set(entries.len() as f64)
            }
        }
    }
}

pub type NextBatch = (Vec<Metadata>, Batch);

#[derive(Debug)]
enum QueueCommand {
    Close,
    Append(Box<Entry>, Span),
    NextBatch {
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

pub(crate) fn prune_canceled_batch((metadata, batch): NextBatch) -> Option<NextBatch> {
    if metadata.iter().all(|m| !m.response_tx.is_closed()) {
        return Some((metadata, batch));
    }
    let mut kept = Vec::with_capacity(metadata.len());
    let mut packed = Batch {
        input_ids: Vec::with_capacity(batch.input_ids.len()),
        token_type_ids: Vec::with_capacity(batch.token_type_ids.len()),
        position_ids: Vec::with_capacity(batch.position_ids.len()),
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: Vec::new(),
        raw_indices: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    };
    for (index, entry) in metadata.into_iter().enumerate() {
        if entry.response_tx.is_closed() {
            log_response_dropped_after("after batching / before inference");
            continue;
        }
        let start = batch.cumulative_seq_lengths[index] as usize;
        let end = batch.cumulative_seq_lengths[index + 1] as usize;
        packed
            .input_ids
            .extend_from_slice(&batch.input_ids[start..end]);
        packed
            .token_type_ids
            .extend_from_slice(&batch.token_type_ids[start..end]);
        packed
            .position_ids
            .extend_from_slice(&batch.position_ids[start..end]);
        if !batch.tokens.is_empty() {
            packed.tokens.extend_from_slice(&batch.tokens[start..end]);
        }
        if !batch.offsets.is_empty() {
            packed.offsets.extend_from_slice(&batch.offsets[start..end]);
        }
        packed
            .cumulative_seq_lengths
            .push(packed.input_ids.len() as u32);
        packed.max_length = packed.max_length.max((end - start) as u32);
        if entry.pooling {
            packed.pooled_indices.push(kept.len() as u32);
        } else {
            packed.raw_indices.push(kept.len() as u32);
        }
        kept.push(entry);
    }
    if kept.is_empty() {
        None
    } else {
        Some((kept, packed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_preserves_token_classification_offsets() {
        let (canceled, canceled_rx) = oneshot::channel();
        let (live, _live_rx) = oneshot::channel();
        drop(canceled_rx);
        let metadata = [canceled, live]
            .into_iter()
            .map(|response_tx| Metadata {
                client_batch: None,
                response_tx,
                tokenization: Duration::ZERO,
                queue_time: Instant::now(),
                prompt_tokens: 1,
                pooling: false,
                token_classification: true,
            })
            .collect();
        let batch = Batch {
            input_ids: vec![10, 20],
            token_type_ids: vec![0, 0],
            position_ids: vec![0, 0],
            cumulative_seq_lengths: vec![0, 1, 2],
            max_length: 1,
            pooled_indices: vec![],
            raw_indices: vec![0, 1],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec!["canceled".into(), "live".into()],
            offsets: vec![(0, 8), (2, 6)],
        };
        let (metadata, batch) = prune_canceled_batch((metadata, batch)).unwrap();
        assert_eq!(metadata.len(), 1);
        assert!(metadata[0].token_classification);
        assert_eq!(batch.input_ids, vec![20]);
        assert_eq!(batch.raw_indices, vec![0]);
        assert_eq!(batch.tokens, vec!["live"]);
        assert_eq!(batch.offsets, vec![(2, 6)]);
        assert_eq!(target_for_backlog(100_000, Some(5_000), 8), Some(5_000));
        assert_eq!(target_for_backlog(160_000, Some(5_000), 8), None);
    }
}
