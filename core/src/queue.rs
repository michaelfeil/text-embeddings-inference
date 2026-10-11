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
    pub decision: Option<text_embeddings_backend::DecisionInput>,
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
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        radix_mlp_threshold: f32,
        max_concurrent_requests: usize,
    ) -> Self {
        Self::with_replicas(
            max_batch_tokens,
            max_batch_requests,
            radix_mlp_threshold,
            max_concurrent_requests,
            1,
        )
    }

    pub fn with_replicas(
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
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    radix_mlp_threshold: f32,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
    early_dispatch_tokens: Option<usize>,
    replicas: usize,
    closed: Arc<AtomicBool>,
) {
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
    // Reuse selection storage; outgoing batch vectors belong to the backend.
    let mut selected: Vec<Entry> = Vec::new();

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

                let mut current_tokens = 0;
                let mut pooled_count = 0;
                let mut token_count = 0;
                let mut offset_count = 0;

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
                        selected.iter().any(|selected| {
                            selected
                                .metadata
                                .client_batch
                                .as_ref()
                                .is_some_and(|client| {
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

                    let total_tokens = current_tokens + entry_tokens;

                    if total_tokens > max_batch_tokens {
                        debug_assert!(
                            entry_tokens <= max_batch_tokens,
                            "Single entry exceeds max batch tokens, leading to stall."
                        );
                        entries.push_front(entry);
                        break;
                    }

                    current_tokens = total_tokens;
                    pooled_count += usize::from(entry.metadata.pooling);
                    token_count += entry.encoding.tokens.len();
                    offset_count += entry.encoding.offsets.len();
                    selected.push(entry);
                    if Some(selected.len()) == max_batch_requests {
                        break;
                    }
                }

                // Size buffers from the selected batch, not the configured maximum.
                // Most requests have no token strings or offsets at all.
                let capacity = selected.len();
                let mut input_ids = Vec::new();
                let mut token_type_ids = Vec::new();
                let mut position_ids = Vec::new();
                let mut tokens = Vec::new();
                let mut offsets = Vec::new();
                let mut multimodal = Vec::with_capacity(capacity);
                let mut pooled_indices = Vec::with_capacity(pooled_count);
                let mut raw_indices = Vec::with_capacity(capacity - pooled_count);
                let mut metadata = Vec::with_capacity(capacity);
                let mut cu_seq_lengths =
                    Vec::with_capacity(if capacity == 0 { 0 } else { capacity + 1 });
                if capacity != 0 {
                    cu_seq_lengths.push(0);
                }
                let mut max_length = 0;

                for (entry_index, entry) in selected.drain(..).enumerate() {
                    let entry_tokens = entry.encoding.input_ids.len();
                    match entry.metadata.pooling {
                        true => pooled_indices.push(entry_index as u32),
                        false => raw_indices.push(entry_index as u32),
                    }
                    max_length = max(max_length, entry_tokens as u32);
                    multimodal.push(entry.encoding.multimodal);
                    if entry_index == 0 {
                        // Move the first encoding's allocations into the batch. A
                        // single request needs neither new token buffers nor copies.
                        input_ids = entry.encoding.input_ids;
                        token_type_ids = entry.encoding.token_type_ids;
                        position_ids = entry.encoding.position_ids;
                        tokens = entry.encoding.tokens;
                        offsets = entry.encoding.offsets;
                        input_ids.reserve_exact(current_tokens - input_ids.len());
                        token_type_ids.reserve_exact(current_tokens - token_type_ids.len());
                        position_ids.reserve_exact(current_tokens - position_ids.len());
                        tokens.reserve_exact(token_count - tokens.len());
                        offsets.reserve_exact(offset_count - offsets.len());
                    } else {
                        input_ids.extend(entry.encoding.input_ids);
                        token_type_ids.extend(entry.encoding.token_type_ids);
                        position_ids.extend(entry.encoding.position_ids);
                        tokens.extend(entry.encoding.tokens);
                        offsets.extend(entry.encoding.offsets);
                    }
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(input_ids.len() as u32);
                }

                // Compute RadixMLP compact representation with BOTH mappings
                let (compact_input_ids, compact_position_ids, scatter_unfold, fold_gather) =
                    if radix_mlp_threshold > 1e-6 && !input_ids.is_empty() {
                        if let Some((compact_ids, compact_pos, scatter, fold)) =
                            crate::radix::compute(
                                &input_ids,
                                &position_ids,
                                &cu_seq_lengths,
                                &multimodal,
                                radix_mlp_pad,
                            )
                        {
                            // Only use if we achieved meaningful compression
                            let compression_ratio =
                                compact_ids.len() as f32 / input_ids.len() as f32;
                            tracing::info!(
                                "RadixMLP compression ratio: {:.2} ({} -> {})",
                                compression_ratio,
                                input_ids.len(),
                                compact_ids.len()
                            );
                            metrics::histogram!("te_radix_mlp_compression_ratio")
                                .record(compression_ratio as f64);
                            if radix_mlp_threshold >= 1.0 || compression_ratio < radix_mlp_threshold
                            {
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
                            multimodal,
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
        multimodal: vec![],
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
        if !batch.multimodal.is_empty() {
            packed.multimodal.push(batch.multimodal[index].clone());
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
    fn test_queue(max_tokens: usize, max_requests: Option<usize>, target: Option<usize>) -> Queue {
        let (queue_sender, queue_receiver) = mpsc::channel(32);
        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        std::thread::spawn(move || {
            queue_blocking_task(
                max_tokens,
                max_requests,
                0.0,
                16,
                queue_receiver,
                target,
                1,
                worker_closed,
            );
        });
        Queue {
            queue_sender,
            closed,
        }
    }

    fn test_entry(
        ids: Vec<u32>,
        pooling: bool,
    ) -> (Entry, oneshot::Receiver<Result<InferResult, BackendError>>) {
        let (response_tx, response_rx) = oneshot::channel();
        let len = ids.len();
        (
            Entry {
                encoding: ValidEncoding {
                    multimodal: None,
                    input_ids: ids,
                    token_type_ids: vec![0; len],
                    position_ids: (0..len as u32).collect(),
                    tokens: vec![],
                    offsets: vec![],
                },
                metadata: Metadata {
                    decision: None,
                    client_batch: None,
                    response_tx,
                    tokenization: Duration::ZERO,
                    queue_time: Instant::now(),
                    prompt_tokens: len,
                    pooling,
                    token_classification: false,
                },
            },
            response_rx,
        )
    }

    #[tokio::test]
    async fn single_request_reuses_encoding_buffers() {
        let queue = test_queue(65_536, Some(1_024), None);
        assert!(queue.next_batch().await.is_none());
        for token_classification in [false, true] {
            let (mut entry, _response) = test_entry(vec![10, 11, 12], !token_classification);
            if token_classification {
                entry.metadata.token_classification = true;
                entry.encoding.tokens = vec!["a".into(), "b".into(), "c".into()];
                entry.encoding.offsets = vec![(0, 1), (2, 3), (4, 5)];
            }
            let pointers = (
                entry.encoding.input_ids.as_ptr(),
                entry.encoding.token_type_ids.as_ptr(),
                entry.encoding.position_ids.as_ptr(),
                entry.encoding.tokens.as_ptr(),
                entry.encoding.offsets.as_ptr(),
            );
            queue.append(entry).await;
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(
                pointers,
                (
                    batch.input_ids.as_ptr(),
                    batch.token_type_ids.as_ptr(),
                    batch.position_ids.as_ptr(),
                    batch.tokens.as_ptr(),
                    batch.offsets.as_ptr(),
                )
            );
            assert_eq!(batch.input_ids, [10, 11, 12]);
            assert_eq!(batch.cumulative_seq_lengths, [0, 3]);
            assert_eq!(metadata.capacity(), 1);
            assert_eq!(batch.cumulative_seq_lengths.capacity(), 2);
            if !token_classification {
                assert_eq!(batch.tokens.capacity(), 0);
                assert_eq!(batch.offsets.capacity(), 0);
                assert_eq!(batch.raw_indices.capacity(), 0);
            }
            assert!(queue.next_batch().await.is_none());
        }
    }

    #[tokio::test]
    async fn packed_batch_preserves_order_and_optional_tokens() {
        let queue = test_queue(65_536, None, None);
        let (first, _first_response) = test_entry(vec![10, 11], true);
        let (mut second, _second_response) = test_entry(vec![20, 21, 22], false);
        second.metadata.token_classification = true;
        second.encoding.tokens = vec!["x".into(), "y".into(), "z".into()];
        second.encoding.offsets = vec![(0, 1), (2, 3), (4, 5)];
        let media = Arc::new(text_embeddings_backend::MultimodalEncoding {
            audios: vec![],
            images: vec![],
            position_ids: [vec![], vec![], vec![]],
            memory: None,
        });
        second.encoding.multimodal = Some(media.clone());
        queue.append(first).await;
        queue.append(second).await;
        let (metadata, batch) = queue.next_batch().await.unwrap();
        assert_eq!(metadata.len(), 2);
        assert!(metadata[1].token_classification);
        assert_eq!(batch.input_ids, [10, 11, 20, 21, 22]);
        assert_eq!(batch.token_type_ids, [0; 5]);
        assert_eq!(batch.position_ids, [0, 1, 0, 1, 2]);
        assert_eq!(batch.cumulative_seq_lengths, [0, 2, 5]);
        assert_eq!(batch.max_length, 3);
        assert_eq!(batch.pooled_indices, [0]);
        assert_eq!(batch.raw_indices, [1]);
        assert_eq!(batch.tokens, ["x", "y", "z"]);
        assert_eq!(batch.offsets, [(0, 1), (2, 3), (4, 5)]);
        assert!(batch.multimodal[0].is_none());
        assert!(Arc::ptr_eq(batch.multimodal[1].as_ref().unwrap(), &media));
        assert_eq!(batch.input_ids.capacity(), 5);
        assert_eq!(batch.tokens.capacity(), 3);
        assert_eq!(batch.offsets.capacity(), 3);
        assert_eq!(batch.pooled_indices.capacity(), 1);
        assert_eq!(batch.raw_indices.capacity(), 1);
    }

    #[tokio::test]
    async fn selection_preserves_limits_and_skips_canceled_entries() {
        for (max_tokens, max_requests, target) in
            [(5, None, None), (100, Some(1), None), (100, None, Some(5))]
        {
            let queue = test_queue(max_tokens, max_requests, target);
            let (canceled, canceled_response) = test_entry(vec![0; 4], true);
            drop(canceled_response);
            queue.append(canceled).await;
            assert!(queue.next_batch().await.is_none());
            let (first, _first_response) = test_entry(vec![1; 3], true);
            let (second, _second_response) = test_entry(vec![2; 3], true);
            queue.append(first).await;
            queue.append(second).await;
            for id in [1, 2] {
                let (metadata, batch) = queue.next_batch().await.unwrap();
                assert_eq!(metadata.len(), 1);
                assert_eq!(batch.input_ids, vec![id; 3]);
                assert_eq!(batch.input_ids.capacity(), 3);
            }
            assert!(queue.next_batch().await.is_none());
        }
    }

    #[tokio::test]
    async fn early_dispatch_keeps_client_batch_together() {
        let queue = test_queue(100, None, Some(3));
        let client = Arc::new(AtomicUsize::new(0));
        let (mut first, _first_response) = test_entry(vec![1; 3], true);
        let (mut second, _second_response) = test_entry(vec![2; 3], true);
        first.metadata.client_batch = Some(client.clone());
        second.metadata.client_batch = Some(client);
        queue.append(first).await;
        queue.append(second).await;
        let (metadata, batch) = queue.next_batch().await.unwrap();
        assert_eq!(metadata.len(), 2);
        assert_eq!(batch.input_ids, [1, 1, 1, 2, 2, 2]);
        assert_eq!(batch.cumulative_seq_lengths, [0, 3, 6]);
    }

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
                decision: None,
            })
            .collect();
        let permits = Arc::new(tokio::sync::Semaphore::new(2));
        let make_media = |value| {
            Arc::new(text_embeddings_backend::MultimodalEncoding {
                audios: vec![],
                images: vec![(
                    0,
                    Arc::new(text_embeddings_backend::ImagePatches {
                        pixels: vec![value],
                        grid_thw: [1, 1, 1],
                        patch_dim: 1,
                        merge_size: 1,
                    }),
                )],
                position_ids: [vec![0], vec![0], vec![0]],
                memory: Some(Arc::new(permits.clone().try_acquire_owned().unwrap())),
            })
        };
        let canceled_media = make_media(1.0);
        let live_media = make_media(2.0);
        let identity = Arc::downgrade(&live_media);
        let batch = Batch {
            multimodal: vec![Some(canceled_media), Some(live_media)],
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
        assert_eq!(batch.multimodal.len(), 1);
        assert!(Arc::ptr_eq(
            batch.multimodal[0].as_ref().unwrap(),
            &identity.upgrade().unwrap()
        ));
        assert_eq!(
            batch.multimodal[0].as_ref().unwrap().images[0].1.pixels,
            vec![2.0]
        );
        assert_eq!(permits.available_permits(), 1);
        drop(batch);
        assert_eq!(permits.available_permits(), 2);
        assert_eq!(target_for_backlog(100_000, Some(5_000), 8), Some(5_000));
        assert_eq!(target_for_backlog(160_000, Some(5_000), 8), None);
    }
    #[test]
    fn cancellation_preserves_decision_metadata() {
        let (canceled, canceled_rx) = oneshot::channel();
        let (live, _live_rx) = oneshot::channel();
        drop(canceled_rx);
        let metadata = [canceled, live]
            .into_iter()
            .enumerate()
            .map(|(i, response_tx)| Metadata {
                client_batch: None,
                response_tx,
                tokenization: Duration::ZERO,
                queue_time: Instant::now(),
                prompt_tokens: i + 2,
                pooling: false,
                token_classification: false,
                decision: Some(text_embeddings_backend::DecisionInput::Laya {
                    question_type: i * 2,
                    markers: if i == 0 { vec![0] } else { vec![1, 2] },
                }),
            })
            .collect();
        let batch = Batch {
            multimodal: vec![],
            input_ids: vec![10, 11, 20, 21, 22],
            token_type_ids: vec![0; 5],
            position_ids: vec![0, 1, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 2, 5],
            max_length: 3,
            pooled_indices: vec![],
            raw_indices: vec![0, 1],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let (metadata, batch) = prune_canceled_batch((metadata, batch)).unwrap();
        assert_eq!(metadata.len(), 1);
        let text_embeddings_backend::DecisionInput::Laya {
            question_type,
            markers,
        } = metadata[0].decision.as_ref().unwrap()
        else {
            panic!("expected Laya metadata")
        };
        assert_eq!(*question_type, 2);
        assert_eq!(*markers, vec![1, 2]);
        assert_eq!(batch.input_ids, vec![20, 21, 22]);
        assert_eq!(batch.raw_indices, vec![0]);
        assert_eq!(batch.cumulative_seq_lengths, vec![0, 3]);
        assert_eq!(batch.position_ids, vec![0, 1, 2]);
    }
}
