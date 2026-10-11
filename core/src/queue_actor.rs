use crate::infer::InferResult;
use crate::log_response_dropped_after;
use crate::tokenization::ValidEncoding;
use std::cmp::max;
use std::collections::{HashMap, VecDeque};
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

/// State belongs exclusively to the queue thread. Canceled entries remain in
/// the counts until removal; this can disable a soft cut, never a hard limit.
struct PendingQueue {
    entries: VecDeque<Entry>,
    tokens: usize,
    clients: Option<HashMap<usize, usize>>,
}

fn client_id(client: &Arc<AtomicUsize>) -> usize {
    Arc::as_ptr(client) as usize
}

impl PendingQueue {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            tokens: 0,
            clients: None,
        }
    }

    fn push(&mut self, entry: Entry) {
        self.tokens += entry.encoding.input_ids.len();
        self.entries.push_back(entry);
    }

    fn pop(&mut self) -> Option<Entry> {
        let entry = self.entries.pop_front()?;
        self.tokens -= entry.encoding.input_ids.len();
        if let (Some(clients), Some(client)) = (&mut self.clients, &entry.metadata.client_batch) {
            let id = client_id(client);
            let count = clients.get_mut(&id).expect("queued client counted");
            *count -= 1;
            if *count == 0 {
                clients.remove(&id);
            }
        }
        Some(entry)
    }

    /// Build lookahead once per selection, only when a soft cut needs it.
    /// Consecutive inputs from one HTTP batch never need this map.
    fn client_is_pending(&mut self, id: usize) -> bool {
        let entries = &self.entries;
        self.clients
            .get_or_insert_with(|| {
                let mut clients = HashMap::new();
                for entry in entries {
                    if let Some(client) = &entry.metadata.client_batch {
                        *clients.entry(client_id(client)).or_default() += 1;
                    }
                }
                clients
            })
            .contains_key(&id)
    }

    fn reject_all(&mut self) {
        for entry in self.entries.drain(..) {
            let _ = entry
                .metadata
                .response_tx
                .send(Err(BackendError::Unhealthy));
        }
        self.tokens = 0;
        self.clients = None;
    }
}

#[derive(Default)]
struct Selection {
    entries: Vec<Entry>,
    tokens: usize,
    pooled: usize,
    token_texts: usize,
    offsets: usize,
}

impl Selection {
    fn unfinished_client(&self, pending: &mut PendingQueue, next_client: Option<usize>) -> bool {
        self.entries
            .iter()
            .filter_map(|entry| entry.metadata.client_batch.as_ref())
            .any(|client| {
                client.load(Ordering::SeqCst) != 0
                    || Some(client_id(client)) == next_client
                    || pending.client_is_pending(client_id(client))
            })
    }

    fn select(
        &mut self,
        pending: &mut PendingQueue,
        max_tokens: usize,
        max_requests: Option<usize>,
        target: Option<usize>,
    ) {
        self.tokens = 0;
        self.pooled = 0;
        self.token_texts = 0;
        self.offsets = 0;
        while let Some(entry) = pending.entries.front() {
            if entry.metadata.response_tx.is_closed() {
                pending.pop();
                log_response_dropped_after("after tokenization / before batching");
                continue;
            }
            let entry_tokens = entry.encoding.input_ids.len();
            if self.tokens + entry_tokens > max_tokens {
                debug_assert!(
                    entry_tokens <= max_tokens,
                    "Single entry exceeds max batch tokens, leading to stall."
                );
                break;
            }
            let next_client = entry.metadata.client_batch.as_ref().map(client_id);
            if should_cut(self.tokens, entry_tokens, target)
                && !self.unfinished_client(pending, next_client)
            {
                break;
            }
            let entry = pending.pop().unwrap();
            self.tokens += entry_tokens;
            self.pooled += usize::from(entry.metadata.pooling);
            self.token_texts += entry.encoding.tokens.len();
            self.offsets += entry.encoding.offsets.len();
            self.entries.push(entry);
            if Some(self.entries.len()) == max_requests {
                break;
            }
        }
        // Lookahead is only needed during selection; appends require no client bookkeeping.
        pending.clients = None;
    }
}

type TokenBuffers = (Vec<u32>, Vec<u32>, Vec<u32>);

fn allocate_token_buffers(capacity: usize) -> TokenBuffers {
    (
        Vec::with_capacity(capacity),
        Vec::with_capacity(capacity),
        Vec::with_capacity(capacity),
    )
}

fn pack_batch(
    selection: &mut Selection,
    spare_token_buffers: &mut TokenBuffers,
    radix_mlp_threshold: f32,
    radix_mlp_pad: Option<usize>,
) -> Option<NextBatch> {
    if selection.entries.is_empty() {
        return None;
    }
    let selected = &mut selection.entries;
    let pooled_count = selection.pooled;
    let token_count = selection.token_texts;
    let offset_count = selection.offsets;
    // Packed batches use the staged numeric buffers. Other storage
    // follows the selected batch; token strings and offsets stay lazy.
    let capacity = selected.len();
    let (mut input_ids, mut token_type_ids, mut position_ids) = if capacity > 1 {
        std::mem::take(spare_token_buffers)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut multimodal = Vec::with_capacity(capacity);
    let mut pooled_indices = Vec::with_capacity(pooled_count);
    let mut raw_indices = Vec::with_capacity(capacity - pooled_count);
    let mut metadata = Vec::with_capacity(capacity);
    let mut cu_seq_lengths = Vec::with_capacity(capacity + 1);
    cu_seq_lengths.push(0);
    let mut max_length = 0;

    for (entry_index, entry) in selected.drain(..).enumerate() {
        let entry_tokens = entry.encoding.input_ids.len();
        match entry.metadata.pooling {
            true => pooled_indices.push(entry_index as u32),
            false => raw_indices.push(entry_index as u32),
        }
        max_length = max(max_length, entry_tokens as u32);
        multimodal.push(entry.encoding.multimodal);
        if capacity == 1 {
            // A single request can transfer its allocations directly.
            input_ids = entry.encoding.input_ids;
            token_type_ids = entry.encoding.token_type_ids;
            position_ids = entry.encoding.position_ids;
        } else {
            input_ids.extend(entry.encoding.input_ids);
            token_type_ids.extend(entry.encoding.token_type_ids);
            position_ids.extend(entry.encoding.position_ids);
        }
        if entry_index == 0 {
            tokens = entry.encoding.tokens;
            offsets = entry.encoding.offsets;
            tokens.reserve_exact(token_count - tokens.len());
            offsets.reserve_exact(offset_count - offsets.len());
        } else {
            tokens.extend(entry.encoding.tokens);
            offsets.extend(entry.encoding.offsets);
        }
        metadata.push(entry.metadata);
        cu_seq_lengths.push(input_ids.len() as u32);
    }

    let mut batch = Batch {
        multimodal,
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths: cu_seq_lengths,
        max_length,
        pooled_indices,
        raw_indices,
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens,
        offsets,
    };
    apply_radix(&mut batch, radix_mlp_threshold, radix_mlp_pad);
    Some((metadata, batch))
}

fn apply_radix(batch: &mut Batch, threshold: f32, pad: Option<usize>) {
    if threshold.is_nan() || threshold <= 1e-6 || batch.input_ids.is_empty() {
        return;
    }
    let Some((ids, positions, scatter, fold)) = crate::radix::compute(
        &batch.input_ids,
        &batch.position_ids,
        &batch.cumulative_seq_lengths,
        &batch.multimodal,
        pad,
    ) else {
        return;
    };
    let ratio = ids.len() as f32 / batch.input_ids.len() as f32;
    tracing::info!(
        "RadixMLP compression ratio: {:.2} ({} -> {})",
        ratio,
        batch.input_ids.len(),
        ids.len()
    );
    metrics::histogram!("te_radix_mlp_compression_ratio").record(ratio as f64);
    if threshold < 1.0 && ratio >= threshold {
        return;
    }
    batch.compact_input_ids = Some(ids);
    batch.compact_position_ids = Some(positions);
    batch.scatter_unfold = Some(scatter);
    batch.fold_gather = Some(fold);
}

// Background task responsible for queue state, selection, and dispatch.
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
    let radix_mlp_pad = std::env::var("RADIX_MLP_PAD")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut pending = PendingQueue::new(max_concurrent_requests);
    let mut selection = Selection::default();
    let mut spare_token_buffers = allocate_token_buffers(max_batch_tokens);
    // Resolve metric handles at each update: the router installs its recorder
    // after constructing this queue. Handles captured here could stay no-op.

    while let Some(cmd) = queue_receiver.blocking_recv() {
        if closed.load(Ordering::SeqCst) {
            pending.reject_all();
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
                pending.push(*entry);
                metrics::gauge!("te_queue_size").set(pending.entries.len() as f64);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();
                let target = target_for_backlog(pending.tokens, early_dispatch_tokens, replicas);
                selection.select(&mut pending, max_batch_tokens, max_batch_requests, target);
                let requests = selection.entries.len();
                let next_batch = pack_batch(
                    &mut selection,
                    &mut spare_token_buffers,
                    radix_mlp_threshold,
                    radix_mlp_pad,
                );
                if let Err(Some((metadata, _))) = response_sender.send(next_batch) {
                    for entry in metadata {
                        let _ = entry.response_tx.send(Err(BackendError::Inference(
                            "Batch pipeline exited before dispatch".into(),
                        )));
                    }
                }
                // Stage the successor only after waking the backend.
                if requests > 1 {
                    spare_token_buffers = allocate_token_buffers(max_batch_tokens);
                }
                metrics::histogram!("te_batch_next_size").record(requests as f64);
                metrics::histogram!("te_batch_next_tokens").record(selection.tokens as f64);
                metrics::gauge!("te_queue_size").set(pending.entries.len() as f64);
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
    fn test_queue(max_tokens: usize) -> Queue {
        let (queue_sender, queue_receiver) = mpsc::channel(32);
        let closed = Arc::new(AtomicBool::new(false));
        let worker_closed = closed.clone();
        std::thread::spawn(move || {
            queue_blocking_task(
                max_tokens,
                None,
                0.0,
                16,
                queue_receiver,
                None,
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

    #[test]
    fn backlog_counts_follow_removal_cancellation_and_close() {
        let mut pending = PendingQueue::new(16);
        let client = Arc::new(AtomicUsize::new(0));
        let (mut first, first_rx) = test_entry(vec![1; 3], true);
        first.metadata.client_batch = Some(client.clone());
        let (mut second, second_rx) = test_entry(vec![2; 4], false);
        second.metadata.client_batch = Some(client.clone());
        pending.push(first);
        pending.push(second);
        assert_eq!(pending.tokens, 7);
        assert!(pending.client_is_pending(client_id(&client)));
        assert_eq!(pending.clients.as_ref().unwrap()[&client_id(&client)], 2);
        drop(first_rx);
        // Cancellation is reconciled on removal, without a backlog scan.
        assert_eq!(pending.tokens, 7);
        let mut selection = Selection::default();
        selection.select(&mut pending, 32, None, Some(5));
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(selection.tokens, 4);
        assert_eq!(pending.tokens, 0);
        assert!(pending.clients.is_none());
        drop(second_rx);
        let (mut third, mut third_rx) = test_entry(vec![3; 2], true);
        third.metadata.client_batch = Some(client);
        pending.push(third);
        pending.reject_all();
        assert_eq!(pending.tokens, 0);
        assert!(pending.clients.is_none());
        assert!(matches!(
            third_rx.try_recv(),
            Ok(Err(BackendError::Unhealthy))
        ));
    }

    #[test]
    fn early_cut_preserves_interleaved_clients_and_hard_limits() {
        let mut pending = PendingQueue::new(16);
        let client = Arc::new(AtomicUsize::new(0));
        let mut receivers = Vec::new();
        for (value, grouped) in [(1, true), (2, false), (3, true), (4, false)] {
            let (mut entry, rx) = test_entry(vec![value; 3], true);
            entry.metadata.client_batch = grouped.then(|| client.clone());
            pending.push(entry);
            receivers.push(rx);
        }
        let mut selection = Selection::default();
        selection.select(&mut pending, 32, None, Some(3));
        assert_eq!(selection.entries.len(), 3);
        assert_eq!(pending.tokens, 3);
        assert!(pending.clients.is_none());
        selection.entries.clear();
        selection.select(&mut pending, 32, None, Some(3));
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(pending.tokens, 0);
        selection.entries.clear();
        // Still-tokenizing clients bypass the soft target, but not token/request limits.
        client.store(1, Ordering::SeqCst);
        for _ in 0..3 {
            let (mut entry, rx) = test_entry(vec![1; 3], true);
            entry.metadata.client_batch = Some(client.clone());
            pending.push(entry);
            receivers.push(rx);
        }
        selection.select(&mut pending, 6, None, Some(3));
        assert_eq!(selection.entries.len(), 2);
        assert_eq!(pending.tokens, 3);
        selection.entries.clear();
        selection.select(&mut pending, 32, Some(1), Some(3));
        assert_eq!(selection.entries.len(), 1);
        assert_eq!(pending.tokens, 0);
        // A single input larger than the soft target is accepted.
        selection.entries.clear();
        let (entry, rx) = test_entry(vec![1; 8], true);
        receivers.push(rx);
        pending.push(entry);
        selection.select(&mut pending, 8, None, Some(3));
        assert_eq!(selection.tokens, 8);
    }

    #[test]
    fn soft_target_is_snapshotted_before_draining_backlog() {
        let mut pending = PendingQueue::new(16);
        let mut receivers = Vec::new();
        for _ in 0..4 {
            let (entry, rx) = test_entry(vec![1; 5_000], true);
            pending.push(entry);
            receivers.push(rx);
        }
        let target = target_for_backlog(pending.tokens, Some(5_000), 1);
        assert_eq!(target, None);
        let mut selection = Selection::default();
        selection.select(&mut pending, 15_000, None, target);
        assert_eq!(selection.tokens, 15_000);
        assert_eq!(pending.tokens, 5_000);
        assert_eq!(
            target_for_backlog(pending.tokens, Some(5_000), 1),
            Some(5_000)
        );
    }

    #[test]
    fn selection_matches_fifo_reference_across_client_patterns() {
        // Compare with the original lookahead policy, without cancellations:
        // canceled entries intentionally stay counted until actor removal.
        for scenario in 0..256usize {
            let clients: Vec<_> = (0..4)
                .map(|i| Arc::new(AtomicUsize::new(usize::from((scenario + i) % 7 == 0))))
                .collect();
            let mut pending = PendingQueue::new(32);
            let mut reference = VecDeque::new();
            let mut receivers = Vec::new();
            let mut seed = scenario as u64 + 1;
            for id in 0..32 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let tokens = 1 + ((seed >> 32) % 16) as usize;
                let group = ((seed >> 40) % 5) as usize;
                let (mut entry, rx) = test_entry(vec![id; tokens], true);
                entry.metadata.client_batch = clients.get(group).cloned();
                pending.push(entry);
                receivers.push(rx);
                reference.push_back((id, tokens, group));
            }
            let max_tokens = 32 + scenario % 64;
            let max_requests = (scenario % 3 == 0).then_some(1 + scenario % 7);
            let target = (scenario % 4 != 0).then_some(8 + scenario % 24);
            let mut selection = Selection::default();
            while !reference.is_empty() {
                let mut expected: Vec<(u32, usize, usize)> = Vec::new();
                let mut tokens = 0;
                while let Some(&(id, count, group)) = reference.front() {
                    let unfinished = expected.iter().any(|&(_, _, selected_group)| {
                        clients.get(selected_group).is_some_and(|client| {
                            client.load(Ordering::SeqCst) != 0
                                || selected_group == group
                                || reference
                                    .iter()
                                    .any(|&(_, _, pending_group)| pending_group == selected_group)
                        })
                    });
                    if should_cut(tokens, count, target) && !unfinished {
                        break;
                    }
                    if tokens + count > max_tokens {
                        break;
                    }
                    reference.pop_front();
                    expected.push((id, count, group));
                    tokens += count;
                    if Some(expected.len()) == max_requests {
                        break;
                    }
                }
                selection.select(&mut pending, max_tokens, max_requests, target);
                assert_eq!(selection.tokens, tokens, "scenario {scenario}");
                assert_eq!(
                    selection
                        .entries
                        .iter()
                        .map(|e| e.encoding.input_ids[0])
                        .collect::<Vec<_>>(),
                    expected.iter().map(|&(id, _, _)| id).collect::<Vec<_>>(),
                    "scenario {scenario}"
                );
                assert_eq!(
                    pending.tokens,
                    reference.iter().map(|&(_, n, _)| n).sum::<usize>()
                );
                selection.entries.clear();
            }
        }
    }

    #[tokio::test]
    async fn staged_buffers_preserve_single_inputs_and_pending_batches() {
        let queue = test_queue(32);
        assert!(queue.next_batch().await.is_none());
        let (canceled, response) = test_entry(vec![0; 2], true);
        drop(response);
        queue.append(canceled).await;
        assert!(queue.next_batch().await.is_none());

        let (single, _response) = test_entry(vec![10, 11], true);
        let pointers = (
            single.encoding.input_ids.as_ptr(),
            single.encoding.token_type_ids.as_ptr(),
            single.encoding.position_ids.as_ptr(),
        );
        queue.append(single).await;
        let (_, single_batch) = queue.next_batch().await.unwrap();
        assert_eq!(
            pointers,
            (
                single_batch.input_ids.as_ptr(),
                single_batch.token_type_ids.as_ptr(),
                single_batch.position_ids.as_ptr(),
            )
        );
        assert_eq!(single_batch.tokens.capacity(), 0);
        assert_eq!(single_batch.offsets.capacity(), 0);

        // Keep previous batches alive while consuming and refilling the spare.
        let mut pending = Vec::new();
        for value in [20, 30, 40] {
            let (first, _first_response) = test_entry(vec![value; 2], true);
            let (mut second, _second_response) = test_entry(vec![value + 1; 3], false);
            second.metadata.token_classification = true;
            second.encoding.tokens = vec!["x".into(), "y".into(), "z".into()];
            second.encoding.offsets = vec![(0, 1), (2, 3), (4, 5)];
            queue.append(first).await;
            queue.append(second).await;
            let (metadata, batch) = queue.next_batch().await.unwrap();
            assert_eq!(metadata.len(), 2);
            assert!(metadata[1].token_classification);
            assert_eq!(
                batch.input_ids,
                [value, value, value + 1, value + 1, value + 1]
            );
            assert_eq!(batch.token_type_ids, [0; 5]);
            assert_eq!(batch.position_ids, [0, 1, 0, 1, 2]);
            assert_eq!(batch.cumulative_seq_lengths, [0, 2, 5]);
            assert_eq!(batch.max_length, 3);
            assert_eq!(batch.pooled_indices, [0]);
            assert_eq!(batch.raw_indices, [1]);
            assert_eq!(batch.tokens, ["x", "y", "z"]);
            assert_eq!(batch.offsets, [(0, 1), (2, 3), (4, 5)]);
            assert_eq!(batch.input_ids.capacity(), 32);
            assert_eq!(batch.token_type_ids.capacity(), 32);
            assert_eq!(batch.position_ids.capacity(), 32);
            assert_eq!(batch.tokens.capacity(), 3);
            assert_eq!(batch.offsets.capacity(), 3);
            pending.push(batch);
            assert!(queue.next_batch().await.is_none());
        }
        assert_eq!(single_batch.input_ids, [10, 11]);
        for (batch, value) in pending.iter().zip([20, 30, 40]) {
            assert_eq!(
                batch.input_ids,
                [value, value, value + 1, value + 1, value + 1]
            );
        }
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
