//! Exclusive replica leases shared by inference and health calls.
//! The pool contains no prepared-batch queue.
//! All transitions use one short lock; model calls and queue operations run outside it.
use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use text_embeddings_backend::{Backend, BackendError, Batch, Embeddings, ModelType};
use tokio::sync::watch;

#[derive(Debug)]
struct Replica {
    busy: bool,
    healthy: bool,
    batches: u64,
    tokens: usize,
    sequences: usize,
    inference_time: Duration,
}

#[derive(Debug)]
struct State {
    replicas: Vec<Replica>,
    idle: VecDeque<usize>,
    closed: bool,
}

impl State {
    fn healthy(&self) -> usize {
        self.replicas.iter().filter(|r| r.healthy).count()
    }
}

#[derive(Debug)]
struct Inner {
    backends: Vec<Backend>,
    state: Mutex<State>,
    changed: watch::Sender<()>,
    health: watch::Sender<bool>,
}

impl Inner {
    fn signal(&self, state: &State) {
        self.health.send_if_modified(|health| {
            let next = !state.closed && state.healthy() == state.replicas.len();
            if *health == next {
                false
            } else {
                *health = next;
                true
            }
        });
        metrics::gauge!("te_backend_replicas_healthy").set(state.healthy() as f64);
        metrics::gauge!("te_backend_replicas_total").set(state.replicas.len() as f64);
        metrics::gauge!("te_backend_replicas_running")
            .set(state.replicas.iter().filter(|r| r.busy).count() as f64);
        self.changed.send_replace(());
    }
}

#[derive(Debug, Clone)]
pub struct BackendPool {
    inner: Arc<Inner>,
    pub padded_model: bool,
    pub radix_mlp_supported: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
}

impl BackendPool {
    pub fn new(backends: Vec<Backend>) -> Result<Self, BackendError> {
        let first = backends.first().ok_or_else(|| {
            BackendError::Start("At least one backend replica is required".into())
        })?;
        if backends.iter().any(|b| {
            b.model_type != first.model_type
                || b.padded_model != first.padded_model
                || b.radix_mlp_supported != first.radix_mlp_supported
        }) {
            return Err(BackendError::Start(
                "Backend replicas have incompatible capabilities".into(),
            ));
        }
        let result = Self {
            padded_model: first.padded_model,
            radix_mlp_supported: first.radix_mlp_supported,
            max_batch_size: backends.iter().filter_map(|b| b.max_batch_size).min(),
            model_type: first.model_type.clone(),
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    replicas: backends
                        .iter()
                        .map(|_| Replica {
                            busy: false,
                            healthy: true,
                            batches: 0,
                            tokens: 0,
                            sequences: 0,
                            inference_time: Duration::ZERO,
                        })
                        .collect(),
                    idle: (0..backends.len()).collect(),
                    closed: false,
                }),
                backends,
                changed: watch::channel(()).0,
                health: watch::channel(true).0,
            }),
        };
        result.inner.signal(&result.inner.state.lock().unwrap());
        Ok(result)
    }

    pub fn len(&self) -> usize {
        self.inner.backends.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.backends.is_empty()
    }
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        let mut receiver = self.inner.health.subscribe();
        // gRPC consumes health through changed(); publish the already-warmed
        // initial state even when no later transition has occurred.
        receiver.mark_changed();
        receiver
    }
    pub fn is_available(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        !state.closed && state.healthy() > 0
    }
    pub fn is_closed(&self) -> bool {
        self.inner.state.lock().unwrap().closed
    }

    pub fn close(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.closed = true;
        self.inner.signal(&state);
    }

    pub(crate) fn changes(&self) -> watch::Receiver<()> {
        self.inner.changed.subscribe()
    }
    pub(crate) async fn acquire_replica(&self, id: usize) -> Result<Execution, BackendError> {
        self.acquire_matching(Some(id)).await
    }
    async fn acquire_matching(&self, target: Option<usize>) -> Result<Execution, BackendError> {
        let mut changed = self.inner.changed.subscribe();
        let start = Instant::now();
        loop {
            changed.borrow_and_update();
            {
                let mut state = self.inner.state.lock().unwrap();
                if state.closed
                    || state.healthy() == 0
                    || target.is_some_and(|id| !state.replicas[id].healthy)
                {
                    return Err(BackendError::Unhealthy);
                }
                let position = state
                    .idle
                    .iter()
                    .position(|&id| target.is_none_or(|target| target == id));
                if let Some(position) = position {
                    let id = state.idle.remove(position).unwrap();
                    state.replicas[id].busy = true;
                    self.inner.signal(&state);
                    metrics::histogram!("te_pipeline_pool_wait_duration")
                        .record(start.elapsed().as_secs_f64());
                    return Ok(Execution {
                        pool: self.clone(),
                        id,
                        finished: false,
                        submitted: false,
                        reserved_at: Instant::now(),
                    });
                }
            }
            let _ = changed.changed().await;
        }
    }

    pub fn observe_preparation(&self, duration: Duration) {
        metrics::histogram!("te_pipeline_preparation_duration").record(duration.as_secs_f64());
    }

    pub async fn health(&self) -> Result<(), BackendError> {
        // Check idle replicas under exclusive leases. Busy replicas retain their
        // most recent status until their current operation completes.
        let executions = {
            let mut state = self.inner.state.lock().unwrap();
            if state.closed || state.healthy() == 0 {
                return Err(BackendError::Unhealthy);
            }
            let ids: Vec<_> = state.idle.drain(..).collect();
            let executions: Vec<_> = ids
                .into_iter()
                .map(|id| {
                    let r = &mut state.replicas[id];
                    r.busy = true;
                    Execution {
                        pool: self.clone(),
                        id,
                        finished: false,
                        submitted: false,
                        reserved_at: Instant::now(),
                    }
                })
                .collect();
            self.inner.signal(&state);
            executions
        };
        // Like inference, keep leases alive if the HTTP health caller cancels.
        let pool = self.clone();
        tokio::spawn(async move {
            for mut execution in executions {
                execution.submitted = true;
                let result = execution.pool.inner.backends[execution.id]
                    .health()
                    .await
                    .map(|_| ((), Duration::ZERO));
                execution.finish(0, 0, &result);
            }
            let state = pool.inner.state.lock().unwrap();
            if state.closed || state.healthy() != state.replicas.len() {
                Err(BackendError::Unhealthy)
            } else {
                Ok(())
            }
        })
        .await
        .map_err(|_| BackendError::Unhealthy)?
    }
}

impl Execution {
    async fn run<T, F, Fut>(
        self,
        tokens: usize,
        sequences: usize,
        call: F,
    ) -> Result<(T, Duration), BackendError>
    where
        T: Send + 'static,
        F: FnOnce(Backend) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(T, Duration), BackendError>> + Send + 'static,
    {
        let mut execution = self;
        execution.submitted = true;
        // The spawned task owns the execution lease until the actual native call
        // returns, even if its client disconnects or the pipeline is canceled.
        tokio::spawn(async move {
            let backend = execution.pool.inner.backends[execution.id].clone();
            execution.started();
            let result = call(backend).await;
            execution.finish(tokens, sequences, &result);
            result
        })
        .await
        .map_err(|_| BackendError::Inference("Replica execution task exited".into()))?
    }

    pub async fn embed(self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let tokens = compute_tokens(&batch, self.pool.padded_model);
        self.run(tokens, batch.len(), move |backend| async move {
            backend.embed(batch).await
        })
        .await
    }

    pub async fn predict(
        self,
        batch: Batch,
    ) -> Result<(text_embeddings_backend::Predictions, Duration), BackendError> {
        let tokens = compute_tokens(&batch, self.pool.padded_model);
        self.run(tokens, batch.len(), move |backend| async move {
            backend.predict(batch).await
        })
        .await
    }
    pub async fn predict_tokens(
        self,
        batch: Batch,
    ) -> Result<(text_embeddings_backend::TokenPredictions, Duration), BackendError> {
        let tokens = compute_tokens(&batch, self.pool.padded_model);
        self.run(tokens, batch.len(), move |backend| async move {
            backend.predict_tokens(batch).await
        })
        .await
    }
}

fn compute_tokens(batch: &Batch, padded: bool) -> usize {
    if padded {
        batch.max_length as usize * batch.len()
    } else {
        batch.input_ids.len()
    }
}
pub(crate) struct Execution {
    pool: BackendPool,
    id: usize,
    finished: bool,
    submitted: bool,
    reserved_at: Instant,
}

impl Execution {
    fn started(&self) {
        metrics::histogram!("te_replica_dispatch_delay_duration", "replica" => self.id.to_string())
            .record(self.reserved_at.elapsed().as_secs_f64());
    }

    fn finish<T>(
        mut self,
        tokens: usize,
        sequences: usize,
        result: &Result<(T, Duration), BackendError>,
    ) {
        let mut state = self.pool.inner.state.lock().unwrap();
        let r = &mut state.replicas[self.id];
        r.busy = false;
        r.healthy = result.is_ok();
        if let Ok((_, duration)) = result {
            metrics::counter!("te_replica_tokens", "replica" => self.id.to_string())
                .increment(tokens as u64);
            if !duration.is_zero() {
                r.batches += 1;
                r.tokens += tokens;
                r.sequences += sequences;
                r.inference_time += *duration;
                if r.batches.is_multiple_of(100) {
                    // Rates use native inference time, excluding queueing and idle time.
                    let inference_seconds = r.inference_time.as_secs_f64();
                    tracing::info!(
                        replica = self.id,
                        batches = r.batches,
                        window_batches = 100,
                        sequences = r.sequences,
                        tokens = r.tokens,
                        inference_seconds,
                        batches_per_second = 100.0 / inference_seconds,
                        sequences_per_second = r.sequences as f64 / inference_seconds,
                        tokens_per_second = r.tokens as f64 / inference_seconds,
                        "Replica inference throughput (last 100 batches)"
                    );
                    r.tokens = 0;
                    r.sequences = 0;
                    r.inference_time = Duration::ZERO;
                }
                metrics::histogram!("te_replica_inference_duration", "replica" => self.id.to_string())
                    .record(duration.as_secs_f64());
                metrics::counter!("te_replica_batches", "replica" => self.id.to_string())
                    .increment(1);
            }
            state.idle.push_back(self.id);
        } else {
            state.closed = true;
            tracing::error!(
                replica = self.id,
                "Backend failure: closing the replica pool"
            );
        }
        self.finished = true;
        self.pool.inner.signal(&state);
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        if !self.finished {
            let mut state = self.pool.inner.state.lock().unwrap();
            let r = &mut state.replicas[self.id];
            r.busy = false;
            if self.submitted {
                r.healthy = false;
                state.closed = true;
            } else if r.healthy {
                state.idle.push_back(self.id);
            }
            self.pool.inner.signal(&state);
        }
    }
}
