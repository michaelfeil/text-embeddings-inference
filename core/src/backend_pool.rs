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
}

impl Inner {
    fn signal(&self, state: &State) {
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
            b.model_type != first.model_type || b.radix_mlp_supported != first.radix_mlp_supported
        }) {
            return Err(BackendError::Start(
                "Backend replicas have incompatible capabilities".into(),
            ));
        }
        let result = Self {
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
                        })
                        .collect(),
                    idle: (0..backends.len()).collect(),
                    closed: false,
                }),
                backends,
                changed: watch::channel(()).0,
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
                        released: false,
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
                        released: false,
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
        F: FnOnce(Backend, tokio::sync::oneshot::Sender<()>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(T, Duration), BackendError>> + Send + 'static,
    {
        let mut execution = self;
        execution.submitted = true;
        // Retain execution ownership even if its caller cancels. A deferred
        // submission may release the model lease, but completion still reports
        // failures and accounts for the full inference duration.
        tokio::spawn(async move {
            let backend = execution.pool.inner.backends[execution.id].clone();
            execution.started();
            let (submitted, submission) = tokio::sync::oneshot::channel();
            let result = call(backend, submitted);
            tokio::pin!(result);
            let result = tokio::select! {
                result = &mut result => result,
                submission = submission => {
                    if submission.is_ok() { execution.release(); }
                    result.await
                }
            };
            execution.finish(tokens, sequences, &result);
            result
        })
        .await
        .map_err(|_| BackendError::Inference("Replica execution task exited".into()))?
    }

    pub async fn embed(self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let tokens = batch.input_ids.len();
        self.run(tokens, batch.len(), move |backend, submitted| async move {
            backend.embed_with_submission(batch, Some(submitted)).await
        })
        .await
    }

    pub async fn decide(
        self,
        batch: Batch,
        inputs: Vec<text_embeddings_backend::DecisionInput>,
    ) -> Result<(Vec<text_embeddings_backend::DecisionOutput>, Duration), BackendError> {
        let tokens = batch.input_ids.len();
        self.run(tokens, batch.len(), move |backend, _submitted| async move {
            backend.decide(batch, inputs).await
        })
        .await
    }

    pub async fn predict(
        self,
        batch: Batch,
    ) -> Result<(text_embeddings_backend::Predictions, Duration), BackendError> {
        let tokens = batch.input_ids.len();
        self.run(tokens, batch.len(), move |backend, submitted| async move {
            backend
                .predict_with_submission(batch, Some(submitted))
                .await
        })
        .await
    }
    pub async fn predict_tokens(
        self,
        batch: Batch,
    ) -> Result<(text_embeddings_backend::TokenPredictions, Duration), BackendError> {
        let tokens = batch.input_ids.len();
        self.run(tokens, batch.len(), move |backend, submitted| async move {
            backend
                .predict_tokens_with_submission(batch, Some(submitted))
                .await
        })
        .await
    }
}

pub(crate) struct Execution {
    pool: BackendPool,
    id: usize,
    finished: bool,
    submitted: bool,
    released: bool,
    reserved_at: Instant,
}

impl Execution {
    fn release(&mut self) {
        let mut state = self.pool.inner.state.lock().unwrap();
        state.replicas[self.id].busy = false;
        if !state.closed && state.replicas[self.id].healthy {
            state.idle.push_back(self.id);
        }
        self.released = true;
        self.pool.inner.signal(&state);
    }

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
        if !self.released {
            r.busy = false;
        }
        if result.is_err() {
            r.healthy = false;
        }
        if let Ok((_, duration)) = result {
            metrics::counter!("te_replica_tokens", "replica" => self.id.to_string())
                .increment(tokens as u64);
            if !duration.is_zero() {
                r.batches += 1;
                if r.batches.is_multiple_of(100) {
                    // Sample the latest batch, using native inference time only.
                    let inference_seconds = duration.as_secs_f64();
                    tracing::info!(
                        replica = self.id,
                        batches = r.batches,
                        sequences,
                        tokens,
                        inference_seconds,
                        tokens_per_second = tokens as f64 / inference_seconds,
                        "Replica inference throughput (latest batch)"
                    );
                }
                metrics::histogram!("te_replica_inference_duration", "replica" => self.id.to_string())
                    .record(duration.as_secs_f64());
                metrics::counter!("te_replica_batches", "replica" => self.id.to_string())
                    .increment(1);
            }
            if !self.released && !state.closed {
                state.idle.push_back(self.id);
            }
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
            if !self.released {
                r.busy = false;
            }
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

#[cfg(test)]
mod completion_tests {
    use super::*;
    fn execution() -> Execution {
        let pool = BackendPool {
            inner: Arc::new(Inner {
                backends: vec![],
                state: Mutex::new(State {
                    replicas: vec![Replica {
                        busy: true,
                        healthy: true,
                        batches: 0,
                    }],
                    idle: VecDeque::new(),
                    closed: false,
                }),
                changed: watch::channel(()).0,
            }),
            model_type: ModelType::Classifier,
            radix_mlp_supported: false,
            max_batch_size: None,
        };
        Execution {
            pool,
            id: 0,
            finished: false,
            submitted: true,
            released: false,
            reserved_at: Instant::now(),
        }
    }
    #[test]
    fn old_completion_does_not_release_the_next_model_lease() {
        let mut previous = execution();
        let pool = previous.pool.clone();
        previous.release();
        {
            let mut state = pool.inner.state.lock().unwrap();
            assert_eq!(state.idle.pop_front(), Some(0));
            state.replicas[0].busy = true;
        }
        previous.finish(1, 1, &Ok(((), Duration::from_millis(1))));
        let state = pool.inner.state.lock().unwrap();
        assert!(state.replicas[0].busy);
        assert!(state.idle.is_empty());
    }
    #[test]
    fn completion_error_closes_pool_without_releasing_new_lease() {
        let mut previous = execution();
        let pool = previous.pool.clone();
        previous.release();
        {
            let mut state = pool.inner.state.lock().unwrap();
            state.idle.clear();
            state.replicas[0].busy = true;
        }
        previous.finish::<()>(1, 1, &Err(BackendError::Inference("copy failed".into())));
        let state = pool.inner.state.lock().unwrap();
        assert!(state.closed);
        assert!(!state.replicas[0].healthy);
        assert!(state.replicas[0].busy);
        assert!(state.idle.is_empty());
    }
}
