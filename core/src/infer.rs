use crate::backend_pool::BackendPool;
use crate::queue::{prune_canceled_batch, Entry, Metadata, NextBatch, Queue};
use crate::tokenization::{EncodingInput, RawEncoding, Tokenization};
use crate::TextEmbeddingsError;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use text_embeddings_backend::{Backend, BackendError, Embedding, ModelType};
use tokenizers::TruncationDirection;
use tokio::sync::{mpsc, oneshot, watch, Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::instrument;

/// Inference struct
#[derive(Debug, Clone)]
pub struct Infer {
    tokenization: Tokenization,
    queue: Queue,
    /// Shared notify
    notify_batching_task: Arc<Notify>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
    backend: BackendPool,
    _lifetime: Arc<PipelineLifetime>,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenization: Tokenization,
        queue: Queue,
        max_concurrent_requests: usize,
        backend: Backend,
    ) -> Self {
        Self::with_pool(
            tokenization,
            queue,
            max_concurrent_requests,
            BackendPool::new(vec![backend]).expect("single backend is compatible"),
        )
    }

    pub fn with_pool(
        tokenization: Tokenization,
        queue: Queue,
        max_concurrent_requests: usize,
        backend: BackendPool,
    ) -> Self {
        let notify_batching_task = Arc::new(Notify::new());
        let (sender, receiver) = mpsc::channel(1);
        let receiver = Arc::new(Mutex::new(receiver));
        tokio::spawn(batching_task(
            backend.clone(),
            queue.clone(),
            notify_batching_task.clone(),
            sender,
        ));
        for replica in 0..backend.len() {
            tokio::spawn(backend_task(
                backend.clone(),
                queue.clone(),
                receiver.clone(),
                replica,
            ));
        }
        let lifetime = Arc::new(PipelineLifetime {
            pool: backend.clone(),
            queue: queue.clone(),
        });
        Self {
            tokenization,
            queue,
            notify_batching_task,
            limit_concurrent_requests: Arc::new(Semaphore::new(max_concurrent_requests)),
            backend,
            _lifetime: lifetime,
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn tokenize<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
        self.tokenization
            .tokenize(inputs.into(), add_special_tokens, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })
    }

    #[instrument(skip(self, ids))]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        self.tokenization
            .decode(ids, skip_special_tokens)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })
    }

    #[instrument(skip(self))]
    pub fn try_acquire_permit(&self) -> Result<OwnedSemaphorePermit, TextEmbeddingsError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "overloaded");
                counter.increment(1);
                tracing::error!("{err}");
                TextEmbeddingsError::from(err)
            })
    }

    #[instrument(skip(self))]
    pub async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .acquire_owned()
            .await
            .expect("Semaphore has been closed. This is a bug.")
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_all<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<AllEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "`embed_all` is not available for SPLADE models".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                false,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::AllEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        let total_time = start_time.elapsed();

        metrics::counter!("te_embed_success").increment(1);
        metrics::histogram!("te_embed_duration").record(total_time.as_secs_f64());
        metrics::histogram!("te_embed_tokenization_duration")
            .record(response.metadata.tokenization.as_secs_f64());
        metrics::histogram!("te_embed_queue_duration")
            .record(response.metadata.queue.as_secs_f64());
        metrics::histogram!("te_embed_inference_duration")
            .record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_sparse<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if !self.is_splade() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model with SPLADE pooling".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::PooledEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_embed_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_embed_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_pooled<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        normalize: bool,
        dimensions: Option<usize>,
        permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() && normalize {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);

            let message = "`normalize` is not available for SPLADE models".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        if let Some(dimensions) = dimensions {
            if dimensions == 0 {
                metrics::counter!("te_request_failure", "err" => "validation").increment(1);
                let message = "`dimensions` should be positive".to_string();
                tracing::error!("{message}");
                return Err(TextEmbeddingsError::Validation(message));
            }
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
                batch_counter,
            )
            .await?;

        let InferResult::PooledEmbedding(mut response) = results else {
            panic!("unexpected enum variant")
        };

        if let Some(mrl_dimensions) = dimensions {
            if mrl_dimensions > response.results.len() {
                metrics::counter!("te_request_failure", "err" => "validation").increment(1);

                let message =
                    "`dimensions` should be smaller than the maximum embedding dimension."
                        .to_string();
                tracing::error!("{message}");

                return Err(TextEmbeddingsError::Validation(message));
            }

            response.results.truncate(mrl_dimensions);
        }

        if normalize {
            // Normalize embedding
            // TODO: Should this be normalized with a background thread etc. instead of doing it synchronously here?
            let scale = (1.0
                / response
                    .results
                    .iter()
                    .map(|v| {
                        let v = *v as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt()) as f32;
            for v in response.results.iter_mut() {
                *v *= scale;
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_embed_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_embed_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    async fn embed<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        pooling: bool,
        start_time: &Instant,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<InferResult, TextEmbeddingsError> {
        if self.is_classifier() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let counter = metrics::counter!("te_embed_count");
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode_embedding(inputs.into(), truncate, truncation_direction, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Append the request to the queue
        self.queue
            .append(Entry {
                metadata: Metadata {
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling,
                    token_classification: false,
                },
                encoding,
            })
            .await;

        match batch_counter {
            None => self.notify_batching_task.notify_one(),
            Some(counter) => {
                if counter.fetch_sub(1, Ordering::SeqCst) == 1 {
                    self.notify_batching_task.notify_one();
                }
            }
        }

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "inference");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        Ok(response)
    }

    #[instrument(skip(self, inputs, _permit))]
    pub async fn predict<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<ClassificationInferResponse, TextEmbeddingsError> {
        if !self.is_classifier() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        let counter = metrics::counter!("te_predict_count");
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, None)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Append the request to the queue
        self.queue
            .append(Entry {
                metadata: Metadata {
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling: true,
                    token_classification: false,
                },
                encoding,
            })
            .await;

        match batch_counter {
            None => self.notify_batching_task.notify_one(),
            Some(counter) => {
                if counter.fetch_sub(1, Ordering::SeqCst) == 1 {
                    self.notify_batching_task.notify_one();
                }
            }
        }

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "inference");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        let InferResult::Classification(mut response) = response else {
            panic!("unexpected enum variant")
        };

        if !raw_scores {
            // Softmax
            if response.results.len() > 1 {
                let max = response
                    .results
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();

                let mut den = 0.0;
                for v in response.results.iter_mut() {
                    *v = (*v - max).exp();
                    den += *v;
                }
                for v in response.results.iter_mut() {
                    *v /= den;
                }
            }
            // Sigmoid
            else {
                response.results[0] = 1.0 / (1.0 + (-response.results[0]).exp());
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_predict_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_predict_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, _permit))]
    pub async fn predict_tokens<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
        batch_counter: Option<Arc<AtomicUsize>>,
    ) -> Result<(Vec<TokenPrediction>, usize, Duration, Duration, Duration), TextEmbeddingsError>
    {
        if !self.is_classifier() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        let counter = metrics::counter!("te_predict_count");
        counter.increment(1);

        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, None)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        let (response_tx, response_rx) = oneshot::channel();

        self.queue
            .append(Entry {
                metadata: Metadata {
                    client_batch: batch_counter.clone(),
                    response_tx,
                    tokenization: start_time.elapsed(),
                    queue_time: Instant::now(),
                    prompt_tokens: encoding.input_ids.len(),
                    pooling: false,
                    token_classification: true,
                },
                encoding,
            })
            .await;

        match batch_counter {
            None => self.notify_batching_task.notify_one(),
            Some(counter) => {
                if counter.fetch_sub(1, Ordering::SeqCst) == 1 {
                    self.notify_batching_task.notify_one();
                }
            }
        }

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "inference");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        let InferResult::TokenClassification(mut response) = response else {
            panic!("unexpected enum variant")
        };

        if !raw_scores {
            for (_, _, scores, _, _) in response.results.iter_mut() {
                if scores.len() > 1 {
                    let max = scores
                        .iter()
                        .copied()
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();

                    let mut den = 0.0;
                    for v in scores.iter_mut() {
                        *v = (*v - max).exp();
                        den += *v;
                    }
                    for v in scores.iter_mut() {
                        *v /= den;
                    }
                } else {
                    scores[0] = 1.0 / (1.0 + (-scores[0]).exp());
                }
            }
        }

        let total_time = start_time.elapsed();

        let counter = metrics::counter!("te_predict_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_predict_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok((
            response.results,
            response.metadata.prompt_tokens,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
        ))
    }

    #[instrument(skip(self))]
    pub fn is_classifier(&self) -> bool {
        matches!(self.backend.model_type, ModelType::Classifier)
    }

    #[instrument(skip(self))]
    pub fn is_splade(&self) -> bool {
        matches!(
            self.backend.model_type,
            ModelType::Embedding(text_embeddings_backend::Pool::Splade)
        )
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> bool {
        self.backend.health().await.is_ok()
    }

    #[instrument(skip(self))]
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        self.backend.health_watcher()
    }
}

#[derive(Debug)]
struct PipelineLifetime {
    pool: BackendPool,
    queue: Queue,
}
impl Drop for PipelineLifetime {
    fn drop(&mut self) {
        self.pool.close();
        self.queue.close();
    }
}

struct DispatchedBatch(Option<NextBatch>);
impl Drop for DispatchedBatch {
    fn drop(&mut self) {
        if let Some((metadata, _)) = self.0.take() {
            for entry in metadata {
                let _ = entry.response_tx.send(Err(BackendError::Unhealthy));
            }
        }
    }
}

async fn batching_task(
    pool: BackendPool,
    queue: Queue,
    notify: Arc<Notify>,
    sender: mpsc::Sender<DispatchedBatch>,
) {
    let mut changed = pool.changes();
    loop {
        let arrival = notify.notified();
        tokio::pin!(arrival);
        arrival.as_mut().enable();
        // Reserve the only shared parking slot before preparing any batch.
        let permit = tokio::select! {
            permit = sender.reserve() => match permit { Ok(p) => p, Err(_) => break },
            _ = changed.changed() => { if pool.is_closed() { break; } continue; }
        };
        if pool.is_closed() {
            break;
        }
        let started = Instant::now();
        if let Some(batch) = queue.next_batch().await {
            pool.observe_preparation(started.elapsed());
            permit.send(DispatchedBatch(Some(batch)));
        } else {
            drop(permit);
            tokio::select! { _ = arrival => {}, _ = changed.changed() => {} }
        }
    }
    queue.close();
}

#[instrument(skip_all, fields(replica))]
async fn backend_task(
    backend: BackendPool,
    queue: Queue,
    receiver: Arc<Mutex<mpsc::Receiver<DispatchedBatch>>>,
    replica: usize,
) {
    let mut changed = backend.changes();
    loop {
        if backend.is_closed() {
            break;
        }
        let next = async { receiver.lock().await.recv().await };
        let mut dispatched = tokio::select! {
            batch = next => match batch { Some(b) => b, None => break },
            _ = changed.changed() => { if backend.is_closed() { break; } continue; }
        };
        let execution = match backend.acquire_replica(replica).await {
            Ok(execution) => execution,
            Err(_) => break,
        };
        let Some(batch) = prune_canceled_batch(dispatched.0.take().unwrap()) else {
            continue;
        };
        match &backend.model_type {
            ModelType::Classifier => {
                let token_classification = batch.0.iter().any(|m| m.token_classification);

                if token_classification {
                    let encoding = batch.1.clone();
                    let results = execution.predict_tokens(batch.1).await;

                    tokio::task::spawn_blocking(move || match results {
                        Ok((mut predictions, inference_duration)) => {
                            batch.0.into_iter().enumerate().for_each(|(i, m)| {
                                let infer_metadata = InferMetadata {
                                    prompt_tokens: m.prompt_tokens,
                                    tokenization: m.tokenization,
                                    queue: m
                                        .queue_time
                                        .elapsed()
                                        .saturating_sub(inference_duration),
                                    inference: inference_duration,
                                };

                                let token_predictions = predictions.remove(&i).expect(
                                    "prediction not found in results. This is a backend bug.",
                                );

                                let start_idx = encoding.cumulative_seq_lengths[i] as usize;
                                let _end_idx = encoding.cumulative_seq_lengths[i + 1] as usize;

                                let token_predictions: Vec<TokenPrediction> = token_predictions
                                    .into_iter()
                                    .enumerate()
                                    .map(|(token_idx, scores)| {
                                        let global_token_idx = start_idx + token_idx;
                                        let token_id = encoding.input_ids[global_token_idx];
                                        let token = encoding
                                            .tokens
                                            .get(global_token_idx)
                                            .cloned()
                                            .unwrap_or_else(|| format!("<token_{}>", token_id));
                                        let start =
                                            encoding.offsets.get(global_token_idx).map(|o| o.0);
                                        let end =
                                            encoding.offsets.get(global_token_idx).map(|o| o.1);
                                        (token, token_id, scores, start, end)
                                    })
                                    .collect();

                                let _ = m.response_tx.send(Ok(InferResult::TokenClassification(
                                    TokenClassificationInferResponse {
                                        results: token_predictions,
                                        metadata: infer_metadata,
                                    },
                                )));
                            });
                        }
                        Err(err) => {
                            batch.0.into_iter().for_each(|m| {
                                let _ = m.response_tx.send(Err(err.clone()));
                            });
                        }
                    });
                } else {
                    let results = execution.predict(batch.1).await;

                    tokio::task::spawn_blocking(move || match results {
                        Ok((mut predictions, inference_duration)) => {
                            batch.0.into_iter().enumerate().for_each(|(i, m)| {
                                let infer_metadata = InferMetadata {
                                    prompt_tokens: m.prompt_tokens,
                                    tokenization: m.tokenization,
                                    queue: m.queue_time.elapsed().saturating_sub(inference_duration),
                                    inference: inference_duration,
                                };

                                let _ = m.response_tx.send(Ok(InferResult::Classification(
                                    ClassificationInferResponse {
                                        results: predictions.remove(&i).expect(
                                            "prediction not found in results. This is a backend bug.",
                                        ),
                                        metadata: infer_metadata,
                                    },
                                )));
                            });
                        }
                        Err(err) => {
                            batch.0.into_iter().for_each(|m| {
                                let _ = m.response_tx.send(Err(err.clone()));
                            });
                        }
                    });
                }
            }
            ModelType::Embedding(_) => {
                let results = execution.embed(batch.1).await;

                tokio::task::spawn_blocking(move || match results {
                    Ok((mut embeddings, inference_duration)) => {
                        batch.0.into_iter().enumerate().for_each(|(i, m)| {
                            let metadata = InferMetadata {
                                prompt_tokens: m.prompt_tokens,
                                tokenization: m.tokenization,
                                queue: m.queue_time.elapsed().saturating_sub(inference_duration),
                                inference: inference_duration,
                            };

                            let results = match embeddings
                                .remove(&i)
                                .expect("embedding not found in results. This is a backend bug.")
                            {
                                Embedding::Pooled(e) => {
                                    InferResult::PooledEmbedding(PooledEmbeddingsInferResponse {
                                        results: e,
                                        metadata,
                                    })
                                }
                                Embedding::All(e) => {
                                    InferResult::AllEmbedding(AllEmbeddingsInferResponse {
                                        results: e,
                                        metadata,
                                    })
                                }
                            };

                            let _ = m.response_tx.send(Ok(results));
                        })
                    }
                    Err(err) => {
                        batch.0.into_iter().for_each(|m| {
                            let _ = m.response_tx.send(Err(err.clone()));
                        });
                    }
                });
            }
        };
    }
    queue.close();
}

#[derive(Debug)]
pub struct InferMetadata {
    pub prompt_tokens: usize,
    pub tokenization: Duration,
    pub queue: Duration,
    pub inference: Duration,
}

#[derive(Debug)]
pub(crate) enum InferResult {
    Classification(ClassificationInferResponse),
    TokenClassification(TokenClassificationInferResponse),
    PooledEmbedding(PooledEmbeddingsInferResponse),
    AllEmbedding(AllEmbeddingsInferResponse),
}

#[derive(Debug)]
pub struct ClassificationInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

/// Token text, token ID, class scores, and optional character offsets.
pub type TokenPrediction = (String, u32, Vec<f32>, Option<usize>, Option<usize>);

#[derive(Debug)]
pub struct TokenClassificationInferResponse {
    pub results: Vec<TokenPrediction>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct PooledEmbeddingsInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct AllEmbeddingsInferResponse {
    pub results: Vec<Vec<f32>>,
    pub metadata: InferMetadata,
}
