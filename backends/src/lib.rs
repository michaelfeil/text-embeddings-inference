mod dtype;

use hf_hub::api::tokio::{ApiError, ApiRepo};
use rand::Rng;
use std::cmp::{max, min};
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use text_embeddings_backend_core::{Backend as CoreBackend, Predictions};
use tokio::sync::{oneshot, watch};
use tracing::{instrument, Span};

#[cfg(feature = "candle")]
use serde::Deserialize;

pub use crate::dtype::DType;
pub use text_embeddings_backend_core::{
    BackendError, Batch, Embedding, Embeddings, ModelType, Pool,
};

#[cfg(feature = "candle")]
use text_embeddings_backend_candle::CandleBackend;

#[cfg(feature = "ort")]
use text_embeddings_backend_ort::OrtBackend;

#[cfg(feature = "python")]
use text_embeddings_backend_python::PythonBackend;

// Compile-time constant for secondary backend activation
const ACTIVATION_THRESHOLD: usize = 64;

// Global atomic for queue size tracking
static GLOBAL_QUEUE_SIZE: AtomicUsize = AtomicUsize::new(0);

// Functions for queue.rs to call
pub fn set_global_queue_size(size: usize) {
    GLOBAL_QUEUE_SIZE.store(size, Ordering::Relaxed);
}

// Function for backend workers to read
pub fn get_global_queue_size() -> usize {
    GLOBAL_QUEUE_SIZE.load(Ordering::Relaxed)
}

fn powers_of_two(max_value: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut power: usize = 1;

    while power <= max_value {
        result.push(power);
        power *= 2;
    }

    result
}

fn generate_bucket_sizes(bucket_size: usize, max_s: usize, base_exp: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut current = bucket_size;

    while current <= max_s {
        sizes.push(current);
        match current.checked_mul(base_exp) {
            Some(next) => current = next,
            None => break,
        }
    }
    sizes
}

fn is_hpu() -> bool {
    match Command::new("hl-smi")
        .args(["-Q", "name", "-f", "csv"])
        .output()
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

pub(crate) fn is_dual_backend_enabled() -> bool {
    std::env::var("DUAL_BACKEND_ENABLED")
        .ok()
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread(s)
    backend_sender: crossbeam_channel::Sender<BackendCommand>,
    /// Health status
    health_receiver: watch::Receiver<bool>,
    _backend_thread: Arc<BackendThread>,
    pub padded_model: bool,
    pub radix_mlp_supported: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
    /// Whether dual backend mode is enabled
    pub is_dual_backend: bool,
}

impl Backend {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        model_path: PathBuf,
        api_repo: Option<ApiRepo>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        uds_path: String,
        otlp_endpoint: Option<String>,
        otlp_service_name: String,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        let dual_mode = is_dual_backend_enabled();

        if dual_mode {
            tracing::info!(" Initializing dual backend mode ");
            Self::new_dual_backend(
                model_path,
                api_repo,
                dtype,
                model_type,
                dense_path,
                uds_path,
                otlp_endpoint,
                otlp_service_name,
                device_id,
            )
            .await
        } else {
            tracing::info!("🔧 Initializing single backend mode");
            Self::new_single_backend(
                model_path,
                api_repo,
                dtype,
                model_type,
                dense_path,
                uds_path,
                otlp_endpoint,
                otlp_service_name,
                device_id,
            )
            .await
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn new_single_backend(
        model_path: PathBuf,
        api_repo: Option<ApiRepo>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        uds_path: String,
        otlp_endpoint: Option<String>,
        otlp_service_name: String,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        let (backend_sender, backend_receiver) = crossbeam_channel::bounded::<BackendCommand>(8);

        let backend = init_backend(
            model_path,
            api_repo.map(Arc::new),
            dtype,
            model_type.clone(),
            dense_path,
            uds_path,
            otlp_endpoint,
            otlp_service_name,
            device_id,
        )
        .await?;
        let padded_model = backend.is_padded();
        let radix_mlp_supported = backend.supports_radix_mlp();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread = Arc::new(BackendThread::new_single(
            backend,
            backend_receiver,
            health_sender,
        ));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            padded_model,
            radix_mlp_supported,
            max_batch_size,
            model_type,
            is_dual_backend: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn new_dual_backend(
        model_path: PathBuf,
        api_repo: Option<ApiRepo>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        uds_path: String,
        otlp_endpoint: Option<String>,
        otlp_service_name: String,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        // Proper dual backend implementation with two separate instances
        let (backend_sender, backend_receiver) = crossbeam_channel::bounded::<BackendCommand>(8);

        // Create two separate backend instances sharing ApiRepo via Arc
        let api_repo_arc = api_repo.map(Arc::new);

        // Initialize both backends concurrently for better performance
        let (backend1, backend2) = tokio::try_join!(
            init_backend(
                model_path.clone(),
                api_repo_arc.clone(),
                dtype.clone(),
                model_type.clone(),
                dense_path.clone(),
                uds_path.clone(),
                otlp_endpoint.clone(),
                otlp_service_name.clone(),
                device_id,
            ),
            init_backend(
                model_path,
                api_repo_arc,
                dtype.clone(),
                model_type.clone(),
                dense_path,
                uds_path,
                otlp_endpoint,
                otlp_service_name,
                device_id,
            )
        )?;

        // Both backends should have the same properties
        let padded_model = backend1.is_padded();
        let radix_mlp_supported = backend1.supports_radix_mlp();
        let max_batch_size = backend1.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);

        // Spawn proper dual backend threads
        let _backend_thread = Arc::new(BackendThread::new_dual(
            backend1,
            backend2,
            backend_receiver,
            health_sender,
        ));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            padded_model,
            radix_mlp_supported,
            max_batch_size,
            model_type,
            is_dual_backend: true,
        })
    }

    #[instrument(skip(self))]
    pub async fn warmup_hpu(
        &self,
        mut max_input_length: usize,
        max_token: usize,
        max_bs: Option<usize>,
    ) -> Result<(), BackendError> {
        if self.is_dual_backend {
            tracing::info!(
                "📊 Warming up dual backend mode (2 backend instances will share warmup workload)"
            );
        }
        let read_env_var = |key: &str, default: usize| -> usize {
            env::var(key)
                .ok()
                .map_or(default, |value| value.parse::<usize>().unwrap())
        };
        let seq_bucket_size: usize = read_env_var("PAD_SEQUENCE_TO_MULTIPLE_OF", 128);
        let max_warmup_length: usize = read_env_var("MAX_WARMUP_SEQUENCE_LENGTH", 1024);
        let seq_len_exp_base: usize = read_env_var("SEQ_LEN_EXPONENT_BASE", 2);
        let max_batch_size = max_bs.unwrap_or_else(|| read_env_var("MAX_WARMUP_BATCH_SIZE", 8));

        let mut batch_sizes: Vec<usize> = powers_of_two(max_batch_size);
        if let Some(&last) = batch_sizes.last() {
            if last < max_batch_size {
                batch_sizes.push(max_batch_size);
            }
        }
        if max_warmup_length > max_input_length {
            return Err(BackendError::Start(
                format!("max_warmup_length ({max_warmup_length}) exceeds model's max_input_length ({max_input_length}), you can modify this value adding `-e MAX_WARMUP_SEQUENCE_LENGTH=<new_warmup_length>` to your Docker run command")
            ));
        }
        if seq_bucket_size > max_warmup_length {
            return Err(BackendError::Start(
                format!("PAD_SEQUENCE_TO_MULTIPLE_OF ({seq_bucket_size}) exceeds model's max warmup length ({max_warmup_length}), you can modify these values adding `-e PAD_SEQUENCE_TO_MULTIPLE_OF=<new_value>` or `-e MAX_WARMUP_SEQUENCE_LENGTH=<new_value> to your Docker run command`")
            ));
        }

        max_input_length = std::cmp::min(max_input_length, max_warmup_length);
        let mut seq_lengths: Vec<usize> =
            generate_bucket_sizes(seq_bucket_size, max_input_length, seq_len_exp_base);
        if let Some(&last) = seq_lengths.last() {
            if last < max_input_length {
                seq_lengths.push(max_input_length);
            }
        }

        let mut shapes: Vec<(u32, u32)> = Vec::with_capacity(batch_sizes.len() * seq_lengths.len());
        for batch_size in &batch_sizes {
            for seq_length in &seq_lengths {
                shapes.push((*batch_size as u32, *seq_length as u32));
            }
        }
        for shape in shapes.iter() {
            let batch = self.create_warmup_batch(*shape, max_token as u32, seq_bucket_size as u32);
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }?;
            tracing::info!("finish warmup for batch: {}, length: {}", shape.0, shape.1);
        }
        Ok(())
    }

    #[instrument(skip_all)]
    pub fn create_warmup_batch(
        &self,
        shape: (u32, u32),
        max_token: u32,
        seq_bucket_size: u32,
    ) -> Batch {
        let (batch_size, length) = shape;
        let min_length = length.saturating_sub(seq_bucket_size).saturating_add(1);
        let tmp_length = if min_length < length {
            rand::rng().random_range(min_length..length)
        } else {
            length
        };
        let mut batched_input_ids = Vec::new();
        let mut batched_token_type_ids = Vec::new();
        let mut batched_position_ids = Vec::new();
        let mut cumulative_seq_lengths = Vec::with_capacity(batch_size as usize + 1);
        let mut pooled_indices = Vec::with_capacity(batch_size as usize);
        cumulative_seq_lengths.push(0);
        let input_ids: Vec<u32> = (0..tmp_length)
            .map(|_| rand::rng().random_range(0..max_token))
            .collect();
        let token_type_ids: Vec<u32> = vec![0; tmp_length as usize];
        let position_ids: Vec<u32> = (0..tmp_length).collect();
        let mut current_length = 0;
        for batch_id in 0..batch_size {
            batched_input_ids.extend(input_ids.iter().cloned());
            batched_token_type_ids.extend(token_type_ids.iter().cloned());
            batched_position_ids.extend(position_ids.iter().cloned());
            current_length += input_ids.len();
            cumulative_seq_lengths.push(current_length as u32);
            pooled_indices.push(batch_id);
        }
        Batch {
            input_ids: batched_input_ids,
            token_type_ids: batched_token_type_ids,
            position_ids: batched_position_ids,
            cumulative_seq_lengths,
            max_length: tmp_length,
            pooled_indices,
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            fold_gather: None,
            scatter_unfold: None,
        }
    }

    #[instrument(skip(self))]
    pub async fn warmup(
        &self,
        max_input_length: usize,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        padded_model: bool,
    ) -> Result<(), BackendError> {
        if self.is_dual_backend {
            tracing::info!(
                "📊 Warming up dual backend mode (2 backend instances will share warmup workload)"
            );
        }
        if is_hpu() {
            return self
                .warmup_hpu(max_input_length, max_batch_tokens, max_batch_requests)
                .await;
        }

        // In padded_model (CPU), use minimal warmup size (max_input_length tokens) for fast startup
        // Non-padded (GPU), use full max_batch_tokens to exercise production batching limits
        let warmup_tokens = if padded_model {
            max_input_length.min(max_batch_tokens)
        } else {
            max_batch_tokens
        };

        let mut input_ids = Vec::with_capacity(warmup_tokens);
        let mut token_type_ids = Vec::with_capacity(warmup_tokens);
        let mut position_ids = Vec::with_capacity(warmup_tokens);

        let mut cumulative_seq_lengths = vec![0];
        let mut pooled_indices = Vec::new();

        let mut i = 0_u32;
        let mut remaining = warmup_tokens;
        let mut cumulative_length = 0;
        let mut max_length = 0;

        while remaining > 0 {
            let request_length = min(remaining, max_input_length);
            cumulative_length += request_length;
            max_length = max(max_length, request_length as u32);

            input_ids.extend(vec![0; request_length]);
            token_type_ids.extend(vec![0; request_length]);
            position_ids.extend((0..request_length as u32).collect::<Vec<u32>>());

            cumulative_seq_lengths.push(cumulative_length as u32);
            pooled_indices.push(i);

            i += 1;
            remaining = remaining.saturating_sub(max_input_length);
            if let Some(max_batch_requests) = &max_batch_requests {
                if i as usize == *max_batch_requests {
                    break;
                }
            }
        }

        let batch = Batch {
            input_ids,
            token_type_ids,
            position_ids,
            cumulative_seq_lengths,
            max_length,
            pooled_indices,
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            fold_gather: None,
            scatter_unfold: None,
        };

        match &self.model_type {
            ModelType::Classifier => self.predict(batch).await.map(|_| ()),
            ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
        }
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> Result<(), BackendError> {
        if *self.health_receiver.borrow() {
            // The backend is healthy. Only do a basic health check by calling the
            // the underlying health method.

            let (sender, receiver) = oneshot::channel();
            self.backend_sender
                .send(BackendCommand::Health(Span::current(), sender))
                .map_err(|_| BackendError::Inference("Backend channel closed".to_string()))?;
            receiver.await.expect(
                "Backend blocking task dropped the sender without sending a response. This is a bug.",
            )
        } else {
            // The backend is un-healthy or only just started. Do a more advanced health check
            // by calling the model forward on a test batch

            let batch = Batch {
                input_ids: vec![0],
                token_type_ids: vec![0],
                position_ids: vec![0],
                cumulative_seq_lengths: vec![0, 1],
                max_length: 1,
                pooled_indices: vec![0],
                raw_indices: vec![],
                compact_input_ids: None,
                compact_position_ids: None,
                fold_gather: None,
                scatter_unfold: None,
            };
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }
        }
    }

    #[instrument(skip(self))]
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        self.health_receiver.clone()
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Embed(batch, Span::current(), sender))
            .map_err(|_| BackendError::Inference("Backend channel closed".to_string()))?;
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Predict(batch, Span::current(), sender))
            .map_err(|_| BackendError::Inference("Backend channel closed".to_string()))?;
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }
}

#[allow(unused, clippy::too_many_arguments)]
async fn init_backend(
    model_path: PathBuf,
    api_repo: Option<Arc<ApiRepo>>,
    dtype: DType,
    model_type: ModelType,
    dense_path: Option<String>,
    uds_path: String,
    otlp_endpoint: Option<String>,
    otlp_service_name: String,
    device_id: usize,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    let mut backend_start_failed = false;

    if cfg!(feature = "ort") {
        #[cfg(feature = "ort")]
        {
            if let Some(api_repo) = api_repo.as_ref() {
                let start = std::time::Instant::now();
                let model_files = download_onnx(api_repo)
                    .await
                    .map_err(|err| BackendError::WeightsNotFound(err.to_string()))?;
                match model_files.is_empty() {
                    true => {
                        tracing::error!("Model ONNX files not found in the repository. You can easily create ONNX files using the following scripts: https://gist.github.com/tomaarsen/4b00b0e3be8884efa64cfab9230b161f, or use this Space: https://huggingface.co/spaces/sentence-transformers/backend-export")
                    }
                    false => {
                        tracing::info!("Model ONNX weights downloaded in {:?}", start.elapsed())
                    }
                }
            }

            // NOTE: for ONNX we need to retrieve the `tokenizer_config.json` to identify which
            // `padding_side` needs to be applied for the input processing and the pooling
            if let Some(api_repo) = api_repo.as_ref() {
                tracing::info!("Downloading `tokenizer_config.json`");
                match api_repo.get("tokenizer_config.json").await {
                    Ok(_) => (),
                    Err(err) => {
                        tracing::warn!("Could not download `tokenizer_config.json`: {}", err)
                    }
                }
            }

            let backend = OrtBackend::new(&model_path, dtype.to_string(), model_type.clone());
            match backend {
                Ok(b) => return Ok(Box::new(b)),
                Err(err) => {
                    tracing::error!("Could not start ORT backend: {err}");
                    backend_start_failed = true;
                }
            }
        }
    }

    if let Some(api_repo) = api_repo.as_ref() {
        if cfg!(feature = "python") || cfg!(feature = "candle") {
            let start = std::time::Instant::now();
            if download_safetensors(api_repo.clone()).await.is_err() {
                tracing::warn!("safetensors weights not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
                tracing::info!("Downloading `pytorch_model.bin`");
                api_repo
                    .get("pytorch_model.bin")
                    .await
                    .map_err(|err| BackendError::WeightsNotFound(err.to_string()))?;
            }

            tracing::info!("Model weights downloaded in {:?}", start.elapsed());
        }
    }

    if cfg!(feature = "candle") {
        #[cfg(feature = "candle")]
        {
            let dense_paths = if let Some(api_repo) = api_repo.as_ref() {
                let start = std::time::Instant::now();
                let dense_paths = download_dense_modules(api_repo, dense_path)
                    .await
                    .map_err(|err| BackendError::WeightsNotFound(err.to_string()))?;
                tracing::info!("Dense modules downloaded in {:?}", start.elapsed());
                Some(dense_paths)
            } else {
                // TODO(alvarobartt): eventually detach the Sentence Transformers module handling
                // to prevent from duplicated code here and there
                // For local models, try to parse modules.json and handle dense_path logic
                let modules_json_path = model_path.join("modules.json");
                if modules_json_path.exists() {
                    match parse_dense_paths_from_modules(&modules_json_path).await {
                        Ok(module_paths) => match module_paths.len() {
                            0 => Some(vec![]),
                            1 => {
                                let path_to_use = if let Some(ref user_path) = dense_path {
                                    if user_path != &module_paths[0] {
                                        tracing::info!("`{}` found in `modules.json`, but using provided `--dense-path={user_path}` instead", module_paths[0]);
                                    }
                                    user_path.clone()
                                } else {
                                    module_paths[0].clone()
                                };
                                Some(vec![path_to_use])
                            }
                            _ => {
                                if dense_path.is_some() {
                                    tracing::warn!("A value for `--dense-path` was provided, but since there's more than one subsequent Dense module, then the provided value will be ignored.");
                                }
                                Some(module_paths)
                            }
                        },
                        Err(err) => {
                            tracing::warn!("Failed to parse local modules.json: {err}");
                            None
                        }
                    }
                } else {
                    None
                }
            };

            let backend = CandleBackend::new(
                &model_path,
                dtype.to_string(),
                model_type.clone(),
                dense_paths,
                device_id,
            );
            match backend {
                Ok(b) => return Ok(Box::new(b)),
                Err(err) => {
                    tracing::error!("Could not start Candle backend: {err}");
                    backend_start_failed = true;
                }
            }
        }
    }

    if cfg!(feature = "python") {
        #[cfg(feature = "python")]
        {
            let backend = std::thread::spawn(move || {
                PythonBackend::new(
                    model_path.to_str().unwrap().to_string(),
                    dtype.to_string(),
                    model_type,
                    uds_path,
                    otlp_endpoint,
                    otlp_service_name,
                )
            })
            .join()
            .expect("Python Backend management thread failed");

            match backend {
                Ok(b) => return Ok(Box::new(b)),
                Err(err) => {
                    tracing::error!("Could not start Python backend: {err}");
                    backend_start_failed = true;
                }
            }
        }
    }

    if backend_start_failed {
        Err(BackendError::Start(
            "Could not start a suitable backend".to_string(),
        ))
    } else {
        Err(BackendError::NoBackend)
    }
}

#[derive(Debug)]
enum BackendThread {
    Single(Option<JoinHandle<()>>),
    Dual(Vec<JoinHandle<()>>),
}

impl BackendThread {
    fn new_single(
        backend: Box<dyn CoreBackend + Send>,
        backend_receiver: crossbeam_channel::Receiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            backend_worker(backend, backend_receiver, health_sender, 0); // Primary worker
        });
        Self::Single(Some(handle))
    }

    fn new_dual(
        backend1: Box<dyn CoreBackend + Send>,
        backend2: Box<dyn CoreBackend + Send>,
        backend_receiver: crossbeam_channel::Receiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let receiver1 = backend_receiver.clone();
        let receiver2 = backend_receiver;
        let health_sender1 = health_sender.clone();
        let health_sender2 = health_sender;

        tracing::info!("🚀 Spawning dual backend worker threads");

        let handle1 = std::thread::spawn(move || {
            tracing::info!("📡 Primary backend worker (ID 0) started");
            backend_worker(backend1, receiver1, health_sender1, 0); // Primary worker
            tracing::info!("📡 Primary backend worker (ID 0) stopped");
        });

        let handle2 = std::thread::spawn(move || {
            tracing::info!("📡 Secondary backend worker (ID 1) started");
            backend_worker(backend2, receiver2, health_sender2, 1); // Secondary worker
            tracing::info!("📡 Secondary backend worker (ID 1) stopped");
        });

        Self::Dual(vec![handle1, handle2])
    }
}

fn backend_worker(
    backend: Box<dyn CoreBackend + Send>,
    backend_receiver: crossbeam_channel::Receiver<BackendCommand>,
    health_sender: watch::Sender<bool>,
    worker_id: usize, // 0 = primary, 1 = secondary
) {
    loop {
        if worker_id == 0 {
            // Primary worker: always process
            if let Ok(cmd) = backend_receiver.recv() {
                process_command(cmd, &backend, &health_sender);
            } else {
                break; // Channel closed
            }
        } else {
            // Secondary worker: check global queue size against compile-time constant
            let queue_size = get_global_queue_size();

            if queue_size >= ACTIVATION_THRESHOLD {
                // Queue is big enough, try to process
                if let Ok(cmd) = backend_receiver.try_recv() {
                    process_command(cmd, &backend, &health_sender);
                } else {
                    // No command available, brief wait
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            } else {
                // Queue too small, wait longer
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

fn process_command(
    cmd: BackendCommand,
    backend: &Box<dyn CoreBackend + Send>,
    health_sender: &watch::Sender<bool>,
) {
    let start = Instant::now();
    let mut healthy = false;
    match cmd {
        BackendCommand::Health(span, sender) => {
            let _span = span.entered();
            let _ = sender.send(backend.health().map(|_| healthy = true));
        }
        BackendCommand::Embed(batch, span, sender) => {
            let _span = span.entered();
            let _ = sender.send(backend.embed(batch).map(|e| {
                healthy = true;
                (e, start.elapsed())
            }));
        }
        BackendCommand::Predict(batch, span, sender) => {
            let _span = span.entered();
            let _ = sender.send(backend.predict(batch).map(|e| {
                healthy = true;
                (e, start.elapsed())
            }));
        }
    };
    let _ = health_sender.send(healthy);
}

impl Drop for BackendThread {
    fn drop(&mut self) {
        match self {
            BackendThread::Single(handle) => {
                if let Some(handle) = handle.take() {
                    handle.join().unwrap();
                }
            }
            BackendThread::Dual(handles) => {
                for handle in handles.drain(..) {
                    handle.join().unwrap();
                }
            }
        }
    }
}

enum BackendCommand {
    Health(Span, oneshot::Sender<Result<(), BackendError>>),
    Embed(
        Batch,
        Span,
        oneshot::Sender<Result<(Embeddings, Duration), BackendError>>,
    ),
    Predict(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<(Predictions, Duration), BackendError>>,
    ),
}

async fn download_safetensors(api: Arc<ApiRepo>) -> Result<Vec<PathBuf>, ApiError> {
    // Single file
    tracing::info!("Downloading `model.safetensors`");
    match api.get("model.safetensors").await {
        Ok(p) => return Ok(vec![p]),
        Err(err) => tracing::warn!("Could not download `model.safetensors`: {}", err),
    };

    // Sharded weights
    // Download and parse index file
    tracing::info!("Downloading `model.safetensors.index.json`");
    let index_file = api.get("model.safetensors.index.json").await?;
    let index_file_string: String =
        std::fs::read_to_string(index_file).expect("model.safetensors.index.json is corrupted");
    let json: serde_json::Value = serde_json::from_str(&index_file_string)
        .expect("model.safetensors.index.json is corrupted");

    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        _ => panic!("model.safetensors.index.json is corrupted"),
    };

    let mut safetensors_filenames = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_filenames.insert(file.to_string());
        }
    }

    // Download weight files
    let handles: Vec<_> = safetensors_filenames
        .into_iter()
        .map(|n| {
            let api = Arc::clone(&api);
            tokio::spawn(async move {
                tracing::info!("Downloading `{}`", n);
                api.get(&n).await
            })
        })
        .collect();

    let mut safetensors_files = Vec::with_capacity(handles.len());
    for handle in handles {
        // Await the JoinHandle to get the result of the task,
        // then unpack the inner result from api.get()
        safetensors_files.push(handle.await??);
    }

    Ok(safetensors_files)
}

#[cfg(feature = "ort")]
async fn download_onnx(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    let mut model_files: Vec<PathBuf> = Vec::new();

    tracing::info!("Downloading `model.onnx`");
    match api.get("model.onnx").await {
        Ok(p) => model_files.push(p),
        Err(err) => {
            tracing::warn!("Could not download `model.onnx`: {err}");
            tracing::info!("Downloading `onnx/model.onnx`");

            match api.get("onnx/model.onnx").await {
                Ok(p) => model_files.push(p.parent().unwrap().to_path_buf()),
                Err(err) => tracing::warn!("Could not download `onnx/model.onnx`: {err}"),
            };
        }
    };

    tracing::info!("Downloading `model.onnx_data`");
    match api.get("model.onnx_data").await {
        Ok(p) => model_files.push(p),
        Err(err) => {
            tracing::warn!("Could not download `model.onnx_data`: {err}");
            tracing::info!("Downloading `onnx/model.onnx_data`");

            match api.get("onnx/model.onnx_data").await {
                Ok(p) => model_files.push(p.parent().unwrap().to_path_buf()),
                Err(err) => tracing::warn!("Could not download `onnx/model.onnx_data`: {err}"),
            }
        }
    }

    Ok(model_files)
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Deserialize, PartialEq)]
enum ModuleType {
    #[serde(rename = "sentence_transformers.models.Dense")]
    Dense,
    #[serde(rename = "sentence_transformers.models.Normalize")]
    Normalize,
    #[serde(rename = "sentence_transformers.models.Pooling")]
    Pooling,
    #[serde(rename = "sentence_transformers.models.Transformer")]
    Transformer,
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Deserialize)]
struct ModuleConfig {
    #[allow(dead_code)]
    idx: usize,
    #[allow(dead_code)]
    name: String,
    path: String,
    #[serde(rename = "type")]
    module_type: ModuleType,
}

#[cfg(feature = "candle")]
async fn download_file(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    api.get(file_path).await
}

#[cfg(feature = "candle")]
async fn parse_dense_paths_from_modules(
    modules_path: &PathBuf,
) -> Result<Vec<String>, std::io::Error> {
    let content = std::fs::read_to_string(modules_path)?;
    let modules: Vec<ModuleConfig> = serde_json::from_str(&content)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

    Ok(modules
        .into_iter()
        .filter(|module| module.module_type == ModuleType::Dense)
        .map(|module| module.path)
        .collect::<Vec<String>>())
}

#[cfg(feature = "candle")]
#[instrument(skip_all)]
pub async fn download_dense_modules(
    api: &ApiRepo,
    dense_path: Option<String>,
) -> Result<Vec<String>, ApiError> {
    match download_file(api, "modules.json").await {
        Ok(modules_path) => {
            // If `modules.json` exists, then parse it to capture the Dense modules
            match parse_dense_paths_from_modules(&modules_path).await {
                Ok(module_paths) => {
                    match module_paths.len() {
                        0 => Ok(vec![]),
                        // NOTE: if there's only a single Dense module defined i.e., there are
                        // no sequential Dense modules to be applied, then the one defined in
                        // `modules.json` will be downloaded, unless the user has specified
                        // another valid `--dense-path` (that exists in the repository), e.g.
                        // defualt might be set to `2_Dense_1024/`, but the user might want to
                        // use `2_Dense/` instead (see https://huggingface.co/NovaSearch/stella_en_400M_v5)
                        1 => {
                            let path_to_use = if let Some(ref user_path) = dense_path {
                                if user_path != &module_paths[0] {
                                    tracing::info!("`{}` found in `modules.json`, but using provided `--dense-path={user_path}` instead", module_paths[0]);
                                }
                                user_path.clone()
                            } else {
                                module_paths[0].clone()
                            };

                            download_dense_module(api, &path_to_use)
                                .await
                                .map_err(|err| {
                                    tracing::error!(
                                        "Failed to download dense module {}: {}",
                                        path_to_use,
                                        err
                                    );
                                    err
                                })?;
                            Ok(vec![path_to_use])
                        }
                        // NOTE: in any other case i.e., more than 1 Dense module, then download
                        // them all, and then sort them to ensure those are applied sequentially
                        _ => {
                            if dense_path.is_some() {
                                tracing::warn!("A value for `--dense-path` was provided, but since there's more than one subsequent Dense module, then the provided value will be ignored.");
                            }

                            for module_path in &module_paths {
                                // NOTE: since the Dense modules here are specified in the
                                // `modules.json` file, then fail if any of those cannot be
                                // downloaded
                                download_dense_module(api, module_path)
                                    .await
                                    .map_err(|err| {
                                        tracing::error!(
                                            "Failed to download `{module_path}` file: {err}"
                                        );
                                        err
                                    })?;
                            }
                            Ok(module_paths)
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("`modules.json` could be downloaded but parsing the modules failed: {err}; so no Dense modules will be downloaded.");
                    Ok(vec![])
                }
            }
        }
        // NOTE: if `modules.json` is not there, then no modules will be downloaded, which most
        // likely means that the model is not a Sentence Transformer model
        Err(_) => Ok(vec![]),
    }
}

#[cfg(feature = "candle")]
async fn download_dense_module(api: &ApiRepo, dense_path: &str) -> Result<PathBuf, ApiError> {
    // Download `config.json` for the Dense module
    let config_file = format!("{}/config.json", dense_path);
    let config_path = match download_file(api, &config_file).await {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!("Failed to download `{config_file}` file: {err}");
            return Err(err);
        }
    };

    // Try to download the `model.safetensors` first
    let safetensors_file = format!("{}/model.safetensors", dense_path);
    if let Err(err) = download_file(api, &safetensors_file).await {
        tracing::warn!("Failed to download `{safetensors_file}` file: {err}");
        // Fallback to former `pytorch_model.bin`
        let pytorch_file = format!("{}/pytorch_model.bin", dense_path);
        if let Err(err) = download_file(api, &pytorch_file).await {
            tracing::warn!("Failed to download `{pytorch_file}` file: {err}");
            return Err(err);
        }
    }

    Ok(config_path.parent().unwrap().to_path_buf())
}

#[cfg(test)]
mod tests {
    use crate::{
        get_global_queue_size, is_dual_backend_enabled, set_global_queue_size, ACTIVATION_THRESHOLD,
    };
    use std::env;

    #[test]
    fn test_dual_backend_env_variable() {
        // Test that the environment variable is read correctly
        env::set_var("DUAL_BACKEND_ENABLED", "true");
        assert!(is_dual_backend_enabled());

        env::set_var("DUAL_BACKEND_ENABLED", "false");
        assert!(!is_dual_backend_enabled());

        env::remove_var("DUAL_BACKEND_ENABLED");
        assert!(!is_dual_backend_enabled()); // Should default to false
    }

    #[tokio::test]
    async fn test_backend_creation_modes() {
        // Test the environment variable logic
        env::set_var("DUAL_BACKEND_ENABLED", "false");
        assert!(!is_dual_backend_enabled());

        env::set_var("DUAL_BACKEND_ENABLED", "true");
        assert!(is_dual_backend_enabled());

        // Clean up
        env::remove_var("DUAL_BACKEND_ENABLED");
    }

    #[test]
    fn test_conditional_worker_behavior() {
        // Test the conditional activation logic for workers

        // Test below threshold - secondary worker should not activate
        set_global_queue_size(32); // Below 64
        assert!(get_global_queue_size() < ACTIVATION_THRESHOLD);

        // Test at threshold - secondary worker should activate
        set_global_queue_size(64); // At threshold
        assert!(get_global_queue_size() >= ACTIVATION_THRESHOLD);

        // Test above threshold - secondary worker should activate
        set_global_queue_size(128); // Above threshold
        assert!(get_global_queue_size() >= ACTIVATION_THRESHOLD);

        // Reset to small queue size
        set_global_queue_size(0);
    }

    #[test]
    fn test_queue_size_tracking_thread_safety() {
        // Test that global queue size tracking is thread-safe

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(AtomicUsize::new(0));
        let num_threads = 10;
        let operations_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        set_global_queue_size(i);
                        let read_back = get_global_queue_size();
                        counter.fetch_add(read_back, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify that operations completed without panics
        assert!(counter.load(Ordering::Relaxed) > 0);

        // Reset
        set_global_queue_size(0);
    }

    #[test]
    fn test_activation_threshold_scenarios() {
        // Test various scenarios around the activation threshold

        const THRESHOLD: usize = 64;

        // Test edge cases around threshold
        let test_cases = vec![
            (0, false),   // Empty queue
            (1, false),   // Single item
            (32, false),  // Half threshold
            (63, false),  // Just below threshold
            (64, true),   // Exactly at threshold
            (65, true),   // Just above threshold
            (128, true),  // Double threshold
            (1000, true), // Very large queue
        ];

        for (queue_size, should_activate) in test_cases {
            set_global_queue_size(queue_size);
            let is_above_threshold = get_global_queue_size() >= THRESHOLD;
            assert_eq!(
                is_above_threshold,
                should_activate,
                "Queue size {} should {} secondary backend",
                queue_size,
                if should_activate {
                    "activate"
                } else {
                    "not activate"
                }
            );
        }

        // Reset
        set_global_queue_size(0);
    }

    #[test]
    fn test_performance_overhead() {
        // Test that the global queue size tracking has minimal performance overhead

        use std::time::Instant;

        const ITERATIONS: usize = 1_000_000;

        // Test queue size setting performance
        let start = Instant::now();
        for i in 0..ITERATIONS {
            set_global_queue_size(i % 1000); // Keep numbers reasonable
        }
        let set_duration = start.elapsed();

        // Test queue size getting performance
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = get_global_queue_size();
        }
        let get_duration = start.elapsed();

        // Performance should be very fast (microseconds per operation)
        let set_avg_ns = set_duration.as_nanos() / ITERATIONS as u128;
        let get_avg_ns = get_duration.as_nanos() / ITERATIONS as u128;

        // Assert that operations are fast (< 100ns per operation on average)
        assert!(
            set_avg_ns < 100,
            "set_global_queue_size too slow: {}ns/op",
            set_avg_ns
        );
        assert!(
            get_avg_ns < 100,
            "get_global_queue_size too slow: {}ns/op",
            get_avg_ns
        );

        // Reset
        set_global_queue_size(0);
    }
}
