mod dtype;

use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::cmp::{max, min};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use text_embeddings_backend_core::Backend as CoreBackend;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{instrument, Span};

#[cfg(feature = "candle")]
use serde::Deserialize;

pub use crate::dtype::DType;
pub use text_embeddings_backend_core::{
    AudioFeatures, BackendError, Batch, ClefField, DecisionInput, DecisionOutput, Embedding,
    Embeddings, ImagePatches, ModelType, MultimodalEncoding, Pool, Predictions, TokenPredictions,
};

#[cfg(feature = "candle")]
use text_embeddings_backend_candle::CandleBackend;

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::Sender<BackendCommand>,
    /// Health status
    health_receiver: watch::Receiver<bool>,
    _backend_thread: Arc<BackendThread>,
    pub radix_mlp_supported: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
}

impl Backend {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        model_path: PathBuf,
        api_repo: Option<ApiRepo>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_shared(
            model_path,
            api_repo.map(Arc::new),
            dtype,
            model_type,
            dense_path,
            device_id,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_shared(
        model_path: PathBuf,
        api_repo: Option<Arc<ApiRepo>>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_shared_with_fp8(
            model_path, api_repo, dtype, model_type, dense_path, device_id, false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_shared_with_fp8(
        model_path: PathBuf,
        api_repo: Option<Arc<ApiRepo>>,
        dtype: DType,
        model_type: ModelType,
        dense_path: Option<String>,
        device_id: usize,
        enable_fp8_dynamic: bool,
    ) -> Result<Self, BackendError> {
        let (backend_sender, backend_receiver) = mpsc::channel(8);

        let backend = init_backend(
            model_path,
            api_repo,
            dtype,
            model_type.clone(),
            dense_path,
            device_id,
            enable_fp8_dynamic,
        )
        .await?;
        let radix_mlp_supported = backend.supports_radix_mlp();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread =
            Arc::new(BackendThread::new(backend, backend_receiver, health_sender));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            radix_mlp_supported,
            max_batch_size,
            model_type,
        })
    }

    #[instrument(skip(self))]
    pub async fn warmup(
        &self,
        max_input_length: usize,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
    ) -> Result<(), BackendError> {
        let warmup_tokens = max_batch_tokens;

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
            multimodal: vec![],
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
            tokens: vec![],
            offsets: vec![],
        };

        match &self.model_type {
            ModelType::Decision => self
                .decide(batch.clone(), vec![DecisionInput::Warmup; batch.len()])
                .await
                .map(|_| ()),
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
                .await
                .expect("No backend receiver. This is a bug.");
            receiver.await.expect(
                "Backend blocking task dropped the sender without sending a response. This is a bug.",
            )
        } else {
            // The backend is un-healthy or only just started. Do a more advanced health check
            // by calling the model forward on a test batch

            let batch = Batch {
                multimodal: vec![],
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
                tokens: vec![],
                offsets: vec![],
            };
            match &self.model_type {
                ModelType::Decision => self
                    .decide(batch.clone(), vec![DecisionInput::Warmup; batch.len()])
                    .await
                    .map(|_| ()),
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }
        }
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .try_send(BackendCommand::Embed(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    pub async fn decide(
        &self,
        batch: Batch,
        inputs: Vec<DecisionInput>,
    ) -> Result<(Vec<DecisionOutput>, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();
        self.backend_sender
            .send(BackendCommand::Decide(
                batch,
                inputs,
                Span::current(),
                sender,
            ))
            .await
            .map_err(|_| BackendError::Unhealthy)?;
        receiver.await.map_err(|_| BackendError::Unhealthy)?
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .try_send(BackendCommand::Predict(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    #[instrument(skip_all)]
    pub async fn predict_tokens(
        &self,
        batch: Batch,
    ) -> Result<(TokenPredictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .try_send(BackendCommand::PredictTokens(
                batch,
                Span::current(),
                sender,
            ))
            .expect("No backend receiver. This is a bug.");
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
    device_id: usize,
    enable_fp8_dynamic: bool,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if enable_fp8_dynamic && !cfg!(feature = "experimental-fp8") {
        return Err(BackendError::Start(
            "Dynamic FP8 requires an experimental-fp8 build".into(),
        ));
    }
    let mut backend_start_failed = false;

    if let Some(api_repo) = api_repo.as_ref() {
        if cfg!(feature = "candle") {
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

            let path = model_path.clone();
            let candle_dtype = dtype.to_string();
            let candle_model_type = model_type.clone();
            let backend = tokio::task::spawn_blocking(move || {
                CandleBackend::new_with_fp8(
                    &path,
                    candle_dtype,
                    candle_model_type,
                    dense_paths,
                    device_id,
                    enable_fp8_dynamic,
                )
            })
            .await
            .map_err(|err| {
                BackendError::Start(format!("Candle initialization worker failed: {err}"))
            })?;
            match backend {
                Ok(b) => return Ok(Box::new(b)),
                Err(err) => {
                    if enable_fp8_dynamic {
                        return Err(err);
                    }
                    tracing::error!("Could not start Candle backend: {err}");
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
struct BackendThread(Option<JoinHandle<()>>);

impl BackendThread {
    fn new(
        backend: Box<dyn CoreBackend + Send>,
        mut backend_receiver: mpsc::Receiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            while let Some(cmd) = backend_receiver.blocking_recv() {
                let start = Instant::now();
                let mut healthy = false;
                match cmd {
                    BackendCommand::Decide(batch, inputs, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.decide(batch, inputs).map(|output| {
                            healthy = true;
                            (output, start.elapsed())
                        }));
                    }
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
                    BackendCommand::PredictTokens(batch, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.predict_tokens(batch).map(|e| {
                            healthy = true;
                            (e, start.elapsed())
                        }));
                    }
                };
                let _ = health_sender.send(healthy);
            }
        });
        Self(Some(handle))
    }
}

impl Drop for BackendThread {
    fn drop(&mut self) {
        self.0.take().unwrap().join().unwrap();
    }
}

enum BackendCommand {
    Decide(
        Batch,
        Vec<DecisionInput>,
        Span,
        oneshot::Sender<Result<(Vec<DecisionOutput>, Duration), BackendError>>,
    ),
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
    PredictTokens(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<(TokenPredictions, Duration), BackendError>>,
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

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Deserialize, PartialEq)]
enum ModuleType {
    #[serde(rename = "sentence_transformers.models.Dense")]
    Dense,
    #[serde(
        rename = "sentence_transformers.models.Normalize",
        alias = "sentence_transformers.base.modules.normalize.Normalize"
    )]
    Normalize,
    #[serde(
        rename = "sentence_transformers.models.Pooling",
        alias = "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
    )]
    Pooling,
    #[serde(
        rename = "sentence_transformers.models.Transformer",
        alias = "sentence_transformers.base.modules.transformer.Transformer"
    )]
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

pub fn supports_device_replication() -> bool {
    cfg!(all(
        feature = "candle",
        any(
            feature = "cuda",
            feature = "flash-attn",
            feature = "flash-attn-v1"
        )
    ))
}

pub fn visible_cuda_device_count() -> Result<usize, BackendError> {
    #[cfg(all(
        feature = "candle",
        any(feature = "cuda", feature = "flash-attn", feature = "flash-attn-v1")
    ))]
    {
        text_embeddings_backend_candle::visible_cuda_device_count()
    }
    #[cfg(not(all(
        feature = "candle",
        any(feature = "cuda", feature = "flash-attn", feature = "flash-attn-v1")
    )))]
    {
        Err(BackendError::Start(
            "GPU replicas require the Candle CUDA backend".into(),
        ))
    }
}
