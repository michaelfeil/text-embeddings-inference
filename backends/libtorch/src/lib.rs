//! Experimental native BERT inference. Python and exported TorchScript models are not required.
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    fs,
    path::Path,
    ptr::NonNull,
};
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
    TokenPredictions,
};

#[repr(C)]
struct NativeConfig {
    hidden: i64,
    heads: i64,
    layers: i64,
    intermediate: i64,
    vocab: i64,
    positions: i64,
    types: i64,
    epsilon: f64,
    activation: i32,
}
extern "C" {
    fn tei_error() -> *const c_char;
    fn tei_device_count(device: *const c_char) -> i64;
    fn tei_create(config: *const NativeConfig, device: *const c_char, dtype: i32) -> *mut c_void;
    fn tei_weight(
        handle: *mut c_void,
        name: *const c_char,
        data: *const u8,
        shape: *const i64,
        rank: usize,
        dtype: i32,
    ) -> i32;
    fn tei_ready(handle: *mut c_void) -> i32;
    fn tei_forward(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        pool: i32,
        pooled: *const i64,
        pooled_count: usize,
        raw: *const i64,
        raw_count: usize,
        output: *mut f32,
    ) -> i32;
    fn tei_destroy(handle: *mut c_void);
}

fn native_error() -> String {
    // SAFETY: the bridge owns a thread-local, NUL-terminated string valid until the next call.
    unsafe { CStr::from_ptr(tei_error()).to_string_lossy().into_owned() }
}
fn start(error: impl ToString) -> BackendError {
    BackendError::Start(error.to_string())
}
fn inference(error: impl ToString) -> BackendError {
    BackendError::Inference(error.to_string())
}

pub fn device_count(device: &str) -> Result<usize, BackendError> {
    let device = CString::new(device).map_err(start)?;
    // SAFETY: the C string is valid for the duration of the synchronous call.
    let count = unsafe { tei_device_count(device.as_ptr()) };
    usize::try_from(count).map_err(|_| start(native_error()))
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    hidden_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    intermediate_size: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    type_vocab_size: i64,
    layer_norm_eps: f64,
    hidden_act: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    position_embedding_type: Option<String>,
}

pub struct LibtorchBackend {
    handle: NonNull<c_void>,
    config: Config,
    pool: Pool,
}
// SAFETY: the handle owns its tensors and is used by TEI's single backend worker.
// It can move between threads; it is deliberately not Sync. InferenceMode is set per native call.
unsafe impl Send for LibtorchBackend {}

impl Drop for LibtorchBackend {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and destroyed exactly once.
        unsafe { tei_destroy(self.handle.as_ptr()) };
    }
}

impl LibtorchBackend {
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
    ) -> Result<Self, BackendError> {
        if device != "cpu" && device != "auto" && !device.starts_with("cuda:") {
            return Err(start("LibTorch 2.14.1 varlen attention requires CUDA; only CPU reference and CUDA are supported"));
        }
        if device.starts_with("cuda:") && !matches!(dtype, "float16" | "bfloat16") {
            return Err(start("CUDA varlen attention requires float16 or bfloat16"));
        }
        let pool =
            match model_type {
                ModelType::Embedding(pool @ (Pool::Cls | Pool::Mean | Pool::LastToken)) => pool,
                _ => return Err(start(
                    "LibTorch currently supports BERT embeddings with cls/mean/last-token pooling",
                )),
            };
        let config: Config =
            serde_json::from_slice(&fs::read(path.join("config.json")).map_err(start)?)
                .map_err(start)?;
        if config.model_type != "bert"
            || config.is_decoder
            || config.add_cross_attention
            || config
                .position_embedding_type
                .as_deref()
                .is_some_and(|p| p != "absolute")
        {
            return Err(start(
                "LibTorch currently supports encoder BERT with absolute position embeddings",
            ));
        }
        if [
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.vocab_size,
            config.max_position_embeddings,
            config.type_vocab_size,
        ]
        .iter()
        .any(|&v| v <= 0)
            || config.hidden_size % config.num_attention_heads != 0
            || !config.layer_norm_eps.is_finite()
            || config.layer_norm_eps <= 0.0
        {
            return Err(start("Invalid BERT dimensions or layer_norm_eps"));
        }
        if path.join("modules.json").exists() {
            let modules: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("modules.json")).map_err(start)?)
                    .map_err(start)?;
            for module in modules
                .as_array()
                .ok_or_else(|| start("modules.json must contain an array"))?
            {
                let kind = module["type"]
                    .as_str()
                    .ok_or_else(|| start("Missing module type"))?;
                if !matches!(
                    kind,
                    "sentence_transformers.models.Transformer"
                        | "sentence_transformers.models.Pooling"
                        | "sentence_transformers.models.Normalize"
                ) {
                    return Err(start(format!(
                        "LibTorch does not yet support Sentence Transformers module {kind}"
                    )));
                }
            }
        }
        let activation = match config.hidden_act.as_str() {
            "gelu" => 0,
            "gelu_new" | "gelu_pytorch_tanh" => 1,
            "relu" => 2,
            other => return Err(start(format!("Unsupported BERT activation {other}"))),
        };
        let native = NativeConfig {
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            layers: config.num_hidden_layers,
            intermediate: config.intermediate_size,
            vocab: config.vocab_size,
            positions: config.max_position_embeddings,
            types: config.type_vocab_size,
            epsilon: config.layer_norm_eps,
            activation,
        };
        let dtype = match dtype {
            "float32" | "auto" => 0,
            "float16" => 1,
            "bfloat16" => 2,
            _ => return Err(start("Unsupported LibTorch dtype")),
        };
        let device = CString::new(device).map_err(start)?;
        // SAFETY: config and device stay valid during creation, and the bridge copies them.
        let handle = NonNull::new(unsafe { tei_create(&native, device.as_ptr(), dtype) })
            .ok_or_else(|| start(native_error()))?;
        let model = Self {
            handle,
            config,
            pool,
        };
        let index = path.join("model.safetensors.index.json");
        let files = if index.exists() {
            let index: serde_json::Value =
                serde_json::from_slice(&fs::read(index).map_err(start)?).map_err(start)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| start("Invalid safetensors index"))?;
            let mut files = std::collections::BTreeSet::new();
            for file in map.values() {
                let file = file
                    .as_str()
                    .ok_or_else(|| start("Invalid shard filename"))?;
                // Restrict shard references to files inside the model directory.
                if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                    return Err(start("Invalid shard filename"));
                }
                files.insert(file.to_owned());
            }
            files.into_iter().collect::<Vec<_>>()
        } else {
            vec!["model.safetensors".to_owned()]
        };
        for file in files {
            let data = fs::read(path.join(file)).map_err(start)?;
            let tensors = SafeTensors::deserialize(&data).map_err(start)?;
            for (name, tensor) in tensors.tensors() {
                // Extra pooler/MLM tensors are not needed for TEI embeddings.
                let key = name.strip_prefix("bert.").unwrap_or(&name);
                if !key.starts_with("embeddings.") && !key.starts_with("encoder.layer.") {
                    continue;
                }
                if key.ends_with("position_ids") || key.ends_with("token_type_ids") {
                    continue;
                }
                let dtype = match tensor.dtype() {
                    Dtype::F32 => 0,
                    Dtype::F16 => 1,
                    Dtype::BF16 => 2,
                    other => return Err(start(format!("Unsupported weight dtype {other:?}"))),
                };
                let shape = tensor
                    .shape()
                    .iter()
                    .map(|&n| i64::try_from(n).map_err(start))
                    .collect::<Result<Vec<_>, _>>()?;
                let name = CString::new(name).map_err(start)?;
                // SAFETY: safetensors validates byte lengths. Native code copies the data before returning.
                if unsafe {
                    tei_weight(
                        handle.as_ptr(),
                        name.as_ptr(),
                        tensor.data().as_ptr(),
                        shape.as_ptr(),
                        shape.len(),
                        dtype,
                    )
                } != 0
                {
                    return Err(start(native_error()));
                }
            }
        }
        // SAFETY: the live handle is owned by model; this checks every required weight and its shape.
        if unsafe { tei_ready(handle.as_ptr()) } != 0 {
            return Err(start(native_error()));
        }
        Ok(model)
    }
}

impl Backend for LibtorchBackend {
    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let ends = &batch.cumulative_seq_lengths;
        if ends.len() < 2
            || ends[0] != 0
            || ends.last().copied() != Some(batch.input_ids.len() as u32)
            || ends.windows(2).any(|w| w[0] >= w[1])
            || batch.token_type_ids.len() != batch.input_ids.len()
            || batch.position_ids.len() != batch.input_ids.len()
        {
            return Err(inference("Invalid packed batch"));
        }
        if batch.multimodal.iter().any(Option::is_some)
            || batch.compact_input_ids.is_some()
            || batch.compact_position_ids.is_some()
            || batch.scatter_unfold.is_some()
            || batch.fold_gather.is_some()
        {
            return Err(inference(
                "LibTorch supports text batches without radix folding",
            ));
        }
        let b = ends.len() - 1;
        let lengths: Vec<i64> = ends.windows(2).map(|w| (w[1] - w[0]) as i64).collect();
        let seq = *lengths.iter().max().unwrap() as usize;
        // Preserve TEI's packed layout: allocations scale with actual tokens, never B * max_length.
        let ids: Vec<i64> = batch.input_ids.iter().map(|&id| id as i64).collect();
        let types: Vec<i64> = batch.token_type_ids.iter().map(|&id| id as i64).collect();
        let positions: Vec<i64> = batch.position_ids.iter().map(|&id| id as i64).collect();
        let cumulative = ends
            .iter()
            .map(|&end| i32::try_from(end).map_err(inference))
            .collect::<Result<Vec<_>, _>>()?;
        if ids.iter().any(|&id| id >= self.config.vocab_size)
            || types.iter().any(|&id| id >= self.config.type_vocab_size)
            || positions
                .iter()
                .any(|&id| id >= self.config.max_position_embeddings)
        {
            return Err(inference(
                "Token/type/position ID exceeds BERT embedding table",
            ));
        }
        let pooled: Vec<i64> = batch.pooled_indices.iter().map(|&n| n as i64).collect();
        let raw: Vec<i64> = batch.raw_indices.iter().map(|&n| n as i64).collect();
        if pooled.iter().chain(&raw).any(|&n| n as usize >= b) {
            return Err(inference("Invalid output index"));
        }
        if pooled
            .iter()
            .chain(&raw)
            .collect::<std::collections::HashSet<_>>()
            .len()
            != pooled.len() + raw.len()
        {
            return Err(inference("Duplicate output indices"));
        }
        let hidden = self.config.hidden_size as usize;
        let rows = pooled.len()
            + raw
                .iter()
                .map(|&i| lengths[i as usize] as usize)
                .sum::<usize>();
        let mut output = vec![
            0f32;
            rows.checked_mul(hidden)
                .ok_or_else(|| inference("Output size overflow"))?
        ];
        let pool = match self.pool {
            Pool::Cls => 0,
            Pool::Mean => 1,
            Pool::LastToken => 2,
            _ => unreachable!(),
        };
        // SAFETY: validated dimensions and indices bound every native read/write. All buffers stay
        // live until this synchronous call returns; exceptions cannot cross the C ABI.
        if unsafe {
            tei_forward(
                self.handle.as_ptr(),
                ids.as_ptr(),
                types.as_ptr(),
                positions.as_ptr(),
                cumulative.as_ptr(),
                b as i64,
                seq as i64,
                pool,
                pooled.as_ptr(),
                pooled.len(),
                raw.as_ptr(),
                raw.len(),
                output.as_mut_ptr(),
            )
        } != 0
        {
            return Err(inference(native_error()));
        }
        let mut result = Embeddings::default();
        let mut offset = 0;
        for index in pooled {
            result.insert(
                index as usize,
                Embedding::Pooled(output[offset..offset + hidden].to_vec()),
            );
            offset += hidden;
        }
        for index in raw {
            let size = lengths[index as usize] as usize * hidden;
            let values = output[offset..offset + size]
                .chunks_exact(hidden)
                .map(<[f32]>::to_vec)
                .collect();
            result.insert(index as usize, Embedding::All(values));
            offset += size;
        }
        Ok(result)
    }
    fn predict(&self, _: Batch) -> Result<Predictions, BackendError> {
        Err(inference("LibTorch classifiers are not implemented"))
    }
    fn predict_tokens(&self, _: Batch) -> Result<TokenPredictions, BackendError> {
        Err(inference("LibTorch token classifiers are not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(feature = "benchmark-cuda"))]
    use text_embeddings_backend_candle::CandleBackend;

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(prefix: &str, sharded: bool) -> Fixture {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tei-libtorch-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let config = serde_json::json!({"model_type":"bert", "hidden_size":16, "num_attention_heads":2,
            "num_hidden_layers":2, "intermediate_size":24, "vocab_size":16, "max_position_embeddings":16,
            "type_vocab_size":2, "layer_norm_eps":1e-5, "hidden_act":"gelu_pytorch_tanh", "pad_token_id":0,
            "hidden_dropout_prob":0.0, "attention_probs_dropout_prob":0.0, "initializer_range":0.02});
        fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut weights = Vec::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let salt = weights.len();
            let values = (0..shape.iter().product())
                .map(|i| {
                    let value = (((i * 17 + salt * 13) % 101) as f32 - 50.) / 200.;
                    if name.ends_with("LayerNorm.weight") {
                        1.0 + value
                    } else {
                        value
                    }
                })
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            weights.push((format!("{prefix}{name}"), shape, values));
        };
        add("embeddings.word_embeddings.weight".into(), vec![16, 16]);
        add("embeddings.position_embeddings.weight".into(), vec![16, 16]);
        add(
            "embeddings.token_type_embeddings.weight".into(),
            vec![2, 16],
        );
        add("embeddings.LayerNorm.weight".into(), vec![16]);
        add("embeddings.LayerNorm.bias".into(), vec![16]);
        for i in 0..2 {
            let p = format!("encoder.layer.{i}.");
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                add(format!("{p}{name}.weight"), vec![16, 16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
                add(format!("{p}{name}.weight"), vec![16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            add(format!("{p}intermediate.dense.weight"), vec![24, 16]);
            add(format!("{p}intermediate.dense.bias"), vec![24]);
            add(format!("{p}output.dense.weight"), vec![16, 24]);
            add(format!("{p}output.dense.bias"), vec![16]);
        }
        let views = weights
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        if sharded {
            let mut map = serde_json::Map::new();
            for (i, slice) in views.chunks(views.len().div_ceil(2)).enumerate() {
                let file = format!("model-{i}.safetensors");
                fs::write(
                    path.join(&file),
                    serialize(slice.iter().map(|(n, t)| (n.clone(), t.clone())), None).unwrap(),
                )
                .unwrap();
                for (name, _) in slice {
                    map.insert(name.clone(), serde_json::json!(file));
                }
            }
            fs::write(
                path.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
        } else {
            fs::write(
                path.join("model.safetensors"),
                serialize(views, None).unwrap(),
            )
            .unwrap();
        }
        Fixture(path)
    }
    fn batch() -> Batch {
        Batch {
            input_ids: vec![1, 5, 7, 2, 4],
            token_type_ids: vec![0, 0, 1, 1, 1],
            position_ids: vec![0, 1, 2, 0, 1],
            cumulative_seq_lengths: vec![0, 3, 5],
            max_length: 3,
            pooled_indices: vec![1],
            raw_indices: vec![0],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    fn values(value: &Embedding) -> Vec<f32> {
        match value {
            Embedding::Pooled(v) => v.clone(),
            Embedding::All(v) => v.iter().flatten().copied().collect(),
        }
    }
    // Candle selects CUDA when compiled with it; CPU parity runs in the default feature build.
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn native_bert_matches_candle_for_ragged_pooled_and_raw_outputs() {
        for (prefix, sharded) in [("", false), ("bert.", true)] {
            let fixture = fixture(prefix, sharded);
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
                let model_type = ModelType::Embedding(pool);
                let torch =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), model_type, None, 0).unwrap();
                let actual = torch.embed(batch()).unwrap();
                let expected = candle.embed(batch()).unwrap();
                for index in [0, 1] {
                    let actual = values(&actual[&index]);
                    let expected = values(&expected[&index]);
                    assert_eq!(actual.len(), expected.len());
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!((a - e).abs() < 2e-4, "index {index}: {a} != {e}");
                    }
                }
                // Packed attention must preserve sequence boundaries.
                let mut short = batch();
                short.input_ids = vec![2, 4];
                short.token_type_ids = vec![1, 1];
                short.position_ids = vec![0, 1];
                short.cumulative_seq_lengths = vec![0, 2];
                short.pooled_indices = vec![0];
                short.raw_indices.clear();
                short.max_length = 2;
                let single = torch.embed(short).unwrap();
                for (a, e) in values(&single[&0]).iter().zip(values(&actual[&1])) {
                    assert!((a - e).abs() < 2e-5);
                }
            }
        }
    }
    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_varlen_matches_cpu_for_ragged_outputs() {
        let fixture = fixture("", false);
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            let model_type = ModelType::Embedding(pool);
            let cpu =
                LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
            let gpu = LibtorchBackend::new(&fixture.0, "float16", model_type, "cuda:0").unwrap();
            let expected = cpu.embed(batch()).unwrap();
            let actual = std::thread::spawn(move || gpu.embed(batch()).unwrap())
                .join()
                .unwrap();
            for index in [0, 1] {
                for (a, e) in values(&actual[&index])
                    .iter()
                    .zip(values(&expected[&index]))
                {
                    assert!((a - e).abs() < 1e-2, "CUDA varlen: {a} != {e}");
                }
            }
        }
    }

    #[test]
    fn malformed_batches_and_native_weight_errors_are_recoverable() {
        let fixture = fixture("", false);
        let torch = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        let mut bad = batch();
        bad.cumulative_seq_lengths = vec![];
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.input_ids[0] = 16;
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.pooled_indices = vec![2];
        assert!(torch.embed(bad).is_err());
        assert!(torch.embed(batch()).is_ok());
        fs::write(
            fixture.0.join("model.safetensors"),
            serialize(Vec::<(String, TensorView<'_>)>::new(), None).unwrap(),
        )
        .unwrap();
        let err = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .err()
        .unwrap();
        assert!(matches!(err, BackendError::Start(_)));
    }
}
