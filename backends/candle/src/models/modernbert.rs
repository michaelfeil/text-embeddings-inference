use std::collections::HashMap;

use crate::layers::{
    apply_rotary, get_cos_sin, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, LayerNormNoBias,
    Linear,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// Preserve the configuration vocabulary. ModernBERT intentionally uses the
// tanh approximation for `gelu`, as well as its explicit approximation aliases.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModernBertActivation {
    Gelu,
    #[serde(rename = "gelu_new", alias = "gelu_pytorch_tanh")]
    GeluApprox,
    Relu,
    Silu,
    Swiglu,
    Tanh,
}

impl Module for ModernBertActivation {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => input.gelu(),
            Self::GeluApprox => input.gelu(),
            Self::Relu => input.relu(),
            Self::Silu => input.silu(),
            Self::Swiglu => candle_nn::ops::swiglu(input),
            Self::Tanh => input.tanh(),
        }
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/configuration_modernbert.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_activation: ModernBertActivation,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub initializer_cutoff_factor: f64,
    pub norm_eps: f64,
    pub norm_bias: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub cls_token_id: usize,
    pub sep_token_id: usize,
    pub global_rope_theta: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    pub embedding_dropout: Option<f64>,
    pub mlp_bias: Option<bool>,
    pub mlp_dropout: Option<f64>,
    pub decoder_bias: Option<bool>,
    pub classifier_pooling: Option<Pool>,
    pub classifier_dropout: Option<f64>,
    pub classifier_bias: Option<bool>,
    pub classifier_activation: HiddenAct,
    pub deterministic_flash_attn: Option<bool>,
    pub sparse_prediction: Option<bool>,
    pub sparse_pred_ignore_index: Option<i64>,
    pub reference_compile: Option<bool>,
    pub num_labels: Option<usize>,
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug)]
pub struct ModernBertEmbeddings {
    tok_embeddings: Embedding,
    norm: LayerNormNoBias,
    span: tracing::Span,
}

impl ModernBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        Ok(Self {
            tok_embeddings: Embedding::new(
                vb.pp("tok_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            norm: LayerNormNoBias::load(vb.pp("norm"), config.hidden_size, config.norm_eps as f32)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        self.norm
            .forward(&self.tok_embeddings.forward(input_ids)?, None)
    }
}

pub struct ModernBertMLP {
    wi: Linear,
    wo: Linear,
    activation: ModernBertActivation,
    span: tracing::Span,
}

impl ModernBertMLP {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let wi_weight = vb
            .pp("Wi")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let wi_bias = vb.pp("Wi").get(config.intermediate_size * 2, "bias").ok();
        let wi = Linear::new(wi_weight, wi_bias, None);

        let wo_weight = vb
            .pp("Wo")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let wo_bias = vb.pp("Wo").get(config.hidden_size, "bias").ok();

        let wo = Linear::new(wo_weight, wo_bias, None);

        let activation = config.hidden_activation;

        Ok(Self {
            wi,
            wo,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.wi.forward(hidden_states)?;

        let gated = match self.activation {
            ModernBertActivation::Gelu | ModernBertActivation::GeluApprox => {
                // Reuse the existing approximation kernel for eligible rank-two
                // Flash Attention projections. Dense rank-three inputs retain
                // the helper's activation-plus-multiply fallback.
                crate::layers::gated_activation(&hidden_states, Some(&HiddenAct::Gelu))?
            }
            _ => {
                let chunks = hidden_states.chunk(2, D::Minus1)?;
                self.activation.forward(&chunks[0])?.mul(&chunks[1])?
            }
        };
        self.wo.forward(&gated)
    }
}

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl ModernBertAttention {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let wqkv_weight = vb
            .pp("Wqkv")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let wqkv_bias = if config.attention_bias {
            vb.pp("Wqkv").get(hidden_size * 3, "bias").ok()
        } else {
            None
        };
        let wqkv: Linear = Linear::new(wqkv_weight, wqkv_bias, None);

        let wo_weight = vb.pp("Wo").get((hidden_size, hidden_size), "weight")?;
        let wo_bias = if config.attention_bias {
            vb.pp("Wo").get(hidden_size, "bias").ok()
        } else {
            None
        };
        let wo = Linear::new(wo_weight, wo_bias, None);

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            wqkv,
            wo,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        rotary_cache: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let device = hidden_states.device();

        let qkv = self.wqkv.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?.transpose(1, 2)?;

        let qkv = qkv.chunk(3, 1)?;
        let query_layer = &qkv[0].contiguous()?;
        let key_layer = &qkv[1].contiguous()?;
        let value_layer = &qkv[2];

        let query_layer = apply_rotary(
            query_layer,
            &rotary_cache.0,
            &rotary_cache.1,
            self.attention_head_size,
        )?;
        let key_layer = apply_rotary(
            key_layer,
            &rotary_cache.0,
            &rotary_cache.1,
            self.attention_head_size,
        )?;

        #[allow(unused_variables)]
        let context_layer =
            if let (Device::Cuda(_), Some(cublaslt)) = (device, get_cublas_lt_wrapper(device)?) {
                #[cfg(feature = "cuda")]
                {
                    let (batch_size, _, seq_len, _) = key_layer.shape().dims4()?;
                    let key_layer = key_layer.flatten(0, 1)?;
                    let query_layer = query_layer.flatten(0, 1)?;
                    let value_layer = value_layer.flatten(0, 1)?;
                    let attention_mask = attention_mask.flatten(0, 1)?;

                    let attention_scores = cublaslt.batch_matmul(
                        &key_layer,
                        &query_layer,
                        None,
                        Some(self.softmax_scale as f32),
                        None,
                        None,
                        None,
                    )?;
                    // cuBLASLt's optional output is a writable buffer, and beta
                    // defaults to zero. Add the mask explicitly so padding/local
                    // windows are honored instead of copying and ignoring it.
                    let attention_scores = attention_scores.add(&attention_mask)?;
                    let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                    let context_layer = cublaslt.batch_matmul(
                        &value_layer.t()?.contiguous()?,
                        &attention_probs,
                        Some(&query_layer),
                        None,
                        None,
                        None,
                        None,
                    )?;

                    context_layer.reshape((
                        batch_size,
                        self.num_attention_heads,
                        seq_len,
                        self.attention_head_size,
                    ))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle::bail!("`cuda` feature is not enabled")
                }
            } else {
                let attn_weights = query_layer.matmul(&key_layer.t()?)?;
                let attn_weights = (attn_weights * self.softmax_scale)?;
                let attn_weights = attn_weights.add(attention_mask)?;
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&value_layer.contiguous()?)
            }?;

        let hidden_states = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;
        let hidden_states = self.wo.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

struct ModernBertEncoderLayer {
    attn_norm: Option<LayerNormNoBias>,
    attn: ModernBertAttention,
    mlp_norm: LayerNormNoBias,
    mlp: ModernBertMLP,

    span: tracing::Span,
}

impl ModernBertEncoderLayer {
    pub fn load(vb: VarBuilder, index: usize, config: &ModernBertConfig) -> Result<Self> {
        let attn_norm = if index != 0 {
            Some(LayerNormNoBias::load(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )?)
        } else {
            None
        };

        let attn = ModernBertAttention::load(vb.pp("attn"), config)?;

        let mlp_norm = LayerNormNoBias::load(
            vb.pp("mlp_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;

        let span = tracing::span!(tracing::Level::TRACE, "layer");

        Ok(ModernBertEncoderLayer {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        rotary_cache: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let attn_norm = if let Some(attn_norm) = &self.attn_norm {
            attn_norm.forward(hidden_states, None)?
        } else {
            hidden_states.clone()
        };

        let attn_outputs = self
            .attn
            .forward(&attn_norm, attention_mask, rotary_cache)?;

        let hidden_states = residual.add(&attn_outputs)?;

        let mlp_output = self
            .mlp
            .forward(&self.mlp_norm.forward(&hidden_states, None)?)?;

        let hidden_states = hidden_states.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

struct ModernBertEncoder {
    layers: Vec<ModernBertEncoderLayer>,

    global_attn_every_n_layers: usize,

    span: tracing::Span,
}

impl ModernBertEncoder {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| ModernBertEncoderLayer::load(vb.pp(format!("{index}")), index, config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(ModernBertEncoder {
            layers,
            global_attn_every_n_layers: config.global_attn_every_n_layers,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
        global_rotaray_cache: &(Tensor, Tensor),
        local_rotaray_cache: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        for (index, layer) in self.layers.iter().enumerate() {
            let use_local_attention = index % self.global_attn_every_n_layers != 0;

            let (attention_mask, rotary_cache) = if use_local_attention {
                (local_attention_mask, local_rotaray_cache)
            } else {
                (global_attention_mask, global_rotaray_cache)
            };

            hidden_states = layer.forward(&hidden_states, attention_mask, rotary_cache)?;
        }

        Ok(hidden_states)
    }
}

pub trait ClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;

    // Token classification uses this hook in the CUDA model.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

pub struct ModernBertClassificationHead {
    dense: Linear,
    norm: LayerNormNoBias,
    classifier: Linear,
    span: tracing::Span,
}

impl ModernBertClassificationHead {
    pub(crate) fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            Some(id2label) => id2label.len(),
            None => config.num_labels.unwrap_or(1),
        };

        let dense_weight = vb
            .pp("head.dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let dense = Linear::new(
            dense_weight,
            None,
            Some(config.classifier_activation.clone()),
        );

        let norm = LayerNormNoBias::load(
            vb.pp("head.norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;

        let classifier_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let classifier_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(classifier_weight, Some(classifier_bias), None);

        Ok(Self {
            dense,
            norm,
            classifier,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }
}

impl ClassificationHead for ModernBertClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = hidden_states.unsqueeze(1)?;

        let hidden_states = self.dense.forward(&hidden_states)?;
        let hidden_states = self.norm.forward(&hidden_states, None)?;
        let hidden_states = self.classifier.forward(&hidden_states)?;

        let hidden_states = hidden_states.squeeze(1)?;

        Ok(hidden_states)
    }

    fn forward_tokens(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.classifier.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct ModernBertModel {
    embeddings: ModernBertEmbeddings,
    encoder: ModernBertEncoder,
    final_norm: LayerNormNoBias,
    pool: Pool,
    classifier: Option<Box<dyn ClassificationHead + Send>>,

    local_attention: usize,
    global_inv_freqs: Tensor,
    local_inv_freqs: Tensor,
    rotary_dim: usize,
    pad_token_id: u32,
    num_attention_heads: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl ModernBertModel {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig, model_type: ModelType) -> Result<Self> {
        let (pool, classifier) = match model_type {
            ModelType::Decision => candle::bail!("Typed decisions require a Laya checkpoint"),
            ModelType::Classifier => {
                let pool: Pool = config.classifier_pooling.clone().unwrap_or(Pool::Cls);

                let classifier: Box<dyn ClassificationHead + Send> =
                    Box::new(ModernBertClassificationHead::load(vb.clone(), config)?);

                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for ModernBert")
                }

                if pool == Pool::LastToken {
                    candle::bail!("`LastToken` is not supported for ModernBert")
                }

                (pool, None)
            }
        };

        let embeddings = ModernBertEmbeddings::load(vb.pp("model.embeddings"), config)
            .or_else(|_| ModernBertEmbeddings::load(vb.pp("embeddings"), config))?;
        let encoder = ModernBertEncoder::load(vb.pp("model.layers"), config)
            .or_else(|_| ModernBertEncoder::load(vb.pp("layers"), config))?;
        let final_norm = LayerNormNoBias::load(
            vb.pp("model.final_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )
        .or_else(|_| {
            LayerNormNoBias::load(
                vb.pp("final_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )
        })?;

        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let global_inv_freqs = get_inv_freqs(
            attention_head_size,
            config.global_rope_theta as f32,
            vb.device(),
            None,
        )?;
        let local_inv_freqs = get_inv_freqs(
            attention_head_size,
            config.local_rope_theta as f32,
            vb.device(),
            None,
        )?;

        Ok(Self {
            embeddings,
            encoder,
            final_norm,
            pool,
            classifier,
            local_attention: config.local_attention,
            global_inv_freqs,
            local_inv_freqs,
            rotary_dim: attention_head_size,
            pad_token_id: config.pad_token_id as u32,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn get_global_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_shape: &(usize, usize),
    ) -> Result<Tensor> {
        let extended_attention_mask = if let Some(attention_mask) = attention_mask {
            attention_mask.squeeze(2)?
        } else {
            Tensor::ones(*input_shape, DType::F32, &self.device)?
        }
        .unsqueeze(1)?
        .unsqueeze(1)?;

        let (bs, seq_len) = *input_shape;
        let extended_attention_mask = extended_attention_mask.broadcast_as((
            bs,
            self.num_attention_heads,
            seq_len,
            seq_len,
        ))?;

        Ok(extended_attention_mask)
    }

    fn get_local_attention_mask(attention_mask: &Tensor, local_attention: usize) -> Result<Tensor> {
        // Keep the dense mask on the encoder device. Moving a broadcast
        // [batch, heads, sequence, sequence] mask through CPU scales quadratically
        // with sequence length and synchronizes every CUDA inference batch.
        let attention_mask = attention_mask.to_dtype(DType::U8)?;

        let mask_shape = attention_mask.shape();
        let (_, _, seq_len, _) = mask_shape.dims4()?;

        // CUDA has no I64 abs kernel; F32 represents supported token positions exactly.
        let rows = Tensor::arange(0f32, seq_len as f32, attention_mask.device())?.unsqueeze(0)?;
        let rows = rows.broadcast_as((seq_len, seq_len))?;

        let distance = (&rows - &rows.t()?)?.abs()?;

        let window_size = (local_attention / 2) as f64;
        let window_mask = distance
            .le(window_size)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(mask_shape)?;

        let zero_tensor = Tensor::zeros_like(&attention_mask)?;
        attention_mask.where_cond(&window_mask, &zero_tensor)
    }

    pub(crate) fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, input_lengths, position_ids, attention_mask) = if batch_size > 1 {
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
            let mut attention_mask = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);

            let mut masking = false;

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = (end - start) as u32;
                input_lengths.push(seq_length as f32);

                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                    attention_mask.push(1.0_f32);
                }

                let padding = batch.max_length - seq_length;
                if padding > 0 {
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(self.pad_token_id);
                        position_ids.push(0);
                        attention_mask.push(0.0_f32);
                    }
                }
            }

            let attention_mask = match masking {
                true => {
                    let attention_mask = Tensor::from_vec(
                        attention_mask,
                        (batch_size, max_length, 1),
                        &self.device,
                    )?
                    .to_dtype(self.dtype)?;

                    Some(attention_mask)
                }
                false => None,
            };

            (input_ids, input_lengths, position_ids, attention_mask)
        } else {
            (
                batch.input_ids,
                vec![max_length as f32],
                batch.position_ids,
                None,
            )
        };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, batch_size * max_length, &self.device)?;
        let mut input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let global_attention_mask = self
            .get_global_attention_mask(attention_mask.as_ref(), &shape)?
            .to_dtype(self.dtype)?;
        let local_attention_mask =
            Self::get_local_attention_mask(&global_attention_mask, self.local_attention)?
                .to_dtype(self.dtype)?;

        let min_value = match self.dtype {
            DType::F32 => f32::MIN as f64,
            _ => -65504.0, // f16 minimum value
        };

        let global_attention_mask = ((1.0 - global_attention_mask)? * min_value)?;
        let local_attention_mask = ((1.0 - local_attention_mask)? * min_value)?;

        let global_rotary_cache =
            get_cos_sin(max_length, &self.global_inv_freqs, self.dtype, true)?;
        let local_rotary_cache = get_cos_sin(max_length, &self.local_inv_freqs, self.dtype, true)?;

        let global_rotary_cache = (
            global_rotary_cache
                .0
                .index_select(&position_ids, 0)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?,
            global_rotary_cache
                .1
                .index_select(&position_ids, 0)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?,
        );

        let local_rotary_cache = (
            local_rotary_cache
                .0
                .index_select(&position_ids, 0)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?,
            local_rotary_cache
                .1
                .index_select(&position_ids, 0)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?,
        );

        let hidden_states = self.embeddings.forward(&input_ids)?;

        let hidden_states = self.encoder.forward(
            &hidden_states,
            &global_attention_mask,
            &local_attention_mask,
            &global_rotary_cache,
            &local_rotary_cache,
        )?;
        let outputs = self.final_norm.forward(&hidden_states, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let pooled_indices_length = batch.pooled_indices.len();
            let mut outputs = outputs.clone();

            let pooled_indices = if has_raw_requests {
                let pooled_indices =
                    Tensor::from_vec(batch.pooled_indices, pooled_indices_length, &self.device)?;

                outputs = outputs.index_select(&pooled_indices, 0)?;
                Some(pooled_indices)
            } else {
                None
            };

            let pooled_embeddings = match self.pool {
                Pool::Cls => outputs.i((.., 0))?,
                Pool::LastToken | Pool::Splade => unreachable!(),
                Pool::Mean => {
                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                            input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        };

                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
            };
            Some(pooled_embeddings)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            let (b, l, h) = outputs.shape().dims3()?;
            let outputs = outputs.reshape((b * l, h))?;

            // We need to remove the padding tokens only if batch_size > 1 and there are some
            // member of the batch that require pooling
            // or if batch_size > 1 and the members of the batch have different lengths
            if (attention_mask.is_some() || has_pooling_requests) && batch_size > 1 {
                let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);

                for i in batch.raw_indices.into_iter() {
                    let start = i * batch.max_length;
                    let i = i as usize;
                    let length =
                        batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                    for j in start..start + length {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(outputs.index_select(&final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for ModernBertModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    #[test]
    fn local_mask_preserves_window_and_padding_on_encoder_device() -> Result<()> {
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "cuda")]
        if std::env::var_os("LAYA_TEST_CUDA").is_some() {
            devices.push(Device::new_cuda(0)?);
        }
        for device in devices.drain(..) {
            let mask = Tensor::from_vec(
                vec![1f32, 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                (2, 1, 1, 5),
                &device,
            )?
            .to_dtype(DType::BF16)?
            .broadcast_as((2, 2, 5, 5))?;
            let actual = ModernBertModel::get_local_attention_mask(&mask, 2)?;
            assert_eq!(actual.device().is_cuda(), device.is_cuda());
            let actual = actual.flatten_all()?.to_vec1::<u8>()?;
            let mut expected = Vec::new();
            for length in [5usize, 3] {
                for _head in 0..2 {
                    for query in 0usize..5 {
                        for key in 0usize..5 {
                            expected.push(u8::from(key < length && query.abs_diff(key) <= 1));
                        }
                    }
                }
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_approximate_gelu(device: &Device) -> Result<()> {
        // Wi yields [x, 1], so the complete gated MLP evaluates GELU(x).
        let mlp = ModernBertMLP {
            wi: Linear::new(
                Tensor::from_slice(&[1f32, 0., 0., 1., 0., 0., 0., 0.], (4, 2), device)?,
                Some(Tensor::from_slice(&[0f32, 0., 1., 1.], 4, device)?),
                None,
            ),
            wo: Linear::new(Tensor::eye(2, DType::F32, device)?, None, None),
            activation: serde_json::from_str("\"gelu\"").unwrap(),
            span: tracing::span!(tracing::Level::TRACE, "test_mlp"),
        };
        let output = mlp
            .forward(&Tensor::from_slice(&[-2f32, 2.], (1, 1, 2), device)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), independently evaluated.
        // Exact erf-GELU differs by about 9.8e-5 at these inputs.
        for (actual, expected) in output.iter().zip([-0.045402307f32, 1.9545977]) {
            assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    fn uniform_attention(device: &Device, dtype: DType) -> Result<ModernBertAttention> {
        // Q = K = 0, V = input, Wo = identity: expected output is the
        // arithmetic mean over precisely the keys allowed by the mask.
        let mut weights = vec![0f32; 12 * 4];
        for i in 0..4 {
            weights[(8 + i) * 4 + i] = 1.;
        }
        Ok(ModernBertAttention {
            wqkv: Linear::new(
                Tensor::from_vec(weights, (12, 4), device)?.to_dtype(dtype)?,
                None,
                None,
            ),
            wo: Linear::new(Tensor::eye(4, dtype, device)?, None, None),
            num_attention_heads: 1,
            attention_head_size: 4,
            softmax_scale: 0.5,
            span: tracing::span!(tracing::Level::TRACE, "test_attention"),
        })
    }

    fn masked_attention(
        attention: &ModernBertAttention,
        device: &Device,
        dtype: DType,
        lengths: &[usize],
        local: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let width = *lengths.iter().max().unwrap();
        let mut hidden = vec![];
        let mut mask = vec![];
        let mut expected = vec![];
        for (batch, &length) in lengths.iter().enumerate() {
            let values: Vec<f32> = (0..width)
                .map(|i| {
                    if i < length {
                        ((i + batch) % 7) as f32 / 8.
                    } else {
                        100.
                    }
                })
                .collect();
            for &value in &values {
                hidden.extend([value; 4]);
            }
            for query in 0..width {
                let allowed: Vec<usize> = (0..length)
                    .filter(|&key| !local || query.abs_diff(key) <= 64)
                    .collect();
                assert!(!allowed.is_empty());
                expected.push(
                    allowed.iter().map(|&key| values[key]).sum::<f32>() / allowed.len() as f32,
                );
                mask.extend((0..width).map(|key| {
                    if allowed.contains(&key) {
                        0f32
                    } else {
                        -65504.
                    }
                }));
            }
        }
        let shape = (lengths.len(), 1, width, width);
        let mask = Tensor::from_vec(mask.clone(), shape, device)?.to_dtype(dtype)?;
        let original_mask = mask.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let hidden =
            Tensor::from_vec(hidden, (lengths.len(), width, 4), device)?.to_dtype(dtype)?;
        let rotary_shape = (lengths.len(), 1, width, 4);
        let rotary = (
            Tensor::ones(rotary_shape, dtype, device)?,
            Tensor::zeros(rotary_shape, dtype, device)?,
        );
        // cuBLASLt may use TF32 for F32 matmuls; keep a tighter CPU oracle.
        let tolerance = match (dtype, device) {
            (DType::F32, Device::Cpu) => 2e-6,
            (DType::F32, _) if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0") => 2e-6,
            // TF32's ten-bit mantissa bounds these outputs (all below 0.5).
            (DType::F32, _) => 5e-4,
            _ => 0.004,
        };
        let mut result = vec![];
        // Reuse the same mask as multiple encoder layers do. Both calls must
        // honor it, and neither may mutate the caller's mask storage.
        for _ in 0..2 {
            result = attention
                .forward(&hidden, &mask, &rotary)?
                .to_dtype(DType::F32)?
                .to_vec3::<f32>()?;
            assert_eq!(
                mask.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                original_mask
            );
            for (index, token) in result.iter().flatten().enumerate() {
                for &actual in token {
                    assert!(
                        (actual - expected[index]).abs() < tolerance,
                        "local={local}, token={index}: {actual} != {}",
                        expected[index]
                    );
                }
            }
        }
        Ok(result)
    }

    fn check_masks(device: &Device, dtype: DType) -> Result<()> {
        let attention = uniform_attention(device, dtype)?;
        for local in [false, true] {
            // Cross the 128-token local window and introduce unequal-length
            // batch padding. The first sequence must match its solo result.
            let batched = masked_attention(&attention, device, dtype, &[133, 160], local)?;
            let solo = masked_attention(&attention, device, dtype, &[133], local)?;
            // cuBLASLt may use TF32 for F32 matmuls; keep a tighter CPU oracle.
            let tolerance = match (dtype, device) {
                (DType::F32, Device::Cpu) => 2e-6,
                (DType::F32, _) if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0") => {
                    2e-6
                }
                // TF32's ten-bit mantissa bounds these outputs (all below 0.5).
                (DType::F32, _) => 5e-4,
                _ => 0.004,
            };
            for (a, b) in batched[0]
                .iter()
                .take(133)
                .flatten()
                .zip(solo[0].iter().flatten())
            {
                assert!((a - b).abs() < tolerance, "batched {a} != solo {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn modernbert_activation_config_preserves_existing_names() -> Result<()> {
        for (name, expected) in [
            ("gelu", ModernBertActivation::Gelu),
            ("gelu_new", ModernBertActivation::GeluApprox),
            ("gelu_pytorch_tanh", ModernBertActivation::GeluApprox),
            ("relu", ModernBertActivation::Relu),
            ("silu", ModernBertActivation::Silu),
            ("swiglu", ModernBertActivation::Swiglu),
            ("tanh", ModernBertActivation::Tanh),
        ] {
            let parsed: ModernBertActivation = serde_json::from_str(&format!("\"{name}\""))
                .expect("previously supported activation name");
            assert_eq!(parsed, expected);
        }
        Ok(())
    }

    #[test]
    fn modernbert_approximate_gelu_cpu() -> Result<()> {
        check_approximate_gelu(&Device::Cpu)
    }

    #[test]
    fn modernbert_padding_locality_and_mask_reuse_cpu() -> Result<()> {
        check_masks(&Device::Cpu, DType::F32)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn modernbert_approximate_gelu_and_attention_masks_cuda() -> Result<()> {
        // Deliberately fail if CUDA/cuBLASLt is unavailable: this test must not
        // silently pass through the CPU fallback it is intended to distinguish.
        let device = Device::new_cuda(0)?;
        assert!(get_cublas_lt_wrapper(&device)?.is_some());
        check_approximate_gelu(&device)?;
        check_masks(&device, DType::F32)?;
        check_masks(&device, DType::BF16)
    }
}
