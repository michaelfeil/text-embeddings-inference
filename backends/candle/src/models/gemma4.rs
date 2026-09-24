use crate::layers::{apply_rotary, get_cos_sin, get_inv_freqs, HiddenAct, Linear};
#[cfg(feature = "flash-attn")]
use crate::layers::{index_select, CompactUnfoldTensors};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

fn default_head_dim() -> usize {
    256
}

fn default_global_head_dim() -> usize {
    512
}

fn default_vocab_size() -> usize {
    262_144
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4RopeParameters {
    pub full_attention: Gemma4RopeLayerParameters,
    pub sliding_attention: Gemma4RopeLayerParameters,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4RopeLayerParameters {
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    pub pad_token_id: u32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub rope_parameters: Gemma4RopeParameters,
    pub sliding_window: usize,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(default)]
    pub id2label: HashMap<String, String>,
}

#[derive(Debug)]
struct Gemma4RmsNorm {
    weight: Option<Tensor>,
    epsilon: f64,
}

impl Gemma4RmsNorm {
    fn load(vb: VarBuilder, hidden_size: usize, epsilon: f64) -> Result<Self> {
        Ok(Self {
            weight: Some(vb.get(hidden_size, "weight")?),
            epsilon,
        })
    }

    fn without_weight(epsilon: f64) -> Self {
        Self {
            weight: None,
            epsilon,
        }
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let states = hidden_states.to_dtype(DType::F32)?;
        let variance = states.sqr()?.mean_keepdim(D::Minus1)?;
        let states = states.broadcast_div(&(variance + self.epsilon)?.sqrt()?)?;
        let states = match &self.weight {
            Some(weight) => states.broadcast_mul(&weight.to_dtype(DType::F32)?)?,
            None => states,
        };
        states.to_dtype(dtype)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionType {
    Full,
    Sliding,
}

impl AttentionType {
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "full_attention" => Ok(Self::Full),
            "sliding_attention" => Ok(Self::Sliding),
            other => candle::bail!("unsupported Gemma4 attention type `{other}`"),
        }
    }
}

struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: Gemma4RmsNorm,
    k_norm: Option<Gemma4RmsNorm>,
    v_norm: Option<Gemma4RmsNorm>,
    attention_type: AttentionType,
    is_kv_shared: bool,
    store_shared_kv: bool,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    sliding_window: usize,
}

impl Gemma4Attention {
    #[cfg(feature = "flash-attn")]
    #[allow(clippy::too_many_arguments)]
    fn forward_varlen(
        &self,
        states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
        max_length: usize,
        causal: bool,
        compact: &CompactUnfoldTensors,
        shared_kv: &mut SharedKv,
    ) -> Result<Tensor> {
        use crate::flash_attn::flash_attn_varlen;
        let compact_len = states.dim(0)?;
        let unfold_heads = |x: Tensor, heads: usize| -> Result<Tensor> {
            // The CUDA gather requires strict row-major strides. A singleton
            // head dimension may retain an arbitrary stride even after
            // `contiguous`, so gather flattened rows and restore heads.
            let flat = x.flatten_from(1)?.contiguous()?;
            let expanded = compact.scatter_unfold(&flat)?;
            expanded.reshape((expanded.dim(0)?, heads, self.head_dim))
        };
        let q = self.q_proj.forward(states)?.reshape((
            1,
            compact_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let q = apply_rotary(&q, cos, sin, self.head_dim)?
            .transpose(1, 2)?
            .squeeze(0)?
            .contiguous()?;
        let q = unfold_heads(q, self.num_attention_heads)?;

        let (k, v) = if self.is_kv_shared {
            shared_kv
                .get(self.attention_type)
                .cloned()
                .ok_or_else(|| candle::Error::Msg("missing Gemma4 shared KV states".into()))?
        } else {
            let k_unrotated = self.k_proj.as_ref().unwrap().forward(states)?.reshape((
                1,
                compact_len,
                self.num_key_value_heads,
                self.head_dim,
            ))?;
            let v_unrotated = match &self.v_proj {
                Some(proj) => proj.forward(states)?.reshape((
                    1,
                    compact_len,
                    self.num_key_value_heads,
                    self.head_dim,
                ))?,
                None => k_unrotated.clone(),
            };
            let k = self
                .k_norm
                .as_ref()
                .unwrap()
                .forward(&k_unrotated)?
                .transpose(1, 2)?;
            let k = apply_rotary(&k, cos, sin, self.head_dim)?
                .transpose(1, 2)?
                .squeeze(0)?
                .contiguous()?;
            let v = self
                .v_norm
                .as_ref()
                .unwrap()
                .forward(&v_unrotated)?
                .squeeze(0)?
                .contiguous()?;
            let k = unfold_heads(k, self.num_key_value_heads)?;
            let v = unfold_heads(v, self.num_key_value_heads)?;
            if self.store_shared_kv {
                shared_kv.set(self.attention_type, k.clone(), v.clone());
            }
            (k, v)
        };
        let (window_left, window_right) = if self.attention_type == AttentionType::Sliding {
            if causal {
                (Some(self.sliding_window.saturating_sub(1)), Some(0))
            } else {
                let half = self.sliding_window / 2;
                (Some(half), Some(half))
            }
        } else {
            (None, None)
        };
        let output = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens,
            cu_seqlens,
            max_length,
            max_length,
            1.0,
            causal,
            window_left,
            window_right,
        )?
        .flatten_from(D::Minus2)?;
        self.o_proj.forward(&compact.fold_gather(&output)?)
    }

    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        let attention_type = AttentionType::from_name(&config.layer_types[layer_idx])?;
        let is_sliding = attention_type == AttentionType::Sliding;
        let head_dim = if is_sliding {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let use_alternative_attention = config.attention_k_eq_v && !is_sliding;
        let num_key_value_heads = if use_alternative_attention {
            config
                .num_global_key_value_heads
                .unwrap_or(config.num_key_value_heads)
        } else {
            config.num_key_value_heads
        };
        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let is_kv_shared = config.num_kv_shared_layers > 0 && layer_idx >= first_shared;
        let store_shared_kv = !is_kv_shared
            && config.num_kv_shared_layers > 0
            && config.layer_types[..first_shared]
                .iter()
                .rposition(|kind| kind == &config.layer_types[layer_idx])
                == Some(layer_idx);

        let load_linear = |vb: VarBuilder, output: usize, input: usize| -> Result<Linear> {
            let weight = vb.get((output, input), "weight")?;
            let bias = if config.attention_bias {
                Some(vb.get(output, "bias")?)
            } else {
                None
            };
            Ok(Linear::new(weight, bias, None))
        };

        let q_proj = load_linear(
            vb.pp("q_proj"),
            config.num_attention_heads * head_dim,
            config.hidden_size,
        )?;
        let (k_proj, v_proj, k_norm, v_norm) = if is_kv_shared {
            (None, None, None, None)
        } else {
            let k_proj = load_linear(
                vb.pp("k_proj"),
                num_key_value_heads * head_dim,
                config.hidden_size,
            )?;
            let v_proj = if use_alternative_attention {
                None
            } else {
                Some(load_linear(
                    vb.pp("v_proj"),
                    num_key_value_heads * head_dim,
                    config.hidden_size,
                )?)
            };
            (
                Some(k_proj),
                v_proj,
                Some(Gemma4RmsNorm::load(
                    vb.pp("k_norm"),
                    head_dim,
                    config.rms_norm_eps,
                )?),
                Some(Gemma4RmsNorm::without_weight(config.rms_norm_eps)),
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj: load_linear(
                vb.pp("o_proj"),
                config.hidden_size,
                config.num_attention_heads * head_dim,
            )?,
            q_norm: Gemma4RmsNorm::load(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?,
            k_norm,
            v_norm,
            attention_type,
            is_kv_shared,
            store_shared_kv,
            head_dim,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads,
            sliding_window: config.sliding_window,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden_states: &Tensor,
        padding_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        causal: bool,
        shared_kv: &mut SharedKv,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let q = apply_rotary(&q, cos, sin, self.head_dim)?;

        let (k, v) = if self.is_kv_shared {
            shared_kv
                .get(self.attention_type)
                .cloned()
                .ok_or_else(|| candle::Error::Msg("missing Gemma4 shared KV states".into()))?
        } else {
            let k_unrotated = self
                .k_proj
                .as_ref()
                .unwrap()
                .forward(hidden_states)?
                .reshape((batch_size, seq_len, self.num_key_value_heads, self.head_dim))?;
            let v_unrotated = match &self.v_proj {
                Some(v_proj) => v_proj.forward(hidden_states)?.reshape((
                    batch_size,
                    seq_len,
                    self.num_key_value_heads,
                    self.head_dim,
                ))?,
                None => k_unrotated.clone(),
            };
            let k = self
                .k_norm
                .as_ref()
                .unwrap()
                .forward(&k_unrotated)?
                .transpose(1, 2)?;
            let k = apply_rotary(&k, cos, sin, self.head_dim)?;
            let v = self
                .v_norm
                .as_ref()
                .unwrap()
                .forward(&v_unrotated)?
                .transpose(1, 2)?;
            if self.store_shared_kv {
                shared_kv.set(self.attention_type, k.clone(), v.clone());
            }
            (k, v)
        };

        let repeat = self.num_attention_heads / self.num_key_value_heads;
        let repeat_kv = |x: Tensor| -> Result<Tensor> {
            if repeat == 1 {
                Ok(x)
            } else {
                let (b, h, s, d) = x.dims4()?;
                x.unsqueeze(2)?
                    .expand((b, h, repeat, s, d))?
                    .reshape((b, h * repeat, s, d))
            }
        };
        let k = repeat_kv(k)?.contiguous()?;
        let v = repeat_kv(v)?.contiguous()?;
        let mut weights = q.matmul(&k.t()?)?;

        let mask = self.attention_mask(
            batch_size,
            seq_len,
            weights.dtype(),
            weights.device(),
            causal,
        )?;
        weights = weights.broadcast_add(&mask)?;
        if let Some(padding_bias) = padding_bias {
            weights = weights.broadcast_add(padding_bias)?;
        }
        // Gemma4 explicitly computes attention softmax in fp32 before casting back.
        let weights = candle_nn::ops::softmax_last_dim(&weights.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        let states = weights.matmul(&v)?;
        self.o_proj
            .forward(&states.transpose(1, 2)?.flatten_from(D::Minus2)?)
    }

    fn attention_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        dtype: DType,
        device: &Device,
        causal: bool,
    ) -> Result<Tensor> {
        let min = if dtype == DType::F32 {
            f32::MIN
        } else {
            -65_504.0
        };
        let sliding = self.attention_type == AttentionType::Sliding;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    let visible = if causal {
                        j <= i && (!sliding || i - j < self.sliding_window)
                    } else if sliding {
                        i.abs_diff(j) <= self.sliding_window / 2
                    } else {
                        true
                    };
                    if visible {
                        0.0
                    } else {
                        min
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?
            .to_dtype(dtype)?
            .expand((batch_size, self.num_attention_heads, seq_len, seq_len))
    }
}

#[derive(Default)]
struct SharedKv {
    full: Option<(Tensor, Tensor)>,
    sliding: Option<(Tensor, Tensor)>,
}

impl SharedKv {
    fn get(&self, kind: AttentionType) -> Option<&(Tensor, Tensor)> {
        match kind {
            AttentionType::Full => self.full.as_ref(),
            AttentionType::Sliding => self.sliding.as_ref(),
        }
    }

    fn set(&mut self, kind: AttentionType, k: Tensor, v: Tensor) {
        match kind {
            AttentionType::Full => self.full = Some((k, v)),
            AttentionType::Sliding => self.sliding = Some((k, v)),
        }
    }
}

struct Gemma4Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    activation: HiddenAct,
    intermediate_size: usize,
}

impl Gemma4Mlp {
    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        let first_shared = config.num_hidden_layers - config.num_kv_shared_layers;
        let double_wide = config.use_double_wide_mlp
            && config.num_kv_shared_layers > 0
            && layer_idx >= first_shared;
        let intermediate_size = config.intermediate_size * if double_wide { 2 } else { 1 };
        let gate = vb
            .pp("gate_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;
        let up = vb
            .pp("up_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;
        let down = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        Ok(Self {
            gate_up_proj: Linear::new(Tensor::cat(&[&gate, &up], 0)?, None, None),
            down_proj: Linear::new(down, None, None),
            activation: config.hidden_activation.clone(),
            intermediate_size,
        })
    }

    fn forward(&self, states: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(states)?;
        let gate =
            self.activation
                .forward(&gate_up.narrow(D::Minus1, 0, self.intermediate_size)?)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct Gemma4PleLayer {
    input_gate: Linear,
    projection: Linear,
    norm: Gemma4RmsNorm,
    activation: HiddenAct,
}

struct Gemma4Layer {
    attention: Gemma4Attention,
    mlp: Gemma4Mlp,
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
    ple: Option<Gemma4PleLayer>,
    layer_scalar: Tensor,
}

impl Gemma4Layer {
    #[cfg(feature = "flash-attn")]
    #[allow(clippy::too_many_arguments)]
    fn forward_varlen(
        &self,
        states: &Tensor,
        per_layer_input: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
        max_length: usize,
        causal: bool,
        compact: &CompactUnfoldTensors,
        shared_kv: &mut SharedKv,
    ) -> Result<Tensor> {
        let residual = states;
        let normalized = self.input_layernorm.forward(states)?;
        let attention = self.attention.forward_varlen(
            &normalized,
            cos,
            sin,
            cu_seqlens,
            max_length,
            causal,
            compact,
            shared_kv,
        )?;
        let states = (residual + self.post_attention_layernorm.forward(&attention)?)?;
        let residual = &states;
        let normalized = self.pre_feedforward_layernorm.forward(&states)?;
        let mlp = self.mlp.forward(&normalized)?;
        let mut states = (residual + self.post_feedforward_layernorm.forward(&mlp)?)?;
        if let (Some(ple), Some(per_layer_input)) = (&self.ple, per_layer_input) {
            let contribution = ple.input_gate.forward(&states)?;
            let contribution = ple.activation.forward(&contribution)?;
            let contribution = (contribution * per_layer_input)?;
            let contribution = ple.projection.forward(&contribution)?;
            let contribution = ple.norm.forward(&contribution)?;
            states = (states + contribution)?;
        }
        states.broadcast_mul(&self.layer_scalar)
    }

    fn load(vb: VarBuilder, config: &Gemma4TextConfig, layer_idx: usize) -> Result<Self> {
        let norm =
            |name: &str| Gemma4RmsNorm::load(vb.pp(name), config.hidden_size, config.rms_norm_eps);
        let ple = if config.hidden_size_per_layer_input > 0 {
            Some(Gemma4PleLayer {
                input_gate: Linear::new(
                    vb.pp("per_layer_input_gate").get(
                        (config.hidden_size_per_layer_input, config.hidden_size),
                        "weight",
                    )?,
                    None,
                    None,
                ),
                projection: Linear::new(
                    vb.pp("per_layer_projection").get(
                        (config.hidden_size, config.hidden_size_per_layer_input),
                        "weight",
                    )?,
                    None,
                    None,
                ),
                norm: norm("post_per_layer_input_norm")?,
                activation: config.hidden_activation.clone(),
            })
        } else {
            None
        };
        Ok(Self {
            attention: Gemma4Attention::load(vb.pp("self_attn"), config, layer_idx)?,
            mlp: Gemma4Mlp::load(vb.pp("mlp"), config, layer_idx)?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            ple,
            layer_scalar: vb.get(1, "layer_scalar")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        states: &Tensor,
        per_layer_input: Option<&Tensor>,
        padding_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        causal: bool,
        shared_kv: &mut SharedKv,
    ) -> Result<Tensor> {
        let residual = states;
        let states = self.input_layernorm.forward(states)?;
        let states = self
            .attention
            .forward(&states, padding_bias, cos, sin, causal, shared_kv)?;
        let states = self.post_attention_layernorm.forward(&states)?;
        let states = (residual + states)?;

        let residual = &states;
        let states = self.pre_feedforward_layernorm.forward(&states)?;
        let states = self.mlp.forward(&states)?;
        let states = self.post_feedforward_layernorm.forward(&states)?;
        let mut states = (residual + states)?;

        if let (Some(ple), Some(per_layer_input)) = (&self.ple, per_layer_input) {
            let contribution = ple.input_gate.forward(&states)?;
            let contribution = ple.activation.forward(&contribution)?;
            let contribution = (contribution * per_layer_input)?;
            let contribution = ple.projection.forward(&contribution)?;
            let contribution = ple.norm.forward(&contribution)?;
            states = (states + contribution)?;
        }
        states.broadcast_mul(&self.layer_scalar)
    }
}

struct Gemma4Ple {
    embeddings: Embedding,
    model_projection: Linear,
    projection_norm: Gemma4RmsNorm,
    num_layers: usize,
    hidden_size: usize,
    scale: f64,
    projection_scale: f64,
}

impl Gemma4Ple {
    fn load(vb: VarBuilder, config: &Gemma4TextConfig) -> Result<Option<Self>> {
        let hidden_size = config.hidden_size_per_layer_input;
        if hidden_size == 0 {
            return Ok(None);
        }
        let packed_size = config.num_hidden_layers * hidden_size;
        Ok(Some(Self {
            embeddings: Embedding::new(
                vb.get(
                    (config.vocab_size_per_layer_input, packed_size),
                    "embed_tokens_per_layer.weight",
                )?,
                packed_size,
            ),
            model_projection: Linear::new(
                vb.get(
                    (packed_size, config.hidden_size),
                    "per_layer_model_projection.weight",
                )?,
                None,
                None,
            ),
            projection_norm: Gemma4RmsNorm::load(
                vb.pp("per_layer_projection_norm"),
                hidden_size,
                config.rms_norm_eps,
            )?,
            num_layers: config.num_hidden_layers,
            hidden_size,
            scale: (hidden_size as f64).sqrt(),
            projection_scale: (config.hidden_size as f64).sqrt().recip(),
        }))
    }

    fn forward(&self, input_ids: &Tensor, input_embeddings: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let token_inputs = (self.embeddings.forward(input_ids)? * self.scale)?.reshape((
            batch,
            seq_len,
            self.num_layers,
            self.hidden_size,
        ))?;
        let projected = (self.model_projection.forward(input_embeddings)? * self.projection_scale)?
            .reshape((batch, seq_len, self.num_layers, self.hidden_size))?;
        let projected = self.projection_norm.forward(&projected)?;
        (token_inputs + projected)? * std::f64::consts::FRAC_1_SQRT_2
    }
}

enum Gemma4Output {
    Embedding(Pool),
    Classifier(Linear),
}

pub struct Gemma4Model {
    embeddings: Embedding,
    embedding_scale: f64,
    ple: Option<Gemma4Ple>,
    layers: Vec<Gemma4Layer>,
    norm: Gemma4RmsNorm,
    output: Gemma4Output,
    final_logit_softcapping: Option<f64>,
    local_rope: (Tensor, Tensor),
    full_rope: (Tensor, Tensor),
    local_head_dim: usize,
    full_head_dim: usize,
    num_attention_heads: usize,
    pad_token_id: u32,
    dtype: DType,
    device: Device,
}

impl Gemma4Model {
    #[cfg(feature = "flash-attn")]
    fn forward_hidden_varlen(
        &self,
        batch: &Batch,
        causal: bool,
    ) -> Result<(Tensor, CompactUnfoldTensors)> {
        if !causal && batch.compact_input_ids.is_some() {
            candle::bail!("Bidirectional Gemma4 inference cannot fold causal prefixes")
        }
        let (input_ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let embeddings = (self.embeddings.forward(&input_ids)? * self.embedding_scale)?;
        let per_layer_inputs = match &self.ple {
            Some(ple) => Some(
                ple.forward(&input_ids.unsqueeze(0)?, &embeddings.unsqueeze(0)?)?
                    .squeeze(0)?,
            ),
            None => None,
        };
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch.len() + 1,
            &self.device,
        )?;
        let positions = &compact.position_ids_compact;
        let rope = |cache: &(Tensor, Tensor)| -> Result<(Tensor, Tensor)> {
            Ok((
                index_select(&cache.0, positions, 0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
                index_select(&cache.1, positions, 0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
            ))
        };
        let local = rope(&self.local_rope)?;
        let full = rope(&self.full_rope)?;
        let mut states = embeddings;
        let mut shared_kv = SharedKv::default();
        for (idx, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = match layer.attention.attention_type {
                AttentionType::Full => (&full.0, &full.1),
                AttentionType::Sliding => (&local.0, &local.1),
            };
            let per_layer = match &per_layer_inputs {
                Some(inputs) => Some(inputs.i((.., idx, ..))?),
                None => None,
            };
            states = layer.forward_varlen(
                &states,
                per_layer.as_ref(),
                cos,
                sin,
                &cu_seqlens,
                batch.max_length as usize,
                causal,
                &compact,
                &mut shared_kv,
            )?;
        }
        Ok((self.norm.forward(&states)?, compact))
    }

    pub fn load(vb: VarBuilder, config: &Gemma4Config, model_type: ModelType) -> Result<Self> {
        let text = &config.text_config;
        if text.enable_moe_block {
            candle::bail!("Gemma4 MoE checkpoints are not supported yet")
        }
        if text.layer_types.len() != text.num_hidden_layers {
            candle::bail!(
                "Gemma4 layer_types has {} entries, expected {}",
                text.layer_types.len(),
                text.num_hidden_layers
            )
        }

        let score = match model_type {
            ModelType::Embedding(pool) => Gemma4Output::Embedding(pool),
            ModelType::Classifier => {
                let num_labels = config.num_labels.unwrap_or(config.id2label.len());
                if num_labels == 0 {
                    candle::bail!("Gemma4 classifier config does not define any labels")
                }
                Gemma4Output::Classifier(Linear::new(
                    vb.pp("score")
                        .get((num_labels, text.hidden_size), "weight")?,
                    None,
                    None,
                ))
            }
        };

        let vb = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb
        };
        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((text.vocab_size, text.hidden_size), "weight")?,
            text.hidden_size,
        );
        let ple = Gemma4Ple::load(vb.clone(), text)?;
        let layers = (0..text.num_hidden_layers)
            .map(|idx| Gemma4Layer::load(vb.pp(format!("layers.{idx}")), text, idx))
            .collect::<Result<Vec<_>>>()?;
        let norm = Gemma4RmsNorm::load(vb.pp("norm"), text.hidden_size, text.rms_norm_eps)?;

        let local_inv = get_inv_freqs(
            text.head_dim,
            text.rope_parameters.sliding_attention.rope_theta,
            vb.device(),
            None,
        )?;
        let local_rope = get_cos_sin(text.max_position_embeddings, &local_inv, vb.dtype(), true)?;
        let factor = text
            .rope_parameters
            .full_attention
            .partial_rotary_factor
            .unwrap_or(1.0);
        let rotary_pairs = (factor * text.global_head_dim as f32 / 2.0) as usize;
        let full_inv: Vec<f32> = (0..text.global_head_dim / 2)
            .map(|i| {
                if i < rotary_pairs {
                    1.0 / text
                        .rope_parameters
                        .full_attention
                        .rope_theta
                        .powf((2 * i) as f32 / text.global_head_dim as f32)
                } else {
                    0.0
                }
            })
            .collect();
        let full_inv = Tensor::from_vec(full_inv, (1, text.global_head_dim / 2), vb.device())?;
        let full_rope = get_cos_sin(text.max_position_embeddings, &full_inv, vb.dtype(), true)?;

        Ok(Self {
            embeddings,
            embedding_scale: (text.hidden_size as f64).sqrt(),
            ple,
            layers,
            norm,
            output: score,
            final_logit_softcapping: text.final_logit_softcapping,
            local_rope,
            full_rope,
            local_head_dim: text.head_dim,
            full_head_dim: text.global_head_dim,
            num_attention_heads: text.num_attention_heads,
            pad_token_id: text.pad_token_id,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    fn forward_hidden(&self, batch: &Batch, causal: bool) -> Result<(Tensor, Vec<usize>)> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let mut ids = Vec::with_capacity(batch_size * max_length);
        let mut positions = Vec::with_capacity(batch_size * max_length);
        let mut padding = Vec::with_capacity(batch_size * max_length);
        let mut lengths = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            let length = end - start;
            lengths.push(length);
            ids.extend_from_slice(&batch.input_ids[start..end]);
            positions.extend_from_slice(&batch.position_ids[start..end]);
            padding.extend(std::iter::repeat_n(0f32, length));
            ids.extend(std::iter::repeat_n(self.pad_token_id, max_length - length));
            positions.extend(std::iter::repeat_n(0u32, max_length - length));
            padding.extend(std::iter::repeat_n(-65_504f32, max_length - length));
        }
        let input_ids = Tensor::from_vec(ids, (batch_size, max_length), &self.device)?;
        let position_ids = Tensor::from_vec(positions, (batch_size, max_length), &self.device)?;
        let padding_bias = Tensor::from_vec(padding, (batch_size, 1, 1, max_length), &self.device)?
            .to_dtype(self.dtype)?
            .expand((batch_size, self.num_attention_heads, max_length, max_length))?;

        let input_embeddings = (self.embeddings.forward(&input_ids)? * self.embedding_scale)?;
        let per_layer_inputs = match &self.ple {
            Some(ple) => Some(ple.forward(&input_ids, &input_embeddings)?),
            None => None,
        };
        let rope = |cache: &(Tensor, Tensor), dim: usize| -> Result<(Tensor, Tensor)> {
            let flat = position_ids.flatten_all()?;
            let cos = cache
                .0
                .index_select(&flat, 0)?
                .reshape((batch_size, 1, max_length, dim))?;
            let sin = cache
                .1
                .index_select(&flat, 0)?
                .reshape((batch_size, 1, max_length, dim))?;
            Ok((cos, sin))
        };
        let local = rope(&self.local_rope, self.local_head_dim)?;
        let full = rope(&self.full_rope, self.full_head_dim)?;

        let mut states = input_embeddings;
        let mut shared_kv = SharedKv::default();
        for (idx, layer) in self.layers.iter().enumerate() {
            let (cos, sin) = match layer.attention.attention_type {
                AttentionType::Full => (&full.0, &full.1),
                AttentionType::Sliding => (&local.0, &local.1),
            };
            let per_layer = match &per_layer_inputs {
                Some(inputs) => Some(inputs.i((.., .., idx, ..))?),
                None => None,
            };
            states = layer.forward(
                &states,
                per_layer.as_ref(),
                Some(&padding_bias),
                cos,
                sin,
                causal,
                &mut shared_kv,
            )?;
        }
        Ok((self.norm.forward(&states)?, lengths))
    }

    #[cfg(feature = "flash-attn")]
    fn embed_batch_varlen(
        &self,
        batch: Batch,
        pool: Pool,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (states, compact) = self.forward_hidden_varlen(&batch, false)?;
        let outputs = compact.scatter_unfold(&states)?;
        let pooled = if batch.pooled_indices.is_empty() {
            None
        } else {
            let values = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let start = batch.cumulative_seq_lengths[i as usize] as usize;
                    let end = batch.cumulative_seq_lengths[i as usize + 1] as usize;
                    match pool {
                        Pool::Cls => outputs.i(start)?.unsqueeze(0),
                        Pool::LastToken => outputs.i(end - 1)?.unsqueeze(0),
                        Pool::Mean => {
                            outputs.narrow(0, start, end - start)?.sum_keepdim(0)?
                                / (end - start) as f64
                        }
                        Pool::Splade => candle::bail!("Splade pooling is not supported for Gemma4"),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        let raw = if batch.raw_indices.is_empty() {
            None
        } else {
            let values = batch
                .raw_indices
                .iter()
                .map(|&i| {
                    let start = batch.cumulative_seq_lengths[i as usize] as usize;
                    let end = batch.cumulative_seq_lengths[i as usize + 1] as usize;
                    outputs.narrow(0, start, end - start)
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        Ok((pooled, raw))
    }

    fn embed_batch(&self, batch: Batch, pool: Pool) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if self.device.is_cuda() {
            #[cfg(feature = "flash-attn")]
            {
                return self.embed_batch_varlen(batch, pool);
            }
            #[cfg(not(feature = "flash-attn"))]
            candle::bail!("Gemma4 CUDA inference requires FlashAttention")
        }
        let (outputs, lengths) = self.forward_hidden(&batch, false)?;
        let pooled = if batch.pooled_indices.is_empty() {
            None
        } else {
            let values = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    match pool {
                        Pool::Cls => outputs.i((i, 0))?.unsqueeze(0),
                        Pool::LastToken => outputs.i((i, lengths[i] - 1))?.unsqueeze(0),
                        Pool::Mean => {
                            outputs.i((i, ..lengths[i]))?.sum_keepdim(0)? / lengths[i] as f64
                        }
                        Pool::Splade => candle::bail!("Splade pooling is not supported for Gemma4"),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        let raw = if batch.raw_indices.is_empty() {
            None
        } else {
            let values = batch
                .raw_indices
                .iter()
                .map(|&i| outputs.i((i as usize, ..lengths[i as usize])))
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&values, 0)?)
        };
        Ok((pooled, raw))
    }

    fn predict_batch(&self, batch: Batch, score: &Linear) -> Result<Tensor> {
        if self.device.is_cuda() {
            #[cfg(feature = "flash-attn")]
            {
                let (states, compact) = self.forward_hidden_varlen(&batch, true)?;
                let states = compact.scatter_unfold(&states)?;
                let indices: Vec<u32> = batch
                    .cumulative_seq_lengths
                    .windows(2)
                    .map(|bounds| bounds[1] - 1)
                    .collect();
                let indices = Tensor::from_vec(indices, batch.len(), &self.device)?;
                let logits = score.forward(&index_select(&states, &indices, 0)?)?;
                return match self.final_logit_softcapping {
                    Some(cap) => (logits / cap)?.tanh()? * cap,
                    None => Ok(logits),
                };
            }
            #[cfg(not(feature = "flash-attn"))]
            candle::bail!("Gemma4 CUDA inference requires FlashAttention")
        }
        let (outputs, lengths) = self.forward_hidden(&batch, true)?;
        let last = lengths
            .iter()
            .enumerate()
            .map(|(i, &length)| outputs.i((i, length - 1))?.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?;
        let logits = score.forward(&Tensor::cat(&last, 0)?)?;
        match self.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }
}

impl Model for Gemma4Model {
    fn supports_radix_mlp(&self) -> bool {
        self.device.is_cuda() && cfg!(feature = "flash-attn")
    }

    fn is_padded(&self) -> bool {
        !(self.device.is_cuda() && cfg!(feature = "flash-attn"))
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        match &self.output {
            Gemma4Output::Embedding(pool) => self.embed_batch(batch, pool.clone()),
            Gemma4Output::Classifier(_) => {
                candle::bail!("`embed` is not available for a Gemma4 classifier")
            }
        }
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.output {
            Gemma4Output::Classifier(score) => self.predict_batch(batch, score),
            Gemma4Output::Embedding(_) => {
                candle::bail!("`predict` is not available for a Gemma4 embedding model")
            }
        }
    }
}
