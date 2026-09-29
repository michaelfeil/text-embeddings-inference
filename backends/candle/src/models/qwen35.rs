use super::qwen35_config::{Qwen35Config, Qwen35TextConfig};
use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, index_select, CompactUnfoldTensors, Linear};
use crate::models::{Model, Qwen3Config};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct Norm {
    weight: Tensor,
    eps: f64,
}
impl Norm {
    fn load(vb: VarBuilder, width: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(width, "weight")?
                .to_dtype(DType::F32)?
                .affine(1., 1.)?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.to_dtype(DType::F32)?;
        let inv = (f.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?;
        f.broadcast_mul(&inv)?
            .broadcast_mul(&self.weight)?
            .to_dtype(x.dtype())
    }
}
fn linear(vb: VarBuilder, input: usize, output: usize) -> Result<Linear> {
    Ok(Linear::new(vb.get((output, input), "weight")?, None, None))
}
struct Dense {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Dense {
    fn load(vb: VarBuilder, h: usize, i: usize) -> Result<Self> {
        Ok(Self {
            gate: linear(vb.pp("gate_proj"), h, i)?,
            up: linear(vb.pp("up_proj"), h, i)?,
            down: linear(vb.pp("down_proj"), i, h)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?)
    }
}
struct FullAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: Norm,
    kn: Norm,
    heads: usize,
    kv: usize,
    dim: usize,
    rotary: usize,
}
impl FullAttention {
    fn load(vb: VarBuilder, c: &Qwen35TextConfig) -> Result<Self> {
        let (h, n, k, d) = (
            c.hidden_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        );
        Ok(Self {
            q: linear(vb.pp("q_proj"), h, 2 * n * d)?,
            k: linear(vb.pp("k_proj"), h, k * d)?,
            v: linear(vb.pp("v_proj"), h, k * d)?,
            o: linear(vb.pp("o_proj"), n * d, h)?,
            qn: Norm::load(vb.pp("q_norm"), d, c.rms_norm_eps)?,
            kn: Norm::load(vb.pp("k_norm"), d, c.rms_norm_eps)?,
            heads: n,
            kv: k,
            dim: d,
            rotary: c.rotary_dim(),
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cu: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let qg = self.q.forward(x)?.reshape((t, self.heads, 2 * self.dim))?;
        let q = self.qn.forward(&qg.narrow(2, 0, self.dim)?)?;
        let gate = qg
            .narrow(2, self.dim, self.dim)?
            .reshape((t, self.heads * self.dim))?;
        let k = self
            .kn
            .forward(&self.k.forward(x)?.reshape((t, self.kv, self.dim))?)?;
        let v = self.v.forward(x)?.reshape((t, self.kv, self.dim))?;
        let rotate = |x: &Tensor| -> Result<Tensor> {
            let r = x.narrow(2, 0, self.rotary)?;
            let half = self.rotary / 2;
            let other = Tensor::cat(&[r.narrow(2, half, half)?.neg()?, r.narrow(2, 0, half)?], 2)?;
            let rotated = (r.broadcast_mul(&cos.unsqueeze(1)?)?
                + other.broadcast_mul(&sin.unsqueeze(1)?)?)?;
            Tensor::cat(
                &[rotated, x.narrow(2, self.rotary, self.dim - self.rotary)?],
                2,
            )?
            .contiguous()
        };
        let q = compact.scatter_unfold(&rotate(&q)?)?;
        let k = compact.scatter_unfold(&rotate(&k)?)?;
        let v = compact.scatter_unfold(&v)?;
        let a = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu,
            cu,
            max_s,
            max_s,
            (self.dim as f32).sqrt().recip(),
            true,
            None,
            None,
        )?
        .flatten_from(1)?;
        let a = compact.fold_gather(&a)?;
        self.o.forward(&(a * candle_nn::ops::sigmoid(&gate)?)?)
    }
}
struct DeltaAttention {
    qkv: Linear,
    z: Linear,
    ab: Linear,
    out: Linear,
    delta: crate::layers::qwen35_gdn::GatedDelta,
}
impl DeltaAttention {
    fn load(vb: VarBuilder, c: &Qwen35TextConfig) -> Result<Self> {
        let (kh, vh, h) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.hidden_size,
        );
        let channels = (2 * kh + vh) * 128;
        let ab = Tensor::cat(
            &[
                vb.pp("in_proj_a").get((vh, h), "weight")?,
                vb.pp("in_proj_b").get((vh, h), "weight")?,
            ],
            0,
        )?;
        Ok(Self {
            qkv: linear(vb.pp("in_proj_qkv"), h, channels)?,
            z: linear(vb.pp("in_proj_z"), h, vh * 128)?,
            ab: Linear::new(ab, None, None),
            out: linear(vb.pp("out_proj"), vh * 128, h)?,
            delta: crate::layers::qwen35_gdn::GatedDelta {
                conv: vb
                    .pp("conv1d")
                    .get((channels, 1, c.linear_conv_kernel_dim), "weight")?
                    .reshape((channels, c.linear_conv_kernel_dim))?,
                a_log: vb.to_dtype(DType::F32).get(vh, "A_log")?,
                dt_bias: vb.get(vh, "dt_bias")?.to_dtype(DType::F32)?,
                norm: vb.pp("norm").get(128, "weight")?,
                key_heads: kh,
                value_heads: vh,
                epsilon: c.rms_norm_eps as f32,
            },
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        lengths: &[u32],
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let qkv = compact.scatter_unfold(&self.qkv.forward(x)?)?;
        let z = compact.scatter_unfold(&self.z.forward(x)?)?;
        let ab = compact.scatter_unfold(&self.ab.forward(x)?)?;
        self.out
            .forward(&compact.fold_gather(&self.delta.forward(&qkv, &z, &ab, lengths)?)?)
    }
}
enum Attention {
    Full(FullAttention),
    Linear(DeltaAttention),
}
struct Layer {
    attn: Attention,
    input_norm: Norm,
    post_norm: Norm,
    moe: super::qwen3_moe::Qwen3Moe,
    shared: Dense,
    shared_gate: Linear,
}
impl Layer {
    fn load(
        vb: VarBuilder,
        c: &Qwen35TextConfig,
        index: usize,
        moe_config: &Qwen3Config,
    ) -> Result<Self> {
        Ok(Self {
            attn: match c.layer_types[index].as_str() {
                "linear_attention" => {
                    Attention::Linear(DeltaAttention::load(vb.pp("linear_attn"), c)?)
                }
                "full_attention" => Attention::Full(FullAttention::load(vb.pp("self_attn"), c)?),
                _ => candle::bail!("Unknown Qwen3.5 attention layer"),
            },
            input_norm: Norm::load(vb.pp("input_layernorm"), c.hidden_size, c.rms_norm_eps)?,
            post_norm: Norm::load(
                vb.pp("post_attention_layernorm"),
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            moe: super::qwen3_moe::Qwen3Moe::load(vb.pp("mlp"), moe_config)?,
            shared: Dense::load(
                vb.pp("mlp.shared_expert"),
                c.hidden_size,
                c.shared_expert_intermediate_size,
            )?,
            shared_gate: linear(vb.pp("mlp.shared_expert_gate"), c.hidden_size, 1)?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cu: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        batch: &Batch,
        compact: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let normalized = self.input_norm.forward(x)?;
        let attention = match &self.attn {
            Attention::Full(a) => a.forward(
                &normalized,
                cu,
                cos,
                sin,
                batch.max_length as usize,
                compact,
            )?,
            Attention::Linear(a) => {
                a.forward(&normalized, &batch.cumulative_seq_lengths, compact)?
            }
        };
        let residual = (x + attention)?;
        let n = self.post_norm.forward(&residual)?;
        let shared = self
            .shared
            .forward(&n)?
            .broadcast_mul(&candle_nn::ops::sigmoid(&self.shared_gate.forward(&n)?)?)?;
        residual + (self.moe.forward(&n)? + shared)?
    }
}
pub struct Qwen35Model {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: Norm,
    linear_output_projection: Option<Linear>,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pool: Pool,
    pub device: Device,
    use_bidirectional_attention: bool,
    span: tracing::Span,
}
impl Qwen35Model {
    pub fn load(vb: VarBuilder, config: &Qwen35Config, model_type: ModelType) -> Result<Self> {
        config.validate()?;
        let c = config.text();
        if !matches!(vb.device(), Device::Cuda(_)) || vb.dtype() != DType::BF16 {
            candle::bail!("Qwen3.5-MoE currently requires BF16 CUDA")
        }
        let pool = match model_type {
            ModelType::Embedding(p) => p,
            _ => candle::bail!("Qwen3.5-MoE requires embedding mode"),
        };
        let model = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb.clone()
        };
        let embeddings = Embedding::new(
            model
                .pp("embed_tokens")
                .get((c.vocab_size, c.hidden_size), "weight")?,
            c.hidden_size,
        );
        let moe_config = c.moe_config()?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(model.pp(format!("layers.{i}")), c, i, &moe_config))
            .collect::<Result<Vec<_>>>()?;
        let norm = Norm::load(model.pp("norm"), c.hidden_size, c.rms_norm_eps)?;
        let inv = get_inv_freqs(
            c.rotary_dim(),
            c.rope_parameters.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) =
            get_cos_sin(c.max_position_embeddings, &inv, vb.dtype(), true)?;
        Ok(Self {
            embeddings,
            layers,
            norm,
            linear_output_projection: None,
            cos_cache,
            sin_cache,
            pool,
            device: vb.device().clone(),
            use_bidirectional_attention: false,
            span: tracing::span!(tracing::Level::TRACE, "qwen35"),
        })
    }
    fn forward_hidden(&self, batch: &Batch) -> Result<(Tensor, CompactUnfoldTensors)> {
        let (ids, compact) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut hidden = self.embeddings.forward(&ids)?.contiguous()?;
        let cu = Tensor::new(batch.cumulative_seq_lengths.as_slice(), &self.device)?;
        #[cfg(feature = "fa4")]
        let _fa4_batch = crate::fa4_native::prepare_batch(&cu, &batch.cumulative_seq_lengths)?;
        let cos = index_select(&self.cos_cache, &compact.position_ids_compact, 0)?;
        let sin = index_select(&self.sin_cache, &compact.position_ids_compact, 0)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cu, &cos, &sin, batch, &compact)?;
        }
        Ok((self.norm.forward(&hidden)?, compact))
    }
    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let (outputs, compact_tensors) = self.forward_hidden(&batch)?;
        let batch_size = batch.cumulative_seq_lengths.len() - 1;

        let outputs = if let Some(linear_output_projection) = &self.linear_output_projection {
            linear_output_projection.forward(&outputs)?
        } else {
            outputs
        };

        self.pool_outputs(batch, outputs, compact_tensors, batch_size)
    }

    fn pool_outputs(
        &self,
        batch: Batch,
        outputs: Tensor,
        compact_tensors: CompactUnfoldTensors,
        batch_size: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;
        // Expand final outputs to original layout for pooling/raw extraction
        let outputs = compact_tensors.scatter_unfold(&outputs)?;
        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                // CLS and LastToken pooling
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        // Get token indices form cu_seqlens
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        // If raw_indices is empty, we don't need to do anything with
                        // the pooled_indices
                        if has_raw_requests {
                            // We need the pooled indices to select the correct cls indices
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            // Only select indices that requires pooling
                            indices = index_select(&indices, &pooled_indices, 0)?
                        }

                        // Select tokens
                        Some(index_select(&outputs, &indices, 0)?)
                    } else {
                        Some(
                            match self.pool {
                                Pool::Cls => outputs.i(0)?,
                                Pool::LastToken => {
                                    outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                                }
                                _ => unreachable!(),
                            }
                            .unsqueeze(0)?,
                        )
                    }
                }
                // Mean pooling
                Pool::Mean => Some(crate::layers::mean_pool(
                    &outputs,
                    &batch.cumulative_seq_lengths,
                    &batch.pooled_indices,
                )?),
                Pool::Splade => {
                    unreachable!();
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                // Create indexing vector for the embeddings
                let shape = batch.input_ids.len();
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    // Get start/end token index of this specific member of the batch
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(index_select(&outputs, &final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for Qwen35Model {
    fn is_padded(&self) -> bool {
        false
    }

    fn supports_radix_mlp(&self) -> bool {
        !self.use_bidirectional_attention
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
