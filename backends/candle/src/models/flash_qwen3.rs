use crate::flash_attn::flash_attn_varlen;
use crate::layers::{
    get_cos_sin, get_inv_freqs, index_select, CompactUnfoldTensors, HiddenAct, Linear, RMSNorm,
};
use crate::models::{Model, Qwen3Config};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    q_norm: RMSNorm,
    k_norm: RMSNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,
    use_bidirectional_attention: bool,

    span: tracing::Span,
}

impl Qwen3Attention {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        if config.use_sliding_window {
            candle::bail!("Sliding window is not supported for Qwen3");
        }

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get(
            (num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let query_bias = if config.attention_bias {
            Some(vb.pp("q_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let q_proj = Linear::new(query_weight, query_bias, None);

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let key_bias = if config.attention_bias {
            Some(
                vb.pp("k_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let k_proj = Linear::new(key_weight, key_bias, None);

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_bias = if config.attention_bias {
            Some(
                vb.pp("v_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let v_proj = Linear::new(value_weight, value_bias, None);

        let o_proj_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let o_proj = Linear::new(o_proj_weight, None, None);

        let q_norm = RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
        let k_norm = RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            attention_head_size,
            softmax_scale,
            use_bidirectional_attention: config.use_bidirectional_attention,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact_tensors: &CompactUnfoldTensors,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq_len, heads, head_dim]
        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

        let q = q.reshape(
            [
                input_shape,
                &[self.num_attention_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let k = k.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let v = v.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;

        // Apply normalization layers
        let (q, _) = self.q_norm.forward(&q, None)?;
        let (k, _) = self.k_norm.forward(&k, None)?;

        // Apply RoPE in COMPACT space
        apply_rotary_inplace(&q, &k, &cos, &sin, true)?;

        // Expand Q, K, V to ORIGINAL layout for attention
        let q = compact_tensors.scatter_unfold(&q)?;
        let k = compact_tensors.scatter_unfold(&k)?;
        let v = compact_tensors.scatter_unfold(&v)?;

        let attention = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            !self.use_bidirectional_attention,
            None,
            None,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        // Compact attention output back to COMPACT layout before o_proj
        let attention = compact_tensors.fold_gather(&attention)?;

        self.o_proj.forward(&attention)
    }
}

struct Qwen3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl Qwen3MLP {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let intermediate_size = config.intermediate_size;

        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
            act: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(1, 0, self.intermediate_size)?;
        let up_states = gate_up_states.narrow(1, self.intermediate_size, self.intermediate_size)?;

        let gate_states = self.act.forward(&gate_states)?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

struct Qwen3Layer {
    attention: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl Qwen3Layer {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let attention = Qwen3Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Qwen3MLP::load(vb.pp("mlp"), config)?;

        let input_layer_norm = RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layer_norm = RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layer_norm,
            post_attention_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
        compact_tensors: &CompactUnfoldTensors,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (normed_hidden_states, res) = self.input_layer_norm.forward(hidden_states, residual)?;

        let attn_output = self.attention.forward(
            &normed_hidden_states,
            cu_seqlens,
            cos,
            sin,
            max_s,
            compact_tensors,
        )?;

        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&res))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        Ok((mlp_output, attn_res))
    }
}

pub struct FlashQwen3Model {
    lm_head: Option<Tensor>,
    eos_token_id: u32,
    embeddings: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RMSNorm,
    linear_output_projection: Option<Linear>,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pool: Pool,
    pub device: Device,
    use_bidirectional_attention: bool,

    span: tracing::Span,
}

impl FlashQwen3Model {
    pub fn load(vb: VarBuilder, config: &Qwen3Config, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashQwen3 requires Cuda"),
        }

        if !matches!(vb.dtype(), DType::F16 | DType::BF16) {
            candle::bail!("FlashQwen3 requires DType::F16 or DType::BF16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Qwen3")
            }
            ModelType::Embedding(pool) => pool,
        };

        let lm_head = if vb.contains_tensor("lm_head.weight") {
            Some(
                vb.pp("lm_head")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
            )
        } else {
            None
        };

        // The Qwen3-Reranker models contain the `model` key
        // https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea
        let root = vb.clone();
        let vb = if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb
        };

        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        // Reuse the embedding allocation when the checkpoint ties its LM head.
        let lm_head = lm_head.or_else(|| {
            config
                .tie_word_embeddings
                .then(|| embeddings.embeddings().clone())
        });

        let layers = (0..config.num_hidden_layers)
            .map(|index| Qwen3Layer::load(vb.pp(format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let linear_output_projection = config.load_output_projection(&root, &vb)?;

        let inv_freqs = get_inv_freqs(
            layers[0].attention.attention_head_size,
            config.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) = get_cos_sin(
            config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false,
        )?;

        Ok(Self {
            lm_head,
            eos_token_id: config.eos_token_id as u32,
            embeddings,
            layers,
            norm,
            linear_output_projection,
            cos_cache,
            sin_cache,
            pool,
            device: vb.device().clone(),
            use_bidirectional_attention: config.use_bidirectional_attention,
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
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

    fn forward_hidden(&self, batch: &Batch) -> Result<(Tensor, CompactUnfoldTensors)> {
        let batch_size = batch.cumulative_seq_lengths.len() - 1;

        // Create compact/unfold tensors and get embeddings
        let (input_ids, compact_tensors) = CompactUnfoldTensors::from_batch(batch, &self.device)?;
        let mut hidden_states = self.embeddings.forward(&input_ids)?.contiguous()?;

        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        // sin and cos are applied on the compact formation, therefore should be on the compact array
        let cos = index_select(&self.cos_cache, &compact_tensors.position_ids_compact, 0)?;
        let sin = index_select(&self.sin_cache, &compact_tensors.position_ids_compact, 0)?;

        let mut residual = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                &cu_seqlens,
                &cos,
                &sin,
                batch.max_length as usize,
                &compact_tensors,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        let (outputs, _) = self.norm.forward(&hidden_states, residual.as_ref())?;

        Ok((outputs, compact_tensors))
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
                Pool::Mean => {
                    if batch_size > 1 {
                        // for each request that requires pooling
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .into_iter()
                            .map(|i| {
                                let i = i as usize;
                                let start = batch.cumulative_seq_lengths[i];
                                let len = batch.cumulative_seq_lengths[i + 1] - start;

                                // Mean
                                let embeddings = outputs.narrow(0, start as usize, len as usize)?;
                                embeddings.sum_keepdim(0)? / (len as f64)
                            })
                            .collect();

                        // Concatenate all results
                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some((outputs.sum_keepdim(0)? / (batch.max_length as f64))?)
                    }
                }
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

impl Model for FlashQwen3Model {
    fn decision_prompt_style(&self) -> Option<text_embeddings_backend_core::DecisionPromptStyle> {
        self.supports_decision_scoring()
            .then_some(text_embeddings_backend_core::DecisionPromptStyle::Qwen3)
    }

    fn supports_decision_scoring(&self) -> bool {
        !self.use_bidirectional_attention
            && self.linear_output_projection.is_none()
            && self.lm_head.is_some()
    }

    fn score_options(&self, batch: Batch, prompt_lengths: &[usize]) -> Result<Vec<f32>> {
        if self.use_bidirectional_attention || self.linear_output_projection.is_some() {
            candle::bail!("Decision scoring requires an unmodified causal language model")
        }
        let weight = self
            .lm_head
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("Model has no language-model head".into()))?;
        if prompt_lengths.is_empty()
            || prompt_lengths.len() + 1 != batch.cumulative_seq_lengths.len()
            || batch
                .cumulative_seq_lengths
                .windows(2)
                .zip(prompt_lengths)
                .any(|(w, &len)| len == 0 || (w[1] - w[0]) as usize <= len)
        {
            candle::bail!("Each branch must contain a nonempty prompt and continuation")
        }
        let (hidden, _) = self.forward_hidden(&batch)?;
        // Evaluate each unique predictor row once, including full-vocabulary normalization.
        // Small chunks bound temporary logits memory independently of prompt length.
        let mut rows = Vec::new();
        let mut row_map = std::collections::HashMap::new();
        let mut edges = Vec::new();
        for (bounds, &prompt_length) in batch.cumulative_seq_lengths.windows(2).zip(prompt_lengths)
        {
            let mut branch = Vec::new();
            for pos in (bounds[0] as usize + prompt_length - 1)..(bounds[1] as usize) {
                let row = batch.scatter_unfold.as_ref().map_or(pos as u32, |s| s[pos]);
                let next = rows.len();
                let index = *row_map.entry(row).or_insert_with(|| {
                    rows.push(row);
                    next
                });
                let target = if pos + 1 == bounds[1] as usize {
                    self.eos_token_id
                } else {
                    batch.input_ids[pos + 1]
                };
                branch.push((index, target as usize));
            }
            edges.push(branch);
        }
        let mut scores = vec![0f32; edges.len()];
        let mut targets_by_row = vec![Vec::new(); rows.len()];
        for (branch_index, branch) in edges.iter().enumerate() {
            for &(row, target) in branch {
                targets_by_row[row].push((branch_index, target));
            }
        }
        let vocab_size = weight.dim(0)?;
        for (chunk_index, chunk) in rows.chunks(32).enumerate() {
            let indices = Tensor::from_vec(chunk.to_vec(), chunk.len(), &self.device)?;
            let states = index_select(&hidden, &indices, 0)?;
            let logits = states.matmul(&weight.t()?)?.to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, 1)?.flatten_all()?;
            let start = chunk_index * 32;
            let mut selected = Vec::new();
            let mut branches = Vec::new();
            for (local_row, targets) in targets_by_row[start..start + chunk.len()]
                .iter()
                .enumerate()
            {
                for &(branch, target) in targets {
                    selected.push((local_row * vocab_size + target) as u32);
                    branches.push(branch);
                }
            }
            let indices = Tensor::from_vec(selected.clone(), selected.len(), &self.device)?;
            // Transfer only requested edges, never the full vocabulary matrix.
            let values = log_probs.index_select(&indices, 0)?.to_vec1::<f32>()?;
            for (branch, value) in branches.into_iter().zip(values) {
                scores[branch] += value;
            }
        }
        Ok(scores)
    }

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

#[cfg(all(test, feature = "flash-attn"))]
mod decision_tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_model(tied: bool) -> Result<FlashQwen3Model> {
        let device = Device::new_cuda(0)?;
        let config: Qwen3Config = serde_json::from_value(serde_json::json!({
            "attention_bias":false, "vocab_size":64, "head_dim":128,
            "hidden_size":128, "intermediate_size":256, "num_hidden_layers":2,
            "num_attention_heads":1, "num_key_value_heads":1, "hidden_act":"silu",
            "max_position_embeddings":128, "rms_norm_eps":0.000001,
            "rope_theta":10000.0, "use_sliding_window":false, "eos_token_id":63,
            "tie_word_embeddings":tied
        }))
        .unwrap();
        let mut weights = HashMap::new();
        let mut add = |name: String, shape: Vec<usize>, norm: bool| -> Result<()> {
            let seed = weights.len() + 1;
            let values: Vec<f32> = (0..shape.iter().product())
                .map(|i| {
                    if norm {
                        1.0
                    } else {
                        (((i * 17 + seed * 131) as f32) * 0.013).sin() * 0.04
                    }
                })
                .collect();
            weights.insert(
                name,
                Tensor::from_vec(values, shape, &device)?.to_dtype(DType::F16)?,
            );
            Ok(())
        };
        add("model.embed_tokens.weight".into(), vec![64, 128], false)?;
        if !tied {
            add("lm_head.weight".into(), vec![64, 128], false)?;
        }
        add("model.norm.weight".into(), vec![128], true)?;
        for layer in 0..2 {
            let prefix = format!("model.layers.{layer}");
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                add(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    vec![128, 128],
                    false,
                )?;
            }
            for norm in ["q_norm", "k_norm"] {
                add(format!("{prefix}.self_attn.{norm}.weight"), vec![128], true)?;
            }
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{prefix}.{norm}.weight"), vec![128], true)?;
            }
            for projection in ["gate_proj", "up_proj"] {
                add(
                    format!("{prefix}.mlp.{projection}.weight"),
                    vec![256, 128],
                    false,
                )?;
            }
            add(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![128, 256],
                false,
            )?;
        }
        FlashQwen3Model::load(
            VarBuilder::from_tensors(weights, DType::F16, &device),
            &config,
            ModelType::Embedding(Pool::LastToken),
        )
    }

    fn batch(sequences: &[Vec<u32>], compact: bool) -> Batch {
        let mut b = Batch {
            input_ids: vec![],
            token_type_ids: vec![],
            position_ids: vec![],
            cumulative_seq_lengths: vec![0],
            max_length: 0,
            pooled_indices: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let mut prefixes = HashMap::new();
        let (mut ids, mut positions, mut scatter, mut fold) = (vec![], vec![], vec![], vec![]);
        for sequence in sequences {
            for (pos, &token) in sequence.iter().enumerate() {
                let next = ids.len() as u32;
                let row = *prefixes
                    .entry(sequence[..=pos].to_vec())
                    .or_insert_with(|| {
                        ids.push(token);
                        positions.push(pos as u32);
                        fold.push(b.input_ids.len() as u32);
                        next
                    });
                scatter.push(row);
                b.input_ids.push(token);
                b.position_ids.push(pos as u32);
                b.token_type_ids.push(0);
            }
            b.cumulative_seq_lengths.push(b.input_ids.len() as u32);
            b.max_length = b.max_length.max(sequence.len() as u32);
        }
        if compact {
            b.compact_input_ids = Some(ids);
            b.compact_position_ids = Some(positions);
            b.scatter_unfold = Some(scatter);
            b.fold_gather = Some(fold);
        }
        b
    }

    #[test]
    fn decision_scores_match_independent_complete_sequence_likelihoods() -> Result<()> {
        for tied in [false, true] {
            let mut model = tiny_model(tied)?;
            assert!(model.supports_decision_scoring());
            // Shared option prefixes, prefix-of-another option, different lengths,
            // and >32 predictor rows exercise EOS and chunk boundaries.
            let mut long = vec![1, 2, 3];
            long.extend((4..44).map(|i| i % 60));
            let sequences = vec![vec![1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 5, 6], long];
            let prompt_lengths = [2, 3, 1, 2];
            let scores = model.score_options(batch(&sequences, true), &prompt_lengths)?;
            let expanded_scores = model.score_options(batch(&sequences, false), &prompt_lengths)?;
            for (i, sequence) in sequences.iter().enumerate() {
                let (hidden, _) = model.forward_hidden(&batch(&[sequence.clone()], false))?;
                let logits = hidden
                    .matmul(&model.lm_head.as_ref().unwrap().t()?)?
                    .to_dtype(DType::F32)?;
                let probabilities = candle_nn::ops::log_softmax(&logits, 1)?.to_vec2::<f32>()?;
                let mut expected = probabilities[sequence.len() - 1][63];
                for token in prompt_lengths[i]..sequence.len() {
                    expected += probabilities[token - 1][sequence[token] as usize];
                }
                assert!(
                    (scores[i] - expected).abs() < 0.05,
                    "compact {} vs reference {}",
                    scores[i],
                    expected
                );
                assert!((expanded_scores[i] - expected).abs() < 0.05);
            }
            assert!(
                (scores[0] - scores[2]).abs() > 0.1,
                "Nonuniform model must distinguish options"
            );
            model.lm_head = None;
            assert!(!model.supports_decision_scoring());
        }
        Ok(())
    }
}
