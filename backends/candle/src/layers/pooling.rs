use candle::{DType, Result, Tensor};

/// Compute mean pooling over a variable-length (unpadded) tensor.
///
/// This function converts to F32 for numerical stability before computing the mean,
/// which is critical for F16 models where summing many values can overflow.
///
/// # Arguments
/// * `outputs` - Tensor of shape [total_tokens, hidden_dim]
/// * `cumulative_seq_lengths` - Cumulative sequence lengths [batch_size + 1]
/// * `pooled_indices` - Indices of sequences that need pooling
///
/// # Returns
/// * Tensor of shape [num_pooled, hidden_dim] in F32
pub fn mean_pooling_varlen(
    outputs: &Tensor,
    cumulative_seq_lengths: &[u32],
    pooled_indices: &[u32],
) -> Result<Tensor> {
    let out_dtype = outputs.dtype();
    let outputs_f32 = outputs.to_dtype(DType::F32)?;
    let batch_size = cumulative_seq_lengths.len() - 1;

    let pooled = if batch_size > 1 {
        let results: Result<Vec<Tensor>> = pooled_indices
            .iter()
            .map(|&i| {
                let i = i as usize;
                let start = cumulative_seq_lengths[i];
                let len = cumulative_seq_lengths[i + 1] - start;

                let embeddings = outputs_f32.narrow(0, start as usize, len as usize)?;
                embeddings.sum_keepdim(0)? / (len.max(1) as f64)
            })
            .collect();

        Tensor::cat(&results?, 0)?
    } else {
        let len = cumulative_seq_lengths[1] - cumulative_seq_lengths[0];
        (outputs_f32.sum_keepdim(0)? / (len.max(1) as f64))?
    };

    pooled.to_dtype(out_dtype)
}
