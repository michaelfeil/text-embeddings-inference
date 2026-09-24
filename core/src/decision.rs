//! Atomic, exhaustive candidate batches. Attention uses the expanded layout;
//! RadixMLP folds identical causal prefixes for projections and MLPs.
use crate::TextEmbeddingsError;
use text_embeddings_backend::Batch;

/// A group is a joint Cartesian product; groups never condition on each other.
pub struct Group {
    pub name: String,
    context: std::sync::Arc<str>,
    schema: serde_json::Value,
    pub options: crate::decision_schema::Options,
}

impl Group {
    pub fn prompt(&self, style: text_embeddings_backend::DecisionPromptStyle) -> String {
        // Context precedes the group's schema so independent groups share its
        // causal prefix. Render lazily to avoid copying context for every group.
        let context = &self.context;
        let schema = &self.schema;
        let instructions = format!("Answer the questions using the context and their descriptions. Return only a JSON object matching the supplied schema.\nContext:\n{context}\n\nQuestions (JSON Schema):\n{schema}");
        match style {
            text_embeddings_backend::DecisionPromptStyle::Gemma4 => {
                format!("<bos><|turn>user\n{instructions}<turn|>\n<|turn>model\n")
            }
            text_embeddings_backend::DecisionPromptStyle::Qwen3 => format!(
                "<|im_start|>system\n\
             Answer the questions using the context and their descriptions. \
             Return only a JSON object matching the supplied schema.\n\
             <|im_end|>\n<|im_start|>user\nContext:\n{context}\n\n\
             Questions (JSON Schema):\n{schema}<|im_end|>\n<|im_start|>assistant\n"
            ),
        }
    }
}

pub struct ScoredGroup {
    pub name: String,
    pub options: Vec<String>,
    pub log_scores: Vec<f32>,
}

pub fn groups(
    context: &str,
    questions: std::collections::BTreeMap<String, serde_json::Value>,
    max_options: usize,
) -> Result<Vec<Group>, TextEmbeddingsError> {
    use serde_json::{json, Map, Value};
    let invalid = |s: &str| TextEmbeddingsError::Validation(s.into());
    if context.trim().is_empty() || questions.is_empty() {
        return Err(invalid("context and questions must be nonempty"));
    }
    let mut grouped = std::collections::BTreeMap::<String, Map<String, Value>>::new();
    for (name, mut schema) in questions {
        if name.trim().is_empty() {
            return Err(invalid("Question names must be nonempty"));
        }
        let object = schema
            .as_object_mut()
            .ok_or_else(|| invalid("Each question must be a finite JSON Schema object"))?;
        let group = match object.remove("group") {
            None => "default".to_string(),
            Some(Value::String(name)) if !name.trim().is_empty() => name,
            _ => return Err(invalid("group must be a nonempty string")),
        };
        grouped.entry(group).or_default().insert(name, schema);
    }
    let context: std::sync::Arc<str> = context.into();
    grouped
        .into_iter()
        .map(|(name, properties)| {
            let required: Vec<_> = properties.keys().cloned().collect();
            let schema = json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            });
            let options = crate::decision_schema::options(&schema, max_options)?;
            Ok(Group {
                name,
                context: context.clone(),
                schema,
                options,
            })
        })
        .collect()
}

pub struct EncodedGroup {
    pub prompt: Vec<u32>,
    pub options: Vec<Vec<u32>>,
}

pub fn option_batch(
    groups: &[EncodedGroup],
    max_batch_tokens: usize,
    max_length: usize,
) -> Result<(Batch, Vec<usize>), TextEmbeddingsError> {
    let invalid = |message: &str| TextEmbeddingsError::Validation(message.into());
    if groups.is_empty()
        || groups.iter().any(|g| {
            g.prompt.is_empty() || g.options.is_empty() || g.options.iter().any(Vec::is_empty)
        })
    {
        return Err(invalid("prompt and every option must contain tokens"));
    }
    let mut total = 0usize;
    for (prompt, option) in groups
        .iter()
        .flat_map(|g| g.options.iter().map(move |o| (&g.prompt, o)))
    {
        let length = prompt
            .len()
            .checked_add(option.len())
            .ok_or_else(|| invalid("Token count overflow"))?;
        total = total
            .checked_add(length)
            .ok_or_else(|| invalid("Token count overflow"))?;
        if length > max_length {
            return Err(invalid("A prompt + option exceeds the model context limit"));
        }
        if total > max_batch_tokens || total > u32::MAX as usize {
            return Err(invalid(
                "Expanded options exceed max_batch_tokens; no branches were evaluated",
            ));
        }
    }
    let mut batch = Batch {
        input_ids: Vec::with_capacity(total),
        token_type_ids: vec![0; total],
        position_ids: Vec::with_capacity(total),
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
    for (prompt, option) in groups
        .iter()
        .flat_map(|g| g.options.iter().map(move |o| (&g.prompt, o)))
    {
        let length = (prompt.len() + option.len()) as u32;
        batch.input_ids.extend_from_slice(prompt);
        batch.input_ids.extend_from_slice(option);
        batch.position_ids.extend(0..length);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(length);
    }
    let (ids, positions, scatter, fold) = radix_mlp::compute_fold_and_scatter(
        &batch.input_ids,
        &batch.position_ids,
        &batch.cumulative_seq_lengths,
        None,
    );
    batch.compact_input_ids = Some(ids);
    batch.compact_position_ids = Some(positions);
    batch.scatter_unfold = Some(scatter);
    batch.fold_gather = Some(fold);
    let prompt_lengths = groups
        .iter()
        .flat_map(|g| std::iter::repeat_n(g.prompt.len(), g.options.len()))
        .collect();
    Ok((batch, prompt_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option_batch(
        prompt: &[u32],
        options: &[Vec<u32>],
        budget: usize,
        length: usize,
    ) -> Result<Batch, TextEmbeddingsError> {
        super::option_batch(
            &[EncodedGroup {
                prompt: prompt.to_vec(),
                options: options.to_vec(),
            }],
            budget,
            length,
        )
        .map(|v| v.0)
    }

    #[test]
    fn broadcasts_all_options_and_shares_prefixes() {
        let b = option_batch(&[1, 2], &[vec![3, 4], vec![3, 5], vec![6]], 11, 4).unwrap();
        assert_eq!(b.input_ids, [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 6]);
        assert_eq!(b.cumulative_seq_lengths, [0, 4, 8, 11]);
        let compact = b.compact_input_ids.unwrap();
        assert_eq!(compact.len(), 6);
        let scatter = b.scatter_unfold.unwrap();
        for (i, &row) in scatter.iter().enumerate() {
            assert_eq!(compact[row as usize], b.input_ids[i]);
        }
        assert_eq!(scatter[0], scatter[4]);
        assert_eq!(scatter[2], scatter[6]);
        assert_ne!(scatter[3], scatter[7]);
        let groups = [
            EncodedGroup {
                prompt: vec![1, 2],
                options: vec![vec![3], vec![4]],
            },
            EncodedGroup {
                prompt: vec![1, 5, 6],
                options: vec![vec![7]],
            },
        ];
        let (batch, lengths) = super::option_batch(&groups, 10, 4).unwrap();
        assert_eq!(lengths, [2, 2, 3]);
        assert_eq!(batch.cumulative_seq_lengths, [0, 3, 6, 10]);
        assert!(super::option_batch(&groups, 9, 4).is_err());
    }

    #[test]
    fn question_groups_are_joint_inside_and_independent_outside() {
        let questions = serde_json::json!({
            "action": {"description":"Apply the refund policy", "enum":["approve","reject","escalate"]},
            "urgent": {"type":"boolean"},
            "language": {"group":"language", "enum":["en","de","other"]}
        });
        let mut planned = groups(
            "Refund after 45 days",
            serde_json::from_value(questions.clone()).unwrap(),
            100,
        )
        .unwrap();
        assert_eq!(planned.len(), 2);
        assert_eq!(planned[0].name, "default");
        let style = text_embeddings_backend::DecisionPromptStyle::Qwen3;
        assert!(planned[0].prompt(style).contains("Apply the refund policy"));
        assert!(!planned[0].prompt(style).contains("language"));
        assert!(!planned[1].prompt(style).contains("Apply the refund policy"));
        let gemma = planned[0].prompt(text_embeddings_backend::DecisionPromptStyle::Gemma4);
        assert!(gemma.starts_with("<bos><|turn>user\n"));
        assert!(gemma.ends_with("<turn|>\n<|turn>model\n"));
        let joint: Vec<_> = planned[0].options.by_ref().collect();
        assert_eq!(joint.len(), 6);
        assert_eq!(planned[1].options.by_ref().count(), 3);
        assert!(joint.contains(&r#"{"action":"reject","urgent":true}"#.to_string()));
        let mut ungrouped = questions;
        ungrouped["language"]
            .as_object_mut()
            .unwrap()
            .remove("group");
        let mut all = groups(
            "Refund after 45 days",
            serde_json::from_value(ungrouped).unwrap(),
            100,
        )
        .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].options.by_ref().count(), 18);
    }

    #[test]
    fn bounds_expanded_tokens_before_compaction() {
        assert!(option_batch(&[1, 2], &[vec![3], vec![4]], 5, 3).is_err());
        assert!(option_batch(&[1, 2], &[vec![3], vec![4]], 6, 3).is_ok());
        assert!(option_batch(&[1, 2], &[vec![3]], 100, 2).is_err());
        assert!(option_batch(&[], &[vec![3]], 100, 2).is_err());
        assert!(option_batch(&[1], &[], 100, 2).is_err());
        assert!(option_batch(&[1], &[vec![]], 100, 2).is_err());
    }
}
