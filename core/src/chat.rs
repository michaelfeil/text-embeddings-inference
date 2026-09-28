//! Model-owned chat rendering for decision scoring. No assistant continuation.
use crate::TextEmbeddingsError;
use fastokens::{ChatTemplateOptions, ChatTemplateRenderer};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

fn invalid(message: impl Into<String>) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(message.into())
}
pub fn validate_messages(messages: &[Message]) -> Result<(), TextEmbeddingsError> {
    if messages.is_empty() || messages.iter().all(|m| m.content.trim().is_empty()) {
        return Err(invalid("messages must contain a nonempty conversation"));
    }
    Ok(())
}

pub struct ChatTemplate {
    renderer: ChatTemplateRenderer,
    special_tokens: Map<String, Value>,
}
impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatTemplate")
    }
}
impl ChatTemplate {
    pub fn load(root: &Path) -> Result<Self, TextEmbeddingsError> {
        let config: Value = match std::fs::read_to_string(root.join("tokenizer_config.json")) {
            Ok(s) => serde_json::from_str(&s)
                .map_err(|e| invalid(format!("Invalid tokenizer_config.json: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(invalid(format!("Cannot read tokenizer_config.json: {e}"))),
        };
        let template = match std::fs::read_to_string(root.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                select_template(&config["chat_template"])?
            }
            Err(e) => return Err(invalid(format!("Cannot read chat_template.jinja: {e}"))),
        };
        let mut special_tokens = Map::new();
        for config in [
            config,
            match std::fs::read_to_string(root.join("special_tokens_map.json")) {
                Ok(s) => serde_json::from_str(&s)
                    .map_err(|e| invalid(format!("Invalid special_tokens_map.json: {e}")))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
                Err(e) => return Err(invalid(format!("Cannot read special_tokens_map.json: {e}"))),
            },
        ] {
            if let Some(object) = config.as_object() {
                for (key, value) in object {
                    if key.ends_with("_token") {
                        if let Some(text) = value
                            .as_str()
                            .or_else(|| value.get("content").and_then(Value::as_str))
                        {
                            special_tokens.insert(key.clone(), Value::String(text.into()));
                        }
                    }
                }
            }
        }
        Self::new(&template, special_tokens)
    }
    pub fn new(
        template: &str,
        special_tokens: Map<String, Value>,
    ) -> Result<Self, TextEmbeddingsError> {
        let renderer = ChatTemplateRenderer::new(template)
            .map_err(|e| invalid(format!("Invalid model chat template: {e}")))?;
        Ok(Self {
            renderer,
            special_tokens,
        })
    }
    pub fn render(
        &self,
        messages: &[Message],
        candidate: Option<&str>,
    ) -> Result<String, TextEmbeddingsError> {
        validate_messages(messages)?;
        let render = |turns: &[Message], generation: bool| {
            let options = ChatTemplateOptions {
                add_generation_prompt: generation,
                continue_final_message: false,
                special_tokens: self.special_tokens.clone(),
                extra_context: Map::from_iter([("enable_thinking".into(), Value::Bool(false))]),
                ..Default::default()
            };
            self.renderer
                .render(serde_json::to_value(turns).unwrap(), options)
                .map_err(|e| invalid(format!("Model chat template rejected messages: {e}")))
        };
        let prompt = render(messages, true)?;
        let Some(candidate) = candidate else {
            return Ok(prompt);
        };
        let mut turns = messages.to_vec();
        turns.push(Message {
            role: Role::Assistant,
            content: candidate.into(),
        });
        let completed = render(&turns, false)?;
        if completed.starts_with(&prompt) {
            return Ok(completed);
        }

        // Some checkpoint templates add an empty reasoning channel only to the
        // generation prompt. Derive the assistant body boundary from that same
        // template and preserve its generation prefix plus completed-turn suffix.
        // Never guess special tokens or accept a changed conversation prefix.
        let mut marker = "TEI_ASSISTANT_BODY_BOUNDARY".to_string();
        while messages.iter().any(|m| m.content.contains(&marker)) || candidate.contains(&marker) {
            marker.push('_');
        }
        turns.last_mut().unwrap().content = marker.clone();
        let probe = render(&turns, false)?;
        let mut occurrences = probe.match_indices(&marker);
        let Some((boundary, _)) = occurrences.next() else {
            return Err(invalid("Chat template does not preserve assistant content"));
        };
        if occurrences.next().is_some() || boundary == 0 {
            return Err(invalid(
                "Chat template has an ambiguous assistant content boundary",
            ));
        }
        let prefix = &probe[..boundary];
        let suffix = &probe[boundary + marker.len()..];
        if !prompt.starts_with(prefix)
            || !completed.starts_with(prefix)
            || !completed.ends_with(suffix)
            || completed.len() < prefix.len() + suffix.len()
        {
            return Err(invalid("Chat template changes conversation or assistant boundaries between prompt and completion"));
        }
        Ok(format!("{prompt}{}", &completed[prefix.len()..]))
    }
}
fn select_template(value: &Value) -> Result<String, TextEmbeddingsError> {
    let template = value
        .as_str()
        .or_else(|| value.get("default").and_then(Value::as_str))
        .or_else(|| {
            value.as_array()?.iter().find(|v| v["name"] == "default")?["template"].as_str()
        });
    template.filter(|s| !s.is_empty()).map(str::to_owned).ok_or_else(|| invalid("Decisions require a model chat_template.jinja or a default chat_template in tokenizer_config.json"))
}

/// Never score separately tokenized text that differs from the actual conversation.
pub fn continuation<'a>(
    prompt: &[u32],
    completed: &'a [u32],
) -> Result<&'a [u32], TextEmbeddingsError> {
    if prompt.is_empty() || !completed.starts_with(prompt) || completed.len() <= prompt.len() {
        return Err(invalid("Model chat template/tokenizer does not preserve the assistant generation prefix; this decision template is unsupported"));
    }
    Ok(&completed[prompt.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_template_owned_generation_only_prefix() {
        let renderer = ChatTemplate::new("{% for m in messages %}<{{m.role}}>{{m.content}}</turn>{% endfor %}{% if add_generation_prompt %}<assistant><empty-thought>{% endif %}", Map::new()).unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: "Decide".into(),
        }];
        assert_eq!(
            renderer.render(&messages, Some("{}")).unwrap(),
            "<user>Decide</turn><assistant><empty-thought>{}</turn>"
        );
        let incompatible = ChatTemplate::new("{% if add_generation_prompt %}different{% else %}{% for m in messages %}<{{m.role}}>{{m.content}}</turn>{% endfor %}{% endif %}", Map::new()).unwrap();
        assert!(incompatible.render(&messages, Some("{}")).is_err());
    }
    #[test]
    fn real_roles_and_complete_assistant_turn_are_preserved() {
        let renderer = ChatTemplate::new("{{ bos_token }}{% for m in messages %}<{{m.role}}>{{m.content}}</turn>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}", Map::from_iter([("bos_token".into(), Value::String("<bos>".into()))])).unwrap();
        let history = vec![
            Message {
                role: Role::System,
                content: "Policy".into(),
            },
            Message {
                role: Role::User,
                content: "héllo".into(),
            },
            Message {
                role: Role::Assistant,
                content: "Hello".into(),
            },
            Message {
                role: Role::User,
                content: "Decide".into(),
            },
        ];
        let prompt = renderer.render(&history, None).unwrap();
        assert_eq!(prompt,"<bos><system>Policy</turn><user>héllo</turn><assistant>Hello</turn><user>Decide</turn><assistant>");
        assert_eq!(
            renderer.render(&history, Some("{}")).unwrap(),
            format!("{prompt}{{}}</turn>")
        );
        assert!(continuation(&[1, 2], &[1, 3, 4]).is_err());
        assert_eq!(continuation(&[1, 2], &[1, 2, 3, 4]).unwrap(), &[3, 4]);
        assert!(validate_messages(&[]).is_err());
        assert!(serde_json::from_str::<Message>(r#"{"role":"tool","content":"x"}"#).is_err());
        assert!(select_template(&serde_json::json!({"tool_use":"x"})).is_err());
        assert_eq!(
            select_template(&serde_json::json!([{"name":"default","template":"x"}])).unwrap(),
            "x"
        );
    }
}
