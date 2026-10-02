//! OneJev's qev-labels-v2 text protocol, using the shared decision queue.
use super::*;
use text_embeddings_core::{
    chat::ChatProcessor,
    input::{ContentPart, MessageContent, MessageRole},
};

const SYSTEM: &str = "Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.";
const SPLIT_LABELS: &[&str] = &[
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ",
    "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
];

pub struct Onejev {
    tokenizer: Tokenizer,
    chat: ChatProcessor,
    codes: Vec<String>,
    token_ids: Vec<u32>,
    max_input_length: usize,
}

impl Onejev {
    pub(super) fn load(
        path: &Path,
        mut tokenizer: Tokenizer,
        max_input_length: usize,
    ) -> anyhow::Result<Self> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let chat = ChatProcessor::load(path)
            .map_err(anyhow::Error::msg)?
            .ok_or_else(|| anyhow::anyhow!("OneJev requires the checkpoint's chat template"))?;
        let mut codes = (b'A'..=b'Z')
            .map(|c| (c as char).to_string())
            .chain(
                (b'A'..=b'Z')
                    .flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char)))
                    .filter(|s| !SPLIT_LABELS.contains(&s.as_str())),
            )
            .take(256)
            .collect::<Vec<_>>();
        let ids = codes
            .iter()
            .map(|code| {
                let encoded = tokenizer
                    .encode(code.as_str(), false)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(
                    if encoded.len() == 1
                        && tokenizer
                            .decode(encoded.get_ids(), false)
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?
                            == *code
                    {
                        Some(encoded.get_ids()[0])
                    } else {
                        None
                    },
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        anyhow::ensure!(
            ids[..26].iter().all(Option::is_some),
            "OneJev requires single-token A-Z answer labels"
        );
        if ids.iter().any(Option::is_none) {
            codes.truncate(26);
        }
        let token_ids = ids
            .into_iter()
            .take(codes.len())
            .map(Option::unwrap)
            .collect();
        Ok(Self {
            tokenizer,
            chat,
            codes,
            token_ids,
            max_input_length,
        })
    }

    pub(super) fn prepare(&self, request: SystemOneRequest) -> Result<Vec<Question>, String> {
        let state = match request.state {
            ModelInput::Text(text) => text,
            ModelInput::Messages(messages) => {
                if messages.messages.is_empty() {
                    return Err("State messages must not be empty".into());
                }
                for message in &messages.messages {
                    if !matches!(message.role, MessageRole::User | MessageRole::Assistant) {
                        return Err("OneJev state accepts user and assistant roles".into());
                    }
                    if let MessageContent::Parts(parts) = &message.content {
                        if parts.iter().any(|p| !matches!(p, ContentPart::Text { .. })) {
                            return Err("OneJev currently supports text only".into());
                        }
                    }
                }
                serde_json::to_string_pretty(&messages).map_err(|e| e.to_string())?
            }
        };
        if state.chars().count() > 50_000 {
            return Err("State exceeds 50000 characters".into());
        }
        if request.head_max_len.is_some() {
            return Err("head_max_len is specific to Laya".into());
        }
        if request.questions.is_empty() {
            return Err("OneJev requires at least one question".into());
        }
        let limit = request
            .max_len
            .unwrap_or(self.max_input_length)
            .min(self.max_input_length);
        let mut total_options = 0;
        request.questions.into_iter().map(|(id, spec)| {
            let spec = spec.as_object().ok_or("Question must be an object")?;
            if spec.keys().any(|key| !matches!(key.as_str(), "type" | "instructions" | "criteria")) { return Err("OneJev question supports only type, instructions and criteria".into()); }
            let kind = spec.get("type").and_then(Value::as_str).ok_or("Question needs type")?;
            let default = match kind {
                "choice" => "Which option applies to the state?",
                "score" => "Which level describes the state?",
                "noul" => "Is the statement true, or is the answer to the question yes?",
                _ => return Err("Question type must be choice, noul or score".into()),
            };
            let instructions = entry(spec.get("instructions").unwrap_or(&Value::Null))?;
            let instructions = if instructions.is_empty() { default } else { &instructions };
            let criteria = spec.get("criteria").cloned().unwrap_or(Value::Null);
            let (labels, names, descriptions): (Vec<String>, Vec<String>, Vec<String>) = match kind {
                "choice" => {
                    let options = criteria.as_object().ok_or("Choice criteria must be an object")?;
                    let labels = options.keys().cloned().collect::<Vec<_>>();
                    (labels.clone(), labels, options.values().map(entry).collect::<Result<_, _>>()?)
                }
                "score" => {
                    let options = criteria.as_array().ok_or("Score criteria must be an array")?;
                    ((0..options.len()).map(|i| i.to_string()).collect(), (0..options.len()).map(|i| format!("level {i}")).collect(), options.iter().map(entry).collect::<Result<_, _>>()?)
                }
                "noul" => {
                    if !criteria.is_null() && (criteria.as_object().is_none_or(|v| v.len() != 2 || !v.contains_key("true") || !v.contains_key("false"))) { return Err("Noul criteria must contain exactly false and true".into()); }
                    let descriptions = [("true", "the statement is true / the answer is yes"), ("false", "the statement is false / the answer is no")].into_iter().map(|(key, default)| {
                        match criteria.get(key).filter(|v| !v.is_null()) { Some(value) => entry(value), None => Ok(default.into()) }
                    }).collect::<Result<_, _>>()?;
                    (vec!["false".into(), "true".into()], vec!["yes".into(), "no".into()], descriptions)
                }
                _ => unreachable!(),
            };
            let n = labels.len();
            total_options += n;
            if n == 0 || n > self.codes.len().min(255) || (kind == "score" && n > 10) || total_options > 512 { return Err(format!("OneJev supports 1-{} options per question and 512 per request", self.codes.len().min(255))); }
            let options = self.codes.iter().zip(names.iter().zip(&descriptions)).map(|(code, (name, description))| {
                if description.is_empty() { format!("{code}. {name}") } else { format!("{code}. {name}: {description}") }
            }).collect::<Vec<_>>().join("\n");
            let header = if kind == "score" { "Rate the state:" } else { "Question:" };
            let word = if n <= 26 { "letter" } else { "label" };
            let user = format!("<state>\n{state}\n</state>\n\n{header} {instructions}\n\nOptions:\n{options}\n\nAnswer with one {word}: {}.", self.codes[..n].join(", "));
            if ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"].iter().any(|token| user.contains(token)) { return Err("State and questions must not contain reserved chat or media tokens".into()); }
            let prompt = self.chat.render_decision(SYSTEM, &user)?;
            let ids = self.tokenizer.encode(prompt.as_str(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
            if ids.is_empty() || ids.len() > limit { return Err(format!("OneJev prompt has {} tokens, exceeding max_len {limit}; prompts are never truncated", ids.len())); }
            let length = ids.len();
            Ok(Question { id, kind: kind.into(), labels, criteria, compute_chars: prompt.chars().count(),
                encoding: ValidEncoding { multimodal: None, input_ids: ids, token_type_ids: vec![0; length], position_ids: (0..length as u32).collect(), tokens: vec![], offsets: vec![] },
                input: DecisionInput::OptionTokens { token_ids: self.token_ids[..n].to_vec() } })
        }).collect()
    }
}

fn entry(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.trim().into()),
        Value::Object(_) | Value::Array(_) => {
            serde_json::to_string_pretty(value).map_err(|e| e.to_string())
        }
        _ => Err("Instructions and criteria must be strings, objects, arrays or null".into()),
    }
}

pub(super) fn answer(question: &Question, mut output: DecisionOutput) -> Result<Value, String> {
    // OneJev trains A=yes/B=no; the shared answer formatter expects false/true.
    if question.kind == "noul" && output.logits.len() == 2 {
        output.logits.swap(0, 1);
    }
    rune::answer(question, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires OneJev checkpoint and upstream prompt fixtures"]
    fn checkpoint_matches_upstream_prompts() -> anyhow::Result<()> {
        let path = std::env::var("ONEJEV_CHECKPOINT_DIR")?;
        let fixtures: Vec<Value> =
            serde_json::from_slice(&std::fs::read(std::env::var("ONEJEV_REFERENCE_FILE")?)?)?;
        let tokenizer = Tokenizer::from_file(Path::new(&path).join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let model = Onejev::load(Path::new(&path), tokenizer, 32768)?;
        for case in fixtures {
            let request: SystemOneRequest = serde_json::from_value(case["request"].clone())?;
            let questions = model.prepare(request).map_err(anyhow::Error::msg)?;
            for reference in case["fixtures"].as_array().unwrap() {
                let question = questions
                    .iter()
                    .find(|q| q.id == reference["id"].as_str().unwrap())
                    .unwrap();
                let expected: Vec<u32> = serde_json::from_value(reference["input_ids"].clone())?;
                assert_eq!(question.encoding.input_ids, expected, "{}", question.id);
                let DecisionInput::OptionTokens { token_ids } = &question.input else {
                    panic!("expected option tokens");
                };
                assert_eq!(
                    *token_ids,
                    serde_json::from_value::<Vec<u32>>(reference["token_ids"].clone())?
                );
                let logits = serde_json::from_value(reference["logits"].clone())?;
                let actual = answer(
                    question,
                    DecisionOutput {
                        logits,
                        action_probability: 1.0,
                    },
                )
                .map_err(anyhow::Error::msg)?;
                let expected = &case["answers"][&question.id];
                for field in ["noul", "score", "confidence"] {
                    if let Some(value) = expected[field].as_f64() {
                        assert!((actual[field].as_f64().unwrap() - value).abs() < 1e-6);
                    }
                }
                assert_eq!(actual["choice"], expected["choice"]);
            }
        }
        let media = json!({"state":{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]},"questions":{"q":{"type":"noul"}}});
        assert!(model
            .prepare(serde_json::from_value(media)?)
            .err()
            .unwrap()
            .contains("text only"));
        let oversized =
            json!({"state":"Invoice is paid.","max_len":1,"questions":{"q":{"type":"noul"}}});
        assert!(model
            .prepare(serde_json::from_value(oversized)?)
            .err()
            .unwrap()
            .contains("never truncated"));
        Ok(())
    }
}
