//! Jev-compatible content at the HTTP boundary. Native conversations remain explicit.
use infinity_v3_core::input::ModelInput;
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

/// Text or structured JSON data. Objects containing `messages` are reserved for
/// Infinity's native conversation extension and validated as `MessageInput`.
#[derive(Deserialize, utoipa::ToSchema)]
#[serde(untagged)]
#[allow(dead_code)] // Documents the wire shape; adapters continue using ModelInput.
pub enum SystemOneState {
    Text(String),
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

pub fn deserialize_state<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<ModelInput, D::Error> {
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(text) => Ok(ModelInput::Text(text)),
        Value::Object(ref object) if object.contains_key("messages") => {
            serde_json::from_value(value).map_err(serde::de::Error::custom)
        }
        Value::Object(_) | Value::Array(_) => Ok(ModelInput::Text(super::render(&value))),
        _ => Err(serde::de::Error::custom(
            "state must be a string, object or array",
        )),
    }
}

pub fn instructions(value: Option<&Value>) -> Result<String, String> {
    match value {
        Some(value @ (Value::String(_) | Value::Object(_) | Value::Array(_))) => {
            Ok(super::render(value))
        }
        _ => Err("Question instructions must be a string, object or array".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::systemone::SystemOneRequest;
    use serde_json::json;

    fn request(state: Value) -> Result<SystemOneRequest, serde_json::Error> {
        serde_json::from_value(json!({"state":state,"questions":{}}))
    }

    #[test]
    fn structured_state_is_rendered_without_changing_text_or_order() {
        for state in [
            json!("hello"),
            json!({"z":"a,b:c","a":["é",false]}),
            json!([1,{"text":"hello"}]),
        ] {
            let expected = super::super::render(&state);
            let parsed = request(state).unwrap();
            assert!(matches!(parsed.state, ModelInput::Text(text) if text == expected));
            assert!(parsed.model.is_none());
        }
        for state in [Value::Null, json!(true), json!(42)] {
            assert!(request(state).is_err());
        }
    }

    #[test]
    fn native_messages_are_never_silently_flattened() {
        assert!(matches!(
            request(json!({"messages":[{"role":"user","content":"hello"}]}))
                .unwrap()
                .state,
            ModelInput::Messages(_)
        ));
        for state in [
            json!({"messages":"bad"}),
            json!({"messages":[{"role":"invalid","content":"hello"}]}),
            json!({"messages":[],"extra":true}),
        ] {
            assert!(request(state).is_err());
        }
    }

    #[test]
    fn structured_instructions_reject_invalid_scalars() {
        for value in [
            json!("question"),
            json!({"question":"Which?","context":[1,2]}),
            json!(["Which?",{"context":true}]),
        ] {
            assert_eq!(
                instructions(Some(&value)).unwrap(),
                super::super::render(&value)
            );
        }
        for value in [Value::Null, json!(true), json!(1)] {
            assert!(instructions(Some(&value)).is_err());
        }
        assert!(instructions(None).is_err());
    }
}
