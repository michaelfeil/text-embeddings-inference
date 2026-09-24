//! Enumeration of finite JSON schemas. Reject unsupported constructs rather than
//! silently evaluating only a subset of the requested output space.
use crate::TextEmbeddingsError;
use serde_json::{Map, Value};

/// A compact plan; complete combinations are only materialized as they are consumed.
pub struct Options {
    plan: Plan,
    next: usize,
}

struct Plan {
    count: usize,
    kind: Kind,
}

enum Kind {
    Values(Vec<Value>),
    Object(Vec<(String, Plan)>),
}

impl Plan {
    fn value(&self, mut index: usize) -> Value {
        match &self.kind {
            Kind::Values(values) => values[index].clone(),
            Kind::Object(properties) => {
                let mut object = Map::new();
                for (key, child) in properties.iter().rev() {
                    object.insert(key.clone(), child.value(index % child.count));
                    index /= child.count;
                }
                Value::Object(object)
            }
        }
    }
}

impl Iterator for Options {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.next == self.plan.count {
            return None;
        }
        let value = self.plan.value(self.next).to_string();
        self.next += 1;
        Some(value)
    }
}

pub fn options(schema: &Value, max_options: usize) -> Result<Options, TextEmbeddingsError> {
    Ok(Options {
        plan: expand(schema, max_options)?,
        next: 0,
    })
}

fn invalid(message: &str) -> TextEmbeddingsError {
    TextEmbeddingsError::Validation(message.into())
}

fn expand(schema: &Value, limit: usize) -> Result<Plan, TextEmbeddingsError> {
    let object = schema
        .as_object()
        .ok_or_else(|| invalid("Schema must be an object"))?;
    for key in object.keys() {
        if ![
            "type",
            "enum",
            "const",
            "properties",
            "required",
            "additionalProperties",
            "title",
            "description",
        ]
        .contains(&key.as_str())
        {
            return Err(invalid(
                "Unsupported schema keyword; use finite enums, booleans, or required objects",
            ));
        }
    }
    if object
        .keys()
        .any(|k| ["properties", "required", "additionalProperties"].contains(&k.as_str()))
        && (object.get("type").and_then(Value::as_str) != Some("object")
            || object.contains_key("enum")
            || object.contains_key("const"))
    {
        return Err(invalid(
            "Object constraints require an object schema without enum or const",
        ));
    }
    let values = if let Some(value) = object.get("const") {
        if object.contains_key("enum") || object.contains_key("properties") {
            return Err(invalid("const cannot be combined with enum or properties"));
        }
        vec![value.clone()]
    } else if let Some(values) = object.get("enum") {
        if object.contains_key("properties") {
            return Err(invalid("enum cannot be combined with properties"));
        }
        {
            let values = values
                .as_array()
                .ok_or_else(|| invalid("enum must be an array"))?;
            if values.len() > limit {
                return Err(invalid("Schema option count exceeds the server limit"));
            }
            values.clone()
        }
    } else {
        match object.get("type").and_then(Value::as_str) {
            Some("boolean") => vec![Value::Bool(false), Value::Bool(true)],
            Some("object") => {
                if object.get("additionalProperties") != Some(&Value::Bool(false)) {
                    return Err(invalid(
                        "Object schemas require additionalProperties: false",
                    ));
                }
                let props = object
                    .get("properties")
                    .and_then(Value::as_object)
                    .ok_or_else(|| invalid("Object schemas require properties"))?;
                let required = object
                    .get("required")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("Every property must be required"))?;
                if required.len() != props.len()
                    || props
                        .keys()
                        .any(|key| required.iter().filter(|v| v.as_str() == Some(key)).count() != 1)
                {
                    return Err(invalid("Every property must be required exactly once"));
                }
                let mut count = 1usize;
                let mut properties = Vec::with_capacity(props.len());
                for (key, child) in props {
                    let child = expand(child, limit)?;
                    count = count
                        .checked_mul(child.count)
                        .ok_or_else(|| invalid("Schema cardinality overflow"))?;
                    if count > limit {
                        return Err(invalid("Schema option count exceeds the server limit"));
                    }
                    properties.push((key.clone(), child));
                }
                if count > limit {
                    return Err(invalid("Schema option count exceeds the server limit"));
                }
                return Ok(Plan {
                    count,
                    kind: Kind::Object(properties),
                });
            }
            _ => {
                return Err(invalid(
                    "Schema must have a finite enum, const, boolean, or required object",
                ))
            }
        }
    };
    if values.is_empty() || values.len() > limit {
        return Err(invalid(
            "Schema option count exceeds the server limit or enum is empty",
        ));
    }
    if let Some(kind) = object.get("type") {
        let kind = kind
            .as_str()
            .ok_or_else(|| invalid("type must be a string"))?;
        for value in &values {
            let valid = match kind {
                "string" => value.is_string(),
                "boolean" => value.is_boolean(),
                "object" => value.is_object(),
                "array" => value.is_array(),
                "null" => value.is_null(),
                "number" => value.is_number(),
                "integer" => value.as_f64().is_some_and(|n| n.fract() == 0.0),
                _ => false,
            };
            if !valid {
                return Err(invalid("Schema value does not match type"));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    if values.iter().any(|v| !seen.insert(v.to_string())) {
        return Err(invalid("Schema enum contains duplicate values"));
    }
    Ok(Plan {
        count: values.len(),
        kind: Kind::Values(values),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn enumerates_entire_cartesian_product() {
        let schema = json!({"type":"object", "additionalProperties":false, "required":["action","urgent"], "properties":{
            "action":{"enum":["approve","reject"]}, "urgent":{"type":"boolean"}
        }});
        let options: Vec<_> = options(&schema, 4).unwrap().collect();
        assert_eq!(options.len(), 4);
        assert!(options.contains(&r#"{"action":"reject","urgent":true}"#.into()));
        assert!(super::options(&schema, 3).is_err());
    }
    #[test]
    fn plans_large_products_without_materializing_them() {
        let properties: Map<String, Value> = (0..20)
            .map(|i| (format!("flag{i}"), json!({"type":"boolean"})))
            .collect();
        let required: Vec<_> = properties.keys().cloned().collect();
        let schema = json!({"type":"object", "properties":properties,
            "required":required, "additionalProperties":false});
        let mut choices = options(&schema, 1 << 20).unwrap();
        assert_eq!(choices.plan.count, 1 << 20);
        let Kind::Object(properties) = &choices.plan.kind else {
            panic!("expected object plan")
        };
        assert_eq!(properties.len(), 20);
        assert!(choices.next().is_some());
        assert!(choices.next().is_some());
        assert_eq!(choices.next, 2);
    }

    #[test]
    fn rejects_unbounded_or_ignored_constraints() {
        for schema in [
            json!({"type":"string"}),
            json!({"enum":[1,2],"minimum":2}),
            json!({"type":"string","enum":[1]}),
            json!({"enum":[]}),
        ] {
            assert!(options(&schema, 100).is_err());
        }
    }
}
