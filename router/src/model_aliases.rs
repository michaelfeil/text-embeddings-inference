use crate::DecisionProtocol;
use anyhow::Result;
use std::path::Path;
use text_embeddings_backend::Pool;

/// Resolve CLI shorthands before downloading model artifacts.
pub(crate) fn resolve(
    model_id: String,
    protocol: Option<DecisionProtocol>,
    pooling: Option<&Pool>,
) -> Result<(String, Option<DecisionProtocol>)> {
    // Local directories take precedence over registered shorthands.
    if Path::new(&model_id).is_dir() || model_id != "pplx-decider-v1.1-27b" {
        return Ok((model_id, protocol));
    }
    anyhow::ensure!(
        pooling.is_none(),
        "The Decider alias cannot be used with --pooling"
    );
    anyhow::ensure!(
        protocol
            .as_ref()
            .is_none_or(|p| matches!(p, DecisionProtocol::Pplx)),
        "The Decider alias requires --decision-protocol pplx"
    );
    Ok((
        "perplexity-ai/pplx-decider-v1.1-27b".into(),
        Some(DecisionProtocol::Pplx),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decider_alias_selects_checkpoint_and_protocol() -> Result<()> {
        for protocol in [None, Some(DecisionProtocol::Pplx)] {
            let (id, protocol) = resolve("pplx-decider-v1.1-27b".into(), protocol, None)?;
            assert_eq!(id, "perplexity-ai/pplx-decider-v1.1-27b");
            assert!(matches!(protocol, Some(DecisionProtocol::Pplx)));
        }
        Ok(())
    }

    #[test]
    fn decider_alias_rejects_conflicting_options() {
        for protocol in [
            DecisionProtocol::Rune,
            DecisionProtocol::Onejev,
            DecisionProtocol::Clef,
        ] {
            assert!(resolve("pplx-decider-v1.1-27b".into(), Some(protocol), None).is_err());
        }
        assert!(resolve("pplx-decider-v1.1-27b".into(), None, Some(&Pool::Mean)).is_err());
    }

    #[test]
    fn ordinary_model_ids_and_local_directories_are_preserved() -> Result<()> {
        for id in ["Qwen/Qwen3-Embedding-0.6B", "/tmp"] {
            let (actual, protocol) = resolve(id.into(), None, Some(&Pool::Mean))?;
            assert_eq!(actual, id);
            assert!(protocol.is_none());
        }
        Ok(())
    }
}
