//! CPU reference adapter: JSON lines in, rendered text and token IDs out.
use std::io::{self, BufRead};
use text_embeddings_core::chat::{ChatTemplate, Message};
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let root = std::path::PathBuf::from(std::env::args().nth(1).ok_or("model root required")?);
    let renderer = ChatTemplate::load(&root)?;
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))?;
    for line in io::stdin().lock().lines() {
        let request: serde_json::Value = serde_json::from_str(&line?)?;
        let messages: Vec<Message> = serde_json::from_value(request["messages"].clone())?;
        let text = renderer.render(&messages, request["candidate"].as_str())?;
        let ids = tokenizer.encode(text.as_str(), false)?.get_ids().to_vec();
        println!("{}", serde_json::json!({"text":text,"ids":ids}));
    }
    Ok(())
}
