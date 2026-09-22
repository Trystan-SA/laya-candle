//! Loading the checkpoint tokenizer and the four special tokens the sequence format needs.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::error::{Error, Result};

/// The special tokens that frame a decision sequence.
#[derive(Clone, Debug)]
pub struct SpecialTokens {
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,
    /// The literal mask token, stripped out of any user text so it cannot forge a marker.
    pub mask_token: String,
}

/// Load `tokenizer.json` plus the special tokens named in `tokenizer_config.json`.
///
/// The config is optional: the defaults below cover a `tokenizer.json` that stands on its own.
pub fn load(tok_path: &Path, cfg_path: Option<&Path>) -> Result<(Tokenizer, SpecialTokens)> {
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| {
        Error::Tokenizer(format!("{}: {e}", tok_path.display()))
    })?;

    let cfg: Value = match cfg_path {
        Some(path) => {
            let raw = std::fs::read_to_string(path).map_err(|e| Error::io(path.display(), e))?;
            serde_json::from_str(&raw).map_err(|e| Error::json(path.display(), e))?
        }
        None => Value::Null,
    };

    // A special token is stored either as a plain string or as an AddedToken object.
    let token_text = |key: &str, fallback: &str| -> String {
        match cfg.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => o
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or(fallback)
                .to_string(),
            _ => fallback.to_string(),
        }
    };

    let resolve = |key: &str, fallback: &str| -> Result<(u32, String)> {
        let text = token_text(key, fallback);
        let id = tokenizer.token_to_id(&text).ok_or_else(|| {
            Error::Tokenizer(format!(
                "{}: the tokenizer has no id for {key} = {text:?}",
                tok_path.display()
            ))
        })?;
        Ok((id, text))
    };

    let (cls_id, _) = resolve("cls_token", "[CLS]")?;
    let (sep_id, _) = resolve("sep_token", "[SEP]")?;
    let (mask_id, mask_token) = resolve("mask_token", "[MASK]")?;
    let (pad_id, _) = resolve("pad_token", "[PAD]")?;

    Ok((tokenizer, SpecialTokens { cls_id, sep_id, mask_id, pad_id, mask_token }))
}
