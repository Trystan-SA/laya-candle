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
    let mut tokenizer = Tokenizer::from_file(tok_path)
        .map_err(|e| Error::Tokenizer(format!("{}: {e}", tok_path.display())))?;
    // The sequence builder does its own truncation and padding; a `tokenizer.json` that ships
    // either would otherwise pad every encode to `max_length` for nothing.
    tokenizer.with_truncation(None)?;
    tokenizer.with_padding(None);

    let cfg: Value = match cfg_path {
        Some(path) => crate::error::read_json(path)?,
        None => Value::Null,
    };

    // A special token is stored either as a plain string or as an AddedToken object.
    let token_text = |key: &str, fallback: &str| -> String {
        match cfg.get(key) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => {
                o.get("content").and_then(Value::as_str).unwrap_or(fallback).to_string()
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{scratch_dir, tiny_id, write_tiny_tokenizer};
    use serde_json::json;

    #[test]
    fn special_tokens_default_to_the_usual_names() {
        let dir = scratch_dir("tok-defaults");
        let (tok, sp) = load(&write_tiny_tokenizer(&dir, None), None).unwrap();
        assert_eq!(
            (sp.cls_id, sp.sep_id, sp.mask_id, sp.pad_id),
            (tiny_id("[CLS]"), tiny_id("[SEP]"), tiny_id("[MASK]"), tiny_id("[PAD]"))
        );
        assert_eq!(sp.mask_token, "[MASK]");
        assert_eq!(
            tok.encode_fast("hello world", false).unwrap().get_ids(),
            &[tiny_id("hello"), tiny_id("world")]
        );
    }

    #[test]
    fn a_config_may_name_tokens_as_strings_or_added_token_objects() {
        let dir = scratch_dir("tok-config");
        let tok = write_tiny_tokenizer(&dir, None);
        let cfg = dir.join("tokenizer_config.json");

        let named = json!({"mask_token": {"content": "[MASK]"}, "cls_token": "[CLS]"});
        std::fs::write(&cfg, named.to_string()).unwrap();
        let (_, sp) = load(&tok, Some(&cfg)).unwrap();
        assert_eq!((sp.mask_id, sp.cls_id), (tiny_id("[MASK]"), tiny_id("[CLS]")));

        std::fs::write(&cfg, json!({"mask_token": "<mask>"}).to_string()).unwrap();
        let err = load(&tok, Some(&cfg)).unwrap_err();
        assert!(matches!(err, Error::Tokenizer(ref m) if m.contains("mask_token")), "{err}");
    }

    #[test]
    fn a_shipped_truncation_rule_is_cleared() {
        let dir = scratch_dir("tok-trunc");
        let (tok, _) = load(&write_tiny_tokenizer(&dir, Some(2)), None).unwrap();
        assert_eq!(tok.encode_fast("hello world hello world", false).unwrap().get_ids().len(), 4);
    }
}
