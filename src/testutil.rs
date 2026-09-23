//! Helpers shared by the unit tests: a scratch directory and a tokenizer small enough to read.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

/// A fresh, empty directory under the system temp dir, removed again when dropped.
///
/// Tests run in parallel, so the name carries a per-process counter as well as `name`: two tests
/// asking for the same `name` never share a directory.
pub(crate) struct ScratchDir(PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for ScratchDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for ScratchDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

pub(crate) fn scratch_dir(name: &str) -> ScratchDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("laya-test-{}-{n}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("creating a scratch dir");
    ScratchDir(dir)
}

/// Every word the tiny tokenizer knows, in id order; the four specials come first.
pub(crate) const TINY_VOCAB: &[&str] = &[
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[UNK]",
    "choice",
    "score",
    "noul",
    "question",
    ":",
    "pick",
    "one",
    "level",
    "a",
    "b",
    "c",
    "hello",
    "world",
    "yes",
    "no",
    "the",
    "statement",
    "holds",
    "does",
    "not",
    "true",
    "false",
];

/// The id the tiny tokenizer gives `word`.
pub(crate) fn tiny_id(word: &str) -> u32 {
    TINY_VOCAB
        .iter()
        .position(|w| *w == word)
        .unwrap_or_else(|| panic!("{word:?} is not in TINY_VOCAB")) as u32
}

/// A whitespace `WordLevel` tokenizer: one token per word, `[UNK]` for anything else.
///
/// `truncate_to` adds a truncation rule to the file, to check the loader clears it.
pub(crate) fn tiny_tokenizer_json(truncate_to: Option<usize>) -> String {
    let vocab: serde_json::Map<String, serde_json::Value> =
        TINY_VOCAB.iter().enumerate().map(|(i, w)| (w.to_string(), json!(i))).collect();
    let added: Vec<_> = TINY_VOCAB[..4]
        .iter()
        .enumerate()
        .map(|(i, w)| {
            json!({"id": i, "content": w, "single_word": false, "lstrip": false, "rstrip": false,
                   "normalized": false, "special": true})
        })
        .collect();
    let truncation = truncate_to.map(
        |n| json!({"direction": "Right", "max_length": n, "strategy": "LongestFirst", "stride": 0}),
    );
    json!({
        "version": "1.0",
        "truncation": truncation,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    })
    .to_string()
}

/// Write the tiny tokenizer into `dir` and return its path.
pub(crate) fn write_tiny_tokenizer(dir: &Path, truncate_to: Option<usize>) -> PathBuf {
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, tiny_tokenizer_json(truncate_to)).expect("writing tokenizer.json");
    path
}

/// The tiny tokenizer, loaded the way a checkpoint's would be, without touching the disk.
pub(crate) fn tiny_tokenizer() -> (tokenizers::Tokenizer, crate::tokenizer::SpecialTokens) {
    let tok: tokenizers::Tokenizer =
        tiny_tokenizer_json(None).parse().expect("parsing the tiny tokenizer");
    crate::tokenizer::prepare(tok, &serde_json::Value::Null, "tiny tokenizer")
        .expect("preparing the tiny tokenizer")
}

/// Write a complete, randomly initialised checkpoint small enough to build in milliseconds.
pub(crate) fn write_tiny_checkpoint(dir: &Path) {
    let enc = json!({
        "model_type": "modernbert", "vocab_size": TINY_VOCAB.len(), "hidden_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 32,
        "max_position_embeddings": 64, "global_attn_every_n_layers": 2, "local_attention": 8,
    });
    let agent = json!({"encoder": "tiny", "head_layers": 1, "max_len": 64, "head_max_len": 32});

    std::fs::create_dir_all(dir.join("encoder")).expect("creating encoder/");
    std::fs::create_dir_all(dir.join("tokenizer")).expect("creating tokenizer/");
    std::fs::write(dir.join("encoder/config.json"), enc.to_string()).expect("encoder config");
    std::fs::write(dir.join("rl_agent_config.json"), agent.to_string()).expect("agent config");
    write_tiny_tokenizer(&dir.join("tokenizer"), None);

    let enc_cfg = crate::config::load_encoder_config(&dir.join("encoder/config.json"))
        .expect("the tiny encoder config is valid");
    let weights = crate::model::random_weights(&enc_cfg, 1, 1).expect("initialising weights");
    candle_core::safetensors::save(&weights, dir.join("model.safetensors")).expect("weights");
}
