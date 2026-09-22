//! Helpers shared by the unit tests: a scratch directory and a tokenizer small enough to read.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

/// A fresh, empty directory under the system temp dir.
///
/// Tests run in parallel, so the name carries a per-process counter as well as `name`: two tests
/// asking for the same `name` never share a directory.
pub(crate) fn scratch_dir(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("laya-test-{}-{n}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("creating a scratch dir");
    dir
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

/// The tiny tokenizer, loaded the way a checkpoint's would be.
pub(crate) fn tiny_tokenizer() -> (tokenizers::Tokenizer, crate::tokenizer::SpecialTokens) {
    let dir = scratch_dir("tiny-tokenizer");
    crate::tokenizer::load(&write_tiny_tokenizer(&dir, None), None)
        .expect("loading the tiny tokenizer")
}
