//! Locating the five files a checkpoint is made of, on disk or on the Hugging Face Hub.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Everything the loader needs, resolved to concrete paths.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// `rl_agent_config.json`: head shape, sequence budget and calibration.
    pub agent_config: PathBuf,
    /// `model.safetensors`: backbone and head in one file.
    pub weights: PathBuf,
    /// `encoder/config.json`: the ModernBERT backbone's shape.
    pub encoder_config: PathBuf,
    /// `tokenizer/tokenizer.json`.
    pub tokenizer: PathBuf,
    /// `tokenizer/tokenizer_config.json`, when the checkpoint ships one.
    pub tokenizer_config: Option<PathBuf>,
    /// How to name this checkpoint in errors.
    pub label: String,
}

/// The files a checkpoint is made of, relative to its root.
pub(crate) const FILES: [&str; 5] = [
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

impl Checkpoint {
    /// Resolve a checkpoint laid out as a directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(Error::Checkpoint(format!(
                "{}: not a directory. Point at a checkpoint folder, or use `Agent::from_hub` to \
                 fetch one.",
                dir.display()
            )));
        }
        let required = |name: &str| -> Result<PathBuf> {
            let p = dir.join(name);
            if p.exists() {
                Ok(p)
            } else {
                Err(Error::Checkpoint(format!(
                    "{}: missing {name}. A laya checkpoint ships {}.",
                    dir.display(),
                    FILES.join(", ")
                )))
            }
        };
        let tokenizer_config = dir.join("tokenizer/tokenizer_config.json");
        Ok(Self {
            agent_config: required("rl_agent_config.json")?,
            weights: required("model.safetensors")?,
            encoder_config: required("encoder/config.json")?,
            tokenizer: required("tokenizer/tokenizer.json")?,
            tokenizer_config: tokenizer_config.exists().then_some(tokenizer_config),
            label: dir.display().to_string(),
        })
    }
}

#[cfg(feature = "hub")]
mod hub {
    use super::*;
    use hf_hub::api::sync::ApiBuilder;

    /// How a checkpoint is fetched from the Hub.
    #[derive(Clone, Debug, Default)]
    pub struct HubOptions {
        /// A Hub token, for private repositories. Falls back to `HF_TOKEN`.
        pub token: Option<String>,
        /// Print a progress bar while downloading.
        pub progress: bool,
    }

    impl Checkpoint {
        /// Download a checkpoint from the Hub, or reuse the local Hub cache.
        ///
        /// `subfolder` picks one checkpoint out of a repository that bundles several, the way
        /// `convaiinnovations/laya` bundles `multilingual` and `typed-decisions` alongside the
        /// English one at its root. Only the requested subfolder is fetched.
        pub fn from_hub(repo: &str, subfolder: Option<&str>, opts: &HubOptions) -> Result<Self> {
            let api = ApiBuilder::new()
                .with_progress(opts.progress)
                .with_token(opts.token.clone().or_else(|| std::env::var("HF_TOKEN").ok()))
                .build()
                .map_err(|e| Error::Hub(e.to_string()))?
                .model(repo.to_string());

            let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
            let label = match subfolder {
                Some(s) => format!("{repo}/{s}"),
                None => repo.to_string(),
            };
            let fetch = |name: &str| -> Result<PathBuf> {
                api.get(&format!("{prefix}{name}"))
                    .map_err(|e| Error::Hub(format!("{label}: could not fetch {name}: {e}")))
            };

            Ok(Self {
                agent_config: fetch("rl_agent_config.json")?,
                weights: fetch("model.safetensors")?,
                encoder_config: fetch("encoder/config.json")?,
                tokenizer: fetch("tokenizer/tokenizer.json")?,
                // Optional: a checkpoint whose tokenizer.json carries its own special tokens
                // does not need one, and a missing file must not fail the whole load.
                tokenizer_config: fetch("tokenizer/tokenizer_config.json").ok(),
                label,
            })
        }
    }
}

#[cfg(feature = "hub")]
pub use hub::HubOptions;
