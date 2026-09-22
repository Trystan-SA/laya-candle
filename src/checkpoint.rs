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

const AGENT_CONFIG: &str = "rl_agent_config.json";
const WEIGHTS: &str = "model.safetensors";
const ENCODER_CONFIG: &str = "encoder/config.json";
const TOKENIZER: &str = "tokenizer/tokenizer.json";
/// Optional: a `tokenizer.json` that carries its own special tokens does not need one.
const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config.json";

/// The files a checkpoint is made of, relative to its root.
const FILES: [&str; 5] = [AGENT_CONFIG, WEIGHTS, ENCODER_CONFIG, TOKENIZER, TOKENIZER_CONFIG];

/// How a Hub checkpoint names itself: the repository, plus the subfolder when it has one.
pub(crate) fn hub_label(repo: &str, subfolder: Option<&str>) -> String {
    match subfolder {
        Some(s) => format!("{repo}/{s}"),
        None => repo.to_string(),
    }
}

impl Checkpoint {
    /// Resolve every file through `fetch`, which fails for a file it cannot find.
    fn assemble(label: String, mut fetch: impl FnMut(&str) -> Result<PathBuf>) -> Result<Self> {
        Ok(Self {
            agent_config: fetch(AGENT_CONFIG)?,
            weights: fetch(WEIGHTS)?,
            encoder_config: fetch(ENCODER_CONFIG)?,
            tokenizer: fetch(TOKENIZER)?,
            // A missing optional file must not fail the whole load.
            tokenizer_config: fetch(TOKENIZER_CONFIG).ok(),
            label,
        })
    }

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
        Self::assemble(dir.display().to_string(), |name| {
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
        })
    }
}

/// How a checkpoint is fetched from the Hub.
#[cfg(feature = "hub")]
#[derive(Clone, Debug, Default)]
pub struct HubOptions {
    /// A Hub token, for private repositories. Falls back to `HF_TOKEN`.
    pub token: Option<String>,
    /// Print a progress bar while downloading.
    pub progress: bool,
}

#[cfg(feature = "hub")]
impl Checkpoint {
    /// Download a checkpoint from the Hub, or reuse the local Hub cache.
    ///
    /// `subfolder` picks one checkpoint out of a repository that bundles several, the way
    /// `convaiinnovations/laya` bundles `multilingual` and `typed-decisions` alongside the
    /// English one at its root. Only the requested subfolder is fetched.
    pub fn from_hub(repo: &str, subfolder: Option<&str>, opts: &HubOptions) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(opts.progress)
            .with_token(opts.token.clone().or_else(|| std::env::var("HF_TOKEN").ok()))
            .build()
            .map_err(|e| Error::Hub(e.to_string()))?
            .model(repo.to_string());

        let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
        let label = hub_label(repo, subfolder);
        Self::assemble(label.clone(), |name| {
            api.get(&format!("{prefix}{name}"))
                .map_err(|e| Error::Hub(format!("{label}: could not fetch {name}: {e}")))
        })
    }
}
