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
    /// Resolve every file through `fetch`, which answers `Ok(None)` for a file that does not
    /// exist and `Err` for one it could not check; `missing` names the error for an absent
    /// required file.
    fn assemble(
        label: String,
        mut fetch: impl FnMut(&str) -> Result<Option<PathBuf>>,
        missing: impl Fn(&str) -> Error,
    ) -> Result<Self> {
        let mut required = |name: &str| fetch(name)?.ok_or_else(|| missing(name));
        Ok(Self {
            agent_config: required(AGENT_CONFIG)?,
            weights: required(WEIGHTS)?,
            encoder_config: required(ENCODER_CONFIG)?,
            tokenizer: required(TOKENIZER)?,
            // Only an absent optional file is skipped: a failure to fetch one is still a
            // failure, and would otherwise surface later as the wrong special-token names.
            tokenizer_config: fetch(TOKENIZER_CONFIG)?,
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
        Self::assemble(
            dir.display().to_string(),
            |name| {
                let p = dir.join(name);
                Ok(p.exists().then_some(p))
            },
            |name| {
                Error::Checkpoint(format!(
                    "{}: missing {name}. A laya checkpoint ships {}.",
                    dir.display(),
                    FILES.join(", ")
                ))
            },
        )
    }
}

/// How a checkpoint is fetched from the Hub.
#[cfg(feature = "hub")]
#[derive(Clone, Debug, Default)]
pub struct HubOptions {
    /// A Hub token, for private repositories. Falls back to `HF_TOKEN`, then to the token
    /// `huggingface-cli login` stored.
    pub token: Option<String>,
    /// Print a progress bar while downloading.
    pub progress: bool,
}

/// The Hub cache directory, resolved the way `huggingface_hub` does: `HF_HUB_CACHE`, else
/// `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
#[cfg(feature = "hub")]
fn hub_cache_dir() -> Result<PathBuf> {
    let var = |key: &str| std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = var("HF_HUB_CACHE") {
        return Ok(dir);
    }
    if let Some(home) = var("HF_HOME") {
        return Ok(home.join("hub"));
    }
    // hf-hub's own default panics when there is no home directory; say what to set instead.
    std::env::home_dir().map(|home| home.join(".cache").join("huggingface").join("hub")).ok_or_else(
        || {
            Error::Hub(
                "no home directory to put the Hub cache in; set HF_HOME or HF_HUB_CACHE".into(),
            )
        },
    )
}

/// Whether a failed fetch means the file does not exist in the repository.
#[cfg(feature = "hub")]
fn is_not_found(e: &hf_hub::api::sync::ApiError) -> bool {
    matches!(e, hf_hub::api::sync::ApiError::RequestError(e) if matches!(**e, ureq::Error::Status(404, _)))
}

#[cfg(feature = "hub")]
impl Checkpoint {
    /// Download a checkpoint from the Hub, or reuse the local Hub cache.
    ///
    /// `subfolder` picks one checkpoint out of a repository that bundles several, the way
    /// `convaiinnovations/laya` bundles `multilingual` and `typed-decisions` alongside the
    /// English one at its root. Only the requested subfolder is fetched.
    ///
    /// The cache location, the endpoint (`HF_ENDPOINT`) and the token follow the same
    /// environment variables as the Python `huggingface_hub`.
    pub fn from_hub(repo: &str, subfolder: Option<&str>, opts: &HubOptions) -> Result<Self> {
        let cache = hf_hub::Cache::new(hub_cache_dir()?);
        // `from_cache` already read the token `huggingface-cli login` stored; only replace it
        // with one that was actually given.
        let mut builder =
            hf_hub::api::sync::ApiBuilder::from_cache(cache).with_progress(opts.progress);
        if let Some(endpoint) = std::env::var("HF_ENDPOINT").ok().filter(|e| !e.is_empty()) {
            builder = builder.with_endpoint(endpoint);
        }
        let token = opts.token.clone().or_else(|| std::env::var("HF_TOKEN").ok());
        if let Some(token) = token.filter(|t| !t.is_empty()) {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build().map_err(|e| Error::Hub(e.to_string()))?.model(repo.to_string());

        let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
        let label = hub_label(repo, subfolder);
        Self::assemble(
            label.clone(),
            |name| match api.get(&format!("{prefix}{name}")) {
                Ok(path) => Ok(Some(path)),
                Err(e) if is_not_found(&e) => Ok(None),
                Err(e) => Err(Error::Hub(format!("{label}: could not fetch {name}: {e}"))),
            },
            |name| Error::Hub(format!("{label}: {prefix}{name} is not in the repository")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::scratch_dir;

    fn touch(dir: &Path, name: &str) {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    #[test]
    fn a_missing_directory_is_rejected() {
        let err = Checkpoint::from_dir("/definitely/not/here").unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("not a directory")), "{err}");
    }

    #[test]
    fn the_first_missing_file_is_named() {
        let dir = scratch_dir("cp-missing");
        touch(&dir, AGENT_CONFIG);
        let err = Checkpoint::from_dir(&dir).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains(WEIGHTS)), "{err}");
    }

    #[test]
    fn the_tokenizer_config_is_optional() {
        let dir = scratch_dir("cp-complete");
        for name in &FILES[..4] {
            touch(&dir, name);
        }
        let cp = Checkpoint::from_dir(&dir).unwrap();
        assert_eq!(cp.weights, dir.join(WEIGHTS));
        assert_eq!(cp.label, dir.display().to_string());
        assert!(cp.tokenizer_config.is_none());

        touch(&dir, TOKENIZER_CONFIG);
        let cp = Checkpoint::from_dir(&dir).unwrap();
        assert_eq!(cp.tokenizer_config, Some(dir.join(TOKENIZER_CONFIG)));
    }

    #[test]
    fn hub_checkpoints_are_labelled_repo_slash_subfolder() {
        assert_eq!(hub_label("org/laya", None), "org/laya");
        assert_eq!(hub_label("org/laya", Some("multilingual")), "org/laya/multilingual");
    }
}
