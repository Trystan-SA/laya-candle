//! Error type shared by the whole crate.

/// Everything that can go wrong while loading a checkpoint or answering questions.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path}: invalid JSON: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    /// A checkpoint directory is missing a file, or a config value makes no sense.
    #[error("{0}")]
    Checkpoint(String),

    /// A question definition the engine cannot render into a sequence.
    #[error("question {id:?}: {message}")]
    Question { id: String, message: String },

    /// An unknown model name was passed to the router.
    #[error("{0}")]
    UnknownModel(String),

    #[cfg(feature = "hub")]
    #[error("hugging face hub: {0}")]
    Hub(String),
}

/// `Result` specialised to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn io(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Error::Io { path: path.to_string(), source }
    }

    pub(crate) fn json(path: impl std::fmt::Display, source: serde_json::Error) -> Self {
        Error::Json { path: path.to_string(), source }
    }

    pub(crate) fn question(id: impl Into<String>, message: impl Into<String>) -> Self {
        Error::Question { id: id.into(), message: message.into() }
    }
}

/// Read and parse a JSON file, naming the file in either failure.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let raw = std::fs::read_to_string(path).map_err(|e| Error::io(path.display(), e))?;
    serde_json::from_str(&raw).map_err(|e| Error::json(path.display(), e))
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::scratch_dir;

    #[test]
    fn read_json_names_the_file_in_both_failures() {
        let dir = scratch_dir("read-json");

        let err = read_json::<serde_json::Value>(&dir.join("missing.json")).unwrap_err();
        assert!(
            matches!(err, Error::Io { ref path, .. } if path.ends_with("missing.json")),
            "{err}"
        );

        let broken = dir.join("broken.json");
        std::fs::write(&broken, "{ not json").unwrap();
        let err = read_json::<serde_json::Value>(&broken).unwrap_err();
        assert!(
            matches!(err, Error::Json { ref path, .. } if path.ends_with("broken.json")),
            "{err}"
        );
        assert!(err.to_string().contains("invalid JSON"), "{err}");
    }
}
