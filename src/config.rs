//! Checkpoint configuration: `rl_agent_config.json` and the encoder's `config.json`.

use std::collections::HashMap;
use std::path::Path;

use candle_transformers::models::modernbert;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result, read_json};
use crate::question::QType;

/// The decision-model side of a checkpoint, as stored in `rl_agent_config.json`.
///
/// Deserialising fills any missing field from [`Default`]; [`AgentConfig::load`] additionally
/// requires `encoder` and `head_layers`, as the reference does, because a head depth silently
/// defaulted to 2 would run a truncated head on a checkpoint trained with more layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    /// Hub id of the backbone the checkpoint was trained on, for provenance only.
    pub encoder: String,
    pub head_layers: usize,
    /// Total sequence budget, including the question head and the state.
    pub max_len: usize,
    /// Token budget for the question head (instructions plus every option).
    pub head_max_len: usize,
    /// Named actions the auxiliary head can recommend; its output width is `len + 1`.
    pub act_costs: HashMap<String, f64>,
    /// Per-type calibration temperature, indexed by [`QType::index`].
    pub temperature: Vec<f32>,
    /// Finer calibration, keyed by `"<type>:<option-count bucket>"`, in file order.
    pub temperature_by_options: IndexMap<String, f32>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            encoder: String::new(),
            head_layers: 2,
            max_len: 512,
            head_max_len: 192,
            act_costs: HashMap::new(),
            temperature: vec![1.0; QType::ALL.len()],
            temperature_by_options: IndexMap::new(),
        }
    }
}

/// Keys `rl_agent_config.json` must spell out rather than leave to a default.
const REQUIRED_AGENT_KEYS: [&str; 2] = ["encoder", "head_layers"];

impl AgentConfig {
    /// Read `rl_agent_config.json`, refusing one that leaves out a [`REQUIRED_AGENT_KEYS`] entry.
    pub fn load(path: &Path) -> Result<Self> {
        let v: Value = read_json(path)?;
        let missing: Vec<_> =
            REQUIRED_AGENT_KEYS.into_iter().filter(|k| v.get(*k).is_none()).collect();
        if !missing.is_empty() {
            return Err(Error::Checkpoint(format!(
                "{}: missing configuration keys {missing:?}, so it is not a laya decision \
                 checkpoint config",
                path.display()
            )));
        }
        serde_json::from_value(v).map_err(|e| Error::json(path.display(), e))
    }

    /// Width of the auxiliary action head.
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// Read the encoder's `config.json` and turn it into candle's ModernBERT config.
///
/// Transformers 5 moved the RoPE bases into a nested `rope_parameters` block; both that shape
/// and the older flat `global_rope_theta` / `local_rope_theta` are accepted.
pub fn load_encoder_config(path: &Path) -> Result<modernbert::Config> {
    let v: Value = read_json(path)?;

    let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "modernbert" {
        return Err(Error::Checkpoint(format!(
            "{}: unsupported encoder architecture {model_type:?}; this crate implements the \
             ModernBERT backbone only",
            path.display()
        )));
    }

    let u64_or = |key: &str, fallback: u64| v.get(key).and_then(Value::as_u64).unwrap_or(fallback);
    let usize_at = |key: &str| -> Result<usize> {
        v.get(key).and_then(Value::as_u64).map(|n| n as usize).ok_or_else(|| {
            Error::Checkpoint(format!("{}: missing or invalid {key:?}", path.display()))
        })
    };
    // candle divides by these, so a zero would panic inside the model build instead of failing
    // here with the file named.
    let nonzero = |key: &str, n: usize| -> Result<usize> {
        if n == 0 {
            Err(Error::Checkpoint(format!("{}: {key:?} must be at least 1", path.display())))
        } else {
            Ok(n)
        }
    };

    let rope = |kind: &str, flat: &str, fallback: f64| -> f64 {
        v.get("rope_parameters")
            .and_then(|r| r.get(kind))
            .and_then(|r| r.get("rope_theta"))
            .and_then(Value::as_f64)
            .or_else(|| v.get(flat).and_then(Value::as_f64))
            .unwrap_or(fallback)
    };

    let hidden_size = nonzero("hidden_size", usize_at("hidden_size")?)?;
    let num_attention_heads = nonzero("num_attention_heads", usize_at("num_attention_heads")?)?;
    if hidden_size % num_attention_heads != 0 {
        return Err(Error::Checkpoint(format!(
            "{}: hidden_size {hidden_size} is not a multiple of num_attention_heads \
             {num_attention_heads}",
            path.display()
        )));
    }
    let global_attn_every_n_layers =
        nonzero("global_attn_every_n_layers", u64_or("global_attn_every_n_layers", 3) as usize)?;

    Ok(modernbert::Config {
        vocab_size: usize_at("vocab_size")?,
        hidden_size,
        num_hidden_layers: usize_at("num_hidden_layers")?,
        num_attention_heads,
        intermediate_size: usize_at("intermediate_size")?,
        max_position_embeddings: usize_at("max_position_embeddings")?,
        layer_norm_eps: v
            .get("layer_norm_eps")
            .or_else(|| v.get("norm_eps"))
            .and_then(Value::as_f64)
            .unwrap_or(1e-5),
        pad_token_id: u64_or("pad_token_id", 0) as u32,
        global_attn_every_n_layers,
        global_rope_theta: rope("full_attention", "global_rope_theta", 160_000.0),
        local_attention: u64_or("local_attention", 128) as usize,
        local_rope_theta: rope("sliding_attention", "local_rope_theta", 10_000.0),
        classifier_config: None,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;
    use crate::testutil::scratch_dir;

    #[test]
    fn missing_fields_take_the_documented_defaults() {
        let cfg: AgentConfig =
            serde_json::from_value(json!({"encoder": "answerdotai/ModernBERT-large"})).unwrap();
        assert_eq!(cfg.encoder, "answerdotai/ModernBERT-large");
        assert_eq!((cfg.head_layers, cfg.max_len, cfg.head_max_len), (2, 512, 192));
        assert_eq!(cfg.temperature, vec![1.0; 3]);
        assert_eq!(cfg.n_act(), 1);
    }

    #[test]
    fn a_config_must_name_its_encoder_and_head_depth() {
        let dir = scratch_dir("agent-cfg");
        let path = dir.join("rl_agent_config.json");

        std::fs::write(&path, json!({"encoder": "x", "max_len": 256}).to_string()).unwrap();
        let err = AgentConfig::load(&path).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("head_layers")), "{err}");

        std::fs::write(&path, json!({"encoder": "x", "head_layers": 3}).to_string()).unwrap();
        assert_eq!(AgentConfig::load(&path).unwrap().head_layers, 3);
    }

    fn write(dir: &Path, v: serde_json::Value) -> PathBuf {
        let p = dir.join("config.json");
        std::fs::write(&p, v.to_string()).unwrap();
        p
    }

    fn shape() -> serde_json::Value {
        json!({
            "model_type": "modernbert", "vocab_size": 100, "hidden_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 32,
            "max_position_embeddings": 64,
        })
    }

    #[test]
    fn rope_bases_are_read_from_either_layout() {
        let dir = scratch_dir("enc-rope");

        let mut nested = shape();
        nested["rope_parameters"] = json!({
            "full_attention": {"rope_theta": 123.0},
            "sliding_attention": {"rope_theta": 45.0},
        });
        let cfg = load_encoder_config(&write(&dir, nested)).unwrap();
        assert_eq!((cfg.global_rope_theta, cfg.local_rope_theta), (123.0, 45.0));

        let mut flat = shape();
        flat["global_rope_theta"] = json!(7.0);
        let cfg = load_encoder_config(&write(&dir, flat)).unwrap();
        assert_eq!((cfg.global_rope_theta, cfg.local_rope_theta), (7.0, 10_000.0));
        assert_eq!(
            (cfg.hidden_size, cfg.local_attention, cfg.global_attn_every_n_layers),
            (16, 128, 3)
        );
    }

    #[test]
    fn other_architectures_and_missing_shape_keys_are_rejected() {
        let dir = scratch_dir("enc-bad");

        let mut bert = shape();
        bert["model_type"] = json!("bert");
        let err = load_encoder_config(&write(&dir, bert)).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("unsupported")), "{err}");

        for (key, value) in [
            ("num_attention_heads", json!(0)),
            ("global_attn_every_n_layers", json!(0)),
            ("hidden_size", json!(0)),
            ("num_attention_heads", json!(3)),
        ] {
            let mut bad = shape();
            bad[key] = value;
            let err = load_encoder_config(&write(&dir, bad)).unwrap_err();
            assert!(matches!(err, Error::Checkpoint(ref m) if m.contains(key)), "{key}: {err}");
        }

        let mut headless = shape();
        headless.as_object_mut().unwrap().remove("hidden_size");
        let err = load_encoder_config(&write(&dir, headless)).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("hidden_size")), "{err}");
    }
}
