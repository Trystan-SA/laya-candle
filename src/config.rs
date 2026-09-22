//! Checkpoint configuration: `rl_agent_config.json` and the encoder's `config.json`.

use std::collections::HashMap;
use std::path::Path;

use candle_transformers::models::modernbert;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::calibration::clamp_temperature;
use crate::error::{Error, Result};

/// The decision-model side of a checkpoint, as stored in `rl_agent_config.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Hub id of the backbone the checkpoint was trained on, for provenance only.
    #[serde(default)]
    pub encoder: String,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    /// Total sequence budget, including the question head and the state.
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    /// Token budget for the question head (instructions plus every option).
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    /// Named actions the auxiliary head can recommend; its output width is `len + 1`.
    #[serde(default)]
    pub act_costs: HashMap<String, f64>,
    /// Per-type calibration temperature, indexed by [`crate::QType::index`].
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f32>,
    /// Finer calibration, keyed by `"<type>:<option-count bucket>"`.
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperature() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}

impl AgentConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| Error::io(path.display(), e))?;
        serde_json::from_str(&raw).map_err(|e| Error::json(path.display(), e))
    }

    /// Width of the auxiliary action head.
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }

    /// The temperatures actually applied, clamped into a range that cannot fake confidence.
    ///
    /// Returns the clamped per-type values, the clamped per-bucket values, and a description of
    /// every temperature that had to be clamped so the caller can say so out loud.
    pub fn calibration(&self) -> (Vec<f32>, HashMap<String, f32>, Vec<String>) {
        let mut rejected = Vec::new();
        let by_options: HashMap<String, f32> = self
            .temperature_by_options
            .iter()
            .map(|(k, v)| {
                let c = clamp_temperature(*v);
                if c != *v {
                    rejected.push(format!("{k}={v:.4}"));
                }
                (k.clone(), c)
            })
            .collect();
        let per_type: Vec<f32> = self
            .temperature
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let c = clamp_temperature(*t);
                if c != *t {
                    rejected.push(format!("temperature[{i}]={t:.4}"));
                }
                c
            })
            .collect();
        (per_type, by_options, rejected)
    }
}

/// Read the encoder's `config.json` and turn it into candle's ModernBERT config.
///
/// Transformers 5 moved the RoPE bases into a nested `rope_parameters` block; both that shape
/// and the older flat `global_rope_theta` / `local_rope_theta` are accepted.
pub fn load_encoder_config(path: &Path) -> Result<modernbert::Config> {
    let raw = std::fs::read_to_string(path).map_err(|e| Error::io(path.display(), e))?;
    let v: Value = serde_json::from_str(&raw).map_err(|e| Error::json(path.display(), e))?;

    let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "modernbert" {
        return Err(Error::Checkpoint(format!(
            "{}: unsupported encoder architecture {model_type:?}; this crate implements the \
             ModernBERT backbone only",
            path.display()
        )));
    }

    let usize_at = |key: &str| -> Result<usize> {
        v.get(key)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| Error::Checkpoint(format!("{}: missing or invalid {key:?}", path.display())))
    };

    let rope = |kind: &str, flat: &str, fallback: f64| -> f64 {
        v.get("rope_parameters")
            .and_then(|r| r.get(kind))
            .and_then(|r| r.get("rope_theta"))
            .and_then(Value::as_f64)
            .or_else(|| v.get(flat).and_then(Value::as_f64))
            .unwrap_or(fallback)
    };

    Ok(modernbert::Config {
        vocab_size: usize_at("vocab_size")?,
        hidden_size: usize_at("hidden_size")?,
        num_hidden_layers: usize_at("num_hidden_layers")?,
        num_attention_heads: usize_at("num_attention_heads")?,
        intermediate_size: usize_at("intermediate_size")?,
        max_position_embeddings: usize_at("max_position_embeddings")?,
        layer_norm_eps: v
            .get("layer_norm_eps")
            .or_else(|| v.get("norm_eps"))
            .and_then(Value::as_f64)
            .unwrap_or(1e-5),
        pad_token_id: v.get("pad_token_id").and_then(Value::as_u64).unwrap_or(0) as u32,
        global_attn_every_n_layers: v
            .get("global_attn_every_n_layers")
            .and_then(Value::as_u64)
            .unwrap_or(3) as usize,
        global_rope_theta: rope("full_attention", "global_rope_theta", 160_000.0),
        local_attention: v.get("local_attention").and_then(Value::as_u64).unwrap_or(128) as usize,
        local_rope_theta: rope("sliding_attention", "local_rope_theta", 10_000.0),
        classifier_config: None,
    })
}
