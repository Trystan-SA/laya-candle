//! Checkpoint configuration: `rl_agent_config.json` and the encoder's `config.json`.

use std::collections::HashMap;
use std::path::Path;

use candle_transformers::models::modernbert;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result, read_json};
use crate::question::QType;

/// The decision-model side of a checkpoint, as stored in `rl_agent_config.json`.
///
/// Every field falls back to its [`Default`] when the file leaves it out.
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
    /// Finer calibration, keyed by `"<type>:<option-count bucket>"`.
    pub temperature_by_options: HashMap<String, f32>,
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
            temperature_by_options: HashMap::new(),
        }
    }
}

impl AgentConfig {
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
        pad_token_id: u64_or("pad_token_id", 0) as u32,
        global_attn_every_n_layers: u64_or("global_attn_every_n_layers", 3) as usize,
        global_rope_theta: rope("full_attention", "global_rope_theta", 160_000.0),
        local_attention: u64_or("local_attention", 128) as usize,
        local_rope_theta: rope("sliding_attention", "local_rope_theta", 10_000.0),
        classifier_config: None,
    })
}
