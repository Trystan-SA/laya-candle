//! The inference runtime: load a checkpoint, answer typed questions in one forward pass.

use std::collections::HashMap;
use std::path::Path;

use candle_core::Device;
use indexmap::IndexMap;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::answer::{Action, Answer, Prediction, Usage};
use crate::calibration::{confidence_from_probs, round4, softmax, temp_bucket};
use crate::checkpoint::Checkpoint;
use crate::config::{AgentConfig, load_encoder_config};
use crate::error::{Error, Result};
use crate::model::DecisionModel;
use crate::pyjson::serialize_state;
use crate::question::{QType, Questions};
use crate::sequence;
use crate::tokenizer::SpecialTokens;

/// Prefixes a usable checkpoint must define, so a mismatched file fails at load and not
/// halfway through a prediction.
const REQUIRED_PREFIXES: [&str; 4] = ["encoder.", "type_emb.", "scorer.", "act_head."];

/// A loaded checkpoint, ready to answer questions.
///
/// An `Agent` is immutable once built, so it is cheap to share across threads behind an `Arc`;
/// [`predict`](Agent::predict) takes `&self`.
pub struct Agent {
    model: DecisionModel,
    tokenizer: Tokenizer,
    special: SpecialTokens,
    config: AgentConfig,
    temperature: Vec<f32>,
    temperature_by_options: HashMap<String, f32>,
    clamped_temperatures: Vec<String>,
    label: String,
}

impl Agent {
    /// Load a checkpoint directory onto the best available device.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load(&Checkpoint::from_dir(dir)?, &crate::device::default_device())
    }

    /// Download a checkpoint from the Hugging Face Hub and load it.
    ///
    /// `subfolder` picks one checkpoint out of a repository that bundles several:
    ///
    /// ```no_run
    /// # use laya::Agent;
    /// let english = Agent::from_hub("convaiinnovations/laya", None)?;
    /// let multilingual = Agent::from_hub("convaiinnovations/laya", Some("multilingual"))?;
    /// # Ok::<(), laya::Error>(())
    /// ```
    ///
    #[cfg(feature = "hub")]
    pub fn from_hub(repo: &str, subfolder: Option<&str>) -> Result<Self> {
        Self::from_hub_with(repo, subfolder, &Default::default(), &crate::device::default_device())
    }

    /// [`from_hub`](Agent::from_hub), with an explicit token, progress bar and device.
    #[cfg(feature = "hub")]
    pub fn from_hub_with(
        repo: &str,
        subfolder: Option<&str>,
        opts: &crate::checkpoint::HubOptions,
        device: &Device,
    ) -> Result<Self> {
        Self::load(&Checkpoint::from_hub(repo, subfolder, opts)?, device)
    }

    /// Load a resolved checkpoint onto a specific device.
    pub fn load(cp: &Checkpoint, device: &Device) -> Result<Self> {
        let config = AgentConfig::from_file(&cp.agent_config)?;
        let enc_cfg = load_encoder_config(&cp.encoder_config)?;
        let (tokenizer, special) =
            crate::tokenizer::load(&cp.tokenizer, cp.tokenizer_config.as_deref())?;

        // Read on the CPU first: `VarBuilder` moves each tensor to the target device as the
        // model asks for it, so a partially matching checkpoint never reaches VRAM.
        let weights = candle_core::safetensors::load(&cp.weights, &Device::Cpu)?;
        verify(&weights, &cp.label)?;

        let model = DecisionModel::load(
            weights,
            &enc_cfg,
            config.head_layers,
            config.n_act(),
            device,
        )?;

        let (temperature, temperature_by_options, clamped_temperatures) = config.calibration();
        if !clamped_temperatures.is_empty() {
            eprintln!(
                "[laya] {}: this checkpoint ships temperatures outside [{}, {}] which would \
                 distort confidence; clamping {}. Treat confidence from the affected buckets as \
                 uncalibrated.",
                cp.label,
                crate::calibration::TEMP_MIN,
                crate::calibration::TEMP_MAX,
                clamped_temperatures.join(", ")
            );
        }

        Ok(Self {
            model,
            tokenizer,
            special,
            config,
            temperature,
            temperature_by_options,
            clamped_temperatures,
            label: cp.label.clone(),
        })
    }

    /// The checkpoint's configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Where the model runs.
    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// How this checkpoint names itself in errors and logs.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Calibration buckets whose shipped temperature had to be clamped.
    ///
    /// Confidence coming out of these buckets is not trustworthy; see
    /// [`crate::calibration::TEMP_MIN`].
    pub fn clamped_temperatures(&self) -> &[String] {
        &self.clamped_temperatures
    }

    /// Answer every question against `state`, in a single forward pass.
    ///
    /// `state` is any JSON: a string, or a document whose keys the instructions can refer to.
    ///
    /// ```no_run
    /// # use laya::{Agent, Question, Questions};
    /// # use serde_json::json;
    /// # let agent = Agent::from_dir("checkpoints/laya")?;
    /// let questions = Questions::new()
    ///     .with("refund", Question::noul("Does the user ask for money back?"))
    ///     .with("urgency", Question::score("How urgent is this?")
    ///         .level("not urgent").level("critical"));
    ///
    /// let out = agent.predict(json!({"body": "We were billed twice. Refund us today."}), &questions)?;
    /// println!("{:?}", out.get("refund").and_then(|a| a.as_noul()));
    /// # Ok::<(), laya::Error>(())
    /// ```
    pub fn predict(&self, state: impl Into<Value>, questions: &Questions) -> Result<Prediction> {
        self.predict_value(&state.into(), questions)
    }

    /// [`predict`](Agent::predict) against a state you already hold.
    pub fn predict_value(&self, state: &Value, questions: &Questions) -> Result<Prediction> {
        if questions.is_empty() {
            return Err(Error::Checkpoint("no questions to answer".into()));
        }
        let state_text = serialize_state(state);
        let max_len = self.config.max_len;
        let head_max_len = self.config.head_max_len;

        let items = questions
            .iter()
            .map(|(id, q)| {
                sequence::build(
                    &self.tokenizer,
                    &self.special,
                    &state_text,
                    id,
                    q,
                    max_len,
                    head_max_len,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let batch = sequence::collate(&items, self.special.pad_id);
        let out = self.model.forward(&batch)?;

        let mut answers = IndexMap::with_capacity(questions.len());
        for (row, (id, q)) in questions.iter().enumerate() {
            let k = items[row].markers.len();
            let scale = self
                .temperature_by_options
                .get(&temp_bucket(q.kind, k))
                .copied()
                .unwrap_or_else(|| self.temperature.get(q.kind.index()).copied().unwrap_or(1.0));

            let z: Vec<f32> = out.logits[row][..k].iter().map(|v| v / scale).collect();
            let p = softmax(&z);
            let confidence = round4(confidence_from_probs(&p, k));
            let action = Action { act_probability: round4(out.act_probs[row][0]) };

            let answer = match q.kind {
                QType::Choice => {
                    let labels = q.labels(id)?;
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    Answer::Choice {
                        choice: labels[best].clone(),
                        probabilities: labels
                            .iter()
                            .cloned()
                            .zip(p.iter().map(|v| round4(*v)))
                            .collect(),
                        confidence,
                        action,
                    }
                }
                QType::Score => {
                    let levels = q
                        .criteria
                        .as_ref()
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let score = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum::<f32>();
                    Answer::Score {
                        score: round4(score),
                        legend: levels
                            .into_iter()
                            .enumerate()
                            .map(|(i, c)| (i.to_string(), c))
                            .collect(),
                        probabilities: p
                            .iter()
                            .enumerate()
                            .map(|(i, v)| (i.to_string(), round4(*v)))
                            .collect(),
                        confidence,
                        action,
                    }
                }
                QType::Noul => {
                    let noul = p[1];
                    Answer::Noul {
                        noul: round4(noul),
                        confidence: round4(noul.max(1.0 - noul)),
                        action,
                    }
                }
            };
            answers.insert(id.clone(), answer);
        }

        Ok(Prediction {
            model: "laya-rl-agent".to_string(),
            answers,
            usage: Usage { input_tokens: batch.n_tokens, output_tokens: 0 },
            routing: None,
        })
    }
}

/// Fail early, and say what is actually wrong, when a file is not a decision checkpoint.
fn verify(weights: &HashMap<String, candle_core::Tensor>, label: &str) -> Result<()> {
    for prefix in REQUIRED_PREFIXES {
        if !weights.keys().any(|k| k.starts_with(prefix)) {
            return Err(Error::Checkpoint(format!(
                "{label}: model.safetensors has no {prefix}* parameters, so it is not a laya \
                 decision checkpoint (expected a ModernBERT encoder plus the typed decision head)."
            )));
        }
    }
    Ok(())
}
