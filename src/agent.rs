//! The inference runtime: load a checkpoint, answer typed questions in one forward pass.

use std::collections::HashMap;
use std::path::Path;

use candle_core::Device;
use indexmap::IndexMap;
use serde_json::Value;

use crate::answer::{Action, Answer, Prediction, Usage};
use crate::calibration::{Calibration, confidence_from_probs, round4, softmax};
use crate::checkpoint::Checkpoint;
use crate::config::{AgentConfig, load_encoder_config};
use crate::error::{Error, Result, read_json};
use crate::model::DecisionModel;
use crate::pyjson;
use crate::question::{QType, Questions};
use crate::sequence::Encoder;

/// Prefixes a usable checkpoint must define, so a mismatched file fails at load and not
/// halfway through a prediction.
const REQUIRED_PREFIXES: [&str; 4] = ["encoder.", "type_emb.", "scorer.", "act_head."];

/// A loaded checkpoint, ready to answer questions.
///
/// An `Agent` is immutable once built, so it is cheap to share across threads behind an `Arc`;
/// [`predict`](Agent::predict) takes `&self`.
pub struct Agent {
    model: DecisionModel,
    encoder: Encoder,
    config: AgentConfig,
    calibration: Calibration,
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
        let config: AgentConfig = read_json(&cp.agent_config)?;
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

        let calibration = Calibration::from_config(&config);
        if !calibration.clamped.is_empty() {
            eprintln!(
                "[laya] {}: this checkpoint ships temperatures outside [{}, {}] which would \
                 distort confidence; clamping {}. Treat confidence from the affected buckets as \
                 uncalibrated.",
                cp.label,
                crate::calibration::TEMP_MIN,
                crate::calibration::TEMP_MAX,
                calibration.clamped.join(", ")
            );
        }

        Ok(Self {
            model,
            encoder: Encoder::new(tokenizer, special, config.max_len, config.head_max_len),
            config,
            calibration,
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
        &self.calibration.clamped
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
        let state_ids = self.encoder.encode_state(&pyjson::render(state))?;
        let items = questions
            .iter()
            .map(|(id, q)| self.encoder.build(&state_ids, id, q))
            .collect::<Result<Vec<_>>>()?;

        let batch = self.encoder.collate(&items);
        let out = self.model.forward(&batch)?;

        let mut answers = IndexMap::with_capacity(questions.len());
        for (row, (id, q)) in questions.iter().enumerate() {
            let item = &items[row];
            let k = item.markers.len();
            let scale = self.calibration.temperature(q.kind, k);

            let z: Vec<f32> = out.logits[row][..k].iter().map(|v| v / scale).collect();
            let p = softmax(&z);
            let confidence = round4(confidence_from_probs(&p, k));
            let action = Action { act_probability: round4(out.act_probs[row][0]) };

            let answer = match q.kind {
                QType::Choice => {
                    let choice = item
                        .labels
                        .iter()
                        .zip(&p)
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(label, _)| label.clone())
                        .expect("a choice has at least one option");
                    Answer::Choice { choice, probabilities: distribution(&item.labels, &p), confidence, action }
                }
                QType::Score => {
                    let score = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum::<f32>();
                    // Building the sequence already validated the levels, so this cannot fail here.
                    let levels = q.score_levels(id)?;
                    Answer::Score {
                        score: round4(score),
                        legend: item.labels.iter().cloned().zip(levels.iter().cloned()).collect(),
                        probabilities: distribution(&item.labels, &p),
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

/// One rounded probability per label, in marker order.
fn distribution(labels: &[String], p: &[f32]) -> IndexMap<String, f32> {
    labels.iter().cloned().zip(p.iter().map(|v| round4(*v))).collect()
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
