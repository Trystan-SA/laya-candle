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
use crate::error::{Error, Result};
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
    /// Load a checkpoint directory onto the device `LAYA_DEVICE` names, or the best available.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load(&Checkpoint::from_dir(dir)?, &crate::device::device_from_env()?)
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
        Self::from_hub_with(
            repo,
            subfolder,
            &Default::default(),
            &crate::device::device_from_env()?,
        )
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
        let config = AgentConfig::load(&cp.agent_config)?;
        let enc_cfg = load_encoder_config(&cp.encoder_config)?;
        let (tokenizer, special) =
            crate::tokenizer::load(&cp.tokenizer, cp.tokenizer_config.as_deref())?;

        // Read on the CPU first: `VarBuilder` moves each tensor to the target device as the
        // model asks for it, so a partially matching checkpoint never reaches VRAM.
        let weights = candle_core::safetensors::load(&cp.weights, &Device::Cpu)?;
        verify(&weights, &cp.label)?;
        verify_layout(&weights, &cp.label, config.head_layers, enc_cfg.num_hidden_layers)?;

        let model =
            DecisionModel::load(weights, &enc_cfg, config.head_layers, config.n_act(), device)?;

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
            return Err(Error::NoQuestions);
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
                    // The first of tied maxima, as numpy's `argmax` picks it.
                    let choice = item
                        .labels
                        .iter()
                        .zip(&p)
                        .reduce(|best, cur| if cur.1 > best.1 { cur } else { best })
                        .map(|(label, _)| label.clone())
                        .expect("a choice has at least one option");
                    Answer::Choice {
                        choice,
                        probabilities: distribution(&item.labels, &p),
                        confidence,
                        action,
                    }
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

/// Top-level tensors a checkpoint may hold besides [`REQUIRED_PREFIXES`] and `head.layers.*`:
/// the training-time temperature parameter, which inference reads from the config instead.
const OPTIONAL_KEYS: [&str; 1] = ["temperature"];

/// The layer indices under `prefix` (`"head.layers."` → the `N` of `head.layers.N.*`).
fn layer_indices<'a>(keys: impl Iterator<Item = &'a String>, prefix: &str) -> Vec<usize> {
    let mut found: Vec<usize> =
        keys.filter_map(|k| k.strip_prefix(prefix)?.split('.').next()?.parse().ok()).collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// Refuse a checkpoint whose tensors do not match its config, as the reference's
/// `load_state_dict(strict=True)` does: layers the model would never read, or tensors of no
/// known part, mean the config describes a different model than the weights.
fn verify_layout(
    weights: &HashMap<String, candle_core::Tensor>,
    label: &str,
    head_layers: usize,
    encoder_layers: usize,
) -> Result<()> {
    for (prefix, expected, source) in [
        ("head.layers.", head_layers, "head_layers in rl_agent_config.json"),
        ("encoder.layers.", encoder_layers, "num_hidden_layers in encoder/config.json"),
    ] {
        let found = layer_indices(weights.keys(), prefix);
        if found != (0..expected).collect::<Vec<_>>() {
            return Err(Error::Checkpoint(format!(
                "{label}: {source} says {expected} layers, but model.safetensors has {prefix}* \
                 layers {found:?}"
            )));
        }
    }

    let known = |k: &str| {
        OPTIONAL_KEYS.contains(&k)
            || k.starts_with("head.layers.")
            || REQUIRED_PREFIXES.iter().any(|p| k.starts_with(p))
    };
    let mut unexpected: Vec<_> = weights.keys().filter(|k| !known(k)).collect();
    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(Error::Checkpoint(format!(
            "{label}: model.safetensors holds tensors no part of the model reads: {unexpected:?}"
        )));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    fn weights(keys: &[&str]) -> HashMap<String, Tensor> {
        keys.iter()
            .map(|k| (k.to_string(), Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap()))
            .collect()
    }

    #[test]
    fn a_file_without_the_decision_head_is_refused_at_load() {
        let complete = [
            "encoder.embeddings.weight",
            "type_emb.weight",
            "scorer.1.weight",
            "act_head.0.weight",
        ];
        assert!(verify(&weights(&complete), "cp").is_ok());

        let err = verify(&weights(&complete[..3]), "cp").unwrap_err();
        assert!(
            matches!(err, Error::Checkpoint(ref m) if m.starts_with("cp:") && m.contains("act_head.")),
            "{err}"
        );
    }

    #[test]
    fn layers_the_config_does_not_describe_are_refused() {
        let keys = [
            "encoder.layers.0.attn.Wqkv.weight",
            "encoder.layers.1.attn.Wqkv.weight",
            "head.layers.0.linear1.weight",
            "head.layers.1.linear1.weight",
            "head.layers.2.linear1.weight",
            "type_emb.weight",
            "temperature",
        ];
        assert!(verify_layout(&weights(&keys), "cp", 3, 2).is_ok());

        let err = verify_layout(&weights(&keys), "cp", 2, 2).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("head_layers")), "{err}");
        let err = verify_layout(&weights(&keys), "cp", 3, 3).unwrap_err();
        assert!(
            matches!(err, Error::Checkpoint(ref m) if m.contains("num_hidden_layers")),
            "{err}"
        );

        let mut extra = keys.to_vec();
        extra.push("pooler.weight");
        let err = verify_layout(&weights(&extra), "cp", 3, 2).unwrap_err();
        assert!(matches!(err, Error::Checkpoint(ref m) if m.contains("pooler.weight")), "{err}");
    }

    #[test]
    fn a_tiny_checkpoint_answers_every_primitive() {
        use crate::question::Question;
        let dir = crate::testutil::scratch_dir("agent-e2e");
        crate::testutil::write_tiny_checkpoint(&dir);
        let agent = Agent::load(&Checkpoint::from_dir(&dir).unwrap(), &Device::Cpu).unwrap();

        let qs = Questions::new()
            .with("pick", Question::choice("pick one").bare_option("a").bare_option("b"))
            .with("level", Question::score("pick one").level("a").level("b").level("c"))
            .with("yes", Question::noul("the statement holds"));
        let out = agent.predict("hello world", &qs).unwrap();
        assert_eq!(out.answers.keys().collect::<Vec<_>>(), ["pick", "level", "yes"]);
        assert!(["a", "b"].contains(&out.get("pick").unwrap().as_choice().unwrap()));
        assert!((0.0..=2.0).contains(&out.get("level").unwrap().as_score().unwrap()));
        assert!((0.0..=1.0).contains(&out.get("yes").unwrap().as_noul().unwrap()));

        let err = agent.predict("hello", &Questions::new()).unwrap_err();
        assert!(matches!(err, Error::NoQuestions), "{err}");
    }

    #[test]
    fn a_distribution_is_rounded_and_keeps_label_order() {
        let d = distribution(&["b".to_string(), "a".to_string()], &[0.123456, 0.876544]);
        assert_eq!(d.keys().collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!(d["b"], 0.1235);
    }
}
