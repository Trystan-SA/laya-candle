//! A non-autoregressive decision engine: typed questions, answered in one forward pass.
//!
//! `laya` evaluates *typed questions* over any state — a string, an email, a ticket, a JSON
//! document — with a bidirectional encoder and a decision head. Nothing is generated, so there
//! is nothing to parse and nothing to hallucinate: every answer is one of three primitives,
//! carrying a probability distribution and a calibrated confidence.
//!
//! | primitive | question | answer |
//! |---|---|---|
//! | [`choice`](Question::choice) | pick one label from a named set | the label, plus a probability per option |
//! | [`score`](Question::score) | rate against ordered levels | the expected level, plus a probability per level |
//! | [`noul`](Question::noul) | a boolean | the probability the statement holds |
//!
//! A question set is answered in a single batched forward pass, with no decoding: each option
//! becomes a `[MASK]` marker and the answer is read off the marker's hidden state. Cost is set
//! by how long the input is, not by how much an autoregressive model would have had to write.
//!
//! # Quickstart
//!
//! ```no_run
//! use laya::{Question, Questions, Router};
//! use serde_json::json;
//!
//! let router = Router::builder().preload(laya::ModelName::ALL).build()?;
//!
//! let questions = Questions::new()
//!     .with("department", Question::choice("Which department should handle this?")
//!         .option("billing", "invoices, payments, refunds")
//!         .option("technical", "bugs, outages, system errors")
//!         .option("other", "everything else"))
//!     .with("urgency", Question::score("How urgent is this request?")
//!         .level("not urgent").level("soon").level("critical deadline"))
//!     .with("churn_risk", Question::noul("Does the user threaten to cancel or leave?"));
//!
//! let state = json!({
//!     "subject": "Duplicate charge on invoice #4411",
//!     "body": "We were billed twice for March. Refund the duplicate today or we cancel.",
//! });
//!
//! // The router detects the script and language first, then runs the checkpoint that can
//! // read it: the English checkpoint scores non-Latin scripts at chance while staying
//! // confident, so confidence gating alone cannot catch that mistake.
//! let out = router.predict(state, &questions)?;
//!
//! println!("{:?}", out.get("department").and_then(|a| a.as_choice()));
//! println!("{:?}", out.get("churn_risk").and_then(|a| a.as_noul()));
//! println!("{}", out.routing.as_ref().unwrap().reason);
//! # Ok::<(), laya::Error>(())
//! ```
//!
//! # One checkpoint, no routing
//!
//! ```no_run
//! # #[cfg(feature = "hub")]
//! # fn run() -> Result<(), laya::Error> {
//! use laya::{Agent, presets};
//! use serde_json::json;
//!
//! let agent = Agent::from_hub("convaiinnovations/laya", Some("multilingual"))?;
//! let out = agent.predict(json!({"message": "Ich wurde zweimal belastet"}), &presets::triage())?;
//! # Ok(())
//! # }
//! ```
//!
//! # Checkpoints
//!
//! | name | encoder | params | context | use it for |
//! |---|---|---|---|---|
//! | [`English`](ModelName::English) | ModernBERT-large | 421M | 512 | English |
//! | [`Multilingual`](ModelName::Multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, about 2x faster |
//! | [`TypedDecisions`](ModelName::TypedDecisions) | ModernBERT-large | 421M | 1024 | the four typed-decisions workflows |
//!
//! Weights are fetched from the Hugging Face Hub on first use and cached, or loaded from a
//! directory with [`Agent::from_dir`].
//!
//! # Performance
//!
//! Everything runs in f32 through [candle](https://github.com/huggingface/candle). On a plain
//! CPU build a prediction takes a second or more; enable the feature that matches your machine
//! — `mkl` on Intel, `accelerate` on macOS, `cuda` or `metal` for a GPU — before drawing any
//! conclusion about speed.

#![forbid(unsafe_code)]

pub mod agent;
pub mod answer;
pub mod calibration;
pub mod checkpoint;
pub mod config;
pub mod device;
pub mod error;
pub mod lang;
pub mod presets;
pub mod pyjson;
pub mod question;
pub mod router;

mod model;
mod sequence;
mod tokenizer;

pub use agent::Agent;
pub use answer::{Action, Answer, Prediction, Usage};
pub use checkpoint::Checkpoint;
pub use config::AgentConfig;
pub use device::default_device;
pub use error::{Error, Result};
pub use lang::{Detection, analyse as detect_language, detect_script, is_english};
pub use question::{QType, Question, Questions};
pub use router::{ModelName, ModelSpec, RouteDecision, RouteOptions, Router, RouterBuilder};

#[cfg(feature = "hub")]
pub use checkpoint::HubOptions;
