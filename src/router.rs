//! Sending each request to the checkpoint that can actually read it.
//!
//! The English checkpoint does not gently degrade off English, it collapses: on 20-option
//! intent classification it scores 0.100 on Hindi and 0.103 on Korean against 0.050 for random
//! guessing — and it reports high confidence while doing so (ECE 0.855 on Hindi). Because it
//! stays confident while being wrong, confidence gating cannot save you. Script detection can,
//! and it costs microseconds before the forward pass.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Agent;
use crate::answer::Prediction;
use crate::error::{Error, Result};
use crate::lang::{Detection, analyse};
use crate::question::Questions;

/// The Hub repository that bundles all three checkpoints.
pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

/// The three checkpoints the router chooses between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelName {
    /// ModernBERT-large, 421M, 512 tokens. English.
    English,
    /// mmBERT-base, 322M, 1024 tokens. 100+ languages, and about twice as fast.
    Multilingual,
    /// ModernBERT-large, 421M, 1024 tokens, fine-tuned on four synthetic workflows.
    TypedDecisions,
}

impl ModelName {
    /// All three, in the order they are preloaded.
    pub const ALL: [ModelName; 3] =
        [ModelName::English, ModelName::Multilingual, ModelName::TypedDecisions];

    pub fn as_str(self) -> &'static str {
        match self {
            ModelName::English => "english",
            ModelName::Multilingual => "multilingual",
            ModelName::TypedDecisions => "typed-decisions",
        }
    }

    /// Resolve a name, accepting the aliases people are likely to type.
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_lowercase().replace('_', "-").as_str() {
            "english" | "en" | "laya" | "default" => Ok(ModelName::English),
            "multilingual" | "multi" | "ml" | "laya-multilingual" => Ok(ModelName::Multilingual),
            "typed-decisions" | "typed" | "decisions" | "laya-typed-decisions" => {
                Ok(ModelName::TypedDecisions)
            }
            other => Err(Error::UnknownModel(format!(
                "unknown model {other:?}; choose english, multilingual or typed-decisions"
            ))),
        }
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `pad`, not `write_str`: a name printed in a table needs its width honoured.
        f.pad(self.as_str())
    }
}

/// Where a checkpoint comes from.
#[derive(Clone, Debug)]
pub enum ModelSpec {
    /// A Hub repository, optionally a subfolder inside it.
    Hub { repo: String, subfolder: Option<String> },
    /// A checkpoint directory on disk.
    Dir(PathBuf),
}

impl ModelSpec {
    /// How this spec reads in a routing decision.
    pub fn label(&self) -> String {
        match self {
            ModelSpec::Hub { repo, subfolder: Some(s) } => format!("{repo}/{s}"),
            ModelSpec::Hub { repo, subfolder: None } => repo.clone(),
            ModelSpec::Dir(p) => p.display().to_string(),
        }
    }
}

fn default_models() -> HashMap<ModelName, ModelSpec> {
    HashMap::from([
        (ModelName::English, ModelSpec::Hub { repo: BUNDLE_REPO.into(), subfolder: None }),
        (
            ModelName::Multilingual,
            ModelSpec::Hub { repo: BUNDLE_REPO.into(), subfolder: Some("multilingual".into()) },
        ),
        (
            ModelName::TypedDecisions,
            ModelSpec::Hub { repo: BUNDLE_REPO.into(), subfolder: Some("typed-decisions".into()) },
        ),
    ])
}

/// Question-id signatures of the four typed-decisions workflows.
///
/// A match must be exact, so an unrelated schema that happens to contain `urgency` is never
/// captured.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    ("agent_trace_observability", &["action", "needs_review", "outcome", "risk", "urgency"]),
    ("customer_service", &["action", "category", "churn_risk", "needs_human", "urgency"]),
    ("invoice_processing", &["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
    ("security_incidents", &["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
];

/// Name of the typed-decisions workflow these question ids are, if any.
pub fn match_typed_decisions_workflow(questions: &Questions) -> Option<&'static str> {
    let ids = questions.ids();
    TYPED_DECISION_WORKFLOWS
        .iter()
        .find(|(_, sig)| ids == sig.iter().copied().collect::<BTreeSet<_>>())
        .map(|(name, _)| *name)
}

/// Which checkpoint was chosen, and why.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteDecision {
    pub model: ModelName,
    /// Where that checkpoint comes from.
    pub repo: String,
    /// Plain-language justification, safe to log or return in an API response.
    pub reason: String,
    /// The detection behind the decision; absent when the caller forced the choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detection: Option<Detection>,
    /// The typed-decisions workflow the question ids matched, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<String>,
}

/// Caller overrides for one routing decision.
#[derive(Clone, Debug, Default)]
pub struct RouteOptions {
    /// Force a checkpoint by name.
    pub model: Option<String>,
    /// Force the typed-decisions checkpoint by naming its task.
    pub task: Option<String>,
    /// Declare the language instead of detecting it.
    pub lang: Option<String>,
}

impl RouteOptions {
    pub fn model(name: impl Into<String>) -> Self {
        Self { model: Some(name.into()), ..Default::default() }
    }
    pub fn lang(code: impl Into<String>) -> Self {
        Self { lang: Some(code.into()), ..Default::default() }
    }
    pub fn task(name: impl Into<String>) -> Self {
        Self { task: Some(name.into()), ..Default::default() }
    }
}

#[derive(Default)]
struct Loaded {
    agents: HashMap<ModelName, Arc<Agent>>,
    /// Least recently used first.
    order: Vec<ModelName>,
}

/// Lazily loads checkpoints and sends each request to the right one.
///
/// A cold checkpoint build costs seconds while detection costs microseconds, so at the default
/// `max_loaded` of 1 traffic that alternates languages rebuilds a model on *every* request. For
/// a server, [`preload`](Router::preload) instead.
pub struct Router {
    models: HashMap<ModelName, ModelSpec>,
    device: Device,
    max_loaded: Mutex<usize>,
    default: ModelName,
    auto_task_detection: bool,
    loaded: Mutex<Loaded>,
}

/// Builder for a [`Router`].
pub struct RouterBuilder {
    models: HashMap<ModelName, ModelSpec>,
    device: Device,
    max_loaded: usize,
    default: ModelName,
    auto_task_detection: bool,
    preload: Vec<ModelName>,
}

impl Default for RouterBuilder {
    fn default() -> Self {
        Self {
            models: default_models(),
            device: crate::device::default_device(),
            max_loaded: 1,
            default: ModelName::English,
            auto_task_detection: false,
            preload: Vec::new(),
        }
    }
}

impl RouterBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point one checkpoint somewhere else — a local directory, or your own Hub repository.
    pub fn model(mut self, name: ModelName, spec: ModelSpec) -> Self {
        self.models.insert(name, spec);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How many checkpoints stay resident. The least recently used is evicted beyond this.
    pub fn max_loaded(mut self, n: usize) -> Self {
        self.max_loaded = n.max(1);
        self
    }

    /// Which checkpoint answers a state with no letters in it.
    pub fn default_model(mut self, name: ModelName) -> Self {
        self.default = name;
        self
    }

    /// Let question ids select the `typed-decisions` checkpoint.
    ///
    /// Off by default: that checkpoint is fine-tuned on four specific synthetic workflows and
    /// should not become a silent default for anything that happens to share their question ids.
    pub fn auto_task_detection(mut self, on: bool) -> Self {
        self.auto_task_detection = on;
        self
    }

    /// Build these checkpoints up front, so no request ever pays a model load.
    pub fn preload(mut self, names: impl IntoIterator<Item = ModelName>) -> Self {
        self.preload.extend(names);
        self
    }

    pub fn build(self) -> Result<Router> {
        let max_loaded = self.max_loaded.max(self.preload.len()).max(1);
        let router = Router {
            models: self.models,
            device: self.device,
            max_loaded: Mutex::new(max_loaded),
            default: self.default,
            auto_task_detection: self.auto_task_detection,
            loaded: Mutex::new(Loaded::default()),
        };
        for name in self.preload {
            router.load(name)?;
        }
        Ok(router)
    }
}

impl Router {
    /// A router over the three published checkpoints, loading them on demand.
    pub fn new() -> Result<Self> {
        RouterBuilder::new().build()
    }

    pub fn builder() -> RouterBuilder {
        RouterBuilder::new()
    }

    /// The checkpoints currently resident, least recently used first.
    pub fn loaded(&self) -> Vec<ModelName> {
        self.loaded.lock().unwrap().order.clone()
    }

    /// Build a checkpoint now, reusing it if it is already resident.
    ///
    /// Concurrent callers share one `Agent` rather than building duplicates.
    pub fn load(&self, name: ModelName) -> Result<Arc<Agent>> {
        let mut loaded = self.loaded.lock().unwrap();
        if let Some(agent) = loaded.agents.get(&name) {
            let agent = Arc::clone(agent);
            touch(&mut loaded, name);
            return Ok(agent);
        }

        let spec = self
            .models
            .get(&name)
            .ok_or_else(|| Error::UnknownModel(format!("{name} is not configured on this router")))?;
        let agent = Arc::new(build(spec, &self.device)?);
        loaded.agents.insert(name, Arc::clone(&agent));
        loaded.order.push(name);
        let max = *self.max_loaded.lock().unwrap();
        evict(&mut loaded, max);
        Ok(agent)
    }

    /// Build several checkpoints up front, raising the residency cap to fit them.
    pub fn preload(&self, names: impl IntoIterator<Item = ModelName>) -> Result<()> {
        let names: Vec<_> = names.into_iter().collect();
        {
            let mut max = self.max_loaded.lock().unwrap();
            *max = (*max).max(names.len());
        }
        for name in names {
            self.load(name)?;
        }
        Ok(())
    }

    /// Free one checkpoint, or all of them.
    pub fn unload(&self, name: Option<ModelName>) {
        let mut loaded = self.loaded.lock().unwrap();
        match name {
            None => {
                loaded.agents.clear();
                loaded.order.clear();
            }
            Some(n) => {
                loaded.agents.remove(&n);
                loaded.order.retain(|k| *k != n);
            }
        }
    }

    /// Decide which checkpoint to use, without loading or running anything.
    ///
    /// Precedence: explicit `model`, then explicit `task`, then a detected workflow (opt-in),
    /// then explicit `lang`, then the detected script and language, then the default.
    pub fn route(
        &self,
        state: &Value,
        questions: &Questions,
        opts: &RouteOptions,
    ) -> Result<RouteDecision> {
        let decide = |model: ModelName, reason: String, detection, workflow| RouteDecision {
            model,
            repo: self.models.get(&model).map(ModelSpec::label).unwrap_or_default(),
            reason,
            detection,
            workflow,
        };

        if let Some(name) = &opts.model {
            let m = ModelName::parse(name)?;
            return Ok(decide(m, format!("explicit model={name:?}"), None, None));
        }
        if let Some(task) = &opts.task {
            let normalised = task.to_lowercase().replace('-', "_");
            let m = if normalised == "typed_decisions" {
                ModelName::TypedDecisions
            } else {
                ModelName::parse(task)?
            };
            return Ok(decide(m, format!("explicit task={task:?}"), None, None));
        }

        let workflow = match_typed_decisions_workflow(questions).map(str::to_string);
        if self.auto_task_detection && let Some(wf) = &workflow {
            return Ok(decide(
                ModelName::TypedDecisions,
                format!("question ids match the {wf:?} typed-decisions workflow"),
                None,
                workflow.clone(),
            ));
        }

        if let Some(lang) = &opts.lang {
            let head = lang.to_lowercase();
            let head = head.split('-').next().unwrap_or("");
            let m = if matches!(head, "en" | "eng" | "english") {
                ModelName::English
            } else {
                ModelName::Multilingual
            };
            return Ok(decide(m, format!("explicit lang={lang:?}"), None, workflow));
        }

        let det = analyse(state);
        let (model, reason) = if det.script == "unknown" {
            (self.default, format!("no letters detected in state; using default ({})", self.default))
        } else if det.script != "latin" {
            (
                ModelName::Multilingual,
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            let reason = match &det.language {
                Some(lg) => format!("Latin script but language looks like {lg:?}, not English"),
                // Unidentified Latin-script language: routed on the non-English letters alone,
                // because no stopword list here covers it.
                None => format!(
                    "Latin script, language not identified but {:.0}% non-English letters; not \
                     safe for the English checkpoint",
                    100.0 * det.diacritic_rate
                ),
            };
            (ModelName::Multilingual, reason)
        } else {
            (ModelName::English, "English Latin text".to_string())
        };

        Ok(decide(model, reason, Some(det), workflow))
    }

    /// Route, then answer every question in one forward pass on the chosen checkpoint.
    pub fn predict(&self, state: impl Into<Value>, questions: &Questions) -> Result<Prediction> {
        self.predict_with(&state.into(), questions, &RouteOptions::default())
    }

    /// [`predict`](Router::predict), with caller overrides.
    pub fn predict_with(
        &self,
        state: &Value,
        questions: &Questions,
        opts: &RouteOptions,
    ) -> Result<Prediction> {
        let decision = self.route(state, questions, opts)?;
        let agent = self.load(decision.model)?;
        let mut out = agent.predict_value(state, questions)?;
        out.routing = Some(decision);
        Ok(out)
    }
}

fn touch(loaded: &mut Loaded, name: ModelName) {
    loaded.order.retain(|k| *k != name);
    loaded.order.push(name);
}

fn evict(loaded: &mut Loaded, max_loaded: usize) {
    while loaded.order.len() > max_loaded {
        let victim = loaded.order.remove(0);
        loaded.agents.remove(&victim);
    }
}

fn build(spec: &ModelSpec, device: &Device) -> Result<Agent> {
    match spec {
        ModelSpec::Dir(path) => Agent::load(&crate::checkpoint::Checkpoint::from_dir(path)?, device),
        #[cfg(feature = "hub")]
        ModelSpec::Hub { repo, subfolder } => Agent::from_hub_with(
            repo,
            subfolder.as_deref(),
            &Default::default(),
            device,
        ),
        #[cfg(not(feature = "hub"))]
        ModelSpec::Hub { repo, .. } => Err(Error::Checkpoint(format!(
            "{repo} is a Hugging Face Hub repository, but this build has the `hub` feature off. \
             Enable it, or point the router at a local directory with `ModelSpec::Dir`."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::question::Question;
    use serde_json::json;

    fn router() -> Router {
        // Routing decisions never touch the weights, so nothing is loaded here.
        RouterBuilder::new().build().unwrap()
    }

    fn questions() -> Questions {
        Questions::new().with("urgent", Question::noul("Is this urgent?"))
    }

    #[test]
    fn a_non_latin_script_goes_to_the_multilingual_checkpoint() {
        let d = router()
            .route(&json!({"body": "मुझसे दो बार शुल्क लिया गया"}), &questions(), &Default::default())
            .unwrap();
        assert_eq!(d.model, ModelName::Multilingual);
        assert!(d.reason.contains("devanagari"), "{}", d.reason);
    }

    #[test]
    fn english_goes_to_the_english_checkpoint() {
        let d = router()
            .route(&json!("We were billed twice and would like a refund"), &questions(), &Default::default())
            .unwrap();
        assert_eq!(d.model, ModelName::English);
    }

    #[test]
    fn an_explicit_model_wins_over_detection() {
        let d = router()
            .route(&json!("मुझसे दो बार शुल्क लिया गया"), &questions(), &RouteOptions::model("english"))
            .unwrap();
        assert_eq!(d.model, ModelName::English);
        assert!(d.detection.is_none());
    }

    #[test]
    fn a_workflow_is_only_selected_when_opted_in() {
        let wf: Questions = ["action", "category", "churn_risk", "needs_human", "urgency"]
            .into_iter()
            .map(|id| (id.to_string(), Question::noul("x").into()))
            .collect();
        assert_eq!(match_typed_decisions_workflow(&wf), Some("customer_service"));

        let off = router().route(&json!("hello there friend"), &wf, &Default::default()).unwrap();
        assert_eq!(off.model, ModelName::English);
        assert_eq!(off.workflow.as_deref(), Some("customer_service"));

        let on = RouterBuilder::new().auto_task_detection(true).build().unwrap();
        let d = on.route(&json!("hello there friend"), &wf, &Default::default()).unwrap();
        assert_eq!(d.model, ModelName::TypedDecisions);
    }

    #[test]
    fn a_model_name_honours_format_width() {
        assert_eq!(format!("[{:<14}]", ModelName::English), "[english       ]");
        assert_eq!(format!("[{:>14}]", ModelName::TypedDecisions), "[typed-decisions]");
    }

    #[test]
    fn aliases_resolve() {
        assert_eq!(ModelName::parse("EN").unwrap(), ModelName::English);
        assert_eq!(ModelName::parse("typed_decisions").unwrap(), ModelName::TypedDecisions);
        assert!(ModelName::parse("gpt").is_err());
    }
}
