//! Sending each request to the checkpoint that can actually read it.
//!
//! The English checkpoint does not gently degrade off English, it collapses: on 20-option
//! intent classification it scores 0.100 on Hindi and 0.103 on Korean against 0.050 for random
//! guessing — and it reports high confidence while doing so (ECE 0.855 on Hindi). Because it
//! stays confident while being wrong, confidence gating cannot save you. Script detection can,
//! and it costs microseconds before the forward pass.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use candle_core::Device;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Agent;
use crate::answer::Prediction;
use crate::checkpoint::{Checkpoint, hub_label};
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
            ModelSpec::Hub { repo, subfolder } => hub_label(repo, subfolder.as_deref()),
            ModelSpec::Dir(p) => p.display().to_string(),
        }
    }
}

/// One spec per checkpoint, indexed by `ModelName as usize`. Every checkpoint always has one.
type Specs = [ModelSpec; ModelName::ALL.len()];

/// The bundle repository: English at the root, every other checkpoint in a subfolder of its name.
fn default_models() -> Specs {
    ModelName::ALL.map(|name| ModelSpec::Hub {
        repo: BUNDLE_REPO.into(),
        subfolder: (name != ModelName::English).then(|| name.as_str().to_string()),
    })
}

/// Question-id signatures of the four typed-decisions workflows, each sorted.
///
/// A match must be exact, so an unrelated schema that happens to contain `urgency` is never
/// captured.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    ("agent_trace_observability", &["action", "needs_review", "outcome", "risk", "urgency"]),
    ("customer_service", &["action", "category", "churn_risk", "needs_human", "urgency"]),
    (
        "invoice_processing",
        &["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    ),
    (
        "security_incidents",
        &["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
    ),
];

/// Name of the typed-decisions workflow these question ids are, if any.
pub fn match_typed_decisions_workflow(questions: &Questions) -> Option<&'static str> {
    // Both sides are sorted, so an exact match is an element-wise one.
    let ids = questions.ids();
    TYPED_DECISION_WORKFLOWS
        .iter()
        .find(|(_, sig)| ids.iter().copied().eq(sig.iter().copied()))
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
    /// An older spelling of `model`: it takes a checkpoint name, not a workflow name.
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

/// The resident checkpoints, least recently used first, never more than `max`.
struct Loaded {
    agents: IndexMap<ModelName, Arc<Agent>>,
    max: usize,
}

impl Loaded {
    /// The resident agent for `name`, marked as just used.
    fn touch(&mut self, name: ModelName) -> Option<Arc<Agent>> {
        let agent = self.agents.shift_remove(&name)?;
        self.agents.insert(name, Arc::clone(&agent));
        Some(agent)
    }

    /// Make `agent` resident, evicting the least recently used beyond the cap.
    fn insert(&mut self, name: ModelName, agent: Arc<Agent>) {
        self.agents.insert(name, agent);
        while self.agents.len() > self.max {
            self.agents.shift_remove_index(0);
        }
    }
}

/// Lazily loads checkpoints and sends each request to the right one.
///
/// A cold checkpoint build costs seconds while detection costs microseconds, so at the default
/// `max_loaded` of 1 traffic that alternates languages rebuilds a model on *every* request. For
/// a server, [`preload`](Router::preload) instead.
pub struct Router {
    models: Specs,
    device: Device,
    default: ModelName,
    auto_task_detection: bool,
    /// Held only for bookkeeping, never across a build or a download.
    loaded: Mutex<Loaded>,
    /// One per checkpoint, held while it builds, so concurrent requests for the same
    /// checkpoint share one build while other checkpoints stay servable.
    building: [Mutex<()>; ModelName::ALL.len()],
}

/// Take a lock even if a thread panicked while holding it. Nothing here can be left half
/// updated by a panic, so one failed build must not take the whole router down with it.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Builder for a [`Router`].
pub struct RouterBuilder {
    models: Specs,
    /// `None` until set: `build` then resolves `LAYA_DEVICE`, so a bad value is an error.
    device: Option<Device>,
    max_loaded: usize,
    default: ModelName,
    auto_task_detection: bool,
    preload: Vec<ModelName>,
}

impl Default for RouterBuilder {
    fn default() -> Self {
        Self {
            models: default_models(),
            device: None,
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
        self.models[name as usize] = spec;
        self
    }

    /// Where every checkpoint runs. Unset, the router uses the device `LAYA_DEVICE` names, or
    /// the best available one; see [`crate::DeviceChoice`] to pick one by name.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
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
        let router = Router {
            models: self.models,
            device: match self.device {
                Some(d) => d,
                None => crate::device::device_from_env()?,
            },
            default: self.default,
            auto_task_detection: self.auto_task_detection,
            loaded: Mutex::new(Loaded { agents: IndexMap::new(), max: self.max_loaded }),
            building: Default::default(),
        };
        router.preload(self.preload)?;
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

    /// Where `name` is loaded from.
    fn spec(&self, name: ModelName) -> &ModelSpec {
        &self.models[name as usize]
    }

    /// Where every checkpoint runs.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The checkpoints currently resident, least recently used first.
    pub fn loaded(&self) -> Vec<ModelName> {
        lock(&self.loaded).agents.keys().copied().collect()
    }

    /// Build a checkpoint now, reusing it if it is already resident.
    ///
    /// Concurrent callers share one `Agent` rather than building duplicates, and a build never
    /// blocks requests for a checkpoint that is already resident.
    pub fn load(&self, name: ModelName) -> Result<Arc<Agent>> {
        if let Some(agent) = lock(&self.loaded).touch(name) {
            return Ok(agent);
        }

        let _building = lock(&self.building[name as usize]);
        // Another caller may have finished building it while this one waited.
        if let Some(agent) = lock(&self.loaded).touch(name) {
            return Ok(agent);
        }
        let agent = Arc::new(build(self.spec(name), &self.device)?);
        lock(&self.loaded).insert(name, Arc::clone(&agent));
        Ok(agent)
    }

    /// Build several checkpoints up front, raising the residency cap so that neither they nor
    /// any checkpoint already resident is evicted to make room.
    pub fn preload(&self, names: impl IntoIterator<Item = ModelName>) -> Result<()> {
        let names: Vec<_> = names.into_iter().collect();
        {
            let mut loaded = lock(&self.loaded);
            let mut wanted: std::collections::BTreeSet<_> = loaded.agents.keys().copied().collect();
            wanted.extend(names.iter().copied());
            loaded.max = loaded.max.max(wanted.len());
        }
        for name in names {
            self.load(name)?;
        }
        Ok(())
    }

    /// Free one checkpoint, or all of them.
    pub fn unload(&self, name: Option<ModelName>) {
        let mut loaded = lock(&self.loaded);
        match name {
            None => loaded.agents.clear(),
            Some(n) => {
                loaded.agents.shift_remove(&n);
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
        let decide =
            |model: ModelName, reason: String, detection, workflow: Option<&str>| RouteDecision {
                model,
                repo: self.spec(model).label(),
                reason,
                detection,
                workflow: workflow.map(str::to_string),
            };

        // `task` is an older spelling of `model`; both name a checkpoint.
        let forced = opts.model.as_deref().map(|m| ("model", m));
        if let Some((what, name)) = forced.or_else(|| opts.task.as_deref().map(|t| ("task", t))) {
            let m = ModelName::parse(name)?;
            return Ok(decide(m, format!("explicit {what}={name:?}"), None, None));
        }

        let workflow = match_typed_decisions_workflow(questions);
        if let Some(wf) = workflow.filter(|_| self.auto_task_detection) {
            return Ok(decide(
                ModelName::TypedDecisions,
                format!("question ids match the {wf:?} typed-decisions workflow"),
                None,
                workflow,
            ));
        }

        if let Some(lang) = &opts.lang {
            let head = lang.to_lowercase();
            // `en-US`, `en_US` and `en` all name English.
            let head = head.split(['-', '_']).next().unwrap_or("");
            let m = if matches!(head, "en" | "eng" | "english") {
                ModelName::English
            } else {
                ModelName::Multilingual
            };
            return Ok(decide(m, format!("explicit lang={lang:?}"), None, workflow));
        }

        let det = analyse(state);
        let (model, reason) = if det.script == "unknown" {
            (
                self.default,
                format!("no letters detected in state; using default ({})", self.default),
            )
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

fn build(spec: &ModelSpec, device: &Device) -> Result<Agent> {
    let cp = match spec {
        ModelSpec::Dir(path) => Checkpoint::from_dir(path)?,
        #[cfg(feature = "hub")]
        ModelSpec::Hub { repo, subfolder } => {
            Checkpoint::from_hub(repo, subfolder.as_deref(), &Default::default())?
        }
        #[cfg(not(feature = "hub"))]
        ModelSpec::Hub { repo, .. } => {
            return Err(Error::Checkpoint(format!(
                "{repo} is a Hugging Face Hub repository, but this build has the `hub` feature \
                 off. Enable it, or point the router at a local directory with `ModelSpec::Dir`."
            )));
        }
    };
    Agent::load(&cp, device)
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
            .route(
                &json!("We were billed twice and would like a refund"),
                &questions(),
                &Default::default(),
            )
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
    fn workflow_signatures_are_sorted() {
        // `match_typed_decisions_workflow` compares element-wise against sorted question ids.
        for (name, sig) in TYPED_DECISION_WORKFLOWS {
            assert!(sig.windows(2).all(|w| w[0] < w[1]), "{name}: {sig:?}");
        }
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

    #[test]
    fn an_explicit_lang_picks_the_checkpoint_without_detection() {
        for code in ["en-US", "en_US", "EN"] {
            let en =
                router().route(&json!("मुझसे"), &questions(), &RouteOptions::lang(code)).unwrap();
            assert_eq!(en.model, ModelName::English, "{code}");
            assert!(en.detection.is_none());
        }

        let de =
            router().route(&json!("hello there"), &questions(), &RouteOptions::lang("de")).unwrap();
        assert_eq!(de.model, ModelName::Multilingual);
        assert!(de.reason.contains("lang=\"de\""), "{}", de.reason);
    }

    #[test]
    fn task_is_an_alias_for_model() {
        let d = router()
            .route(&json!("x"), &questions(), &RouteOptions::task("typed_decisions"))
            .unwrap();
        assert_eq!(d.model, ModelName::TypedDecisions);
        assert!(d.reason.starts_with("explicit task="), "{}", d.reason);
        assert!(router().route(&json!("x"), &questions(), &RouteOptions::model("gpt")).is_err());
    }

    #[test]
    fn a_letterless_state_goes_to_the_configured_default() {
        let r = RouterBuilder::new().default_model(ModelName::Multilingual).build().unwrap();
        let d = r.route(&json!({"amount": 4411}), &questions(), &Default::default()).unwrap();
        assert_eq!(d.model, ModelName::Multilingual);
        assert!(d.reason.contains("no letters"), "{}", d.reason);
    }

    #[test]
    fn decisions_name_where_the_checkpoint_comes_from() {
        let r = RouterBuilder::new()
            .model(ModelName::English, ModelSpec::Dir("/opt/laya".into()))
            .build()
            .unwrap();
        let d = r.route(&json!("hello there friend"), &questions(), &Default::default()).unwrap();
        assert_eq!(d.repo, "/opt/laya");

        let d = r.route(&json!("मुझसे"), &questions(), &Default::default()).unwrap();
        assert_eq!(d.repo, format!("{BUNDLE_REPO}/multilingual"));
    }

    /// A router whose English and multilingual checkpoints are tiny ones on disk.
    fn tiny_router(dir: &std::path::Path) -> RouterBuilder {
        let english = dir.join("english");
        let multilingual = dir.join("multilingual");
        crate::testutil::write_tiny_checkpoint(&english);
        crate::testutil::write_tiny_checkpoint(&multilingual);
        RouterBuilder::new()
            .device(Device::Cpu)
            .model(ModelName::English, ModelSpec::Dir(english))
            .model(ModelName::Multilingual, ModelSpec::Dir(multilingual))
    }

    #[test]
    fn preloading_in_steps_keeps_what_is_already_resident() {
        let dir = crate::testutil::scratch_dir("router-preload");
        let r = tiny_router(&dir).preload([ModelName::English]).build().unwrap();
        r.preload([ModelName::Multilingual]).unwrap();
        assert_eq!(r.loaded(), vec![ModelName::English, ModelName::Multilingual]);

        // A later on-demand load reuses the resident agent rather than rebuilding it.
        let a = r.load(ModelName::English).unwrap();
        assert!(Arc::ptr_eq(&a, &r.load(ModelName::English).unwrap()));
    }

    #[test]
    fn a_panic_while_building_does_not_take_the_router_down() {
        let dir = crate::testutil::scratch_dir("router-poison");
        let r = tiny_router(&dir).preload([ModelName::English]).build().unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = r.loaded.lock().unwrap();
            let _building = r.building[ModelName::Multilingual as usize].lock().unwrap();
            panic!("a build blew up");
        }));
        assert!(r.loaded.is_poisoned());
        assert_eq!(r.loaded(), vec![ModelName::English]);
        r.load(ModelName::Multilingual).unwrap();
        r.unload(Some(ModelName::English));
        assert_eq!(r.loaded(), vec![ModelName::Multilingual]);
    }

    #[test]
    fn nothing_is_resident_until_a_load_is_asked_for() {
        let r = router();
        assert!(r.loaded().is_empty());
        r.unload(None);
        r.unload(Some(ModelName::English));
        assert!(r.loaded().is_empty());
    }
}
