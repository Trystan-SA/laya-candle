//! The three decision primitives and how they render into option texts.
//!
//! A question is one of `choice` (pick a label), `score` (rate against ordered levels) or
//! `noul` (a calibrated boolean). Every variant is answered by scoring one `[MASK]` marker per
//! option, so the *rendering* of the options is part of the model input and has to be stable.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::pyjson;

/// The kind of decision a question asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    /// Pick exactly one label out of a named set.
    Choice,
    /// Rate the state against ordered descriptive levels.
    Score,
    /// A boolean question, answered as the probability that it holds.
    Noul,
}

impl QType {
    /// Index used by the model's type embedding. Must match the training order.
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }

    /// The word that opens the rendered question head.
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// One typed question.
///
/// `instructions` and `criteria` are kept as raw JSON because the reference implementation
/// accepts structured values there and renders them as JSON into the prompt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QType,
    pub instructions: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
}

impl Question {
    /// Start a `choice` question; add labels with [`ChoiceBuilder::option`].
    pub fn choice(instructions: impl Into<String>) -> ChoiceBuilder {
        ChoiceBuilder { instructions: instructions.into(), options: Map::new() }
    }

    /// Start a `score` question; add ordered levels with [`ScoreBuilder::level`].
    pub fn score(instructions: impl Into<String>) -> ScoreBuilder {
        ScoreBuilder { instructions: instructions.into(), levels: Vec::new() }
    }

    /// A `noul` (boolean) question.
    pub fn noul(instructions: impl Into<String>) -> NoulBuilder {
        NoulBuilder { instructions: instructions.into(), when_true: None, when_false: None }
    }

    /// The instruction text as the model sees it: strings pass through, anything else is JSON.
    pub fn instructions_text(&self) -> String {
        match &self.instructions {
            Value::String(s) => s.clone(),
            other => pyjson::dumps(other),
        }
    }

    /// The labels an answer can carry, in marker order.
    ///
    /// `choice` returns its criterion keys, `score` the level indices as strings and `noul`
    /// `["false", "true"]`.
    pub fn labels(&self, id: &str) -> Result<Vec<String>> {
        match self.kind {
            QType::Choice => Ok(self.choice_entries(id)?.into_iter().map(|(k, _)| k).collect()),
            QType::Score => Ok((0..self.score_levels(id)?.len()).map(|i| i.to_string()).collect()),
            QType::Noul => Ok(vec!["false".to_string(), "true".to_string()]),
        }
    }

    /// The option texts, one per `[MASK]` marker, in label order.
    ///
    /// This is a direct port of the reference `render_options`: the exact strings matter,
    /// because they are tokenised into the sequence the model scores.
    pub fn render_options(&self, id: &str) -> Result<Vec<String>> {
        match self.kind {
            QType::Choice => Ok(self
                .choice_entries(id)?
                .into_iter()
                .map(|(k, v)| match v {
                    // Only null and "" mean "no description": 0 and false are real criteria.
                    None => k,
                    Some(v) => format!("{k}: {}", render_criterion(&v)),
                })
                .collect()),
            QType::Score => Ok(self
                .score_levels(id)?
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect()),
            QType::Noul => {
                let crit = self.criteria.as_ref().and_then(Value::as_object);
                let side = |key: &str, fallback: &str| match crit.and_then(|c| c.get(key)) {
                    Some(v) if !is_blank(v) => format!("{key}: {}", render_criterion(v)),
                    _ => format!("{key}: {fallback}"),
                };
                Ok(vec![
                    side("false", "no, the statement does not hold"),
                    side("true", "yes, the statement holds"),
                ])
            }
        }
    }

    fn choice_entries(&self, id: &str) -> Result<Vec<(String, Option<Value>)>> {
        match self.criteria.as_ref() {
            // A bare list of labels is shorthand for "no description for any of them".
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok((s.clone(), None)),
                    other => Ok((pyjson::dumps(other), None)),
                })
                .collect(),
            Some(Value::Object(map)) => Ok(map
                .iter()
                .map(|(k, v)| (k.clone(), if is_blank(v) { None } else { Some(v.clone()) }))
                .collect()),
            _ => Err(Error::question(
                id,
                "a choice question needs `criteria`: either a map of label -> description or a list of labels",
            )),
        }
    }

    fn score_levels(&self, id: &str) -> Result<&Vec<Value>> {
        match self.criteria.as_ref() {
            Some(Value::Array(items)) if !items.is_empty() => Ok(items),
            _ => Err(Error::question(
                id,
                "a score question needs `criteria`: a non-empty list of ordered level descriptions",
            )),
        }
    }
}

fn is_blank(v: &Value) -> bool {
    matches!(v, Value::Null) || matches!(v, Value::String(s) if s.is_empty())
}

/// Render one criterion value: strings pass through, anything structured becomes JSON.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

/// An ordered set of questions, answered together in one forward pass.
///
/// Order is preserved end to end: it is the order the answers come back in, and questions are
/// batched as rows in the order they were inserted.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Questions(pub IndexMap<String, Question>);

impl Questions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a question under `id`, replacing any question already there.
    pub fn with(mut self, id: impl Into<String>, q: impl Into<Question>) -> Self {
        self.0.insert(id.into(), q.into());
        self
    }

    /// Add a question under `id` in place.
    pub fn insert(&mut self, id: impl Into<String>, q: impl Into<Question>) -> &mut Self {
        self.0.insert(id.into(), q.into());
        self
    }

    /// Parse the JSON schema used by the Python package: `{"<id>": {"type": ..., ...}}`.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| Error::json("questions", e))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> indexmap::map::Iter<'_, String, Question> {
        self.0.iter()
    }

    /// The question ids of this set, used by the router to recognise a known workflow.
    pub(crate) fn ids(&self) -> std::collections::BTreeSet<&str> {
        self.0.keys().map(String::as_str).collect()
    }
}

impl<'a> IntoIterator for &'a Questions {
    type Item = (&'a String, &'a Question);
    type IntoIter = indexmap::map::Iter<'a, String, Question>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromIterator<(String, Question)> for Questions {
    fn from_iter<T: IntoIterator<Item = (String, Question)>>(iter: T) -> Self {
        Questions(iter.into_iter().collect())
    }
}

// ----------------------------------------------------------------------- builders

/// Builder for a [`QType::Choice`] question.
pub struct ChoiceBuilder {
    instructions: String,
    options: Map<String, Value>,
}

impl ChoiceBuilder {
    /// Add a label with a description of when it applies.
    pub fn option(mut self, label: impl Into<String>, description: impl Into<String>) -> Self {
        self.options.insert(label.into(), Value::String(description.into()));
        self
    }

    /// Add a label that speaks for itself.
    pub fn bare_option(mut self, label: impl Into<String>) -> Self {
        self.options.insert(label.into(), Value::Null);
        self
    }
}

impl From<ChoiceBuilder> for Question {
    fn from(b: ChoiceBuilder) -> Self {
        Question {
            kind: QType::Choice,
            instructions: Value::String(b.instructions),
            criteria: Some(Value::Object(b.options)),
        }
    }
}

/// Builder for a [`QType::Score`] question.
pub struct ScoreBuilder {
    instructions: String,
    levels: Vec<Value>,
}

impl ScoreBuilder {
    /// Append the next level. Levels are ordered from 0 upwards.
    pub fn level(mut self, description: impl Into<String>) -> Self {
        self.levels.push(Value::String(description.into()));
        self
    }
}

impl From<ScoreBuilder> for Question {
    fn from(b: ScoreBuilder) -> Self {
        Question {
            kind: QType::Score,
            instructions: Value::String(b.instructions),
            criteria: Some(Value::Array(b.levels)),
        }
    }
}

/// Builder for a [`QType::Noul`] question.
pub struct NoulBuilder {
    instructions: String,
    when_true: Option<String>,
    when_false: Option<String>,
}

impl NoulBuilder {
    /// Describe what "true" means, instead of the generic wording.
    pub fn when_true(mut self, description: impl Into<String>) -> Self {
        self.when_true = Some(description.into());
        self
    }

    /// Describe what "false" means, instead of the generic wording.
    pub fn when_false(mut self, description: impl Into<String>) -> Self {
        self.when_false = Some(description.into());
        self
    }
}

impl From<NoulBuilder> for Question {
    fn from(b: NoulBuilder) -> Self {
        let mut crit = Map::new();
        if let Some(f) = b.when_false {
            crit.insert("false".into(), Value::String(f));
        }
        if let Some(t) = b.when_true {
            crit.insert("true".into(), Value::String(t));
        }
        Question {
            kind: QType::Noul,
            instructions: Value::String(b.instructions),
            criteria: if crit.is_empty() { None } else { Some(Value::Object(crit)) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn choice_options_render_label_and_description() {
        let q: Question = Question::choice("Which team?")
            .option("billing", "invoices, payments")
            .bare_option("other")
            .into();
        assert_eq!(
            q.render_options("dept").unwrap(),
            vec!["billing: invoices, payments".to_string(), "other".to_string()]
        );
        assert_eq!(q.labels("dept").unwrap(), vec!["billing", "other"]);
    }

    #[test]
    fn choice_criteria_order_is_preserved() {
        let q: Question =
            serde_json::from_str(r#"{"type":"choice","instructions":"x","criteria":{"z":null,"a":null,"m":null}}"#)
                .unwrap();
        assert_eq!(q.labels("q").unwrap(), vec!["z", "a", "m"]);
    }

    #[test]
    fn choice_accepts_a_bare_list_of_labels() {
        let q: Question =
            serde_json::from_str(r#"{"type":"choice","instructions":"x","criteria":["a","b"]}"#).unwrap();
        assert_eq!(q.render_options("q").unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn score_options_are_numbered_levels() {
        let q: Question = Question::score("How urgent?").level("not urgent").level("critical").into();
        assert_eq!(
            q.render_options("u").unwrap(),
            vec!["level 0: not urgent".to_string(), "level 1: critical".to_string()]
        );
    }

    #[test]
    fn noul_falls_back_to_generic_wording() {
        let q: Question = Question::noul("Does the user want a refund?").into();
        assert_eq!(
            q.render_options("r").unwrap(),
            vec![
                "false: no, the statement does not hold".to_string(),
                "true: yes, the statement holds".to_string(),
            ]
        );
    }

    #[test]
    fn noul_uses_its_own_wording_when_given() {
        let q: Question = Question::noul("Phishing?").when_true("a scam").when_false("legitimate").into();
        assert_eq!(
            q.render_options("p").unwrap(),
            vec!["false: legitimate".to_string(), "true: a scam".to_string()]
        );
    }

    #[test]
    fn a_structured_criterion_renders_as_json_not_a_debug_repr() {
        let q = Question {
            kind: QType::Choice,
            instructions: json!("x"),
            criteria: Some(json!({"billing": {"desc": "invoices", "examples": 2}})),
        };
        assert_eq!(
            q.render_options("q").unwrap(),
            vec![r#"billing: {"desc": "invoices", "examples": 2}"#.to_string()]
        );
    }

    #[test]
    fn a_choice_without_criteria_is_rejected() {
        let q = Question { kind: QType::Choice, instructions: json!("x"), criteria: None };
        assert!(q.render_options("dept").is_err());
    }
}
