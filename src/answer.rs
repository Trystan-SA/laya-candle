//! What a prediction returns.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::router::RouteDecision;

/// The auxiliary action head's read on a question.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Action {
    /// Probability the model would act on this answer rather than escalate it.
    pub act_probability: f32,
}

/// One answer, shaped by the primitive that was asked.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// The picked label, with a probability for every option.
    Choice { choice: String, probabilities: IndexMap<String, f32>, confidence: f32, action: Action },
    /// The expected level, with the level descriptions it was scored against.
    Score {
        score: f32,
        legend: IndexMap<String, Value>,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
        action: Action,
    },
    /// The probability the statement holds.
    Noul { noul: f32, confidence: f32, action: Action },
}

impl Answer {
    /// Calibrated confidence in this answer, between 0 and 1.
    ///
    /// For `choice` and `score` this is normalised entropy over the options; for `noul` it is
    /// the distance of the probability from a coin flip.
    pub fn confidence(&self) -> f32 {
        match self {
            Answer::Choice { confidence, .. }
            | Answer::Score { confidence, .. }
            | Answer::Noul { confidence, .. } => *confidence,
        }
    }

    /// The auxiliary action head's read.
    pub fn action(&self) -> Action {
        match self {
            Answer::Choice { action, .. }
            | Answer::Score { action, .. }
            | Answer::Noul { action, .. } => *action,
        }
    }

    /// The picked label, for a `choice` answer.
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Answer::Choice { choice, .. } => Some(choice),
            _ => None,
        }
    }

    /// The expected level, for a `score` answer.
    pub fn as_score(&self) -> Option<f32> {
        match self {
            Answer::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// The probability the statement holds, for a `noul` answer.
    pub fn as_noul(&self) -> Option<f32> {
        match self {
            Answer::Noul { noul, .. } => Some(*noul),
            _ => None,
        }
    }

    /// Probability assigned to one option, by label (or by level index for `score`).
    pub fn probability(&self, label: &str) -> Option<f32> {
        match self {
            Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
                probabilities.get(label).copied()
            }
            Answer::Noul { noul, .. } => match label {
                "true" => Some(*noul),
                "false" => Some(1.0 - *noul),
                _ => None,
            },
        }
    }
}

/// How much of the sequence budget a prediction used.
///
/// `output_tokens` is always zero: nothing is generated, which is the point.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// The result of one forward pass over a set of questions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    /// Identifier of the runtime that produced this.
    pub model: String,
    /// One answer per question, in the order the questions were given.
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    /// Present when the prediction came through a [`crate::Router`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RouteDecision>,
}

impl Prediction {
    /// The answer to one question.
    pub fn get(&self, id: &str) -> Option<&Answer> {
        self.answers.get(id)
    }

    /// Every question whose answer is below `threshold`, for confidence gating.
    ///
    /// The probabilities are trained against strictly proper scoring rules, so gating on them is
    /// meaningful — but only within a checkpoint's competence: the English checkpoint scores
    /// non-Latin scripts at chance *while staying confident*, which is what the router is for.
    pub fn below_confidence(&self, threshold: f32) -> Vec<&str> {
        self.answers
            .iter()
            .filter(|(_, a)| a.confidence() < threshold)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Pretty-printed JSON, in the same shape the Python package returns.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a Prediction always serialises")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn action() -> Action {
        Action { act_probability: 1.0 }
    }

    fn noul(p: f32) -> Answer {
        Answer::Noul { noul: p, confidence: p.max(1.0 - p), action: action() }
    }

    #[test]
    fn probabilities_are_read_by_label_for_every_primitive() {
        let choice = Answer::Choice {
            choice: "a".into(),
            probabilities: [("a".to_string(), 0.75), ("b".to_string(), 0.25)].into_iter().collect(),
            confidence: 0.5,
            action: action(),
        };
        assert_eq!(choice.as_choice(), Some("a"));
        assert_eq!(choice.probability("b"), Some(0.25));
        assert_eq!(choice.probability("zzz"), None);

        let n = noul(0.75);
        assert_eq!(n.as_noul(), Some(0.75));
        assert_eq!(n.as_choice(), None);
        assert_eq!(n.probability("true"), Some(0.75));
        assert_eq!(n.probability("false"), Some(0.25));
        assert_eq!(n.probability("maybe"), None);
    }

    #[test]
    fn below_confidence_lists_the_weak_answers_in_question_order() {
        let answers: IndexMap<String, Answer> =
            [("sure", noul(0.99)), ("meh", noul(0.55)), ("coin", noul(0.5))]
                .into_iter()
                .map(|(id, a)| (id.to_string(), a))
                .collect();
        let p = Prediction {
            model: "test".into(),
            answers,
            usage: Usage { input_tokens: 1, output_tokens: 0 },
            routing: None,
        };
        assert_eq!(p.below_confidence(0.9), vec!["meh", "coin"]);
        assert_eq!(p.get("sure").map(Answer::confidence), Some(0.99));
        assert!(p.get("nope").is_none());
    }

    #[test]
    fn json_carries_the_primitive_as_a_type_tag() {
        let v = serde_json::to_value(noul(0.75)).unwrap();
        assert_eq!(
            v,
            json!({"type": "noul", "noul": 0.75, "confidence": 0.75, "action": {"act_probability": 1.0}})
        );
    }
}
