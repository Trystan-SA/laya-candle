//! End-to-end tests against a real checkpoint.
//!
//! These are `#[ignore]`d: they need roughly 1.7 GB of weights and a slow CPU forward pass, so
//! `cargo test` stays offline and fast. Run them with:
//!
//! ```text
//! cargo test --release -- --ignored --nocapture
//! ```
//!
//! Set `LAYA_CHECKPOINT` to a local checkpoint directory to test that instead of the Hub.

use laya::{Agent, Question, Questions};
use serde_json::json;

fn agent() -> Agent {
    match std::env::var("LAYA_CHECKPOINT") {
        Ok(dir) => Agent::from_dir(dir).expect("loading LAYA_CHECKPOINT"),
        Err(_) => Agent::from_hub("convaiinnovations/laya", None).expect("loading from the Hub"),
    }
}

fn questions() -> Questions {
    Questions::new()
        .with(
            "department",
            Question::choice("Which department should handle this request?")
                .option("billing", "invoices, payments, refunds")
                .option("technical", "bugs, outages, system errors")
                .option("sales", "pricing, new contracts")
                .option("other", "everything else"),
        )
        .with(
            "urgency",
            Question::score("How urgent is this request?")
                .level("not urgent")
                .level("soon")
                .level("critical deadline or blocking issue"),
        )
        .with("churn_risk", Question::noul("Does the user threaten to cancel or leave?"))
        .with("refund_requested", Question::noul("Does the user explicitly request a refund?"))
}

#[test]
#[ignore = "downloads a checkpoint and runs a forward pass"]
fn answers_have_the_shape_the_questions_asked_for() {
    let agent = agent();
    let out = agent
        .predict(json!({"body": "We were billed twice for March. Please refund the duplicate."}), &questions())
        .unwrap();

    assert_eq!(out.answers.len(), 4);
    assert_eq!(out.usage.output_tokens, 0);
    assert!(out.usage.input_tokens > 0);

    let dept = out.get("department").unwrap();
    assert!(dept.as_choice().is_some());
    let total: f32 = ["billing", "technical", "sales", "other"]
        .iter()
        .map(|k| dept.probability(k).expect("every option carries a probability"))
        .sum();
    assert!((total - 1.0).abs() < 1e-2, "choice probabilities must sum to 1, got {total}");

    let urgency = out.get("urgency").unwrap().as_score().unwrap();
    assert!((0.0..=2.0).contains(&urgency), "a 3-level score lives in [0, 2], got {urgency}");

    for id in ["churn_risk", "refund_requested"] {
        let p = out.get(id).unwrap().as_noul().unwrap();
        assert!((0.0..=1.0).contains(&p), "{id} must be a probability, got {p}");
    }
}

/// The real test of a port: opposite states have to move the answers in opposite directions.
/// A forward pass wired up wrongly still returns well-formed numbers; it just stops discriminating.
#[test]
#[ignore = "downloads a checkpoint and runs a forward pass"]
fn opposite_states_separate() {
    let agent = agent();
    let questions = questions();

    let angry = agent
        .predict(
            json!({"subject": "Duplicate charge on invoice #4411",
                   "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan."}),
            &questions,
        )
        .unwrap();
    let happy = agent
        .predict(
            json!({"subject": "Thanks!",
                   "body": "Just wanted to say the new dashboard is great. No issues at all, keep it up."}),
            &questions,
        )
        .unwrap();

    println!("angry: {}", angry.to_json());
    println!("happy: {}", happy.to_json());

    assert_eq!(angry.get("department").unwrap().as_choice(), Some("billing"));

    let refund = |p: &laya::Prediction| p.get("refund_requested").unwrap().as_noul().unwrap();
    assert!(
        refund(&angry) > refund(&happy) + 0.3,
        "an explicit refund request must outscore a thank-you note: {} vs {}",
        refund(&angry),
        refund(&happy)
    );

    let churn = |p: &laya::Prediction| p.get("churn_risk").unwrap().as_noul().unwrap();
    assert!(
        churn(&angry) > churn(&happy) + 0.3,
        "a cancellation threat must outscore a thank-you note: {} vs {}",
        churn(&angry),
        churn(&happy)
    );

    let urgency = |p: &laya::Prediction| p.get("urgency").unwrap().as_score().unwrap();
    assert!(
        urgency(&angry) > urgency(&happy),
        "a same-day deadline must outscore a compliment: {} vs {}",
        urgency(&angry),
        urgency(&happy)
    );
}

/// The router must send a non-Latin state to the checkpoint that can read it, and the answer
/// must still be right.
#[test]
#[ignore = "downloads two checkpoints and runs two forward passes"]
fn the_router_reads_a_language_the_english_checkpoint_cannot() {
    use laya::{ModelName, Router};

    let router = Router::new().unwrap();
    let questions = questions();

    let hindi = router
        .predict(json!({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}), &questions)
        .unwrap();
    let routing = hindi.routing.as_ref().unwrap();
    assert_eq!(routing.model, ModelName::Multilingual, "{}", routing.reason);
    assert_eq!(hindi.get("department").unwrap().as_choice(), Some("billing"));
}
