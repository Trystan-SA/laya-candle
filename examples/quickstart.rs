//! Your first prediction: load a checkpoint, ask three questions, read the answers.
//!
//! ```text
//! cargo run --release --example quickstart
//! ```

mod common;

use laya::{Agent, Question, Questions};
use serde_json::json;

fn main() -> Result<(), laya::Error> {
    let mut report = common::Report::new();

    // Downloaded on first use and cached by hf-hub; later runs load from disk.
    let agent = Agent::from_hub("convaiinnovations/laya", None)?;
    report.lap("checkpoint ready");

    // A question set. The criteria are read by the model, so write them for the model:
    // short, concrete, and distinct from each other.
    let questions = Questions::new()
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
        .with("churn_risk", Question::noul("Does the user threaten to cancel or leave?"));

    // The state is any JSON. Keys are visible to the model, so name them the way your
    // instructions refer to them.
    let email = json!({
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
    });

    // One call, one forward pass, all three answers.
    let out = agent.predict(email, &questions)?;
    report.lap("answered 3 questions");

    // Each primitive has its own accessor, and every answer carries a confidence.
    let department = out.get("department").unwrap();
    println!("\n  department  {}", department.as_choice().unwrap());
    println!("              confidence {:.2}", department.confidence());
    println!("              billing at {:.2}, technical at {:.2}",
        department.probability("billing").unwrap(),
        department.probability("technical").unwrap());

    println!("  urgency     {:.2} / 2", out.get("urgency").unwrap().as_score().unwrap());
    println!("  churn_risk  {:.3}", out.get("churn_risk").unwrap().as_noul().unwrap());
    println!("\n  {} input tokens, {} generated", out.usage.input_tokens, out.usage.output_tokens);

    report.finish();
    Ok(())
}
