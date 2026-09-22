//! The three primitives, side by side, with the full distribution behind each answer.
//!
//! A decision model does not return a word — it returns a distribution. That is what makes the
//! answers gateable, and it is the part a generated string throws away.
//!
//! ```text
//! cargo run --release --example primitives
//! ```

mod common;

use laya::{Agent, Answer, Question, Questions};
use serde_json::json;

fn main() -> Result<(), laya::Error> {
    let mut report = common::Report::new();
    let agent = Agent::from_hub("convaiinnovations/laya", None)?;
    report.lap("checkpoint ready");

    let state = json!({
        "subject": "Still waiting",
        "body": "This is the third time I write about the double charge. Nobody answers. \
                 Refund me today or I am done with you.",
    });

    let questions = Questions::new()
        // choice — pick one label out of a named set. The criteria are read by the model, so
        // they are part of the question, not documentation.
        .with(
            "intent",
            Question::choice("What does the customer want?")
                .option("refund", "money returned or a duplicate charge reversed")
                .option("technical_help", "a bug, outage or integration problem")
                .option("information", "general information, pricing or how-to")
                .option("cancellation", "wants to cancel or downgrade"),
        )
        // score — an ordinal rubric. The answer is the *expected* level, so 2.4 means "between
        // annoyed and furious, closer to annoyed".
        .with(
            "frustration",
            Question::score("How frustrated does the customer sound?")
                .level("calm and neutral")
                .level("concerned but civil")
                .level("clearly annoyed")
                .level("very angry or using strong language"),
        )
        // noul — a boolean, answered as a probability rather than a yes or a no.
        .with("refund_requested", Question::noul("Does the customer ask for money back?"))
        // A noul can describe its own sides when the generic wording is too vague.
        .with(
            "is_repeat_contact",
            Question::noul("Has the customer written about this before?")
                .when_true("they refer to an earlier unanswered message")
                .when_false("this is the first contact about the issue"),
        );

    let out = agent.predict(state, &questions)?;
    report.lap(&format!("answered {} questions", questions.len()));
    println!("\n{} input tokens, {} generated\n", out.usage.input_tokens, out.usage.output_tokens);

    for (id, answer) in &out.answers {
        match answer {
            Answer::Choice { choice, probabilities, confidence, .. } => {
                println!("{id}  [choice]  -> {choice}   confidence {confidence:.3}");
                for (label, p) in probabilities {
                    println!("      {label:<16} {p:.4} {}", bar(*p));
                }
            }
            Answer::Score { score, legend, probabilities, confidence, .. } => {
                let top = legend.len().saturating_sub(1);
                println!("{id}  [score]   -> {score:.2} / {top}   confidence {confidence:.3}");
                for (level, p) in probabilities {
                    let text = legend.get(level).and_then(|v| v.as_str()).unwrap_or("");
                    println!("      level {level}: {text:<38} {p:.4} {}", bar(*p));
                }
            }
            Answer::Noul { noul, confidence, .. } => {
                println!("{id}  [noul]    -> {noul:.4}   confidence {confidence:.3}");
                println!("      true             {noul:.4} {}", bar(*noul));
                println!("      false            {:.4} {}", 1.0 - noul, bar(1.0 - noul));
            }
        }
        // The auxiliary action head reads the answer's own shape and says whether it would act
        // on it or hand it over.
        println!("      act_probability {:.3}\n", answer.action().act_probability);
    }

    println!("note: `confidence` for choice and score is 1 - H(p)/ln(k), so a two-way tie reads 0");
    println!("      and a decided answer reads near 1. For noul it is the distance from 0.5.");

    report.finish();
    Ok(())
}

fn bar(p: f32) -> String {
    "#".repeat((p * 30.0).round() as usize)
}
