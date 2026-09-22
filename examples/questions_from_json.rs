//! Questions as data: the same JSON schema the Python package and the `laya` CLI use.
//!
//! Question sets belong in configuration, not in a rebuild. They round-trip through serde, so a
//! file written for the Python package works here unchanged, and a set built in Rust can be
//! written back out for the CLI.
//!
//! ```text
//! cargo run --release --example questions_from_json
//! ```

mod common;

use laya::{Question, Questions, Router};
use serde_json::json;

fn main() -> Result<(), laya::Error> {
    let mut report = common::Report::new();

    // 1. Load a question set written for the Python package.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data/questions.json");
    let raw = std::fs::read_to_string(path).expect("examples/data/questions.json ships with the crate");
    let questions = Questions::from_json(&raw)?;

    println!("loaded {} questions from {path}", questions.len());
    for (id, q) in questions.iter() {
        println!("  {id:<18} {:<7} {} option(s)", q.kind.name(), q.render_options(id)?.len());
        for option in q.render_options(id)? {
            println!("      {option}");
        }
    }

    // 2. A set built in Rust serialises back to exactly that schema, so the CLI can read it:
    //    `laya predict -s @email.json -q @my-questions.json`
    let built = Questions::new()
        .with(
            "language_of_reply",
            Question::choice("Which language should the reply be written in?")
                .option("en", "English")
                .option("fr", "French")
                .option("de", "German"),
        )
        .with("needs_manager", Question::noul("Does this need a manager's sign-off?"));
    println!("\nbuilt in Rust, written back out as JSON:\n{}", serde_json::to_string_pretty(&built).unwrap());

    // 3. Both shapes run the same way.
    // A lazy router: nothing is downloaded or built until the first prediction needs it.
    let router = Router::new()?;
    report.lap("router built (checkpoints load on first use)");
    let state = json!({
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Please refund the duplicate today.",
    });

    let out = router.predict(state.clone(), &questions)?;
    report.lap("first prediction, including the checkpoint load");
    println!("\nfrom the file:");
    for (id, answer) in &out.answers {
        println!("  {id:<18} confidence {:.2}", answer.confidence());
    }

    let out = router.predict(state, &built)?;
    report.lap("second prediction, checkpoint already resident");
    println!("\nbuilt in Rust:");
    println!(
        "  language_of_reply  {}",
        out.get("language_of_reply").unwrap().as_choice().unwrap()
    );
    println!("  needs_manager      {:.3}", out.get("needs_manager").unwrap().as_noul().unwrap());

    report.finish();
    Ok(())
}
