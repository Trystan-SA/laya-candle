//! Any language, without choosing a model yourself.
//!
//! `Router` looks at the script and the language first — microseconds of pure Rust, no weights
//! touched — and then runs the checkpoint that can actually read the text. Use it whenever the
//! input is not guaranteed to be English.
//!
//! ```text
//! cargo run --release --example multilingual
//! ```

mod common;

use laya::{ModelName, Question, Questions, Router};
use serde_json::json;

const INBOX: [(&str, &str); 5] = [
    ("English", "I was charged twice this month, please refund the duplicate."),
    ("French", "J'ai été débité deux fois ce mois-ci, merci de me rembourser le doublon."),
    ("German", "Guten Tag, wir möchten auf den Enterprise-Tarif wechseln. Bitte ein Angebot."),
    ("Hindi", "इस महीने मुझसे दो बार शुल्क लिया गया, कृपया अतिरिक्त राशि वापस करें।"),
    ("Japanese", "今月は二重に請求されました。重複分を返金してください。"),
];

fn main() -> Result<(), laya::Error> {
    let mut report = common::Report::new();

    // Preload if you serve mixed traffic. At the default residency of one checkpoint, every
    // language switch would rebuild a model — seconds per request instead of milliseconds.
    let router = Router::builder()
        .preload([ModelName::English, ModelName::Multilingual])
        .build()?;
    report.lap("2 checkpoints resident");

    let questions = Questions::new().with(
        "department",
        Question::choice("Which department should handle this request?")
            .option("billing", "invoices, payments, refunds")
            .option("technical", "bugs, outages, system errors")
            .option("sales", "pricing, new contracts")
            .option("other", "everything else"),
    );

    println!();
    for (language, text) in INBOX {
        // Ask what would happen, without running anything. Useful for logging and for tests.
        let plan = router.route(&json!({ "body": text }), &questions, &Default::default())?;
        println!("  {language:<9} {:<13} {}", plan.model, plan.reason);

        // Then actually answer. `routing` records the same decision on the result.
        let out = router.predict(json!({ "body": text }), &questions)?;
        let answer = out.get("department").unwrap();
        println!(
            "            -> {} ({:.0}% confident)\n",
            answer.as_choice().unwrap(),
            100.0 * answer.confidence()
        );
    }
    report.lap("routed and answered 5 requests");

    // Override the router when you already know: from an Accept-Language header, a user
    // setting, or a field on the record.
    let forced = router.predict_with(
        &json!({ "body": "Merci, tout fonctionne parfaitement." }),
        &questions,
        &laya::RouteOptions::lang("fr"),
    )?;
    println!("  forced lang=fr -> {}", forced.routing.unwrap().reason);

    report.finish();
    Ok(())
}
