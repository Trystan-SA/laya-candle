//! A support desk triaging its inbox.
//!
//! Every ticket is read once — one forward pass answers all six questions — and the result
//! drives a real policy: which team gets it, what priority it lands at, and whether a human
//! has to look at it before anything is automated.
//!
//! ```text
//! cargo run --release --example support_triage
//! ```

mod common;

use std::time::Instant;

use laya::{Answer, ModelName, Prediction, Question, Questions, Router};
use serde_json::{Value, json};

/// Below this, a `choice` is not a routing decision — it is a question for a human.
/// Normalised entropy over five options: 0.70 means the winner is well clear of the rest.
const TEAM_FLOOR: f32 = 0.70;
/// A `noul` carries its own confidence as the distance from a coin flip, so it reads directly.
const BOOLEAN_FLOOR: f32 = 0.75;

struct Ticket {
    id: &'static str,
    plan: &'static str,
    state: Value,
}

fn inbox() -> Vec<Ticket> {
    vec![
        Ticket {
            id: "SUP-1041",
            plan: "enterprise",
            state: json!({
                "subject": "Duplicate charge on invoice #4411",
                "body": "We were billed twice for March. Refund the duplicate today or we move to a competitor.",
            }),
        },
        Ticket {
            id: "SUP-1042",
            plan: "free",
            state: json!({
                "subject": "Webhooks stopped firing",
                "body": "Since this morning none of our webhooks arrive. Nothing changed on our side. Production is down.",
            }),
        },
        Ticket {
            id: "SUP-1043",
            plan: "pro",
            state: json!({
                "subject": "Merci !",
                "body": "Juste un mot pour dire que le nouveau tableau de bord est excellent. Rien à signaler, continuez comme ça.",
            }),
        },
        Ticket {
            id: "SUP-1044",
            plan: "pro",
            state: json!({
                "subject": "Preisliste",
                "body": "Guten Tag, wir würden gerne auf den Enterprise-Tarif wechseln. Können Sie uns ein Angebot für 50 Sitze schicken?",
            }),
        },
        Ticket {
            id: "SUP-1045",
            plan: "enterprise",
            state: json!({
                "body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। यह बहुत परेशान करने वाला है।",
            }),
        },
    ]
}

/// The six questions a triage decision needs. All of them are answered together.
fn triage_questions() -> Questions {
    Questions::new()
        .with(
            "team",
            Question::choice("Which team should own this ticket?")
                .option("billing", "invoices, charges, refunds, payment methods")
                .option("technical", "bugs, outages, integrations, API errors")
                .option("account", "logins, permissions, seats, data exports")
                .option("sales", "pricing, upgrades, quotes, new contracts")
                .option("other", "none of the above fits"),
        )
        .with(
            "urgency",
            Question::score("How urgent is this ticket?")
                .level("no time pressure")
                .level("should be handled this week")
                .level("needs attention today")
                .level("production is blocked or a deadline is being missed"),
        )
        .with(
            "frustration",
            Question::score("How frustrated does the customer sound?")
                .level("calm and neutral")
                .level("concerned but civil")
                .level("clearly annoyed")
                .level("very angry or using strong language"),
        )
        .with("refund_requested", Question::noul("Does the customer ask for money back?"))
        .with(
            "churn_risk",
            Question::noul("Does the customer threaten to cancel or move to a competitor?"),
        )
        .with(
            "needs_engineer",
            Question::noul("Does answering this require someone who can read logs or code?")
                .when_true("a system fault has to be investigated")
                .when_false("a support agent can answer from the product"),
        )
}

/// What the triage decided, and why.
struct Triage {
    team: String,
    priority: &'static str,
    queue: String,
    flags: Vec<String>,
}

fn decide(out: &Prediction, plan: &str) -> Triage {
    let team = out.get("team").unwrap();
    let urgency = out.get("urgency").unwrap().as_score().unwrap();
    let frustration = out.get("frustration").unwrap().as_score().unwrap();
    let churn = out.get("churn_risk").unwrap().as_noul().unwrap();
    let refund = out.get("refund_requested").unwrap().as_noul().unwrap();
    let engineer = out.get("needs_engineer").unwrap().as_noul().unwrap();

    // The score is ordinal, so the expected level is a usable number: weight it, add the
    // boolean risks, and let the plan break ties.
    let pressure =
        urgency + 0.5 * frustration + 2.0 * churn + if plan == "enterprise" { 0.8 } else { 0.0 };
    let priority = match pressure {
        p if p >= 3.5 => "P1",
        p if p >= 2.0 => "P2",
        p if p >= 1.0 => "P3",
        _ => "P4",
    };

    // An answer the model is not sure about is not a routing decision, it is a question for a
    // human. Each primitive gets the floor that suits it: a five-option `choice` and a boolean
    // do not measure confidence on the same scale, and a 4-level `score` is read as a number
    // rather than gated at all.
    let team_sure = team.confidence() >= TEAM_FLOOR;
    let queue = if !team_sure {
        "human triage".to_string()
    } else if engineer > 0.6 {
        format!("{}-oncall", team.as_choice().unwrap())
    } else {
        team.as_choice().unwrap().to_string()
    };

    let mut flags = Vec::new();
    if churn > 0.5 {
        flags.push(format!("churn {churn:.2}"));
    }
    if refund > 0.5 {
        flags.push(format!("refund {refund:.2}"));
    }
    for id in ["refund_requested", "churn_risk", "needs_engineer"] {
        let a = out.get(id).unwrap();
        if a.confidence() < BOOLEAN_FLOOR {
            flags.push(format!("{id} undecided"));
        }
    }
    if !team_sure {
        flags.push(format!("team undecided ({:.2})", team.confidence()));
    }

    Triage { team: team.as_choice().unwrap().to_string(), priority, queue, flags }
}

fn main() -> Result<(), laya::Error> {
    // Detection costs microseconds; a cold checkpoint build costs seconds. An inbox that mixes
    // languages must preload, or it rebuilds a model on every switch.
    let mut report = common::Report::new();
    let router =
        Router::builder().preload([ModelName::English, ModelName::Multilingual]).build()?;
    report.lap("2 checkpoints resident");
    println!();

    let questions = triage_questions();
    let mut queues: Vec<(String, String)> = Vec::new();

    for ticket in inbox() {
        let t = Instant::now();
        let out = router.predict(ticket.state.clone(), &questions)?;
        let elapsed = t.elapsed();
        let routing = out.routing.as_ref().unwrap();
        let triage = decide(&out, ticket.plan);

        println!(
            "{} [{}]  {} -> {}  ({:.0} ms, {} tokens)",
            ticket.id,
            ticket.plan,
            routing.model,
            triage.queue,
            elapsed.as_secs_f64() * 1000.0,
            out.usage.input_tokens,
        );
        println!("   routing   {}", routing.reason);
        println!(
            "   priority  {}   team {} ({:.0}% confident)",
            triage.priority,
            triage.team,
            100.0 * out.get("team").unwrap().confidence(),
        );
        println!(
            "   signals   urgency {:.2}/3   frustration {:.2}/3   engineer {:.2}",
            out.get("urgency").unwrap().as_score().unwrap(),
            out.get("frustration").unwrap().as_score().unwrap(),
            out.get("needs_engineer").unwrap().as_noul().unwrap(),
        );
        if !triage.flags.is_empty() {
            println!("   flags     {}", triage.flags.join(" | "));
        }
        println!("   options   {}", distribution(out.get("team").unwrap()));
        println!();

        queues.push((triage.queue, format!("{} {}", triage.priority, ticket.id)));
    }

    report.lap(&format!("triaged {} tickets", queues.len()));

    queues.sort();
    println!("\n--- work queues ---");
    let mut current = String::new();
    for (queue, item) in &queues {
        if *queue != current {
            current.clone_from(queue);
            println!("{queue}:");
        }
        println!("   {item}");
    }

    report.finish();
    Ok(())
}

/// The full distribution behind a choice, which is what makes a near-miss visible.
fn distribution(answer: &Answer) -> String {
    match answer {
        Answer::Choice { probabilities, .. } => {
            let mut ranked: Vec<_> = probabilities.iter().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(a.1));
            ranked
                .iter()
                .take(3)
                .map(|(label, p)| format!("{label} {p:.2}"))
                .collect::<Vec<_>>()
                .join("  ")
        }
        _ => String::new(),
    }
}
