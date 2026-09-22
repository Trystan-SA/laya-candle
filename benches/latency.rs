//! How fast is it, and what does it cost in memory?
//!
//! Measures the two numbers that decide whether this fits a workload: the price of loading a
//! checkpoint, and the marginal cost of one more question in the same call.
//!
//! ```text
//! cargo bench --bench latency
//! ```

#[path = "../examples/common/mod.rs"]
mod common;

use std::time::Instant;

use laya::{Agent, ModelName, Questions, presets};
use serde_json::json;

/// Discard this many passes before measuring.
const WARMUP: usize = 2;
/// Report the median of this many.
const RUNS: usize = 7;

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn state() -> serde_json::Value {
    json!({
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan. \
                 This is the third message and nobody has answered.",
    })
}

/// Fifteen genuinely different questions: three presets whose ids do not collide.
fn pool() -> Questions {
    let pool: Questions = presets::triage()
        .iter()
        .chain(presets::guard().iter())
        .chain(presets::moderation().iter())
        .map(|(id, q)| (id.clone(), q.clone()))
        .collect();
    assert_eq!(pool.len(), 15, "the three presets must not share question ids");
    pool
}

fn main() -> Result<(), laya::Error> {
    let before = common::rss().unwrap_or(0);
    println!("laya-rs latency benchmark");
    println!(
        "build: {}",
        if cfg!(debug_assertions) { "debug — numbers are meaningless" } else { "release" }
    );

    // --- load cost -------------------------------------------------------------------------
    println!("\n## checkpoint load\n");
    println!("{:<16} {:>9} {:>12} {:>12}", "checkpoint", "load s", "rss after", "peak rss");
    let mut agents = Vec::new();
    for (name, subfolder) in
        [(ModelName::English, None), (ModelName::Multilingual, Some("multilingual"))]
    {
        let t = Instant::now();
        let agent = Agent::from_hub("convaiinnovations/laya", subfolder)?;
        let load = t.elapsed().as_secs_f64();
        println!(
            "{:<16} {load:>9.2} {:>12} {:>12}",
            name.to_string(),
            common::bytes(common::rss().unwrap_or(0)),
            common::bytes(common::peak_rss().unwrap_or(0)),
        );
        agents.push((name, agent));
    }
    println!(
        "\n(both resident at once: {} over the {} this process started at)",
        common::bytes(common::rss().unwrap_or(0).saturating_sub(before)),
        common::bytes(before),
    );

    // --- marginal cost of a question -------------------------------------------------------
    let pool = pool();
    let state = state();
    for (name, agent) in &agents {
        println!("\n## {name}: cost of asking more questions in one call\n");
        println!(
            "{:>9} {:>11} {:>13} {:>8} {:>10}",
            "questions", "median ms", "ms/question", "tokens", "vs 1 q"
        );

        let mut baseline = 0.0;
        for n in [1usize, 2, 5, 10, 15] {
            let questions: Questions =
                pool.iter().take(n).map(|(id, q)| (id.clone(), q.clone())).collect();
            for _ in 0..WARMUP {
                agent.predict(state.clone(), &questions)?;
            }
            let mut times = Vec::with_capacity(RUNS);
            let mut tokens = 0;
            for _ in 0..RUNS {
                let t = Instant::now();
                let out = agent.predict(state.clone(), &questions)?;
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                tokens = out.usage.input_tokens;
            }
            let ms = median(times);
            if n == 1 {
                baseline = ms;
            }
            println!(
                "{n:>9} {ms:>11.0} {:>13.1} {tokens:>8} {:>9.1}x",
                ms / n as f64,
                ms / baseline
            );
        }
    }

    println!(
        "\nRead the `vs 1 q` column honestly: every question is its own row in the batch and\n\
         carries its own copy of the state, so the encoder work does scale with the question\n\
         count. Batching buys perhaps 1.5x in ms/question by filling the matrix multiplies\n\
         better — it does not make the extra questions free.\n\n\
         What the architecture buys is elsewhere: there is no decoding. The answer is read off\n\
         the [MASK] markers in the same pass, so cost is set by the input length, not by how\n\
         much an autoregressive model would have had to write.\n\n\
         On CPU the encoder dominates; enable `mkl`, `accelerate`, `cuda` or `metal` before\n\
         comparing these numbers to anything."
    );
    println!("\npeak rss for the whole run: {}", common::bytes(common::peak_rss().unwrap_or(0)));
    Ok(())
}
