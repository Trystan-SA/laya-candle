//! `laya` — answer typed questions about a state from the command line.

use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use laya::{Answer, ModelName, ModelSpec, Prediction, Questions, RouteOptions, Router, presets};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "laya", version, about = "Typed decisions in one forward pass", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Answer a set of questions about a state.
    Predict(PredictArgs),
    /// Show which checkpoint a state would be routed to, without loading anything.
    Route(RouteArgs),
    /// List the built-in question sets, or print one as JSON.
    Presets {
        /// A preset name; omit to list them all.
        name: Option<String>,
    },
}

#[derive(clap::Args)]
struct PredictArgs {
    /// The state: literal text, @file (JSON or text), or - for stdin.
    #[arg(short, long)]
    state: String,

    /// Questions: a preset name, @file.json, or - for stdin.
    #[arg(short, long, default_value = "triage")]
    questions: String,

    /// Force a checkpoint instead of routing: english, multilingual or typed-decisions.
    #[arg(short, long)]
    model: Option<String>,

    /// Declare the language instead of detecting it.
    #[arg(long)]
    lang: Option<String>,

    /// Load a checkpoint directory instead of fetching from the Hub. Implies --model.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<PathBuf>,

    /// Print the full result as JSON instead of a summary.
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct RouteArgs {
    /// The state: literal text, @file (JSON or text), or - for stdin.
    #[arg(short, long)]
    state: String,

    /// Questions, which can select a typed-decisions workflow: preset name, @file.json, or -.
    #[arg(short, long, default_value = "triage")]
    questions: String,

    /// Let matching question ids select the typed-decisions checkpoint.
    #[arg(long)]
    auto_task_detection: bool,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Predict(args) => predict(args),
        Command::Route(args) => route(args),
        Command::Presets { name } => show_presets(name),
    }
}

fn predict(args: PredictArgs) -> anyhow::Result<()> {
    let state = read_state(&args.state)?;
    let questions = read_questions(&args.questions)?;

    let mut router = Router::builder();
    let mut opts =
        RouteOptions { model: args.model.clone(), lang: args.lang.clone(), ..Default::default() };
    if let Some(dir) = &args.checkpoint {
        // The directory stands in for whichever checkpoint `--model` names (English by default),
        // and that checkpoint is forced so routing never looks elsewhere.
        let name = match &args.model {
            Some(m) => ModelName::parse(m)?,
            None => ModelName::English,
        };
        router = router.model(name, ModelSpec::Dir(dir.clone()));
        opts.model = Some(name.as_str().to_string());
    }
    let out = router.build()?.predict_with(&state, &questions, &opts)?;

    if args.json {
        println!("{}", out.to_json());
    } else {
        print_summary(&out);
    }
    Ok(())
}

fn route(args: RouteArgs) -> anyhow::Result<()> {
    let state = read_state(&args.state)?;
    let questions = read_questions(&args.questions)?;
    // Routing never touches the weights, so nothing is downloaded here.
    let router = Router::builder().auto_task_detection(args.auto_task_detection).build()?;
    let decision = router.route(&state, &questions, &Default::default())?;
    println!("{}", serde_json::to_string_pretty(&decision)?);
    Ok(())
}

fn show_presets(name: Option<String>) -> anyhow::Result<()> {
    match name {
        None => {
            for n in presets::names() {
                let qs = presets::by_name(n).expect("a listed preset always resolves");
                let ids: Vec<&str> = qs.iter().map(|(id, _)| id.as_str()).collect();
                println!("{n:<12} {}", ids.join(", "));
            }
        }
        Some(n) => {
            let qs = presets::by_name(&n)
                .with_context(|| format!("unknown preset {n:?}; try one of {}", preset_names()))?;
            println!("{}", serde_json::to_string_pretty(&qs)?);
        }
    }
    Ok(())
}

fn preset_names() -> String {
    presets::names().collect::<Vec<_>>().join(", ")
}

/// `-` reads stdin and `@path` reads that file; anything else is `None`, for the caller to take
/// literally.
fn read_source(arg: &str) -> anyhow::Result<Option<String>> {
    if arg == "-" {
        return read_stdin().map(Some);
    }
    match arg.strip_prefix('@') {
        Some(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {path}")).map(Some)
        }
        None => Ok(None),
    }
}

/// A state is literal text unless it names a file or stdin, in which case JSON is tried first.
fn read_state(arg: &str) -> anyhow::Result<Value> {
    Ok(match read_source(arg)? {
        Some(raw) => serde_json::from_str(&raw).unwrap_or(Value::String(raw)),
        None => Value::String(arg.to_string()),
    })
}

fn read_questions(arg: &str) -> anyhow::Result<Questions> {
    let Some(raw) = read_source(arg)? else {
        return presets::by_name(arg).with_context(|| {
            format!(
                "{arg:?} is neither a preset nor a file; presets are {}, and a file is given as \
                 @path.json",
                preset_names()
            )
        });
    };
    let questions = Questions::from_json(&raw)?;
    if questions.is_empty() {
        bail!("no questions to answer");
    }
    Ok(questions)
}

fn read_stdin() -> anyhow::Result<String> {
    std::io::read_to_string(std::io::stdin()).context("reading stdin")
}

fn print_summary(out: &Prediction) {
    if let Some(r) = &out.routing {
        println!("routed to {} — {}\n", r.model, r.reason);
    }
    let width = out.answers.keys().map(String::len).max().unwrap_or(0);
    for (id, answer) in &out.answers {
        let (kind, value) = match answer {
            Answer::Choice { choice, .. } => ("choice", choice.clone()),
            Answer::Score { score, legend, .. } => {
                ("score", format!("{score:.2} / {}", legend.len().saturating_sub(1)))
            }
            Answer::Noul { noul, .. } => ("noul", format!("{noul:.3}")),
        };
        println!("{id:<width$}  {kind:<6}  {value:<28}  confidence {:.2}", answer.confidence());
    }
    println!("\n{} input tokens, 0 generated", out.usage.input_tokens);
}
