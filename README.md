# laya-rs

Use this crate to classify, score or check a piece of text from your Rust code: give it a message and a few questions, and it answers each one with a label, a score or a yes/no probability you can branch on. It helps with triaging support tickets, sorting emails, moderating posts, guarding prompts before they reach an LLM, and any other case where you need a reliable decision rather than generated text.

**Typed decisions in one forward pass, in pure Rust.**

Ask typed questions about any state (string, email, ticket, JSON) and get back a label, a score
or a probability, each with a calibrated confidence. Because nothing is generated, the answer never
needs parsing and cannot hallucinate.

Pure-Rust port of [Laya](https://github.com/NandhaKishorM/laya). Loads Laya's checkpoints
directly from the Hugging Face Hub via [candle](https://github.com/huggingface/candle).

| primitive | you ask | you get |
|---|---|---|
| `choice` | pick one label from a set | the label + a probability per option |
| `score`  | rate against ordered levels | the expected level + a probability per level |
| `noul`   | a yes/no question | the probability it holds |

All questions are answered in a single batched forward pass. Cost scales with input length, not
output length.

## Install

```toml
[dependencies]
laya-rs = "0.1"
serde_json = "1"
```

Rust 1.88+. A C compiler is needed once (`candle-core` pulls in `onig`).

Library only, no CLI:

```toml
laya-rs = { version = "0.1", default-features = false, features = ["hub"] }
```

## Quickstart

```rust
use laya::{ModelName, Question, Questions, Router};
use serde_json::json;

let router = Router::builder().preload([ModelName::English, ModelName::Multilingual]).build()?;

let questions = Questions::new()
    .with("department", Question::choice("Which department should handle this request?")
        .option("billing",   "invoices, payments, refunds")
        .option("technical", "bugs, outages, system errors")
        .option("other",     "everything else"))
    .with("urgency", Question::score("How urgent is this request?")
        .level("not urgent").level("soon").level("critical deadline or blocking issue"))
    .with("churn_risk", Question::noul("Does the user threaten to cancel or leave?"));

let out = router.predict(json!({
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
}), &questions)?;

out.get("department").unwrap().as_choice();     // Some("billing")
out.get("department").unwrap().confidence();    // 0.86
out.get("urgency").unwrap().as_score();         // Some(1.44)
out.get("churn_risk").unwrap().as_noul();       // Some(0.825)
```

Other ways to build:

```rust
// Questions from the same JSON schema as the Python package
let questions = laya::Questions::from_json(&std::fs::read_to_string("questions.json")?)?;

// One checkpoint, no routing
let agent = laya::Agent::from_hub("convaiinnovations/laya", Some("multilingual"))?;

// From a local directory
let agent = laya::Agent::from_dir("checkpoints/laya")?;
```

`Agent` is immutable and `predict` takes `&self`: share one behind an `Arc`.

Gate on confidence:

```rust
for id in out.below_confidence(0.85) {
    escalate_to_a_human(id);
}
```

## Where Rust matters

The decision model is the same one the Python package runs. What changes with a native crate is
where it can go and what it can sit inside of:

- **A request-path middleware.** `predict` takes `&self`, so one checkpoint behind an `Arc`
  serves every worker of an axum, actix or tonic service. Screen prompts, route tickets or gate a
  webhook inside the process that received it, without a round trip to a sidecar service.

  ```rust
  let agent = Arc::new(Agent::from_hub("convaiinnovations/laya", None)?);
  let guard = presets::guard();

  // in each handler:
  let out = agent.predict(json!({"prompt": body}), &guard)?;
  if out.get("prompt_injection").and_then(|a| a.as_noul()).unwrap_or(0.0) > 0.9 {
      return Err(StatusCode::FORBIDDEN);
  }
  ```

- **A single binary at the edge.** There is no interpreter or runtime to ship alongside it. The
  crate, a weights directory and `Agent::from_dir` run on a box with no network, a kiosk, or a
  container whose image is the binary plus five files.

- **Text that already flows through Rust.** A proxy, an API gateway, a Kafka consumer, a Discord
  or Slack bot, a log shipper: a typed decision can be added where the text is, without
  introducing a second language to the deployment.

- **Multilingual traffic without a model registry.** `Router` detects the script and language in
  microseconds of pure Rust before touching any weights, then runs the checkpoint that can read
  the text. One code path handles 100+ languages.

- **Batch jobs and shell pipelines.** The `laya` CLI reads questions as JSON and states from
  stdin, so a cron job or a `find | xargs laya predict` labels a corpus with no code at all.

- **Confidence you can branch on.** Every answer comes back as a distribution rather than a string. The gating
  logic (escalate, retry, hand to a human) is ordinary Rust over ordinary numbers rather than a
  regex over generated prose.

## CLI

```console
$ cargo install laya-rs

$ laya predict -s @examples/data/email.json -q triage
routed to english — English Latin text

intent            choice  refund      confidence 0.99
is_urgent         noul    0.222       confidence 0.78
frustration       score   1.72 / 3    confidence 0.32
refund_requested  noul    0.891       confidence 0.89
churn_risk        noul    0.360       confidence 0.64
```

- `laya predict -s <text|@file|-> -q <preset|@file|-> [--model NAME] [--lang xx] [--checkpoint DIR] [--json]`
- `laya route -s <state>`: show the routing decision without running the model
- `laya presets [name]`: list built-in question sets (`triage`, `email`, `guard`, `moderation`, `router`)

## Examples and benchmarks

```console
cargo run --release --example quickstart
cargo bench --bench latency        # load time, memory, cost per question
cargo bench --bench reliability    # checks that opposite states actually separate
```

| example | shows |
|---|---|
| `quickstart` | load, ask three questions, read the answers |
| `support_triage` | route tickets, set priority, escalate the unsure ones |
| `primitives` | `choice` / `score` / `noul` with full distributions |
| `guardrails` | allow / review / block prompts in front of an LLM |
| `multilingual` | let `Router` pick the checkpoint, or override it |
| `questions_from_json` | questions as JSON configuration |

## Checkpoints

| name | encoder | params | context | use for |
|---|---|---|---|---|
| `english` | ModernBERT-large | 421M | 512 | English |
| `multilingual` | mmBERT-base | 322M | 1024 | 100+ languages, ~2x faster |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

Weights come from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya),
cached by `hf-hub` on first use. Any directory with the same five-file layout loads, including
your own fine-tunes.

## Routing

The English checkpoint collapses to near-random on non-Latin scripts **while staying confident**,
so confidence gating cannot catch it. `Router` detects script and language in microseconds before
the forward pass and picks the right checkpoint.

```rust
laya::detect_language(&json!("Der Kunde wurde zweimal belastet")).is_english;  // false
```

Pass `RouteOptions::lang("de")` or `RouteOptions::model("multilingual")` when you already know.
Call `preload` on servers seeing mixed languages, or the router rebuilds a checkpoint on every
language switch.

## Confidence caveats

- **Only valid inside a checkpoint's competence.** Route first.
- **Shipped temperatures below 1 sharpen instead of soften.** They are clamped to `[0.5, 5.0]`;
  clamped buckets are reported at load and listed by `Agent::clamped_temperatures()`. Treat their
  confidence as uncalibrated.
- **Measure before you gate.** The `reliability` bench flags `harm_severity` in the `guard` preset
  as not separating attacks from benign prompts.

## Performance

Everything runs in f32 on the CPU by default. On 24 cores, no BLAS:

| | load | resident | 1 question | 15 questions |
|---|---|---|---|---|
| `english` | 2.1 s | 1.8 GiB | 1209 ms | 12.0 s (801 ms/q) |
| `multilingual` | 2.7 s | 1.3 GiB | 606 ms | 7.3 s (486 ms/q) |

Enable the feature that matches your hardware before judging speed:

```toml
laya-rs = { version = "0.1", features = ["mkl"] }        # Intel CPU
laya-rs = { version = "0.1", features = ["accelerate"] } # macOS
laya-rs = { version = "0.1", features = ["cuda"] }       # NVIDIA
laya-rs = { version = "0.1", features = ["metal"] }      # Apple GPU
```

Each question is its own batch row carrying a copy of the state, so 15 questions cost about 10x
one. Sequences pad to the longest row in the batch, not to `max_len`.

## Fidelity to the reference

Sequence format, decision head, calibration and routing are ported from Laya's implementation:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

Differences from the Python package on GPU: f32 instead of autocast fp16 (small probability
shifts, same answers), and clamped temperatures.

```console
cargo test                                          # offline, < 1 s
cargo test --release -- --ignored --nocapture       # end-to-end, downloads ~1.7 GB
```

## Licence

Apache-2.0, matching Laya. See `NOTICE` for attribution.
