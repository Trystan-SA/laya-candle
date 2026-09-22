# laya-rs

**Typed decisions in one forward pass — in Rust, with no Python and no server.**

`laya-rs` answers *typed questions* about any state (a string, an email, a ticket, a JSON
document) using a bidirectional encoder and a decision head. Nothing is generated, so there is
nothing to parse and nothing to hallucinate: every answer is one of three primitives and comes
back with a probability distribution and a calibrated confidence.

It is a pure-Rust port of [Laya](https://github.com/NandhaKishorM/laya) — the open-source answer
to TypeSafe's Jev — and it loads Laya's published checkpoints directly from the Hugging Face Hub.
No PyTorch, no ONNX Runtime, no sidecar process: add the crate and call it.

| primitive | you ask | you get |
|---|---|---|
| `choice` | pick one label from a named set | the label, plus a probability per option |
| `score`  | rate the state against ordered levels | the expected level, plus a probability per level |
| `noul`   | a boolean question | the probability the statement holds |

A question set is answered in a **single batched forward pass, with no decoding**: each option
becomes a `[MASK]` marker and the answer is read off that marker's hidden state. Cost is set by
how long the input is, not by how much an autoregressive model would have had to write.

---

## Install

```toml
[dependencies]
laya-rs = "0.1"
serde_json = "1"
```

Rust 1.85+. A C compiler is needed the first time: `candle-core` pulls in `onig`, which builds a
small C library.

Library-only, without the CLI:

```toml
laya-rs = { version = "0.1", default-features = false, features = ["hub"] }
```

---

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
out.routing.unwrap().reason;                    // "English Latin text"
```

Questions also load from the same JSON schema the Python package uses, so an existing question
file works unchanged:

```rust
let questions = laya::Questions::from_json(&std::fs::read_to_string("questions.json")?)?;
```

One checkpoint, no routing:

```rust
let agent = laya::Agent::from_hub("convaiinnovations/laya", Some("multilingual"))?;
let out = agent.predict(json!({"message": "Ich wurde zweimal belastet"}), &laya::presets::triage())?;
```

Or from a directory you already have:

```rust
let agent = laya::Agent::from_dir("checkpoints/laya")?;
```

An `Agent` is immutable once built and `predict` takes `&self`, so share one behind an `Arc`
across threads.

---

## Command line

```console
$ cargo install laya-rs

$ laya predict -s @examples/data/email.json -q triage
routed to english — English Latin text

intent            choice  refund                        confidence 0.99
is_urgent         noul    0.222                         confidence 0.78
frustration       score   1.72 / 3                      confidence 0.32
refund_requested  noul    0.891                         confidence 0.89
churn_risk        noul    0.360                         confidence 0.64

379 input tokens, 0 generated
```

- `laya predict -s <text|@file|-> -q <preset|@file|-> [--model NAME] [--lang xx] [--checkpoint DIR] [--json]`
- `laya route -s <state>` — show the routing decision without downloading or running anything
- `laya presets [name]` — list the built-in question sets, or print one as JSON

Built-in presets: `triage`, `email`, `guard`, `moderation`, `router`.

---

## Examples

Six runnable programs in `examples/`, each printing how long it took and what it cost in memory.

```console
cargo run --release --example quickstart
```

| example | what it shows |
|---|---|
| `quickstart` | load a checkpoint, ask three questions, read the answers |
| `support_triage` | a full support desk: route each ticket to a team, set a priority, send the unsure ones to a human |
| `primitives` | `choice` / `score` / `noul` side by side, with the full distribution behind each answer |
| `guardrails` | screen prompts in front of an LLM: allow, review or block |
| `multilingual` | let `Router` pick the checkpoint, and override it when you already know the language |
| `questions_from_json` | questions as configuration, in the same JSON schema the Python package and the CLI use |

Every one ends with a line like:

```text
  [ 2.39 s]  checkpoint ready   (rss 1.8 GiB)
  [ 889 ms]  answered 3 questions   (rss 1.8 GiB)

  total 3.28 s wall, rss 1.8 GiB, peak 2.4 GiB
```

Memory is read from `/proc/self/status`, so the numbers appear on Linux and are quietly skipped
elsewhere.

---

## Benchmarks

`benches/` measures the model rather than showing how to use it.

```console
cargo bench --bench latency        # what it costs
cargo bench --bench reliability    # whether it decides anything
```

`latency` reports checkpoint load time, resident and peak memory, and the cost of asking more
questions in one call. On 24 CPU cores, f32, no BLAS feature enabled:

| | load | resident | 1 question | 15 questions |
|---|---|---|---|---|
| `english` (421M) | 2.1 s | 1.8 GiB | 1209 ms | 12.0 s (801 ms/q) |
| `multilingual` (322M) | 2.7 s | 1.3 GiB | 606 ms | 7.3 s (486 ms/q) |

`reliability` measures separation — the thing a broken forward pass loses first, since it still
returns well-formed probabilities either way. It exits non-zero if a check fails:

```text
question            complaint     praise        gap
refund_requested        0.906      0.016      0.890
churn_risk              0.692      0.009      0.683
urgency                 1.491      0.477      1.014

routed  8/8 correct, mean confidence 0.81
english 7/8 correct, mean confidence 0.56
```

It also reports a preset question that does **not** separate: `harm_severity` scores attacks and
benign prompts 0.105 apart, so no threshold on it is a guardrail. The benchmark says so out loud
rather than quietly passing — measure a question on your own traffic before it gates anything.

---

## Checkpoints

| name | encoder | params | context | use it for |
|---|---|---|---|---|
| `english` | ModernBERT-large | 421M | 512 | English |
| `multilingual` | mmBERT-base | 322M | 1024 | 100+ languages, about 2x faster |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | the four typed-decisions workflows |

Weights come from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) and
are cached by `hf-hub` on first use. Only the requested subfolder is downloaded.

A checkpoint directory is five files:

```
rl_agent_config.json     head shape, sequence budget, calibration temperatures
model.safetensors        backbone + decision head
encoder/config.json      the ModernBERT backbone's shape
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

Anything with that layout loads, so a checkpoint you fine-tune yourself works with no changes
here.

---

## Why route

The English checkpoint does not gently degrade off English — it collapses. On 20-option intent
classification it scores 0.100 on Hindi and 0.103 on Korean against 0.050 for random guessing,
**and reports high confidence while doing so**. Because it stays confident while being wrong,
confidence gating cannot save you. Script detection can, and it costs microseconds of pure Rust
before the forward pass:

```rust
laya::detect_language(&json!("Der Kunde wurde zweimal belastet")).is_english;  // false
router.route(&state, &questions, &Default::default())?.reason;
// "Latin script but language looks like \"de\", not English"
```

Detection is exact for script and best-effort for the language of Latin text. Pass
`RouteOptions::lang("de")` or `RouteOptions::model("multilingual")` when you already know.

A cold checkpoint build costs seconds while detection costs microseconds, so a server that sees
mixed languages should `preload` — otherwise the default residency of one model rebuilds on
every language switch.

---

## Confidence, and where it stops being trustworthy

The probabilities are trained against strictly proper scoring rules, so gating on them is
meaningful:

```rust
for id in out.below_confidence(0.85) {
    escalate_to_a_human(id);
}
```

Two caveats this crate makes explicit rather than silently absorbing:

- **Confidence is only valid inside a checkpoint's competence.** A model answering a script it
  cannot read is confidently wrong, and no threshold catches that. Route first.
- **A shipped temperature below 1 sharpens rather than softens.** The English checkpoint's
  `choice:11+` bucket is fitted at 0.1006, which multiplies the logits roughly tenfold: a 0.24
  top probability would be published as 0.99. Temperatures are clamped to `[0.5, 5.0]`, and any
  clamped bucket is reported at load and listed by `Agent::clamped_temperatures()`. Treat
  confidence from those buckets as uncalibrated.

---

## Performance

Everything runs in f32 through [candle](https://github.com/huggingface/candle). candle's
ModernBERT builds its attention masks in f32 unconditionally, so a half-precision backbone would
fail on the first broadcast; f32 is also what the reference implementation falls back to off
CUDA.

That makes a plain CPU build slow — see the `latency` table above. Enable the feature that
matches your machine before drawing any conclusion about speed:

```toml
laya-rs = { version = "0.1", features = ["mkl"] }        # Intel CPU
laya-rs = { version = "0.1", features = ["accelerate"] } # macOS
laya-rs = { version = "0.1", features = ["cuda"] }       # NVIDIA
laya-rs = { version = "0.1", features = ["metal"] }      # Apple GPU
```

Two things worth knowing before sizing a deployment:

- **Sequences pad to the longest question in the batch, not to `max_len`**, so short states stay
  cheap.
- **Questions are not free.** Each one is its own row carrying its own copy of the state, so
  fifteen questions cost about ten times one. Batching buys roughly 1.5x in ms/question by
  filling the matrix multiplies better. What the architecture buys is the absence of decoding:
  the answer is read off the `[MASK]` markers in the same pass.

An `Agent` holds its weights resident — about 1.8 GiB for `english`, 1.3 GiB for `multilingual`,
with a load-time peak roughly 0.6 GiB above that. `Router` keeps one checkpoint resident by
default and evicts least-recently-used; `preload` raises the cap to fit what you preload.

---

## Fidelity to the reference

The sequence format, decision head, calibration buckets, confidence formula and routing rules are
ported from Laya's published implementation and match it structurally:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

Each `[MASK]` is a marker; the hidden state at that position is what the scorer reads. States are
serialised with Python's `json.dumps(..., ensure_ascii=False)` separators — `{"a": 1, "b": 2}`,
spaces included — because a different string is a different tokenisation, which is a different
prediction.

Known differences from running the Python package on a GPU:

- **f32, not autocast fp16/bf16.** The reference runs the encoder under autocast on CUDA. Small
  probability shifts are expected; the answers and their ordering are not affected.
- **Temperatures are clamped.** See above. The reference applies the shipped values as they are.
- The illustrative numbers in Laya's own README were produced on a T4 in fp16 against whatever
  checkpoint revision was current then, so treat them as indicative rather than as a fixture.

`cargo test` runs the ported logic offline in well under a second. The end-to-end tests are
`#[ignore]`d because they pull ~1.7 GB of weights:

```console
cargo test --release -- --ignored --nocapture
```

They check that the answers are well-formed, that opposite states separate (a forward pass wired
up wrongly still returns well-formed numbers — it just stops discriminating), and that the router
sends Hindi to the checkpoint that can read it and still answers `billing`.

---

## Licence

Apache-2.0, matching Laya. See `NOTICE` for attribution: this is a port, the checkpoints are the
upstream authors' work, and the ModernBERT backbone comes from `candle-transformers`.
