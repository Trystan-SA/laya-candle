//! Building the token sequence a decision is read from.
//!
//! The layout is fixed by training:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! Each `[MASK]` is a *marker*: the hidden state at that position is what the scorer reads, so
//! one marker per option and the model answers every option of every question in one pass.

use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::question::Question;
use crate::tokenizer::SpecialTokens;

/// At most this many tokens of any single option text.
const MAX_OPTION_TOKENS: usize = 48;
/// The question head never shrinks below this many instruction tokens.
const MIN_HEAD_TOKENS: usize = 8;

/// One question, encoded and ready to be batched.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
}

fn truncate(mut v: Vec<u32>, n: usize) -> Vec<u32> {
    v.truncate(n);
    v
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false)?.get_ids().to_vec())
}

/// Encode one question against one state.
///
/// `state` is the already-flattened state text. `truncate_left` keeps the *end* of a state that
/// does not fit, which is what you want for a conversation.
pub(crate) fn build(
    tok: &Tokenizer,
    sp: &SpecialTokens,
    state: &str,
    id: &str,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<Item> {
    let options = q.render_options(id)?;
    let n_options = options.len();
    if n_options == 0 {
        return Err(Error::question(id, "has no options to score"));
    }

    // Any literal mask token in user text would create a marker the scorer would read as an
    // option boundary, so it is blanked out everywhere text enters the sequence.
    let scrub = |s: &str| s.replace(&sp.mask_token, " ");

    let head_text = format!("{} question: {}", q.kind.name(), scrub(&q.instructions_text()));
    let mut head_ids = encode(tok, &head_text)?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(n_options);
    for opt in &options {
        let body = truncate(encode(tok, &format!(" {}", scrub(opt)))?, MAX_OPTION_TOKENS);
        let mut ids = Vec::with_capacity(body.len() + 1);
        ids.push(sp.mask_id);
        ids.extend(body);
        opt_ids.push(ids);
    }

    let total: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as i64 - total as i64;
    if opt_budget < 16 {
        // The options alone are eating the head budget: share what is left evenly between them
        // rather than letting the first options starve the last.
        let per = std::cmp::max(4, (head_max_len as i64 - 16) / n_options as i64) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        let total: usize = opt_ids.iter().map(Vec::len).sum();
        opt_budget = head_max_len as i64 - total as i64;
    }
    head_ids.truncate(std::cmp::max(MIN_HEAD_TOKENS as i64, opt_budget) as usize);

    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls_id);
    ids.extend_from_slice(&head_ids);
    ids.push(sp.sep_id);

    let mut markers = Vec::with_capacity(n_options);
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(sp.sep_id);

    let room = max_len.saturating_sub(ids.len() + 1);
    let state_ids = encode(tok, &scrub(state))?;
    let kept = if state_ids.len() <= room {
        state_ids.as_slice()
    } else if truncate_left {
        &state_ids[state_ids.len() - room..]
    } else {
        &state_ids[..room]
    };
    ids.extend_from_slice(kept);
    ids.push(sp.sep_id);
    ids.truncate(max_len);

    markers.retain(|&m| m < max_len);
    if markers.len() != n_options {
        return Err(Error::question(
            id,
            format!(
                "its {n_options} options do not fit in head_max_len={head_max_len}; \
                 shorten the option descriptions or use a checkpoint with a larger head budget"
            ),
        ));
    }

    Ok(Item { ids, markers, qtype: q.kind.index() })
}

/// Pad a batch of encoded questions into rectangular tensors' worth of data.
pub(crate) struct Batch {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<f32>,
    pub marker_pos: Vec<u32>,
    pub marker_mask: Vec<u8>,
    pub qtype: Vec<u32>,
    pub batch: usize,
    pub seq_len: usize,
    pub k_max: usize,
    pub n_tokens: usize,
}

pub(crate) fn collate(items: &[Item], pad_id: u32) -> Batch {
    let batch = items.len();
    let seq_len = items.iter().map(|i| i.ids.len()).max().unwrap_or(0);
    // A question with a single option still needs two slots: the action head reads a top-1 and a
    // top-2 probability, and the padded slot supplies the missing one as a hard zero.
    let k_max = items.iter().map(|i| i.markers.len()).max().unwrap_or(0).max(2);

    let mut input_ids = vec![pad_id; batch * seq_len];
    let mut attention_mask = vec![0f32; batch * seq_len];
    let mut marker_pos = vec![0u32; batch * k_max];
    let mut marker_mask = vec![0u8; batch * k_max];
    let mut qtype = Vec::with_capacity(batch);
    let mut n_tokens = 0;

    for (r, it) in items.iter().enumerate() {
        let row = r * seq_len;
        for (c, &t) in it.ids.iter().enumerate() {
            input_ids[row + c] = t;
            attention_mask[row + c] = 1.0;
        }
        n_tokens += it.ids.len();
        let mrow = r * k_max;
        for (c, &m) in it.markers.iter().enumerate() {
            marker_pos[mrow + c] = m as u32;
            marker_mask[mrow + c] = 1;
        }
        qtype.push(it.qtype as u32);
    }

    Batch { input_ids, attention_mask, marker_pos, marker_mask, qtype, batch, seq_len, k_max, n_tokens }
}
