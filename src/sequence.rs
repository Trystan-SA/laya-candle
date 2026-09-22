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

use std::borrow::Cow;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::question::{QType, Question};
use crate::tokenizer::SpecialTokens;

/// At most this many tokens of any single option text.
const MAX_OPTION_TOKENS: usize = 48;
/// The question head never shrinks below this many instruction tokens.
const MIN_HEAD_TOKENS: usize = 8;
/// The state text is cut at this many bytes per sequence slot before it is tokenised. No BPE
/// token in these vocabularies comes anywhere near this long, so every token that could still
/// fit in the sequence lies inside the kept prefix; the rest would be tokenised and dropped.
const MAX_STATE_BYTES_PER_TOKEN: usize = 64;

/// One question, encoded and ready to be batched.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    /// One label per marker, in marker order.
    pub labels: Vec<String>,
}

/// Turns questions and states into sequences that fit a checkpoint's budget.
pub(crate) struct Encoder {
    tok: Tokenizer,
    sp: SpecialTokens,
    /// Total sequence budget, including the question head and the state.
    max_len: usize,
    /// Token budget for the question head (instructions plus every option).
    head_max_len: usize,
}

impl Encoder {
    pub(crate) fn new(tok: Tokenizer, sp: SpecialTokens, max_len: usize, head_max_len: usize) -> Self {
        Self { tok, sp, max_len, head_max_len }
    }

    /// Any literal mask token in user text would create a marker the scorer would read as an
    /// option boundary, so it is blanked out everywhere text enters the sequence.
    fn scrub<'a>(&self, s: &'a str) -> Cow<'a, str> {
        if s.contains(&self.sp.mask_token) {
            Cow::Owned(s.replace(&self.sp.mask_token, " "))
        } else {
            Cow::Borrowed(s)
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Only the ids are read, so the offsets `encode` would also compute are skipped.
        Ok(self.tok.encode_fast(text, false)?.get_ids().to_vec())
    }

    /// Encode the flattened state once; every question in a batch shares it.
    pub(crate) fn encode_state(&self, state: &str) -> Result<Vec<u32>> {
        let mut cut = state.len().min(self.max_len * MAX_STATE_BYTES_PER_TOKEN);
        while !state.is_char_boundary(cut) {
            cut -= 1;
        }
        self.encode(&self.scrub(&state[..cut]))
    }

    /// Encode one question over an already-encoded state.
    ///
    /// A state that does not fit is cut at the end.
    pub(crate) fn build(&self, state_ids: &[u32], id: &str, q: &Question) -> Result<Item> {
        let (labels, texts): (Vec<String>, Vec<String>) = q.options(id)?.into_iter().unzip();
        let n_options = texts.len();

        let head_text = format!("{} question: {}", q.kind.name(), self.scrub(&q.instructions_text()));
        let mut head_ids = self.encode(&head_text)?;

        let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(n_options);
        for text in &texts {
            let mut body = self.encode(&format!(" {}", self.scrub(text)))?;
            body.truncate(MAX_OPTION_TOKENS);
            let mut ids = Vec::with_capacity(body.len() + 1);
            ids.push(self.sp.mask_id);
            ids.extend(body);
            opt_ids.push(ids);
        }

        let head_max_len = self.head_max_len;
        let total: usize = opt_ids.iter().map(Vec::len).sum();
        let mut opt_budget = head_max_len.saturating_sub(total);
        if opt_budget < 16 {
            // The options alone are eating the head budget: share what is left evenly between them
            // rather than letting the first options starve the last.
            let per = (head_max_len.saturating_sub(16) / n_options).max(4);
            for o in &mut opt_ids {
                o.truncate(per);
            }
            let total: usize = opt_ids.iter().map(Vec::len).sum();
            opt_budget = head_max_len.saturating_sub(total);
        }
        head_ids.truncate(opt_budget.max(MIN_HEAD_TOKENS));

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(self.sp.cls_id);
        ids.extend_from_slice(&head_ids);
        ids.push(self.sp.sep_id);

        let mut markers = Vec::with_capacity(n_options);
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.push(self.sp.sep_id);

        // Markers grow with the sequence, so the last one is the first to fall off the end.
        if markers.last().is_some_and(|&m| m >= self.max_len) {
            return Err(Error::question(
                id,
                format!(
                    "its {n_options} options do not fit in head_max_len={head_max_len}; \
                     shorten the option descriptions or use a checkpoint with a larger head budget"
                ),
            ));
        }

        let room = self.max_len.saturating_sub(ids.len() + 1);
        ids.extend_from_slice(&state_ids[..state_ids.len().min(room)]);
        ids.push(self.sp.sep_id);
        ids.truncate(self.max_len);

        Ok(Item { ids, markers, qtype: q.kind, labels })
    }

    /// Pad a batch of encoded questions into rectangular tensors' worth of data.
    pub(crate) fn collate(&self, items: &[Item]) -> Batch {
        let batch = items.len();
        let seq_len = items.iter().map(|i| i.ids.len()).max().unwrap_or(0);
        // A question with a single option still needs two slots: the action head reads a top-1
        // and a top-2 probability, and the padded slot supplies the missing one as a hard zero.
        let k_max = items.iter().map(|i| i.markers.len()).max().unwrap_or(0).max(2);

        let mut input_ids = vec![self.sp.pad_id; batch * seq_len];
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
            qtype.push(it.qtype.index() as u32);
        }

        Batch { input_ids, attention_mask, marker_pos, marker_mask, qtype, batch, seq_len, k_max, n_tokens }
    }
}

/// A batch of encoded questions, padded rectangular.
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
