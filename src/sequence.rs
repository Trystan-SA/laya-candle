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
/// The state is first tokenised from a prefix of this many bytes per sequence slot, so a huge
/// state is not tokenised in full only to be cut. Ordinary text runs at a few bytes per token,
/// but the English vocabulary holds 128-byte and longer tokens (rules of dashes, for one), so
/// the prefix grows until it yields more tokens than the sequence can hold.
const STATE_BYTES_PER_TOKEN: usize = 16;

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
    pub(crate) fn new(
        tok: Tokenizer,
        sp: SpecialTokens,
        max_len: usize,
        head_max_len: usize,
    ) -> Self {
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
    ///
    /// Only the first `max_len` tokens can ever be used, so the state is tokenised from a
    /// prefix that yields more than that, not in full. The token straddling the cut may differ
    /// from the full tokenisation, which is why the prefix must yield strictly more.
    pub(crate) fn encode_state(&self, state: &str) -> Result<Vec<u32>> {
        let mut budget = self.max_len.max(1) * STATE_BYTES_PER_TOKEN;
        loop {
            let mut cut = state.len().min(budget);
            while !state.is_char_boundary(cut) {
                cut -= 1;
            }
            let ids = self.encode(&self.scrub(&state[..cut]))?;
            if cut == state.len() || ids.len() > self.max_len {
                return Ok(ids);
            }
            budget = budget.saturating_mul(2);
        }
    }

    /// Encode one question over an already-encoded state.
    ///
    /// A state that does not fit is cut at the end.
    pub(crate) fn build(&self, state_ids: &[u32], id: &str, q: &Question) -> Result<Item> {
        let (labels, texts): (Vec<String>, Vec<String>) = q.options(id)?.into_iter().unzip();
        let n_options = texts.len();

        let head_text =
            format!("{} question: {}", q.kind.name(), self.scrub(&q.instructions_text()));
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

        Batch {
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
            batch,
            seq_len,
            k_max,
            n_tokens,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{tiny_id, tiny_tokenizer};

    fn encoder(max_len: usize, head_max_len: usize) -> Encoder {
        let (tok, sp) = tiny_tokenizer();
        Encoder::new(tok, sp, max_len, head_max_len)
    }

    fn ids(words: &str) -> Vec<u32> {
        words.split_whitespace().map(tiny_id).collect()
    }

    #[test]
    fn the_layout_is_cls_head_sep_markers_sep_state_sep() {
        let enc = encoder(64, 32);
        let q: Question = Question::choice("pick one").bare_option("a").bare_option("b").into();
        let state = enc.encode_state("hello world").unwrap();
        let item = enc.build(&state, "q", &q).unwrap();

        let mut expected = ids("[CLS] choice question : pick one [SEP]");
        let m0 = expected.len();
        expected.extend(ids("[MASK] a"));
        let m1 = expected.len();
        expected.extend(ids("[MASK] b [SEP] hello world [SEP]"));

        assert_eq!(item.ids, expected);
        assert_eq!(item.markers, vec![m0, m1]);
        assert_eq!(item.labels, vec!["a", "b"]);
        assert_eq!(item.qtype, QType::Choice);
    }

    #[test]
    fn a_mask_token_in_user_text_cannot_forge_a_marker() {
        let enc = encoder(64, 32);
        let q: Question = Question::noul("pick [MASK] one").into();
        let state = enc.encode_state("hello [MASK] world").unwrap();
        let item = enc.build(&state, "q", &q).unwrap();

        let mask = tiny_id("[MASK]");
        assert_eq!(item.ids.iter().filter(|&&t| t == mask).count(), 2, "{:?}", item.ids);
        assert_eq!(item.markers.len(), 2);
        assert_eq!(item.labels, vec!["false", "true"]);
    }

    #[test]
    fn a_long_state_is_cut_to_max_len_and_still_ends_with_sep() {
        let enc = encoder(24, 16);
        let q: Question = Question::noul("pick one").into();
        let state = enc.encode_state(&"hello world ".repeat(50)).unwrap();
        let item = enc.build(&state, "q", &q).unwrap();

        assert_eq!(item.ids.len(), 24);
        assert_eq!(*item.ids.last().unwrap(), tiny_id("[SEP]"));
        assert!(item.markers.iter().all(|&m| m < 24), "{:?}", item.markers);
    }

    #[test]
    fn options_that_overflow_the_sequence_are_an_error() {
        let enc = encoder(8, 64);
        let q: Question =
            Question::choice("pick one").bare_option("a").bare_option("b").bare_option("c").into();
        let err = enc.build(&[], "dept", &q).unwrap_err();
        assert!(matches!(err, Error::Question { ref id, .. } if id == "dept"), "{err}");
    }

    #[test]
    fn option_bodies_are_capped_at_max_option_tokens() {
        let enc = encoder(512, 192);
        let q: Question = Question::choice("pick one").option("b", "a ".repeat(60)).into();
        let item = enc.build(&[], "q", &q).unwrap();

        let m0 = item.markers[0];
        let end = item.ids[m0..].iter().position(|&t| t == tiny_id("[SEP]")).unwrap();
        assert_eq!(end, MAX_OPTION_TOKENS + 1, "marker plus a capped body");
    }

    #[test]
    fn collate_pads_rows_and_reserves_two_marker_slots() {
        let enc = encoder(64, 32);
        let state = enc.encode_state("hello world").unwrap();
        let one: Question = Question::choice("pick one").bare_option("a").into();
        let three: Question =
            Question::choice("pick one").bare_option("a").bare_option("b").bare_option("c").into();
        let items = vec![
            enc.build(&state, "one", &one).unwrap(),
            enc.build(&state, "three", &three).unwrap(),
        ];

        let batch = enc.collate(&items);
        assert_eq!((batch.batch, batch.k_max), (2, 3));
        assert_eq!(batch.seq_len, items[1].ids.len());
        assert_eq!(batch.n_tokens, items[0].ids.len() + items[1].ids.len());

        // The shorter row is padded and masked out past its own tokens.
        let (short, long) = (items[0].ids.len(), batch.seq_len);
        assert!(batch.input_ids[short..long].iter().all(|&t| t == tiny_id("[PAD]")));
        assert_eq!(batch.attention_mask[..long].iter().sum::<f32>() as usize, short);
        assert_eq!(&batch.marker_mask, &[1, 0, 0, 1, 1, 1]);
        assert_eq!(batch.qtype, vec![0, 0]);

        // A single one-option question still gets two marker slots for the action head.
        assert_eq!(enc.collate(&items[..1]).k_max, 2);
    }

    #[test]
    fn encode_state_cuts_at_a_char_boundary() {
        // Three-byte characters in words of four, so the prefix cut (5 * 16, then doubled)
        // lands inside a character.
        let enc = encoder(5, 2);
        let ids = enc.encode_state(&"€€€€ ".repeat(200)).unwrap();
        assert!(ids.len() > 5, "{}", ids.len());
    }

    #[test]
    fn long_tokens_do_not_starve_the_state() {
        // Each 255-byte word is one token, far longer than the first prefix allows per slot.
        let enc = encoder(8, 2);
        let state = format!("{} hello", "x".repeat(255)).repeat(20);
        let got = enc.encode_state(&state).unwrap();
        assert!(got.len() > 8, "{}", got.len());

        // A state that ends first is tokenised whole.
        assert_eq!(enc.encode_state("hello world").unwrap(), ids("hello world"));
    }
}
