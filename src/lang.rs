//! Dependency-free script and language detection, used to pick a checkpoint.
//!
//! Routing only needs one decision: *is this English Latin text, or is it something the English
//! checkpoint cannot read?* The English checkpoint does not gently degrade off English, it
//! collapses — on 20-option intent classification it scores 0.100 on Hindi against 0.050 for
//! random guessing, while reporting high confidence. So **script** is the primary signal and it
//! is detected exactly; the Latin-script language guess is a stopword and diacritic heuristic and
//! is explicitly best effort.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Unicode blocks the English (50k English BPE) checkpoint cannot read.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    ("cyrillic", &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    ("arabic", &[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    ("hangul", &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
    ("kana", &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
    ("han", &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
];

/// Function words. Latin-script languages overlap heavily (`de` / `la` / `le` / `un` / `e` /
/// `que`), so a margin is required before calling something non-English. Order matters: it
/// decides ties between languages with equal hit counts.
const STOPWORDS: &[(&str, &[&str])] = &[
    ("en", &["the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
             "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
             "will", "can", "would", "there", "their", "what", "which", "please", "we", "i"]),
    ("fr", &["le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
             "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"]),
    ("de", &["der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
             "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"]),
    ("es", &["el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
             "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"]),
    ("pt", &["os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
             "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"]),
    ("it", &["il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
             "questo", "questa", "anche", "come", "più", "nella", "alla"]),
    ("nl", &["het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
             "door", "maar", "ook", "worden", "deze", "naar", "wordt"]),
    // Romanian words its Romance neighbours do not share: `la`, `o`, `un`, `de`, `pe`, `ca` are
    // deliberately left out so adding `ro` cannot steal a French or Spanish state.
    ("ro", &["și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
             "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
             "acesta", "aceasta", "mi", "ți", "vă", "nu"]),
];

/// Letters ordinary English does not use.
///
/// This is the signal that catches a Latin-script language no stopword list here covers
/// (Romanian, Polish, Czech, Turkish, Baltic, …) — the difference between routing it to the
/// multilingual checkpoint and silently handing it to the one that cannot read it.
const NON_EN_DIACRITICS: &str = concat!(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ", // Western European
    "ăâîșțşţ",                          // Romanian
    "ąćęłńśźż",                         // Polish
    "čďěňřšťůž",                        // Czech / Slovak
    "őű",                               // Hungarian
    "ğı",                               // Turkish (text is lowercased before matching)
    "āēģīķļņūž",                        // Baltic
    "đ",                                // Serbo-Croatian / Vietnamese
);

/// A diacritic rate above this is taken as evidence the text is not English, even when no
/// stopword list matches it.
pub const NON_EN_DIACRITIC_RATE: f32 = 0.02;

/// How much text detection looks at.
const MAX_DETECTION_CHARS: usize = 4000;
/// How deep detection walks into a nested state.
const MAX_DEPTH: usize = 6;

/// What detection concluded about a state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Detection {
    /// Dominant script: `"latin"`, `"han"`, `"devanagari"`, … or `"unknown"` when there are no letters.
    pub script: String,
    /// Fraction of alphabetic characters belonging to each script found.
    pub script_profile: BTreeMap<String, f32>,
    /// Best-effort language code for Latin text; `None` when undecided.
    pub language: Option<String>,
    /// Whether the English checkpoint can be expected to read this state.
    pub is_english: bool,
    /// True when no language could be named — which is *not* the same as English.
    pub language_undecided: bool,
    /// Share of characters that are non-English letters.
    pub diacritic_rate: f32,
    /// Share of alphabetic characters outside the Latin script.
    pub non_latin_fraction: f32,
}

fn round4(v: f32) -> f32 {
    (v * 1e4).round() / 1e4
}

fn script_of(c: char) -> Option<&'static str> {
    let cp = c as u32;
    if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
        return Some("latin");
    }
    SCRIPT_RANGES
        .iter()
        .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
        .map(|(name, _)| *name)
}

/// Count alphabetic characters per script, keeping first-seen order for tie-breaking.
fn script_counts(text: &str) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for c in text.chars().filter(|c| c.is_alphabetic()) {
        let Some(name) = script_of(c) else { continue };
        match counts.iter_mut().find(|(n, _)| *n == name) {
            Some((_, n)) => *n += 1,
            None => counts.push((name, 1)),
        }
    }
    counts
}

/// The string leaves of a state, in order. Keys are ignored: they are usually English.
fn text_leaves(state: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > MAX_DEPTH {
        return;
    }
    match state {
        Value::String(s) => out.push(s.clone()),
        Value::Object(map) => {
            for v in map.values() {
                text_leaves(v, depth + 1, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                text_leaves(v, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Flatten a state into the text detection reads.
pub fn state_text(state: &Value) -> String {
    let mut leaves = Vec::new();
    text_leaves(state, 0, &mut leaves);
    leaves.join(" ").chars().take(MAX_DETECTION_CHARS).collect()
}

/// Dominant script of `text`, or `"unknown"` when it holds no letters.
pub fn detect_script(text: &str) -> String {
    let counts = script_counts(text);
    if counts.is_empty() {
        return "unknown".to_string();
    }
    // Python's `max` keeps the first maximum in insertion order, and so does this.
    counts
        .iter()
        .copied()
        .reduce(|best, cur| if cur.1 > best.1 { cur } else { best })
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Fraction of alphabetic characters belonging to each detected script.
pub fn script_profile(text: &str) -> BTreeMap<String, f32> {
    let counts = script_counts(text);
    let total: usize = counts.iter().map(|(_, n)| *n).sum();
    if total == 0 {
        return BTreeMap::new();
    }
    counts
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(name, n)| (name.to_string(), n as f32 / total as f32))
        .collect()
}

struct LatinProfile {
    language: Option<String>,
    diacritic_rate: f32,
    looks_non_english: bool,
}

fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphabetic() {
            cur.extend(c.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn latin_profile(text: &str) -> LatinProfile {
    let ws = words(text);
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    let diac = lowered.iter().filter(|c| NON_EN_DIACRITICS.contains(**c)).count();
    let diacritic_rate = diac as f32 / std::cmp::max(1, lowered.len()) as f32;
    let looks_non_english = diacritic_rate >= NON_EN_DIACRITIC_RATE;

    if ws.len() < 4 {
        return LatinProfile { language: None, diacritic_rate, looks_non_english };
    }

    let hits = |list: &[&str]| ws.iter().filter(|w| list.contains(&w.as_str())).count();
    let en = STOPWORDS.iter().find(|(lg, _)| *lg == "en").map(|(_, l)| hits(l)).unwrap_or(0);
    let (mut best_lg, best) = STOPWORDS
        .iter()
        .filter(|(lg, _)| *lg != "en")
        .map(|(lg, list)| (Some(*lg), hits(list)))
        .reduce(|best, cur| if cur.1 > best.1 { cur } else { best })
        .unwrap_or((None, 0));
    // No stopword hit for any non-English language is no evidence for a *particular* one.
    // Naming the winner of a 0-0 tie invented a language, so stay undecided and let the
    // diacritic rate speak.
    if best == 0 {
        best_lg = None;
    }

    let language = match best_lg {
        // A non-English language needs a clear margin over English function words.
        Some(lg) if best >= std::cmp::max(2, en + 2) => Some(lg.to_string()),
        // Two hits are required here too: one shared function word ("para" in Turkish text)
        // named Spanish on the strength of the diacritics alone, which is a guess dressed as
        // a detection.
        Some(lg) if looks_non_english && best >= std::cmp::max(2, en) => Some(lg.to_string()),
        _ if en > 0 && !looks_non_english => Some("en".to_string()),
        _ => None,
    };

    LatinProfile { language, diacritic_rate, looks_non_english }
}

/// Full detection result for a state.
pub fn analyse(state: &Value) -> Detection {
    let text = state_text(state);
    let profile = script_profile(&text);
    let script = detect_script(&text);
    let non_latin = if profile.is_empty() {
        0.0
    } else {
        round4(1.0 - profile.get("latin").copied().unwrap_or(0.0))
    };

    if script == "unknown" {
        return Detection {
            script,
            script_profile: profile,
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
        };
    }
    if script != "latin" {
        return Detection {
            script,
            script_profile: profile,
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
        };
    }

    let lat = latin_profile(&text);
    let undecided = lat.language.is_none();
    // Undecided is not English. Treating it as English sent every Latin-script language we hold
    // no stopwords for to the checkpoint that cannot read it, silently. When nothing identifies
    // the language, non-English letters are enough to prefer the multilingual checkpoint; text
    // with no such letters (including short English) still goes to the English one.
    let is_english =
        lat.language.as_deref() == Some("en") || (undecided && !lat.looks_non_english);
    Detection {
        script,
        script_profile: profile,
        language: lat.language,
        is_english,
        language_undecided: undecided,
        diacritic_rate: round4(lat.diacritic_rate),
        non_latin_fraction: non_latin,
    }
}

/// True when the English checkpoint can be expected to read this state.
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_latin_scripts_are_detected_exactly() {
        assert_eq!(detect_script("मुझसे दो बार शुल्क लिया गया"), "devanagari");
        assert_eq!(detect_script("请尽快退款"), "han");
        assert_eq!(detect_script("환불해 주세요"), "hangul");
        assert_eq!(detect_script("الرجاء رد المبلغ"), "arabic");
        assert_eq!(detect_script("1234 !!"), "unknown");
    }

    #[test]
    fn english_is_routed_to_the_english_checkpoint() {
        let d = analyse(&json!({"body": "We were billed twice for March and would like a refund."}));
        assert_eq!(d.script, "latin");
        assert_eq!(d.language.as_deref(), Some("en"));
        assert!(d.is_english);
    }

    #[test]
    fn a_latin_script_language_is_not_english() {
        let d = analyse(&json!("Der Kunde wurde zweimal belastet und ist nicht zufrieden"));
        assert_eq!(d.language.as_deref(), Some("de"));
        assert!(!d.is_english);
    }

    #[test]
    fn an_unidentified_latin_language_is_caught_by_its_letters() {
        // No Romanian stopwords hit here, but the diacritics do.
        let d = analyse(&json!("Clientul a fost taxat de două ori și dorește rambursarea sumei"));
        assert!(!d.is_english, "{d:?}");
    }

    #[test]
    fn a_state_with_no_letters_stays_on_the_default() {
        let d = analyse(&json!({"amount": 4411, "ok": false}));
        assert_eq!(d.script, "unknown");
        assert!(d.is_english);
    }

    #[test]
    fn detection_reads_nested_string_leaves() {
        let d = analyse(&json!({"thread": [{"body": "मुझसे दो बार शुल्क लिया गया"}]}));
        assert_eq!(d.script, "devanagari");
        assert!(!d.is_english);
    }
}
