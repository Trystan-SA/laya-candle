//! Turning raw marker logits into calibrated probabilities.

use crate::config::AgentConfig;
use crate::question::QType;

/// A fitted temperature below 1 sharpens the logits instead of softening them.
///
/// The shipped `choice:11+` bucket of the English checkpoint is 0.1006, which multiplies the
/// logits roughly tenfold: a 0.24 top probability would be published as 0.99, so a caller gating
/// on confidence is told a coin flip is a certainty. No honest calibration needs to sharpen that
/// hard, so temperatures are confined to a range that cannot manufacture confidence.
pub const TEMP_MIN: f32 = 0.5;
/// Upper end of the usable temperature range. See [`TEMP_MIN`].
pub const TEMP_MAX: f32 = 5.0;

/// `t` confined to `[TEMP_MIN, TEMP_MAX]`, falling back to 1.0 when it is not a usable number.
pub fn clamp_temperature(t: f32) -> f32 {
    if !t.is_finite() { 1.0 } else { t.clamp(TEMP_MIN, TEMP_MAX) }
}

/// The option-count bucket a temperature is fitted for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionBucket {
    Two,
    ThreeToFive,
    SixToTen,
    ElevenPlus,
}

impl OptionBucket {
    pub const ALL: [OptionBucket; 4] = [
        OptionBucket::Two,
        OptionBucket::ThreeToFive,
        OptionBucket::SixToTen,
        OptionBucket::ElevenPlus,
    ];

    /// The bucket a question with `k` options falls into.
    pub fn from_count(k: usize) -> Self {
        match k {
            0..=2 => OptionBucket::Two,
            3..=5 => OptionBucket::ThreeToFive,
            6..=10 => OptionBucket::SixToTen,
            _ => OptionBucket::ElevenPlus,
        }
    }

    /// The bucket's name in `temperature_by_options` keys.
    pub fn name(self) -> &'static str {
        match self {
            OptionBucket::Two => "2",
            OptionBucket::ThreeToFive => "3-5",
            OptionBucket::SixToTen => "6-10",
            OptionBucket::ElevenPlus => "11+",
        }
    }
}

/// Read a `temperature_by_options` key, `"<type>:<bucket>"`; `None` for one this crate does
/// not know.
fn parse_bucket(key: &str) -> Option<(QType, OptionBucket)> {
    let (kind, bucket) = key.split_once(':')?;
    let kind = QType::ALL.into_iter().find(|q| q.name() == kind)?;
    let bucket = OptionBucket::ALL.into_iter().find(|b| b.name() == bucket)?;
    Some((kind, bucket))
}

/// A checkpoint's temperatures, clamped and resolved once at load.
///
/// A question's temperature is its `"<type>:<bucket>"` entry when the checkpoint ships one, else
/// the per-type value, else 1.0. The reference implementation applies that precedence on every
/// prediction; here it is folded into one table so the hot path only indexes.
#[derive(Clone, Debug)]
pub struct Calibration {
    table: [[f32; OptionBucket::ALL.len()]; QType::ALL.len()],
    /// Every shipped temperature that had to be clamped, as `"<bucket>=<value>"`.
    pub clamped: Vec<String>,
}

impl Calibration {
    pub fn from_config(cfg: &AgentConfig) -> Self {
        let mut clamped = Vec::new();
        let mut clamp = |label: String, t: f32| {
            let c = clamp_temperature(t);
            if c != t {
                clamped.push(format!("{label}={t:.4}"));
            }
            c
        };

        let mut table = [[1.0f32; OptionBucket::ALL.len()]; QType::ALL.len()];
        for kind in QType::ALL {
            if let Some(&t) = cfg.temperature.get(kind.index()) {
                table[kind.index()] =
                    [clamp(format!("temperature[{}]", kind.index()), t); OptionBucket::ALL.len()];
            }
        }
        for (key, &t) in &cfg.temperature_by_options {
            if let Some((kind, bucket)) = parse_bucket(key) {
                table[kind.index()][bucket as usize] = clamp(key.clone(), t);
            }
        }
        Self { table, clamped }
    }

    /// The temperature to divide a `kind` question's logits by when it has `k` options.
    pub fn temperature(&self, kind: QType, k: usize) -> f32 {
        self.table[kind.index()][OptionBucket::from_count(k) as usize]
    }
}

/// Numerically stable softmax over a slice.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    }
    p
}

/// Confidence as normalised Shannon entropy: `1 - H(p) / ln(k)`.
///
/// A single-option question is fully decided by construction, so it scores 1.0.
pub fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = p[..k].iter().map(|&v| -v * v.clamp(1e-12, 1.0).ln()).sum();
    (1.0 - ent / (k as f32).ln()).clamp(0.0, 1.0)
}

/// Round to four decimals, the precision the reference implementation publishes.
pub fn round4(v: f32) -> f32 {
    (v * 1e4).round() / 1e4
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_sharpening_temperature_is_refused() {
        assert_eq!(clamp_temperature(0.1006), TEMP_MIN);
        assert_eq!(clamp_temperature(1.9063), 1.9063);
        assert_eq!(clamp_temperature(42.0), TEMP_MAX);
        assert_eq!(clamp_temperature(f32::NAN), 1.0);
        assert_eq!(clamp_temperature(f32::INFINITY), 1.0);
    }

    #[test]
    fn buckets_match_the_reference_names() {
        assert_eq!(OptionBucket::from_count(2).name(), "2");
        assert_eq!(OptionBucket::from_count(4).name(), "3-5");
        assert_eq!(OptionBucket::from_count(7).name(), "6-10");
        assert_eq!(OptionBucket::from_count(20).name(), "11+");
        assert_eq!(parse_bucket("score:3-5"), Some((QType::Score, OptionBucket::ThreeToFive)));
        assert_eq!(parse_bucket("bogus:9"), None);
    }

    #[test]
    fn a_bucket_overrides_its_type_and_the_rest_fall_back() {
        let cfg: AgentConfig = serde_json::from_value(json!({
            "temperature": [1.5, 2.0, 1.0],
            "temperature_by_options": {"choice:11+": 0.1006, "score:2": 3.0, "bogus:9": 42.0},
        }))
        .unwrap();
        let cal = Calibration::from_config(&cfg);
        assert_eq!(cal.temperature(QType::Choice, 4), 1.5);
        assert_eq!(cal.temperature(QType::Choice, 20), TEMP_MIN);
        assert_eq!(cal.temperature(QType::Score, 2), 3.0);
        assert_eq!(cal.temperature(QType::Score, 5), 2.0);
        assert_eq!(cal.temperature(QType::Noul, 2), 1.0);
        assert_eq!(cal.clamped, vec!["choice:11+=0.1006".to_string()]);
    }

    #[test]
    fn confidence_spans_uniform_to_certain() {
        assert!(confidence_from_probs(&[0.25; 4], 4).abs() < 1e-6);
        assert!((confidence_from_probs(&[1.0, 0.0, 0.0, 0.0], 4) - 1.0).abs() < 1e-6);
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[3.0, 1.0, -2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1] && p[1] > p[2]);
    }
}
