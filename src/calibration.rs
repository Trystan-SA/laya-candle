//! Turning raw marker logits into calibrated probabilities.

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

/// The calibration bucket a question falls into: `"<type>:<option-count>"`.
pub fn temp_bucket(kind: QType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", kind.name())
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
        assert_eq!(temp_bucket(QType::Noul, 2), "noul:2");
        assert_eq!(temp_bucket(QType::Choice, 4), "choice:3-5");
        assert_eq!(temp_bucket(QType::Choice, 7), "choice:6-10");
        assert_eq!(temp_bucket(QType::Choice, 20), "choice:11+");
        assert_eq!(temp_bucket(QType::Score, 3), "score:3-5");
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
