//! The decision model: a ModernBERT backbone plus the typed decision head.
//!
//! The head is a faithful port of the PyTorch module the checkpoints were trained with:
//! a type embedding added to every position, two pre-norm transformer layers, a scorer that
//! reads the `[MASK]` markers, and an auxiliary action head fed by the scorer's own confidence.

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert;

use crate::error::Result;
use crate::question::QType;

/// Everything in this crate runs in f32.
///
/// candle's ModernBERT builds its attention masks in f32 unconditionally, so a half-precision
/// backbone would fail on the very first `broadcast_add`. f32 is also what the reference
/// implementation falls back to on CPU and MPS.
pub(crate) const DTYPE: DType = DType::F32;

/// `nn.TransformerEncoderLayer` keeps this eps, and so must we. A bare eps is an affine,
/// mean-removing `LayerNormConfig`, which is what PyTorch's `LayerNorm` is.
const HEAD_LN_EPS: f64 = 1e-5;
/// Pad positions are pushed this far below the real scores before the softmax.
const MASKED_SCORE: f64 = -1e9;
/// Markers that do not exist for a given question are scored this low, as in the reference.
const MASKED_OPTION_LOGIT: f32 = -1e4;

/// `nn.MultiheadAttention` with the packed `in_proj_weight` PyTorch stores.
struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl MultiheadAttention {
    fn load(vb: VarBuilder, hidden: usize, n_heads: usize) -> Result<Self> {
        let w = vb.get((3 * hidden, hidden), "in_proj_weight")?;
        let b = vb.get(3 * hidden, "in_proj_bias")?;
        Ok(Self {
            in_proj: Linear::new(w, Some(b)),
            out_proj: candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?,
            n_heads,
            head_dim: hidden / n_heads,
        })
    }

    /// `bias` is an additive `[B, 1, 1, L]` mask: 0 on real tokens, very negative on padding.
    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let qkv = self.in_proj.forward(xs)?;
        let split = |offset: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, offset * d, d)?
                .reshape((b, l, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);

        let scale = (self.head_dim as f64).powf(-0.5);
        let att = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax_last_dim(&att.broadcast_add(bias)?)?;

        let out = att.matmul(&v)?.transpose(1, 2)?.reshape((b, l, d))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

/// `nn.TransformerEncoderLayer(..., norm_first=True)`; the activation is ReLU, PyTorch's default.
struct HeadLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize, n_heads: usize) -> Result<Self> {
        let ff = 4 * hidden;
        Ok(Self {
            self_attn: MultiheadAttention::load(vb.pp("self_attn"), hidden, n_heads)?,
            linear1: candle_nn::linear(hidden, ff, vb.pp("linear1"))?,
            linear2: candle_nn::linear(ff, hidden, vb.pp("linear2"))?,
            norm1: candle_nn::layer_norm(hidden, HEAD_LN_EPS, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(hidden, HEAD_LN_EPS, vb.pp("norm2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.self_attn.forward(&self.norm1.forward(xs)?, bias)?)?;
        let ff = self.linear2.forward(&self.linear1.forward(&self.norm2.forward(&xs)?)?.relu()?)?;
        Ok((xs + ff)?)
    }
}

/// What one forward pass produces.
pub(crate) struct Forward {
    /// `[B, K]` per-option logits, with absent options pushed to `MASKED_OPTION_LOGIT`.
    pub logits: Vec<Vec<f32>>,
    /// `[B, n_act]` probabilities from the auxiliary action head.
    pub act_probs: Vec<Vec<f32>>,
}

/// The full decision model.
pub(crate) struct DecisionModel {
    encoder: modernbert::ModernBert,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer_fc1: Linear,
    scorer_fc2: Linear,
    act_fc1: Linear,
    act_fc2: Linear,
    hidden: usize,
    device: Device,
}

impl DecisionModel {
    /// Build the model from a checkpoint's tensors.
    ///
    /// `weights` is the raw `model.safetensors` map; encoder keys are rewritten from
    /// `encoder.*` to the `encoder.model.*` layout candle's ModernBERT expects.
    pub(crate) fn load(
        weights: std::collections::HashMap<String, Tensor>,
        enc_cfg: &modernbert::Config,
        head_layers: usize,
        n_act: usize,
        device: &Device,
    ) -> Result<Self> {
        let remapped: std::collections::HashMap<String, Tensor> = weights
            .into_iter()
            .map(|(k, v)| match k.strip_prefix("encoder.") {
                Some(rest) => (format!("encoder.model.{rest}"), v),
                None => (k, v),
            })
            .collect();

        let vb = VarBuilder::from_tensors(remapped, DTYPE, device);
        let hidden = enc_cfg.hidden_size;
        // The training code derives the head's head count from the width, not from the encoder.
        let n_heads = std::cmp::max(1, hidden / 64);

        let encoder = modernbert::ModernBert::load(vb.pp("encoder"), enc_cfg)?;
        let head = (0..head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), hidden, n_heads))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            encoder,
            head,
            type_emb: candle_nn::embedding(QType::ALL.len(), hidden, vb.pp("type_emb"))?,
            scorer_norm: candle_nn::layer_norm(hidden, HEAD_LN_EPS, vb.pp("scorer.0"))?,
            scorer_fc1: candle_nn::linear(hidden, hidden, vb.pp("scorer.1"))?,
            scorer_fc2: candle_nn::linear(hidden, 1, vb.pp("scorer.3"))?,
            act_fc1: candle_nn::linear(hidden + 4, 256, vb.pp("act_head.0"))?,
            act_fc2: candle_nn::linear(256, n_act, vb.pp("act_head.2"))?,
            hidden,
            device: device.clone(),
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn forward(&self, batch: &crate::sequence::Batch) -> Result<Forward> {
        let dev = &self.device;
        let (b, l, k) = (batch.batch, batch.seq_len, batch.k_max);

        let input_ids = Tensor::from_slice(&batch.input_ids, (b, l), dev)?;
        let attention_mask = Tensor::from_slice(&batch.attention_mask, (b, l), dev)?;
        let marker_pos = Tensor::from_slice(&batch.marker_pos, (b, k), dev)?;
        let marker_mask = Tensor::from_slice(&batch.marker_mask, (b, k), dev)?;
        let qtype = Tensor::from_slice(&batch.qtype, b, dev)?;

        let hs = self.encoder.forward(&input_ids, &attention_mask)?.to_dtype(DTYPE)?;

        // The question type is a global signal, so it is added to every position.
        let type_vec = self.type_emb.forward(&qtype)?.unsqueeze(1)?;
        let mut hs = hs.broadcast_add(&type_vec)?;

        // Padding is masked out of the head's attention exactly as `src_key_padding_mask` does.
        let bias = ((1.0 - &attention_mask)? * MASKED_SCORE)?.reshape((b, 1, 1, l))?;
        for layer in &self.head {
            hs = layer.forward(&hs, &bias)?;
        }

        // One hidden state per marker, then a scalar score per option.
        let idx = marker_pos.unsqueeze(2)?.expand((b, k, self.hidden))?.contiguous()?;
        let markers = hs.gather(&idx, 1)?;
        let logits = self
            .scorer_fc2
            .forward(&self.scorer_fc1.forward(&self.scorer_norm.forward(&markers)?)?.gelu_erf()?)?
            .squeeze(2)?;

        let masked = Tensor::full(MASKED_OPTION_LOGIT, (b, k), dev)?;
        let logits = marker_mask.where_cond(&logits, &masked)?;

        // Confidence features for the action head, mirroring the training-time definition.
        let p = softmax_last_dim(&logits)?;
        let n_options = marker_mask.to_dtype(DTYPE)?.sum_keepdim(1)?.clamp(2f32, f32::MAX)?;
        let ent = (p.clamp(1e-9f32, 1f32)?.log()? * &p)?
            .sum_keepdim(1)?
            .neg()?
            .broadcast_div(&n_options.log()?)?;
        let (sorted, _) = p.sort_last_dim(false)?;
        let top1 = sorted.narrow(1, 0, 1)?;
        let top2 = sorted.narrow(1, 1, 1)?;
        let feats = Tensor::cat(&[top1.clone(), (&top1 - &top2)?, ent, (n_options / 255.0)?], 1)?;

        let pooled = hs.i((.., 0, ..))?;
        let act_logits = self
            .act_fc2
            .forward(&self.act_fc1.forward(&Tensor::cat(&[pooled, feats], 1)?)?.gelu_erf()?)?;
        let act_probs = softmax_last_dim(&act_logits)?;

        Ok(Forward { logits: logits.to_vec2::<f32>()?, act_probs: act_probs.to_vec2::<f32>()? })
    }
}
