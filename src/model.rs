//! The decision model: a ModernBERT backbone plus the typed decision head.
//!
//! The head is a faithful port of the PyTorch module the checkpoints were trained with:
//! a type embedding added to every position, two pre-norm transformer layers, a scorer that
//! reads the `[MASK]` markers, and an auxiliary action head fed by the scorer's own confidence.

use std::collections::HashMap;
use std::path::Path;

use candle_core::safetensors::Load;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::attention::{Kernel, Packing, attend};
use crate::encoder::ModernBert;
use crate::error::{Error, Result};
use crate::question::QType;

/// Everything in this crate runs in f32, which is what the reference implementation falls back
/// to on CPU and MPS.
pub(crate) const DTYPE: DType = DType::F32;

/// Read `model.safetensors` onto the CPU, every floating-point tensor already in [`DTYPE`].
///
/// The checkpoints ship half-precision weights. Widening them here, in parallel and straight
/// from the file bytes, skips both the intermediate half-precision copy and candle's
/// single-threaded `to_dtype`; the widening is exact, so the weights are the same either way.
pub(crate) fn load_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path.display(), e))?;
    let file = SafeTensors::deserialize(&bytes)
        .map_err(|e| Error::Checkpoint(format!("{}: {e}", path.display())))?;

    fn widen<const N: usize>(data: &[u8], to_f32: impl Fn([u8; N]) -> f32 + Sync) -> Vec<f32> {
        data.par_chunks_exact(N)
            .with_min_len(1 << 16)
            .map(|b| to_f32(b.try_into().expect("chunks are exactly N bytes")))
            .collect()
    }

    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let data = view.data();
            let widened = match view.dtype() {
                Dtype::F16 => widen(data, |b| half::f16::from_le_bytes(b).to_f32()),
                Dtype::BF16 => widen(data, |b| half::bf16::from_le_bytes(b).to_f32()),
                Dtype::F32 => widen(data, f32::from_le_bytes),
                _ => return Ok((name, view.load(&Device::Cpu)?)),
            };
            Ok((name, Tensor::from_vec(widened, view.shape(), &Device::Cpu)?))
        })
        .collect()
}

/// `nn.TransformerEncoderLayer` keeps this eps, and so must we. A bare eps is an affine,
/// mean-removing `LayerNormConfig`, which is what PyTorch's `LayerNorm` is.
const HEAD_LN_EPS: f64 = 1e-5;
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

    /// Padding never takes part, exactly as `src_key_padding_mask` keeps it out of the reference.
    fn forward(&self, xs: &Tensor, packing: &Packing) -> Result<Tensor> {
        let tokens = xs.dim(0)?;
        let qkv = self.in_proj.forward(xs)?.reshape((tokens, 3, self.n_heads, self.head_dim))?;
        let part = |i: usize| qkv.narrow(1, i, 1)?.squeeze(1);
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);

        let scale = (self.head_dim as f64).powf(-0.5);
        Ok(self.out_proj.forward(&attend(&q, &k, &v, packing, scale, false)?)?)
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

    fn forward(&self, xs: &Tensor, packing: &Packing) -> Result<Tensor> {
        let xs = (xs + self.self_attn.forward(&self.norm1.forward(xs)?, packing)?)?;
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
    encoder: ModernBert,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer_fc1: Linear,
    scorer_fc2: Linear,
    act_fc1: Linear,
    act_fc2: Linear,
    hidden: usize,
    device: Device,
    kernel: Kernel,
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

        Self::from_vb(
            VarBuilder::from_tensors(remapped, DTYPE, device),
            enc_cfg,
            head_layers,
            n_act,
        )
    }

    /// Build the model from whatever `vb` holds, in the in-memory `encoder.model.*` layout.
    fn from_vb(
        vb: VarBuilder,
        enc_cfg: &modernbert::Config,
        head_layers: usize,
        n_act: usize,
    ) -> Result<Self> {
        let hidden = enc_cfg.hidden_size;
        // The training code derives the head's head count from the width, not from the encoder.
        let n_heads = std::cmp::max(1, hidden / 64);

        let encoder = ModernBert::load(vb.pp("encoder"), enc_cfg)?;
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
            device: vb.device().clone(),
            kernel: Kernel::for_device(vb.device()),
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn forward(&self, batch: &crate::sequence::Batch) -> Result<Forward> {
        let dev = &self.device;
        let (b, k) = (batch.batch, batch.k_max);
        let packing = Packing::new(&batch.lens, self.encoder.window(), self.kernel, dev)?;

        let input_ids = Tensor::from_slice(&batch.input_ids, batch.n_tokens, dev)?;
        let marker_pos = Tensor::from_slice(&batch.marker_pos, b * k, dev)?;
        let marker_mask = Tensor::from_slice(&batch.marker_mask, (b, k), dev)?;
        let qtype = Tensor::from_slice(&batch.qtype, b, dev)?;

        let hs = self.encoder.forward(&input_ids, &packing)?;

        // The question type is a global signal, so it is added to every token of its row.
        let type_vec = self.type_emb.forward(&qtype)?.index_select(packing.row_of_token(), 0)?;
        let mut hs = (hs + type_vec)?;

        for layer in &self.head {
            hs = layer.forward(&hs, &packing)?;
        }

        // One hidden state per marker, then a scalar score per option.
        let markers = hs.index_select(&marker_pos, 0)?.reshape((b, k, self.hidden))?;
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

        let pooled = hs.index_select(packing.row_starts(), 0)?;
        let act_logits = self
            .act_fc2
            .forward(&self.act_fc1.forward(&Tensor::cat(&[pooled, feats], 1)?)?.gelu_erf()?)?;
        let act_probs = softmax_last_dim(&act_logits)?;

        Ok(Forward { logits: logits.to_vec2::<f32>()?, act_probs: act_probs.to_vec2::<f32>()? })
    }
}

/// Randomly initialised tensors for a model of this shape, keyed as `model.safetensors` stores
/// them, so tests can load a real (if tiny) checkpoint without downloading one.
#[cfg(test)]
pub(crate) fn random_weights(
    enc_cfg: &modernbert::Config,
    head_layers: usize,
    n_act: usize,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DTYPE, &Device::Cpu);
    DecisionModel::from_vb(vb, enc_cfg, head_layers, n_act)?;
    let tensors = varmap.data().lock().expect("no other thread holds the VarMap");
    Ok(tensors
        .iter()
        .map(|(k, v)| {
            let key = match k.strip_prefix("encoder.model.") {
                Some(rest) => format!("encoder.{rest}"),
                None => k.clone(),
            };
            (key, v.as_tensor().clone())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{Item, collate};

    fn tiny_config() -> modernbert::Config {
        modernbert::Config {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            intermediate_size: 24,
            max_position_embeddings: 64,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160_000.0,
            local_attention: 8,
            local_rope_theta: 10_000.0,
            classifier_config: None,
        }
    }

    fn item(len: usize, n_options: usize, qtype: QType) -> Item {
        Item {
            ids: (0..len).map(|i| ((i * 7 + len) % 32) as u32).collect(),
            markers: (0..n_options).map(|i| 2 + 3 * i).collect(),
            qtype,
            labels: (0..n_options).map(|i| i.to_string()).collect(),
        }
    }

    #[test]
    fn weights_load_widened_to_f32_and_otherwise_unchanged() {
        let dev = &Device::Cpu;
        let values = Tensor::new(&[0.1f32, -2.5, 65504.0, 1e-7], dev).unwrap();
        let stored: HashMap<String, Tensor> = [
            ("f16", values.to_dtype(DType::F16).unwrap()),
            ("bf16", values.to_dtype(DType::BF16).unwrap()),
            ("f32", values.reshape((2, 2)).unwrap()),
            ("ids", Tensor::new(&[7u32, 9], dev).unwrap()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let dir = crate::testutil::scratch_dir("load-weights");
        let path = dir.join("model.safetensors");
        candle_core::safetensors::save(&stored, &path).unwrap();

        let loaded = load_weights(&path).unwrap();
        for (name, original) in &stored {
            let got = &loaded[name];
            assert_eq!(got.dims(), original.dims(), "{name}");
            let expected = match original.dtype() {
                DType::U32 => original.clone(),
                _ => original.to_dtype(DType::F32).unwrap(),
            };
            assert_eq!(got.dtype(), expected.dtype(), "{name}");
            let flat = |t: &Tensor| t.flatten_all().unwrap().to_dtype(DType::F64).unwrap();
            assert_eq!(
                flat(got).to_vec1::<f64>().unwrap(),
                flat(&expected).to_vec1::<f64>().unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_packed_row_reads_the_same_as_the_row_alone() {
        let cfg = tiny_config();
        let mut model =
            DecisionModel::load(random_weights(&cfg, 2, 2).unwrap(), &cfg, 2, 2, &Device::Cpu)
                .unwrap();
        // Rows of different lengths, all longer than the sliding window.
        let items =
            [item(13, 2, QType::Noul), item(21, 4, QType::Choice), item(9, 3, QType::Score)];

        let mut by_kernel = Vec::new();
        for kernel in [Kernel::Varlen, Kernel::Dense] {
            model.kernel = kernel;
            let together = model.forward(&collate(&items)).unwrap();
            for (row, it) in items.iter().enumerate() {
                let alone = model.forward(&collate(std::slice::from_ref(it))).unwrap();
                let k = it.markers.len();
                for (a, b) in together.logits[row][..k].iter().zip(&alone.logits[0][..k]) {
                    assert!((a - b).abs() < 1e-4, "{kernel:?} row {row}: {a} vs {b}");
                }
                for (a, b) in together.act_probs[row].iter().zip(&alone.act_probs[0]) {
                    assert!((a - b).abs() < 1e-5, "{kernel:?} row {row}: {a} vs {b}");
                }
            }
            by_kernel.push(together);
        }

        let (varlen, dense) = (&by_kernel[0], &by_kernel[1]);
        for (row, it) in items.iter().enumerate() {
            let k = it.markers.len();
            for (a, b) in varlen.logits[row][..k].iter().zip(&dense.logits[row][..k]) {
                assert!((a - b).abs() < 1e-4, "row {row}: varlen {a} vs dense {b}");
            }
        }
    }
}
