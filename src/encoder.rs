//! The ModernBERT backbone, run over packed rows.
//!
//! Adapted from candle-transformers' `modernbert`, whose weights layout and arithmetic it keeps.
//! What changes is the shape everything flows in: rows are packed end to end as
//! `[tokens, hidden]` rather than padded to `[B, L, hidden]` (see [`crate::attention`]), so no
//! layer computes anything for a padding position, and attention runs through the kernel that
//! suits the device.

use candle_core::{DType, Device, Tensor};
use candle_nn::rotary_emb::rope_thd;
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::Config;

use crate::attention::{Packing, attend};
use crate::error::Result;
use crate::kernels::{Norm, geglu};

/// RoPE's rotation for every position up to `max_position_embeddings`.
struct Rope {
    /// `[max_positions, head_dim / 2]`.
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(dtype: DType, config: &Config, theta: f64, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> =
            (0..dim).step_by(2).map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32).collect();
        let n_freq = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n_freq), device)?.to_dtype(dtype)?;
        let max_pos = config.max_position_embeddings;
        let t =
            Tensor::arange(0u32, max_pos as u32, device)?.to_dtype(dtype)?.reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    /// The rotation of every packed token at its own position, as `[tokens, 1, head_dim / 2]`:
    /// [`rope_thd`] then reads each token as a batch of one.
    fn at(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let gather = |table: &Tensor| -> Result<Tensor> {
            Ok(table.index_select(positions, 0)?.unsqueeze(1)?)
        };
        Ok((gather(&self.cos)?, gather(&self.sin)?))
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(
        &self,
        xs: &Tensor,
        packing: &Packing,
        (cos, sin): &(Tensor, Tensor),
        local: bool,
    ) -> Result<Tensor> {
        let tokens = xs.dim(0)?;
        let qkv = xs.apply(&self.qkv)?.reshape((tokens, 3, self.n_heads, self.head_dim))?;
        let part = |i: usize| qkv.narrow(1, i, 1);
        let rotate = |i: usize| -> Result<Tensor> {
            Ok(rope_thd(&part(i)?.contiguous()?, cos, sin)?.squeeze(1)?)
        };
        let (q, k, v) = (rotate(0)?, rotate(1)?, part(2)?.squeeze(1)?);

        let scale = (self.head_dim as f64).powf(-0.5);
        Ok(attend(&q, &k, &v, packing, scale, local)?.apply(&self.proj)?)
    }
}

struct Layer {
    attn: Attention,
    /// Layer 0 normalises in the embeddings instead, so it has none.
    attn_norm: Option<Norm>,
    mlp_norm: Norm,
    wi: Linear,
    wo: Linear,
    local: bool,
}

impl Layer {
    fn load(vb: VarBuilder, config: &Config, local: bool) -> Result<Self> {
        let (h, eps) = (config.hidden_size, config.layer_norm_eps);
        Ok(Self {
            attn: Attention {
                qkv: candle_nn::linear_no_bias(h, 3 * h, vb.pp("attn.Wqkv"))?,
                proj: candle_nn::linear_no_bias(h, h, vb.pp("attn.Wo"))?,
                n_heads: config.num_attention_heads,
                head_dim: h / config.num_attention_heads,
            },
            attn_norm: Norm::load(h, eps, vb.pp("attn_norm")).ok(),
            mlp_norm: Norm::load(h, eps, vb.pp("mlp_norm"))?,
            wi: candle_nn::linear_no_bias(h, 2 * config.intermediate_size, vb.pp("mlp.Wi"))?,
            wo: candle_nn::linear_no_bias(config.intermediate_size, h, vb.pp("mlp.Wo"))?,
            local,
        })
    }

    fn forward(&self, xs: &Tensor, packing: &Packing, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let xs = (xs + self.attn.forward(&normed, packing, rope, self.local)?)?;

        let h = geglu(&xs.apply(&self.mlp_norm)?.apply(&self.wi)?)?.apply(&self.wo)?;
        Ok((xs + h)?)
    }
}

/// The ModernBERT encoder.
pub(crate) struct ModernBert {
    embeddings: Embedding,
    norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    global_rope: Rope,
    local_rope: Rope,
    window: usize,
}

impl ModernBert {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (h, eps) = (config.hidden_size, config.layer_norm_eps);
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let local = i % config.global_attn_every_n_layers != 0;
                Layer::load(vb.pp(format!("model.layers.{i}")), config, local)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings: candle_nn::embedding(
                config.vocab_size,
                h,
                vb.pp("model.embeddings.tok_embeddings"),
            )?,
            norm: Norm::load(h, eps, vb.pp("model.embeddings.norm"))?,
            layers,
            final_norm: Norm::load(h, eps, vb.pp("model.final_norm"))?,
            global_rope: Rope::new(vb.dtype(), config, config.global_rope_theta, vb.device())?,
            local_rope: Rope::new(vb.dtype(), config, config.local_rope_theta, vb.device())?,
            window: config.local_attention / 2,
        })
    }

    /// How many tokens a sliding-window layer sees on either side of each token.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// `[tokens]` packed ids in, `[tokens, hidden]` final hidden states out.
    pub(crate) fn forward(&self, input_ids: &Tensor, packing: &Packing) -> Result<Tensor> {
        let global = self.global_rope.at(packing.positions())?;
        let local = self.local_rope.at(packing.positions())?;

        let mut xs = self.embeddings.forward(input_ids)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, packing, if layer.local { &local } else { &global })?;
        }
        Ok(xs.apply(&self.final_norm)?)
    }
}
