//! Attention over packed rows.
//!
//! A batch is not padded to its longest row: the rows sit end to end as one `[tokens, hidden]`
//! matrix, so the linear layers, which hold nearly all of the compute, never touch a padding
//! position. Attention is the only step that needs to know where one row ends and the next
//! begins, and [`Packing`] is what tells it.

use candle_core::{Device, Tensor};
use candle_nn::attention::flash_attn_varlen_cpu;
use candle_nn::ops::softmax_last_dim;

use crate::error::Result;

/// How attention scores are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// candle's fused variable-length kernel: every token attends to its own row only, and no
    /// score matrix is materialised. CPU only.
    ///
    /// The batched matmul it replaces runs one small multithreaded GEMM per (row, head), and
    /// on a CPU the thread hand-off of each one costs more than the arithmetic.
    Varlen,
    /// Scatter back to a padded `[B, H, L, L]` grid and run two batched matmuls, with padding
    /// and the sliding window masked out additively. Runs on any device.
    Dense,
}

impl Kernel {
    /// The fastest kernel that runs on `device`.
    pub(crate) fn for_device(device: &Device) -> Self {
        if device.is_cpu() { Kernel::Varlen } else { Kernel::Dense }
    }
}

/// Where each row of a batch sits once the rows are laid end to end.
pub(crate) struct Packing {
    /// Tokens in each row.
    seqlens: Tensor,
    max_len: usize,
    /// How far a token sees in a sliding-window layer, on either side.
    window: usize,
    /// The position of every token within its own row, which is what RoPE rotates by.
    positions: Tensor,
    /// The row every token belongs to.
    row_of_token: Tensor,
    /// The first token of every row: its `[CLS]`.
    row_starts: Tensor,
    /// The padded layout, which only [`Kernel::Dense`] needs.
    dense: Option<DenseLayout>,
}

struct DenseLayout {
    /// For every slot of the padded `[B, L]` grid, the packed token it holds. Padding slots
    /// repeat token 0; they are masked out of every key and never read back.
    pad: Tensor,
    /// For every packed token, its slot in the padded grid.
    unpad: Tensor,
    /// `[B, 1, 1, L]`, added to the scores of a global layer: 0 on tokens, `f32::MIN` on padding.
    global_bias: Tensor,
    /// `[B, 1, L, L]`, the same plus `-inf` beyond the window, for a sliding-window layer.
    local_bias: Tensor,
}

impl Packing {
    /// Lay out rows of `lens` tokens for `kernel`, with sliding-window layers seeing `window`
    /// tokens on either side.
    pub(crate) fn new(
        lens: &[usize],
        window: usize,
        kernel: Kernel,
        device: &Device,
    ) -> Result<Self> {
        let total: usize = lens.iter().sum();
        let max_len = lens.iter().copied().max().unwrap_or(0);

        let mut positions = Vec::with_capacity(total);
        let mut row_of_token = Vec::with_capacity(total);
        let mut row_starts = Vec::with_capacity(lens.len());
        for (row, &len) in lens.iter().enumerate() {
            row_starts.push(positions.len() as u32);
            positions.extend(0..len as u32);
            row_of_token.extend(std::iter::repeat_n(row as u32, len));
        }
        let seqlens: Vec<u32> = lens.iter().map(|&l| l as u32).collect();

        let dense = match kernel {
            Kernel::Varlen => None,
            Kernel::Dense => Some(DenseLayout::new(lens, max_len, window, device)?),
        };

        Ok(Self {
            seqlens: Tensor::from_vec(seqlens, lens.len(), device)?,
            max_len,
            window,
            positions: Tensor::from_vec(positions, total, device)?,
            row_of_token: Tensor::from_vec(row_of_token, total, device)?,
            row_starts: Tensor::from_vec(row_starts, lens.len(), device)?,
            dense,
        })
    }

    pub(crate) fn positions(&self) -> &Tensor {
        &self.positions
    }

    pub(crate) fn row_of_token(&self) -> &Tensor {
        &self.row_of_token
    }

    pub(crate) fn row_starts(&self) -> &Tensor {
        &self.row_starts
    }
}

impl DenseLayout {
    fn new(lens: &[usize], max_len: usize, window: usize, device: &Device) -> Result<Self> {
        let (b, l) = (lens.len(), max_len);
        let mut pad = vec![0u32; b * l];
        let mut unpad = Vec::with_capacity(lens.iter().sum());
        let mut key = vec![f32::MIN; b * l];
        let mut start = 0;
        for (row, &len) in lens.iter().enumerate() {
            for pos in 0..len {
                pad[row * l + pos] = (start + pos) as u32;
                unpad.push((row * l + pos) as u32);
                key[row * l + pos] = 0.0;
            }
            start += len;
        }
        let window_mask: Vec<f32> = (0..l)
            .flat_map(|i| {
                (0..l).map(move |j| if i.abs_diff(j) > window { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();

        let global_bias = Tensor::from_vec(key, (b, 1, 1, l), device)?;
        let local_bias =
            global_bias.broadcast_add(&Tensor::from_vec(window_mask, (l, l), device)?)?;
        let n_unpad = unpad.len();
        Ok(Self {
            pad: Tensor::from_vec(pad, b * l, device)?,
            unpad: Tensor::from_vec(unpad, n_unpad, device)?,
            global_bias,
            local_bias,
        })
    }
}

/// Softmax attention of every token over the tokens of its own row.
///
/// `q`, `k` and `v` are `[tokens, heads, head_dim]`, and the result is
/// `[tokens, heads * head_dim]`. Scores are scaled by `scale`; `local` restricts each token to
/// the keys within [`Packing`]'s window, as ModernBERT's sliding-window layers do.
pub(crate) fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    packing: &Packing,
    scale: f64,
    local: bool,
) -> Result<Tensor> {
    let (tokens, heads, head_dim) = q.dims3()?;
    let width = heads * head_dim;

    let Some(dense) = &packing.dense else {
        let window = local.then_some(packing.window);
        let out = flash_attn_varlen_cpu(
            q,
            k,
            v,
            None,
            &packing.seqlens,
            &packing.seqlens,
            packing.max_len,
            packing.max_len,
            scale as f32,
            false,
            window,
            window,
        )?;
        return Ok(out.reshape((tokens, width))?);
    };

    let (b, l) = (packing.seqlens.dim(0)?, packing.max_len);
    let pad = |x: &Tensor| -> Result<Tensor> {
        Ok(x.reshape((tokens, width))?
            .index_select(&dense.pad, 0)?
            .reshape((b, l, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    };
    let (q, k, v) = (pad(q)?, pad(k)?, pad(v)?);

    let bias = if local { &dense.local_bias } else { &dense.global_bias };
    let scores = (q * scale)?.matmul(&k.t()?)?.broadcast_add(bias)?;
    let out = softmax_last_dim(&scores)?.matmul(&v)?;
    Ok(out.transpose(1, 2)?.reshape((b * l, width))?.index_select(&dense.unpad, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar().unwrap()
    }

    #[test]
    fn the_varlen_and_dense_kernels_agree() {
        // Uneven rows, one of them wider than the window on both sides.
        let (lens, window) = ([5, 11, 3], 2);
        let total = lens.iter().sum::<usize>();
        let rand = || Tensor::randn(0f32, 1., (total, 2, 4), &Device::Cpu).unwrap();
        let (q, k, v) = (rand(), rand(), rand());

        let run = |kernel: Kernel, local: bool| {
            let packing = Packing::new(&lens, window, kernel, &Device::Cpu).unwrap();
            attend(&q, &k, &v, &packing, 0.5, local).unwrap()
        };
        for local in [false, true] {
            let (varlen, dense) = (run(Kernel::Varlen, local), run(Kernel::Dense, local));
            assert_eq!(varlen.dims(), &[total, 8]);
            assert!(
                max_diff(&varlen, &dense) < 1e-5,
                "local={local}: {}",
                max_diff(&varlen, &dense)
            );
        }
        // The window does cut the long row: a global layer would read more of it.
        assert!(max_diff(&run(Kernel::Varlen, false), &run(Kernel::Varlen, true)) > 1e-3);
    }

    #[test]
    fn a_packing_knows_where_every_row_starts() {
        let packing = Packing::new(&[3, 1, 2], 4, Kernel::Varlen, &Device::Cpu).unwrap();
        let v = |t: &Tensor| t.to_vec1::<u32>().unwrap();
        assert_eq!(v(packing.positions()), [0, 1, 2, 0, 0, 1]);
        assert_eq!(v(packing.row_of_token()), [0, 0, 0, 1, 2, 2]);
        assert_eq!(v(packing.row_starts()), [0, 3, 4]);
    }
}
