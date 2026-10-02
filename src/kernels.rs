//! Fused CPU kernels for the elementwise steps candle runs on a single thread.
//!
//! candle's CPU unary and binary ops do not parallelise, and its bias-free LayerNorm is
//! composed of seven of them, so on a many-core CPU these steps end up costing as much as the
//! matmuls around them. Each kernel here makes one parallel pass over the rows instead. Any
//! other device keeps candle's own ops.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, D, DType, Layout, Shape, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};
use rayon::prelude::*;

use crate::error::Result;

/// Fewer elements than this per task and waking a thread costs more than the work it takes.
const MIN_TASK_ELEMS: usize = 1 << 15;

/// Rows of `width` elements to hand each task.
fn rows_per_task(width: usize) -> usize {
    (MIN_TASK_ELEMS / width.max(1)).max(1)
}

/// ModernBERT's gated MLP activation: `gelu(a) * b`, `a` and `b` being the two halves of the
/// last dimension.
pub(crate) fn geglu(xs: &Tensor) -> Result<Tensor> {
    if xs.device().is_cpu() && xs.dtype() == DType::F32 {
        return Ok(xs.contiguous()?.apply_op1_no_bwd(&GeGlu)?);
    }
    let halves = xs.chunk(2, D::Minus1)?;
    Ok((halves[0].gelu_erf()? * &halves[1])?)
}

struct GeGlu;

impl CustomOp1 for GeGlu {
    fn name(&self) -> &'static str {
        "geglu"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let src = f32_rows(storage, layout, "geglu")?;
        let mut dims = layout.dims().to_vec();
        let width = dims.last().copied().unwrap_or(0);
        if width % 2 != 0 {
            candle_core::bail!("geglu: the last dimension ({width}) must split in two halves");
        }
        let half = width / 2;
        if let Some(last) = dims.last_mut() {
            *last = half;
        }

        let mut dst = vec![0f32; src.len() / 2];
        let rows = dst.par_chunks_mut(half.max(1)).zip(src.par_chunks(width.max(1)));
        rows.with_min_len(rows_per_task(width)).for_each(|(out, row)| {
            let (x, gate) = row.split_at(half);
            for ((o, &x), &g) in out.iter_mut().zip(x).zip(gate) {
                *o = gelu_erf(x) * g;
            }
        });
        Ok((CpuStorage::F32(dst), Shape::from(dims)))
    }
}

/// candle's own `gelu_erf` formula, so the CPU kernel and every other device agree bit for bit.
#[inline]
fn gelu_erf(v: f32) -> f32 {
    (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

/// A bias-free LayerNorm over the last dimension, as ModernBERT uses everywhere.
pub(crate) struct Norm {
    module: LayerNorm,
    /// The same weights as plain floats, for the CPU kernel.
    cpu: Option<RowNorm>,
}

impl Norm {
    pub(crate) fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Self::new(vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?, eps)
    }

    fn new(weight: Tensor, eps: f64) -> Result<Self> {
        let cpu = if weight.device().is_cpu() && weight.dtype() == DType::F32 {
            Some(RowNorm { weight: weight.to_vec1()?, eps })
        } else {
            None
        };
        Ok(Self { module: LayerNorm::new_no_bias(weight, eps), cpu })
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match &self.cpu {
            Some(op) if xs.device().is_cpu() => xs.contiguous()?.apply_op1_no_bwd(op),
            _ => self.module.forward(xs),
        }
    }
}

struct RowNorm {
    weight: Vec<f32>,
    eps: f64,
}

impl CustomOp1 for RowNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let src = f32_rows(storage, layout, "layer-norm")?;
        let width = self.weight.len();
        if layout.dims().last() != Some(&width) {
            candle_core::bail!("layer-norm: expected rows of {width}, got {:?}", layout.dims());
        }

        // Two passes, mean then the variance around it, accumulated in f64: the residual stream
        // carries outliers large enough that the one-pass `E[x²] - E[x]²` would lose digits.
        let mut dst = vec![0f32; src.len()];
        let rows = dst.par_chunks_mut(width.max(1)).zip(src.par_chunks(width.max(1)));
        rows.with_min_len(rows_per_task(width)).for_each(|(out, row)| {
            let n = row.len() as f64;
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for ((o, &v), &w) in out.iter_mut().zip(row).zip(&self.weight) {
                *o = ((f64::from(v) - mean) * inv_std) as f32 * w;
            }
        });
        Ok((CpuStorage::F32(dst), layout.shape().clone()))
    }
}

/// The contiguous f32 data a kernel reads.
fn f32_rows<'a>(
    storage: &'a CpuStorage,
    layout: &Layout,
    op: &str,
) -> candle_core::Result<&'a [f32]> {
    let CpuStorage::F32(data) = storage else {
        candle_core::bail!("{op}: f32 only, got {:?}", storage.dtype());
    };
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("{op}: the input must be contiguous"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn sample(rows: usize, width: usize) -> Tensor {
        // Large offsets on a few columns, as ModernBERT's residual stream has.
        let data: Vec<f32> = (0..rows * width)
            .map(|i| {
                let x = ((i * 7919) % 1000) as f32 / 100.0 - 5.0;
                if i % width == 3 { x * 200.0 } else { x }
            })
            .collect();
        Tensor::from_vec(data, (rows, width), &Device::Cpu).unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar().unwrap()
    }

    #[test]
    fn geglu_matches_candles_composed_ops_bit_for_bit() {
        let xs = sample(5, 12);
        let halves = xs.chunk(2, D::Minus1).unwrap();
        let reference = (halves[0].gelu_erf().unwrap() * &halves[1]).unwrap();
        let fused = geglu(&xs).unwrap();
        assert_eq!(fused.dims(), &[5, 6]);
        assert_eq!(max_diff(&fused, &reference), 0.0);
    }

    #[test]
    fn layer_norm_matches_candles_module() {
        // A weight with some shape, so a misplaced or missing multiply would show.
        let w: Vec<f32> = (0..64).map(|i| 0.5 + i as f32 / 64.0).collect();
        let norm = Norm::new(Tensor::from_vec(w, 64, &Device::Cpu).unwrap(), 1e-5).unwrap();
        assert!(norm.cpu.is_some());

        let xs = sample(9, 64);
        let fused = norm.forward(&xs).unwrap();
        let reference = norm.module.forward(&xs).unwrap();
        assert!(max_diff(&fused, &reference) < 1e-4, "{}", max_diff(&fused, &reference));
    }
}
