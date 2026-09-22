//! Choosing where to run.

use candle_core::Device;

/// The best device this build can use: CUDA, then Metal, then the CPU.
///
/// Compile with `--features cuda` or `--features metal` to make the accelerators available;
/// without them this is always the CPU.
pub fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}
