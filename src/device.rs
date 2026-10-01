//! Choosing where to run.
//!
//! A device is named `auto`, `cpu`, `cuda`, `cuda:N`, `metal` or `metal:N`. The library, the
//! CLI's `--device` and the `LAYA_DEVICE` environment variable all take the same spelling.

use std::str::FromStr;

use candle_core::Device;

use crate::error::{Error, Result};

/// The environment variable that picks the device when the caller does not.
pub const DEVICE_ENV: &str = "LAYA_DEVICE";

/// Where the model should run, before it is resolved into a candle [`Device`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The best accelerator this build can reach, else the CPU.
    #[default]
    Auto,
    Cpu,
    /// An NVIDIA GPU, by index. Needs the `cuda` feature.
    Cuda(usize),
    /// An Apple GPU, by index. Needs the `metal` feature.
    Metal(usize),
}

impl FromStr for DeviceChoice {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let spec = s.trim().to_lowercase();
        let (kind, index) = match spec.split_once(':') {
            Some((kind, n)) => {
                let n = n.parse().map_err(|_| {
                    Error::Device(format!("{s:?}: the device index must be a number"))
                })?;
                (kind, Some(n))
            }
            None => (spec.as_str(), None),
        };
        match (kind, index) {
            ("auto", None) => Ok(DeviceChoice::Auto),
            ("cpu", None) => Ok(DeviceChoice::Cpu),
            ("cuda" | "gpu", n) => Ok(DeviceChoice::Cuda(n.unwrap_or(0))),
            ("metal" | "mps", n) => Ok(DeviceChoice::Metal(n.unwrap_or(0))),
            _ => Err(Error::Device(format!(
                "unknown device {s:?}; choose auto, cpu, cuda, cuda:N, metal or metal:N"
            ))),
        }
    }
}

impl std::fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceChoice::Auto => f.write_str("auto"),
            DeviceChoice::Cpu => f.write_str("cpu"),
            DeviceChoice::Cuda(n) => write!(f, "cuda:{n}"),
            DeviceChoice::Metal(n) => write!(f, "metal:{n}"),
        }
    }
}

impl DeviceChoice {
    /// The choice `LAYA_DEVICE` names, or [`Auto`](DeviceChoice::Auto) when it is unset.
    pub fn from_env() -> Result<Self> {
        match std::env::var(DEVICE_ENV) {
            Ok(v) if !v.trim().is_empty() => v.parse().map_err(|e| match e {
                Error::Device(m) => Error::Device(format!("{DEVICE_ENV}: {m}")),
                other => other,
            }),
            _ => Ok(DeviceChoice::Auto),
        }
    }

    /// Open the device.
    ///
    /// An explicit accelerator that cannot be opened is an error, never a silent CPU run: a
    /// build without its feature, a missing driver or a bad index all say so. `Auto` falls
    /// back to the CPU, and warns when an accelerator was compiled in but failed to open.
    pub fn resolve(self) -> Result<Device> {
        match self {
            DeviceChoice::Cpu => Ok(Device::Cpu),
            DeviceChoice::Cuda(n) => CUDA.open(n),
            DeviceChoice::Metal(n) => METAL.open(n),
            DeviceChoice::Auto => Ok(default_device()),
        }
    }
}

/// A GPU backend: its name, whether this build has its feature, and candle's opener, which
/// exists either way and fails without the feature.
#[derive(Clone, Copy)]
struct Accelerator {
    name: &'static str,
    compiled: bool,
    new: fn(usize) -> candle_core::Result<Device>,
}

const CUDA: Accelerator =
    Accelerator { name: "cuda", compiled: cfg!(feature = "cuda"), new: Device::new_cuda };
const METAL: Accelerator =
    Accelerator { name: "metal", compiled: cfg!(feature = "metal"), new: Device::new_metal };

impl Accelerator {
    /// Open GPU `n`, saying why when it cannot be.
    fn open(self, n: usize) -> Result<Device> {
        let name = self.name;
        if !self.compiled {
            return Err(Error::Device(format!(
                "{name}:{n} was asked for, but this build has the `{name}` feature off; rebuild \
                 with `--features {name}`"
            )));
        }
        (self.new)(n).map_err(|e| Error::Device(format!("cannot open {name}:{n}: {e}")))
    }
}

/// The device `LAYA_DEVICE` names, or the best one this build can reach when it is unset.
///
/// Compile with `--features cuda` or `--features metal` to make the accelerators available.
pub fn device_from_env() -> Result<Device> {
    DeviceChoice::from_env()?.resolve()
}

/// The best device this build can use: CUDA, then Metal, then the CPU, warning about any
/// compiled-in accelerator that failed to open.
///
/// Ignores `LAYA_DEVICE` and never fails; prefer [`device_from_env`], which honours it.
pub fn default_device() -> Device {
    let compiled = || [CUDA, METAL].into_iter().filter(|a| a.compiled);
    for Accelerator { name, new, .. } in compiled() {
        match new(0) {
            Ok(d) => return d,
            Err(e) => {
                eprintln!("[laya] built with `{name}` but {name}:0 did not open ({e}); trying next")
            }
        }
    }
    if compiled().next().is_some() {
        eprintln!("[laya] no accelerator available; running on the CPU");
    }
    Device::Cpu
}

/// A short name for a device, for logs and the CLI.
pub fn describe(device: &Device) -> String {
    let choice = match device.location() {
        candle_core::DeviceLocation::Cpu => DeviceChoice::Cpu,
        candle_core::DeviceLocation::Cuda { gpu_id } => DeviceChoice::Cuda(gpu_id),
        candle_core::DeviceLocation::Metal { gpu_id } => DeviceChoice::Metal(gpu_id),
    };
    choice.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_names_parse() {
        for (s, want) in [
            ("auto", DeviceChoice::Auto),
            ("CPU", DeviceChoice::Cpu),
            ("cuda", DeviceChoice::Cuda(0)),
            ("cuda:1", DeviceChoice::Cuda(1)),
            ("gpu", DeviceChoice::Cuda(0)),
            ("metal", DeviceChoice::Metal(0)),
            ("mps:2", DeviceChoice::Metal(2)),
        ] {
            assert_eq!(s.parse::<DeviceChoice>().unwrap(), want, "{s}");
        }
        for bad in ["tpu", "cuda:x", "cpu:1", ""] {
            assert!(bad.parse::<DeviceChoice>().is_err(), "{bad}");
        }
        assert_eq!(DeviceChoice::Cuda(1).to_string(), "cuda:1");
    }

    #[test]
    fn the_cpu_always_resolves() {
        assert!(DeviceChoice::Cpu.resolve().unwrap().is_cpu());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn an_explicit_gpu_without_its_feature_is_an_error_not_a_cpu_run() {
        let err = DeviceChoice::Cuda(0).resolve().unwrap_err();
        assert!(matches!(err, Error::Device(ref m) if m.contains("--features cuda")), "{err}");
    }
}
