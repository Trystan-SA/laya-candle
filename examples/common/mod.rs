//! What every example prints: how long each step took, and what it cost in memory.
//!
//! Not a Cargo target — `examples/common/mod.rs` has no `main`, so auto-discovery skips it.
//! Examples pull it in with a plain `mod common;`.

// Each target uses the part of this it needs; the rest is not dead, just unused here.
#![allow(dead_code)]

use std::time::Instant;

/// A running report: lap the interesting steps, then finish.
pub struct Report {
    started: Instant,
    last: Instant,
}

impl Report {
    pub fn new() -> Self {
        let now = Instant::now();
        Self { started: now, last: now }
    }

    /// Print the time since the previous lap, and the memory held right now.
    pub fn lap(&mut self, what: &str) {
        let now = Instant::now();
        println!(
            "  [{:>7}]  {what}{}",
            secs(now.duration_since(self.last).as_secs_f64()),
            match rss() {
                Some(b) => format!("   (rss {})", bytes(b)),
                None => String::new(),
            }
        );
        self.last = now;
    }

    /// Print the totals: wall clock for the whole run, and the high-water mark of memory.
    pub fn finish(&self) {
        let total = self.started.elapsed().as_secs_f64();
        let mem = match (rss(), peak_rss()) {
            (Some(now), Some(peak)) => format!("rss {}, peak {}", bytes(now), bytes(peak)),
            _ => "memory not readable on this platform".to_string(),
        };
        println!("\n  total {} wall, {mem}", secs(total));
    }
}

impl Default for Report {
    fn default() -> Self {
        Self::new()
    }
}

fn secs(s: f64) -> String {
    if s < 1.0 { format!("{:.0} ms", s * 1000.0) } else { format!("{s:.2} s") }
}

/// Bytes as the size a human would say out loud.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    format!("{v:.1} {}", UNITS[unit])
}

/// Resident set size right now, in bytes.
pub fn rss() -> Option<u64> {
    proc_status("VmRSS:")
}

/// The largest resident set size this process has ever held, in bytes.
///
/// This is the number that decides whether the process fits: loading a checkpoint peaks well
/// above what it settles at.
pub fn peak_rss() -> Option<u64> {
    proc_status("VmHWM:")
}

/// Read one `kB` field out of `/proc/self/status`. Linux only; `None` anywhere else.
fn proc_status(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with(field))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}
