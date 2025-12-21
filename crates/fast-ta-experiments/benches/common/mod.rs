//! Shared benchmark utilities and standard data sizes for fast-ta experiments.
//!
//! This module provides:
//! - Standard data sizes for consistent benchmarking across experiments
//! - Criterion configuration presets for different benchmark scenarios
//! - Helper functions for benchmark setup and data generation
//!
//! # Standard Data Sizes
//!
//! All experiments should use these standard sizes for consistency:
//! - `SIZE_1K`: 1,000 points - Quick iteration, development testing
//! - `SIZE_10K`: 10,000 points - Small dataset, representative of short-term trading
//! - `SIZE_100K`: 100,000 points - Medium dataset, multi-year intraday data
//! - `SIZE_1M`: 1,000,000 points - Large dataset, stress testing
//!
//! # Example Usage
//!
//! ```ignore
//! use std::hint::black_box;
//! use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
//!
//! // Import common utilities (path depends on benchmark location)
//! mod common;
//! use common::{DATA_SIZES, SIZE_10K, criterion_config};
//!
//! fn my_benchmark(c: &mut Criterion) {
//!     // Iterate through standard sizes
//!     for &size in &DATA_SIZES {
//!         let data = generate_data(size);
//!         c.bench_with_input(
//!             BenchmarkId::new("my_operation", size),
//!             &data,
//!             |b, data| b.iter(|| black_box(operation(black_box(data)))),
//!         );
//!     }
//! }
//! ```

use std::time::Duration;

// ============================================================================
// Standard Data Sizes
// ============================================================================

/// Small dataset: 1,000 points
/// Use for quick iteration during development and basic correctness checks.
pub const SIZE_1K: usize = 1_000;

/// Medium-small dataset: 10,000 points
/// Represents short-term trading data (e.g., a few weeks of minute bars).
pub const SIZE_10K: usize = 10_000;

/// Medium dataset: 100,000 points
/// Represents multi-year intraday data or several decades of daily data.
pub const SIZE_100K: usize = 100_000;

/// Large dataset: 1,000,000 points
/// For stress testing and measuring performance at scale.
pub const SIZE_1M: usize = 1_000_000;

/// Standard data sizes array for iterating in benchmarks.
/// Includes all four sizes for comprehensive performance profiling.
pub const DATA_SIZES: [usize; 4] = [SIZE_1K, SIZE_10K, SIZE_100K, SIZE_1M];

/// Reduced data sizes for quick benchmarks (excludes 1M).
/// Use when full benchmark suite takes too long during development.
pub const QUICK_DATA_SIZES: [usize; 3] = [SIZE_1K, SIZE_10K, SIZE_100K];

// ============================================================================
// Criterion Configuration
// ============================================================================

/// Default measurement time for benchmarks (5 seconds).
/// Provides good statistical confidence for most operations.
pub const DEFAULT_MEASUREMENT_TIME: Duration = Duration::from_secs(5);

/// Extended measurement time for benchmarks with high variance (10 seconds).
/// Use for operations that show inconsistent timing.
pub const EXTENDED_MEASUREMENT_TIME: Duration = Duration::from_secs(10);

/// Warm-up time before measurements begin (3 seconds).
/// Allows CPU caches and branch predictors to stabilize.
pub const WARMUP_TIME: Duration = Duration::from_secs(3);

/// Short warm-up for quick benchmarks (1 second).
pub const QUICK_WARMUP_TIME: Duration = Duration::from_secs(1);

/// Default sample size for benchmarks.
/// Higher values provide better statistical confidence but take longer.
pub const DEFAULT_SAMPLE_SIZE: usize = 100;

/// Reduced sample size for long-running benchmarks.
pub const REDUCED_SAMPLE_SIZE: usize = 50;

/// Minimum sample size for very long-running benchmarks (1M+ elements).
pub const MINIMUM_SAMPLE_SIZE: usize = 20;

// ============================================================================
// Benchmark Group Names
// ============================================================================

/// Group name for E01 baseline indicator benchmarks.
pub const GROUP_E01_BASELINE: &str = "e01_baseline";

/// Group name for E02 RunningStat fusion benchmarks.
pub const GROUP_E02_RUNNING_STAT: &str = "e02_running_stat";

/// Group name for E03 EMA fusion benchmarks.
pub const GROUP_E03_EMA_FUSION: &str = "e03_ema_fusion";

/// Group name for E04 rolling extrema benchmarks.
pub const GROUP_E04_ROLLING_EXTREMA: &str = "e04_rolling_extrema";

/// Group name for E05 plan overhead benchmarks.
pub const GROUP_E05_PLAN_OVERHEAD: &str = "e05_plan_overhead";

/// Group name for E06 memory writes benchmarks.
pub const GROUP_E06_MEMORY_WRITES: &str = "e06_memory_writes";

/// Group name for E07 end-to-end benchmarks.
pub const GROUP_E07_END_TO_END: &str = "e07_end_to_end";

// ============================================================================
// Common Indicator Periods
// ============================================================================

/// Standard SMA/EMA periods used across benchmarks.
pub const STANDARD_PERIODS: [usize; 4] = [10, 20, 50, 200];

/// MACD fast period (standard: 12).
pub const MACD_FAST_PERIOD: usize = 12;

/// MACD slow period (standard: 26).
pub const MACD_SLOW_PERIOD: usize = 26;

/// MACD signal period (standard: 9).
pub const MACD_SIGNAL_PERIOD: usize = 9;

/// RSI standard period.
pub const RSI_PERIOD: usize = 14;

/// ATR standard period.
pub const ATR_PERIOD: usize = 14;

/// Bollinger Bands standard period.
pub const BOLLINGER_PERIOD: usize = 20;

/// Bollinger Bands standard deviation multiplier.
pub const BOLLINGER_STDDEV: f64 = 2.0;

/// Stochastic %K period.
pub const STOCHASTIC_K_PERIOD: usize = 14;

/// Stochastic %D period (smoothing of %K).
pub const STOCHASTIC_D_PERIOD: usize = 3;

// ============================================================================
// Seed Values for Reproducibility
// ============================================================================

/// Default seed for reproducible benchmark data generation.
/// Using a constant seed ensures consistent results across runs.
pub const DEFAULT_SEED: u64 = 42;

/// Alternative seeds for testing reproducibility.
pub const ALT_SEEDS: [u64; 3] = [12345, 98765, 31415];

// ============================================================================
// Helper Functions
// ============================================================================

/// Returns an appropriate sample size based on data size.
///
/// Larger datasets take longer per iteration, so we reduce sample count
/// to keep total benchmark time reasonable.
#[inline]
pub const fn sample_size_for_data_size(data_size: usize) -> usize {
    match data_size {
        0..=10_000 => DEFAULT_SAMPLE_SIZE,
        10_001..=100_000 => REDUCED_SAMPLE_SIZE,
        _ => MINIMUM_SAMPLE_SIZE,
    }
}

/// Returns an appropriate measurement time based on data size.
///
/// Larger datasets may need more time to achieve stable measurements.
#[inline]
pub const fn measurement_time_for_data_size(data_size: usize) -> Duration {
    match data_size {
        0..=10_000 => DEFAULT_MEASUREMENT_TIME,
        _ => EXTENDED_MEASUREMENT_TIME,
    }
}

/// Formats a data size for display in benchmark IDs.
///
/// # Examples
///
/// - 1000 -> "1K"
/// - 10000 -> "10K"
/// - 100000 -> "100K"
/// - 1000000 -> "1M"
pub fn format_size(size: usize) -> String {
    match size {
        s if s >= 1_000_000 => format!("{}M", s / 1_000_000),
        s if s >= 1_000 => format!("{}K", s / 1_000),
        s => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_sizes_are_ordered() {
        assert!(SIZE_1K < SIZE_10K);
        assert!(SIZE_10K < SIZE_100K);
        assert!(SIZE_100K < SIZE_1M);
    }

    #[test]
    fn test_data_sizes_array() {
        assert_eq!(DATA_SIZES.len(), 4);
        assert_eq!(DATA_SIZES[0], SIZE_1K);
        assert_eq!(DATA_SIZES[3], SIZE_1M);
    }

    #[test]
    fn test_quick_data_sizes_excludes_1m() {
        assert_eq!(QUICK_DATA_SIZES.len(), 3);
        assert!(!QUICK_DATA_SIZES.contains(&SIZE_1M));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1_000), "1K");
        assert_eq!(format_size(10_000), "10K");
        assert_eq!(format_size(100_000), "100K");
        assert_eq!(format_size(1_000_000), "1M");
        assert_eq!(format_size(500), "500");
    }

    #[test]
    fn test_sample_size_for_data_size() {
        assert_eq!(sample_size_for_data_size(SIZE_1K), DEFAULT_SAMPLE_SIZE);
        assert_eq!(sample_size_for_data_size(SIZE_10K), DEFAULT_SAMPLE_SIZE);
        assert_eq!(sample_size_for_data_size(SIZE_100K), REDUCED_SAMPLE_SIZE);
        assert_eq!(sample_size_for_data_size(SIZE_1M), MINIMUM_SAMPLE_SIZE);
    }

    #[test]
    fn test_measurement_time_for_data_size() {
        assert_eq!(
            measurement_time_for_data_size(SIZE_1K),
            DEFAULT_MEASUREMENT_TIME
        );
        assert_eq!(
            measurement_time_for_data_size(SIZE_10K),
            DEFAULT_MEASUREMENT_TIME
        );
        assert_eq!(
            measurement_time_for_data_size(SIZE_100K),
            EXTENDED_MEASUREMENT_TIME
        );
        assert_eq!(
            measurement_time_for_data_size(SIZE_1M),
            EXTENDED_MEASUREMENT_TIME
        );
    }

    #[test]
    fn test_standard_periods() {
        // Verify common trading periods are included
        assert!(STANDARD_PERIODS.contains(&10));
        assert!(STANDARD_PERIODS.contains(&20));
        assert!(STANDARD_PERIODS.contains(&50));
        assert!(STANDARD_PERIODS.contains(&200));
    }

    #[test]
    fn test_macd_periods() {
        // Standard MACD uses 12, 26, 9
        assert_eq!(MACD_FAST_PERIOD, 12);
        assert_eq!(MACD_SLOW_PERIOD, 26);
        assert_eq!(MACD_SIGNAL_PERIOD, 9);
        assert!(MACD_FAST_PERIOD < MACD_SLOW_PERIOD);
    }
}
