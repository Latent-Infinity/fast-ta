//! E02: RunningStat Fusion Benchmarks
//!
//! This experiment compares fused vs separate pass approaches for computing
//! rolling statistics (mean, variance, standard deviation).
//!
//! # Hypothesis
//!
//! Computing mean, variance, and stddev in a single pass using Welford's algorithm
//! (fused approach) should be faster than computing them separately because:
//!
//! 1. **Reduced memory bandwidth**: Single pass reads input data once vs twice
//! 2. **Better cache locality**: Data stays hot in cache during fused computation
//! 3. **Amortized loop overhead**: One loop vs multiple loops
//!
//! # Methodology
//!
//! We benchmark three approaches:
//!
//! 1. **Fused (Welford)**: Using `rolling_stats()` from the running_stat kernel
//!    - Computes mean, variance, stddev in a single pass
//!    - Uses numerically stable Welford's algorithm
//!
//! 2. **Separate Passes (Naive)**: Computing each statistic independently
//!    - SMA for mean (rolling sum approach)
//!    - rolling_stddev for standard deviation
//!    - Variance computed as stddev²
//!
//! 3. **Bollinger (Reference)**: Using Bollinger Bands computation
//!    - Uses rolling sum + sum-of-squares approach
//!    - Represents the "traditional" approach for mean + stddev
//!
//! # Success Criteria
//!
//! - **GO**: Fused approach achieves ≥20% speedup over separate passes
//! - **NO-GO**: Speedup is <10% or fused is slower
//! - **INVESTIGATE**: Speedup is 10-20%
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e02_running_stat/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E02_running_stat/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E02_RUNNING_STAT, BOLLINGER_PERIOD,
};

// Core library imports
use fast_ta_core::indicators::{bollinger, sma};
use fast_ta_core::indicators::bollinger::rolling_stddev;
use fast_ta_core::kernels::running_stat::rolling_stats;

// Data generators
use fast_ta_experiments::data::generate_random_walk;

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Standard period for rolling statistics benchmarks (matches Bollinger default)
const ROLLING_PERIOD: usize = BOLLINGER_PERIOD; // 20

/// Standard deviation multiplier for Bollinger comparison
const NUM_STD_DEV: f64 = 2.0;

// ============================================================================
// Data Preparation
// ============================================================================

/// Pre-generated price data to avoid including data generation in benchmark timing.
struct BenchmarkData {
    prices: Vec<f64>,
}

impl BenchmarkData {
    fn new(size: usize, seed: u64) -> Self {
        Self {
            prices: generate_random_walk(size, seed),
        }
    }
}

// ============================================================================
// Fused Approach Benchmarks (Welford's Algorithm)
// ============================================================================

/// Benchmark the fused rolling statistics using Welford's algorithm.
/// This computes mean, variance, and stddev in a single pass.
fn bench_fused_welford(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/fused_welford", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_20", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_stats(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Separate Passes Benchmarks
// ============================================================================

/// Benchmark computing mean and stddev as separate passes.
/// This simulates what a user would do without fusion.
fn bench_separate_passes(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/separate_passes", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("sma_plus_stddev", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    // First pass: compute SMA (mean)
                    let mean = black_box(sma(black_box(&data.prices), ROLLING_PERIOD).unwrap());

                    // Second pass: compute rolling stddev
                    let stddev = black_box(rolling_stddev(black_box(&data.prices), ROLLING_PERIOD).unwrap());

                    // Compute variance from stddev (element-wise square)
                    let variance: Vec<f64> = stddev.iter().map(|&s| s * s).collect();

                    black_box((mean, stddev, variance))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Bollinger Bands Reference Benchmark
// ============================================================================

/// Benchmark using Bollinger Bands computation (sum + sum-of-squares approach).
/// This provides a reference for the "traditional" implementation.
fn bench_bollinger_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/bollinger_reference", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("bollinger_bands", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(bollinger(black_box(&data.prices), ROLLING_PERIOD, NUM_STD_DEV).unwrap())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Component Benchmarks (for detailed analysis)
// ============================================================================

/// Benchmark just the SMA computation.
fn bench_sma_only(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/components/sma", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_20", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(sma(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark just the rolling stddev computation.
fn bench_stddev_only(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/components/stddev", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_20", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_stddev(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Different Period Comparisons
// ============================================================================

/// Benchmark fused approach with different periods to see if fusion benefit scales.
fn bench_fused_different_periods(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/period_comparison", GROUP_E02_RUNNING_STAT));

    let periods = [5, 10, 20, 50, 100];
    let size = 100_000; // Use 100K for period comparison
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &period in &periods {
        group.bench_with_input(
            BenchmarkId::new("fused_welford", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_stats(black_box(&data.prices), period).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("separate_passes", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| {
                    let mean = black_box(sma(black_box(&data.prices), period).unwrap());
                    let stddev = black_box(rolling_stddev(black_box(&data.prices), period).unwrap());
                    let variance: Vec<f64> = stddev.iter().map(|&s| s * s).collect();
                    black_box((mean, stddev, variance))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

/// Benchmark to measure elements per second throughput for both approaches.
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/throughput", GROUP_E02_RUNNING_STAT));

    // Use larger sizes for throughput measurement
    let throughput_sizes = [10_000, 100_000, 1_000_000];

    for &size in &throughput_sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("fused", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_stats(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("separate", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let mean = black_box(sma(black_box(&data.prices), ROLLING_PERIOD).unwrap());
                    let stddev = black_box(rolling_stddev(black_box(&data.prices), ROLLING_PERIOD).unwrap());
                    let variance: Vec<f64> = stddev.iter().map(|&s| s * s).collect();
                    black_box((mean, stddev, variance))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Allocation Comparison
// ============================================================================

/// Compare approaches when using pre-allocated buffers (zero-alloc scenario).
fn bench_preallocated(c: &mut Criterion) {
    use fast_ta_core::kernels::running_stat::{rolling_stats_into, RollingStatOutput};

    let mut group = c.benchmark_group(format!("{}/preallocated", GROUP_E02_RUNNING_STAT));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("fused_into", format_size(size)),
            &data,
            |b, data| {
                // Pre-allocate output buffer
                let mut output = RollingStatOutput {
                    mean: vec![0.0_f64; size],
                    variance: vec![0.0_f64; size],
                    stddev: vec![0.0_f64; size],
                };
                b.iter(|| {
                    black_box(
                        rolling_stats_into(black_box(&data.prices), ROLLING_PERIOD, &mut output)
                            .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("separate_into", format_size(size)),
            &data,
            |b, data| {
                // Pre-allocate output buffers
                let mut mean_output = vec![0.0_f64; size];
                let mut stddev_output = vec![0.0_f64; size];
                let mut variance_output = vec![0.0_f64; size];

                b.iter(|| {
                    use fast_ta_core::indicators::sma::sma_into;
                    use fast_ta_core::indicators::bollinger::rolling_stddev_into;

                    black_box(sma_into(black_box(&data.prices), ROLLING_PERIOD, &mut mean_output).unwrap());
                    black_box(rolling_stddev_into(black_box(&data.prices), ROLLING_PERIOD, &mut stddev_output).unwrap());

                    // Compute variance from stddev
                    for i in 0..size {
                        variance_output[i] = stddev_output[i] * stddev_output[i];
                    }

                    // Prevent the compiler from optimizing away variance computation
                    black_box(variance_output[size - 1])
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = running_stat_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_fused_welford,
        bench_separate_passes,
        bench_bollinger_reference,
        bench_sma_only,
        bench_stddev_only,
        bench_fused_different_periods,
        bench_throughput,
        bench_preallocated
);

criterion_main!(running_stat_benches);
