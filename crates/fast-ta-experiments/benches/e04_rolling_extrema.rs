//! E04: Rolling Extrema Benchmarks
//!
//! This experiment compares deque-based O(n) rolling extrema vs naive O(n×k) scan.
//!
//! # Hypothesis
//!
//! Computing rolling max/min using a monotonic deque algorithm should be significantly
//! faster than the naive scan approach because:
//!
//! 1. **Amortized O(1) per element**: Each element is pushed and popped at most once
//! 2. **O(k) vs O(n×k) comparison**: Naive scan compares k elements per output
//! 3. **Better cache behavior**: Deque is small (O(k)) and stays in cache
//!
//! # Methodology
//!
//! We benchmark several approaches:
//!
//! 1. **Deque-based (Optimized)**: Using `rolling_max()` and `rolling_min()` with
//!    monotonic deque implementation
//!    - Time complexity: O(n) amortized
//!    - Space complexity: O(n) for output + O(k) for deque
//!
//! 2. **Naive Scan**: Using `rolling_max_naive()` and `rolling_min_naive()` which
//!    scan the entire window for each output element
//!    - Time complexity: O(n × k) where k is the period
//!    - Space complexity: O(n) for output
//!
//! 3. **Fused Extrema**: Using `rolling_extrema()` which computes both max and min
//!    in a single pass vs calling rolling_max and rolling_min separately
//!
//! # Success Criteria
//!
//! - **GO**: Deque-based approach achieves ≥5× speedup (especially at larger periods)
//! - **NO-GO**: Speedup is <2× or deque is slower
//! - **INVESTIGATE**: Speedup is 2-5×
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e04_rolling_extrema/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E04_rolling_extrema/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E04_ROLLING_EXTREMA, STOCHASTIC_K_PERIOD,
};

// Core library imports
use fast_ta_core::kernels::rolling_extrema::{
    rolling_max, rolling_max_into, rolling_max_naive,
    rolling_min, rolling_min_into, rolling_min_naive,
    rolling_extrema, rolling_extrema_into, RollingExtremaOutput,
};

// Data generators
use fast_ta_experiments::data::generate_random_walk;

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Standard period for rolling extrema benchmarks (matches Stochastic %K period)
const ROLLING_PERIOD: usize = STOCHASTIC_K_PERIOD; // 14

/// Periods to test for scaling analysis (small to large)
const TEST_PERIODS: [usize; 5] = [5, 14, 50, 100, 200];

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
// Rolling Max: Deque vs Naive
// ============================================================================

/// Benchmark deque-based rolling max (optimized O(n) algorithm).
fn bench_rolling_max_deque(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/rolling_max_deque", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark naive O(n×k) rolling max scan.
fn bench_rolling_max_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/rolling_max_naive", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max_naive(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Rolling Min: Deque vs Naive
// ============================================================================

/// Benchmark deque-based rolling min (optimized O(n) algorithm).
fn bench_rolling_min_deque(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/rolling_min_deque", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_min(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark naive O(n×k) rolling min scan.
fn bench_rolling_min_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/rolling_min_naive", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_min_naive(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Fused Extrema: Both Max and Min Together
// ============================================================================

/// Benchmark fused rolling extrema (both max and min in single pass).
fn bench_fused_extrema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/fused_extrema", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_extrema(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark separate rolling max + rolling min calls (for comparison with fused).
fn bench_separate_extrema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/separate_extrema", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_14", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let max_result = black_box(rolling_max(black_box(&data.prices), ROLLING_PERIOD).unwrap());
                    let min_result = black_box(rolling_min(black_box(&data.prices), ROLLING_PERIOD).unwrap());
                    black_box((max_result, min_result))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Period Scaling Analysis
// ============================================================================

/// Benchmark to analyze how speedup scales with period size.
/// The naive approach should slow down linearly with period, while deque stays constant.
fn bench_period_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/period_scaling", GROUP_E04_ROLLING_EXTREMA));

    let size = 100_000; // Fixed size for period scaling analysis
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &period in &TEST_PERIODS {
        // Deque approach
        group.bench_with_input(
            BenchmarkId::new("deque", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max(black_box(&data.prices), period).unwrap()))
            },
        );

        // Naive approach
        group.bench_with_input(
            BenchmarkId::new("naive", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max_naive(black_box(&data.prices), period).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Data Size Scaling Analysis
// ============================================================================

/// Benchmark to verify O(n) vs O(n×k) scaling with data size.
fn bench_data_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/data_size_scaling", GROUP_E04_ROLLING_EXTREMA));

    // Test with a larger period to amplify the difference
    let period = 50;

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Deque approach
        group.bench_with_input(
            BenchmarkId::new("deque_period_50", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max(black_box(&data.prices), period).unwrap()))
            },
        );

        // Naive approach
        group.bench_with_input(
            BenchmarkId::new("naive_period_50", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max_naive(black_box(&data.prices), period).unwrap()))
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
    let mut group = c.benchmark_group(format!("{}/throughput", GROUP_E04_ROLLING_EXTREMA));

    let throughput_sizes = [10_000, 100_000, 1_000_000];

    for &size in &throughput_sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("deque", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("naive", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max_naive(black_box(&data.prices), ROLLING_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Pre-allocated Buffer Comparison
// ============================================================================

/// Compare deque vs naive when using pre-allocated buffers (zero-alloc scenario).
fn bench_preallocated(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/preallocated", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Deque with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("rolling_max_into", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    black_box(
                        rolling_max_into(black_box(&data.prices), ROLLING_PERIOD, &mut output)
                            .unwrap(),
                    )
                })
            },
        );

        // Deque rolling min with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("rolling_min_into", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    black_box(
                        rolling_min_into(black_box(&data.prices), ROLLING_PERIOD, &mut output)
                            .unwrap(),
                    )
                })
            },
        );

        // Fused extrema with pre-allocated buffer
        group.bench_with_input(
            BenchmarkId::new("rolling_extrema_into", format_size(size)),
            &data,
            |b, data| {
                let mut output = RollingExtremaOutput {
                    max: vec![0.0_f64; size],
                    min: vec![0.0_f64; size],
                };
                b.iter(|| {
                    black_box(
                        rolling_extrema_into(black_box(&data.prices), ROLLING_PERIOD, &mut output)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Large Period Extreme Case
// ============================================================================

/// Benchmark with very large period to maximize the difference between O(n) and O(n×k).
fn bench_large_period(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/large_period", GROUP_E04_ROLLING_EXTREMA));

    let size = 100_000;
    let large_periods = [100, 200, 500, 1000];
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &period in &large_periods {
        // Deque approach
        group.bench_with_input(
            BenchmarkId::new("deque", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max(black_box(&data.prices), period).unwrap()))
            },
        );

        // Naive approach
        group.bench_with_input(
            BenchmarkId::new("naive", format!("period_{}", period)),
            &data,
            |b, data| {
                b.iter(|| black_box(rolling_max_naive(black_box(&data.prices), period).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Stochastic Oscillator Use Case
// ============================================================================

/// Benchmark the typical Stochastic Oscillator use case:
/// Computing highest high and lowest low over a lookback period.
fn bench_stochastic_use_case(c: &mut Criterion) {
    use fast_ta_experiments::data::generate_ohlcv;

    let mut group = c.benchmark_group(format!("{}/stochastic_use_case", GROUP_E04_ROLLING_EXTREMA));

    for &size in &QUICK_DATA_SIZES {
        // generate_ohlcv returns an Ohlcv struct with columnar data (Vec<f64> fields)
        let ohlcv = generate_ohlcv(size, DEFAULT_SEED);
        // Use the columnar high/low vectors directly
        let highs = ohlcv.high.clone();
        let lows = ohlcv.low.clone();

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Deque approach for Stochastic
        group.bench_with_input(
            BenchmarkId::new("deque_highest_lowest", format_size(size)),
            &(highs.clone(), lows.clone()),
            |b, (highs, lows)| {
                b.iter(|| {
                    let highest_high = black_box(rolling_max(black_box(highs), STOCHASTIC_K_PERIOD).unwrap());
                    let lowest_low = black_box(rolling_min(black_box(lows), STOCHASTIC_K_PERIOD).unwrap());
                    black_box((highest_high, lowest_low))
                })
            },
        );

        // Naive approach for Stochastic
        group.bench_with_input(
            BenchmarkId::new("naive_highest_lowest", format_size(size)),
            &(highs, lows),
            |b, (highs, lows)| {
                b.iter(|| {
                    let highest_high = black_box(rolling_max_naive(black_box(highs), STOCHASTIC_K_PERIOD).unwrap());
                    let lowest_low = black_box(rolling_min_naive(black_box(lows), STOCHASTIC_K_PERIOD).unwrap());
                    black_box((highest_high, lowest_low))
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
    name = rolling_extrema_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_rolling_max_deque,
        bench_rolling_max_naive,
        bench_rolling_min_deque,
        bench_rolling_min_naive,
        bench_fused_extrema,
        bench_separate_extrema,
        bench_period_scaling,
        bench_data_size_scaling,
        bench_throughput,
        bench_preallocated,
        bench_large_period,
        bench_stochastic_use_case
);

criterion_main!(rolling_extrema_benches);
