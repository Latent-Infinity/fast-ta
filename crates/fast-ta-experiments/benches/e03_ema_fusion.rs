//! E03: EMA Fusion Benchmarks
//!
//! This experiment compares fused vs independent EMA-family computations.
//!
//! # Hypothesis
//!
//! Computing multiple EMA-family indicators in a single pass (fused approach)
//! should be faster than computing them independently because:
//!
//! 1. **Reduced memory bandwidth**: Single pass reads input data once vs multiple times
//! 2. **Better cache locality**: Data stays hot in cache during fused computation
//! 3. **Amortized loop overhead**: One loop vs multiple loops
//! 4. **Reused intermediate results**: EMA values reused for DEMA, TEMA, MACD
//!
//! # Methodology
//!
//! We benchmark several approaches:
//!
//! 1. **Fused Multi-EMA**: Using `ema_multi()` to compute multiple EMAs together
//!    - Computes all EMAs with different periods in a single pass
//!    - Time complexity: O(n × k) with single data read
//!
//! 2. **Separate EMAs**: Computing each EMA independently
//!    - Multiple calls to `ema()` with different periods
//!    - Time complexity: O(n × k) with k data reads
//!
//! 3. **Fused EMA/DEMA/TEMA**: Using `ema_fusion()` to compute all together
//!    - Computes EMA, DEMA, TEMA in a single fused pass
//!    - Reuses EMA(EMA) and EMA(EMA(EMA)) computations
//!
//! 4. **Fused MACD**: Using `macd_fusion()` for MACD computation
//!    - Computes fast EMA, slow EMA, signal together
//!    - Compares to standard MACD implementation
//!
//! # Success Criteria
//!
//! - **GO**: Fused approach achieves ≥15% speedup for ≥10 EMAs
//! - **NO-GO**: Speedup is <10% or fused is slower
//! - **INVESTIGATE**: Speedup is 10-15%
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e03_ema_fusion/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E03_ema_fusion/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E03_EMA_FUSION, MACD_FAST_PERIOD, MACD_SIGNAL_PERIOD, MACD_SLOW_PERIOD,
    STANDARD_PERIODS,
};

// Core library imports
use fast_ta_core::indicators::ema::ema;
use fast_ta_core::indicators::macd::macd;
use fast_ta_core::kernels::ema_fusion::{ema_fusion, ema_multi, macd_fusion};

// Data generators
use fast_ta_experiments::data::generate_random_walk;

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Number of EMAs to compute for scalability tests (3, 5, 10, 20)
const EMA_COUNTS: [usize; 4] = [3, 5, 10, 20];

/// Standard period for EMA/DEMA/TEMA tests
const FUSION_PERIOD: usize = 12;

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

/// Generates N periods centered around typical EMA periods.
fn generate_periods(count: usize) -> Vec<usize> {
    // Generate periods: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, ...
    (0..count).map(|i| 5 + i * 5).collect()
}

// ============================================================================
// Multi-EMA Benchmarks: Fused vs Separate
// ============================================================================

/// Benchmark fused multi-EMA computation using ema_multi().
fn bench_fused_multi_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/fused_multi_ema", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        for &count in &EMA_COUNTS {
            let periods = generate_periods(count);

            group.bench_with_input(
                BenchmarkId::new(format!("{}_emas", count), format_size(size)),
                &data,
                |b, data| {
                    b.iter(|| black_box(ema_multi(black_box(&data.prices), black_box(&periods)).unwrap()))
                },
            );
        }
    }

    group.finish();
}

/// Benchmark separate EMA computation using multiple ema() calls.
fn bench_separate_emas(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/separate_emas", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        for &count in &EMA_COUNTS {
            let periods = generate_periods(count);

            group.bench_with_input(
                BenchmarkId::new(format!("{}_emas", count), format_size(size)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let results: Vec<Vec<f64>> = periods
                            .iter()
                            .map(|&period| black_box(ema(black_box(&data.prices), period).unwrap()))
                            .collect();
                        black_box(results)
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// EMA/DEMA/TEMA Fusion Benchmarks
// ============================================================================

/// Benchmark fused EMA/DEMA/TEMA computation.
fn bench_fused_ema_dema_tema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/fused_ema_dema_tema", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_12", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema_fusion(black_box(&data.prices), FUSION_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark separate EMA/DEMA/TEMA computation.
/// DEMA = 2 × EMA - EMA(EMA)
/// TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
fn bench_separate_ema_dema_tema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/separate_ema_dema_tema", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_12", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    // Compute EMA
                    let ema1 = black_box(ema(black_box(&data.prices), FUSION_PERIOD).unwrap());

                    // Compute EMA(EMA) for DEMA and TEMA
                    let ema2 = black_box(ema(black_box(&ema1), FUSION_PERIOD).unwrap());

                    // Compute EMA(EMA(EMA)) for TEMA
                    let ema3 = black_box(ema(black_box(&ema2), FUSION_PERIOD).unwrap());

                    // Compute DEMA = 2 × EMA - EMA(EMA)
                    let dema: Vec<f64> = ema1
                        .iter()
                        .zip(&ema2)
                        .map(|(&e1, &e2)| 2.0 * e1 - e2)
                        .collect();

                    // Compute TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
                    let tema: Vec<f64> = ema1
                        .iter()
                        .zip(&ema2)
                        .zip(&ema3)
                        .map(|((&e1, &e2), &e3)| 3.0 * e1 - 3.0 * e2 + e3)
                        .collect();

                    black_box((ema1, dema, tema))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// MACD Fusion Benchmarks
// ============================================================================

/// Benchmark fused MACD computation using macd_fusion().
fn bench_fused_macd(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/fused_macd", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("12_26_9", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(
                        macd_fusion(
                            black_box(&data.prices),
                            MACD_FAST_PERIOD,
                            MACD_SLOW_PERIOD,
                            MACD_SIGNAL_PERIOD,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark standard MACD indicator (baseline).
fn bench_standard_macd(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/standard_macd", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("12_26_9", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(
                        macd(
                            black_box(&data.prices),
                            MACD_FAST_PERIOD,
                            MACD_SLOW_PERIOD,
                            MACD_SIGNAL_PERIOD,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scaling Analysis (Number of EMAs)
// ============================================================================

/// Benchmark to measure how speedup scales with the number of EMAs.
fn bench_ema_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/ema_count_scaling", GROUP_E03_EMA_FUSION));

    let size = 100_000; // Fixed size for scaling analysis
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &count in &EMA_COUNTS {
        let periods = generate_periods(count);

        // Fused approach
        group.bench_with_input(
            BenchmarkId::new("fused", format!("{}_emas", count)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema_multi(black_box(&data.prices), black_box(&periods)).unwrap()))
            },
        );

        // Separate approach
        group.bench_with_input(
            BenchmarkId::new("separate", format!("{}_emas", count)),
            &data,
            |b, data| {
                b.iter(|| {
                    let results: Vec<Vec<f64>> = periods
                        .iter()
                        .map(|&period| black_box(ema(black_box(&data.prices), period).unwrap()))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Period Sensitivity Analysis
// ============================================================================

/// Benchmark to analyze if fusion benefit changes with EMA period.
fn bench_period_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/period_sensitivity", GROUP_E03_EMA_FUSION));

    let size = 100_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);
    let periods_to_test = &STANDARD_PERIODS; // 10, 20, 50, 200

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &period in periods_to_test {
        if period <= size / 3 {
            // EMA/DEMA/TEMA fusion
            group.bench_with_input(
                BenchmarkId::new("fused_ema_dema_tema", format!("period_{}", period)),
                &data,
                |b, data| {
                    b.iter(|| black_box(ema_fusion(black_box(&data.prices), period).unwrap()))
                },
            );

            // Separate EMA/DEMA/TEMA
            group.bench_with_input(
                BenchmarkId::new("separate_ema_dema_tema", format!("period_{}", period)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let ema1 = black_box(ema(black_box(&data.prices), period).unwrap());
                        let ema2 = black_box(ema(black_box(&ema1), period).unwrap());
                        let ema3 = black_box(ema(black_box(&ema2), period).unwrap());

                        let dema: Vec<f64> = ema1
                            .iter()
                            .zip(&ema2)
                            .map(|(&e1, &e2)| 2.0 * e1 - e2)
                            .collect();

                        let tema: Vec<f64> = ema1
                            .iter()
                            .zip(&ema2)
                            .zip(&ema3)
                            .map(|((&e1, &e2), &e3)| 3.0 * e1 - 3.0 * e2 + e3)
                            .collect();

                        black_box((ema1, dema, tema))
                    })
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

/// Benchmark to measure elements per second throughput for both approaches.
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/throughput", GROUP_E03_EMA_FUSION));

    let throughput_sizes = [10_000, 100_000, 1_000_000];
    let periods = generate_periods(10); // 10 EMAs for throughput test

    for &size in &throughput_sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("fused_10_emas", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema_multi(black_box(&data.prices), black_box(&periods)).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("separate_10_emas", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let results: Vec<Vec<f64>> = periods
                        .iter()
                        .map(|&period| black_box(ema(black_box(&data.prices), period).unwrap()))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Pre-allocated Buffer Comparison
// ============================================================================

/// Compare fused vs separate when using pre-allocated buffers.
fn bench_preallocated(c: &mut Criterion) {
    use fast_ta_core::indicators::ema::ema_into;
    use fast_ta_core::kernels::ema_fusion::{ema_multi_into, EmaFusionOutput, ema_fusion_into};

    let mut group = c.benchmark_group(format!("{}/preallocated", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);
        let periods = vec![10, 20, 30]; // 3 EMAs for preallocated test

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Fused multi-EMA with pre-allocated buffers
        group.bench_with_input(
            BenchmarkId::new("fused_multi_ema_into", format_size(size)),
            &data,
            |b, data| {
                let mut output1 = vec![0.0_f64; size];
                let mut output2 = vec![0.0_f64; size];
                let mut output3 = vec![0.0_f64; size];
                let mut outputs: Vec<&mut [f64]> = vec![&mut output1, &mut output2, &mut output3];
                b.iter(|| {
                    black_box(
                        ema_multi_into(black_box(&data.prices), black_box(&periods), &mut outputs)
                            .unwrap(),
                    )
                })
            },
        );

        // Separate EMAs with pre-allocated buffers
        group.bench_with_input(
            BenchmarkId::new("separate_ema_into", format_size(size)),
            &data,
            |b, data| {
                let mut output1 = vec![0.0_f64; size];
                let mut output2 = vec![0.0_f64; size];
                let mut output3 = vec![0.0_f64; size];

                b.iter(|| {
                    black_box(ema_into(black_box(&data.prices), 10, &mut output1).unwrap());
                    black_box(ema_into(black_box(&data.prices), 20, &mut output2).unwrap());
                    black_box(ema_into(black_box(&data.prices), 30, &mut output3).unwrap());
                    black_box((&output1, &output2, &output3))
                })
            },
        );

        // Fused EMA/DEMA/TEMA with pre-allocated buffers
        group.bench_with_input(
            BenchmarkId::new("fused_ema_dema_tema_into", format_size(size)),
            &data,
            |b, data| {
                let mut output = EmaFusionOutput {
                    ema: vec![0.0_f64; size],
                    dema: vec![0.0_f64; size],
                    tema: vec![0.0_f64; size],
                };
                b.iter(|| {
                    black_box(
                        ema_fusion_into(black_box(&data.prices), FUSION_PERIOD, &mut output)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Component Benchmarks (Individual Operations)
// ============================================================================

/// Benchmark individual EMA computation to understand baseline.
fn bench_single_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/components/single_ema", GROUP_E03_EMA_FUSION));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_12", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema(black_box(&data.prices), FUSION_PERIOD).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = ema_fusion_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_fused_multi_ema,
        bench_separate_emas,
        bench_fused_ema_dema_tema,
        bench_separate_ema_dema_tema,
        bench_fused_macd,
        bench_standard_macd,
        bench_ema_count_scaling,
        bench_period_sensitivity,
        bench_throughput,
        bench_preallocated,
        bench_single_ema
);

criterion_main!(ema_fusion_benches);
