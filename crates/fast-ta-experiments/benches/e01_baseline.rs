//! E01: Baseline Cost Benchmarks for All 7 Indicators
//!
//! This experiment establishes performance baselines for all 7 technical indicators:
//! - SMA (Simple Moving Average)
//! - EMA (Exponential Moving Average)
//! - RSI (Relative Strength Index)
//! - MACD (Moving Average Convergence Divergence)
//! - ATR (Average True Range)
//! - Bollinger Bands
//! - Stochastic Oscillator
//!
//! # Purpose
//!
//! The baseline benchmarks serve as reference points for:
//! 1. Measuring the cost of individual indicator computations
//! 2. Identifying which indicators are most expensive
//! 3. Providing comparison data for fusion kernel experiments (E02-E04)
//! 4. Establishing performance profiles across different data sizes
//!
//! # Methodology
//!
//! Each indicator is benchmarked with:
//! - Standard data sizes: 1K, 10K, 100K, 1M points
//! - Default/standard periods for each indicator
//! - Reproducible synthetic data (seeded RNG)
//! - black_box() to prevent dead code elimination
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e01_baseline/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E01_baseline/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E01_BASELINE,
};

// Core library imports
use fast_ta_core::indicators::{atr, bollinger, ema, macd, rsi, sma, stochastic_fast};

// Data generators
use fast_ta_experiments::data::{generate_ohlcv, generate_random_walk, Ohlcv};

// Standard indicator periods (from common module)
use common::{
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STDDEV, MACD_FAST_PERIOD, MACD_SIGNAL_PERIOD,
    MACD_SLOW_PERIOD, RSI_PERIOD, STOCHASTIC_D_PERIOD, STOCHASTIC_K_PERIOD,
};

/// Standard SMA period for baseline benchmarks
const SMA_PERIOD: usize = 20;

/// Standard EMA period for baseline benchmarks
const EMA_PERIOD: usize = 20;

// ============================================================================
// Data Preparation
// ============================================================================

/// Pre-generated price data for different sizes to avoid including
/// data generation in benchmark timing.
struct BenchmarkData {
    prices: Vec<f64>,
    ohlcv: Ohlcv,
}

impl BenchmarkData {
    fn new(size: usize, seed: u64) -> Self {
        Self {
            prices: generate_random_walk(size, seed),
            ohlcv: generate_ohlcv(size, seed),
        }
    }
}

// ============================================================================
// SMA Benchmark
// ============================================================================

fn bench_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/sma", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(BenchmarkId::new("period_20", format_size(size)), &data, |b, data| {
            b.iter(|| black_box(sma(black_box(&data.prices), SMA_PERIOD).unwrap()))
        });
    }

    group.finish();
}

// ============================================================================
// EMA Benchmark
// ============================================================================

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/ema", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(BenchmarkId::new("period_20", format_size(size)), &data, |b, data| {
            b.iter(|| black_box(ema(black_box(&data.prices), EMA_PERIOD).unwrap()))
        });
    }

    group.finish();
}

// ============================================================================
// RSI Benchmark
// ============================================================================

fn bench_rsi(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/rsi", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(BenchmarkId::new("period_14", format_size(size)), &data, |b, data| {
            b.iter(|| black_box(rsi(black_box(&data.prices), RSI_PERIOD).unwrap()))
        });
    }

    group.finish();
}

// ============================================================================
// MACD Benchmark
// ============================================================================

fn bench_macd(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/macd", GROUP_E01_BASELINE));

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
// ATR Benchmark
// ============================================================================

fn bench_atr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/atr", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(BenchmarkId::new("period_14", format_size(size)), &data, |b, data| {
            b.iter(|| {
                black_box(
                    atr(
                        black_box(&data.ohlcv.high),
                        black_box(&data.ohlcv.low),
                        black_box(&data.ohlcv.close),
                        ATR_PERIOD,
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

// ============================================================================
// Bollinger Bands Benchmark
// ============================================================================

fn bench_bollinger(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/bollinger", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("period_20_stddev_2", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(
                        bollinger(black_box(&data.prices), BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Stochastic Oscillator Benchmark
// ============================================================================

fn bench_stochastic(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/stochastic", GROUP_E01_BASELINE));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("fast_14_3", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(
                        stochastic_fast(
                            black_box(&data.ohlcv.high),
                            black_box(&data.ohlcv.low),
                            black_box(&data.ohlcv.close),
                            STOCHASTIC_K_PERIOD,
                            STOCHASTIC_D_PERIOD,
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
// Combined All-Indicators Benchmark
// ============================================================================

/// Benchmark running all 7 indicators on the same data
/// This provides a reference for what a typical trading system might compute.
fn bench_all_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/all_indicators", GROUP_E01_BASELINE));

    // Use only 10K and 100K for the combined benchmark to keep runtime reasonable
    for &size in &[10_000_usize, 100_000_usize] {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("sequential", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    // SMA
                    let _sma = black_box(sma(black_box(&data.prices), SMA_PERIOD).unwrap());
                    // EMA
                    let _ema = black_box(ema(black_box(&data.prices), EMA_PERIOD).unwrap());
                    // RSI
                    let _rsi = black_box(rsi(black_box(&data.prices), RSI_PERIOD).unwrap());
                    // MACD
                    let _macd = black_box(
                        macd(
                            black_box(&data.prices),
                            MACD_FAST_PERIOD,
                            MACD_SLOW_PERIOD,
                            MACD_SIGNAL_PERIOD,
                        )
                        .unwrap(),
                    );
                    // ATR
                    let _atr = black_box(
                        atr(
                            black_box(&data.ohlcv.high),
                            black_box(&data.ohlcv.low),
                            black_box(&data.ohlcv.close),
                            ATR_PERIOD,
                        )
                        .unwrap(),
                    );
                    // Bollinger Bands
                    let _bollinger = black_box(
                        bollinger(black_box(&data.prices), BOLLINGER_PERIOD, BOLLINGER_STDDEV)
                            .unwrap(),
                    );
                    // Stochastic
                    let _stochastic = black_box(
                        stochastic_fast(
                            black_box(&data.ohlcv.high),
                            black_box(&data.ohlcv.low),
                            black_box(&data.ohlcv.close),
                            STOCHASTIC_K_PERIOD,
                            STOCHASTIC_D_PERIOD,
                        )
                        .unwrap(),
                    );
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
    name = baseline_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_sma,
        bench_ema,
        bench_rsi,
        bench_macd,
        bench_atr,
        bench_bollinger,
        bench_stochastic,
        bench_all_indicators
);

criterion_main!(baseline_benches);
