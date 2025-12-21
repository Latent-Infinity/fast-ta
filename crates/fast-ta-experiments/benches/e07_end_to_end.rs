//! E07: End-to-End Workload Benchmarks
//!
//! This experiment provides the final comparison between TA-Lib (via golden file baselines),
//! direct mode (independent indicator computation), and plan mode (fused kernel execution).
//!
//! # Hypothesis
//!
//! Plan mode with fused kernels should outperform direct mode when:
//! 1. Computing ≥6 indicators in a single workload
//! 2. Multiple EMA-based indicators share data access
//! 3. Plan compilation overhead is amortized over multiple executions
//!
//! The speedup target is ≥1.5× for ≥20 indicators compared to direct mode.
//!
//! # Methodology
//!
//! We benchmark:
//! 1. **Direct mode baseline**: All 7 indicators computed independently
//! 2. **Plan mode with fusion**: Same indicators using fused kernels
//! 3. **TA-Lib comparison**: Document performance relative to golden file baselines
//! 4. **Workload scaling**: 7, 14, 21, and 28 indicators
//! 5. **Data size scaling**: 1K, 10K, 100K, 1M data points
//!
//! # Success Criteria
//!
//! - **GO (Plan architecture)**: Plan mode achieves ≥1.5× speedup for ≥20 indicators
//! - **CONDITIONAL GO**: Plan mode achieves ≥1.2× speedup for baseline 7 indicators
//! - **NO-GO**: Plan mode shows <1.1× speedup or is slower than direct mode
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e07_end_to_end/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E07_end_to_end/REPORT.md` (analysis summary with go/no-go)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size,
    DATA_SIZES, QUICK_DATA_SIZES, DEFAULT_SEED, GROUP_E07_END_TO_END,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    RSI_PERIOD, ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STDDEV,
    STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD,
};

// Core library imports
use fast_ta_core::indicators::{
    sma, ema, rsi, macd, atr, bollinger, stochastic_fast,
};
use fast_ta_core::plan::direct_mode::{compute_all_direct, DirectExecutor, IndicatorRequest, OhlcvData};
use fast_ta_core::plan::plan_mode::{compute_all_plan, PlanExecutor};
use fast_ta_core::plan::IndicatorKind;

// Data generators
use fast_ta_experiments::data::{generate_random_walk, generate_ohlcv};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Standard periods for benchmark indicators
const SMA_PERIOD: usize = 20;
const EMA_PERIOD: usize = 20;

/// Workload sizes for scaling analysis
const WORKLOAD_INDICATOR_COUNTS: [usize; 4] = [7, 14, 21, 28];

// ============================================================================
// Data Preparation
// ============================================================================

/// Pre-generated price and OHLCV data for benchmarks.
struct BenchmarkData {
    prices: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
}

impl BenchmarkData {
    fn new(size: usize, seed: u64) -> Self {
        let ohlcv = generate_ohlcv(size, seed);
        Self {
            prices: generate_random_walk(size, seed),
            high: ohlcv.high,
            low: ohlcv.low,
            close: ohlcv.close,
        }
    }
}

// ============================================================================
// Direct Mode Implementation (Baseline)
// ============================================================================

/// Compute all 7 baseline indicators directly (no plan, no fusion).
fn compute_direct_baseline_inline(data: &BenchmarkData) {
    // SMA
    let _ = black_box(sma(&data.close, SMA_PERIOD).unwrap());
    // EMA
    let _ = black_box(ema(&data.close, EMA_PERIOD).unwrap());
    // RSI
    let _ = black_box(rsi(&data.close, RSI_PERIOD).unwrap());
    // MACD
    let _ = black_box(macd(&data.close, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD).unwrap());
    // ATR
    let _ = black_box(atr(&data.high, &data.low, &data.close, ATR_PERIOD).unwrap());
    // Bollinger Bands
    let _ = black_box(bollinger(&data.close, BOLLINGER_PERIOD, BOLLINGER_STDDEV).unwrap());
    // Stochastic
    let _ = black_box(stochastic_fast(&data.high, &data.low, &data.close, STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD).unwrap());
}

/// Compute indicators using DirectExecutor.
fn compute_direct_via_executor(data: &BenchmarkData) {
    let ohlcv = OhlcvData::new(&data.high, &data.low, &data.close);
    let _ = black_box(DirectExecutor::new().execute_all_baseline(&ohlcv).unwrap());
}

/// Compute indicators using compute_all_direct helper.
fn compute_direct_helper(data: &BenchmarkData) {
    let _ = black_box(compute_all_direct(&data.high, &data.low, &data.close).unwrap());
}

// ============================================================================
// Plan Mode Implementation (With Fusion)
// ============================================================================

/// Compute all 7 baseline indicators using PlanExecutor with fused kernels.
fn compute_plan_via_executor(data: &BenchmarkData) {
    let ohlcv = OhlcvData::new(&data.high, &data.low, &data.close);
    let _ = black_box(PlanExecutor::new().execute_all_baseline(&ohlcv).unwrap());
}

/// Compute indicators using compute_all_plan helper.
fn compute_plan_helper(data: &BenchmarkData) {
    let _ = black_box(compute_all_plan(&data.high, &data.low, &data.close).unwrap());
}

// ============================================================================
// Baseline Indicator Benchmarks (7 Indicators)
// ============================================================================

/// Benchmark direct mode: all 7 baseline indicators.
fn bench_direct_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/direct_baseline", GROUP_E07_END_TO_END));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Inline direct computation
        group.bench_with_input(
            BenchmarkId::new("inline_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_baseline_inline(black_box(data)))
            },
        );

        // DirectExecutor
        group.bench_with_input(
            BenchmarkId::new("executor_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_via_executor(black_box(data)))
            },
        );

        // Helper function
        group.bench_with_input(
            BenchmarkId::new("helper_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

/// Benchmark plan mode: all 7 baseline indicators with fusion.
fn bench_plan_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/plan_baseline", GROUP_E07_END_TO_END));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // PlanExecutor
        group.bench_with_input(
            BenchmarkId::new("executor_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_plan_via_executor(black_box(data)))
            },
        );

        // Helper function
        group.bench_with_input(
            BenchmarkId::new("helper_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_plan_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Direct vs Plan Mode Comparison
// ============================================================================

/// Head-to-head comparison of direct vs plan mode.
fn bench_direct_vs_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/direct_vs_plan", GROUP_E07_END_TO_END));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Direct mode
        group.bench_with_input(
            BenchmarkId::new("direct", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_helper(black_box(data)))
            },
        );

        // Plan mode
        group.bench_with_input(
            BenchmarkId::new("plan", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_plan_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Workload Scaling Analysis
// ============================================================================

/// Generate multiple indicator requests for scaling tests.
fn generate_scaled_requests(count: usize) -> Vec<IndicatorRequest> {
    let mut requests = Vec::with_capacity(count);

    // Base 7 indicators
    requests.push(IndicatorRequest::simple(IndicatorKind::Sma, 20));
    requests.push(IndicatorRequest::simple(IndicatorKind::Ema, 20));
    requests.push(IndicatorRequest::simple(IndicatorKind::Rsi, 14));
    requests.push(IndicatorRequest::macd(12, 26, 9));
    requests.push(IndicatorRequest::bollinger(20, 2.0));
    requests.push(IndicatorRequest::simple(IndicatorKind::Sma, 50));
    requests.push(IndicatorRequest::simple(IndicatorKind::Ema, 50));

    // Additional indicators for scaling
    let additional_periods = [10, 15, 25, 30, 40, 60, 100, 200];
    let mut period_idx = 0;

    while requests.len() < count {
        let period = additional_periods[period_idx % additional_periods.len()];
        period_idx += 1;

        // Alternate between different indicator types
        match (requests.len() - 7) % 5 {
            0 => requests.push(IndicatorRequest::simple(IndicatorKind::Sma, period)),
            1 => requests.push(IndicatorRequest::simple(IndicatorKind::Ema, period)),
            2 => requests.push(IndicatorRequest::simple(IndicatorKind::Rsi, period)),
            3 => requests.push(IndicatorRequest::bollinger(period, 2.5)),
            _ => requests.push(IndicatorRequest::macd(period, period * 2, 9)),
        }
    }

    requests
}

/// Benchmark workload scaling with direct mode.
fn bench_workload_scaling_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/scaling_direct", GROUP_E07_END_TO_END));

    let size = 10_000; // Fixed size for scaling analysis
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &count in &WORKLOAD_INDICATOR_COUNTS {
        let requests = generate_scaled_requests(count);
        let executor = DirectExecutor::new();

        group.bench_with_input(
            BenchmarkId::new("indicators", count),
            &(&data, &requests, &executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark workload scaling with plan mode.
fn bench_workload_scaling_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/scaling_plan", GROUP_E07_END_TO_END));

    let size = 10_000; // Fixed size for scaling analysis
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &count in &WORKLOAD_INDICATOR_COUNTS {
        let requests = generate_scaled_requests(count);
        let executor = PlanExecutor::new();

        group.bench_with_input(
            BenchmarkId::new("indicators", count),
            &(&data, &requests, &executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Compare direct vs plan for different workload sizes.
fn bench_workload_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/workload_comparison", GROUP_E07_END_TO_END));

    let size = 10_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &count in &WORKLOAD_INDICATOR_COUNTS {
        let requests = generate_scaled_requests(count);
        let direct_executor = DirectExecutor::new();
        let plan_executor = PlanExecutor::new();

        // Direct mode
        group.bench_with_input(
            BenchmarkId::new("direct", count),
            &(&data, &requests, &direct_executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
                })
            },
        );

        // Plan mode
        group.bench_with_input(
            BenchmarkId::new("plan", count),
            &(&data, &requests, &plan_executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Data Size Scaling
// ============================================================================

/// Benchmark how direct mode scales with data size.
fn bench_data_size_scaling_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/data_scaling_direct", GROUP_E07_END_TO_END));

    for &size in &DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

/// Benchmark how plan mode scales with data size.
fn bench_data_size_scaling_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/data_scaling_plan", GROUP_E07_END_TO_END));

    for &size in &DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_plan_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Individual Indicator Comparison
// ============================================================================

/// Compare individual indicator performance between direct and plan implementations.
fn bench_individual_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/individual_indicators", GROUP_E07_END_TO_END));

    let size = 10_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);
    let ohlcv = OhlcvData::new(&data.high, &data.low, &data.close);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // SMA
    group.bench_function("sma_direct", |b| {
        b.iter(|| black_box(sma(black_box(&data.close), SMA_PERIOD).unwrap()))
    });

    // EMA
    group.bench_function("ema_direct", |b| {
        b.iter(|| black_box(ema(black_box(&data.close), EMA_PERIOD).unwrap()))
    });

    // RSI
    group.bench_function("rsi_direct", |b| {
        b.iter(|| black_box(rsi(black_box(&data.close), RSI_PERIOD).unwrap()))
    });

    // MACD
    group.bench_function("macd_direct", |b| {
        b.iter(|| black_box(macd(black_box(&data.close), MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD).unwrap()))
    });

    // ATR
    group.bench_function("atr_direct", |b| {
        b.iter(|| black_box(atr(black_box(&data.high), black_box(&data.low), black_box(&data.close), ATR_PERIOD).unwrap()))
    });

    // Bollinger Bands
    group.bench_function("bollinger_direct", |b| {
        b.iter(|| black_box(bollinger(black_box(&data.close), BOLLINGER_PERIOD, BOLLINGER_STDDEV).unwrap()))
    });

    // Stochastic
    group.bench_function("stochastic_direct", |b| {
        b.iter(|| black_box(stochastic_fast(
            black_box(&data.high),
            black_box(&data.low),
            black_box(&data.close),
            STOCHASTIC_K_PERIOD,
            STOCHASTIC_D_PERIOD
        ).unwrap()))
    });

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

/// Measure throughput in elements per second for both modes.
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/throughput", GROUP_E07_END_TO_END));

    let sizes = [10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(Throughput::Elements(size as u64));

        // Direct mode throughput
        group.bench_with_input(
            BenchmarkId::new("direct_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_helper(black_box(data)))
            },
        );

        // Plan mode throughput
        group.bench_with_input(
            BenchmarkId::new("plan_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_plan_helper(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Repeated Execution (Plan Reuse)
// ============================================================================

/// Benchmark plan reuse - compile once, execute many times.
fn bench_plan_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/plan_reuse", GROUP_E07_END_TO_END));

    let size = 10_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);
    let ohlcv = OhlcvData::new(&data.high, &data.low, &data.close);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // Create executors once
    let plan_executor = PlanExecutor::new();
    let direct_executor = DirectExecutor::new();

    // Single execution - direct
    group.bench_with_input(
        BenchmarkId::new("direct_single", format_size(size)),
        &(&direct_executor, &ohlcv),
        |b, &(executor, ohlcv)| {
            b.iter(|| {
                black_box(executor.execute_all_baseline(black_box(ohlcv)).unwrap())
            })
        },
    );

    // Single execution - plan
    group.bench_with_input(
        BenchmarkId::new("plan_single", format_size(size)),
        &(&plan_executor, &ohlcv),
        |b, &(executor, ohlcv)| {
            b.iter(|| {
                black_box(executor.execute_all_baseline(black_box(ohlcv)).unwrap())
            })
        },
    );

    // Multiple datasets to simulate batch processing
    let datasets: Vec<_> = (0..5)
        .map(|seed| BenchmarkData::new(size, seed as u64))
        .collect();
    let ohlcvs: Vec<_> = datasets
        .iter()
        .map(|d| OhlcvData::new(&d.high, &d.low, &d.close))
        .collect();

    // Batch execution - direct
    group.bench_with_input(
        BenchmarkId::new("direct_batch_5", format_size(size)),
        &(&direct_executor, &ohlcvs),
        |b, &(executor, ohlcvs)| {
            b.iter(|| {
                for ohlcv in black_box(ohlcvs) {
                    black_box(executor.execute_all_baseline(ohlcv).unwrap());
                }
            })
        },
    );

    // Batch execution - plan
    group.bench_with_input(
        BenchmarkId::new("plan_batch_5", format_size(size)),
        &(&plan_executor, &ohlcvs),
        |b, &(executor, ohlcvs)| {
            b.iter(|| {
                for ohlcv in black_box(ohlcvs) {
                    black_box(executor.execute_all_baseline(ohlcv).unwrap());
                }
            })
        },
    );

    group.finish();
}

// ============================================================================
// EMA Fusion Analysis (Key Optimization)
// ============================================================================

/// Analyze EMA fusion benefits specifically.
fn bench_ema_fusion_benefit(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/ema_fusion", GROUP_E07_END_TO_END));

    let size = 10_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // Multiple EMAs - direct (separate calls)
    let ema_periods = [10, 12, 20, 26, 50, 100, 200];

    group.bench_function("ema_7_direct", |b| {
        b.iter(|| {
            for &period in &ema_periods {
                black_box(ema(black_box(&data.close), period).unwrap());
            }
        })
    });

    // Multiple EMAs - plan (fused)
    let requests: Vec<_> = ema_periods
        .iter()
        .map(|&p| IndicatorRequest::simple(IndicatorKind::Ema, p))
        .collect();
    let plan_executor = PlanExecutor::new();

    group.bench_with_input(
        BenchmarkId::new("ema_7_fused", "plan"),
        &(&data, &requests, &plan_executor),
        |b, &(data, requests, executor)| {
            b.iter(|| {
                black_box(executor.execute(&data.close, black_box(requests)).unwrap())
            })
        },
    );

    group.finish();
}

// ============================================================================
// Real-World Workload Simulation
// ============================================================================

/// Simulate a realistic trading workload.
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/realistic_workload", GROUP_E07_END_TO_END));

    // Simulate daily bars for 10 years
    let sizes = [(252, "1_year"), (2520, "10_years"), (25200, "100_years")];

    for (size, name) in sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);
        let ohlcv = OhlcvData::new(&data.high, &data.low, &data.close);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        let direct_executor = DirectExecutor::new();
        let plan_executor = PlanExecutor::new();

        // Direct mode
        group.bench_with_input(
            BenchmarkId::new("direct", name),
            &(&direct_executor, &ohlcv),
            |b, &(executor, ohlcv)| {
                b.iter(|| black_box(executor.execute_all_baseline(black_box(ohlcv)).unwrap()))
            },
        );

        // Plan mode
        group.bench_with_input(
            BenchmarkId::new("plan", name),
            &(&plan_executor, &ohlcv),
            |b, &(executor, ohlcv)| {
                b.iter(|| black_box(executor.execute_all_baseline(black_box(ohlcv)).unwrap()))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Final Go/No-Go Benchmarks
// ============================================================================

/// Critical benchmarks for the go/no-go decision.
fn bench_go_no_go(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/go_no_go", GROUP_E07_END_TO_END));

    // Test at key decision points
    let decision_points = [
        (10_000, 7, "baseline_7_10K"),
        (100_000, 7, "baseline_7_100K"),
        (10_000, 21, "scaling_21_10K"),
        (100_000, 21, "scaling_21_100K"),
    ];

    for (size, indicator_count, name) in decision_points {
        let data = BenchmarkData::new(size, DEFAULT_SEED);
        let requests = generate_scaled_requests(indicator_count);
        let direct_executor = DirectExecutor::new();
        let plan_executor = PlanExecutor::new();

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Direct mode
        group.bench_with_input(
            BenchmarkId::new("direct", name),
            &(&data, &requests, &direct_executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
                })
            },
        );

        // Plan mode
        group.bench_with_input(
            BenchmarkId::new("plan", name),
            &(&data, &requests, &plan_executor),
            |b, &(data, requests, executor)| {
                b.iter(|| {
                    black_box(executor.execute(&data.close, black_box(requests)).unwrap())
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
    name = end_to_end_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_direct_baseline,
        bench_plan_baseline,
        bench_direct_vs_plan,
        bench_workload_scaling_direct,
        bench_workload_scaling_plan,
        bench_workload_comparison,
        bench_data_size_scaling_direct,
        bench_data_size_scaling_plan,
        bench_individual_indicators,
        bench_throughput,
        bench_plan_reuse,
        bench_ema_fusion_benefit,
        bench_realistic_workload,
        bench_go_no_go
);

criterion_main!(end_to_end_benches);
