//! E05: Plan Compilation Overhead Benchmarks
//!
//! This experiment measures the cost of plan infrastructure (registry, DAG construction,
//! topological sort) and calculates the break-even point vs direct indicator computation.
//!
//! # Hypothesis
//!
//! The plan infrastructure incurs a fixed overhead for:
//! 1. Registry population with indicator specifications
//! 2. DAG construction from dependency graph
//! 3. Topological sort to determine execution order
//!
//! This overhead should be amortized when:
//! - Computing many indicators in a single plan
//! - Reusing the same plan across multiple data batches
//! - Taking advantage of kernel fusion opportunities
//!
//! # Methodology
//!
//! We benchmark:
//! 1. **Registry operations**: Time to register indicators
//! 2. **DAG construction**: Time to build dependency graph from registry
//! 3. **Topological sort**: Time to compute execution order
//! 4. **Full plan compilation**: Total overhead from empty registry to execution plan
//! 5. **Direct vs plan comparison**: Compare overhead against actual indicator computation
//!
//! # Success Criteria
//!
//! - **BREAK-EVEN**: Calculate how many executions needed to amortize compilation cost
//! - **RECOMMENDATION**: Provide guidance on when to use plan mode vs direct mode
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e05_plan_overhead/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E05_plan_overhead/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E05_PLAN_OVERHEAD,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    RSI_PERIOD, ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STDDEV,
    STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD,
};

// Core library imports
use fast_ta_core::plan::{IndicatorKind, IndicatorSpec, Registry};
use fast_ta_core::plan::dag::DagBuilder;
use fast_ta_core::indicators::{
    sma, ema, rsi, macd, atr, bollinger, stochastic_fast,
};

// Data generators
use fast_ta_experiments::data::{generate_random_walk, generate_ohlcv};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Standard periods for benchmark indicators
const SMA_PERIOD: usize = 20;
const EMA_PERIOD: usize = 20;

/// Indicator counts to test for scaling analysis
const INDICATOR_COUNTS: [usize; 5] = [1, 5, 10, 20, 50];

/// Dependency depths to test (linear chains)
const DEPENDENCY_DEPTHS: [usize; 4] = [1, 3, 5, 10];

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
// Registry Operations
// ============================================================================

/// Create a registry with N simple SMA indicators.
fn create_simple_registry(count: usize) -> Registry {
    let mut registry = Registry::with_capacity(count);
    for i in 0..count {
        let period = 10 + i; // Varying periods to avoid deduplication
        registry.register(
            format!("sma_{}", period),
            IndicatorSpec::new(IndicatorKind::Sma, period),
        );
    }
    registry
}

/// Create a registry with a mix of indicator types.
fn create_mixed_registry(count: usize) -> Registry {
    let mut registry = Registry::with_capacity(count);
    let indicator_types = [
        IndicatorKind::Sma,
        IndicatorKind::Ema,
        IndicatorKind::Rsi,
        IndicatorKind::Atr,
    ];

    for i in 0..count {
        let kind = indicator_types[i % indicator_types.len()];
        let period = 10 + (i / indicator_types.len());
        registry.register(
            format!("{}_{}", kind.name().to_lowercase(), period),
            IndicatorSpec::new(kind, period),
        );
    }
    registry
}

/// Create a registry with linear dependency chain (A -> B -> C -> ...).
fn create_linear_dependency_registry(depth: usize) -> Registry {
    let mut registry = Registry::with_capacity(depth);

    for i in 0..depth {
        let id = format!("indicator_{}", i);
        let mut spec = IndicatorSpec::new(IndicatorKind::Custom, 10 + i);

        if i > 0 {
            spec = spec.with_dependency(format!("indicator_{}", i - 1));
        }

        registry.register(id, spec);
    }
    registry
}

/// Create a registry with diamond dependency pattern.
fn create_diamond_registry(width: usize) -> Registry {
    let mut registry = Registry::with_capacity(width + 2);

    // Root node
    registry.register("root", IndicatorSpec::new(IndicatorKind::Sma, 10));

    // Middle layer
    for i in 0..width {
        registry.register(
            format!("mid_{}", i),
            IndicatorSpec::new(IndicatorKind::Ema, 10 + i)
                .with_dependency("root"),
        );
    }

    // Leaf node depends on all middle nodes
    let mut leaf_spec = IndicatorSpec::new(IndicatorKind::Custom, 20);
    for i in 0..width {
        leaf_spec = leaf_spec.with_dependency(format!("mid_{}", i));
    }
    registry.register("leaf", leaf_spec);

    registry
}

/// Create a realistic trading indicator registry.
fn create_realistic_registry() -> Registry {
    let mut registry = Registry::with_capacity(10);

    // Common indicators used in trading systems
    registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, SMA_PERIOD));
    registry.register("sma_50", IndicatorSpec::new(IndicatorKind::Sma, 50));
    registry.register("ema_12", IndicatorSpec::new(IndicatorKind::Ema, MACD_FAST_PERIOD));
    registry.register("ema_26", IndicatorSpec::new(IndicatorKind::Ema, MACD_SLOW_PERIOD));
    registry.register("rsi_14", IndicatorSpec::new(IndicatorKind::Rsi, RSI_PERIOD));
    registry.register("atr_14", IndicatorSpec::new(IndicatorKind::Atr, ATR_PERIOD));
    registry.register("macd", IndicatorSpec::macd(MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD));
    registry.register("bb_20", IndicatorSpec::bollinger(BOLLINGER_PERIOD, BOLLINGER_STDDEV));
    registry.register("stoch_14", IndicatorSpec::stochastic_fast(STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD));

    registry
}

// ============================================================================
// Registry Registration Benchmarks
// ============================================================================

/// Benchmark time to register N indicators.
fn bench_registry_registration(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/registry_registration", GROUP_E05_PLAN_OVERHEAD));

    for &count in &INDICATOR_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("simple_indicators", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    black_box(create_simple_registry(count))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mixed_indicators", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    black_box(create_mixed_registry(count))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// DAG Construction Benchmarks
// ============================================================================

/// Benchmark DAG construction from registry.
fn bench_dag_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/dag_construction", GROUP_E05_PLAN_OVERHEAD));

    // Test with varying number of independent indicators
    for &count in &INDICATOR_COUNTS {
        let registry = create_simple_registry(count);

        group.bench_with_input(
            BenchmarkId::new("independent_nodes", count),
            &registry,
            |b, registry| {
                b.iter(|| {
                    black_box(DagBuilder::from_registry(black_box(registry)))
                })
            },
        );
    }

    // Test with linear dependency chains
    for &depth in &DEPENDENCY_DEPTHS {
        let registry = create_linear_dependency_registry(depth);

        group.bench_with_input(
            BenchmarkId::new("linear_chain", depth),
            &registry,
            |b, registry| {
                b.iter(|| {
                    black_box(DagBuilder::from_registry(black_box(registry)))
                })
            },
        );
    }

    // Test with diamond dependencies
    let diamond_widths = [2, 5, 10, 20];
    for &width in &diamond_widths {
        let registry = create_diamond_registry(width);

        group.bench_with_input(
            BenchmarkId::new("diamond_pattern", width),
            &registry,
            |b, registry| {
                b.iter(|| {
                    black_box(DagBuilder::from_registry(black_box(registry)))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Topological Sort Benchmarks
// ============================================================================

/// Benchmark topological sort (build step).
fn bench_topological_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/topological_sort", GROUP_E05_PLAN_OVERHEAD));

    // Test with varying number of independent indicators
    for &count in &INDICATOR_COUNTS {
        let registry = create_simple_registry(count);

        group.bench_with_input(
            BenchmarkId::new("independent_nodes", count),
            &registry,
            |b, registry| {
                b.iter(|| {
                    let builder = DagBuilder::from_registry(black_box(registry));
                    black_box(builder.build().unwrap())
                })
            },
        );
    }

    // Test with linear dependency chains
    for &depth in &DEPENDENCY_DEPTHS {
        let registry = create_linear_dependency_registry(depth);

        group.bench_with_input(
            BenchmarkId::new("linear_chain", depth),
            &registry,
            |b, registry| {
                b.iter(|| {
                    let builder = DagBuilder::from_registry(black_box(registry));
                    black_box(builder.build().unwrap())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Full Plan Compilation Benchmarks
// ============================================================================

/// Benchmark full plan compilation (registry + DAG + sort).
fn bench_full_plan_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/full_compilation", GROUP_E05_PLAN_OVERHEAD));

    for &count in &INDICATOR_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("simple_plan", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let registry = create_simple_registry(count);
                    let builder = DagBuilder::from_registry(&registry);
                    black_box(builder.build().unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mixed_plan", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let registry = create_mixed_registry(count);
                    let builder = DagBuilder::from_registry(&registry);
                    black_box(builder.build().unwrap())
                })
            },
        );
    }

    // Realistic trading system
    group.bench_function("realistic_trading_system", |b| {
        b.iter(|| {
            let registry = create_realistic_registry();
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    group.finish();
}

// ============================================================================
// Plan Reuse Benchmarks
// ============================================================================

/// Benchmark accessing execution order from cached plan.
fn bench_plan_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/plan_reuse", GROUP_E05_PLAN_OVERHEAD));

    // Create plans once, benchmark reuse
    for &count in &INDICATOR_COUNTS {
        let registry = create_simple_registry(count);
        let plan = DagBuilder::from_registry(&registry).build().unwrap();

        group.bench_with_input(
            BenchmarkId::new("execution_order_access", count),
            &plan,
            |b, plan| {
                b.iter(|| {
                    black_box(plan.execution_order())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("iter_order", count),
            &plan,
            |b, plan| {
                b.iter(|| {
                    for id in plan.iter() {
                        black_box(id);
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Direct vs Plan Mode Comparison
// ============================================================================

/// Compute all 7 baseline indicators directly (no plan overhead).
fn compute_direct_baseline(data: &BenchmarkData) {
    // SMA
    let _ = black_box(sma(&data.prices, SMA_PERIOD).unwrap());
    // EMA
    let _ = black_box(ema(&data.prices, EMA_PERIOD).unwrap());
    // RSI
    let _ = black_box(rsi(&data.prices, RSI_PERIOD).unwrap());
    // MACD
    let _ = black_box(macd(&data.prices, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD).unwrap());
    // ATR
    let _ = black_box(atr(&data.high, &data.low, &data.close, ATR_PERIOD).unwrap());
    // Bollinger Bands
    let _ = black_box(bollinger(&data.prices, BOLLINGER_PERIOD, BOLLINGER_STDDEV).unwrap());
    // Stochastic
    let _ = black_box(stochastic_fast(&data.high, &data.low, &data.close, STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD).unwrap());
}

/// Benchmark direct indicator computation (no plan).
fn bench_direct_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/direct_computation", GROUP_E05_PLAN_OVERHEAD));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Individual indicators
        group.bench_with_input(
            BenchmarkId::new("sma_20", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(sma(black_box(&data.prices), SMA_PERIOD).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ema_20", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema(black_box(&data.prices), EMA_PERIOD).unwrap()))
            },
        );

        // All baseline indicators together
        group.bench_with_input(
            BenchmarkId::new("all_7_indicators", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| compute_direct_baseline(black_box(data)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Break-Even Analysis
// ============================================================================

/// Benchmark plan compilation + execution vs direct execution.
/// This helps calculate the break-even point.
fn bench_break_even_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/break_even", GROUP_E05_PLAN_OVERHEAD));

    let size = 10_000; // Fixed size for break-even calculation
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // Measure plan compilation overhead alone
    group.bench_function("plan_compilation_only", |b| {
        b.iter(|| {
            let registry = create_realistic_registry();
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    // Measure direct computation (no plan)
    group.bench_with_input(
        BenchmarkId::new("direct_7_indicators", format_size(size)),
        &data,
        |b, data| {
            b.iter(|| compute_direct_baseline(black_box(data)))
        },
    );

    // Measure plan compilation + single execution
    group.bench_with_input(
        BenchmarkId::new("plan_compile_plus_exec", format_size(size)),
        &data,
        |b, data| {
            b.iter(|| {
                // Compile plan
                let _registry = create_realistic_registry();
                let _builder = DagBuilder::from_registry(&_registry);
                let _plan = black_box(_builder.build().unwrap());

                // Execute indicators (simulated - just direct calls for now)
                compute_direct_baseline(black_box(data))
            })
        },
    );

    // Measure cached plan + multiple executions (simulate reuse)
    let registry = create_realistic_registry();
    let plan = DagBuilder::from_registry(&registry).build().unwrap();

    group.bench_with_input(
        BenchmarkId::new("cached_plan_exec", format_size(size)),
        &(plan, data),
        |b, (_plan, data)| {
            b.iter(|| {
                // Plan is already compiled, just execute
                compute_direct_baseline(black_box(data))
            })
        },
    );

    group.finish();
}

// ============================================================================
// Scaling Analysis
// ============================================================================

/// Benchmark how plan overhead scales with indicator count.
fn bench_indicator_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/indicator_scaling", GROUP_E05_PLAN_OVERHEAD));

    let extended_counts = [1, 2, 5, 10, 20, 50, 100];

    for &count in &extended_counts {
        group.bench_with_input(
            BenchmarkId::new("full_plan_overhead", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let registry = create_simple_registry(count);
                    let builder = DagBuilder::from_registry(&registry);
                    black_box(builder.build().unwrap())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

/// Benchmark plan compilation throughput (plans per second).
fn bench_compilation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/compilation_throughput", GROUP_E05_PLAN_OVERHEAD));

    group
        .throughput(criterion::Throughput::Elements(1))
        .sample_size(100);

    // Small plan (1 indicator)
    group.bench_function("plan_1_indicator", |b| {
        b.iter(|| {
            let registry = create_simple_registry(1);
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    // Medium plan (10 indicators)
    group.bench_function("plan_10_indicators", |b| {
        b.iter(|| {
            let registry = create_simple_registry(10);
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    // Realistic plan
    group.bench_function("plan_realistic_9_indicators", |b| {
        b.iter(|| {
            let registry = create_realistic_registry();
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    // Large plan (50 indicators)
    group.bench_function("plan_50_indicators", |b| {
        b.iter(|| {
            let registry = create_simple_registry(50);
            let builder = DagBuilder::from_registry(&registry);
            black_box(builder.build().unwrap())
        })
    });

    group.finish();
}

// ============================================================================
// Registry Query Benchmarks
// ============================================================================

/// Benchmark registry query operations.
fn bench_registry_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/registry_queries", GROUP_E05_PLAN_OVERHEAD));

    let registry = create_realistic_registry();

    group.bench_function("get_by_id", |b| {
        b.iter(|| {
            black_box(registry.get(black_box("sma_20")))
        })
    });

    group.bench_function("contains_check", |b| {
        b.iter(|| {
            black_box(registry.contains(black_box("macd")))
        })
    });

    group.bench_function("find_by_config", |b| {
        b.iter(|| {
            black_box(registry.find_by_config(black_box("SMA_20")))
        })
    });

    group.bench_function("find_by_kind", |b| {
        b.iter(|| {
            let results: Vec<_> = registry.find_by_kind(IndicatorKind::Sma).collect();
            black_box(results)
        })
    });

    group.bench_function("validate_dependencies", |b| {
        b.iter(|| {
            black_box(registry.validate_dependencies())
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = plan_overhead_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_registry_registration,
        bench_dag_construction,
        bench_topological_sort,
        bench_full_plan_compilation,
        bench_plan_reuse,
        bench_direct_computation,
        bench_break_even_analysis,
        bench_indicator_count_scaling,
        bench_compilation_throughput,
        bench_registry_queries
);

criterion_main!(plan_overhead_benches);
