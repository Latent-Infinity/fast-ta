# Specification: fast-ta Micro-Experiments Implementation

## Overview

This specification implements a comprehensive micro-experiment registry for the fast-ta technical analysis library. The project validates performance hypotheses through 7 experiments (E01-E07) across 5 phases, building from foundation to end-to-end comparison with TA-Lib. The implementation creates a greenfield Rust workspace with benchmark-driven development to generate data for PRD v1.5 validation.

## Workflow Type

**Type**: feature

**Rationale**: This is a comprehensive feature development project building a new technical analysis library from scratch. It involves creating new crates, implementing core indicators, developing kernel fusion strategies, and validating architectural decisions through systematic benchmarking.

## Task Scope

### Services Involved
- **fast-ta-core** (primary) - Core library with indicators, kernels, and plan infrastructure
- **fast-ta-experiments** (primary) - Benchmark suite and experiment harness

### This Task Will:
- [x] Create Cargo workspace with multi-crate structure
- [x] Implement 7 baseline technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Stochastic)
- [x] Build 4 fusion kernels (RunningStat, EMA family, Rolling Extrema)
- [x] Create plan infrastructure with DAG-based execution
- [x] Run 7 micro-experiments (E01-E07) with benchmark data
- [x] Generate experiment reports with go/no-go decisions
- [x] Achieve ≥95% coverage on primitives, ≥90% on kernels
- [x] Validate or invalidate performance hypotheses
- [x] Update PRD to v1.5 with experimental findings

### Out of Scope:
- Production-ready API design (only experiment scaffolding)
- Full indicator library beyond the 7 baseline indicators
- FFI bindings or language interop
- GUI or visualization tools
- Real-time data integration
- Production deployment infrastructure

## Service Context

### fast-ta-core

**Tech Stack:**
- Language: Rust (Edition 2021, MSRV 1.75)
- Framework: None (library crate)
- Dependencies: num-traits, petgraph, rand (for test data generation)
- Key directories: `crates/fast-ta-core/src/{indicators,kernels,plan}`

**Entry Point:** `crates/fast-ta-core/src/lib.rs`

**How to Run:**
```bash
# Run tests
cargo test --package fast-ta-core

# Check coverage
cargo tarpaulin --package fast-ta-core --out Html
```

**Port:** N/A (library)

### fast-ta-experiments

**Tech Stack:**
- Language: Rust (Edition 2021)
- Framework: Criterion.rs (v0.5.1) for benchmarking
- Dependencies: criterion, fast-ta-core, rand (for data generation), serde, serde_json (for JSON output)
- Key directories: `crates/fast-ta-experiments/benches/`, `benches/experiments/E01-E07/`

**Entry Point:** `crates/fast-ta-experiments/benches/*.rs`

**How to Run:**
```bash
# Run all benchmarks
cargo bench --package fast-ta-experiments

# Run specific experiment
cargo bench --package fast-ta-experiments --bench e01_baseline

# View HTML reports
open target/criterion/report/index.html
```

**Port:** N/A (benchmark harness)

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `.gitignore` | root | Add Criterion output directories, benchmark JSON files |
| `docs/product-requirements.md` | root | Update to v1.5 with validated performance hypotheses |

## Files to Create

### Phase 0: Foundation (9 files)
| File | Service | Purpose |
|------|---------|---------|
| `Cargo.toml` | root | Workspace configuration |
| `crates/fast-ta-core/Cargo.toml` | core | Core library manifest |
| `crates/fast-ta-core/src/lib.rs` | core | Library entry point |
| `crates/fast-ta-core/src/error.rs` | core | Error types (InsufficientData, NumericConversion, CyclicDependency) |
| `crates/fast-ta-core/src/traits.rs` | core | Series trait definitions |
| `crates/fast-ta-experiments/Cargo.toml` | experiments | Experiments manifest |
| `crates/fast-ta-experiments/src/data.rs` | experiments | Synthetic data generators |
| `crates/fast-ta-experiments/benches/common/mod.rs` | experiments | Shared benchmark utilities |
| `docs/decisions/001-talib-comparison.md` | root | TA-Lib comparison strategy decision |

### Phase 1: Baseline Indicators (7 files)
| File | Service | Purpose |
|------|---------|---------|
| `crates/fast-ta-core/src/indicators/mod.rs` | core | Indicators module |
| `crates/fast-ta-core/src/indicators/sma.rs` | core | Simple Moving Average |
| `crates/fast-ta-core/src/indicators/ema.rs` | core | Exponential Moving Average |
| `crates/fast-ta-core/src/indicators/rsi.rs` | core | Relative Strength Index |
| `crates/fast-ta-core/src/indicators/macd.rs` | core | MACD indicator |
| `crates/fast-ta-core/src/indicators/atr.rs` | core | Average True Range |
| `crates/fast-ta-core/src/indicators/bollinger.rs` | core | Bollinger Bands |
| `crates/fast-ta-core/src/indicators/stochastic.rs` | core | Stochastic Oscillator |

### Phase 2: Kernels & Experiments (11 files)
| File | Service | Purpose |
|------|---------|---------|
| `crates/fast-ta-core/src/kernels/mod.rs` | core | Kernels module |
| `crates/fast-ta-core/src/kernels/running_stat.rs` | core | Welford's algorithm kernel |
| `crates/fast-ta-core/src/kernels/ema_fusion.rs` | core | Multi-EMA fusion kernel |
| `crates/fast-ta-core/src/kernels/rolling_extrema.rs` | core | Monotonic deque kernel |
| `crates/fast-ta-experiments/benches/e01_baseline.rs` | experiments | E01 benchmark |
| `crates/fast-ta-experiments/benches/e02_running_stat.rs` | experiments | E02 benchmark |
| `crates/fast-ta-experiments/benches/e03_ema_fusion.rs` | experiments | E03 benchmark |
| `crates/fast-ta-experiments/benches/e04_rolling_extrema.rs` | experiments | E04 benchmark |
| `benches/experiments/E01_baseline/REPORT.md` | experiments | E01 findings |
| `benches/experiments/E02_running_stat/REPORT.md` | experiments | E02 findings |
| `benches/experiments/E03_ema_fusion/REPORT.md` | experiments | E03 findings |
| `benches/experiments/E04_rolling_extrema/REPORT.md` | experiments | E04 findings |

### Phase 3: Plan Infrastructure (7 files)
| File | Service | Purpose |
|------|---------|---------|
| `crates/fast-ta-core/src/plan/mod.rs` | core | Plan module |
| `crates/fast-ta-core/src/plan/registry.rs` | core | Indicator registry |
| `crates/fast-ta-core/src/plan/spec.rs` | core | IndicatorSpec struct |
| `crates/fast-ta-core/src/plan/dag.rs` | core | DAG builder with petgraph |
| `crates/fast-ta-experiments/benches/e05_plan_overhead.rs` | experiments | E05 benchmark |
| `crates/fast-ta-experiments/benches/e06_memory_writes.rs` | experiments | E06 benchmark |
| `benches/experiments/E05_plan_overhead/REPORT.md` | experiments | E05 findings |
| `benches/experiments/E06_memory_writes/REPORT.md` | experiments | E06 findings |

### Phase 4: End-to-End (3 files)
| File | Service | Purpose |
|------|---------|---------|
| `crates/fast-ta-experiments/benches/e07_end_to_end.rs` | experiments | E07 benchmark |
| `benches/experiments/E07_end_to_end/REPORT.md` | experiments | E07 findings |
| `crates/fast-ta-experiments/src/talib_baseline.rs` | experiments | TA-Lib comparison wrapper |

### Phase 5: Documentation (2 files)
| File | Service | Purpose |
|------|---------|---------|
| `docs/experiments/SUMMARY.md` | root | Consolidated experiment results |
| `docs/product-requirements.md` | root | Updated PRD v1.5 |

## Patterns to Follow

### Criterion Benchmark Setup

From research phase findings:

```toml
# Cargo.toml pattern
[dev-dependencies]
criterion = "0.5.1"

[[bench]]
name = "benchmark_name"
harness = false  # CRITICAL: Required for Criterion
```

```rust
// Benchmark structure pattern
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn benchmark_function(c: &mut Criterion) {
    let data = generate_test_data();

    c.bench_function("operation_name", |b| {
        b.iter(|| {
            black_box(operation(black_box(&data)))
        })
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

**Key Points:**
- MUST use `black_box()` from `std::hint` to prevent dead code elimination
- Import black_box separately: `use std::hint::black_box;`
- `harness = false` is mandatory in `[[bench]]` section
- Outputs to `./target/criterion/` with HTML and JSON
- Use consistent data sizes: 1K, 10K, 100K, 1M
- For long-running benchmarks, consider `SamplingMode::Flat` to reduce measurement overhead

### Generic Numeric Operations

From research phase findings with num-traits:

```rust
use num_traits::Float;

fn sma<T: Float>(data: &[T], period: usize) -> Result<Vec<T>, Error> {
    if data.len() < period {
        return Err(Error::InsufficientData);
    }

    let mut result = vec![T::nan(); data.len()];
    let mut sum = T::zero();

    for i in 0..period {
        sum = sum + data[i];
    }

    let period_t = T::from(period).ok_or(Error::NumericConversion)?;
    result[period - 1] = sum / period_t;

    for i in period..data.len() {
        sum = sum + data[i] - data[i - period];
        result[i] = sum / period_t;
    }

    Ok(result)
}
```

**Key Points:**
- Use `Float` trait for f32/f64 abstraction
- `NumCast::from()` returns `Option` - ALWAYS use `.ok_or()` or `match`, NEVER `.unwrap()`
- Convert once and reuse the value (e.g., `let period_t = T::from(period).ok_or(Error::NumericConversion)?;`)
- Explicitly handle `NaN` with `Float::is_nan()`
- Return `Result` for error handling
- Define `Error::NumericConversion` variant for failed type conversions

### Petgraph DAG Operations

From research phase findings:

```rust
use petgraph::graph::DiGraph;
use petgraph::algo::toposort;

fn build_execution_order(graph: &DiGraph<IndicatorNode, ()>) -> Result<Vec<NodeIndex>, Error> {
    match toposort(graph, None) {
        Ok(order) => Ok(order),
        Err(cycle) => Err(Error::CyclicDependency(cycle.node_id())),
    }
}
```

**Key Points:**
- `toposort()` returns `Result<Vec<NodeId>, Cycle>` - MUST handle errors
- Use `DiGraph` for directed dependency graphs
- Process nodes in topological order for correctness
- Handle cycle detection explicitly

### Welford's Algorithm (Running Statistics)

Pattern for numerically stable rolling statistics:

```rust
struct RunningStat {
    count: usize,
    mean: f64,
    m2: f64,  // Sum of squared differences from mean
}

impl RunningStat {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count as f64) }
    }

    fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }
}
```

**Key Points:**
- Single-pass algorithm for mean, variance, stddev
- Numerically stable with extreme values
- O(1) per-element update
- Foundation for fused Bollinger Bands + StdDev

## Requirements

### Functional Requirements

#### 1. **Cargo Workspace Structure**
   - Description: Multi-crate workspace with core library and experiments
   - Acceptance:
     - Workspace builds with `cargo build --workspace`
     - Edition 2021, MSRV 1.75
     - `.gitignore` excludes `target/`, `Cargo.lock`, `*.rs.bk`

#### 2. **Synthetic Data Generators**
   - Description: Deterministic test data with seeded RNG
   - Acceptance:
     - `generate_random_walk(n, seed)` produces consistent output
     - `generate_ohlcv(n, seed)` maintains OHLCV invariants (High ≥ Low)
     - `inject_nans(series, ratio, seed)` creates sparse data
     - All generators covered ≥95%

#### 3. **Baseline Indicators (7 Total)**
   - Description: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Stochastic
   - Acceptance:
     - Each indicator has unit tests with reference values
     - O(n) time complexity verified
     - Generic over f32/f64 via `Float` trait
     - Coverage ≥95% on indicator logic
     - First `period-1` values are NaN where appropriate

#### 4. **Fusion Kernels (3 Total)**
   - Description: RunningStat, EMA family, Rolling extrema
   - Acceptance:
     - Property tests verify fused == unfused output
     - Welford's algorithm for numeric stability
     - Monotonic deque for O(n) rolling extrema
     - Coverage ≥90%

#### 5. **Plan Infrastructure**
   - Description: DAG-based execution planner with dependency resolution
   - Acceptance:
     - Topological sort provides valid execution order
     - Cycle detection returns `CyclicDependency` error
     - Linear dependencies (A→B→C) work correctly
     - Diamond dependencies (A→B,C; B,C→D) work correctly

#### 6. **Experiment Benchmarks (E01-E07)**
   - Description: Criterion-based benchmarks with statistical rigor
   - Acceptance:
     - Each experiment outputs JSON to `benches/experiments/EXX/results.json`
     - Each experiment has REPORT.md with go/no-go decision
     - Data sizes: 1K, 10K, 100K, 1M points
     - Confidence intervals included in results

#### 7. **PRD Update**
   - Description: Update product requirements with validated hypotheses
   - Acceptance:
     - Version bumped to 1.5
     - Section 1.3 performance hypotheses marked validated/invalidated
     - Section 6 architecture updated with experiment findings
     - Section 13 timeline reflects actual complexity

### Edge Cases

1. **NaN Handling in Indicators** - Indicators must propagate NaN correctly in lookback periods and maintain it through calculations
2. **Insufficient Data Errors** - Return `Error::InsufficientData` when series length < period
3. **Numeric Conversion Failures** - Return `Error::NumericConversion` when `T::from()` fails (e.g., converting usize to generic Float type)
4. **Numeric Overflow** - Use Welford's algorithm to prevent catastrophic cancellation with extreme values
5. **Empty Sequences** - Return error for zero-length input data
6. **Cyclic Dependencies in DAG** - Detect and report cycles with `CyclicDependency(node_id)`
7. **OHLCV Invariants** - Validate High ≥ Close ≥ Low, High ≥ Open ≥ Low in test data
8. **Period=1 Edge Cases** - SMA(1) should equal input, EMA(1) degenerates
9. **All Gains/All Losses in RSI** - Handle boundary conditions (RSI=100 or RSI=0)

## Implementation Notes

### DO
- Follow TDD: Write tests BEFORE implementations (Tasks X.1, X.3, etc. before X.2, X.4)
- Use `black_box()` in ALL benchmark iterations
- Set `harness = false` in every `[[bench]]` section
- Handle `toposort()` error explicitly - it WILL return `Err` on cycles
- Use `Float` trait for all numeric operations to support f32/f64
- Run `cargo tarpaulin` after each phase to verify coverage targets
- Document go/no-go decisions in REPORT.md files with data justification
- Use seeded RNG for reproducible benchmark data
- Measure statistical confidence intervals with Criterion
- Profile with `perf` counters where available (cache misses, branch mispredictions)

### DON'T
- Skip test tasks - they define acceptance criteria
- Forget `black_box()` - compiler will optimize away your benchmarks
- Use `unwrap()` on `NumCast::from()` - it returns `Option`
- Modify node indices after `petgraph` deletions (use `StableGraph` if needed)
- Create new statistical implementations - use Welford's for variance
- Assume TA-Lib integration is simple - follow decision from Task 0.6
- Skip coverage checks - ≥95% on primitives, ≥90% on kernels is mandatory
- Batch completions - mark tasks complete immediately after finishing
- Create implementations without corresponding tests
- Use naive O(n×k) algorithms without benchmarking first

## Development Environment

### Start Services

```bash
# Install Rust (MSRV 1.75)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install coverage tool
cargo install cargo-tarpaulin

# Create workspace structure
mkdir -p crates/{fast-ta-core,fast-ta-experiments}/src
mkdir -p benches/experiments/{E01_baseline,E02_running_stat,E03_ema_fusion,E04_rolling_extrema,E05_plan_overhead,E06_memory_writes,E07_end_to_end}

# Build workspace
cargo build --workspace

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Check coverage
cargo tarpaulin --workspace --out Html
```

### Service URLs
- N/A (library and benchmark suite, no running services)

### Benchmark Outputs
- HTML Reports: `./target/criterion/report/index.html`
- JSON Data: `./target/criterion/*/base/estimates.json`
- Experiment Reports: `./benches/experiments/EXX/REPORT.md`

### Required Environment Variables
- None (all data is synthetic and seeded)

## Success Criteria

The task is complete when:

1. [x] Phase 0: Workspace builds, data generators working with ≥95% coverage
2. [x] Phase 1: All 7 baseline indicators implemented with ≥95% coverage
3. [x] Phase 2: E01-E04 complete with REPORT.md and go/no-go decisions
4. [x] Phase 3: Plan infrastructure functional, E05-E06 complete
5. [x] Phase 4: E07 complete with TA-Lib comparison
6. [x] Phase 5: PRD v1.5 published with all experiment findings
7. [x] All tests pass: `cargo test --workspace` succeeds
8. [x] Coverage targets met: ≥95% primitives, ≥90% kernels
9. [x] All benchmarks run: `cargo bench --workspace` succeeds
10. [x] No compiler warnings on `cargo clippy --workspace`

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests

| Test | File | What to Verify |
|------|------|----------------|
| Data generator determinism | `crates/fast-ta-experiments/src/data.rs` | Same seed produces identical output |
| SMA correctness | `crates/fast-ta-core/src/indicators/sma.rs` | Matches reference values, handles edge cases |
| EMA correctness | `crates/fast-ta-core/src/indicators/ema.rs` | Standard and Wilder variants match references |
| RSI boundary conditions | `crates/fast-ta-core/src/indicators/rsi.rs` | All gains → 100, all losses → 0 |
| MACD components | `crates/fast-ta-core/src/indicators/macd.rs` | Line, signal, histogram all correct |
| ATR with gaps | `crates/fast-ta-core/src/indicators/atr.rs` | True Range handles overnight gaps |
| Bollinger Bands | `crates/fast-ta-core/src/indicators/bollinger.rs` | Middle, upper, lower bands verified |
| Stochastic variants | `crates/fast-ta-core/src/indicators/stochastic.rs` | Fast, slow, full stochastic all correct |
| Welford stability | `crates/fast-ta-core/src/kernels/running_stat.rs` | Handles extreme values without overflow |
| EMA fusion equivalence | `crates/fast-ta-core/src/kernels/ema_fusion.rs` | Fused == unfused output |
| Rolling extrema correctness | `crates/fast-ta-core/src/kernels/rolling_extrema.rs` | Matches naive O(n×k) scan |
| DAG cycle detection | `crates/fast-ta-core/src/plan/dag.rs` | Returns error on cyclic dependencies |
| DAG topological sort | `crates/fast-ta-core/src/plan/dag.rs` | Valid execution order for diamond DAG |

### Integration Tests

| Test | Services | What to Verify |
|------|----------|----------------|
| Indicator chain | core | SMA → Bollinger Bands pipeline works |
| MACD construction | core | EMA(12), EMA(26) → MACD → Signal |
| Plan execution | core | Registry → DAG → Execution produces correct results |
| Benchmark data flow | experiments ↔ core | Generated data fed to indicators correctly |

### Coverage Tests

| Check | Command | Expected |
|-------|---------|----------|
| Phase 0 coverage | `cargo tarpaulin -p fast-ta-experiments --lib` | ≥95% on data generators |
| Phase 1 coverage | `cargo tarpaulin -p fast-ta-core --lib` | ≥95% on indicators module |
| Phase 2 coverage | `cargo tarpaulin -p fast-ta-core -- kernels` | ≥90% on kernels module |
| Phase 3 coverage | `cargo tarpaulin -p fast-ta-core -- plan` | ≥90% on plan module |

### Benchmark Verification

| Experiment | Output File | Checks |
|------------|-------------|--------|
| E01 Baseline | `benches/experiments/E01_baseline/REPORT.md` | Baseline established for all 7 indicators |
| E02 RunningStat | `benches/experiments/E02_running_stat/REPORT.md` | Go/no-go on ≥20% speedup documented |
| E03 EMA Fusion | `benches/experiments/E03_ema_fusion/REPORT.md` | Go/no-go on ≥15% speedup for ≥10 EMAs |
| E04 Rolling Extrema | `benches/experiments/E04_rolling_extrema/REPORT.md` | O(n) vs O(n×k) comparison shows ≥5× speedup |
| E05 Plan Overhead | `benches/experiments/E05_plan_overhead/REPORT.md` | Break-even point calculated |
| E06 Memory Writes | `benches/experiments/E06_memory_writes/REPORT.md` | Write pattern recommendation |
| E07 End-to-End | `benches/experiments/E07_end_to_end/REPORT.md` | TA-Lib comparison complete, final architecture decision |

### Build Verification

| Check | Command | Expected Output |
|-------|---------|----------------|
| Workspace builds | `cargo build --workspace` | Success with no errors |
| All tests pass | `cargo test --workspace` | All tests passing |
| Benchmarks compile | `cargo bench --workspace --no-run` | All benches compile |
| No warnings | `cargo clippy --workspace -- -D warnings` | Zero warnings |
| Formatting | `cargo fmt --workspace -- --check` | No formatting issues |

### Documentation Verification

| Document | Check | Expected |
|----------|-------|----------|
| `docs/decisions/001-talib-comparison.md` | Exists and complete | Decision documented with rationale |
| `docs/experiments/SUMMARY.md` | Exists and complete | All 7 experiments summarized with data |
| `docs/product-requirements.md` | Version 1.5 | Performance hypotheses updated, architecture revised |
| All REPORT.md files | 7 files exist | Each has go/no-go decision with supporting data |

### QA Sign-off Requirements

- [x] All 13 indicator/kernel unit test suites pass
- [x] All 4 integration test scenarios pass
- [x] Coverage targets met: ≥95% primitives, ≥90% kernels/plan
- [x] All 7 experiment reports exist with go/no-go decisions
- [x] Build verification: workspace builds, tests pass, clippy clean
- [x] Documentation complete: decision docs, experiment summary, PRD v1.5
- [x] No regressions in existing functionality (N/A - greenfield project)
- [x] Code follows Rust 2021 idioms and best practices
- [x] No security vulnerabilities: `cargo audit` passes
- [x] Benchmark outputs verified: HTML reports and JSON data present
- [x] MSRV 1.75 verified: builds on minimum supported Rust version

### Performance Validation

| Metric | Target | Validation Method |
|--------|--------|------------------|
| SMA O(n) | Linear time | Benchmark shows constant time per element |
| EMA O(n) | Linear time | Benchmark shows constant time per element |
| Rolling Extrema | O(n) not O(n×k) | E04 shows amortized O(1) per element |
| RunningStat fusion | ≥20% speedup | E02 benchmark vs separate passes |
| EMA fusion | ≥15% speedup on ≥10 EMAs | E03 benchmark vs independent calls |
| Plan overhead | Break-even <100 executions | E05 calculation documented |
| Full architecture | ≥1.5× speedup on ≥20 indicators | E07 plan mode vs direct mode |

---

**Total Tasks**: 39 tasks across 5 phases
**Estimated Files**: 40+ new files (Rust source, benchmarks, reports, docs)
**Critical Dependencies**: Criterion.rs v0.5.1, petgraph v0.6.x, num-traits v0.2.x
**Coverage Enforcement**: cargo-tarpaulin with phase-specific targets
**Decision Points**: Task 0.6 (TA-Lib comparison strategy) requires human approval

This specification serves as the implementation blueprint for the entire micro-experiments plan, with clear acceptance criteria at each phase for systematic validation.
