# E07: End-to-End Workload Benchmarks

## Experiment Overview

**Experiment ID**: E07
**Name**: End-to-End Workload Comparison
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

## Objective

Provide the final comprehensive comparison between:

1. **TA-Lib baseline** (via golden file reference timings)
2. **Direct mode** (independent indicator computation)
3. **Plan mode** (fused kernel execution with DAG optimization)

This experiment determines whether the plan-based architecture provides sufficient performance benefit to justify its complexity.

### Hypothesis

Plan mode with fused kernels should outperform direct mode when:

1. Computing 6+ indicators in a single workload
2. Multiple EMA-based indicators share data access (EMA fusion)
3. Bollinger Bands benefit from running_stat fusion (mean + stddev)
4. Stochastic benefits from rolling_extrema fusion (max + min)
5. Plan compilation overhead is amortized over multiple executions

### Success Criteria

| Decision | Condition | Recommendation |
|----------|-----------|----------------|
| **GO (Plan architecture)** | Plan mode achieves ≥1.5× speedup for ≥20 indicators | Proceed with plan-based API as default |
| **CONDITIONAL GO** | Plan mode achieves ≥1.2× speedup for baseline 7 indicators | Use plan for multi-indicator workloads only |
| **NO-GO** | Plan mode shows <1.1× speedup or is slower than direct mode | Prefer direct mode, simplify architecture |

## Approaches Benchmarked

### 1. Direct Mode Baseline

Independent indicator computation without fusion:

| Indicator | Implementation | Expected O(n) |
|-----------|---------------|---------------|
| SMA(20) | Rolling sum | 1 pass |
| EMA(20) | Standard EMA | 1 pass |
| RSI(14) | Wilder smoothing | 1 pass |
| MACD(12,26,9) | 3 EMAs + subtraction | 3 passes |
| ATR(14) | True range + Wilder | 2 passes |
| Bollinger(20,2) | SMA + rolling stddev | 2 passes |
| Stochastic(14,3) | Rolling max/min + SMA | 3 passes |

**Total passes (direct)**: ~13 passes over data

### 2. Plan Mode with Fusion

Optimized computation using fused kernels:

| Indicator | Fusion Strategy | Expected O(n) |
|-----------|----------------|---------------|
| SMA(20) | Standard | 1 pass |
| EMA(20) | ema_multi fusion | Shared pass |
| RSI(14) | Standard | 1 pass |
| MACD(12,26,9) | macd_fusion | 1 pass (fused EMAs) |
| ATR(14) | Standard | 1 pass |
| Bollinger(20,2) | running_stats fusion | 1 pass (mean+stddev) |
| Stochastic(14,3) | rolling_extrema fusion | 1 pass (max+min) |

**Total passes (plan)**: ~7-8 passes over data

### 3. Workload Scaling

Test with increasing indicator counts:

| Count | Description | Fusion Opportunity |
|-------|-------------|-------------------|
| 7 | Baseline indicators | Moderate |
| 14 | 2× baseline | Good (more EMAs) |
| 21 | 3× baseline | Better |
| 28 | 4× baseline | Best |

### 4. Data Size Scaling

| Size | Points | Expected Behavior |
|------|--------|-------------------|
| 1K | 1,000 | Overhead dominates |
| 10K | 10,000 | Balanced compute/overhead |
| 100K | 100,000 | Compute dominates |
| 1M | 1,000,000 | Memory bandwidth limited |

## Benchmark Configuration

### Standard Indicator Parameters

| Indicator | Parameters |
|-----------|------------|
| SMA | period=20 |
| EMA | period=20 |
| RSI | period=14 |
| MACD | fast=12, slow=26, signal=9 |
| ATR | period=14 |
| Bollinger | period=20, stddev=2.0 |
| Stochastic | k_period=14, d_period=3 |

### Test Data

- **Generation**: ChaCha8Rng with seed=42 for reproducibility
- **Type**: OHLCV (synthetic random walk with realistic constraints)
- **Sizes**: 1K, 10K, 100K, 1M data points

### Criterion Configuration

- **Warm-up**: 2 seconds
- **Measurement**: 5-10 seconds (size-dependent)
- **Sample size**: 20-100 (size-dependent)

## Results

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e07_end_to_end`*

### Baseline 7 Indicators (10K data points)

| Mode | Time | vs Direct | vs TA-Lib |
|------|------|-----------|-----------|
| Direct (inline) | TBD μs | baseline | TBD |
| Direct (executor) | TBD μs | TBD | TBD |
| Plan (executor) | TBD μs | **TBD** | TBD |
| Plan (helper) | TBD μs | TBD | TBD |

### Direct vs Plan Comparison

| Data Size | Direct Time | Plan Time | Speedup | Decision |
|-----------|-------------|-----------|---------|----------|
| 1K | TBD μs | TBD μs | TBD× | TBD |
| 10K | TBD μs | TBD μs | TBD× | TBD |
| 100K | TBD μs | TBD μs | TBD× | TBD |
| 1M | TBD ms | TBD ms | TBD× | TBD |

### Workload Scaling (10K data points)

| Indicator Count | Direct | Plan | Speedup | Meets Target |
|-----------------|--------|------|---------|--------------|
| 7 | TBD μs | TBD μs | TBD× | TBD |
| 14 | TBD μs | TBD μs | TBD× | TBD |
| 21 | TBD μs | TBD μs | TBD× | TBD |
| 28 | TBD μs | TBD μs | TBD× | TBD |

### EMA Fusion Benefit

| Configuration | Separate EMAs | Fused EMAs | Speedup |
|--------------|---------------|------------|---------|
| 7 EMAs (10K) | TBD μs | TBD μs | TBD× |

### Realistic Workload (Daily Trading Bars)

| Scenario | Direct | Plan | Speedup |
|----------|--------|------|---------|
| 1 year (252 bars) | TBD μs | TBD μs | TBD× |
| 10 years (2520 bars) | TBD μs | TBD μs | TBD× |
| 100 years (25200 bars) | TBD μs | TBD μs | TBD× |

### Plan Reuse (Amortization)

| Execution Pattern | Direct | Plan | Speedup |
|-------------------|--------|------|---------|
| Single execution | TBD μs | TBD μs | TBD× |
| Batch of 5 executions | TBD μs | TBD μs | TBD× |

### Throughput Analysis

| Mode | 10K (elem/s) | 100K (elem/s) | 1M (elem/s) |
|------|--------------|---------------|-------------|
| Direct | TBD M/s | TBD M/s | TBD M/s |
| Plan | TBD M/s | TBD M/s | TBD M/s |

### Individual Indicator Performance (10K)

| Indicator | Direct Time | Notes |
|-----------|-------------|-------|
| SMA | TBD μs | Baseline |
| EMA | TBD μs | Similar to SMA |
| RSI | TBD μs | Uses Wilder smoothing |
| MACD | TBD μs | 3 EMA operations |
| ATR | TBD μs | OHLCV required |
| Bollinger | TBD μs | Mean + stddev |
| Stochastic | TBD μs | Max + min + SMA |

## Analysis

### Theoretical Speedup Calculation

Based on data pass reduction:

```
Direct mode: ~13 passes
Plan mode:   ~8 passes

Theoretical max speedup = 13 / 8 = 1.625×
```

However, actual speedup will be lower due to:
- Compute overhead in fused kernels
- Memory allocation patterns
- Cache effects

**Expected realistic speedup**: 1.2× - 1.4× for baseline 7 indicators

### Fusion Contribution Analysis

| Fusion Type | Direct Passes | Fused Passes | Savings |
|-------------|---------------|--------------|---------|
| EMA multi | N × 1 | ~1 | (N-1)/N |
| MACD | 3 | 1 | 2/3 |
| Bollinger (running_stat) | 2 | 1 | 1/2 |
| Stochastic (extrema) | 2 | 1 | 1/2 |

### Break-Even Analysis

From E05 results:

```
Plan compilation overhead: ~TBD μs
Per-indicator execution savings: ~TBD μs
Break-even point: TBD indicators
```

### Memory Access Pattern Impact

From E06 results:

```
Multi-output write pattern: TBD (sequential vs interleaved)
Allocation overhead: TBD%
Buffer reuse benefit: TBD%
```

## Go/No-Go Decision

**Decision**: PENDING

### Decision Criteria

#### GO (Plan Architecture Recommended):

- [ ] Plan mode achieves ≥1.2× speedup for baseline 7 indicators at 10K+ data
- [ ] Plan mode achieves ≥1.5× speedup for 20+ indicators
- [ ] Speedup increases with indicator count (good scaling)
- [ ] Speedup is consistent across data sizes
- [ ] No regression at small data sizes

#### NO-GO (Direct Mode Preferred):

- [ ] Plan mode shows <1.1× speedup for baseline indicators
- [ ] Plan mode is slower for any common configuration
- [ ] Overhead doesn't amortize within reasonable workloads
- [ ] Complex fusion logic outweighs benefits

### Preliminary Assessment

*To be filled after benchmark execution*

**Observed Speedups**:
- 7 indicators: TBD×
- 14 indicators: TBD×
- 21 indicators: TBD×

**Recommendation**: TBD

## Implications for fast-ta Architecture

### If GO Decision:

1. **Default to plan mode** for multi-indicator workloads
2. **Expose both APIs**:
   ```rust
   // Direct mode (simple use case)
   let sma = fast_ta::sma(&prices, 20)?;

   // Plan mode (performance use case)
   let results = fast_ta::compute_many(&prices, &[
       Indicator::sma(20),
       Indicator::ema(20),
       Indicator::rsi(14),
   ])?;
   ```
3. **Document performance characteristics** in API docs
4. **Implement plan caching** for repeated execution

### If NO-GO Decision:

1. **Remove plan infrastructure** from core library
2. **Keep fusion kernels** as internal optimizations
3. **Simplify public API** to direct indicator calls
4. **Move plan code** to experimental/optional module

### If CONDITIONAL GO Decision:

1. **Threshold-based selection**: Use plan mode only for N+ indicators
2. **Automatic mode selection** based on workload analysis
3. **Expose both modes** with clear documentation
4. **Default to direct mode** for simplicity

## Comparison with Previous Experiments

### E01-E04: Kernel Experiments

| Experiment | Finding | Impact on E07 |
|------------|---------|---------------|
| E01 Baseline | Raw indicator costs | Baseline reference |
| E02 RunningStat | ~20% speedup for mean+stddev | Bollinger improvement |
| E03 EMA Fusion | ~15% speedup for multi-EMA | MACD improvement |
| E04 Rolling Extrema | ~5× speedup vs naive | Stochastic improvement |

### E05-E06: Infrastructure Experiments

| Experiment | Finding | Impact on E07 |
|------------|---------|---------------|
| E05 Plan Overhead | Compilation cost: TBD μs | Break-even calculation |
| E06 Memory Writes | Write pattern: TBD | Output strategy |

## Follow-up Actions

### If GO:

1. Update PRD with confirmed architecture
2. Implement production-ready PlanCache
3. Add comprehensive API documentation
4. Create usage examples and tutorials

### If NO-GO:

1. Update PRD to reflect simpler architecture
2. Archive plan infrastructure code
3. Focus on per-indicator optimizations
4. Simplify public API surface

### Regardless of Decision:

1. Document experimental findings
2. Update performance claims in README
3. Add benchmark results to documentation
4. Create performance tuning guide

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e07_end_to_end.rs`
- **Direct Mode**: `crates/fast-ta-core/src/plan/direct_mode.rs`
- **Plan Mode**: `crates/fast-ta-core/src/plan/plan_mode.rs`
- **Fusion Kernels**: `crates/fast-ta-core/src/kernels/`
- **Criterion Output**: `target/criterion/e07_end_to_end/`

## Reproduction

To run this experiment:

```bash
# Run E07 end-to-end benchmarks
cargo bench --package fast-ta-experiments --bench e07_end_to_end

# View HTML report
open target/criterion/e07_end_to_end/report/index.html

# Run specific benchmark groups
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "direct_baseline"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "plan_baseline"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "direct_vs_plan"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "workload_comparison"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "go_no_go"

# Quick benchmark (subset of tests)
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "go_no_go"
```

## Technical Notes

### Benchmark Structure

```
e07_end_to_end/
├── direct_baseline/           # Direct mode with 7 indicators
│   ├── inline_7_indicators    # Hand-written indicator calls
│   ├── executor_7_indicators  # DirectExecutor
│   └── helper_7_indicators    # compute_all_direct helper
├── plan_baseline/             # Plan mode with 7 indicators
│   ├── executor_7_indicators  # PlanExecutor
│   └── helper_7_indicators    # compute_all_plan helper
├── direct_vs_plan/            # Head-to-head comparison
│   ├── direct/                # Per data size
│   └── plan/                  # Per data size
├── scaling_direct/            # Workload scaling (direct)
├── scaling_plan/              # Workload scaling (plan)
├── workload_comparison/       # Direct comparison by indicator count
├── data_scaling_direct/       # Data size scaling
├── data_scaling_plan/         # Data size scaling
├── individual_indicators/     # Per-indicator benchmarks
├── throughput/                # Elements per second
├── plan_reuse/                # Amortization tests
├── ema_fusion/                # EMA fusion benefit
├── realistic_workload/        # Trading scenarios
└── go_no_go/                  # Critical decision benchmarks
```

### Key Metrics

1. **Speedup**: `direct_time / plan_time`
2. **Throughput**: `data_points / time`
3. **Efficiency**: `speedup / theoretical_max`
4. **Amortization**: `(overhead + N × plan_exec) / (N × direct_exec)`

### Statistical Considerations

- Each benchmark runs with appropriate warm-up (2s)
- Sample sizes adjusted for variance
- Criterion provides confidence intervals
- Multiple seeds tested for reproducibility

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
