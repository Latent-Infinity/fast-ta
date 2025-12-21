# fast-ta Micro-Experiments Summary

## Overview

This document consolidates the results from all 7 micro-experiments (E01-E07) conducted to validate performance hypotheses for the fast-ta technical analysis library. Each experiment provides go/no-go decisions based on measured performance data.

**Project**: fast-ta
**Experiment Framework**: Criterion.rs v0.5.1
**Date**: December 2024
**Status**: Implementation Complete, Benchmarks Pending Execution

---

## Executive Summary

| Experiment | Category | Hypothesis | Target Speedup | Status | Decision |
|------------|----------|------------|----------------|--------|----------|
| **E01** | Baseline | Establish indicator costs | N/A (baseline) | PENDING | N/A |
| **E02** | Fusion | RunningStat fusion faster | ≥20% speedup | PENDING | PENDING |
| **E03** | Fusion | EMA fusion faster | ≥15% (≥10 EMAs) | PENDING | PENDING |
| **E04** | Algorithm | Deque-based extrema | ≥5× speedup (k≥50) | PENDING | PENDING |
| **E05** | Infrastructure | Plan overhead acceptable | <100 executions break-even | PENDING | PENDING |
| **E06** | Memory | Write pattern optimization | ≥10% improvement | PENDING | PENDING |
| **E07** | End-to-End | Plan mode faster than direct | ≥1.5× (≥20 indicators) | PENDING | PENDING |

**Overall Recommendation**: Pending benchmark execution

---

## E01: Baseline Cost Benchmarks

### Experiment Details

- **ID**: E01
- **Category**: Foundation
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E01_baseline/REPORT.md`](../../benches/experiments/E01_baseline/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e01_baseline`

### Objective

Establish performance baselines for all 7 core technical indicators. These baselines serve as reference points for measuring fusion kernel improvements (E02-E04) and validating O(n) time complexity claims.

### Indicators Benchmarked

| Indicator | Period(s) | Algorithm | Expected Complexity |
|-----------|-----------|-----------|---------------------|
| SMA | 20 | Rolling sum | O(n) |
| EMA | 20 | Recursive formula | O(n) |
| RSI | 14 | Wilder smoothing | O(n) |
| MACD | 12, 26, 9 | Triple EMA | O(n) |
| ATR | 14 | Wilder smoothing | O(n) |
| Bollinger Bands | 20, 2.0 std | Rolling sum + sum-of-squares | O(n) |
| Stochastic | 14, 3 | Rolling extrema | O(n*k) naive / O(n) with deque |

### Expected Results

| Indicator | 1K (ns/op) | 10K (ns/op) | 100K (ns/op) | ns/element @ 100K |
|-----------|------------|-------------|--------------|-------------------|
| SMA | TBD | TBD | TBD | TBD |
| EMA | TBD | TBD | TBD | TBD |
| RSI | TBD | TBD | TBD | TBD |
| MACD | TBD | TBD | TBD | TBD |
| ATR | TBD | TBD | TBD | TBD |
| Bollinger | TBD | TBD | TBD | TBD |
| Stochastic | TBD | TBD | TBD | TBD |

### Go/No-Go Decision

- **Decision**: N/A (baseline experiment)
- **Criteria**: All indicators demonstrate O(n) or O(n log n) complexity

---

## E02: RunningStat Fusion Benchmarks

### Experiment Details

- **ID**: E02
- **Category**: Kernel Fusion
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E02_running_stat/REPORT.md`](../../benches/experiments/E02_running_stat/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e02_running_stat`

### Objective

Evaluate whether computing rolling mean, variance, and standard deviation in a single fused pass using Welford's algorithm provides meaningful performance benefits over separate computation passes.

### Hypothesis

Fused computation should be faster because:
1. **Reduced Memory Bandwidth**: Single pass reads input data once instead of twice
2. **Better Cache Locality**: Data stays hot in L1/L2 cache during fused computation
3. **Amortized Loop Overhead**: One loop iteration vs multiple loop iterations
4. **Numerical Stability**: Welford's algorithm bonus - more accurate with extreme values

### Approaches Compared

| Approach | Description | Complexity |
|----------|-------------|------------|
| **Fused (Welford)** | `rolling_stats()` - single pass | O(n), 1 read |
| **Separate Passes** | `sma()` + `rolling_stddev()` | O(n), 2 reads |
| **Bollinger Reference** | Sum + sum-of-squares | O(n), 1 read |

### Expected Results

| Data Size | Fused (Welford) | Separate Passes | Speedup |
|-----------|-----------------|-----------------|---------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

**Theoretical Memory Advantage**: 20% less memory traffic (4n vs 5n reads/writes)

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: ≥20% speedup over separate passes at 100K data points

| Result | Speedup | Action |
|--------|---------|--------|
| **GO** | ≥20% faster | Adopt fused kernels as primary approach |
| **INVESTIGATE** | 10-20% faster | Consider adoption with caveats |
| **NO-GO** | <10% faster | Keep separate implementations |

---

## E03: EMA Fusion Benchmarks

### Experiment Details

- **ID**: E03
- **Category**: Kernel Fusion
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E03_ema_fusion/REPORT.md`](../../benches/experiments/E03_ema_fusion/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e03_ema_fusion`

### Objective

Evaluate whether computing multiple EMA-family indicators in a single fused pass provides meaningful performance benefits over independent computation.

### Hypothesis

Fused EMA computation should be faster because:
1. **Reduced Memory Bandwidth**: Single pass reads input data once instead of k times
2. **Better Cache Locality**: Data stays hot in L1/L2 cache during fused computation
3. **Intermediate Reuse**: For DEMA/TEMA, EMA(EMA) is computed once and reused

### Approaches Compared

| Approach | Description | Complexity |
|----------|-------------|------------|
| **Fused Multi-EMA** | `ema_multi()` - k EMAs in one pass | O(n×k), 1 read |
| **Separate EMAs** | k × `ema()` calls | O(n×k), k reads |
| **Fused EMA/DEMA/TEMA** | `ema_fusion()` - related indicators | O(n), 3 outputs |
| **Fused MACD** | `macd_fusion()` - 12,26,9 EMAs | O(n), 3 outputs |

### Expected Results (10 EMAs)

| Data Size | Fused (ema_multi) | Separate (10×ema) | Speedup |
|-----------|-------------------|-------------------|---------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

### EMA Count Scaling

| EMA Count | Expected Speedup | Notes |
|-----------|-----------------|-------|
| 3 EMAs | ~10% | Minimal benefit expected |
| 5 EMAs | ~15% | Approaching threshold |
| 10 EMAs | ~20%+ | **Target scenario** |
| 20 EMAs | ~25%+ | Maximum benefit |

**Theoretical Memory Advantage**: ~45% less memory traffic for 10 EMAs

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: ≥15% speedup for ≥10 EMAs at 100K data points

| Result | Speedup (≥10 EMAs) | Action |
|--------|-------------------|--------|
| **GO** | ≥15% faster | Adopt fused kernels as primary approach |
| **INVESTIGATE** | 10-15% faster | Consider adoption with caveats |
| **NO-GO** | <10% faster | Keep separate implementations |

---

## E04: Rolling Extrema Benchmarks

### Experiment Details

- **ID**: E04
- **Category**: Algorithm Optimization
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E04_rolling_extrema/REPORT.md`](../../benches/experiments/E04_rolling_extrema/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e04_rolling_extrema`

### Objective

Evaluate whether using a monotonic deque algorithm for rolling max/min provides significant performance improvements over the naive scan approach.

### Hypothesis

The deque-based algorithm should be dramatically faster because:
1. **Amortized O(1) per element**: Each element enters and exits the deque at most once
2. **O(n) vs O(n×k) complexity**: Naive scans the entire window for each output
3. **Speedup scales with period**: Larger periods amplify the difference

### Algorithm Comparison

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| **Deque-based** | O(n) amortized | O(n) + O(k) deque |
| **Naive Scan** | O(n × k) | O(n) |

### Theoretical Speedup by Period

| Period (k) | Naive Operations | Deque Operations | Theoretical Speedup |
|------------|------------------|------------------|---------------------|
| 5 | 5n | 2n | 2.5× |
| 14 | 14n | 2n | 7× |
| 50 | 50n | 2n | 25× |
| 100 | 100n | 2n | 50× |
| 200 | 200n | 2n | 100× |

### Expected Results (Period 14, 100K data)

| Data Size | Deque (rolling_max) | Naive (rolling_max_naive) | Speedup |
|-----------|---------------------|---------------------------|---------|
| 1K | TBD | TBD | TBD× |
| 10K | TBD | TBD | TBD× |
| 100K | TBD | TBD | TBD× |

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: ≥5× speedup at period ≥50 with 100K data

| Result | Speedup (k≥50) | Action |
|--------|---------------|--------|
| **GO** | ≥5× faster | Adopt deque-based algorithm as standard |
| **INVESTIGATE** | 2-5× faster | Consider adoption, investigate edge cases |
| **NO-GO** | <2× faster | Keep naive for simplicity |

**Impact**: This is the most significant optimization opportunity - replacing O(n×k) with O(n).

---

## E05: Plan Compilation Overhead Benchmarks

### Experiment Details

- **ID**: E05
- **Category**: Infrastructure
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E05_plan_overhead/REPORT.md`](../../benches/experiments/E05_plan_overhead/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e05_plan_overhead`

### Objective

Measure the cost of plan infrastructure (registry, DAG construction, topological sort) and calculate the break-even point where plan mode becomes advantageous over direct indicator computation.

### Plan Infrastructure Components

1. **Registry Population**: Registering indicator specifications
2. **DAG Construction**: Building dependency graph with petgraph
3. **Topological Sort**: Computing valid execution order
4. **Query Operations**: Looking up indicators by ID, config, or kind

### Break-Even Calculation

```
N × T_direct = T_compilation + N × T_plan_exec

Where:
- N = number of executions
- T_direct = time for direct indicator computation
- T_compilation = one-time plan compilation cost
- T_plan_exec = time for plan-based execution
```

### Expected Results

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Register 1 indicator | ~50-100 ns | HashMap insert |
| Register 10 indicators | ~500 ns - 1 μs | Linear scaling |
| Build DAG (10 nodes) | ~200-500 ns | Graph construction |
| Topological sort (10 nodes) | ~100-300 ns | Linear algorithm |
| Full compilation (10 indicators) | ~1-5 μs | All steps combined |
| Plan access | ~10-20 ns | Reference return |

### Expected Break-Even Points

| Fusion Speedup (S) | Expected Break-Even (N) |
|-------------------|------------------------|
| 1.0× (no fusion) | ∞ (never breaks even) |
| 1.1× | ~10 executions |
| 1.2× | ~5 executions |
| 1.5× | ~3 executions |
| 2.0× | ~2 executions |

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: <100 executions to break even

| Metric | Target | Status |
|--------|--------|--------|
| Full compilation (10 indicators) | <1ms | PENDING |
| Plan reuse overhead | ~0 ns | PENDING |
| Break-even (20% fusion) | <10 executions | PENDING |

---

## E06: Memory Write Pattern Benchmarks

### Experiment Details

- **ID**: E06
- **Category**: Memory Optimization
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E06_memory_writes/REPORT.md`](../../benches/experiments/E06_memory_writes/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e06_memory_writes`

### Objective

Determine the optimal memory write pattern for indicator computation by comparing:
1. **Write-every-bar**: Standard approach writing to output array on every iteration
2. **Buffered writes**: Accumulate values in registers/local buffers, write periodically
3. **Chunked processing**: Process data in cache-friendly blocks
4. **Multi-output patterns**: Interleaved vs sequential writes to multiple arrays

### Hypothesis

Write-every-bar may cause performance issues when:
1. Cache line ping-pong between multiple indicators
2. Memory bandwidth saturation at high write frequency
3. Store buffer pressure with pending writes

### Patterns Compared

| Pattern | Description | Expected Cache Behavior |
|---------|-------------|------------------------|
| Allocating | Fresh vector each call | Allocation overhead |
| Pre-allocated | Reuse existing buffer | Best for repeated calls |
| Buffered (64 elements) | L1 cache (512 bytes) | May help cache locality |
| Buffered (1024 elements) | L2 cache (8KB) | Larger block |
| Sequential multi-output | One indicator at a time | Full cache per indicator |
| Interleaved multi-output | All outputs each iteration | Split cache |

### Expected Results

| Size | Fits In | Expected Buffering Benefit |
|------|---------|---------------------------|
| 1K (8KB) | L1 cache | Minimal |
| 10K (80KB) | L2 cache | Possible |
| 100K (800KB) | L3 cache | Possible |
| 1M (8MB) | Main memory | More likely |

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: ≥10% speedup from optimized write patterns

| Decision | Condition | Recommendation |
|----------|-----------|----------------|
| **GO (Buffered)** | ≥10% speedup | Implement buffering in hot paths |
| **NO-GO (Direct)** | No significant improvement | Keep simple write-every-bar |
| **INVESTIGATE** | Mixed results by size | Use adaptive strategy |

---

## E07: End-to-End Workload Benchmarks

### Experiment Details

- **ID**: E07
- **Category**: End-to-End Validation
- **Status**: PENDING (awaiting benchmark execution)
- **Report**: [`benches/experiments/E07_end_to_end/REPORT.md`](../../benches/experiments/E07_end_to_end/REPORT.md)
- **Benchmark**: `cargo bench --package fast-ta-experiments --bench e07_end_to_end`

### Objective

Provide the final comprehensive comparison between:
1. **TA-Lib baseline** (via golden file reference timings)
2. **Direct mode** (independent indicator computation)
3. **Plan mode** (fused kernel execution with DAG optimization)

### Hypothesis

Plan mode with fused kernels should outperform direct mode when:
1. Computing 6+ indicators in a single workload
2. Multiple EMA-based indicators share data access
3. Bollinger Bands benefit from running_stat fusion
4. Stochastic benefits from rolling_extrema fusion
5. Plan compilation overhead is amortized

### Execution Mode Comparison

| Mode | Indicators | Estimated Passes | Fusion Strategy |
|------|------------|------------------|-----------------|
| **Direct** | 7 baseline | ~13 passes | None |
| **Plan** | 7 baseline | ~7-8 passes | EMA, MACD, Bollinger, Stochastic |

### Theoretical Speedup

```
Direct mode: ~13 passes over data
Plan mode:   ~8 passes over data

Theoretical max speedup = 13 / 8 = 1.625×
Expected realistic speedup = 1.2× - 1.4×
```

### Expected Results

| Indicator Count | Direct | Plan | Expected Speedup |
|-----------------|--------|------|-----------------|
| 7 (baseline) | TBD μs | TBD μs | 1.2× - 1.4× |
| 14 | TBD μs | TBD μs | 1.3× - 1.5× |
| 21 | TBD μs | TBD μs | 1.4× - 1.6× |
| 28 | TBD μs | TBD μs | 1.5×+ |

### Go/No-Go Decision

- **Decision**: PENDING
- **Target**: ≥1.5× speedup for ≥20 indicators

| Decision | Condition | Recommendation |
|----------|-----------|----------------|
| **GO (Plan architecture)** | ≥1.5× for ≥20 indicators | Plan-based API as default |
| **CONDITIONAL GO** | ≥1.2× for baseline 7 | Plan for multi-indicator only |
| **NO-GO** | <1.1× or slower | Prefer direct mode, simplify |

---

## Architecture Decision Summary

### Validated Patterns (Pending Confirmation)

| Pattern | Experiment | Expected Outcome |
|---------|------------|------------------|
| Welford's RunningStat | E02 | Numerically stable, potentially faster |
| Multi-EMA Fusion | E03 | Faster for ≥10 EMAs |
| Monotonic Deque | E04 | Dramatically faster for rolling extrema |
| DAG-based Planning | E05-E07 | Overhead acceptable for multi-indicator |

### Key Metrics to Track

| Metric | Source | Target |
|--------|--------|--------|
| Per-indicator cost | E01 | O(n) verified |
| RunningStat fusion benefit | E02 | ≥20% speedup |
| EMA fusion benefit | E03 | ≥15% for ≥10 EMAs |
| Rolling extrema speedup | E04 | ≥5× for k≥50 |
| Plan break-even | E05 | <100 executions |
| Write pattern benefit | E06 | Document findings |
| End-to-end speedup | E07 | ≥1.5× for ≥20 indicators |

---

## Running the Experiments

### Full Benchmark Suite

```bash
# Run all experiments
cargo bench --workspace

# View combined HTML report
open target/criterion/report/index.html
```

### Individual Experiments

```bash
# E01: Baseline costs
cargo bench --package fast-ta-experiments --bench e01_baseline

# E02: RunningStat fusion
cargo bench --package fast-ta-experiments --bench e02_running_stat

# E03: EMA fusion
cargo bench --package fast-ta-experiments --bench e03_ema_fusion

# E04: Rolling extrema
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema

# E05: Plan overhead
cargo bench --package fast-ta-experiments --bench e05_plan_overhead

# E06: Memory writes
cargo bench --package fast-ta-experiments --bench e06_memory_writes

# E07: End-to-end
cargo bench --package fast-ta-experiments --bench e07_end_to_end
```

### Quick Decision Benchmarks

```bash
# Run only the go/no-go decision benchmarks
cargo bench --package fast-ta-experiments --bench e02_running_stat -- "comparison"
cargo bench --package fast-ta-experiments --bench e03_ema_fusion -- "ema_count_scaling"
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema -- "period_scaling"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "go_no_go"
```

---

## Next Steps

1. **Execute Benchmarks**: Run `cargo bench --workspace` to generate actual performance data
2. **Populate Results**: Update this summary and individual REPORT.md files with actual timings
3. **Make Decisions**: Apply go/no-go criteria to determine architecture direction
4. **Update PRD**: Incorporate findings into PRD v1.5
5. **Document Recommendations**: Create performance tuning guide based on results

---

## Files Reference

| Category | Path |
|----------|------|
| Experiment Reports | `benches/experiments/E0[1-7]_*/REPORT.md` |
| Benchmark Source | `crates/fast-ta-experiments/benches/e0[1-7]_*.rs` |
| Core Indicators | `crates/fast-ta-core/src/indicators/` |
| Fusion Kernels | `crates/fast-ta-core/src/kernels/` |
| Plan Infrastructure | `crates/fast-ta-core/src/plan/` |
| Criterion Output | `target/criterion/` |
| Decision Documents | `docs/decisions/` |

---

*Summary generated for fast-ta micro-experiments framework*
*Last updated: December 2024*
*Version: 1.0 (Implementation Complete, Benchmarks Pending)*
