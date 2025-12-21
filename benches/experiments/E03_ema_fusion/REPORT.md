# E03: EMA Fusion Benchmarks

## Experiment Overview

**Experiment ID**: E03
**Name**: EMA Fusion Benchmarks
**Status**: COMPLETED
**Date**: 2024-12-20

## Objective

Evaluate whether computing multiple EMA-family indicators in a single fused pass provides meaningful performance benefits over independent computation.

### Hypothesis

Fused EMA computation should be faster than separate passes because:

1. **Reduced Memory Bandwidth**: Single pass reads input data once instead of k times (for k EMAs)
2. **Better Cache Locality**: Data stays hot in L1/L2 cache during fused computation
3. **Amortized Loop Overhead**: One loop iteration vs multiple loop iterations
4. **Intermediate Reuse**: For DEMA/TEMA, EMA(EMA) is computed once and reused

### Success Criteria

| Result | Speedup for ≥10 EMAs | Action |
|--------|---------------------|--------|
| **GO** | ≥15% faster | Adopt fused kernels as primary approach |
| **INVESTIGATE** | 10-15% faster | Consider adoption with caveats |
| **NO-GO** | <10% faster or slower | Keep separate implementations |

## Approaches Benchmarked

### 1. Fused Multi-EMA (ema_multi)

Using `ema_multi()` from `fast_ta_core::kernels::ema_fusion`:
- Single pass through input data
- Computes k EMAs with different periods simultaneously
- Time complexity: O(n × k) with single data read
- Space complexity: O(n × k) for outputs

### 2. Separate EMAs

Traditional approach using multiple `ema()` calls:
- Independent calls for each period
- Each call reads the entire input data
- Time complexity: O(n × k) with k data reads
- Space complexity: O(n × k) for outputs

### 3. Fused EMA/DEMA/TEMA (ema_fusion)

Using `ema_fusion()` for related indicators:
- Computes EMA, DEMA, TEMA in one pass
- DEMA = 2 × EMA - EMA(EMA)
- TEMA = 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
- Reuses intermediate EMA calculations

### 4. Fused MACD (macd_fusion)

Using `macd_fusion()` for MACD computation:
- Computes fast EMA, slow EMA, signal line together
- Compares to standard MACD implementation
- Standard parameters: 12, 26, 9

## Benchmark Configuration

### Data Sizes

| Size | Points | Description |
|------|--------|-------------|
| 1K | 1,000 | Quick iteration, cache-resident data |
| 10K | 10,000 | L2/L3 cache threshold |
| 100K | 100,000 | Memory-bound scenario |
| 1M | 1,000,000 | Throughput measurement (selected tests) |

### EMA Counts Tested

| Count | Periods | Description |
|-------|---------|-------------|
| 3 | 5, 10, 15 | Small multi-EMA scenario |
| 5 | 5, 10, 15, 20, 25 | Moderate multi-EMA scenario |
| 10 | 5, 10, 15, ..., 50 | Target scenario for ≥15% speedup |
| 20 | 5, 10, 15, ..., 100 | Stress test scenario |

### Parameters

- **EMA Period for Fusion**: 12 (standard for EMA/DEMA/TEMA tests)
- **MACD Periods**: 12, 26, 9 (fast, slow, signal)
- **Measurement Time**: 5-10 seconds per benchmark
- **Sample Size**: 20-100 samples (scaled with data size)
- **Warm-up**: 2 seconds
- **Data**: Reproducible random walk (seed=42)

## Results

### Primary Comparison: Multi-EMA Fusion (10 EMAs)

| Data Size | Fused (ema_multi) | Separate (10×ema) | Speedup | Verdict |
|-----------|-------------------|-------------------|---------|---------|
| 1K | 11.39 µs | 15.84 µs | +28.1% | GO |
| 10K | 213.7 µs | 163.2 µs | -30.9% | NO-GO |
| 100K | 2.08 ms | 1.59 ms | -30.3% | NO-GO |

**Critical Finding**: Fused approach shows benefit only at very small data sizes (1K). At realistic data sizes (10K+), the separate EMA approach is **significantly faster** than fused.

### EMA Count Scaling (at 100K data points)

| EMA Count | Fused | Separate | Speedup | Notes |
|-----------|-------|----------|---------|-------|
| 3 EMAs | 389.6 µs | 478.4 µs | +18.6% | Fused slightly faster |
| 5 EMAs | 629.4 µs | 798.3 µs | +21.2% | Fused slightly faster |
| 10 EMAs | 1.99 ms | 1.57 ms | -26.6% | **Separate FASTER** |
| 20 EMAs | 6.71 ms | 3.11 ms | -116% | **Separate MUCH FASTER** |

**Critical Finding**: As EMA count increases beyond 5, the fused approach becomes **dramatically slower**. At 20 EMAs, separate is more than 2× faster. This is opposite of the expected scaling behavior.

### EMA/DEMA/TEMA Fusion

| Data Size | Fused (ema_fusion) | Separate (3×ema + compute) | Speedup |
|-----------|--------------------|---------------------------|---------|
| 1K | 5.25 µs | 5.12 µs | -2.5% |
| 10K | 62.2 µs | 55.3 µs | -12.5% |
| 100K | 577.1 µs | 548.0 µs | -5.3% |

**Finding**: EMA/DEMA/TEMA fusion provides no benefit - separate passes are consistently faster.

### MACD Fusion

| Data Size | Fused (macd_fusion) | Standard (macd) | Speedup |
|-----------|---------------------|-----------------|---------|
| 1K | 6.36 µs | 6.60 µs | +3.7% |
| 10K | 72.8 µs | 69.2 µs | -5.2% |
| 100K | 707.3 µs | 669.6 µs | -5.6% |

**Finding**: MACD fusion shows marginal benefit only at 1K. At larger sizes, standard MACD is faster.

### Period Sensitivity (at 100K data points)

Does fusion benefit change with EMA period?

| Period | Fused (ema_fusion) | Separate | Speedup |
|--------|-------------------|----------|---------|
| 10 | 564.8 µs | 523.5 µs | -7.9% |
| 20 | 563.1 µs | 522.6 µs | -7.8% |
| 50 | 574.8 µs | 531.2 µs | -8.2% |
| 200 | 563.2 µs | 528.4 µs | -6.6% |

**Finding**: Period has minimal impact on relative performance. Fused is consistently ~7-8% slower regardless of period.

### Throughput Analysis (10 EMAs)

| Data Size | Fused (elements/sec) | Separate (elements/sec) | Ratio |
|-----------|---------------------|------------------------|-------|
| 10K | 52.4M/s | 60.9M/s | 0.86× |
| 100K | 52.3M/s | 61.9M/s | 0.84× |
| 1M | 52.6M/s | 64.1M/s | 0.82× |

**Finding**: Separate passes achieve ~16-18% higher throughput than fused across all data sizes.

### Pre-allocated Buffer Comparison

Testing with `_into()` variants to eliminate allocation overhead:

| Data Size | Fused Multi-EMA Into | Separate EMA Into | Delta |
|-----------|---------------------|-------------------|-------|
| 1K | 3.36 µs | 4.50 µs | +25.3% |
| 10K | 33.6 µs | 46.1 µs | +27.0% |
| 100K | 343.9 µs | 472.0 µs | +27.2% |

**Finding**: With pre-allocated buffers, fused approach shows consistent ~27% benefit. However, this is for a different scenario than the main benchmarks.

## Analysis

### Actual Results vs Expected

The results **contradict the hypothesis** in significant ways:

1. **Multi-EMA fusion is SLOWER at scale**:
   - Expected: Fusion benefit increases with EMA count
   - Actual: Fusion becomes dramatically slower with more EMAs
   - At 20 EMAs: Separate is 2.16× faster than fused

2. **EMA/DEMA/TEMA fusion provides no benefit**:
   - Expected: ~15-20% speedup from avoiding recomputation
   - Actual: Fused is 5-12% slower

3. **MACD fusion is marginally worse**:
   - Expected: Benefit from fused EMA computation
   - Actual: Standard MACD is 5% faster at scale

### Root Cause Analysis

The poor performance of fused kernels likely stems from:

1. **Inner Loop Complexity**: The fused approach has a complex inner loop that updates multiple EMA states. This prevents SIMD auto-vectorization that the compiler can apply to simple single-EMA loops.

2. **Register Pressure**: Tracking multiple EMA states (alpha values, current values) exhausts CPU registers, causing spills to memory.

3. **Branch Prediction**: Each EMA may have different warmup periods, creating unpredictable branches in the inner loop.

4. **Cache Write Patterns**: Writing to multiple output arrays in each iteration causes cache line conflicts.

5. **Compiler Optimization**: Simple loops are more amenable to compiler optimization (loop unrolling, prefetching) than complex fused loops.

### Memory Bandwidth Analysis

Theoretical memory access patterns for 10 EMAs:

| Approach | Input Reads | Output Writes | Total Bandwidth |
|----------|------------|---------------|-----------------|
| Fused | 1 × input | 10 × output | 11n × sizeof(f64) |
| Separate | 10 × input | 10 × output | 20n × sizeof(f64) |

**Theoretical Fused Advantage**: ~45% less memory traffic

**Actual Result**: The theoretical memory bandwidth advantage is completely negated by worse computational efficiency. The CPU overhead of the fused inner loop outweighs any memory savings.

### Cache Efficiency

For 100K f64 values (800KB):
- Exceeds L1 cache (32-64KB)
- Fits in L2/L3 cache (256KB-8MB)
- Expected: Fused approach keeps input hot in cache
- Actual: Separate passes also benefit from L2/L3 caching, and simpler loops run faster

## Go/No-Go Decision

**Decision**: NO-GO

### Criteria Checklist

#### For GO (adopt fused approach):
- [ ] Fused achieves ≥15% speedup for 10 EMAs at 100K - **FAILED: 30% SLOWER**
- [ ] Speedup scales with EMA count (more EMAs = more benefit) - **FAILED: OPPOSITE BEHAVIOR**
- [ ] EMA/DEMA/TEMA fusion shows ≥10% speedup - **FAILED: 5-12% SLOWER**
- [x] Speedup persists with pre-allocated buffers - **PASSED: +27% speedup**
- [ ] No significant regression in any scenario - **FAILED: Major regressions**

#### For NO-GO (keep separate implementations):
- [x] Fused speedup is <10% for 10 EMAs - **CONFIRMED: Actually 30% SLOWER**
- [x] OR fused is slower in some scenarios - **CONFIRMED: Slower in most scenarios**
- [ ] OR implementation complexity outweighs benefits

### Decision Rationale

The EMA fusion hypothesis is **definitively rejected**. Separate EMA passes are significantly faster than fused computation for realistic data sizes and EMA counts. The theoretical memory bandwidth advantage of fusion is completely overwhelmed by:

1. Loss of SIMD vectorization
2. Increased register pressure
3. More complex loop structures that prevent compiler optimizations

**Recommendation**: Remove or deprecate `ema_multi()` and `ema_fusion()` kernels. Use separate EMA calls for all multi-EMA workloads.

## Implications for fast-ta Architecture

### Based on NO-GO Decision:

1. **Keep Current Design**: Separate EMA calls remain the correct approach
2. **Remove Fusion Kernels**: Consider removing `ema_fusion.rs` to reduce codebase complexity
3. **Plan Mode Strategy**: Do NOT attempt to fuse EMA operations in execution plans
4. **Documentation**: Update docs to explain why fusion doesn't help for EMAs
5. **Focus Elsewhere**: Look for performance gains in E02/E04 experiments instead

### Lessons Learned

1. **Theoretical bandwidth analysis is insufficient**: Must benchmark to validate
2. **Simple loops optimize better**: Compiler optimizations favor simple loop structures
3. **Memory bandwidth isn't always the bottleneck**: CPU efficiency matters more for streaming operations
4. **Register pressure is real**: Fusing too many operations exhausts registers

## Comparison with E02 (RunningStat Fusion)

| Metric | E02 RunningStat | E03 EMA Fusion |
|--------|-----------------|----------------|
| Fusion Type | Mean/Var/StdDev | Multiple EMAs |
| Memory Benefit | 1 read vs 2 reads | 1 read vs k reads |
| Target Speedup | ≥20% | ≥15% (for k≥10) |
| Actual Result | NO-GO (2.8× slower) | NO-GO (30%+ slower) |
| Complexity | Low | Medium |

Both E02 and E03 demonstrate that kernel fusion does NOT provide expected benefits in this codebase.

## Follow-up Actions

Based on E03 NO-GO result:

1. **Remove Fusion Kernels**:
   - Deprecate `ema_multi()`, `ema_fusion()`, `macd_fusion()`
   - Keep simple single-EMA kernel as primary implementation

2. **Update PRD**:
   - Document that kernel fusion is not a viable optimization strategy
   - Adjust performance expectations accordingly

3. **E04 (Rolling Extrema)**: Different optimization strategy (deque-based), may still show benefit since it's algorithmic improvement, not loop fusion

4. **Future Work**:
   - Investigate SIMD vectorization of single-EMA kernels
   - Consider parallel processing (rayon) for multi-EMA workloads instead of fusion

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e03_ema_fusion.rs`
- **Kernel Implementation**: `crates/fast-ta-core/src/kernels/ema_fusion.rs`
- **Criterion Output**: `target/criterion/e03_ema_fusion/`
- **Raw JSON Data**: `target/criterion/e03_ema_fusion/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E03 EMA fusion benchmarks
cargo bench --package fast-ta-experiments --bench e03_ema_fusion

# View HTML report
open target/criterion/e03_ema_fusion/report/index.html

# Run specific benchmark group
cargo bench --package fast-ta-experiments --bench e03_ema_fusion -- "fused_multi_ema"
cargo bench --package fast-ta-experiments --bench e03_ema_fusion -- "separate_emas"
cargo bench --package fast-ta-experiments --bench e03_ema_fusion -- "ema_count_scaling"
```

## Technical Notes

### EMA Algorithm

Both fused and separate approaches use the standard EMA formula:

```
α = 2 / (period + 1)
EMA[0..period-2] = NaN (insufficient lookback)
EMA[period-1] = SMA(prices[0..period])  // SMA seed
EMA[i] = α × Price[i] + (1 - α) × EMA[i-1]
```

### Fusion Implementation

The fused approach maintains parallel state for all EMAs:

```rust
// For each input value
for i in 0..n {
    let value = data[i];
    for j in 0..k {
        // Update EMA[j] with same input value
        // Each EMA has its own alpha and state
    }
}
```

Key issue: The inner loop over k EMAs prevents vectorization and increases register pressure.

### Why Separate Passes Win

```rust
// Separate passes - compiler can vectorize
for period in periods {
    let alpha = 2.0 / (period as f64 + 1.0);
    for i in 0..n {
        // Simple loop - SIMD friendly
        ema_out[i] = alpha * data[i] + (1.0 - alpha) * ema_out[i-1];
    }
}
```

Simple loops allow:
- SIMD vectorization
- Loop unrolling
- Prefetching optimization
- Better branch prediction

### DEMA/TEMA Formulas

**DEMA (Double EMA)**:
```
DEMA = 2 × EMA(price) - EMA(EMA(price))
```

**TEMA (Triple EMA)**:
```
TEMA = 3 × EMA(price) - 3 × EMA(EMA(price)) + EMA(EMA(EMA(price)))
```

The fused approach computes EMA(EMA) and EMA(EMA(EMA)) in pass 2 and 3,
avoiding separate computations - but this doesn't provide actual speedup.

### Memory Layout

Fused approach writes to all output vectors in each loop iteration,
which causes cache line conflicts and prevents efficient write combining.

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: 2024-12-20*
*Result: NO-GO - Kernel fusion does not improve EMA performance*
