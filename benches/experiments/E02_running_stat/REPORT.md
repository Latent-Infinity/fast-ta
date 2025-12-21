# E02: RunningStat Fusion Benchmarks

## Experiment Overview

**Experiment ID**: E02
**Name**: RunningStat Fusion Benchmarks
**Status**: COMPLETED
**Date**: 2024-12-20

## Objective

Evaluate whether computing rolling mean, variance, and standard deviation in a single fused pass using Welford's algorithm provides meaningful performance benefits over separate computation passes.

### Hypothesis

Fused computation using Welford's algorithm should be faster than separate passes because:

1. **Reduced Memory Bandwidth**: Single pass reads input data once instead of twice
2. **Better Cache Locality**: Data stays hot in L1/L2 cache during fused computation
3. **Amortized Loop Overhead**: One loop iteration vs multiple loop iterations
4. **Numerical Stability**: Welford's algorithm bonus - more accurate with extreme values

### Success Criteria

| Result | Speedup vs Separate Passes | Action |
|--------|---------------------------|--------|
| **GO** | ≥20% faster | Adopt fused kernels as primary approach |
| **INVESTIGATE** | 10-20% faster | Consider adoption with caveats |
| **NO-GO** | <10% faster or slower | Keep separate implementations |

## Approaches Benchmarked

### 1. Fused (Welford's Algorithm)

Using `rolling_stats()` from `fast_ta_core::kernels::running_stat`:
- Single pass through input data
- Computes mean, variance, and stddev simultaneously
- Uses Welford's online algorithm for numerical stability
- Time complexity: O(n)
- Space complexity: O(n) for outputs

### 2. Separate Passes

Traditional approach using existing indicators:
- **Pass 1**: `sma()` for rolling mean
- **Pass 2**: `rolling_stddev()` for rolling standard deviation
- **Post-processing**: Compute variance as stddev²
- Time complexity: O(n) per pass = O(2n) total
- Space complexity: O(n) for each output

### 3. Bollinger Reference

Using Bollinger Bands computation:
- Rolling sum + sum-of-squares approach
- Computes middle band (SMA) and bands (using stddev)
- Represents "industry standard" implementation
- Time complexity: O(n)

## Benchmark Configuration

### Data Sizes

| Size | Points | Description |
|------|--------|-------------|
| 1K | 1,000 | Quick iteration, cache-resident data |
| 10K | 10,000 | L2/L3 cache threshold |
| 100K | 100,000 | Memory-bound scenario |
| 1M | 1,000,000 | Throughput measurement (selected tests) |

### Parameters

- **Rolling Period**: 20 (standard Bollinger period)
- **Measurement Time**: 5-10 seconds per benchmark
- **Sample Size**: 20-100 samples (scaled with data size)
- **Warm-up**: 2 seconds
- **Data**: Reproducible random walk (seed=42)

## Results

### Primary Comparison: Fused vs Separate

| Data Size | Fused (Welford) | Separate Passes | Speedup | Verdict |
|-----------|-----------------|-----------------|---------|---------|
| 1K | 10,036 ns | 3,534 ns | -184% (2.84× slower) | NO-GO |
| 10K | 108,849 ns | 38,460 ns | -183% (2.83× slower) | NO-GO |
| 100K | 1,026,786 ns | 372,516 ns | -176% (2.76× slower) | NO-GO |

**Key Finding**: The fused Welford approach is consistently **~2.8× SLOWER** than the separate SMA + StdDev approach across all data sizes.

### Component Breakdown

Understanding where time is spent in the separate approach:

| Component | 1K | 10K | 100K | % of Separate Total |
|-----------|-----|------|-------|---------------------|
| SMA (mean) | 1,371 ns | 14,316 ns | 134,812 ns | ~39% |
| rolling_stddev | 1,981 ns | 21,007 ns | 206,731 ns | ~60% |
| variance calc | ~182 ns | ~3,137 ns | ~31,973 ns | ~1% |
| **Total** | 3,534 ns | 38,460 ns | 372,516 ns | 100% |

**Note**: The existing SMA and rolling_stddev implementations are highly optimized with efficient sliding window algorithms that outperform the Welford online algorithm.

### Bollinger Comparison

| Data Size | Fused (Welford) | Bollinger | Delta |
|-----------|-----------------|-----------|-------|
| 1K | 10,036 ns | 2,733 ns | -267% (3.67× slower) |
| 10K | 108,849 ns | 31,419 ns | -246% (3.46× slower) |
| 100K | 1,026,786 ns | 279,538 ns | -267% (3.67× slower) |

**Note**: Bollinger Bands using sum-of-squares approach is even faster than separate passes, likely due to simpler arithmetic operations and better cache utilization.

### Period Sensitivity (at 100K data points)

| Period | Fused | Separate | Speedup |
|--------|-------|----------|---------|
| 5 | 987,426 ns | 363,535 ns | -172% (2.72× slower) |
| 10 | 1,022,642 ns | 370,358 ns | -176% (2.76× slower) |
| 20 | 1,027,773 ns | 369,944 ns | -178% (2.78× slower) |
| 50 | 970,435 ns | 359,111 ns | -170% (2.70× slower) |
| 100 | 1,000,445 ns | 339,350 ns | -195% (2.95× slower) |

**Note**: The fused approach remains slower across all period sizes. The performance gap is consistent, indicating the overhead is inherent to the Welford algorithm implementation.

### Throughput Analysis

| Data Size | Fused (elements/sec) | Separate (elements/sec) | Ratio |
|-----------|---------------------|------------------------|-------|
| 10K | 94.4M | 265.5M | 0.36× |
| 100K | 100.1M | 278.8M | 0.36× |
| 1M | 98.8M | 294.1M | 0.34× |

**Note**: Separate passes achieve 2.8-3× higher throughput across all data sizes.

### Pre-allocated Buffer Comparison

Testing with `_into()` variants to eliminate allocation overhead:

| Data Size | Fused Into | Separate Into | Delta |
|-----------|------------|---------------|-------|
| 1K | 10,164 ns | 3,509 ns | -190% (2.90× slower) |
| 10K | 108,596 ns | 36,042 ns | -201% (3.01× slower) |
| 100K | 1,028,042 ns | 379,392 ns | -171% (2.71× slower) |

**Note**: Pre-allocation does not change the fundamental performance characteristics. The fused approach remains significantly slower.

## Analysis

### Observed Results vs Expected

**Hypothesis was incorrect.** The expected benefits of fusion did not materialize:

1. **Single-pass did NOT help** (1K-10K):
   - The Welford algorithm's per-element overhead dominates
   - Simple arithmetic in SMA/StdDev is faster than Welford's divisions

2. **Memory bandwidth NOT the bottleneck** (100K+):
   - Two separate passes are still faster than one fused pass
   - The extra memory read is cheaper than Welford's computation

3. **Welford vs Sum-of-Squares** (Bollinger comparison):
   - Sum-of-squares (Bollinger) is ~30% faster than separate passes
   - Welford is ~3.5× slower than sum-of-squares
   - Numerical stability comes at a significant performance cost

### Why Welford is Slower

The Welford algorithm performs more operations per element:

```
Welford (per element):
  count += 1
  delta = x - mean           # 1 subtraction
  mean += delta / count      # 1 division, 1 addition
  delta2 = x - mean          # 1 subtraction (using updated mean)
  m2 += delta * delta2       # 1 multiply, 1 addition

SMA + StdDev (per element):
  SMA: sum += x - old; mean = sum / n  # 2 ops amortized
  StdDev: Similar sliding window trick
```

The division operation in Welford's algorithm is expensive (15-20 cycles on modern CPUs), while the sliding window approach uses primarily additions and subtractions (1 cycle each).

### Scaling Analysis

To verify O(n) complexity, compare 10K to 100K (10x data):

| Approach | 10K→100K Ratio | O(n) Verified? |
|----------|----------------|----------------|
| Fused | 9.4× | Yes |
| Separate | 9.7× | Yes |
| Bollinger | 8.9× | Yes |

All approaches show O(n) scaling behavior.

### Memory Bandwidth Estimation

Theoretical memory access patterns:

| Approach | Reads | Writes | Total Bandwidth |
|----------|-------|--------|-----------------|
| Fused | 1 × input | 3 × output | 4n × sizeof(f64) |
| Separate | 2 × input | 3 × output | 5n × sizeof(f64) |
| Bollinger | 1 × input | 3 × output | 4n × sizeof(f64) |

**Actual observation**: Memory bandwidth is NOT the bottleneck. The computational overhead of Welford's algorithm dominates.

## Go/No-Go Decision

**Decision**: NO-GO

### Criteria Checklist

#### For GO (adopt fused approach):
- [ ] Fused achieves ≥20% speedup over separate at 100K
- [ ] Speedup is consistent across data sizes (1K, 10K, 100K)
- [ ] Speedup persists with pre-allocated buffers
- [ ] No significant regression in any scenario

#### For NO-GO (keep separate implementations):
- [x] Fused speedup is <10% (actually 2.8× SLOWER)
- [x] Fused is slower in ALL scenarios
- [x] Implementation complexity outweighs benefits (no benefits exist)

### Rationale

The Welford-based fused kernel is **2.8× slower** than the separate SMA + StdDev approach. This significant performance regression makes the fused approach unsuitable for production use.

The existing separate implementations using sliding window algorithms are highly optimized and should be retained.

## Implications for fast-ta Architecture

### NO-GO Decision Applied:

1. **Keep Current Design**: Separate SMA and rolling_stddev implementations are optimal
2. **Bollinger Bands**: Already optimized; no refactoring needed
3. **Focus Elsewhere**: Look for gains in other kernels (E03, E04)
4. **Consider Numerical Stability Trade-offs**: For extreme value scenarios, Welford may still be preferred despite performance cost

### Recommendations:

1. **Do NOT adopt fused Welford kernel** for mean/variance/stddev computation
2. **Retain sliding window algorithms** in SMA and rolling_stddev
3. **Consider sum-of-squares approach** (like Bollinger) for even better performance where numerical stability is acceptable
4. **Document the trade-off**: Welford provides numerical stability at ~3× performance cost

## Follow-up Actions

1. **E03 (EMA Fusion)**: Different algorithm characteristics; evaluate independently
2. **E04 (Rolling Extrema)**: Deque-based approach may have different outcomes
3. **Document findings**: Update PRD with performance guidance

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e02_running_stat.rs`
- **Kernel Implementation**: `crates/fast-ta-core/src/kernels/running_stat.rs`
- **Criterion Output**: `target/criterion/e02_running_stat/`
- **Raw JSON Data**: `target/criterion/e02_running_stat/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E02 RunningStat fusion benchmarks
cargo bench --package fast-ta-experiments --bench e02_running_stat

# View HTML report
open target/criterion/e02_running_stat/report/index.html

# Run specific benchmark group
cargo bench --package fast-ta-experiments --bench e02_running_stat -- "fused_welford"
cargo bench --package fast-ta-experiments --bench e02_running_stat -- "separate_passes"
```

## Technical Notes

### Welford's Algorithm

The fused kernel uses Welford's online algorithm:

```
For each new value x:
  count += 1
  delta = x - mean
  mean += delta / count
  delta2 = x - mean  # Using updated mean
  m2 += delta * delta2

Variance = m2 / count
StdDev = sqrt(Variance)
```

Key properties:
- Numerically stable (no catastrophic cancellation)
- Single-pass (O(n) time, O(1) auxiliary space)
- Sliding window support via inverse operation
- **Performance trade-off**: ~3× slower than sum-of-squares

### Sum-of-Squares Approach (Bollinger)

Traditional approach:

```
Variance = (sum_sq / n) - (sum / n)²
```

Potential numerical issues:
- Catastrophic cancellation when variance is small relative to mean²
- May produce negative variance due to floating-point errors

**Performance advantage**: Simple arithmetic operations, highly cache-efficient.

### Sliding Window Approach (SMA/StdDev)

Used by current implementations:

```
For each new value x, old value x_old:
  sum = sum + x - x_old
  mean = sum / n
```

- Amortized O(1) per element
- Minimal division operations
- **Best performance** for this use case

### Memory Layout

Both approaches output 3 vectors (mean, variance, stddev).
The fused approach writes all three in a single loop iteration,
but this benefit is overwhelmed by the computational overhead of Welford's algorithm.

---

*Report generated for fast-ta micro-experiments framework*
*Completed: 2024-12-20*
