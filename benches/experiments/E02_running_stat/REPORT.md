# E02: RunningStat Fusion Benchmarks

## Experiment Overview

**Experiment ID**: E02
**Name**: RunningStat Fusion Benchmarks
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

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

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e02_running_stat`*

### Primary Comparison: Fused vs Separate

| Data Size | Fused (Welford) | Separate Passes | Speedup | Verdict |
|-----------|-----------------|-----------------|---------|---------|
| 1K | TBD ns | TBD ns | TBD% | TBD |
| 10K | TBD ns | TBD ns | TBD% | TBD |
| 100K | TBD ns | TBD ns | TBD% | TBD |

### Component Breakdown

Understanding where time is spent in the separate approach:

| Component | 1K | 10K | 100K | % of Separate Total |
|-----------|-----|------|-------|---------------------|
| SMA (mean) | TBD | TBD | TBD | TBD% |
| rolling_stddev | TBD | TBD | TBD | TBD% |
| variance calc | TBD | TBD | TBD | TBD% |
| **Total** | TBD | TBD | TBD | 100% |

### Bollinger Comparison

| Data Size | Fused (Welford) | Bollinger | Delta |
|-----------|-----------------|-----------|-------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

### Period Sensitivity (at 100K data points)

| Period | Fused | Separate | Speedup |
|--------|-------|----------|---------|
| 5 | TBD | TBD | TBD% |
| 10 | TBD | TBD | TBD% |
| 20 | TBD | TBD | TBD% |
| 50 | TBD | TBD | TBD% |
| 100 | TBD | TBD | TBD% |

### Throughput Analysis

| Data Size | Fused (elements/sec) | Separate (elements/sec) | Ratio |
|-----------|---------------------|------------------------|-------|
| 10K | TBD | TBD | TBD |
| 100K | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD |

### Pre-allocated Buffer Comparison

Testing with `_into()` variants to eliminate allocation overhead:

| Data Size | Fused Into | Separate Into | Delta |
|-----------|------------|---------------|-------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

## Analysis

### Expected Results

Based on algorithm analysis:

1. **Fused should win on small data** (1K-10K):
   - Cache-resident data benefits from single-pass access
   - Loop overhead reduction is proportionally larger

2. **Fused should maintain advantage on large data** (100K+):
   - Memory bandwidth becomes the bottleneck
   - Single read vs double read is significant

3. **Welford vs Sum-of-Squares** (Bollinger comparison):
   - Sum-of-squares is slightly simpler arithmetically
   - Welford has better numerical stability
   - Performance should be comparable

### Scaling Analysis

To verify O(n) complexity, compare 10K to 100K (10x data):
- **Expected**: ~10x time increase for both approaches
- **If fused has advantage**: Advantage should be consistent across scales

| Approach | 10K→100K Ratio | O(n) Verified? |
|----------|----------------|----------------|
| Fused | TBD | TBD |
| Separate | TBD | TBD |
| Bollinger | TBD | TBD |

### Memory Bandwidth Estimation

Theoretical memory access patterns:

| Approach | Reads | Writes | Total Bandwidth |
|----------|-------|--------|-----------------|
| Fused | 1 × input | 3 × output | 4n × sizeof(f64) |
| Separate | 2 × input | 3 × output | 5n × sizeof(f64) |
| Bollinger | 1 × input | 3 × output | 4n × sizeof(f64) |

**Theoretical Fused Advantage**: 20% less memory traffic

## Go/No-Go Decision

**Decision**: PENDING

### Criteria Checklist

#### For GO (adopt fused approach):
- [ ] Fused achieves ≥20% speedup over separate at 100K
- [ ] Speedup is consistent across data sizes (1K, 10K, 100K)
- [ ] Speedup persists with pre-allocated buffers
- [ ] No significant regression in any scenario

#### For NO-GO (keep separate implementations):
- [ ] Fused speedup is <10%
- [ ] OR fused is slower in some scenarios
- [ ] OR implementation complexity outweighs benefits

## Implications for fast-ta Architecture

### If GO:

1. **Bollinger Bands**: Refactor to use `rolling_stats()` internally
2. **Future Indicators**: Use fused kernel as building block
3. **Plan Mode**: Fuse statistics computation across indicator DAG

### If NO-GO:

1. **Keep Current Design**: Separate SMA and rolling_stddev are fine
2. **Focus Elsewhere**: Look for gains in other kernels (E03, E04)
3. **Consider Numerical Stability**: May still prefer Welford for extreme values

## Follow-up Actions

After E02 completes:

1. **If GO**:
   - Consider Bollinger Bands refactoring
   - Document fusion pattern for other developers
   - Update performance recommendations

2. **E03 (EMA Fusion)**: Apply learnings about single-pass benefits
3. **E04 (Rolling Extrema)**: Different fusion strategy (deque-based)

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

### Sum-of-Squares Approach (Bollinger)

Traditional approach:

```
Variance = (sum_sq / n) - (sum / n)²
```

Potential numerical issues:
- Catastrophic cancellation when variance is small relative to mean²
- May produce negative variance due to floating-point errors

### Memory Layout

Both approaches output 3 vectors (mean, variance, stddev).
The fused approach writes all three in a single loop iteration,
which may have better write combining and cache behavior.

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
