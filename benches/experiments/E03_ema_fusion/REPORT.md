# E03: EMA Fusion Benchmarks

## Experiment Overview

**Experiment ID**: E03
**Name**: EMA Fusion Benchmarks
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

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

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e03_ema_fusion`*

### Primary Comparison: Multi-EMA Fusion (10 EMAs)

| Data Size | Fused (ema_multi) | Separate (10×ema) | Speedup | Verdict |
|-----------|-------------------|-------------------|---------|---------|
| 1K | TBD ns | TBD ns | TBD% | TBD |
| 10K | TBD ns | TBD ns | TBD% | TBD |
| 100K | TBD ns | TBD ns | TBD% | TBD |

### EMA Count Scaling (at 100K data points)

| EMA Count | Fused | Separate | Speedup | Notes |
|-----------|-------|----------|---------|-------|
| 3 EMAs | TBD | TBD | TBD% | Minimal benefit expected |
| 5 EMAs | TBD | TBD | TBD% | Growing benefit |
| 10 EMAs | TBD | TBD | TBD% | **Target scenario** |
| 20 EMAs | TBD | TBD | TBD% | Maximum benefit |

### EMA/DEMA/TEMA Fusion

| Data Size | Fused (ema_fusion) | Separate (3×ema + compute) | Speedup |
|-----------|--------------------|---------------------------|---------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

### MACD Fusion

| Data Size | Fused (macd_fusion) | Standard (macd) | Speedup |
|-----------|---------------------|-----------------|---------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

### Period Sensitivity (at 100K data points)

Does fusion benefit change with EMA period?

| Period | Fused (ema_fusion) | Separate | Speedup |
|--------|-------------------|----------|---------|
| 10 | TBD | TBD | TBD% |
| 20 | TBD | TBD | TBD% |
| 50 | TBD | TBD | TBD% |
| 200 | TBD | TBD | TBD% |

### Throughput Analysis (10 EMAs)

| Data Size | Fused (elements/sec) | Separate (elements/sec) | Ratio |
|-----------|---------------------|------------------------|-------|
| 10K | TBD | TBD | TBD |
| 100K | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD |

### Pre-allocated Buffer Comparison

Testing with `_into()` variants to eliminate allocation overhead:

| Data Size | Fused Multi-EMA Into | Separate EMA Into | Delta |
|-----------|---------------------|-------------------|-------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

## Analysis

### Expected Results

Based on algorithm analysis:

1. **Multi-EMA fusion benefit increases with EMA count**:
   - 3 EMAs: ~10% benefit (marginal)
   - 5 EMAs: ~15% benefit (approaching threshold)
   - 10 EMAs: ~20%+ benefit (target scenario)
   - 20 EMAs: ~25%+ benefit (significant)

2. **EMA/DEMA/TEMA fusion should show consistent benefit**:
   - Avoids recomputing EMA(EMA) separately
   - ~15-20% speedup expected

3. **MACD fusion may show smaller benefit**:
   - Only 2 EMAs (fast + slow) are computed together
   - Benefit comes from fused EMA computation

### Memory Bandwidth Analysis

Theoretical memory access patterns for 10 EMAs:

| Approach | Input Reads | Output Writes | Total Bandwidth |
|----------|------------|---------------|-----------------|
| Fused | 1 × input | 10 × output | 11n × sizeof(f64) |
| Separate | 10 × input | 10 × output | 20n × sizeof(f64) |

**Theoretical Fused Advantage**: ~45% less memory traffic

### Cache Efficiency

For 100K f64 values (800KB):
- Exceeds L1 cache (32-64KB)
- Fits in L2/L3 cache (256KB-8MB)
- Fused approach keeps input hot in cache

## Go/No-Go Decision

**Decision**: PENDING

### Criteria Checklist

#### For GO (adopt fused approach):
- [ ] Fused achieves ≥15% speedup for 10 EMAs at 100K
- [ ] Speedup scales with EMA count (more EMAs = more benefit)
- [ ] EMA/DEMA/TEMA fusion shows ≥10% speedup
- [ ] Speedup persists with pre-allocated buffers
- [ ] No significant regression in any scenario

#### For NO-GO (keep separate implementations):
- [ ] Fused speedup is <10% for 10 EMAs
- [ ] OR fused is slower in some scenarios
- [ ] OR implementation complexity outweighs benefits

## Implications for fast-ta Architecture

### If GO:

1. **Multi-Indicator Workloads**: Use ema_multi() when computing many EMAs
2. **DEMA/TEMA Indicators**: Add fused DEMA/TEMA as first-class indicators
3. **Plan Mode Optimization**: Detect EMA groups and fuse in execution plan
4. **Ribbons/Bands**: EMA ribbons (8 EMAs) should use fusion

### If NO-GO:

1. **Keep Current Design**: Separate EMA calls are fine
2. **Focus on E02/E04**: Look for gains in RunningStat and Rolling Extrema
3. **Simplicity**: Simpler code may be worth small performance cost

## Comparison with E02 (RunningStat Fusion)

| Metric | E02 RunningStat | E03 EMA Fusion |
|--------|-----------------|----------------|
| Fusion Type | Mean/Var/StdDev | Multiple EMAs |
| Memory Benefit | 1 read vs 2 reads | 1 read vs k reads |
| Target Speedup | ≥20% | ≥15% (for k≥10) |
| Complexity | Low | Medium |

## Follow-up Actions

After E03 completes:

1. **If GO**:
   - Consider adding EMA ribbon indicator
   - Optimize MACD to use internal fusion
   - Document fusion patterns for developers

2. **If NO-GO**:
   - Evaluate if E04 (Rolling Extrema) provides sufficient gains
   - Consider removing ema_fusion kernel to reduce complexity

3. **E04 (Rolling Extrema)**: Different optimization strategy (deque-based)

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

Key optimization: Single read of `data[i]` is used for all k EMA updates.

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
avoiding separate computations.

### Memory Layout

Fused approach writes to all output vectors in each loop iteration,
which may have implications for write combining and cache behavior.
Pre-allocated buffers help measure pure computation overhead.

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
