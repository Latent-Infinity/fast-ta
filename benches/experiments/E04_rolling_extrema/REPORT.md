# E04: Rolling Extrema Benchmarks

## Experiment Overview

**Experiment ID**: E04
**Name**: Rolling Extrema (Deque-based vs Naive Scan)
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

## Objective

Evaluate whether using a monotonic deque algorithm for rolling max/min provides significant performance improvements over the naive scan approach.

### Hypothesis

The deque-based algorithm should be dramatically faster than naive scan because:

1. **Amortized O(1) per element**: Each element enters and exits the deque at most once
2. **O(n) vs O(n×k) complexity**: Naive scans the entire window for each output
3. **Speedup scales with period**: Larger periods amplify the difference
4. **Small working set**: Deque of size O(k) fits in cache

### Success Criteria

| Result | Speedup (at period ≥50) | Action |
|--------|------------------------|--------|
| **GO** | ≥5× faster | Adopt deque-based algorithm as standard |
| **INVESTIGATE** | 2-5× faster | Consider adoption, investigate edge cases |
| **NO-GO** | <2× faster or slower | Keep naive for simplicity |

## Algorithm Comparison

### Deque-Based (Optimized)

Using `rolling_max()` / `rolling_min()` with monotonic deque:

```
Invariant: Deque contains indices in decreasing order of values (for max)
           Front of deque is always the maximum in current window

For each element i:
  1. Pop expired elements from front (index < i - period + 1)
  2. Pop elements from back that are <= current value
  3. Push current index to back
  4. Front is the maximum for this window
```

- **Time Complexity**: O(n) amortized (each element pushed/popped at most once)
- **Space Complexity**: O(n) for output + O(k) for deque

### Naive Scan

Using `rolling_max_naive()` / `rolling_min_naive()`:

```
For each position i:
  Scan all k elements in window [i-k+1, i]
  Find maximum/minimum
```

- **Time Complexity**: O(n × k) where k is the period
- **Space Complexity**: O(n) for output

### Theoretical Speedup

| Period (k) | Naive Operations | Deque Operations | Theoretical Speedup |
|------------|------------------|------------------|---------------------|
| 5 | 5n | 2n | 2.5× |
| 14 | 14n | 2n | 7× |
| 50 | 50n | 2n | 25× |
| 100 | 100n | 2n | 50× |
| 200 | 200n | 2n | 100× |

Note: Actual speedup may differ due to:
- Cache effects (smaller deque vs larger window)
- Branch prediction overhead in deque maintenance
- Memory access patterns

## Approaches Benchmarked

### 1. Rolling Max (Deque vs Naive)

Direct comparison of rolling maximum algorithms.

### 2. Rolling Min (Deque vs Naive)

Direct comparison of rolling minimum algorithms.

### 3. Fused Extrema vs Separate Calls

Comparing `rolling_extrema()` (both max and min in single pass) vs calling `rolling_max()` and `rolling_min()` separately.

### 4. Period Scaling

How speedup changes across periods: 5, 14, 50, 100, 200.

### 5. Large Period Extreme Case

Testing with periods 100-1000 to verify O(n) vs O(n×k) scaling.

### 6. Stochastic Oscillator Use Case

Real-world use case: computing highest high and lowest low for Stochastic %K.

## Benchmark Configuration

### Data Sizes

| Size | Points | Description |
|------|--------|-------------|
| 1K | 1,000 | Quick iteration, cache-resident data |
| 10K | 10,000 | L2/L3 cache threshold |
| 100K | 100,000 | Memory-bound scenario |
| 1M | 1,000,000 | Throughput measurement (selected tests) |

### Periods Tested

| Period | Use Case |
|--------|----------|
| 5 | Short-term swing high/low |
| 14 | Stochastic Oscillator default |
| 50 | Medium-term channel |
| 100 | Long-term support/resistance |
| 200 | 200-day high/low |

### Parameters

- **Standard Period**: 14 (Stochastic %K default)
- **Measurement Time**: 5-10 seconds per benchmark
- **Sample Size**: 20-100 samples (scaled with data size)
- **Warm-up**: 2 seconds
- **Data**: Reproducible random walk (seed=42)

## Results

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e04_rolling_extrema`*

### Primary Comparison: Rolling Max (Period 14)

| Data Size | Deque (rolling_max) | Naive (rolling_max_naive) | Speedup | Verdict |
|-----------|---------------------|---------------------------|---------|---------|
| 1K | TBD ns | TBD ns | TBD× | TBD |
| 10K | TBD ns | TBD ns | TBD× | TBD |
| 100K | TBD ns | TBD ns | TBD× | TBD |

### Period Scaling (at 100K data points)

| Period | Deque | Naive | Speedup | Notes |
|--------|-------|-------|---------|-------|
| 5 | TBD | TBD | TBD× | Minimal benefit expected |
| 14 | TBD | TBD | TBD× | Standard Stochastic period |
| 50 | TBD | TBD | TBD× | **Target scenario** |
| 100 | TBD | TBD | TBD× | Significant benefit |
| 200 | TBD | TBD | TBD× | Maximum benefit |

### Large Period Extreme Case (at 100K data points)

| Period | Deque | Naive | Speedup |
|--------|-------|-------|---------|
| 100 | TBD | TBD | TBD× |
| 200 | TBD | TBD | TBD× |
| 500 | TBD | TBD | TBD× |
| 1000 | TBD | TBD | TBD× |

### Fused Extrema vs Separate

| Data Size | Fused (rolling_extrema) | Separate (max + min) | Speedup |
|-----------|------------------------|---------------------|---------|
| 1K | TBD | TBD | TBD% |
| 10K | TBD | TBD | TBD% |
| 100K | TBD | TBD | TBD% |

### Throughput Analysis

| Data Size | Deque (elements/sec) | Naive (elements/sec) | Ratio |
|-----------|---------------------|---------------------|-------|
| 10K | TBD | TBD | TBD |
| 100K | TBD | TBD | TBD |
| 1M | TBD | TBD | TBD |

### Pre-allocated Buffer Comparison

| Data Size | rolling_max_into | rolling_extrema_into | Notes |
|-----------|------------------|---------------------|-------|
| 1K | TBD | TBD | Allocation eliminated |
| 10K | TBD | TBD | |
| 100K | TBD | TBD | |

### Stochastic Use Case (highest high + lowest low)

| Data Size | Deque (OHLCV) | Naive (OHLCV) | Speedup |
|-----------|---------------|---------------|---------|
| 1K | TBD | TBD | TBD× |
| 10K | TBD | TBD | TBD× |
| 100K | TBD | TBD | TBD× |

## Analysis

### Expected Results

Based on algorithm analysis:

1. **Speedup should scale linearly with period**:
   - Period 5: ~2-3× speedup
   - Period 14: ~5-7× speedup
   - Period 50: ~20-30× speedup
   - Period 100: ~40-50× speedup
   - Period 200: ~80-100× speedup

2. **Fused extrema should be ~40-50% faster than separate**:
   - Single pass through data vs two passes
   - Both deques maintained simultaneously

3. **Speedup should be consistent across data sizes**:
   - Both algorithms are O(n) or O(n×k)
   - Larger data sizes may show better deque performance due to cache

### Complexity Verification

If the deque algorithm is truly O(n):
- Doubling the data size should double the time
- Changing the period should NOT significantly affect time

If the naive algorithm is truly O(n×k):
- Doubling the data size should double the time
- Doubling the period should also double the time

### Memory Access Patterns

| Algorithm | Memory Pattern | Cache Efficiency |
|-----------|---------------|------------------|
| Deque | Sequential input read, small deque updates | Excellent |
| Naive | Sequential output, random window reads | Poor for large k |

## Go/No-Go Decision

**Decision**: PENDING

### Criteria Checklist

#### For GO (adopt deque-based approach):

- [ ] Deque achieves ≥5× speedup at period 50 with 100K data
- [ ] Speedup scales approximately linearly with period
- [ ] No regression at small periods (period ≤ 10)
- [ ] Fused extrema outperforms separate calls
- [ ] Pre-allocated buffers show consistent improvement

#### For NO-GO (keep naive implementation):

- [ ] Speedup is <2× at period 50
- [ ] OR deque is slower at small periods
- [ ] OR deque shows unexpected performance characteristics

## Implications for fast-ta Architecture

### If GO:

1. **Stochastic Indicator**: Use deque-based rolling extrema
2. **Channels/Bands**: Use for Donchian channels, ATR calculations
3. **Swing Detection**: Fast highest-high/lowest-low lookups
4. **Memory Efficiency**: Rolling extrema needs O(k) space per channel

### If NO-GO:

1. **Keep Naive for Small Periods**: If speedup only shows at large k
2. **Consider Hybrid**: Naive for k < 20, deque for k >= 20
3. **Simplicity**: Naive is easier to understand and maintain

## Comparison with Other Experiments

| Experiment | Algorithm Type | Expected Speedup | Complexity Reduction |
|------------|---------------|------------------|---------------------|
| E02 RunningStat | Fusion | ≥20% | 2 passes → 1 pass |
| E03 EMA Fusion | Fusion | ≥15% (≥10 EMAs) | k passes → 1 pass |
| **E04 Rolling Extrema** | **Better Algorithm** | **≥5× (at k≥50)** | **O(n×k) → O(n)** |

E04 is unique among the experiments because it replaces a naive algorithm with a fundamentally better one (O(n) vs O(n×k)), rather than just fusing passes.

## Real-World Applications

### Stochastic Oscillator

```rust
// Without optimization: O(n×k) for each
let highest_high = rolling_max_naive(highs, 14);
let lowest_low = rolling_min_naive(lows, 14);

// With optimization: O(n) for each
let highest_high = rolling_max(highs, 14);
let lowest_low = rolling_min(lows, 14);

// Or fused: O(n) for both together
let extrema = rolling_extrema(highs, 14); // For highest high
// (Would need separate call for lows)
```

### Donchian Channels

```rust
// Upper channel: highest high over period
// Lower channel: lowest low over period
let channels = rolling_extrema(prices, 20);
```

### Williams %R

Similar to Stochastic, requires rolling high and low over lookback period.

## Follow-up Actions

After E04 completes:

1. **If GO**:
   - Update Stochastic implementation to use deque-based extrema
   - Consider adding Donchian Channels indicator
   - Document deque algorithm for maintainers

2. **If NO-GO**:
   - Investigate hybrid approach
   - Consider whether implementation complexity is worth small gains

3. **E05 (Plan Overhead)**: Different focus (overhead measurement, not optimization)

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e04_rolling_extrema.rs`
- **Kernel Implementation**: `crates/fast-ta-core/src/kernels/rolling_extrema.rs`
- **Criterion Output**: `target/criterion/e04_rolling_extrema/`
- **Raw JSON Data**: `target/criterion/e04_rolling_extrema/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E04 rolling extrema benchmarks
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema

# View HTML report
open target/criterion/e04_rolling_extrema/report/index.html

# Run specific benchmark group
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema -- "rolling_max_deque"
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema -- "period_scaling"
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema -- "large_period"
```

## Technical Notes

### Monotonic Deque Algorithm

The monotonic deque maintains these invariants:

**For Rolling Maximum:**
- Indices in deque are in increasing order (oldest to newest)
- Values at those indices are in decreasing order
- Front of deque always contains the index of the maximum value in current window

**Key Operations:**
```
push_max(index, data):
  // Remove expired elements from front
  while front < index - period + 1: pop_front()

  // Remove smaller elements from back
  while data[back] <= data[index]: pop_back()

  // Add current element
  push_back(index)

  // Front is the maximum
  return data[front]
```

### NaN Handling

The deque algorithm handles NaN values by:
- Skipping NaN values during push (not added to deque)
- If deque is empty after expiration, returning NaN
- This matches the behavior expected for financial time series

### Memory Layout

The `MonotonicDeque<T>` structure:
- Uses `VecDeque<usize>` to store indices (not values)
- Indices are 8 bytes on 64-bit systems
- Maximum deque size is period (k)
- Memory: ~8 × k bytes for the deque itself

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
