# E01: Baseline Cost Benchmarks

## Experiment Overview

**Experiment ID**: E01
**Name**: Baseline Cost Benchmarks
**Status**: COMPLETED
**Date**: 2024-12-20

## Objective

Establish performance baselines for all 7 core technical indicators in the fast-ta library. These baselines will serve as reference points for:

1. Measuring individual indicator computation costs
2. Identifying performance bottlenecks
3. Comparing against fusion kernel improvements (E02-E04)
4. Validating O(n) time complexity claims

## Indicators Benchmarked

| Indicator | Period(s) | Algorithm | Expected Complexity |
|-----------|-----------|-----------|---------------------|
| SMA | 20 | Rolling sum | O(n) |
| EMA | 20 | Recursive formula | O(n) |
| RSI | 14 | Wilder smoothing | O(n) |
| MACD | 12, 26, 9 | Triple EMA | O(n) |
| ATR | 14 | Wilder smoothing | O(n) |
| Bollinger Bands | 20, 2.0 std | Rolling sum + sum-of-squares | O(n) |
| Stochastic | 14, 3 | Rolling extrema | O(n*k) naive / O(n) with deque |

## Methodology

### Data Sizes

| Size | Points | Description |
|------|--------|-------------|
| 1K | 1,000 | Quick iteration, development testing |
| 10K | 10,000 | Short-term trading (few weeks of minute bars) |
| 100K | 100,000 | Multi-year intraday or decades of daily |

*Note: 1M (1,000,000) is excluded from quick benchmarks but included in full suite.*

### Configuration

- **Measurement time**: 5-10 seconds per benchmark (scaled with data size)
- **Sample size**: 20-100 samples (reduced for larger datasets)
- **Warm-up**: 2 seconds
- **Data generation**: Seeded random walk (seed=42 for reproducibility)
- **Dead code prevention**: All results wrapped in `black_box()`

## Results

*Results populated from: `cargo bench --package fast-ta-experiments --bench e01_baseline`*

### Individual Indicator Performance

| Indicator | 1K (ns/op) | 10K (ns/op) | 100K (ns/op) | ns/element @ 100K |
|-----------|------------|-------------|--------------|-------------------|
| SMA | 1,406 | 14,274 | 139,365 | 1.39 |
| EMA | 1,718 | 18,717 | 171,445 | 1.71 |
| RSI | 4,485 | 49,349 | 522,581 | 5.23 |
| MACD | 7,409 | 80,746 | 761,453 | 7.61 |
| ATR | 4,968 | 50,483 | 509,189 | 5.09 |
| Bollinger | 2,962 | 33,643 | 300,809 | 3.01 |
| Stochastic | 8,271 | 86,871 | 894,210 | 8.94 |

### Combined Performance (All 7 Indicators)

| Data Size | Total Time | Per-Indicator Avg |
|-----------|------------|-------------------|
| 10K | 308.7 µs | 44.1 µs |
| 100K | 3.43 ms | 490.3 µs |

### Throughput Analysis

| Indicator | Elements/sec @ 100K | Memory Bandwidth |
|-----------|---------------------|------------------|
| SMA | 717.5 M/s | ~5.7 GB/s |
| EMA | 583.3 M/s | ~4.7 GB/s |
| RSI | 191.4 M/s | ~1.5 GB/s |
| MACD | 131.3 M/s | ~1.1 GB/s |
| ATR | 196.4 M/s | ~1.6 GB/s |
| Bollinger | 332.4 M/s | ~2.7 GB/s |
| Stochastic | 111.8 M/s | ~0.9 GB/s |

*Note: Memory bandwidth estimated assuming 8 bytes per f64 element read.*

## Analysis

### Complexity Verification

To verify O(n) complexity, we compare scaling from 10K to 100K (10x data):
- **Expected**: ~10x time increase
- **Actual**: All indicators show near-linear scaling (8.9x to 10.6x)

| Indicator | 10K->100K Ratio | O(n) Verified? |
|-----------|-----------------|----------------|
| SMA | 9.76x | Yes |
| EMA | 9.16x | Yes |
| RSI | 10.59x | Yes |
| MACD | 9.43x | Yes |
| ATR | 10.09x | Yes |
| Bollinger | 8.94x | Yes |
| Stochastic | 10.29x | Yes |

### Relative Performance

Ranking of indicators by per-element cost (fastest to slowest):

1. **SMA** - 1.39 ns/element (fastest, simple rolling sum)
2. **EMA** - 1.71 ns/element (single pass, recursive formula)
3. **Bollinger** - 3.01 ns/element (rolling sum + variance)
4. **ATR** - 5.09 ns/element (True Range + Wilder smoothing)
5. **RSI** - 5.23 ns/element (gains/losses + Wilder smoothing)
6. **MACD** - 7.61 ns/element (computes 3 EMAs)
7. **Stochastic** - 8.94 ns/element (rolling extrema computation)

### Bottleneck Identification

**Most Expensive Operations**:
- **Stochastic**: Rolling min/max extrema detection is the most expensive at 8.94 ns/element
- **MACD**: Computing 3 separate EMAs (fast, slow, signal) adds significant overhead at 7.61 ns/element
- **RSI/ATR**: Wilder smoothing with gains/losses separation costs ~5 ns/element

**Potential Optimization Targets**:
- **Stochastic**: Could benefit from deque-based O(n) rolling extrema instead of O(n*k) naive approach
- **MACD**: EMA fusion could reduce memory passes when computing multiple EMAs
- **Bollinger**: RunningStat kernel could fuse mean/variance computation

## Expected Outcomes

Based on algorithm analysis, we expect:

1. **SMA and EMA**: Fastest indicators (single pass, simple arithmetic) - Confirmed
2. **RSI**: Moderate cost (requires gains/losses separation + Wilder smoothing) - Confirmed
3. **MACD**: Higher cost (computes 3 EMAs) - Confirmed
4. **ATR**: Moderate (True Range + Wilder smoothing) - Confirmed
5. **Bollinger Bands**: Higher cost (SMA + rolling stddev) - Actually faster than expected
6. **Stochastic**: Highest cost with naive O(n*k) rolling extrema - Confirmed

## Go/No-Go Decision

**Decision**: GO

### Criteria for GO

- [x] All indicators demonstrate O(n) or O(n log n) complexity
- [x] Baseline timings are captured for all 7 indicators
- [x] Results are reproducible across multiple runs
- [x] No indicators exceed 1000 ns/element for 100K data (all under 9 ns/element)
- [x] Combined benchmark completes in reasonable time (3.43 ms for 100K)

### Criteria for NO-GO

- [ ] Any indicator shows worse than O(n*k) complexity - Not observed
- [ ] Benchmark harness issues prevent accurate measurement - No issues
- [ ] Results vary significantly (>20%) between runs - Results are stable

## Follow-up Actions

After E01 completes:

1. **E02 (RunningStat)**: Use Bollinger baseline (3.01 ns/element) to measure fusion benefit
2. **E03 (EMA Fusion)**: Use EMA/MACD baselines (1.71/7.61 ns/element) for multi-EMA comparison
3. **E04 (Rolling Extrema)**: Use Stochastic baseline (8.94 ns/element) to measure deque improvement

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e01_baseline.rs`
- **Criterion Output**: `target/criterion/e01_baseline_*/`
- **Raw JSON Data**: `target/criterion/e01_baseline_*/*/new/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E01 baseline benchmarks
cargo bench --package fast-ta-experiments --bench e01_baseline

# View HTML report
open target/criterion/e01_baseline_sma/report/index.html

# View specific indicator report
open target/criterion/e01_baseline_sma/period_20/report/index.html
```

## Notes

- Benchmarks use `QUICK_DATA_SIZES` (1K, 10K, 100K) by default
- For full benchmarks including 1M, modify the benchmark to use `DATA_SIZES`
- Results may vary based on system load, CPU thermal throttling, etc.
- Multiple runs recommended to establish statistical confidence

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: 2024-12-20*
