# E01: Baseline Cost Benchmarks

## Experiment Overview

**Experiment ID**: E01
**Name**: Baseline Cost Benchmarks
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

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

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e01_baseline`*

### Individual Indicator Performance

| Indicator | 1K (ns/op) | 10K (ns/op) | 100K (ns/op) | ns/element @ 100K |
|-----------|------------|-------------|--------------|-------------------|
| SMA | TBD | TBD | TBD | TBD |
| EMA | TBD | TBD | TBD | TBD |
| RSI | TBD | TBD | TBD | TBD |
| MACD | TBD | TBD | TBD | TBD |
| ATR | TBD | TBD | TBD | TBD |
| Bollinger | TBD | TBD | TBD | TBD |
| Stochastic | TBD | TBD | TBD | TBD |

### Combined Performance (All 7 Indicators)

| Data Size | Total Time | Per-Indicator Avg |
|-----------|------------|-------------------|
| 10K | TBD | TBD |
| 100K | TBD | TBD |

### Throughput Analysis

| Indicator | Elements/sec @ 100K | Memory Bandwidth |
|-----------|---------------------|------------------|
| SMA | TBD | TBD |
| EMA | TBD | TBD |
| RSI | TBD | TBD |
| MACD | TBD | TBD |
| ATR | TBD | TBD |
| Bollinger | TBD | TBD |
| Stochastic | TBD | TBD |

## Analysis

### Complexity Verification

To verify O(n) complexity, we compare scaling from 10K to 100K (10x data):
- **Expected**: ~10x time increase
- **Actual**: TBD

| Indicator | 10K->100K Ratio | O(n) Verified? |
|-----------|-----------------|----------------|
| SMA | TBD | TBD |
| EMA | TBD | TBD |
| RSI | TBD | TBD |
| MACD | TBD | TBD |
| ATR | TBD | TBD |
| Bollinger | TBD | TBD |
| Stochastic | TBD | TBD |

### Relative Performance

Ranking of indicators by per-element cost (fastest to slowest):

1. TBD
2. TBD
3. TBD
4. TBD
5. TBD
6. TBD
7. TBD

### Bottleneck Identification

**Most Expensive Operations**:
- TBD

**Potential Optimization Targets**:
- TBD

## Expected Outcomes

Based on algorithm analysis, we expect:

1. **SMA and EMA**: Fastest indicators (single pass, simple arithmetic)
2. **RSI**: Moderate cost (requires gains/losses separation + Wilder smoothing)
3. **MACD**: Higher cost (computes 3 EMAs)
4. **ATR**: Moderate (True Range + Wilder smoothing)
5. **Bollinger Bands**: Higher cost (SMA + rolling stddev)
6. **Stochastic**: Highest cost with naive O(n*k) rolling extrema

## Go/No-Go Decision

**Decision**: PENDING

### Criteria for GO

- [ ] All indicators demonstrate O(n) or O(n log n) complexity
- [ ] Baseline timings are captured for all 7 indicators
- [ ] Results are reproducible across multiple runs
- [ ] No indicators exceed 1000 ns/element for 100K data
- [ ] Combined benchmark completes in reasonable time

### Criteria for NO-GO

- [ ] Any indicator shows worse than O(n*k) complexity
- [ ] Benchmark harness issues prevent accurate measurement
- [ ] Results vary significantly (>20%) between runs

## Follow-up Actions

After E01 completes:

1. **E02 (RunningStat)**: Use Bollinger baseline to measure fusion benefit
2. **E03 (EMA Fusion)**: Use EMA/MACD baselines for multi-EMA comparison
3. **E04 (Rolling Extrema)**: Use Stochastic baseline to measure deque improvement

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e01_baseline.rs`
- **Criterion Output**: `target/criterion/e01_baseline/`
- **Raw JSON Data**: `target/criterion/e01_baseline/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E01 baseline benchmarks
cargo bench --package fast-ta-experiments --bench e01_baseline

# View HTML report
open target/criterion/e01_baseline/report/index.html

# View specific indicator report
open target/criterion/e01_baseline/sma/report/index.html
```

## Notes

- Benchmarks use `QUICK_DATA_SIZES` (1K, 10K, 100K) by default
- For full benchmarks including 1M, modify the benchmark to use `DATA_SIZES`
- Results may vary based on system load, CPU thermal throttling, etc.
- Multiple runs recommended to establish statistical confidence

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
