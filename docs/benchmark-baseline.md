# fast-ta Performance Baseline

Benchmark results establishing performance baselines for all indicators.
Run with `cargo bench -p fast-ta` to reproduce.

## Test Environment

- Date: 2025-12-24
- Platform: macOS (Darwin)
- Rust: stable
- Profile: release with LTO
- TA-Lib: v0.4.0 (via ta-lib-sys FFI)

## TA-Lib Comparison (100K elements)

Comprehensive FFI comparison between fast-ta and TA-Lib C library across 35 indicators.

### Moving Averages

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| SMA(20) | 235 µs | 96 µs | 0.41× | TA-Lib |
| EMA(20) | 140 µs | 124 µs | 0.88× | TA-Lib |
| WMA(20) | 122 µs | 119 µs | 0.97× | TA-Lib |
| DEMA(20) | 280 µs | 262 µs | 0.94× | TA-Lib |
| TEMA(20) | 340 µs | 384 µs | **1.13×** | **fast-ta** |
| TRIMA(20) | 218 µs | 180 µs | 0.83× | TA-Lib |
| KAMA(20) | 402 µs | 215 µs | 0.53× | TA-Lib |
| T3(20) | 1.23 ms | 172 µs | 0.14× | TA-Lib |

### Momentum

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| RSI(14) | 376 µs | 427 µs | **1.14×** | **fast-ta** |
| MACD(12,26,9) | 395 µs | 419 µs | **1.06×** | **fast-ta** |
| MOM(14) | 19 µs | 15 µs | 0.79× | TA-Lib |
| ROC(14) | 45 µs | 53 µs | **1.18×** | **fast-ta** |
| CMO(14) | 1.05 ms | 422 µs | 0.40× | TA-Lib |
| APO(12,26) | 149 µs | 263 µs | **1.76×** | **fast-ta** |
| TRIX(14) | 647 µs | 412 µs | 0.64× | TA-Lib |

### Trend

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| ADX(14) | 419 µs | 416 µs | ~1.0× | Tie |
| DX(14) | 492 µs | 407 µs | 0.83× | TA-Lib |
| AROON(14) | 712 µs | 450 µs | 0.63× | TA-Lib |
| CCI(20) | 1.06 ms | 720 µs | 0.68× | TA-Lib |

### Volatility

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| ATR(14) | 374 µs | 396 µs | **1.06×** | **fast-ta** |
| TRANGE | 46 µs | 30 µs | 0.65× | TA-Lib |
| Bollinger(20,2) | 269 µs | 382 µs | **1.42×** | **fast-ta** |

### Stochastic

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| Stochastic(14,3,3) | 1.32 ms | 650 µs | 0.49× | TA-Lib |
| StochFast(14,3) | 1.07 ms | 545 µs | 0.51× | TA-Lib |
| Williams %R(14) | 746 µs | 424 µs | 0.57× | TA-Lib |
| ULTOSC(7,14,28) | 2.12 ms | 367 µs | 0.17× | TA-Lib |

### Volume

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| OBV | 257 µs | 73 µs | 0.28× | TA-Lib |
| AD | 80 µs | 93 µs | **1.15×** | **fast-ta** |
| MFI(14) | 989 µs | 141 µs | 0.14× | TA-Lib |

### Statistics

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| VAR(20) | 1.36 ms | 130 µs | 0.10× | TA-Lib |
| LINEARREG(20) | 515 µs | 441 µs | 0.86× | TA-Lib |
| TSF(20) | 517 µs | 438 µs | 0.85× | TA-Lib |

### Other

| Indicator | fast-ta | TA-Lib | Ratio | Winner |
|-----------|---------|--------|-------|--------|
| MIDPOINT(20) | 426 µs | 597 µs | **1.40×** | **fast-ta** |
| MIDPRICE(20) | 440 µs | 418 µs | 0.95× | TA-Lib |
| BOP | 56 µs | 56 µs | ~1.0× | Tie |

### Summary

**Overall: fast-ta wins 9/35 (26%), TA-Lib wins 24/35 (69%), Tie 2/35 (6%)**

**fast-ta is faster on:**
- TEMA, RSI, MACD, ROC, APO, ATR, Bollinger, AD, MIDPOINT

**TA-Lib is faster on:**
- Most simple moving averages (SMA, EMA, WMA, DEMA, TRIMA, KAMA, T3)
- Stochastic variants and rolling-window indicators
- Volume indicators using simple cumulation (OBV, MFI)
- Statistical functions with optimized variance algorithms

### Analysis

- **Moving Averages**: TA-Lib uses highly optimized C with SIMD. Our pure Rust
  implementation prioritizes correctness and maintainability. TEMA is an exception
  where fast-ta's triple-EMA fusing is more efficient.
- **RSI/ATR**: fast-ta's Wilder smoothing implementation is more cache-efficient.
- **MACD/APO**: fast-ta uses fused computation for better cache efficiency.
- **Bollinger**: fast-ta computes rolling variance more efficiently.
- **Stochastic/Williams %R**: TA-Lib's O(n×k) is faster for small periods due to
  simpler implementation; fast-ta's deque approach has overhead at typical period sizes.
- **AD (Accumulation/Distribution)**: fast-ta's vectorized approach wins.
- **MIDPOINT**: fast-ta's rolling extrema is faster than TA-Lib's implementation.
- **VAR/MFI**: TA-Lib uses highly optimized variance/rolling sum algorithms.
- **T3**: TA-Lib uses a single-pass T3 implementation; fast-ta chains 6 EMAs.

## Results Summary

All indicators demonstrate **O(n) linear time complexity** with consistent
throughput across input sizes (100 to 100,000 elements).

### Single-Series Indicators

| Indicator | 100 elem | 1K elem | 10K elem | 100K elem | Throughput |
|-----------|----------|---------|----------|-----------|------------|
| SMA(20)   | 232 ns   | 2.8 µs  | 30 µs    | 297 µs    | ~337 Melem/s |
| EMA(20)   | 149 ns   | 1.6 µs  | 17 µs    | 161 µs    | ~621 Melem/s |
| RSI(14)   | 395 ns   | 4.3 µs  | 44 µs    | 429 µs    | ~233 Melem/s |
| MACD      | 406 ns   | 4.4 µs  | 47 µs    | 468 µs    | ~214 Melem/s |
| Bollinger | 317 ns   | 2.9 µs  | 31 µs    | 289 µs    | ~346 Melem/s |

### OHLC Indicators

| Indicator   | 100 elem | 1K elem | 10K elem | 100K elem | Throughput |
|-------------|----------|---------|----------|-----------|------------|
| ATR(14)     | 391 ns   | 4.3 µs  | 42 µs    | 397 µs    | ~252 Melem/s |
| Stochastic  | 1.18 µs  | 13.0 µs | 134 µs   | 1.31 ms   | ~76 Melem/s |
| ADX(14)     | 434 ns   | 4.2 µs  | 44 µs    | 437 µs    | ~229 Melem/s |
| Williams %R | 720 ns   | 6.5 µs  | 70 µs    | 684 µs    | ~146 Melem/s |
| Donchian    | 723 ns   | 6.2 µs  | 70 µs    | 679 µs    | ~147 Melem/s |

### Volume Indicators

| Indicator | 100 elem | 1K elem | 10K elem | 100K elem | Throughput |
|-----------|----------|---------|----------|-----------|------------|
| OBV       | 141 ns   | 1.9 µs  | 28 µs    | 275 µs    | ~364 Melem/s |
| VWAP      | 180 ns   | 1.6 µs  | 18 µs    | 179 µs    | ~559 Melem/s |

## Key Observations

1. **Linear Scaling**: All indicators scale linearly (10x input = ~10x time)
2. **High Throughput**: Fastest indicators (EMA, VWAP) exceed 550 Melem/s
3. **Consistent Performance**: Throughput remains stable across input sizes
4. **Memory Efficient**: Pre-allocation strategy avoids runtime allocations

## Notable Improvements Since Last Baseline

- **Donchian**: 2-3× faster (optimized monotonic deque implementation)
- **Williams %R**: Improved at larger sizes (10K+)
- **MACD**: 20% faster overall

## Complexity Analysis

| Indicator   | Time Complexity | Space Complexity | Notes |
|-------------|-----------------|------------------|-------|
| SMA         | O(n)            | O(n)             | Ring buffer optimization |
| EMA         | O(n)            | O(n)             | Single-pass, SMA seed |
| RSI         | O(n)            | O(n)             | Wilder's smoothing |
| MACD        | O(n)            | O(n)             | 3 EMAs + histogram |
| Bollinger   | O(n)            | O(n)             | Uses SMA + std dev |
| ATR         | O(n)            | O(n)             | True Range + EMA |
| Stochastic  | O(n)            | O(n)             | Rolling extrema |
| ADX         | O(n)            | O(n)             | +DI/-DI + Wilder |
| Williams %R | O(n)            | O(n)             | Rolling extrema |
| Donchian    | O(n)            | O(n)             | Monotonic deque |
| OBV         | O(n)            | O(n)             | Cumulative |
| VWAP        | O(n)            | O(n)             | Cumulative |

## Reproducing Results

```bash
# Full benchmark suite
cargo bench -p fast-ta

# Specific indicator
cargo bench -p fast-ta -- sma

# Quick verification (test mode)
cargo bench -p fast-ta -- --test
```

## Regression Detection

Compare against this baseline using:

```bash
cargo bench -p fast-ta -- --save-baseline baseline
# ... make changes ...
cargo bench -p fast-ta -- --baseline baseline
```

Criterion will report any significant regressions (>5% slower).
