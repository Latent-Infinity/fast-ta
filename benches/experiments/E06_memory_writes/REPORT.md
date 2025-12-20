# E06: Memory Write Pattern Benchmarks

## Experiment Overview

**Experiment ID**: E06
**Name**: Memory Write Pattern Benchmarks
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

## Objective

Determine the optimal memory write pattern for indicator computation by comparing:

1. **Write-every-bar**: Standard approach writing to output array on every iteration
2. **Buffered writes**: Accumulate values in registers/local buffers, write periodically
3. **Chunked processing**: Process data in cache-friendly blocks
4. **Multi-output patterns**: Interleaved vs sequential writes to multiple arrays

### Hypothesis

Write-every-bar may cause performance issues when:

1. **Cache line ping-pong**: Multiple indicators writing to different arrays cause L1 cache evictions
2. **Memory bandwidth saturation**: High-frequency writes to main memory limit throughput
3. **Store buffer pressure**: CPU store buffers fill up with pending writes

Buffered writes may improve performance by:

1. **Better cache utilization**: Writing in bursts keeps data cache-hot
2. **Reduced memory bandwidth**: Fewer store operations to main memory
3. **Better ILP**: Deferred writes allow more computation to proceed in parallel

### Success Criteria

| Decision | Condition | Recommendation |
|----------|-----------|----------------|
| **GO (Buffered)** | Buffered writes achieve ≥10% speedup | Implement buffering in hot paths |
| **NO-GO (Direct)** | No significant improvement or slowdown | Keep simple write-every-bar pattern |
| **INVESTIGATE** | Mixed results by data size | Use adaptive strategy based on size |

## Approaches Benchmarked

### 1. Write-Every-Bar Patterns

| Variant | Description | Expected Use Case |
|---------|-------------|-------------------|
| Allocating | Fresh vector allocation each call | Simple API, one-shot computation |
| Pre-allocated | Reuse existing buffer | High-frequency computation |
| With capacity | Allocate with known capacity | Reduce reallocation overhead |

### 2. Buffered Write Patterns

| Buffer Size | Cache Level Target | Notes |
|-------------|-------------------|-------|
| 64 elements | L1 cache (512 bytes) | Fits in L1 cache line |
| 256 elements | L1 cache (2KB) | Multiple L1 lines |
| 1024 elements | L2 cache (8KB) | Small L2 footprint |
| 4096 elements | L2 cache (32KB) | Larger L2 block |

### 3. Multi-Output Patterns

| Pattern | Description | Cache Behavior |
|---------|-------------|----------------|
| Sequential | Complete one indicator before next | Full cache for each indicator |
| Interleaved | Write to all outputs in each iteration | Split cache between outputs |
| Parallel outputs | Multiple independent computations | Test memory bandwidth limits |

### 4. Chunked Processing

Process data in cache-friendly chunks to maximize data reuse before eviction.

## Benchmark Configuration

### Data Sizes

| Size | Points | Description | Expected Cache Behavior |
|------|--------|-------------|------------------------|
| 1K | 1,000 | 8KB | Fits in L1 cache |
| 10K | 10,000 | 80KB | Fits in L2 cache |
| 100K | 100,000 | 800KB | Fits in L3 cache |
| 1M | 1,000,000 | 8MB | Exceeds L3, main memory |

### Indicator Periods

- SMA: 20 (standard)
- EMA: 20 (standard)
- Bollinger: 20, 2σ (triple output)

### Parallel Output Counts

- 1 output (baseline)
- 2 outputs (e.g., SMA + EMA)
- 4 outputs (e.g., multiple SMAs)
- 8 outputs (stress test)

## Results

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e06_memory_writes`*

### Write-Every-Bar: Allocating vs Pre-allocated

| Indicator | Size | Allocating | Pre-allocated | Speedup |
|-----------|------|------------|---------------|---------|
| SMA | 1K | TBD ns | TBD ns | TBD% |
| SMA | 10K | TBD ns | TBD ns | TBD% |
| SMA | 100K | TBD ns | TBD ns | TBD% |
| EMA | 1K | TBD ns | TBD ns | TBD% |
| EMA | 10K | TBD ns | TBD ns | TBD% |
| EMA | 100K | TBD ns | TBD ns | TBD% |
| Bollinger | 1K | TBD ns | TBD ns | TBD% |
| Bollinger | 10K | TBD ns | TBD ns | TBD% |
| Bollinger | 100K | TBD ns | TBD ns | TBD% |

### Buffered vs Direct Writes

| Buffer Size | Time | vs Direct | Notes |
|-------------|------|-----------|-------|
| No buffer | TBD μs | baseline | |
| 64 elements | TBD μs | TBD% | L1 cache size |
| 256 elements | TBD μs | TBD% | |
| 1024 elements | TBD μs | TBD% | |
| 4096 elements | TBD μs | TBD% | |

### Multi-Output: Sequential vs Interleaved

| Pattern | 4 Outputs Time | Per-Output Overhead |
|---------|----------------|---------------------|
| Sequential | TBD μs | TBD |
| Interleaved | TBD μs | TBD |
| **Speedup** | **TBD%** | |

### Parallel Output Scaling (100K data points)

| Output Count | Total Time | Time per Output | Scaling Factor |
|--------------|------------|-----------------|----------------|
| 1 | TBD μs | TBD μs | 1.0× |
| 2 | TBD μs | TBD μs | TBD× |
| 4 | TBD μs | TBD μs | TBD× |
| 8 | TBD μs | TBD μs | TBD× |

### Chunked Processing

| Chunk Size | Time | vs Unchunked |
|------------|------|--------------|
| Unchunked | TBD μs | baseline |
| 64 | TBD μs | TBD% |
| 256 | TBD μs | TBD% |
| 1024 | TBD μs | TBD% |
| 4096 | TBD μs | TBD% |

### Memory Access Patterns (100K data points)

| Pattern | Time | vs Optimal |
|---------|------|------------|
| Sequential read, sequential write | TBD μs | baseline |
| Sequential read, strided write | TBD μs | TBD% slower |
| Multi-pass sequential | TBD μs | TBD |
| Single-pass multi-output | TBD μs | TBD |

### Throughput Analysis

| Configuration | 10K (elem/s) | 100K (elem/s) | 1M (elem/s) |
|---------------|--------------|---------------|-------------|
| Single output | TBD | TBD | TBD |
| Triple output (Bollinger) | TBD | TBD | TBD |

### Allocation Overhead

| Size | Fresh Alloc | Reused Alloc | With Capacity | Alloc Cost |
|------|-------------|--------------|---------------|------------|
| 1K | TBD μs | TBD μs | TBD μs | TBD% |
| 10K | TBD μs | TBD μs | TBD μs | TBD% |
| 100K | TBD μs | TBD μs | TBD μs | TBD% |

## Analysis

### Expected Results

Based on cache architecture analysis:

#### L1 Cache Behavior (32KB typical)
- **1K elements** (8KB): Entire working set fits in L1
- Buffering overhead likely exceeds benefit
- Pre-allocation more important than write pattern

#### L2 Cache Behavior (256KB-1MB typical)
- **10K elements** (80KB): Input fits in L2
- Output may evict input during write-every-bar
- Buffering could help maintain input in L2

#### L3 Cache Behavior (8-32MB typical)
- **100K elements** (800KB): Everything fits in L3
- Memory bandwidth not a bottleneck
- Write pattern matters less

#### Main Memory Behavior
- **1M elements** (8MB): Exceeds typical L3
- Memory bandwidth becomes critical
- Buffering and prefetching become important

### Cache Line Considerations

- **Cache line size**: 64 bytes (8 f64 values)
- **Partial line writes**: May cause read-modify-write cycles
- **Write combining**: Modern CPUs combine adjacent writes

### Instruction-Level Parallelism

- **Out-of-order execution**: Stores can proceed independently
- **Store buffer**: Decouples stores from execution
- **Memory-level parallelism**: Multiple outstanding stores

## Go/No-Go Decision

**Decision**: PENDING

### Criteria Checklist

#### For BUFFERED WRITES Recommendation:
- [ ] Buffered writes show ≥10% improvement at 100K+ data points
- [ ] Optimal buffer size identified
- [ ] No significant slowdown at smaller sizes

#### For WRITE-EVERY-BAR Recommendation:
- [ ] Buffering provides <5% improvement
- [ ] Added complexity not justified
- [ ] Compiler already optimizes effectively

#### For ADAPTIVE STRATEGY Recommendation:
- [ ] Clear size threshold where buffering helps
- [ ] Pattern differs between single/multi-output
- [ ] Memory bandwidth is limiting factor

## Implications for fast-ta Architecture

### If BUFFERED WRITES Win:

1. **Internal buffers**: Implement ring buffer for output
2. **Flush strategy**: Flush on chunk boundary or completion
3. **Buffer size tuning**: Auto-detect based on cache size
4. **API consideration**: Hide buffering from public API

### If WRITE-EVERY-BAR Wins:

1. **Keep simple**: Direct array writes are optimal
2. **Pre-allocation API**: Provide `_into` variants
3. **Buffer pools**: Let caller manage buffers
4. **No internal state**: Stateless computation preferred

### Multi-Output Recommendations

Based on interleaved vs sequential results:

| Result | Recommendation |
|--------|---------------|
| Interleaved faster | Fuse related indicators into single loop |
| Sequential faster | Compute indicators independently |
| Equal | Choose based on code simplicity |

## Write Pattern Recommendation

*To be filled after benchmark execution*

### Recommended Default Pattern

```rust
// TBD: Based on benchmark results, recommend one of:

// Option A: Simple write-every-bar (if buffering doesn't help)
pub fn indicator(data: &[f64], period: usize) -> Result<Vec<f64>> {
    let mut output = vec![f64::NAN; data.len()];
    // ... compute with direct writes
    Ok(output)
}

// Option B: Buffered writes (if buffering helps)
pub fn indicator(data: &[f64], period: usize) -> Result<Vec<f64>> {
    const BUFFER_SIZE: usize = OPTIMAL_SIZE; // From benchmarks
    let mut buffer = [0.0_f64; BUFFER_SIZE];
    // ... compute with buffered writes
    Ok(output)
}

// Option C: Adaptive (if size-dependent)
pub fn indicator(data: &[f64], period: usize) -> Result<Vec<f64>> {
    if data.len() > THRESHOLD {
        indicator_buffered(data, period)
    } else {
        indicator_direct(data, period)
    }
}
```

### API Recommendations

| Scenario | Recommended API |
|----------|----------------|
| Single indicator, one-shot | `indicator(data, period)` |
| Single indicator, repeated | `indicator_into(data, period, &mut output)` |
| Multiple indicators | `compute_batch(&data, &specs, &mut outputs)` |
| Streaming data | `Indicator::new(period).update(value)` |

## Comparison with Other Experiments

| Experiment | Focus | Relationship to E06 |
|------------|-------|---------------------|
| E01 Baseline | Raw indicator cost | Baseline for comparison |
| E02 RunningStat | Fusion benefit | Single pass reduces writes |
| E03 EMA Fusion | Multi-EMA fusion | Interleaved output writes |
| E04 Rolling Extrema | Algorithm choice | Independent of write pattern |
| E05 Plan Overhead | Infrastructure cost | May affect buffer pool design |
| **E06 Memory Writes** | **Write patterns** | **Core memory strategy** |
| E07 End-to-End | Full comparison | Validates E06 choices |

## Follow-up Actions

After E06 completes:

1. **If buffered writes win**:
   - Implement internal buffer abstraction
   - Add buffer size configuration
   - Profile with perf counters (cache misses)

2. **If write-every-bar wins**:
   - Focus optimization on fusion (E02, E03)
   - Ensure `_into` variants are well-optimized
   - Consider SIMD for write operations

3. **Additional profiling**:
   - Measure L1/L2/L3 cache miss rates
   - Profile store buffer utilization
   - Test on different CPU architectures

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e06_memory_writes.rs`
- **Indicator Implementations**: `crates/fast-ta-core/src/indicators/`
- **Criterion Output**: `target/criterion/e06_memory_writes/`
- **Raw JSON Data**: `target/criterion/e06_memory_writes/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E06 memory writes benchmarks
cargo bench --package fast-ta-experiments --bench e06_memory_writes

# View HTML report
open target/criterion/e06_memory_writes/report/index.html

# Run specific benchmark group
cargo bench --package fast-ta-experiments --bench e06_memory_writes -- "write_every_bar"
cargo bench --package fast-ta-experiments --bench e06_memory_writes -- "buffered_writes"
cargo bench --package fast-ta-experiments --bench e06_memory_writes -- "multi_output"
cargo bench --package fast-ta-experiments --bench e06_memory_writes -- "chunked"
cargo bench --package fast-ta-experiments --bench e06_memory_writes -- "access_patterns"
```

## Technical Notes

### Memory Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU Core                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Registers (few ns latency)                                │  │
│  │ - Accumulator variables (sum, mean, etc.)                 │  │
│  │ - Loop indices                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L1 Data Cache (~4 cycles, 32KB)                           │  │
│  │ - Current input window (period elements)                  │  │
│  │ - Current output chunk (if buffered)                      │  │
│  │ - Cache line: 64 bytes = 8 f64 values                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L2 Cache (~12 cycles, 256KB-1MB)                          │  │
│  │ - Larger input/output chunks                              │  │
│  │ - Prefetched data                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ L3 Cache (~40 cycles, 8-32MB)                             │  │
│  │ - Shared across cores                                     │  │
│  │ - Entire arrays for 100K data                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Main Memory (~100+ cycles)                                │  │
│  │ - 1M+ data points                                         │  │
│  │ - Memory bandwidth: 50-100 GB/s (DDR4/DDR5)               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Write Pattern Trade-offs

| Pattern | Pros | Cons |
|---------|------|------|
| Write-every-bar | Simple, streaming output | May evict input from cache |
| Buffered writes | Better cache utilization | Added complexity, buffer management |
| Chunked processing | Cache-friendly blocks | Chunk boundary handling |

### Compiler Optimizations

Modern compilers (rustc + LLVM) may:

- **Vectorize loops**: Process 4+ elements per instruction (AVX2/AVX-512)
- **Unroll loops**: Reduce branch overhead
- **Prefetch**: Insert prefetch instructions for predictable access
- **Combine stores**: Merge adjacent stores to full cache lines

### Memory Bandwidth Calculation

For 1M f64 elements:
- Input read: 8 MB
- Output write: 8 MB
- Total: 16 MB per indicator

At 50 GB/s bandwidth:
- Minimum time: 16 MB / 50 GB/s = 0.32 ms
- Actual time includes compute, so bandwidth is rarely limiting for single indicators

For 8 parallel outputs:
- Total: 8 MB + 64 MB = 72 MB
- At 50 GB/s: 1.44 ms minimum
- Memory bandwidth becomes more relevant

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
