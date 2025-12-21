//! E06: Memory Write Pattern Benchmarks
//!
//! This experiment compares different memory write strategies for indicator output:
//! - **Write-every-bar**: Standard approach writing to output array on every iteration
//! - **Buffered writes**: Accumulate values in registers/local buffers, write periodically
//!
//! # Hypothesis
//!
//! Write-every-bar may cause cache line ping-pong and memory bus saturation when
//! computing multiple indicators that all write to different output arrays. Buffered
//! writes may improve performance by:
//!
//! 1. **Better cache utilization**: Writing in bursts keeps cache lines hot
//! 2. **Reduced memory bandwidth**: Fewer individual store operations
//! 3. **Better instruction-level parallelism**: Deferred writes allow more computation
//!
//! However, the benefits depend on:
//! - L1/L2 cache sizes and line widths
//! - Memory access patterns of the specific indicator
//! - Compiler optimizations (vectorization, loop unrolling)
//!
//! # Methodology
//!
//! We benchmark:
//!
//! 1. **Single output write patterns**: SMA with write-every-bar vs buffered
//! 2. **Multiple output write patterns**: Multiple indicators writing simultaneously
//! 3. **Chunked processing**: Processing data in cache-friendly chunks
//! 4. **Streaming writes**: Sequential vs strided memory access
//!
//! # Success Criteria
//!
//! - **GO (Use buffered writes)**: Buffered approach achieves ≥10% speedup
//! - **NO-GO (Stick with write-every-bar)**: No significant improvement or slowdown
//! - **INVESTIGATE**: Mixed results depending on data size
//!
//! # Output
//!
//! Results are written to:
//! - `target/criterion/e06_memory_writes/` (Criterion HTML/JSON reports)
//! - `benches/experiments/E06_memory_writes/REPORT.md` (analysis summary)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Include common utilities
mod common;
use common::{
    format_size, measurement_time_for_data_size, sample_size_for_data_size, QUICK_DATA_SIZES,
    DEFAULT_SEED, GROUP_E06_MEMORY_WRITES, BOLLINGER_PERIOD,
};

// Core library imports
use fast_ta_core::indicators::{sma, ema, bollinger};
use fast_ta_core::indicators::sma::sma_into;
use fast_ta_core::indicators::ema::ema_into;
use fast_ta_core::indicators::bollinger::bollinger_into;

// Data generators
use fast_ta_experiments::data::generate_random_walk;

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Standard period for write pattern benchmarks
const SMA_PERIOD: usize = 20;
const EMA_PERIOD: usize = 20;

/// Chunk sizes to test for chunked processing
const CHUNK_SIZES: [usize; 4] = [64, 256, 1024, 4096];

/// Number of parallel outputs for multi-output tests
const PARALLEL_OUTPUT_COUNTS: [usize; 4] = [1, 2, 4, 8];

// ============================================================================
// Data Preparation
// ============================================================================

/// Pre-generated price data for benchmarks.
struct BenchmarkData {
    prices: Vec<f64>,
}

impl BenchmarkData {
    fn new(size: usize, seed: u64) -> Self {
        Self {
            prices: generate_random_walk(size, seed),
        }
    }
}

// ============================================================================
// Write-Every-Bar Pattern (Baseline)
// ============================================================================

/// Standard write-every-bar approach - allocates new output vector.
/// This is the baseline for comparison.
fn bench_write_every_bar_allocating(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/write_every_bar/allocating", GROUP_E06_MEMORY_WRITES));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        group.bench_with_input(
            BenchmarkId::new("sma", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(sma(black_box(&data.prices), SMA_PERIOD).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ema", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(ema(black_box(&data.prices), EMA_PERIOD).unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bollinger", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| black_box(bollinger(black_box(&data.prices), BOLLINGER_PERIOD, 2.0).unwrap()))
            },
        );
    }

    group.finish();
}

/// Write-every-bar with pre-allocated buffer.
/// Tests if allocation overhead dominates.
fn bench_write_every_bar_preallocated(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/write_every_bar/preallocated", GROUP_E06_MEMORY_WRITES));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Pre-allocated SMA
        group.bench_with_input(
            BenchmarkId::new("sma", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    black_box(sma_into(black_box(&data.prices), SMA_PERIOD, &mut output).unwrap())
                })
            },
        );

        // Pre-allocated EMA
        group.bench_with_input(
            BenchmarkId::new("ema", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    black_box(ema_into(black_box(&data.prices), EMA_PERIOD, &mut output).unwrap())
                })
            },
        );

        // Pre-allocated Bollinger
        group.bench_with_input(
            BenchmarkId::new("bollinger", format_size(size)),
            &data,
            |b, data| {
                use fast_ta_core::indicators::bollinger::BollingerOutput;
                let mut output = BollingerOutput {
                    middle: vec![0.0_f64; size],
                    upper: vec![0.0_f64; size],
                    lower: vec![0.0_f64; size],
                };
                b.iter(|| {
                    black_box(bollinger_into(black_box(&data.prices), BOLLINGER_PERIOD, 2.0, &mut output).unwrap())
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Buffered Write Pattern
// ============================================================================

/// Manually buffered write - accumulate in small buffer, write in bursts.
/// Simulates what we might do with explicit buffering.
fn bench_buffered_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/buffered_writes", GROUP_E06_MEMORY_WRITES));

    let size = 100_000; // Fixed size for buffer comparison
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &buffer_size in &CHUNK_SIZES {
        group.bench_with_input(
            BenchmarkId::new("buffered_sma", format!("buf_{}", buffer_size)),
            &(&data, buffer_size),
            |b, &(data, buf_size)| {
                let mut output = vec![0.0_f64; size];
                let mut buffer = vec![0.0_f64; buf_size];

                b.iter(|| {
                    // Simulated buffered SMA: compute into small buffer, then copy
                    let period = SMA_PERIOD;
                    let period_f = period as f64;
                    let mut sum = 0.0_f64;

                    // Initialize first period values
                    for i in 0..period {
                        sum += data.prices[i];
                        output[i] = f64::NAN;
                    }
                    output[period - 1] = sum / period_f;

                    // Process in chunks
                    let mut buf_idx = 0;
                    for i in period..size {
                        sum = sum + data.prices[i] - data.prices[i - period];
                        buffer[buf_idx] = sum / period_f;
                        buf_idx += 1;

                        if buf_idx == buf_size {
                            // Flush buffer to output
                            let start = i - buf_size + 1;
                            output[start..=i].copy_from_slice(&buffer[..buf_size]);
                            buf_idx = 0;
                        }
                    }

                    // Flush remaining
                    if buf_idx > 0 {
                        let start = size - buf_idx;
                        output[start..].copy_from_slice(&buffer[..buf_idx]);
                    }

                    black_box(output[size - 1])
                })
            },
        );
    }

    // Baseline: direct write (no buffering)
    group.bench_with_input(
        BenchmarkId::new("direct_sma", "no_buffer"),
        &data,
        |b, data| {
            let mut output = vec![0.0_f64; size];
            b.iter(|| {
                let period = SMA_PERIOD;
                let period_f = period as f64;
                let mut sum = 0.0_f64;

                for i in 0..period {
                    sum += data.prices[i];
                    output[i] = f64::NAN;
                }
                output[period - 1] = sum / period_f;

                for i in period..size {
                    sum = sum + data.prices[i] - data.prices[i - period];
                    output[i] = sum / period_f;
                }

                black_box(output[size - 1])
            })
        },
    );

    group.finish();
}

// ============================================================================
// Multiple Output Streams
// ============================================================================

/// Benchmark multiple indicators writing to different output arrays.
/// Tests memory bandwidth saturation with multiple write streams.
fn bench_multi_output_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/multi_output", GROUP_E06_MEMORY_WRITES));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        for &output_count in &PARALLEL_OUTPUT_COUNTS {
            group.bench_with_input(
                BenchmarkId::new("parallel_writes", format!("{}_outputs_{}", format_size(size), output_count)),
                &(&data, output_count),
                |b, &(data, count)| {
                    // Pre-allocate output buffers
                    let mut outputs: Vec<Vec<f64>> = (0..count)
                        .map(|_| vec![0.0_f64; size])
                        .collect();

                    b.iter(|| {
                        // Compute multiple SMAs with different periods
                        for (i, output) in outputs.iter_mut().enumerate() {
                            let period = 10 + i * 5; // Periods: 10, 15, 20, 25, ...
                            let _ = sma_into(&data.prices, period, output);
                        }
                        black_box(outputs.last().map(|v| v.last().copied()).flatten())
                    })
                },
            );
        }
    }

    group.finish();
}

/// Interleaved multi-output vs sequential multi-output.
/// Tests if interleaving writes to different buffers is better or worse.
fn bench_interleaved_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/interleaved_vs_sequential", GROUP_E06_MEMORY_WRITES));

    let size = 100_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // Sequential: complete each indicator before starting next
    group.bench_with_input(
        BenchmarkId::new("sequential", "4_outputs"),
        &data,
        |b, data| {
            let mut output1 = vec![0.0_f64; size];
            let mut output2 = vec![0.0_f64; size];
            let mut output3 = vec![0.0_f64; size];
            let mut output4 = vec![0.0_f64; size];

            b.iter(|| {
                // Complete indicator 1
                let _ = sma_into(&data.prices, 10, &mut output1);
                // Complete indicator 2
                let _ = sma_into(&data.prices, 15, &mut output2);
                // Complete indicator 3
                let _ = sma_into(&data.prices, 20, &mut output3);
                // Complete indicator 4
                let _ = sma_into(&data.prices, 25, &mut output4);

                black_box((output1[size - 1], output2[size - 1], output3[size - 1], output4[size - 1]))
            })
        },
    );

    // Interleaved: write one value to each buffer in a loop
    group.bench_with_input(
        BenchmarkId::new("interleaved", "4_outputs"),
        &data,
        |b, data| {
            let mut output1 = vec![0.0_f64; size];
            let mut output2 = vec![0.0_f64; size];
            let mut output3 = vec![0.0_f64; size];
            let mut output4 = vec![0.0_f64; size];

            b.iter(|| {
                // Initialize sums
                let mut sum1 = 0.0_f64;
                let mut sum2 = 0.0_f64;
                let mut sum3 = 0.0_f64;
                let mut sum4 = 0.0_f64;

                // Initial sums for each period
                for i in 0..25 {
                    if i < 10 { sum1 += data.prices[i]; }
                    if i < 15 { sum2 += data.prices[i]; }
                    if i < 20 { sum3 += data.prices[i]; }
                    sum4 += data.prices[i];

                    // Write NaN for lookback period
                    if i < 9 { output1[i] = f64::NAN; }
                    if i < 14 { output2[i] = f64::NAN; }
                    if i < 19 { output3[i] = f64::NAN; }
                    if i < 24 { output4[i] = f64::NAN; }
                }

                // Write first valid values
                output1[9] = sum1 / 10.0;
                output2[14] = sum2 / 15.0;
                output3[19] = sum3 / 20.0;
                output4[24] = sum4 / 25.0;

                // Interleaved rolling computation
                for i in 25..size {
                    // Update all sums
                    sum1 = sum1 + data.prices[i] - data.prices[i - 10];
                    sum2 = sum2 + data.prices[i] - data.prices[i - 15];
                    sum3 = sum3 + data.prices[i] - data.prices[i - 20];
                    sum4 = sum4 + data.prices[i] - data.prices[i - 25];

                    // Write to all outputs (interleaved)
                    output1[i] = sum1 / 10.0;
                    output2[i] = sum2 / 15.0;
                    output3[i] = sum3 / 20.0;
                    output4[i] = sum4 / 25.0;
                }

                black_box((output1[size - 1], output2[size - 1], output3[size - 1], output4[size - 1]))
            })
        },
    );

    group.finish();
}

// ============================================================================
// Chunked Processing
// ============================================================================

/// Benchmark chunked processing where data is processed in cache-friendly blocks.
fn bench_chunked_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/chunked_processing", GROUP_E06_MEMORY_WRITES));

    let size = 100_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    for &chunk_size in &CHUNK_SIZES {
        group.bench_with_input(
            BenchmarkId::new("chunked_sma", format!("chunk_{}", chunk_size)),
            &(&data, chunk_size),
            |b, &(data, chunk)| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    chunked_sma(&data.prices, SMA_PERIOD, chunk, &mut output);
                    black_box(output[size - 1])
                })
            },
        );
    }

    // Baseline: no chunking
    group.bench_with_input(
        BenchmarkId::new("unchunked_sma", "full"),
        &data,
        |b, data| {
            let mut output = vec![0.0_f64; size];
            b.iter(|| {
                let _ = sma_into(&data.prices, SMA_PERIOD, &mut output);
                black_box(output[size - 1])
            })
        },
    );

    group.finish();
}

/// Helper function for chunked SMA computation.
fn chunked_sma(data: &[f64], period: usize, chunk_size: usize, output: &mut [f64]) {
    let n = data.len();
    let period_f = period as f64;

    // Handle initial period
    let mut sum = 0.0_f64;
    for i in 0..period.min(n) {
        sum += data[i];
        if i < period - 1 {
            output[i] = f64::NAN;
        }
    }
    if period <= n {
        output[period - 1] = sum / period_f;
    }

    // Process in chunks
    let mut i = period;
    while i < n {
        let chunk_end = (i + chunk_size).min(n);

        // Process this chunk
        for j in i..chunk_end {
            sum = sum + data[j] - data[j - period];
            output[j] = sum / period_f;
        }

        i = chunk_end;
    }
}

// ============================================================================
// Memory Access Pattern Analysis
// ============================================================================

/// Compare sequential vs strided memory access patterns.
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/access_patterns", GROUP_E06_MEMORY_WRITES));

    let size = 100_000;
    let data = BenchmarkData::new(size, DEFAULT_SEED);

    group
        .sample_size(sample_size_for_data_size(size))
        .measurement_time(measurement_time_for_data_size(size));

    // Sequential read, sequential write (optimal)
    group.bench_with_input(
        BenchmarkId::new("seq_read_seq_write", format_size(size)),
        &data,
        |b, data| {
            let mut output = vec![0.0_f64; size];
            b.iter(|| {
                for i in 0..size {
                    output[i] = data.prices[i] * 2.0;
                }
                black_box(output[size - 1])
            })
        },
    );

    // Sequential read, strided write (poor cache utilization for writes)
    let stride = 64; // Cache line size in f64s (64 * 8 = 512 bytes)
    group.bench_with_input(
        BenchmarkId::new("seq_read_strided_write", format!("stride_{}", stride)),
        &data,
        |b, data| {
            let mut output = vec![0.0_f64; size];
            b.iter(|| {
                // Write in strided pattern
                for s in 0..stride {
                    let mut i = s;
                    let mut j = 0;
                    while i < size {
                        output[i] = data.prices[j] * 2.0;
                        i += stride;
                        j += 1;
                    }
                }
                black_box(output[size - 1])
            })
        },
    );

    // Multiple passes: simulate write-every-bar with cache pollution
    group.bench_with_input(
        BenchmarkId::new("multi_pass_sequential", "3_passes"),
        &data,
        |b, data| {
            let mut output1 = vec![0.0_f64; size];
            let mut output2 = vec![0.0_f64; size];
            let mut output3 = vec![0.0_f64; size];
            b.iter(|| {
                // Pass 1
                for i in 0..size {
                    output1[i] = data.prices[i] * 2.0;
                }
                // Pass 2
                for i in 0..size {
                    output2[i] = data.prices[i] * 3.0;
                }
                // Pass 3
                for i in 0..size {
                    output3[i] = data.prices[i] * 4.0;
                }
                black_box((output1[size - 1], output2[size - 1], output3[size - 1]))
            })
        },
    );

    // Single pass with multiple outputs
    group.bench_with_input(
        BenchmarkId::new("single_pass_multi_output", "3_outputs"),
        &data,
        |b, data| {
            let mut output1 = vec![0.0_f64; size];
            let mut output2 = vec![0.0_f64; size];
            let mut output3 = vec![0.0_f64; size];
            b.iter(|| {
                for i in 0..size {
                    let val = data.prices[i];
                    output1[i] = val * 2.0;
                    output2[i] = val * 3.0;
                    output3[i] = val * 4.0;
                }
                black_box((output1[size - 1], output2[size - 1], output3[size - 1]))
            })
        },
    );

    group.finish();
}

// ============================================================================
// Throughput Analysis
// ============================================================================

/// Measure write throughput in elements per second.
fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/throughput", GROUP_E06_MEMORY_WRITES));

    let throughput_sizes = [10_000, 100_000, 1_000_000];

    for &size in &throughput_sizes {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size))
            .throughput(criterion::Throughput::Elements(size as u64));

        // Single output
        group.bench_with_input(
            BenchmarkId::new("single_output", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    let _ = sma_into(&data.prices, SMA_PERIOD, &mut output);
                    black_box(output[size - 1])
                })
            },
        );

        // Multiple outputs (Bollinger = 3 outputs)
        group.bench_with_input(
            BenchmarkId::new("triple_output", format_size(size)),
            &data,
            |b, data| {
                use fast_ta_core::indicators::bollinger::BollingerOutput;
                let mut output = BollingerOutput {
                    middle: vec![0.0_f64; size],
                    upper: vec![0.0_f64; size],
                    lower: vec![0.0_f64; size],
                };
                b.iter(|| {
                    let _ = bollinger_into(&data.prices, BOLLINGER_PERIOD, 2.0, &mut output);
                    black_box((output.middle[size - 1], output.upper[size - 1], output.lower[size - 1]))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Allocation vs Pre-allocation Comparison
// ============================================================================

/// Compare allocation overhead across different scenarios.
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("{}/allocation_overhead", GROUP_E06_MEMORY_WRITES));

    for &size in &QUICK_DATA_SIZES {
        let data = BenchmarkData::new(size, DEFAULT_SEED);

        group
            .sample_size(sample_size_for_data_size(size))
            .measurement_time(measurement_time_for_data_size(size));

        // Fresh allocation each iteration
        group.bench_with_input(
            BenchmarkId::new("fresh_alloc", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = sma(&data.prices, SMA_PERIOD).unwrap();
                    black_box(result)
                })
            },
        );

        // Reused allocation (simulates buffer pool)
        group.bench_with_input(
            BenchmarkId::new("reused_alloc", format_size(size)),
            &data,
            |b, data| {
                let mut output = vec![0.0_f64; size];
                b.iter(|| {
                    let _ = sma_into(&data.prices, SMA_PERIOD, &mut output);
                    black_box(output[size - 1])
                })
            },
        );

        // Vec with capacity (no reallocation, but still allocation)
        group.bench_with_input(
            BenchmarkId::new("with_capacity", format_size(size)),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut output = Vec::with_capacity(size);
                    output.resize(size, 0.0_f64);
                    let _ = sma_into(&data.prices, SMA_PERIOD, &mut output);
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = memory_writes_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .with_plots();
    targets =
        bench_write_every_bar_allocating,
        bench_write_every_bar_preallocated,
        bench_buffered_writes,
        bench_multi_output_writes,
        bench_interleaved_vs_sequential,
        bench_chunked_processing,
        bench_memory_access_patterns,
        bench_write_throughput,
        bench_allocation_overhead
);

criterion_main!(memory_writes_benches);
