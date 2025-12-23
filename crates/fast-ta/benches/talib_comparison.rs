//! Performance comparison between fast-ta and TA-Lib (via FFI).
//!
//! Run with: `cargo bench -p fast-ta --bench talib_comparison`
//!
//! Requires TA-Lib to be installed on the system.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// TA-Lib FFI bindings
use ta_lib_sys::{
    ATR, BBANDS, EMA, MACD, RSI, SMA, STOCH,
    MAType,
};

// fast-ta indicators
use fast_ta::indicators::{
    atr::atr,
    bollinger::bollinger,
    ema::ema,
    macd::macd,
    rsi::rsi,
    sma::sma,
    stochastic::stochastic,
};

/// Generate synthetic price data for benchmarks.
fn generate_close_prices(size: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(size);
    let mut price = 100.0;
    for i in 0..size {
        let delta = ((i as f64 * 0.1).sin() * 2.0) + ((i as f64 * 0.03).cos() * 1.5);
        price += delta;
        price = price.max(10.0);
        data.push(price);
    }
    data
}

/// Generate OHLC data for benchmarks.
fn generate_ohlcv(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);
    let mut open = Vec::with_capacity(size);

    let mut price = 100.0;
    for i in 0..size {
        let delta = ((i as f64 * 0.1).sin() * 2.0) + ((i as f64 * 0.03).cos() * 1.5);
        price += delta;
        price = price.max(10.0);

        let h = price + 1.0 + (i as f64 * 0.07).sin().abs();
        let l = price - 1.0 - (i as f64 * 0.05).cos().abs();
        let c = price + ((i as f64 * 0.02).tan() * 0.5).clamp(-0.8, 0.8);
        let o = price + ((i as f64 * 0.04).sin() * 0.3);

        high.push(h);
        low.push(l);
        close.push(c);
        open.push(o);
    }

    (open, high, low, close)
}

const SIZES: &[usize] = &[1_000, 10_000, 100_000];

// =============================================================================
// SMA Comparison
// =============================================================================

fn bench_sma_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma_comparison");
    let period: i32 = 20;

    for &size in SIZES {
        let data = generate_close_prices(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(
            BenchmarkId::new("fast-ta", size),
            &data,
            |b, data| b.iter(|| sma(black_box(data), black_box(period as usize))),
        );

        // TA-Lib
        group.bench_with_input(BenchmarkId::new("ta-lib", size), &data, |b, data| {
            b.iter(|| {
                let mut out_begin: i32 = 0;
                let mut out_nb_element: i32 = 0;
                let mut output = vec![0.0f64; data.len()];

                unsafe {
                    SMA(
                        0,
                        (data.len() - 1) as i32,
                        data.as_ptr(),
                        period,
                        &mut out_begin,
                        &mut out_nb_element,
                        output.as_mut_ptr(),
                    );
                }
                black_box(output)
            })
        });
    }
    group.finish();
}

// =============================================================================
// EMA Comparison
// =============================================================================

fn bench_ema_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema_comparison");
    let period: i32 = 20;

    for &size in SIZES {
        let data = generate_close_prices(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(
            BenchmarkId::new("fast-ta", size),
            &data,
            |b, data| b.iter(|| ema(black_box(data), black_box(period as usize))),
        );

        // TA-Lib
        group.bench_with_input(BenchmarkId::new("ta-lib", size), &data, |b, data| {
            b.iter(|| {
                let mut out_begin: i32 = 0;
                let mut out_nb_element: i32 = 0;
                let mut output = vec![0.0f64; data.len()];

                unsafe {
                    EMA(
                        0,
                        (data.len() - 1) as i32,
                        data.as_ptr(),
                        period,
                        &mut out_begin,
                        &mut out_nb_element,
                        output.as_mut_ptr(),
                    );
                }
                black_box(output)
            })
        });
    }
    group.finish();
}

// =============================================================================
// RSI Comparison
// =============================================================================

fn bench_rsi_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsi_comparison");
    let period: i32 = 14;

    for &size in SIZES {
        let data = generate_close_prices(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(
            BenchmarkId::new("fast-ta", size),
            &data,
            |b, data| b.iter(|| rsi(black_box(data), black_box(period as usize))),
        );

        // TA-Lib
        group.bench_with_input(BenchmarkId::new("ta-lib", size), &data, |b, data| {
            b.iter(|| {
                let mut out_begin: i32 = 0;
                let mut out_nb_element: i32 = 0;
                let mut output = vec![0.0f64; data.len()];

                unsafe {
                    RSI(
                        0,
                        (data.len() - 1) as i32,
                        data.as_ptr(),
                        period,
                        &mut out_begin,
                        &mut out_nb_element,
                        output.as_mut_ptr(),
                    );
                }
                black_box(output)
            })
        });
    }
    group.finish();
}

// =============================================================================
// MACD Comparison
// =============================================================================

fn bench_macd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("macd_comparison");
    let fast_period: i32 = 12;
    let slow_period: i32 = 26;
    let signal_period: i32 = 9;

    for &size in SIZES {
        let data = generate_close_prices(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(BenchmarkId::new("fast-ta", size), &data, |b, data| {
            b.iter(|| {
                macd(
                    black_box(data),
                    black_box(fast_period as usize),
                    black_box(slow_period as usize),
                    black_box(signal_period as usize),
                )
            })
        });

        // TA-Lib
        group.bench_with_input(BenchmarkId::new("ta-lib", size), &data, |b, data| {
            b.iter(|| {
                let mut out_begin: i32 = 0;
                let mut out_nb_element: i32 = 0;
                let mut macd_out = vec![0.0f64; data.len()];
                let mut signal_out = vec![0.0f64; data.len()];
                let mut hist_out = vec![0.0f64; data.len()];

                unsafe {
                    MACD(
                        0,
                        (data.len() - 1) as i32,
                        data.as_ptr(),
                        fast_period,
                        slow_period,
                        signal_period,
                        &mut out_begin,
                        &mut out_nb_element,
                        macd_out.as_mut_ptr(),
                        signal_out.as_mut_ptr(),
                        hist_out.as_mut_ptr(),
                    );
                }
                black_box((macd_out, signal_out, hist_out))
            })
        });
    }
    group.finish();
}

// =============================================================================
// ATR Comparison
// =============================================================================

fn bench_atr_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("atr_comparison");
    let period: i32 = 14;

    for &size in SIZES {
        let (_, high, low, close) = generate_ohlcv(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(
            BenchmarkId::new("fast-ta", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| atr(black_box(*h), black_box(*l), black_box(*c), black_box(period as usize)))
            },
        );

        // TA-Lib
        group.bench_with_input(
            BenchmarkId::new("ta-lib", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| {
                    let mut out_begin: i32 = 0;
                    let mut out_nb_element: i32 = 0;
                    let mut output = vec![0.0f64; h.len()];

                    unsafe {
                        ATR(
                            0,
                            (h.len() - 1) as i32,
                            h.as_ptr(),
                            l.as_ptr(),
                            c.as_ptr(),
                            period,
                            &mut out_begin,
                            &mut out_nb_element,
                            output.as_mut_ptr(),
                        );
                    }
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// Bollinger Bands Comparison
// =============================================================================

fn bench_bollinger_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bollinger_comparison");
    let period: i32 = 20;
    let num_std: f64 = 2.0;

    for &size in SIZES {
        let data = generate_close_prices(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(BenchmarkId::new("fast-ta", size), &data, |b, data| {
            b.iter(|| bollinger(black_box(data), black_box(period as usize), black_box(num_std)))
        });

        // TA-Lib
        group.bench_with_input(BenchmarkId::new("ta-lib", size), &data, |b, data| {
            b.iter(|| {
                let mut out_begin: i32 = 0;
                let mut out_nb_element: i32 = 0;
                let mut upper = vec![0.0f64; data.len()];
                let mut middle = vec![0.0f64; data.len()];
                let mut lower = vec![0.0f64; data.len()];

                unsafe {
                    BBANDS(
                        0,
                        (data.len() - 1) as i32,
                        data.as_ptr(),
                        period,
                        num_std,
                        num_std,
                        MAType::MAType_SMA,
                        &mut out_begin,
                        &mut out_nb_element,
                        upper.as_mut_ptr(),
                        middle.as_mut_ptr(),
                        lower.as_mut_ptr(),
                    );
                }
                black_box((upper, middle, lower))
            })
        });
    }
    group.finish();
}

// =============================================================================
// Stochastic Comparison
// =============================================================================

fn bench_stochastic_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("stochastic_comparison");
    let k_period: i32 = 14;
    let d_period: i32 = 3;
    let k_slowing: i32 = 3;

    for &size in SIZES {
        let (_, high, low, close) = generate_ohlcv(size);
        group.throughput(Throughput::Elements(size as u64));

        // fast-ta
        group.bench_with_input(
            BenchmarkId::new("fast-ta", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| {
                    stochastic(
                        black_box(*h),
                        black_box(*l),
                        black_box(*c),
                        black_box(k_period as usize),
                        black_box(d_period as usize),
                        black_box(k_slowing as usize),
                    )
                })
            },
        );

        // TA-Lib
        group.bench_with_input(
            BenchmarkId::new("ta-lib", size),
            &(&high, &low, &close),
            |b, (h, l, c)| {
                b.iter(|| {
                    let mut out_begin: i32 = 0;
                    let mut out_nb_element: i32 = 0;
                    let mut k_out = vec![0.0f64; h.len()];
                    let mut d_out = vec![0.0f64; h.len()];

                    unsafe {
                        STOCH(
                            0,
                            (h.len() - 1) as i32,
                            h.as_ptr(),
                            l.as_ptr(),
                            c.as_ptr(),
                            k_period,
                            k_slowing,
                            MAType::MAType_SMA,
                            d_period,
                            MAType::MAType_SMA,
                            &mut out_begin,
                            &mut out_nb_element,
                            k_out.as_mut_ptr(),
                            d_out.as_mut_ptr(),
                        );
                    }
                    black_box((k_out, d_out))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sma_comparison,
    bench_ema_comparison,
    bench_rsi_comparison,
    bench_macd_comparison,
    bench_atr_comparison,
    bench_bollinger_comparison,
    bench_stochastic_comparison,
);

criterion_main!(benches);
