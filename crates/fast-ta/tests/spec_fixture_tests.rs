//! Spec fixture tests - authoritative source of truth for indicator behavior.
//!
//! These tests verify indicator behavior against hand-constructed test fixtures
//! derived from the PRD specification. They are the authoritative source of truth.

use fast_ta::indicators::{
    atr::{atr, atr_lookback, atr_min_len, true_range, true_range_lookback},
    bollinger::{bollinger, bollinger_lookback, bollinger_min_len},
    ema::{ema, ema_lookback, ema_min_len},
    macd::{macd, macd_line_lookback, macd_min_len, macd_signal_lookback},
    rsi::{rsi, rsi_lookback, rsi_min_len},
    sma::{sma, sma_lookback, sma_min_len},
    stochastic::{stochastic_d_lookback, stochastic_fast, stochastic_k_lookback, stochastic_min_len},
};
use fast_ta::kernels::rolling_extrema::{rolling_extrema_lookback, rolling_extrema_min_len};

const EPSILON: f64 = 1e-10;
const LOOSE_EPSILON: f64 = 1e-6;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() < eps
}

// ==================== EMA SMA Seed Tests ====================
// Spec: spec_ema_sma_seed.json
// PRD §4.7: EMA initialization uses SMA of first period values as seed

#[test]
fn spec_ema_sma_seed_period_3_basic() {
    let input = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = ema(&input, 3).unwrap();

    // First 2 values are NaN
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());

    // First valid EMA = SMA seed = (1+2+3)/3 = 2.0
    assert!(
        approx_eq(result[2], 2.0, EPSILON),
        "EMA seed should be SMA = 2.0, got {}",
        result[2]
    );
}

#[test]
fn spec_ema_sma_seed_constant_values() {
    let input = vec![10.0_f64; 7];
    let result = ema(&input, 5).unwrap();

    // First 4 values are NaN
    for i in 0..4 {
        assert!(result[i].is_nan());
    }

    // Constant input: EMA = SMA = 10.0
    for i in 4..7 {
        assert!(
            approx_eq(result[i], 10.0, EPSILON),
            "Constant EMA should be 10.0, got {} at index {}",
            result[i],
            i
        );
    }
}

// ==================== RSI Extreme Tests ====================
// Spec: spec_rsi_extremes.json
// PRD §4.6: All gains → RSI = 100, All losses → RSI = 0, No movement → RSI = 50

#[test]
fn spec_rsi_all_gains_100() {
    let input = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0, 15.0];
    let result = rsi(&input, 3).unwrap();

    // First 3 values are NaN (period for RSI)
    for i in 0..3 {
        assert!(result[i].is_nan());
    }

    // All gains → RSI = 100
    for i in 3..6 {
        assert!(
            approx_eq(result[i], 100.0, LOOSE_EPSILON),
            "All gains RSI should be 100, got {} at index {}",
            result[i],
            i
        );
    }
}

#[test]
fn spec_rsi_all_losses_0() {
    let input = vec![15.0_f64, 14.0, 13.0, 12.0, 11.0, 10.0];
    let result = rsi(&input, 3).unwrap();

    // All losses → RSI = 0
    for i in 3..6 {
        assert!(
            approx_eq(result[i], 0.0, LOOSE_EPSILON),
            "All losses RSI should be 0, got {} at index {}",
            result[i],
            i
        );
    }
}

#[test]
fn spec_rsi_no_movement_50() {
    let input = vec![50.0_f64; 6];
    let result = rsi(&input, 3).unwrap();

    // No movement → RSI = 50 (neutral)
    for i in 3..6 {
        assert!(
            approx_eq(result[i], 50.0, LOOSE_EPSILON),
            "No movement RSI should be 50, got {} at index {}",
            result[i],
            i
        );
    }
}

// ==================== ATR Gap Tests ====================
// Spec: spec_atr_gap.json
// PRD §4.6: TR = max(H-L, |H-prev_close|, |L-prev_close|)

#[test]
fn spec_atr_gap_up_included() {
    let high = vec![10.0_f64, 15.0, 16.0];
    let low = vec![9.0_f64, 14.0, 15.0];
    let close = vec![9.5_f64, 14.5, 15.5];

    let tr = true_range(&high, &low, &close).unwrap();

    // First TR is NaN (no previous close)
    assert!(tr[0].is_nan());

    // Gap up: prev_close=9.5, high=15 → |15-9.5|=5.5 > (15-14)=1
    assert!(
        approx_eq(tr[1], 5.5, EPSILON),
        "Gap up TR should be 5.5, got {}",
        tr[1]
    );

    // Normal range: high-low = 16-15 = 1, but |16-14.5|=1.5 is max
    assert!(
        approx_eq(tr[2], 1.5, EPSILON),
        "TR[2] should be 1.5, got {}",
        tr[2]
    );
}

#[test]
fn spec_atr_gap_down_included() {
    let high = vec![20.0_f64, 14.0, 13.0];
    let low = vec![19.0_f64, 12.0, 11.0];
    let close = vec![19.5_f64, 13.0, 12.0];

    let tr = true_range(&high, &low, &close).unwrap();

    // Gap down: prev_close=19.5, low=12 → |12-19.5|=7.5 > (14-12)=2
    assert!(
        approx_eq(tr[1], 7.5, EPSILON),
        "Gap down TR should be 7.5, got {}",
        tr[1]
    );
}

#[test]
fn spec_atr_no_gap_standard_range() {
    let high = vec![50.0_f64, 51.0, 52.0];
    let low = vec![48.0_f64, 49.0, 50.0];
    let close = vec![49.0_f64, 50.0, 51.0];

    let tr = true_range(&high, &low, &close).unwrap();

    // No gap: TR = High - Low = 2.0
    assert!(
        approx_eq(tr[1], 2.0, EPSILON),
        "No gap TR should be 2.0, got {}",
        tr[1]
    );
    assert!(
        approx_eq(tr[2], 2.0, EPSILON),
        "No gap TR should be 2.0, got {}",
        tr[2]
    );
}

// ==================== Bollinger Bands Width Tests ====================
// Spec: spec_bollinger_width.json

#[test]
fn spec_bollinger_constant_price_zero_width() {
    let input = vec![50.0_f64; 5];
    let result = bollinger(&input, 3, 2.0).unwrap();

    // Constant price: stddev = 0, so upper = middle = lower
    for i in 2..5 {
        assert!(
            approx_eq(result.middle[i], 50.0, EPSILON),
            "Middle should be 50.0"
        );
        assert!(
            approx_eq(result.upper[i], 50.0, EPSILON),
            "Upper should equal middle for zero stddev"
        );
        assert!(
            approx_eq(result.lower[i], 50.0, EPSILON),
            "Lower should equal middle for zero stddev"
        );
    }
}

#[test]
fn spec_bollinger_symmetric_bands() {
    let input = vec![20.0_f64, 21.0, 22.0, 21.5, 22.5];
    let result = bollinger(&input, 3, 2.0).unwrap();

    // Bands should be symmetric around the middle
    for i in 2..5 {
        if !result.middle[i].is_nan() {
            let upper_diff = result.upper[i] - result.middle[i];
            let lower_diff = result.middle[i] - result.lower[i];
            assert!(
                approx_eq(upper_diff, lower_diff, EPSILON),
                "Bands should be symmetric at index {}: upper_diff={}, lower_diff={}",
                i,
                upper_diff,
                lower_diff
            );
        }
    }
}

// ==================== Stochastic Boundary Tests ====================
// Spec: spec_stochastic_boundary.json
// PRD §4.6: %K = 100 when close = highest high, %K = 0 when close = lowest low

#[test]
fn spec_stochastic_close_at_high_k100() {
    let high = vec![10.0_f64, 12.0, 15.0, 14.0, 13.0];
    let low = vec![8.0_f64, 9.0, 10.0, 11.0, 10.0];
    let close = vec![9.0_f64, 11.0, 15.0, 13.0, 13.0];

    let result = stochastic_fast(&high, &low, &close, 3, 3).unwrap();

    // When close equals highest high in window, %K = 100
    assert!(
        approx_eq(result.k[2], 100.0, LOOSE_EPSILON),
        "Close at high should give %K = 100, got {}",
        result.k[2]
    );
}

#[test]
fn spec_stochastic_close_at_low_k0() {
    let high = vec![15.0_f64, 14.0, 13.0, 12.0, 11.0];
    let low = vec![10.0_f64, 9.0, 8.0, 7.0, 6.0];
    let close = vec![12.0_f64, 10.0, 8.0, 8.0, 7.0];

    let result = stochastic_fast(&high, &low, &close, 3, 3).unwrap();

    // When close equals lowest low in window, %K = 0
    assert!(
        approx_eq(result.k[2], 0.0, LOOSE_EPSILON),
        "Close at low should give %K = 0, got {}",
        result.k[2]
    );
}

// ==================== Lookback Period Tests ====================
// Spec: spec_lookback.json
// PRD §4.5: Output shape is trimmed, first lookback values are NaN

#[test]
fn spec_lookback_sma() {
    let input: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let period = 5;
    let result = sma(&input, period).unwrap();

    // SMA lookback = period - 1 = 4
    let expected_lookback = period - 1;
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, expected_lookback,
        "SMA should have {} NaN values, got {}",
        expected_lookback, nan_count
    );
}

#[test]
fn spec_lookback_ema() {
    let input: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let period = 5;
    let result = ema(&input, period).unwrap();

    // EMA lookback = period - 1 = 4
    let expected_lookback = period - 1;
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, expected_lookback,
        "EMA should have {} NaN values, got {}",
        expected_lookback, nan_count
    );
}

#[test]
fn spec_lookback_rsi() {
    let input: Vec<f64> = (0..30).map(|x| x as f64).collect();
    let period = 14;
    let result = rsi(&input, period).unwrap();

    // RSI lookback = period = 14
    let expected_lookback = period;
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, expected_lookback,
        "RSI should have {} NaN values, got {}",
        expected_lookback, nan_count
    );
}

#[test]
fn spec_lookback_macd() {
    let input: Vec<f64> = (0..50).map(|x| x as f64).collect();
    let result = macd(&input, 12, 26, 9).unwrap();

    // MACD line lookback = slow_period - 1 = 25
    let macd_nan_count = result.macd_line.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        macd_nan_count, 25,
        "MACD line should have 25 NaN values, got {}",
        macd_nan_count
    );

    // Signal line lookback = slow_period + signal_period - 2 = 33
    let signal_nan_count = result.signal_line.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        signal_nan_count, 33,
        "Signal line should have 33 NaN values, got {}",
        signal_nan_count
    );
}

#[test]
fn spec_lookback_atr() {
    let high: Vec<f64> = (0..30).map(|x| 100.0 + x as f64).collect();
    let low: Vec<f64> = (0..30).map(|x| 98.0 + x as f64).collect();
    let close: Vec<f64> = (0..30).map(|x| 99.0 + x as f64).collect();
    let period = 14;
    let result = atr(&high, &low, &close, period).unwrap();

    // ATR lookback = period = 14
    let expected_lookback = period;
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, expected_lookback,
        "ATR should have {} NaN values, got {}",
        expected_lookback, nan_count
    );
}

#[test]
fn spec_lookback_bollinger() {
    let input: Vec<f64> = (0..30).map(|x| x as f64).collect();
    let period = 20;
    let result = bollinger(&input, period, 2.0).unwrap();

    // Bollinger lookback = period - 1 = 19
    let expected_lookback = period - 1;
    let nan_count = result.middle.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, expected_lookback,
        "Bollinger should have {} NaN values, got {}",
        expected_lookback, nan_count
    );
}

#[test]
fn spec_lookback_stochastic() {
    let n = 30;
    let high: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 98.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 99.0 + x as f64).collect();
    let k_period = 14;
    let d_period = 3;
    let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();

    // %K lookback = k_period - 1 = 13
    let k_nan_count = result.k.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        k_nan_count,
        k_period - 1,
        "%K should have {} NaN values, got {}",
        k_period - 1,
        k_nan_count
    );

    // %D lookback = k_period + d_period - 2 = 15
    let d_nan_count = result.d.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        d_nan_count,
        k_period + d_period - 2,
        "%D should have {} NaN values, got {}",
        k_period + d_period - 2,
        d_nan_count
    );
}

// ==================== Lookback Function Tests ====================
// These tests verify that the lookback() functions return correct values

#[test]
fn spec_lookback_functions_sma() {
    assert_eq!(sma_lookback(5), 4);
    assert_eq!(sma_lookback(14), 13);
    assert_eq!(sma_lookback(1), 0);
    assert_eq!(sma_min_len(5), 5);
    assert_eq!(sma_min_len(14), 14);
}

#[test]
fn spec_lookback_functions_ema() {
    assert_eq!(ema_lookback(5), 4);
    assert_eq!(ema_lookback(14), 13);
    assert_eq!(ema_lookback(1), 0);
    assert_eq!(ema_min_len(5), 5);
    assert_eq!(ema_min_len(14), 14);
}

#[test]
fn spec_lookback_functions_rsi() {
    assert_eq!(rsi_lookback(14), 14);
    assert_eq!(rsi_lookback(5), 5);
    assert_eq!(rsi_lookback(1), 1);
    assert_eq!(rsi_min_len(14), 15);
    assert_eq!(rsi_min_len(5), 6);
}

#[test]
fn spec_lookback_functions_macd() {
    // Standard MACD (12, 26, 9)
    assert_eq!(macd_line_lookback(26), 25);
    assert_eq!(macd_signal_lookback(26, 9), 33);
    assert_eq!(macd_min_len(26, 9), 34);
}

#[test]
fn spec_lookback_functions_atr() {
    assert_eq!(atr_lookback(14), 14);
    assert_eq!(atr_lookback(5), 5);
    assert_eq!(atr_min_len(14), 15);
    assert_eq!(atr_min_len(5), 6);
    assert_eq!(true_range_lookback(), 1);
}

#[test]
fn spec_lookback_functions_bollinger() {
    assert_eq!(bollinger_lookback(20), 19);
    assert_eq!(bollinger_lookback(5), 4);
    assert_eq!(bollinger_min_len(20), 20);
    assert_eq!(bollinger_min_len(5), 5);
}

#[test]
fn spec_lookback_functions_stochastic() {
    // Standard stochastic (14, 3)
    assert_eq!(stochastic_k_lookback(14), 13);
    assert_eq!(stochastic_d_lookback(14, 3), 15);
    assert_eq!(stochastic_min_len(14, 3), 16);
}

#[test]
fn spec_lookback_functions_rolling_extrema() {
    assert_eq!(rolling_extrema_lookback(5), 4);
    assert_eq!(rolling_extrema_lookback(14), 13);
    assert_eq!(rolling_extrema_min_len(5), 5);
    assert_eq!(rolling_extrema_min_len(14), 14);
}

// ==================== Consistency Tests: lookback() matches actual NaN count ====================

#[test]
fn spec_lookback_consistency_sma() {
    let input: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let period = 5;
    let result = sma(&input, period).unwrap();
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, sma_lookback(period));
}

#[test]
fn spec_lookback_consistency_ema() {
    let input: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let period = 5;
    let result = ema(&input, period).unwrap();
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, ema_lookback(period));
}

#[test]
fn spec_lookback_consistency_rsi() {
    let input: Vec<f64> = (0..30).map(|x| x as f64).collect();
    let period = 14;
    let result = rsi(&input, period).unwrap();
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, rsi_lookback(period));
}

#[test]
fn spec_lookback_consistency_macd() {
    let input: Vec<f64> = (0..50).map(|x| x as f64).collect();
    let fast = 12;
    let slow = 26;
    let signal = 9;
    let result = macd(&input, fast, slow, signal).unwrap();

    let macd_nan = result.macd_line.iter().filter(|x| x.is_nan()).count();
    assert_eq!(macd_nan, macd_line_lookback(slow));

    let signal_nan = result.signal_line.iter().filter(|x| x.is_nan()).count();
    assert_eq!(signal_nan, macd_signal_lookback(slow, signal));
}

#[test]
fn spec_lookback_consistency_atr() {
    let n = 30;
    let high: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 98.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 99.0 + x as f64).collect();
    let period = 14;

    let result = atr(&high, &low, &close, period).unwrap();
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, atr_lookback(period));

    let tr = true_range(&high, &low, &close).unwrap();
    let tr_nan = tr.iter().filter(|x| x.is_nan()).count();
    assert_eq!(tr_nan, true_range_lookback());
}

#[test]
fn spec_lookback_consistency_bollinger() {
    let input: Vec<f64> = (0..30).map(|x| x as f64).collect();
    let period = 20;
    let result = bollinger(&input, period, 2.0).unwrap();
    let nan_count = result.middle.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, bollinger_lookback(period));
}

#[test]
fn spec_lookback_consistency_stochastic() {
    let n = 30;
    let high: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 98.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 99.0 + x as f64).collect();
    let k_period = 14;
    let d_period = 3;

    let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();

    let k_nan = result.k.iter().filter(|x| x.is_nan()).count();
    assert_eq!(k_nan, stochastic_k_lookback(k_period));

    let d_nan = result.d.iter().filter(|x| x.is_nan()).count();
    assert_eq!(d_nan, stochastic_d_lookback(k_period, d_period));
}

// ==================== Output Length Contract Tests ====================
// Contract: output.len() == input.len()
// All indicators return full-length output with NaN for lookback positions

#[test]
fn output_length_contract_sma() {
    for input_len in [5, 10, 50, 100] {
        let input: Vec<f64> = (0..input_len).map(|x| x as f64).collect();
        for period in [3, 5, 14, 20] {
            if input_len >= period {
                let result = sma(&input, period).unwrap();
                assert_eq!(
                    result.len(),
                    input.len(),
                    "SMA output length should match input length"
                );
            }
        }
    }
}

#[test]
fn output_length_contract_ema() {
    for input_len in [5, 10, 50, 100] {
        let input: Vec<f64> = (0..input_len).map(|x| x as f64).collect();
        for period in [3, 5, 14, 20] {
            if input_len >= period {
                let result = ema(&input, period).unwrap();
                assert_eq!(
                    result.len(),
                    input.len(),
                    "EMA output length should match input length"
                );
            }
        }
    }
}

#[test]
fn output_length_contract_rsi() {
    for input_len in [10, 20, 50, 100] {
        let input: Vec<f64> = (0..input_len).map(|x| x as f64).collect();
        for period in [3, 7, 14] {
            if input_len >= period + 1 {
                let result = rsi(&input, period).unwrap();
                assert_eq!(
                    result.len(),
                    input.len(),
                    "RSI output length should match input length"
                );
            }
        }
    }
}

#[test]
fn output_length_contract_macd() {
    let input: Vec<f64> = (0..50).map(|x| x as f64).collect();
    let result = macd(&input, 12, 26, 9).unwrap();

    assert_eq!(
        result.macd_line.len(),
        input.len(),
        "MACD line output length should match input length"
    );
    assert_eq!(
        result.signal_line.len(),
        input.len(),
        "MACD signal output length should match input length"
    );
    assert_eq!(
        result.histogram.len(),
        input.len(),
        "MACD histogram output length should match input length"
    );
}

#[test]
fn output_length_contract_atr() {
    let n = 50;
    let high: Vec<f64> = (0..n).map(|x| 110.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 90.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();

    for period in [7, 14, 20] {
        let result = atr(&high, &low, &close, period).unwrap();
        assert_eq!(
            result.len(),
            n,
            "ATR output length should match input length"
        );
    }
}

#[test]
fn output_length_contract_bollinger() {
    for input_len in [20, 30, 50, 100] {
        let input: Vec<f64> = (0..input_len).map(|x| x as f64).collect();
        for period in [10, 20] {
            if input_len >= period {
                let result = bollinger(&input, period, 2.0).unwrap();
                assert_eq!(
                    result.middle.len(),
                    input.len(),
                    "Bollinger middle band output length should match input length"
                );
                assert_eq!(
                    result.upper.len(),
                    input.len(),
                    "Bollinger upper band output length should match input length"
                );
                assert_eq!(
                    result.lower.len(),
                    input.len(),
                    "Bollinger lower band output length should match input length"
                );
            }
        }
    }
}

#[test]
fn output_length_contract_stochastic() {
    let n = 50;
    let high: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 98.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 99.0 + x as f64).collect();

    let result = stochastic_fast(&high, &low, &close, 14, 3).unwrap();

    assert_eq!(result.k.len(), n, "Stochastic %K output length should match input length");
    assert_eq!(result.d.len(), n, "Stochastic %D output length should match input length");
}

#[test]
fn output_length_contract_true_range() {
    let n = 50;
    let high: Vec<f64> = (0..n).map(|x| 110.0 + x as f64).collect();
    let low: Vec<f64> = (0..n).map(|x| 90.0 + x as f64).collect();
    let close: Vec<f64> = (0..n).map(|x| 100.0 + x as f64).collect();

    let result = true_range(&high, &low, &close).unwrap();
    assert_eq!(
        result.len(),
        n,
        "True Range output length should match input length"
    );
}
