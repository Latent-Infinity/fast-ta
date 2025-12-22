//! Numeric policy compliance tests.
//!
//! These tests verify compliance with the numeric policy defined in PRD §4.3 and §4.5.
//! The policy covers NaN propagation, infinity handling, and edge case behaviors.

use fast_ta::indicators::{
    atr::{atr, true_range},
    bollinger::bollinger,
    ema::ema,
    macd::macd,
    rsi::rsi,
    sma::sma,
    stochastic::stochastic_fast,
};
use fast_ta::kernels::rolling_extrema::{rolling_max, rolling_min};

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

// ==================== PRD §4.3: NaN Propagation Tests ====================
// "NaN in window: Output NaN for that position"

#[test]
fn numeric_policy_nan_propagation_sma() {
    let data = vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0];
    let result = sma(&data, 3).unwrap();

    // SMA with period 3 should produce NaN when any value in window contains NaN
    // Window [2.0, NaN, 4.0] at index 3 should produce NaN
    assert!(result[3].is_nan(), "SMA should propagate NaN at index 3");
    // Window [NaN, 4.0, 5.0] at index 4 should produce NaN
    assert!(result[4].is_nan(), "SMA should propagate NaN at index 4");
    // Window [4.0, 5.0, 6.0] at index 5 should be valid
    assert!(
        !result[5].is_nan(),
        "SMA should be valid once NaN leaves window"
    );
}

#[test]
fn numeric_policy_nan_propagation_ema() {
    let data = vec![1.0_f64, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0];
    let result = ema(&data, 3).unwrap();

    // Once NaN appears, EMA should propagate NaN forward (due to recursive nature)
    // EMA at index 2 is the SMA seed (valid)
    assert!(!result[2].is_nan(), "EMA seed should be valid before NaN");
    // EMA at index 3 encounters NaN in data
    assert!(result[3].is_nan(), "EMA should be NaN when input is NaN");
    // EMA at index 4 depends on prior NaN EMA
    assert!(
        result[4].is_nan(),
        "EMA should propagate NaN forward due to recursion"
    );
}

#[test]
fn numeric_policy_nan_propagation_rsi() {
    let data = vec![10.0_f64, 11.0, 12.0, f64::NAN, 14.0, 15.0, 16.0, 17.0, 18.0];
    let result = rsi(&data, 3).unwrap();

    // RSI should propagate NaN when NaN appears in the input
    assert!(result[3].is_nan(), "RSI should be NaN at position with NaN");
}

#[test]
fn numeric_policy_nan_propagation_macd() {
    let mut data: Vec<f64> = (0..40).map(|x| x as f64).collect();
    data[20] = f64::NAN;
    let result = macd(&data, 12, 26, 9).unwrap();

    // MACD should propagate NaN after encountering NaN in slow EMA calculation
    // The NaN at index 20 will affect the slow EMA
    let nan_after = result
        .macd_line
        .iter()
        .skip(21)
        .all(|x| x.is_nan() || !x.is_finite());
    assert!(nan_after || result.macd_line[20..].iter().any(|x| x.is_nan()));
}

#[test]
fn numeric_policy_nan_propagation_bollinger() {
    let data = vec![20.0_f64, 21.0, f64::NAN, 22.0, 23.0, 24.0, 25.0];
    let result = bollinger(&data, 3, 2.0).unwrap();

    // Bollinger should produce NaN when window contains NaN
    assert!(
        result.middle[3].is_nan(),
        "Bollinger middle should be NaN at index 3"
    );
    assert!(
        result.upper[3].is_nan(),
        "Bollinger upper should be NaN at index 3"
    );
    assert!(
        result.lower[3].is_nan(),
        "Bollinger lower should be NaN at index 3"
    );
}

#[test]
fn numeric_policy_nan_propagation_atr() {
    let high = vec![11.0_f64, 12.0, f64::NAN, 14.0, 15.0, 16.0, 17.0];
    let low = vec![9.0_f64, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let close = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let tr = true_range(&high, &low, &close).unwrap();

    // True range should be NaN when high contains NaN
    assert!(tr[2].is_nan(), "TR should be NaN when high is NaN");

    let atr_result = atr(&high, &low, &close, 3).unwrap();
    // ATR calculation should propagate NaN
    assert!(atr_result[3].is_nan(), "ATR should propagate NaN");
}

#[test]
fn numeric_policy_nan_propagation_stochastic() {
    let high = vec![15.0_f64, 16.0, f64::NAN, 18.0, 19.0, 20.0, 21.0];
    let low = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let close = vec![12.0_f64, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];

    let result = stochastic_fast(&high, &low, &close, 3, 3).unwrap();

    // %K should be NaN when window contains NaN in any of high/low/close
    assert!(result.k[3].is_nan(), "%K should propagate NaN at index 3");
    assert!(result.k[4].is_nan(), "%K should propagate NaN at index 4");
}

#[test]
fn numeric_policy_nan_handling_rolling_extrema() {
    // Note: Rolling extrema has a DIFFERENT NaN policy than other indicators.
    // It skips NaN values and returns the max/min of non-NaN values in the window.
    // This is intentional for use in Stochastic oscillator which needs to find
    // extrema even when some values are missing.

    let data = vec![1.0_f64, f64::NAN, 3.0, 4.0, 5.0];

    let max_result = rolling_max(&data, 3).unwrap();
    let min_result = rolling_min(&data, 3).unwrap();

    // Window [1, NaN, 3]: max should be 3, min should be 1 (NaN skipped)
    assert!(
        approx_eq(max_result[2], 3.0, EPSILON),
        "Rolling max should skip NaN and return 3.0"
    );
    assert!(
        approx_eq(min_result[2], 1.0, EPSILON),
        "Rolling min should skip NaN and return 1.0"
    );

    // Window [NaN, 3, 4]: max should be 4, min should be 3
    assert!(
        approx_eq(max_result[3], 4.0, EPSILON),
        "Rolling max should skip NaN and return 4.0"
    );
    assert!(
        approx_eq(min_result[3], 3.0, EPSILON),
        "Rolling min should skip NaN and return 3.0"
    );
}

#[test]
fn numeric_policy_nan_all_nan_window_rolling_extrema() {
    // When ALL values in window are NaN, result should be NaN
    let data = vec![f64::NAN, f64::NAN, f64::NAN, 4.0, 5.0];

    let max_result = rolling_max(&data, 3).unwrap();

    // All-NaN window should produce NaN
    assert!(
        max_result[2].is_nan(),
        "All-NaN window should produce NaN in rolling max"
    );

    // Once valid values enter window, should produce valid result
    assert!(
        approx_eq(max_result[4], 5.0, EPSILON),
        "Rolling max should be valid after NaN leaves window"
    );
}

// ==================== PRD §4.3: Infinity Handling Tests ====================
// "Infinity in input: Propagate to output"

#[test]
fn numeric_policy_infinity_propagation_sma() {
    let data = vec![1.0_f64, 2.0, f64::INFINITY, 4.0, 5.0];
    let result = sma(&data, 3).unwrap();

    // SMA with infinity should produce infinity
    assert!(result[2].is_infinite(), "SMA should propagate infinity");
    assert!(
        result[3].is_infinite(),
        "SMA should propagate infinity at index 3"
    );
}

#[test]
fn numeric_policy_infinity_propagation_ema() {
    let data = vec![1.0_f64, 2.0, 3.0, f64::INFINITY, 5.0, 6.0];
    let result = ema(&data, 3).unwrap();

    // EMA with infinity should produce infinity or NaN (depending on computation)
    assert!(
        result[3].is_infinite() || result[3].is_nan(),
        "EMA should handle infinity"
    );
}

#[test]
fn numeric_policy_negative_infinity() {
    let data = vec![1.0_f64, 2.0, f64::NEG_INFINITY, 4.0, 5.0];
    let result = sma(&data, 3).unwrap();

    // Negative infinity should also propagate
    assert!(
        result[2].is_infinite() && result[2] < 0.0,
        "SMA should propagate negative infinity"
    );
}

// ==================== PRD §4.5: RSI Edge Cases ====================
// "All gains (avg_loss = 0): RSI = 100"
// "All losses (avg_gain = 0): RSI = 0"

#[test]
fn numeric_policy_rsi_all_gains_equals_100() {
    let data = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let result = rsi(&data, 5).unwrap();

    // All gains: every change is positive, so RSI = 100
    for i in 5..result.len() {
        assert!(
            approx_eq(result[i], 100.0, LOOSE_EPSILON),
            "RSI should be 100 for all gains, got {} at index {}",
            result[i],
            i
        );
    }
}

#[test]
fn numeric_policy_rsi_all_losses_equals_0() {
    let data = vec![20.0_f64, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0];
    let result = rsi(&data, 5).unwrap();

    // All losses: every change is negative, so RSI = 0
    for i in 5..result.len() {
        assert!(
            approx_eq(result[i], 0.0, LOOSE_EPSILON),
            "RSI should be 0 for all losses, got {} at index {}",
            result[i],
            i
        );
    }
}

#[test]
fn numeric_policy_rsi_no_movement_equals_50() {
    let data = vec![50.0_f64; 10];
    let result = rsi(&data, 5).unwrap();

    // No movement: avg_gain = avg_loss = 0, RSI = 50 (neutral)
    for i in 5..result.len() {
        assert!(
            approx_eq(result[i], 50.0, LOOSE_EPSILON),
            "RSI should be 50 for no movement, got {} at index {}",
            result[i],
            i
        );
    }
}

// ==================== PRD §4.5: Stochastic Edge Cases ====================
// "Denominator = 0 (high == low): %K = 50"

#[test]
fn numeric_policy_stochastic_flat_price_k50() {
    let high = vec![50.0_f64; 10];
    let low = vec![50.0_f64; 10];
    let close = vec![50.0_f64; 10];

    let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

    // When high == low == close (flat price), %K = 50
    for i in 4..result.k.len() {
        if !result.k[i].is_nan() {
            assert!(
                approx_eq(result.k[i], 50.0, LOOSE_EPSILON),
                "%K should be 50 for flat price, got {} at index {}",
                result.k[i],
                i
            );
        }
    }
}

// ==================== PRD §4.5: Bollinger Edge Cases ====================
// "StdDev = 0 (constant prices): Upper = Lower = Middle"

#[test]
fn numeric_policy_bollinger_constant_price_bands_collapse() {
    let data = vec![100.0_f64; 10];
    let result = bollinger(&data, 5, 2.0).unwrap();

    // Constant price: stddev = 0, so all bands equal the mean
    for i in 4..result.middle.len() {
        if !result.middle[i].is_nan() {
            assert!(
                approx_eq(result.middle[i], 100.0, EPSILON),
                "Middle should be 100"
            );
            assert!(
                approx_eq(result.upper[i], 100.0, EPSILON),
                "Upper should equal middle when stddev = 0"
            );
            assert!(
                approx_eq(result.lower[i], 100.0, EPSILON),
                "Lower should equal middle when stddev = 0"
            );
        }
    }
}

#[test]
fn numeric_policy_bollinger_symmetric_bands() {
    let data = vec![20.0_f64, 22.0, 18.0, 24.0, 16.0, 26.0, 14.0];
    let result = bollinger(&data, 3, 2.0).unwrap();

    // Bands should always be symmetric around middle
    for i in 2..result.middle.len() {
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

// ==================== PRD §4.5: ATR Edge Cases ====================
// "First value initialization: SMA of first n True Ranges"

#[test]
fn numeric_policy_atr_first_value_is_sma() {
    let high = vec![11.0_f64, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
    let low = vec![9.0_f64, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let close = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let tr = true_range(&high, &low, &close).unwrap();
    let atr_result = atr(&high, &low, &close, 3).unwrap();

    // First ATR value should be SMA of first 3 true ranges
    // TR[0] is NaN, TR[1] and TR[2] are valid
    // ATR[3] should be average of TR[1], TR[2], TR[3]
    if !atr_result[3].is_nan() {
        let expected_sma = (tr[1] + tr[2] + tr[3]) / 3.0;
        assert!(
            approx_eq(atr_result[3], expected_sma, EPSILON),
            "First ATR value should be SMA of first 3 TR values"
        );
    }
}

// ==================== Additional Numeric Edge Cases ====================

#[test]
fn numeric_policy_very_large_values() {
    let data = vec![1e300_f64, 2e300, 3e300, 4e300, 5e300];
    let result = sma(&data, 3).unwrap();

    // Should handle very large values without overflow
    assert!(
        !result[2].is_nan() && !result[2].is_infinite(),
        "SMA should handle large values"
    );
    assert!(
        approx_eq(result[2], 2e300, 1e290),
        "SMA of large values should be correct"
    );
}

#[test]
fn numeric_policy_very_small_values() {
    let data = vec![1e-300_f64, 2e-300, 3e-300, 4e-300, 5e-300];
    let result = sma(&data, 3).unwrap();

    // Should handle very small values without underflow
    assert!(
        !result[2].is_nan(),
        "SMA should handle small values without underflow"
    );
    assert!(
        approx_eq(result[2], 2e-300, 1e-310),
        "SMA of small values should be correct"
    );
}

#[test]
fn numeric_policy_mixed_signs() {
    let data = vec![-5.0_f64, -3.0, 0.0, 3.0, 5.0];
    let result = sma(&data, 3).unwrap();

    // Should handle mixed positive and negative values
    assert!(
        approx_eq(result[2], (-5.0 - 3.0 + 0.0) / 3.0, EPSILON),
        "SMA should handle mixed signs"
    );
    assert!(
        approx_eq(result[3], (-3.0 + 0.0 + 3.0) / 3.0, EPSILON),
        "SMA should handle crossing zero"
    );
}

#[test]
fn numeric_policy_all_nan_window() {
    let data = vec![f64::NAN, f64::NAN, f64::NAN, 4.0, 5.0, 6.0];
    let result = sma(&data, 3).unwrap();

    // All-NaN window should produce NaN
    assert!(result[2].is_nan(), "All-NaN window should produce NaN");
    // But once NaN leaves window, output should be valid
    assert!(
        approx_eq(result[5], 5.0, EPSILON),
        "SMA should be valid after NaN leaves window"
    );
}

#[test]
fn numeric_policy_rsi_bounds() {
    // Generate random-ish price movements
    let data: Vec<f64> = vec![
        50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0, 45.0, 56.0, 44.0, 57.0, 43.0,
    ];
    let result = rsi(&data, 5).unwrap();

    // RSI should always be in [0, 100]
    for (i, &val) in result.iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "RSI at index {} = {} is out of bounds",
                i,
                val
            );
        }
    }
}

#[test]
fn numeric_policy_stochastic_bounds() {
    let high = vec![
        55.0_f64, 56.0, 54.0, 57.0, 53.0, 58.0, 52.0, 59.0, 51.0, 60.0,
    ];
    let low = vec![
        45.0_f64, 46.0, 44.0, 47.0, 43.0, 48.0, 42.0, 49.0, 41.0, 50.0,
    ];
    let close = vec![
        50.0_f64, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0, 46.0, 55.0,
    ];

    let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

    // %K and %D should always be in [0, 100]
    for (i, &val) in result.k.iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "%K at index {} = {} is out of bounds",
                i,
                val
            );
        }
    }
    for (i, &val) in result.d.iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "%D at index {} = {} is out of bounds",
                i,
                val
            );
        }
    }
}
