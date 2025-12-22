//! Property-based tests for all indicators using proptest.
//!
//! These tests verify invariant properties that must hold for all valid inputs,
//! using randomly generated test data to find edge cases.

use proptest::prelude::*;

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

// ==================== Test Data Generators ====================

/// Generate a random price series (all positive values)
fn arb_price_series(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(1.0..1000.0_f64, min_len..=max_len)
}

/// Generate a random OHLC series with valid constraints (high >= open, close; low <= open, close)
fn arb_ohlc_series(
    min_len: usize,
    max_len: usize,
) -> impl Strategy<Value = (Vec<f64>, Vec<f64>, Vec<f64>)> {
    prop::collection::vec(
        (1.0..1000.0_f64, 0.0..0.1_f64, 0.0..0.1_f64),
        min_len..=max_len,
    )
    .prop_map(|data| {
        let mut high = Vec::with_capacity(data.len());
        let mut low = Vec::with_capacity(data.len());
        let mut close = Vec::with_capacity(data.len());

        for (base, high_pct, low_pct) in data {
            let h = base * (1.0 + high_pct);
            let l = base * (1.0 - low_pct);
            let c = base; // close at base price
            high.push(h);
            low.push(l);
            close.push(c);
        }

        (high, low, close)
    })
}

// ==================== SMA Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// SMA output length equals input length
    #[test]
    fn prop_sma_output_length(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = sma(&data, period).unwrap();
            prop_assert_eq!(result.len(), data.len());
        }
    }

    /// SMA has exactly period-1 NaN values at the start
    #[test]
    fn prop_sma_nan_count(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = sma(&data, period).unwrap();
            let nan_count = result.iter().filter(|x| x.is_nan()).count();
            prop_assert_eq!(nan_count, period - 1);
        }
    }

    /// SMA of constant values equals that constant
    #[test]
    fn prop_sma_constant_input(constant in 1.0..1000.0_f64, len in 5usize..50, period in 1usize..=10) {
        if len >= period {
            let data = vec![constant; len];
            let result = sma(&data, period).unwrap();

            for i in (period - 1)..len {
                prop_assert!(
                    (result[i] - constant).abs() < 1e-10,
                    "SMA of constant {} at index {} is {}", constant, i, result[i]
                );
            }
        }
    }
}

// ==================== EMA Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// EMA output length equals input length
    #[test]
    fn prop_ema_output_length(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = ema(&data, period).unwrap();
            prop_assert_eq!(result.len(), data.len());
        }
    }

    /// EMA has exactly period-1 NaN values at the start
    #[test]
    fn prop_ema_nan_count(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = ema(&data, period).unwrap();
            let nan_count = result.iter().filter(|x| x.is_nan()).count();
            prop_assert_eq!(nan_count, period - 1);
        }
    }

    /// EMA of constant values equals that constant
    #[test]
    fn prop_ema_constant_input(constant in 1.0..1000.0_f64, len in 5usize..50, period in 1usize..=10) {
        if len >= period {
            let data = vec![constant; len];
            let result = ema(&data, period).unwrap();

            for i in (period - 1)..len {
                prop_assert!(
                    (result[i] - constant).abs() < 1e-10,
                    "EMA of constant {} at index {} is {}", constant, i, result[i]
                );
            }
        }
    }
}

// ==================== RSI Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// RSI output length equals input length
    #[test]
    fn prop_rsi_output_length(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period + 1 {
            let result = rsi(&data, period).unwrap();
            prop_assert_eq!(result.len(), data.len());
        }
    }

    /// RSI values are in range [0, 100]
    #[test]
    fn prop_rsi_bounded(data in arb_price_series(5, 100), period in 1usize..=10) {
        if data.len() >= period + 1 {
            let result = rsi(&data, period).unwrap();

            for (i, &val) in result.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0 && val <= 100.0,
                        "RSI at index {} is out of bounds: {}", i, val
                    );
                }
            }
        }
    }

    /// RSI of strictly increasing prices equals 100
    #[test]
    fn prop_rsi_all_gains(start in 1.0..100.0_f64, step in 0.1..5.0_f64, len in 5usize..20, period in 1usize..=5) {
        if len >= period + 1 {
            let data: Vec<f64> = (0..len).map(|i| start + step * (i as f64)).collect();
            let result = rsi(&data, period).unwrap();

            for i in period..len {
                prop_assert!(
                    (result[i] - 100.0).abs() < 1e-6,
                    "RSI of increasing prices at {} should be 100, got {}", i, result[i]
                );
            }
        }
    }

    /// RSI of strictly decreasing prices equals 0
    #[test]
    fn prop_rsi_all_losses(start in 100.0..200.0_f64, step in 0.1..5.0_f64, len in 5usize..20, period in 1usize..=5) {
        if len >= period + 1 {
            let data: Vec<f64> = (0..len).map(|i| start - step * (i as f64)).filter(|&x| x > 0.0).collect();
            if data.len() >= period + 1 {
                let result = rsi(&data, period).unwrap();

                for i in period..data.len() {
                    prop_assert!(
                        result[i].abs() < 1e-6,
                        "RSI of decreasing prices at {} should be 0, got {}", i, result[i]
                    );
                }
            }
        }
    }
}

// ==================== MACD Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// MACD output lengths equal input length
    #[test]
    fn prop_macd_output_length(data in arb_price_series(40, 100)) {
        let fast = 12;
        let slow = 26;
        let signal = 9;
        let min_required = slow + signal - 1;

        if data.len() >= min_required {
            let result = macd(&data, fast, slow, signal).unwrap();
            prop_assert_eq!(result.macd_line.len(), data.len());
            prop_assert_eq!(result.signal_line.len(), data.len());
            prop_assert_eq!(result.histogram.len(), data.len());
        }
    }

    /// MACD histogram = MACD line - signal line
    #[test]
    fn prop_macd_histogram_definition(data in arb_price_series(50, 100)) {
        let fast = 12;
        let slow = 26;
        let signal = 9;

        if data.len() >= slow + signal - 1 {
            let result = macd(&data, fast, slow, signal).unwrap();

            for i in 0..data.len() {
                if !result.histogram[i].is_nan() {
                    let expected = result.macd_line[i] - result.signal_line[i];
                    prop_assert!(
                        (result.histogram[i] - expected).abs() < 1e-10,
                        "Histogram[{}] = {} != MACD - Signal = {}",
                        i, result.histogram[i], expected
                    );
                }
            }
        }
    }
}

// ==================== Bollinger Bands Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Bollinger bands maintain upper >= middle >= lower
    #[test]
    fn prop_bollinger_band_order(data in arb_price_series(25, 100), period in 5usize..=20) {
        if data.len() >= period {
            let result = bollinger(&data, period, 2.0).unwrap();

            for i in (period - 1)..data.len() {
                if !result.middle[i].is_nan() {
                    prop_assert!(
                        result.upper[i] >= result.middle[i],
                        "Upper {} < Middle {} at index {}", result.upper[i], result.middle[i], i
                    );
                    prop_assert!(
                        result.middle[i] >= result.lower[i],
                        "Middle {} < Lower {} at index {}", result.middle[i], result.lower[i], i
                    );
                }
            }
        }
    }

    /// Bollinger bands are symmetric around middle
    #[test]
    fn prop_bollinger_symmetric(data in arb_price_series(25, 100), period in 5usize..=20) {
        if data.len() >= period {
            let result = bollinger(&data, period, 2.0).unwrap();

            for i in (period - 1)..data.len() {
                if !result.middle[i].is_nan() {
                    let upper_diff = result.upper[i] - result.middle[i];
                    let lower_diff = result.middle[i] - result.lower[i];
                    prop_assert!(
                        (upper_diff - lower_diff).abs() < 1e-10,
                        "Bands not symmetric at {}: upper_diff={}, lower_diff={}",
                        i, upper_diff, lower_diff
                    );
                }
            }
        }
    }
}

// ==================== ATR Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// ATR is always non-negative
    #[test]
    fn prop_atr_non_negative((high, low, close) in arb_ohlc_series(20, 100), period in 1usize..=14) {
        if high.len() >= period + 1 {
            let result = atr(&high, &low, &close, period).unwrap();

            for (i, &val) in result.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0,
                        "ATR at index {} is negative: {}", i, val
                    );
                }
            }
        }
    }

    /// True Range is always non-negative
    #[test]
    fn prop_true_range_non_negative((high, low, close) in arb_ohlc_series(5, 100)) {
        let result = true_range(&high, &low, &close).unwrap();

        for (i, &val) in result.iter().enumerate() {
            if !val.is_nan() {
                prop_assert!(
                    val >= 0.0,
                    "True Range at index {} is negative: {}", i, val
                );
            }
        }
    }
}

// ==================== Stochastic Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Stochastic %K is in range [0, 100]
    #[test]
    fn prop_stochastic_k_bounded((high, low, close) in arb_ohlc_series(20, 100)) {
        let k_period = 14;
        let d_period = 3;

        if high.len() >= k_period {
            let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();

            for (i, &val) in result.k.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0 && val <= 100.0,
                        "%K at index {} is out of bounds: {}", i, val
                    );
                }
            }
        }
    }

    /// Stochastic %D is in range [0, 100]
    #[test]
    fn prop_stochastic_d_bounded((high, low, close) in arb_ohlc_series(20, 100)) {
        let k_period = 14;
        let d_period = 3;

        if high.len() >= k_period {
            let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();

            for (i, &val) in result.d.iter().enumerate() {
                if !val.is_nan() {
                    prop_assert!(
                        val >= 0.0 && val <= 100.0,
                        "%D at index {} is out of bounds: {}", i, val
                    );
                }
            }
        }
    }
}

// ==================== Rolling Extrema Properties ====================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Rolling max >= rolling min
    #[test]
    fn prop_rolling_max_gte_min(data in arb_price_series(10, 100), period in 1usize..=10) {
        if data.len() >= period {
            let max_result = rolling_max(&data, period).unwrap();
            let min_result = rolling_min(&data, period).unwrap();

            for i in (period - 1)..data.len() {
                prop_assert!(
                    max_result[i] >= min_result[i],
                    "Max {} < Min {} at index {}", max_result[i], min_result[i], i
                );
            }
        }
    }

    /// Rolling max is at least as large as current value
    #[test]
    fn prop_rolling_max_gte_current(data in arb_price_series(10, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = rolling_max(&data, period).unwrap();

            for i in (period - 1)..data.len() {
                if !result[i].is_nan() {
                    prop_assert!(
                        result[i] >= data[i],
                        "Rolling max {} < current value {} at index {}",
                        result[i], data[i], i
                    );
                }
            }
        }
    }

    /// Rolling min is at most as large as current value
    #[test]
    fn prop_rolling_min_lte_current(data in arb_price_series(10, 100), period in 1usize..=10) {
        if data.len() >= period {
            let result = rolling_min(&data, period).unwrap();

            for i in (period - 1)..data.len() {
                if !result[i].is_nan() {
                    prop_assert!(
                        result[i] <= data[i],
                        "Rolling min {} > current value {} at index {}",
                        result[i], data[i], i
                    );
                }
            }
        }
    }
}
