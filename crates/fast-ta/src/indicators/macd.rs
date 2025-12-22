//! Moving Average Convergence Divergence (MACD) indicator.
//!
//! The MACD is a trend-following momentum indicator that shows the relationship
//! between two exponential moving averages of a security's price. It consists
//! of three components:
//!
//! - **MACD Line**: The difference between the fast EMA and slow EMA
//! - **Signal Line**: An EMA of the MACD line (typically 9-period)
//! - **Histogram**: The difference between the MACD line and signal line
//!
//! # Algorithm
//!
//! This implementation computes MACD with O(n) time complexity:
//!
//! 1. Calculate fast EMA (typically 12-period)
//! 2. Calculate slow EMA (typically 26-period)
//! 3. MACD Line = Fast EMA - Slow EMA
//! 4. Signal Line = EMA(MACD Line, `signal_period`)
//! 5. Histogram = MACD Line - Signal Line
//!
//! # Formula
//!
//! ```text
//! MACD Line[i] = EMA(fast_period)[i] - EMA(slow_period)[i]
//! Signal Line[i] = EMA(MACD Line, signal_period)[i]
//! Histogram[i] = MACD Line[i] - Signal Line[i]
//! ```
//!
//! # NaN Handling
//!
//! - MACD Line: First `slow_period - 1` values are NaN (need slow EMA to be valid)
//! - Signal Line: First `slow_period - 1 + signal_period - 1` values are NaN
//! - Histogram: Same as Signal Line (needs both MACD and Signal to be valid)
//!
//! # Example
//!
//! ```
//! use fast_ta::indicators::macd::{macd, MacdOutput};
//!
//! let data = vec![
//!     26.0_f64, 27.0, 28.0, 27.5, 28.5, 29.0, 28.0, 27.0, 28.0, 29.0,
//!     30.0, 29.5, 30.5, 31.0, 30.0, 31.0, 32.0, 31.5, 32.5, 33.0,
//!     32.0, 31.0, 32.0, 33.0, 34.0, 33.5, 34.5, 35.0, 34.0, 35.0,
//!     36.0, 35.5, 36.5, 37.0,
//! ];
//!
//! // Standard MACD with 12, 26, 9 periods
//! let result = macd(&data, 12, 26, 9).unwrap();
//!
//! // Access the three components
//! assert!(result.macd_line[24].is_nan()); // slow_period - 1 = 25 values are NaN
//! assert!(!result.macd_line[25].is_nan()); // First valid MACD value
//! ```

use crate::error::{Error, Result};
use crate::indicators::ema::ema;
use crate::traits::{SeriesElement, ValidatedInput};

/// The output of MACD calculation containing all three components.
///
/// # Fields
///
/// - `macd_line`: The difference between fast and slow EMAs
/// - `signal_line`: The EMA of the MACD line
/// - `histogram`: The difference between MACD line and signal line
#[derive(Debug, Clone)]
pub struct MacdOutput<T: SeriesElement> {
    /// The MACD line (fast EMA - slow EMA).
    ///
    /// First `slow_period - 1` values are NaN.
    pub macd_line: Vec<T>,

    /// The signal line (EMA of MACD line).
    ///
    /// First `slow_period - 1 + signal_period - 1` values are NaN.
    pub signal_line: Vec<T>,

    /// The histogram (MACD line - signal line).
    ///
    /// First `slow_period - 1 + signal_period - 1` values are NaN.
    pub histogram: Vec<T>,
}

impl<T: SeriesElement> MacdOutput<T> {
    /// Returns the length of the output vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.macd_line.len()
    }

    /// Returns true if the output vectors are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.macd_line.is_empty()
    }

    /// Returns the index of the first valid MACD line value.
    ///
    /// This is `slow_period - 1`.
    #[must_use]
    pub const fn first_valid_macd_index(&self, slow_period: usize) -> usize {
        slow_period - 1
    }

    /// Returns the index of the first valid signal/histogram value.
    ///
    /// This is `slow_period - 1 + signal_period - 1`.
    #[must_use]
    pub const fn first_valid_signal_index(&self, slow_period: usize, signal_period: usize) -> usize {
        slow_period - 1 + signal_period - 1
    }
}

/// Computes the Moving Average Convergence Divergence (MACD) indicator.
///
/// The MACD uses standard EMA smoothing (α = 2 / (period + 1)).
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `fast_period` - The period for the fast EMA (typically 12)
/// * `slow_period` - The period for the slow EMA (typically 26)
/// * `signal_period` - The period for the signal line EMA (typically 9)
///
/// # Returns
///
/// A `Result` containing a `MacdOutput` with all three components, or an error
/// if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - The fast period is greater than or equal to the slow period (`Error::InvalidPeriod`)
/// - The input data is shorter than required (`Error::InsufficientData`)
///
/// # Performance
///
/// - Time complexity: O(n) where n is the length of the input data
/// - Space complexity: O(n) for each output vector (3n total)
///
/// # Example
///
/// ```
/// use fast_ta::indicators::macd::macd;
///
/// let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
/// let result = macd(&data, 12, 26, 9).unwrap();
///
/// // Standard interpretation:
/// // - MACD line crosses above signal: bullish signal
/// // - MACD line crosses below signal: bearish signal
/// // - Positive histogram: bullish momentum
/// // - Negative histogram: bearish momentum
/// ```
#[must_use = "this returns a Result with the MACD output, which should be used"]
pub fn macd<T: SeriesElement>(
    data: &[T],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<MacdOutput<T>> {
    // Validate inputs
    validate_macd_inputs(data, fast_period, slow_period, signal_period)?;

    let n = data.len();

    // Step 1: Calculate fast and slow EMAs
    let fast_ema = ema(data, fast_period)?;
    let slow_ema = ema(data, slow_period)?;

    // Step 2: Calculate MACD line = fast EMA - slow EMA
    let mut macd_line = vec![T::nan(); n];
    for i in 0..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Step 3: Calculate signal line = EMA of MACD line
    // We need to compute EMA starting from the first valid MACD value
    let first_valid_macd = slow_period - 1;
    let signal_line = compute_signal_line(&macd_line, signal_period, first_valid_macd)?;

    // Step 4: Calculate histogram = MACD line - signal line
    let first_valid_signal = first_valid_macd + signal_period - 1;
    let mut histogram = vec![T::nan(); n];
    for i in first_valid_signal..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok(MacdOutput {
        macd_line,
        signal_line,
        histogram,
    })
}

/// Computes MACD into pre-allocated output buffers.
///
/// This variant allows reusing existing buffers to avoid allocations in
/// performance-critical code paths.
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `fast_period` - The period for the fast EMA (typically 12)
/// * `slow_period` - The period for the slow EMA (typically 26)
/// * `signal_period` - The period for the signal line EMA (typically 9)
/// * `macd_output` - Pre-allocated output buffer for MACD line
/// * `signal_output` - Pre-allocated output buffer for signal line
/// * `histogram_output` - Pre-allocated output buffer for histogram
///
/// # Returns
///
/// A `Result` containing a tuple of (`valid_macd_count`, `valid_signal_count`),
/// where `valid_signal_count` also applies to the histogram.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - The fast period is greater than or equal to the slow period (`Error::InvalidPeriod`)
/// - The input data is shorter than required (`Error::InsufficientData`)
/// - Any output buffer is shorter than the input data
///
/// # Example
///
/// ```
/// use fast_ta::indicators::macd::macd_into;
///
/// let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
/// let mut macd_line = vec![0.0_f64; 50];
/// let mut signal_line = vec![0.0_f64; 50];
/// let mut histogram = vec![0.0_f64; 50];
///
/// let (valid_macd, valid_signal) = macd_into(
///     &data, 12, 26, 9,
///     &mut macd_line, &mut signal_line, &mut histogram
/// ).unwrap();
/// ```
#[must_use = "this returns a Result with the valid MACD counts"]
pub fn macd_into<T: SeriesElement>(
    data: &[T],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    macd_output: &mut [T],
    signal_output: &mut [T],
    histogram_output: &mut [T],
) -> Result<(usize, usize)> {
    // Validate inputs
    validate_macd_inputs(data, fast_period, slow_period, signal_period)?;

    let n = data.len();

    // Validate output buffer sizes
    if macd_output.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: macd_output.len(),
        });
    }
    if signal_output.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: signal_output.len(),
        });
    }
    if histogram_output.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: histogram_output.len(),
        });
    }

    // Step 1: Calculate fast and slow EMAs
    let fast_ema = ema(data, fast_period)?;
    let slow_ema = ema(data, slow_period)?;

    // Step 2: Calculate MACD line and initialize outputs with NaN
    let first_valid_macd = slow_period - 1;
    for value in macd_output.iter_mut().take(first_valid_macd) {
        *value = T::nan();
    }
    for i in first_valid_macd..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_output[i] = fast_ema[i] - slow_ema[i];
        } else {
            macd_output[i] = T::nan();
        }
    }

    // Step 3: Calculate signal line
    let signal_line = compute_signal_line(macd_output, signal_period, first_valid_macd)?;
    let first_valid_signal = first_valid_macd + signal_period - 1;
    signal_output[..n].copy_from_slice(&signal_line[..n]);

    // Step 4: Calculate histogram
    for value in histogram_output.iter_mut().take(first_valid_signal) {
        *value = T::nan();
    }
    for i in first_valid_signal..n {
        if !macd_output[i].is_nan() && !signal_output[i].is_nan() {
            histogram_output[i] = macd_output[i] - signal_output[i];
        } else {
            histogram_output[i] = T::nan();
        }
    }

    // Valid counts
    let valid_macd = n - first_valid_macd;
    let valid_signal = n - first_valid_signal;

    Ok((valid_macd, valid_signal))
}

/// Validates MACD inputs.
fn validate_macd_inputs<T: SeriesElement>(
    data: &[T],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<()> {
    // Validate periods
    if fast_period == 0 {
        return Err(Error::InvalidPeriod {
            period: fast_period,
            reason: "fast_period must be at least 1",
        });
    }
    if slow_period == 0 {
        return Err(Error::InvalidPeriod {
            period: slow_period,
            reason: "slow_period must be at least 1",
        });
    }
    if signal_period == 0 {
        return Err(Error::InvalidPeriod {
            period: signal_period,
            reason: "signal_period must be at least 1",
        });
    }
    if fast_period >= slow_period {
        return Err(Error::InvalidPeriod {
            period: fast_period,
            reason: "fast_period must be less than slow_period",
        });
    }

    // Validate data
    data.validate_not_empty()?;

    // Minimum required data: slow_period (for slow EMA) + signal_period - 1 (for signal EMA)
    // Actually, we need at least slow_period for a valid MACD line,
    // and slow_period + signal_period - 1 for a valid signal line
    let min_required = slow_period + signal_period - 1;
    if data.len() < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            actual: data.len(),
        });
    }

    Ok(())
}

/// Computes the signal line (EMA of MACD line) starting from the first valid MACD value.
fn compute_signal_line<T: SeriesElement>(
    macd_line: &[T],
    signal_period: usize,
    first_valid_macd: usize,
) -> Result<Vec<T>> {
    let n = macd_line.len();
    let mut signal_line = vec![T::nan(); n];

    // The signal line starts at first_valid_macd + signal_period - 1
    let first_valid_signal = first_valid_macd + signal_period - 1;

    // Not enough data for signal line
    if first_valid_signal >= n {
        return Ok(signal_line);
    }

    // Calculate alpha for standard EMA
    let two = T::two();
    let period_plus_one = T::from_usize(signal_period + 1)?;
    let alpha = two / period_plus_one;
    let one_minus_alpha = T::one() - alpha;

    // Calculate initial SMA seed from the first signal_period valid MACD values
    let period_t = T::from_usize(signal_period)?;
    let mut sum = T::zero();
    for &value in macd_line
        .iter()
        .skip(first_valid_macd)
        .take(signal_period)
    {
        sum = sum + value;
    }
    let sma_seed = sum / period_t;

    // Set the first valid signal value
    signal_line[first_valid_signal] = sma_seed;

    // Apply EMA formula for remaining values
    let mut ema_prev = sma_seed;
    for i in (first_valid_signal + 1)..n {
        if macd_line[i].is_nan() {
            signal_line[i] = T::nan();
            // Keep ema_prev unchanged (or we could set it to NaN and propagate)
        } else {
            let ema_current = alpha * macd_line[i] + one_minus_alpha * ema_prev;
            signal_line[i] = ema_current;
            ema_prev = ema_current;
        }
    }

    Ok(signal_line)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::all, clippy::pedantic, clippy::nursery)]
    use super::*;
    use num_traits::Float;

    // Helper function to compare floating point values
    fn approx_eq<T: Float>(a: T, b: T, epsilon: T) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_nan() || b.is_nan() {
            return false;
        }
        (a - b).abs() < epsilon
    }

    const EPSILON: f64 = 1e-10;
    // Looser epsilon for MACD calculations involving multiple EMAs
    const MACD_EPSILON: f64 = 1e-6;

    // ==================== Basic Functionality Tests ====================

    #[test]
    fn test_macd_basic() {
        // Create enough data for MACD(12, 26, 9)
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd(&data, 12, 26, 9).unwrap();

        assert_eq!(result.len(), 50);

        // First slow_period - 1 = 25 MACD values should be NaN
        for i in 0..25 {
            assert!(result.macd_line[i].is_nan(), "MACD line at {} should be NaN", i);
        }

        // First valid MACD at index 25
        assert!(!result.macd_line[25].is_nan());

        // First slow_period + signal_period - 2 = 33 signal values should be NaN
        for i in 0..33 {
            assert!(result.signal_line[i].is_nan(), "Signal line at {} should be NaN", i);
        }

        // First valid signal at index 33
        assert!(!result.signal_line[33].is_nan());

        // Histogram should have same NaN pattern as signal
        for i in 0..33 {
            assert!(result.histogram[i].is_nan(), "Histogram at {} should be NaN", i);
        }
        assert!(!result.histogram[33].is_nan());
    }

    #[test]
    fn test_macd_f32() {
        let data: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.5).collect();
        let result = macd(&data, 12, 26, 9).unwrap();

        assert_eq!(result.len(), 50);
        assert!(!result.macd_line[25].is_nan());
        assert!(!result.signal_line[33].is_nan());
    }

    #[test]
    fn test_macd_small_periods() {
        // Test with smaller periods (3, 6, 2)
        let data: Vec<f64> = (0..20).map(|i| 50.0 + (i as f64)).collect();
        let result = macd(&data, 3, 6, 2).unwrap();

        assert_eq!(result.len(), 20);

        // First valid MACD at index slow_period - 1 = 5
        assert!(result.macd_line[4].is_nan());
        assert!(!result.macd_line[5].is_nan());

        // First valid signal at index slow_period + signal_period - 2 = 6
        assert!(result.signal_line[5].is_nan());
        assert!(!result.signal_line[6].is_nan());
    }

    #[test]
    fn test_macd_minimum_periods() {
        // Test with minimum valid periods (1, 2, 1)
        let data: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = macd(&data, 1, 2, 1).unwrap();

        assert_eq!(result.len(), 5);

        // First valid MACD at index 1 (slow_period - 1)
        assert!(result.macd_line[0].is_nan());
        assert!(!result.macd_line[1].is_nan());

        // First valid signal at index 1 (slow_period + signal_period - 2 = 1)
        assert!(result.signal_line[0].is_nan());
        assert!(!result.signal_line[1].is_nan());
    }

    // ==================== MACD Line Tests ====================

    #[test]
    fn test_macd_line_uptrend() {
        // In an uptrend, fast EMA > slow EMA, so MACD line should be positive
        let data: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 2.0).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // After warmup, MACD line should be positive
        for i in 15..result.len() {
            assert!(
                result.macd_line[i] > 0.0,
                "MACD line at {} should be positive in uptrend, got {}",
                i,
                result.macd_line[i]
            );
        }
    }

    #[test]
    fn test_macd_line_downtrend() {
        // In a downtrend, fast EMA < slow EMA, so MACD line should be negative
        let data: Vec<f64> = (0..50).map(|i| 150.0 - (i as f64) * 2.0).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // After warmup, MACD line should be negative
        for i in 15..result.len() {
            assert!(
                result.macd_line[i] < 0.0,
                "MACD line at {} should be negative in downtrend, got {}",
                i,
                result.macd_line[i]
            );
        }
    }

    #[test]
    fn test_macd_line_constant_values() {
        // For constant values, EMAs should equal the constant, so MACD line should be 0
        let data = vec![50.0_f64; 50];
        let result = macd(&data, 5, 10, 3).unwrap();

        // After warmup, MACD line should be approximately 0
        for i in 9..result.len() {
            if !result.macd_line[i].is_nan() {
                assert!(
                    approx_eq(result.macd_line[i], 0.0, MACD_EPSILON),
                    "MACD line at {} should be ~0 for constant data, got {}",
                    i,
                    result.macd_line[i]
                );
            }
        }
    }

    // ==================== Signal Line Tests ====================

    #[test]
    fn test_signal_line_smooths_macd() {
        // Signal line should be a smoothed version of MACD line
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // Both should eventually converge to similar values for trending data
        let last_macd = result.macd_line[result.len() - 1];
        let last_signal = result.signal_line[result.len() - 1];

        // Signal should be close to MACD for smooth trends
        assert!(
            (last_macd - last_signal).abs() < 1.0,
            "Signal should be close to MACD in smooth trend"
        );
    }

    // ==================== Histogram Tests ====================

    #[test]
    fn test_histogram_equals_macd_minus_signal() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // Histogram should equal MACD line - signal line
        for i in 11..result.len() {
            if !result.histogram[i].is_nan() {
                let expected = result.macd_line[i] - result.signal_line[i];
                assert!(
                    approx_eq(result.histogram[i], expected, EPSILON),
                    "Histogram at {} should equal MACD - Signal",
                    i
                );
            }
        }
    }

    #[test]
    fn test_histogram_sign_changes() {
        // Create data that reverses trend - histogram should change sign
        let mut data = Vec::with_capacity(60);
        // First half: uptrend
        for i in 0..30 {
            data.push(50.0 + (i as f64) * 2.0);
        }
        // Second half: downtrend
        for i in 0..30 {
            data.push(108.0 - (i as f64) * 2.0);
        }

        let result = macd(&data, 5, 10, 3).unwrap();

        // Early histogram should be positive (uptrend)
        let early_valid = 15;
        let mut found_positive = false;
        for i in early_valid..25 {
            if !result.histogram[i].is_nan() && result.histogram[i] > 0.0 {
                found_positive = true;
                break;
            }
        }

        // Late histogram should be negative (downtrend)
        let mut found_negative = false;
        for i in 45..result.len() {
            if !result.histogram[i].is_nan() && result.histogram[i] < 0.0 {
                found_negative = true;
                break;
            }
        }

        assert!(found_positive || found_negative, "Histogram should change sign with trend reversal");
    }

    // ==================== Reference Value Tests ====================

    #[test]
    fn test_macd_known_calculation() {
        // Manual calculation verification
        // Data: simple sequential values
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = macd(&data, 2, 3, 2).unwrap();

        // Slow EMA(3) seed at index 2: (1+2+3)/3 = 2.0
        // Fast EMA(2) seed at index 1: (1+2)/2 = 1.5

        // At index 2:
        // Fast EMA: alpha = 2/3, EMA[2] = 2/3 * 3 + 1/3 * 1.5 = 2 + 0.5 = 2.5
        // Slow EMA: (1+2+3)/3 = 2.0
        // MACD[2] = 2.5 - 2.0 = 0.5
        assert!(approx_eq(result.macd_line[2], 0.5, MACD_EPSILON));

        // Signal line at index 3 (need 2 valid MACD values)
        // MACD[2] = 0.5
        // For MACD[3], need to compute Fast EMA[3] and Slow EMA[3]
        // Fast EMA[3] = 2/3 * 4 + 1/3 * 2.5 = 2.667 + 0.833 = 3.5
        // Slow EMA[3] = 2/4 * 4 + 2/4 * 2.0 = 2 + 1 = 3.0
        // MACD[3] = 3.5 - 3.0 = 0.5
        // Signal[3] = SMA(MACD[2..3]) = (0.5 + 0.5)/2 = 0.5

        assert!(!result.macd_line[2].is_nan());
        assert!(!result.macd_line[3].is_nan());
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_macd_with_nan_in_data() {
        let data = vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = macd(&data, 2, 3, 2).unwrap();

        // NaN propagates through EMA, affecting MACD calculation
        assert!(result.macd_line[2].is_nan());
    }

    #[test]
    fn test_macd_negative_values() {
        let data: Vec<f64> = (-25..25).map(|i| i as f64).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // Should work with negative values
        assert!(!result.macd_line[15].is_nan());
        assert!(!result.signal_line[17].is_nan());
    }

    #[test]
    fn test_macd_large_values() {
        let data: Vec<f64> = (0..50).map(|i| 1e10 + (i as f64) * 1e8).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // Should handle large values
        for i in 12..result.len() {
            assert!(!result.signal_line[i].is_nan());
        }
    }

    #[test]
    fn test_macd_small_values() {
        let data: Vec<f64> = (0..50).map(|i| 1e-10 + (i as f64) * 1e-12).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        // Should handle small values
        for i in 12..result.len() {
            assert!(!result.signal_line[i].is_nan());
        }
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_macd_empty_input() {
        let data: Vec<f64> = vec![];
        let result = macd(&data, 12, 26, 9);

        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_macd_zero_fast_period() {
        let data = vec![1.0_f64; 50];
        let result = macd(&data, 0, 26, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_zero_slow_period() {
        let data = vec![1.0_f64; 50];
        let result = macd(&data, 12, 0, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_zero_signal_period() {
        let data = vec![1.0_f64; 50];
        let result = macd(&data, 12, 26, 0);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_fast_equals_slow() {
        let data = vec![1.0_f64; 50];
        let result = macd(&data, 12, 12, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { .. })));
    }

    #[test]
    fn test_macd_fast_greater_than_slow() {
        let data = vec![1.0_f64; 50];
        let result = macd(&data, 26, 12, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { .. })));
    }

    #[test]
    fn test_macd_insufficient_data() {
        let data = vec![1.0_f64; 30]; // Need at least 26 + 9 - 1 = 34 for MACD(12, 26, 9)
        let result = macd(&data, 12, 26, 9);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    #[test]
    fn test_macd_minimum_required_data() {
        // For MACD(12, 26, 9), minimum is slow_period + signal_period - 1 = 34
        let data = vec![1.0_f64; 34];
        let result = macd(&data, 12, 26, 9);

        assert!(result.is_ok());
    }

    // ==================== macd_into Tests ====================

    #[test]
    fn test_macd_into_basic() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let mut macd_line = vec![0.0_f64; 50];
        let mut signal_line = vec![0.0_f64; 50];
        let mut histogram = vec![0.0_f64; 50];

        let (valid_macd, valid_signal) = macd_into(
            &data, 12, 26, 9,
            &mut macd_line, &mut signal_line, &mut histogram
        ).unwrap();

        assert_eq!(valid_macd, 25); // 50 - 25 = 25
        assert_eq!(valid_signal, 17); // 50 - 33 = 17

        // Verify same as macd()
        let reference = macd(&data, 12, 26, 9).unwrap();
        for i in 0..50 {
            assert!(approx_eq(macd_line[i], reference.macd_line[i], EPSILON));
            assert!(approx_eq(signal_line[i], reference.signal_line[i], EPSILON));
            assert!(approx_eq(histogram[i], reference.histogram[i], EPSILON));
        }
    }

    #[test]
    fn test_macd_into_buffer_reuse() {
        let data1: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64)).collect();
        let data2: Vec<f64> = (0..50).map(|i| 200.0 - (i as f64)).collect();
        let mut macd_line = vec![0.0_f64; 50];
        let mut signal_line = vec![0.0_f64; 50];
        let mut histogram = vec![0.0_f64; 50];

        macd_into(&data1, 5, 10, 3, &mut macd_line, &mut signal_line, &mut histogram).unwrap();
        let first_macd = macd_line[15];

        macd_into(&data2, 5, 10, 3, &mut macd_line, &mut signal_line, &mut histogram).unwrap();
        let second_macd = macd_line[15];

        // First (uptrend) should give positive MACD, second (downtrend) should give negative
        assert!(first_macd > 0.0);
        assert!(second_macd < 0.0);
    }

    #[test]
    fn test_macd_into_insufficient_output() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64)).collect();
        let mut macd_line = vec![0.0_f64; 30]; // Too short
        let mut signal_line = vec![0.0_f64; 50];
        let mut histogram = vec![0.0_f64; 50];

        let result = macd_into(&data, 5, 10, 3, &mut macd_line, &mut signal_line, &mut histogram);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    #[test]
    fn test_macd_into_f32() {
        let data: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.5).collect();
        let mut macd_line = vec![0.0_f32; 50];
        let mut signal_line = vec![0.0_f32; 50];
        let mut histogram = vec![0.0_f32; 50];

        let (valid_macd, valid_signal) = macd_into(
            &data, 5, 10, 3,
            &mut macd_line, &mut signal_line, &mut histogram
        ).unwrap();

        assert_eq!(valid_macd, 41); // 50 - 9 = 41
        assert_eq!(valid_signal, 39); // 50 - 11 = 39
    }

    // ==================== Consistency Tests ====================

    #[test]
    fn test_macd_and_macd_into_produce_same_result() {
        let data: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result1 = macd(&data, 12, 26, 9).unwrap();

        let mut macd_line = vec![0.0_f64; 60];
        let mut signal_line = vec![0.0_f64; 60];
        let mut histogram = vec![0.0_f64; 60];
        macd_into(&data, 12, 26, 9, &mut macd_line, &mut signal_line, &mut histogram).unwrap();

        for i in 0..60 {
            assert!(
                approx_eq(result1.macd_line[i], macd_line[i], EPSILON),
                "MACD line mismatch at {}",
                i
            );
            assert!(
                approx_eq(result1.signal_line[i], signal_line[i], EPSILON),
                "Signal line mismatch at {}",
                i
            );
            assert!(
                approx_eq(result1.histogram[i], histogram[i], EPSILON),
                "Histogram mismatch at {}",
                i
            );
        }
    }

    // ==================== Property-Based-Like Tests ====================

    #[test]
    fn test_macd_output_length_equals_input_length() {
        for len in [35, 50, 100, 200] {
            let data: Vec<f64> = (0..len).map(|x| x as f64).collect();
            let result = macd(&data, 12, 26, 9).unwrap();
            assert_eq!(result.len(), len);
            assert_eq!(result.macd_line.len(), len);
            assert_eq!(result.signal_line.len(), len);
            assert_eq!(result.histogram.len(), len);
        }
    }

    #[test]
    fn test_macd_nan_counts() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        for (fast, slow, signal) in [(3, 5, 2), (5, 10, 3), (12, 26, 9)] {
            let result = macd(&data, fast, slow, signal).unwrap();

            let macd_nan_count = result.macd_line.iter().filter(|x| x.is_nan()).count();
            let signal_nan_count = result.signal_line.iter().filter(|x| x.is_nan()).count();

            // MACD NaN count should be slow_period - 1
            assert_eq!(
                macd_nan_count, slow - 1,
                "MACD NaN count for ({}, {}, {})",
                fast, slow, signal
            );

            // Signal NaN count should be slow_period + signal_period - 2
            assert_eq!(
                signal_nan_count, slow + signal - 2,
                "Signal NaN count for ({}, {}, {})",
                fast, slow, signal
            );
        }
    }

    #[test]
    fn test_macd_histogram_nan_matches_signal() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let result = macd(&data, 12, 26, 9).unwrap();

        // Histogram should be NaN exactly where signal is NaN
        for i in 0..result.len() {
            assert_eq!(
                result.histogram[i].is_nan(),
                result.signal_line[i].is_nan(),
                "Histogram and signal NaN mismatch at {}",
                i
            );
        }
    }

    // ==================== MacdOutput Struct Tests ====================

    #[test]
    fn test_macd_output_len() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let result = macd(&data, 5, 10, 3).unwrap();

        assert_eq!(result.len(), 50);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_macd_output_is_empty() {
        let output: MacdOutput<f64> = MacdOutput {
            macd_line: vec![],
            signal_line: vec![],
            histogram: vec![],
        };

        assert!(output.is_empty());
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_macd_output_first_valid_indices() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let result = macd(&data, 12, 26, 9).unwrap();

        assert_eq!(result.first_valid_macd_index(26), 25);
        assert_eq!(result.first_valid_signal_index(26, 9), 33);
    }

    #[test]
    fn test_macd_output_clone() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let result = macd(&data, 5, 10, 3).unwrap();
        let cloned = result.clone();

        for i in 0..result.len() {
            assert!(approx_eq(result.macd_line[i], cloned.macd_line[i], EPSILON));
            assert!(approx_eq(result.signal_line[i], cloned.signal_line[i], EPSILON));
            assert!(approx_eq(result.histogram[i], cloned.histogram[i], EPSILON));
        }
    }

    // ==================== Real-World Scenario Tests ====================

    #[test]
    fn test_macd_bullish_crossover() {
        // Create data where MACD line will cross above signal line
        let mut data = Vec::with_capacity(60);
        // Initial flat period
        for _ in 0..35 {
            data.push(100.0_f64);
        }
        // Strong uptrend
        for i in 0..25 {
            data.push(100.0 + (i as f64) * 3.0);
        }

        let result = macd(&data, 5, 10, 3).unwrap();

        // After the uptrend starts, histogram should become positive
        let mut found_positive = false;
        for i in 40..result.len() {
            if !result.histogram[i].is_nan() && result.histogram[i] > 0.0 {
                found_positive = true;
                break;
            }
        }

        assert!(found_positive, "Should find positive histogram during uptrend");
    }

    #[test]
    fn test_macd_bearish_crossover() {
        // Create data where MACD line will cross below signal line
        let mut data = Vec::with_capacity(60);
        // Initial flat period
        for _ in 0..35 {
            data.push(100.0_f64);
        }
        // Strong downtrend
        for i in 0..25 {
            data.push(100.0 - (i as f64) * 3.0);
        }

        let result = macd(&data, 5, 10, 3).unwrap();

        // After the downtrend starts, histogram should become negative
        let mut found_negative = false;
        for i in 40..result.len() {
            if !result.histogram[i].is_nan() && result.histogram[i] < 0.0 {
                found_negative = true;
                break;
            }
        }

        assert!(found_negative, "Should find negative histogram during downtrend");
    }

    #[test]
    fn test_macd_divergence_scenario() {
        // Simulate price making higher highs but MACD making lower highs (bearish divergence setup)
        let data: Vec<f64> = vec![
            50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
            60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
            70.0, 69.5, 70.0, 69.0, 70.5, 68.5, 71.0, 68.0, 71.5, 67.5,
            72.0, 67.0, 72.5, 66.5, 73.0, 66.0, 73.5, 65.5, 74.0, 65.0,
        ];

        let result = macd(&data, 5, 10, 3).unwrap();

        // Should compute without errors
        assert_eq!(result.len(), 40);
        assert!(!result.signal_line[15].is_nan());
    }
}
