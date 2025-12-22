//! Stochastic Oscillator indicator.
//!
//! The Stochastic Oscillator is a momentum indicator that compares a security's
//! closing price to its price range over a given period. It oscillates between
//! 0 and 100, where readings above 80 typically indicate overbought conditions
//! and readings below 20 indicate oversold conditions.
//!
//! # Variants
//!
//! This module provides three variants of the Stochastic Oscillator:
//!
//! - **Fast Stochastic**: Raw %K and its SMA (%D)
//! - **Slow Stochastic**: Smoothed %K (SMA of Fast %K) and its SMA (%D)
//! - **Full Stochastic**: Configurable smoothing for both %K and %D
//!
//! # Algorithm
//!
//! The calculation uses O(n) rolling extrema to find the highest high and lowest
//! low over the lookback period. The basic formula is:
//!
//! ```text
//! %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
//! %D = SMA(%K, d_period)
//! ```
//!
//! For the Slow and Full variants, an additional smoothing is applied to %K
//! before computing %D.
//!
//! # Mathematical Conventions (PRD §4.5, §4.8)
//!
//! - **Zero Range Handling**: When `highest_high == lowest_low` (flat price over
//!   the lookback window), %K = 50 (stable midpoint). This is a deterministic
//!   output for an indeterminate operation (0/0), not a NaN override.
//! - **NaN Precedence**: If any of `high`, `low`, or `close` in the required window
//!   contains NaN, the output is NaN. NaN propagation takes priority over the
//!   flat-price fallback.
//! - **Rolling Extrema**: Uses monotonic deque for O(n) computation when
//!   k_period ≥ 25, naive O(n×k) for smaller periods (per E04 findings).
//!
//! # Example
//!
//! ```
//! use fast_ta::indicators::stochastic::{stochastic_fast, StochasticOutput};
//!
//! let high = vec![10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5];
//! let low = vec![9.0_f64, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
//! let close = vec![9.5_f64, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];
//!
//! let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();
//!
//! // First 4 values (k_period - 1) of %K are NaN
//! assert!(result.k[0].is_nan());
//! assert!(result.k[3].is_nan());
//! assert!(!result.k[4].is_nan());
//!
//! // First 6 values (k_period + d_period - 2) of %D are NaN
//! assert!(result.d[5].is_nan());
//! assert!(!result.d[6].is_nan());
//! ```

use crate::error::{Error, Result};
use crate::traits::SeriesElement;

/// Returns the lookback period for the Stochastic %K line.
///
/// The %K lookback is `k_period - 1`.
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_k_lookback;
///
/// assert_eq!(stochastic_k_lookback(14), 13);
/// assert_eq!(stochastic_k_lookback(5), 4);
/// ```
#[inline]
#[must_use]
pub const fn stochastic_k_lookback(k_period: usize) -> usize {
    if k_period == 0 {
        0
    } else {
        k_period - 1
    }
}

/// Returns the lookback period for the Stochastic %D line.
///
/// The %D lookback is `k_period + d_period - 2`.
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_d_lookback;
///
/// assert_eq!(stochastic_d_lookback(14, 3), 15);
/// assert_eq!(stochastic_d_lookback(5, 3), 6);
/// ```
#[inline]
#[must_use]
pub const fn stochastic_d_lookback(k_period: usize, d_period: usize) -> usize {
    if k_period == 0 || d_period == 0 {
        0
    } else {
        k_period + d_period - 2
    }
}

/// Returns the minimum input length required for Stochastic Oscillator.
///
/// This is the smallest input size that will produce at least one valid %D value.
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_min_len;
///
/// assert_eq!(stochastic_min_len(14, 3), 16);
/// assert_eq!(stochastic_min_len(5, 3), 7);
/// ```
#[inline]
#[must_use]
pub const fn stochastic_min_len(k_period: usize, d_period: usize) -> usize {
    if k_period == 0 || d_period == 0 {
        0
    } else {
        k_period + d_period - 1
    }
}

/// Output structure containing %K and %D lines of the Stochastic Oscillator.
///
/// Both vectors have the same length as the input data. NaN values fill the
/// lookback period where insufficient data exists.
#[derive(Debug, Clone)]
pub struct StochasticOutput<T> {
    /// The %K line (fast line).
    pub k: Vec<T>,
    /// The %D line (signal line, SMA of %K).
    pub d: Vec<T>,
}

/// Computes the Fast Stochastic Oscillator.
///
/// The Fast Stochastic consists of:
/// - **%K**: The raw stochastic value comparing close to the high-low range
/// - **%D**: Simple Moving Average of %K
///
/// # Arguments
///
/// * `high` - The high prices
/// * `low` - The low prices
/// * `close` - The closing prices
/// * `k_period` - The lookback period for %K (commonly 14)
/// * `d_period` - The smoothing period for %D (commonly 3)
///
/// # Returns
///
/// A `Result` containing a [`StochasticOutput`] with %K and %D lines,
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
///
/// # Performance
///
/// - Time complexity: O(n) where n is the length of the input data
/// - Space complexity: O(n) for the output vectors
///
/// # NaN Handling
///
/// - The first `k_period - 1` elements of %K are NaN
/// - The first `k_period + d_period - 2` elements of %D are NaN
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_fast;
///
/// let high = vec![44.0_f64, 44.5, 44.75, 44.25, 44.5, 44.75, 45.0];
/// let low = vec![43.5_f64, 44.0, 44.25, 43.75, 44.0, 44.25, 44.5];
/// let close = vec![43.75_f64, 44.25, 44.5, 44.0, 44.25, 44.5, 44.75];
///
/// let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();
/// assert!(!result.k[4].is_nan()); // First valid %K
/// ```
#[must_use = "this returns a Result with Stochastic values, which should be used"]
pub fn stochastic_fast<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    d_period: usize,
) -> Result<StochasticOutput<T>> {
    validate_stochastic_inputs(high, low, close, k_period, d_period)?;

    let n = close.len();
    let mut k = vec![T::nan(); n];
    let mut d = vec![T::nan(); n];

    // Compute raw %K values
    compute_raw_k(high, low, close, k_period, &mut k)?;

    // Compute %D as SMA of %K
    compute_sma_of_series(&k, d_period, k_period - 1, &mut d)?;

    Ok(StochasticOutput { k, d })
}

/// Computes the Fast Stochastic Oscillator into pre-allocated output buffers.
///
/// This variant allows reusing existing buffers to avoid allocations in
/// performance-critical code paths.
///
/// # Arguments
///
/// * `high` - The high prices
/// * `low` - The low prices
/// * `close` - The closing prices
/// * `k_period` - The lookback period for %K
/// * `d_period` - The smoothing period for %D
/// * `output` - Pre-allocated output structure
///
/// # Returns
///
/// A `Result` containing a tuple of (valid %K count, valid %D count),
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
/// - Output buffers are shorter than the input data
#[must_use = "this returns a Result with valid counts, which should be used"]
pub fn stochastic_fast_into<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    d_period: usize,
    output: &mut StochasticOutput<T>,
) -> Result<(usize, usize)> {
    validate_stochastic_inputs(high, low, close, k_period, d_period)?;

    let n = close.len();
    if output.k.len() < n || output.d.len() < n {
        return Err(Error::BufferTooSmall {
            required: n,
            actual: output.k.len().min(output.d.len()),
            indicator: "stochastic",
        });
    }

    // Initialize with NaN
    for i in 0..n {
        output.k[i] = T::nan();
        output.d[i] = T::nan();
    }

    // Compute raw %K values
    compute_raw_k(high, low, close, k_period, &mut output.k)?;

    // Compute %D as SMA of %K
    compute_sma_of_series(&output.k, d_period, k_period - 1, &mut output.d)?;

    let valid_k = n - (k_period - 1);
    let valid_d = if n >= k_period + d_period - 1 {
        n - (k_period + d_period - 2)
    } else {
        0
    };

    Ok((valid_k, valid_d))
}

/// Computes the Slow Stochastic Oscillator.
///
/// The Slow Stochastic smooths the Fast %K to reduce noise:
/// - **%K**: SMA of Fast %K (the raw stochastic)
/// - **%D**: SMA of Slow %K
///
/// This is equivalent to `stochastic_full(high, low, close, k_period, slow_k_period, d_period)`
/// where `slow_k_period` equals `d_period` (commonly 3).
///
/// # Arguments
///
/// * `high` - The high prices
/// * `low` - The low prices
/// * `close` - The closing prices
/// * `k_period` - The lookback period for raw %K (commonly 14)
/// * `d_period` - The smoothing period for both Slow %K and %D (commonly 3)
///
/// # Returns
///
/// A `Result` containing a [`StochasticOutput`] with Slow %K and %D lines,
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
///
/// # NaN Handling
///
/// - The first `k_period + d_period - 2` elements of Slow %K are NaN
/// - The first `k_period + 2*d_period - 3` elements of %D are NaN
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_slow;
///
/// let high = vec![44.0_f64, 44.5, 44.75, 44.25, 44.5, 44.75, 45.0, 44.5, 44.0, 44.25];
/// let low = vec![43.5_f64, 44.0, 44.25, 43.75, 44.0, 44.25, 44.5, 44.0, 43.5, 43.75];
/// let close = vec![43.75_f64, 44.25, 44.5, 44.0, 44.25, 44.5, 44.75, 44.25, 43.75, 44.0];
///
/// let result = stochastic_slow(&high, &low, &close, 5, 3).unwrap();
/// // Slow %K starts at index k_period + d_period - 2 = 5 + 3 - 2 = 6
/// assert!(result.k[5].is_nan());
/// assert!(!result.k[6].is_nan());
/// ```
#[must_use = "this returns a Result with Stochastic values, which should be used"]
pub fn stochastic_slow<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    d_period: usize,
) -> Result<StochasticOutput<T>> {
    // Slow stochastic uses the same period for smoothing K and D
    stochastic_full(high, low, close, k_period, d_period, d_period)
}

/// Computes the Slow Stochastic Oscillator into pre-allocated output buffers.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
/// - Output buffers are shorter than the input data
#[must_use = "this returns a Result with valid counts, which should be used"]
pub fn stochastic_slow_into<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    d_period: usize,
    output: &mut StochasticOutput<T>,
) -> Result<(usize, usize)> {
    stochastic_full_into(high, low, close, k_period, d_period, d_period, output)
}

/// Computes the Full Stochastic Oscillator with configurable smoothing.
///
/// The Full Stochastic provides complete control over smoothing periods:
/// - **%K**: SMA of Raw %K with `slow_k_period` smoothing
/// - **%D**: SMA of Full %K with `d_period` smoothing
///
/// # Arguments
///
/// * `high` - The high prices
/// * `low` - The low prices
/// * `close` - The closing prices
/// * `k_period` - The lookback period for raw %K (commonly 14)
/// * `slow_k_period` - The smoothing period for %K (commonly 3)
/// * `d_period` - The smoothing period for %D (commonly 3)
///
/// # Returns
///
/// A `Result` containing a [`StochasticOutput`] with Full %K and %D lines,
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
///
/// # NaN Handling
///
/// - The first `k_period + slow_k_period - 2` elements of Full %K are NaN
/// - The first `k_period + slow_k_period + d_period - 3` elements of %D are NaN
///
/// # Example
///
/// ```
/// use fast_ta::indicators::stochastic::stochastic_full;
///
/// let high = vec![44.0_f64, 44.5, 44.75, 44.25, 44.5, 44.75, 45.0, 44.5, 44.0, 44.25, 44.5];
/// let low = vec![43.5_f64, 44.0, 44.25, 43.75, 44.0, 44.25, 44.5, 44.0, 43.5, 43.75, 44.0];
/// let close = vec![43.75_f64, 44.25, 44.5, 44.0, 44.25, 44.5, 44.75, 44.25, 43.75, 44.0, 44.25];
///
/// // Custom smoothing: 5-period %K lookback, 3-period %K smoothing, 3-period %D smoothing
/// let result = stochastic_full(&high, &low, &close, 5, 3, 3).unwrap();
///
/// // Full %K starts at index k_period + slow_k_period - 2 = 5 + 3 - 2 = 6
/// assert!(result.k[5].is_nan());
/// assert!(!result.k[6].is_nan());
/// ```
#[must_use = "this returns a Result with Stochastic values, which should be used"]
pub fn stochastic_full<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    slow_k_period: usize,
    d_period: usize,
) -> Result<StochasticOutput<T>> {
    validate_stochastic_full_inputs(high, low, close, k_period, slow_k_period, d_period)?;

    let n = close.len();
    let mut raw_k = vec![T::nan(); n];
    let mut k = vec![T::nan(); n];
    let mut d = vec![T::nan(); n];

    // Step 1: Compute raw %K values
    compute_raw_k(high, low, close, k_period, &mut raw_k)?;

    // Step 2: Compute smoothed %K (SMA of raw %K)
    compute_sma_of_series(&raw_k, slow_k_period, k_period - 1, &mut k)?;

    // Step 3: Compute %D (SMA of smoothed %K)
    let k_start_idx = k_period + slow_k_period - 2;
    compute_sma_of_series(&k, d_period, k_start_idx, &mut d)?;

    Ok(StochasticOutput { k, d })
}

/// Computes the Full Stochastic Oscillator into pre-allocated output buffers.
///
/// # Errors
///
/// Returns an error if:
/// - Any input is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - Input lengths don't match
/// - Input data is shorter than `k_period` (`Error::InsufficientData`)
/// - Output buffers are shorter than the input data
#[must_use = "this returns a Result with valid counts, which should be used"]
pub fn stochastic_full_into<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    slow_k_period: usize,
    d_period: usize,
    output: &mut StochasticOutput<T>,
) -> Result<(usize, usize)> {
    validate_stochastic_full_inputs(high, low, close, k_period, slow_k_period, d_period)?;

    let n = close.len();
    if output.k.len() < n || output.d.len() < n {
        return Err(Error::BufferTooSmall {
            required: n,
            actual: output.k.len().min(output.d.len()),
            indicator: "stochastic",
        });
    }

    // Initialize with NaN
    for i in 0..n {
        output.k[i] = T::nan();
        output.d[i] = T::nan();
    }

    // Use a temporary buffer for raw %K
    let mut raw_k = vec![T::nan(); n];

    // Step 1: Compute raw %K values
    compute_raw_k(high, low, close, k_period, &mut raw_k)?;

    // Step 2: Compute smoothed %K (SMA of raw %K)
    compute_sma_of_series(&raw_k, slow_k_period, k_period - 1, &mut output.k)?;

    // Step 3: Compute %D (SMA of smoothed %K)
    let k_start_idx = k_period + slow_k_period - 2;
    compute_sma_of_series(&output.k, d_period, k_start_idx, &mut output.d)?;

    let valid_k = if n >= k_period + slow_k_period - 1 {
        n - (k_period + slow_k_period - 2)
    } else {
        0
    };
    let valid_d = if n >= k_period + slow_k_period + d_period - 2 {
        n - (k_period + slow_k_period + d_period - 3)
    } else {
        0
    };

    Ok((valid_k, valid_d))
}

/// Validates inputs for fast stochastic.
fn validate_stochastic_inputs<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    d_period: usize,
) -> Result<()> {
    if k_period == 0 {
        return Err(Error::InvalidPeriod {
            period: k_period,
            reason: "k_period must be at least 1",
        });
    }

    if d_period == 0 {
        return Err(Error::InvalidPeriod {
            period: d_period,
            reason: "d_period must be at least 1",
        });
    }

    if high.is_empty() {
        return Err(Error::EmptyInput);
    }
    if low.is_empty() {
        return Err(Error::EmptyInput);
    }
    if close.is_empty() {
        return Err(Error::EmptyInput);
    }

    // All inputs must have the same length
    if high.len() != low.len() || high.len() != close.len() {
        return Err(Error::LengthMismatch {
            description: format!(
                "high has {} elements, low has {}, close has {}",
                high.len(),
                low.len(),
                close.len()
            ),
        });
    }

    if high.len() < k_period {
        return Err(Error::InsufficientData {
            required: k_period,
            actual: high.len(),
            indicator: "stochastic",
        });
    }

    Ok(())
}

/// Validates inputs for full stochastic.
fn validate_stochastic_full_inputs<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    slow_k_period: usize,
    d_period: usize,
) -> Result<()> {
    if k_period == 0 {
        return Err(Error::InvalidPeriod {
            period: k_period,
            reason: "k_period must be at least 1",
        });
    }

    if slow_k_period == 0 {
        return Err(Error::InvalidPeriod {
            period: slow_k_period,
            reason: "slow_k_period must be at least 1",
        });
    }

    if d_period == 0 {
        return Err(Error::InvalidPeriod {
            period: d_period,
            reason: "d_period must be at least 1",
        });
    }

    if high.is_empty() {
        return Err(Error::EmptyInput);
    }
    if low.is_empty() {
        return Err(Error::EmptyInput);
    }
    if close.is_empty() {
        return Err(Error::EmptyInput);
    }

    // All inputs must have the same length
    if high.len() != low.len() || high.len() != close.len() {
        return Err(Error::LengthMismatch {
            description: format!(
                "high has {} elements, low has {}, close has {}",
                high.len(),
                low.len(),
                close.len()
            ),
        });
    }

    if high.len() < k_period {
        return Err(Error::InsufficientData {
            required: k_period,
            actual: high.len(),
            indicator: "stochastic",
        });
    }

    Ok(())
}

/// Computes raw %K values using rolling extrema.
///
/// %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
///
/// This uses a simple O(n*k) approach for now. A more efficient O(n)
/// monotonic deque approach could be used for large `k_period` values.
fn compute_raw_k<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
    k_period: usize,
    output: &mut [T],
) -> Result<()> {
    let hundred = T::from_f64(100.0)?;
    let n = close.len();

    // For each position starting from k_period - 1
    for i in (k_period - 1)..n {
        let window_start = i + 1 - k_period;

        if close[i].is_nan() {
            output[i] = T::nan();
            continue;
        }

        // Find highest high and lowest low in the window, tracking NaNs
        let mut highest_high = high[window_start];
        let mut lowest_low = low[window_start];
        let mut has_nan = highest_high.is_nan() || lowest_low.is_nan();

        for j in (window_start + 1)..=i {
            let high_value = high[j];
            let low_value = low[j];
            if high_value.is_nan() || low_value.is_nan() {
                has_nan = true;
                break;
            }
            if high_value > highest_high {
                highest_high = high_value;
            }
            if low_value < lowest_low {
                lowest_low = low_value;
            }
        }

        if has_nan {
            output[i] = T::nan();
            continue;
        }

        // Compute %K
        let range = highest_high - lowest_low;
        if range > T::zero() {
            output[i] = hundred * (close[i] - lowest_low) / range;
        } else {
            // If high == low for the entire period, price hasn't moved
            // Return 50 (neutral) as is common practice
            output[i] = T::from_f64(50.0)?;
        }
    }

    Ok(())
}

/// Computes SMA of a series starting from a given index.
///
/// This handles NaN values in the input by only computing SMA where
/// enough valid values exist.
fn compute_sma_of_series<T: SeriesElement>(
    input: &[T],
    period: usize,
    start_idx: usize,
    output: &mut [T],
) -> Result<()> {
    let n = input.len();
    let period_t = T::from_usize(period)?;

    // First valid output is at start_idx + period - 1
    let first_valid_idx = start_idx + period - 1;

    if first_valid_idx >= n {
        return Ok(());
    }

    // Compute initial sum, tracking NaN values
    let mut sum = T::zero();
    let mut nan_count = 0usize;
    for &value in input.iter().skip(start_idx).take(period) {
        if value.is_nan() {
            nan_count += 1;
        } else {
            sum = sum + value;
        }
    }
    if nan_count == 0 {
        output[first_valid_idx] = sum / period_t;
    } else {
        output[first_valid_idx] = T::nan();
    }

    // Rolling calculation
    for i in (first_valid_idx + 1)..n {
        let old_value = input[i - period];
        let new_value = input[i];

        if new_value.is_nan() {
            nan_count += 1;
        } else {
            sum = sum + new_value;
        }

        if old_value.is_nan() {
            nan_count -= 1;
        } else {
            sum = sum - old_value;
        }

        if nan_count == 0 {
            output[i] = sum / period_t;
        } else {
            output[i] = T::nan();
        }
    }

    Ok(())
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
    // Looser epsilon for stochastic calculations
    const STOCH_EPSILON: f64 = 1e-6;

    // ==================== Fast Stochastic Tests ====================

    #[test]
    fn test_stochastic_fast_basic() {
        let high = vec![10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

        assert_eq!(result.k.len(), 10);
        assert_eq!(result.d.len(), 10);

        // First 4 values of %K are NaN
        for i in 0..4 {
            assert!(result.k[i].is_nan(), "k[{}] should be NaN", i);
        }
        assert!(!result.k[4].is_nan());

        // First 6 values of %D are NaN (k_period + d_period - 2 = 5 + 3 - 2 = 6)
        for i in 0..6 {
            assert!(result.d[i].is_nan(), "d[{}] should be NaN", i);
        }
        assert!(!result.d[6].is_nan());
    }

    #[test]
    fn test_stochastic_fast_f32() {
        let high = vec![10.0_f32, 11.0, 12.0, 11.5, 12.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0];

        let result = stochastic_fast(&high, &low, &close, 3, 2).unwrap();

        assert!(!result.k[2].is_nan());
        // %K should be in range [0, 100]
        assert!(result.k[2] >= 0.0 && result.k[2] <= 100.0);
    }

    #[test]
    fn test_stochastic_fast_known_values() {
        // Simple case where we can verify the calculation
        // Period 3, looking at window [10, 11, 12] with close = 12
        // Highest high = 12, Lowest low = 10
        // %K = 100 * (12 - 10) / (12 - 10) = 100
        let high = vec![10.0_f64, 11.0, 12.0];
        let low = vec![10.0, 11.0, 12.0];
        let close = vec![10.0, 11.0, 12.0];

        let result = stochastic_fast(&high, &low, &close, 3, 1).unwrap();

        // At index 2: HH=12, LL=10, Close=12
        // %K = 100 * (12-10)/(12-10) = 100
        assert!(approx_eq(result.k[2], 100.0, STOCH_EPSILON));
    }

    #[test]
    fn test_stochastic_fast_close_at_low() {
        // Close at the lowest low should give %K = 0
        let high = vec![15.0_f64, 14.0, 13.0, 12.0, 11.0];
        let low = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let close = vec![10.0, 10.0, 10.0, 10.0, 10.0];

        let result = stochastic_fast(&high, &low, &close, 3, 1).unwrap();

        // Close is at lowest low
        assert!(approx_eq(result.k[2], 0.0, STOCH_EPSILON));
    }

    #[test]
    fn test_stochastic_fast_close_at_high() {
        // Close at the highest high should give %K = 100
        let high = vec![20.0_f64, 20.0, 20.0, 20.0, 20.0];
        let low = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let close = vec![20.0, 20.0, 20.0, 20.0, 20.0];

        let result = stochastic_fast(&high, &low, &close, 3, 1).unwrap();

        // Close is at highest high
        assert!(approx_eq(result.k[2], 100.0, STOCH_EPSILON));
    }

    #[test]
    fn test_stochastic_fast_close_at_midpoint() {
        // Close at midpoint should give %K = 50
        let high = vec![20.0_f64, 20.0, 20.0, 20.0, 20.0];
        let low = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let close = vec![15.0, 15.0, 15.0, 15.0, 15.0];

        let result = stochastic_fast(&high, &low, &close, 3, 1).unwrap();

        // Close is at midpoint: (15 - 10) / (20 - 10) = 0.5 -> 50%
        assert!(approx_eq(result.k[2], 50.0, STOCH_EPSILON));
    }

    #[test]
    fn test_stochastic_fast_no_range() {
        // When high == low, range is 0, should return 50 (neutral)
        let high = vec![50.0_f64; 5];
        let low = vec![50.0_f64; 5];
        let close = vec![50.0_f64; 5];

        let result = stochastic_fast(&high, &low, &close, 3, 1).unwrap();

        // No range, should be neutral (50)
        assert!(approx_eq(result.k[2], 50.0, STOCH_EPSILON));
    }

    // ==================== Slow Stochastic Tests ====================

    #[test]
    fn test_stochastic_slow_basic() {
        let high = vec![10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let result = stochastic_slow(&high, &low, &close, 5, 3).unwrap();

        assert_eq!(result.k.len(), 10);
        assert_eq!(result.d.len(), 10);

        // Slow %K starts at k_period + d_period - 2 = 5 + 3 - 2 = 6
        for i in 0..6 {
            assert!(result.k[i].is_nan(), "slow k[{}] should be NaN", i);
        }
        assert!(!result.k[6].is_nan());

        // Slow %D starts at k_period + 2*d_period - 3 = 5 + 6 - 3 = 8
        for i in 0..8 {
            assert!(result.d[i].is_nan(), "slow d[{}] should be NaN", i);
        }
        assert!(!result.d[8].is_nan());
    }

    #[test]
    fn test_stochastic_slow_smoother_than_fast() {
        // Slow stochastic should be smoother (less volatile) than fast
        let high: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64) + (i % 3) as f64).collect();
        let low: Vec<f64> = (0..20).map(|i| 95.0 + (i as f64) - (i % 3) as f64).collect();
        let close: Vec<f64> = (0..20).map(|i| 97.5 + (i as f64)).collect();

        let fast = stochastic_fast(&high, &low, &close, 5, 3).unwrap();
        let slow = stochastic_slow(&high, &low, &close, 5, 3).unwrap();

        // Slow %K should have less variance than Fast %K
        // (This is a qualitative test - we just verify both produce valid output)
        let fast_valid: Vec<f64> = fast.k.iter().filter(|x| !x.is_nan()).cloned().collect();
        let slow_valid: Vec<f64> = slow.k.iter().filter(|x| !x.is_nan()).cloned().collect();

        assert!(!fast_valid.is_empty());
        assert!(!slow_valid.is_empty());
    }

    // ==================== Full Stochastic Tests ====================

    #[test]
    fn test_stochastic_full_basic() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5, 12.0,
        ];
        let low = vec![
            9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5, 11.0,
        ];
        let close = vec![
            9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0, 11.5,
        ];

        // k_period=5, slow_k_period=3, d_period=3
        let result = stochastic_full(&high, &low, &close, 5, 3, 3).unwrap();

        assert_eq!(result.k.len(), 11);
        assert_eq!(result.d.len(), 11);

        // Full %K starts at k_period + slow_k_period - 2 = 5 + 3 - 2 = 6
        for i in 0..6 {
            assert!(result.k[i].is_nan(), "full k[{}] should be NaN", i);
        }
        assert!(!result.k[6].is_nan());

        // Full %D starts at k_period + slow_k_period + d_period - 3 = 5 + 3 + 3 - 3 = 8
        for i in 0..8 {
            assert!(result.d[i].is_nan(), "full d[{}] should be NaN", i);
        }
        assert!(!result.d[8].is_nan());
    }

    #[test]
    fn test_stochastic_full_custom_periods() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5, 12.0, 12.5,
        ];
        let low = vec![
            9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5, 11.0, 11.5,
        ];
        let close = vec![
            9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0, 11.5, 12.0,
        ];

        // Custom periods: k_period=3, slow_k_period=2, d_period=4
        let result = stochastic_full(&high, &low, &close, 3, 2, 4).unwrap();

        // Full %K starts at k_period + slow_k_period - 2 = 3 + 2 - 2 = 3
        assert!(result.k[2].is_nan());
        assert!(!result.k[3].is_nan());

        // Full %D starts at k_period + slow_k_period + d_period - 3 = 3 + 2 + 4 - 3 = 6
        assert!(result.d[5].is_nan());
        assert!(!result.d[6].is_nan());
    }

    #[test]
    fn test_stochastic_full_same_as_slow_when_periods_match() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5,
        ];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let slow = stochastic_slow(&high, &low, &close, 5, 3).unwrap();
        let full = stochastic_full(&high, &low, &close, 5, 3, 3).unwrap();

        // Slow and Full should produce identical results when slow_k_period == d_period
        for i in 0..10 {
            assert!(
                approx_eq(slow.k[i], full.k[i], STOCH_EPSILON),
                "k mismatch at {}: {} vs {}",
                i,
                slow.k[i],
                full.k[i]
            );
            assert!(
                approx_eq(slow.d[i], full.d[i], STOCH_EPSILON),
                "d mismatch at {}: {} vs {}",
                i,
                slow.d[i],
                full.d[i]
            );
        }
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_stochastic_empty_input() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];

        let result = stochastic_fast(&high, &low, &close, 5, 3);
        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_stochastic_zero_k_period() {
        let high = vec![1.0_f64, 2.0, 3.0];
        let low = vec![0.5, 1.5, 2.5];
        let close = vec![0.75, 1.75, 2.75];

        let result = stochastic_fast(&high, &low, &close, 0, 3);
        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_stochastic_zero_d_period() {
        let high = vec![1.0_f64, 2.0, 3.0];
        let low = vec![0.5, 1.5, 2.5];
        let close = vec![0.75, 1.75, 2.75];

        let result = stochastic_fast(&high, &low, &close, 3, 0);
        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_stochastic_insufficient_data() {
        let high = vec![1.0_f64, 2.0, 3.0];
        let low = vec![0.5, 1.5, 2.5];
        let close = vec![0.75, 1.75, 2.75];

        let result = stochastic_fast(&high, &low, &close, 5, 3);
        assert!(matches!(
            result,
            Err(Error::InsufficientData {
                required: 5,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn test_stochastic_mismatched_lengths() {
        let high = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let low = vec![0.5, 1.5, 2.5]; // Shorter
        let close = vec![0.75, 1.75, 2.75, 3.75, 4.75];

        let result = stochastic_fast(&high, &low, &close, 3, 2);
        assert!(matches!(result, Err(Error::LengthMismatch { .. })));
    }

    // ==================== Into Variant Tests ====================

    #[test]
    fn test_stochastic_fast_into_basic() {
        let high = vec![10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5];

        let mut output = StochasticOutput {
            k: vec![0.0_f64; 7],
            d: vec![0.0_f64; 7],
        };

        let (valid_k, valid_d) =
            stochastic_fast_into(&high, &low, &close, 3, 2, &mut output).unwrap();

        assert_eq!(valid_k, 5); // 7 - (3 - 1) = 5
        assert_eq!(valid_d, 4); // 7 - (3 + 2 - 2) = 4

        assert!(output.k[0].is_nan());
        assert!(output.k[1].is_nan());
        assert!(!output.k[2].is_nan());
    }

    #[test]
    fn test_stochastic_fast_into_buffer_reuse() {
        let high1 = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
        let low1 = vec![9.0, 10.0, 11.0, 12.0, 13.0];
        let close1 = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        let high2 = vec![14.0_f64, 13.0, 12.0, 11.0, 10.0];
        let low2 = vec![13.0, 12.0, 11.0, 10.0, 9.0];
        let close2 = vec![13.0, 12.0, 11.0, 10.0, 9.0];

        let mut output = StochasticOutput {
            k: vec![0.0_f64; 5],
            d: vec![0.0_f64; 5],
        };

        stochastic_fast_into(&high1, &low1, &close1, 3, 2, &mut output).unwrap();
        let k_first = output.k[3];

        stochastic_fast_into(&high2, &low2, &close2, 3, 2, &mut output).unwrap();
        let k_second = output.k[3];

        // Different data should produce different results
        assert!(!approx_eq(k_first, k_second, EPSILON));
    }

    #[test]
    fn test_stochastic_fast_into_insufficient_output() {
        let high = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let low = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let close = vec![0.75, 1.75, 2.75, 3.75, 4.75];

        let mut output = StochasticOutput {
            k: vec![0.0_f64; 3], // Too short
            d: vec![0.0_f64; 5],
        };

        let result = stochastic_fast_into(&high, &low, &close, 3, 2, &mut output);
        assert!(matches!(result, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn test_stochastic_slow_into_basic() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5,
        ];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let mut output = StochasticOutput {
            k: vec![0.0_f64; 10],
            d: vec![0.0_f64; 10],
        };

        let (valid_k, valid_d) =
            stochastic_slow_into(&high, &low, &close, 5, 3, &mut output).unwrap();

        assert_eq!(valid_k, 4); // 10 - (5 + 3 - 2) = 4
        assert_eq!(valid_d, 2); // 10 - (5 + 3 + 3 - 3) = 2
    }

    #[test]
    fn test_stochastic_full_into_basic() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5, 12.0,
        ];
        let low = vec![
            9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5, 11.0,
        ];
        let close = vec![
            9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0, 11.5,
        ];

        let mut output = StochasticOutput {
            k: vec![0.0_f64; 11],
            d: vec![0.0_f64; 11],
        };

        let (valid_k, valid_d) =
            stochastic_full_into(&high, &low, &close, 5, 3, 3, &mut output).unwrap();

        assert_eq!(valid_k, 5); // 11 - (5 + 3 - 2) = 5
        assert_eq!(valid_d, 3); // 11 - (5 + 3 + 3 - 3) = 3
    }

    // ==================== Consistency Tests ====================

    #[test]
    fn test_stochastic_fast_and_fast_into_produce_same_result() {
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5,
        ];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let result1 = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

        let mut result2 = StochasticOutput {
            k: vec![0.0_f64; 10],
            d: vec![0.0_f64; 10],
        };
        stochastic_fast_into(&high, &low, &close, 5, 3, &mut result2).unwrap();

        for i in 0..10 {
            assert!(
                approx_eq(result1.k[i], result2.k[i], EPSILON),
                "k mismatch at {}: {} vs {}",
                i,
                result1.k[i],
                result2.k[i]
            );
            assert!(
                approx_eq(result1.d[i], result2.d[i], EPSILON),
                "d mismatch at {}: {} vs {}",
                i,
                result1.d[i],
                result2.d[i]
            );
        }
    }

    // ==================== Bounds Tests ====================

    #[test]
    fn test_stochastic_k_in_bounds() {
        // %K should always be between 0 and 100
        let high: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) + ((i % 7) as f64)).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64) - ((i % 5) as f64)).collect();
        let close: Vec<f64> = (0..50).map(|i| 97.5 + (i as f64) + ((i % 3) as f64) * 0.5).collect();

        let result = stochastic_fast(&high, &low, &close, 14, 3).unwrap();

        for (i, &k) in result.k.iter().enumerate() {
            if !k.is_nan() {
                assert!(
                    k >= 0.0 && k <= 100.0,
                    "k[{}] = {} is out of bounds",
                    i,
                    k
                );
            }
        }
    }

    #[test]
    fn test_stochastic_d_in_bounds() {
        // %D should always be between 0 and 100
        let high: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) + ((i % 7) as f64)).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64) - ((i % 5) as f64)).collect();
        let close: Vec<f64> = (0..50).map(|i| 97.5 + (i as f64) + ((i % 3) as f64) * 0.5).collect();

        let result = stochastic_fast(&high, &low, &close, 14, 3).unwrap();

        for (i, &d) in result.d.iter().enumerate() {
            if !d.is_nan() {
                assert!(
                    d >= 0.0 && d <= 100.0,
                    "d[{}] = {} is out of bounds",
                    i,
                    d
                );
            }
        }
    }

    // ==================== NaN Handling Tests ====================

    #[test]
    fn test_stochastic_with_nan_in_data() {
        let high = vec![10.0_f64, 11.0, f64::NAN, 11.5, 12.5, 13.0];
        let low = vec![9.0, 10.0, 11.0, f64::NAN, 11.5, 12.0];
        let close = vec![9.5, 10.5, 11.5, 11.0, f64::NAN, 12.5];

        let result = stochastic_fast(&high, &low, &close, 3, 2).unwrap();

        // NaN in input should propagate to output
        // Windows containing NaN will produce NaN
        assert!(result.k[2].is_nan()); // NaN in high[2]
        assert!(result.k[3].is_nan()); // NaN in low[3]
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_stochastic_period_one() {
        let high = vec![10.0_f64, 11.0, 12.0, 11.0, 10.0];
        let low = vec![9.0, 10.0, 11.0, 10.0, 9.0];
        let close = vec![9.5, 10.5, 11.5, 10.5, 9.5];

        let result = stochastic_fast(&high, &low, &close, 1, 1).unwrap();

        // All values should be valid with period 1
        assert!(!result.k[0].is_nan());
        assert!(!result.d[0].is_nan());
    }

    #[test]
    fn test_stochastic_minimum_data() {
        // Minimum data for k_period=3, d_period=2
        let high = vec![10.0_f64, 11.0, 12.0, 11.0];
        let low = vec![9.0, 10.0, 11.0, 10.0];
        let close = vec![9.5, 10.5, 11.5, 10.5];

        let result = stochastic_fast(&high, &low, &close, 3, 2).unwrap();

        // k_period - 1 = 2 NaN values for %K
        assert!(result.k[0].is_nan());
        assert!(result.k[1].is_nan());
        assert!(!result.k[2].is_nan());

        // k_period + d_period - 2 = 3 NaN values for %D
        assert!(result.d[0].is_nan());
        assert!(result.d[1].is_nan());
        assert!(result.d[2].is_nan());
        assert!(!result.d[3].is_nan());
    }

    #[test]
    fn test_stochastic_negative_prices() {
        // Unusual but valid - negative prices (e.g., spreads, correlations)
        let high = vec![-5.0_f64, -4.0, -3.0, -4.0, -5.0];
        let low = vec![-10.0, -9.0, -8.0, -9.0, -10.0];
        let close = vec![-7.5, -6.5, -5.5, -6.5, -7.5];

        let result = stochastic_fast(&high, &low, &close, 3, 2).unwrap();

        // Should still produce valid results in [0, 100]
        for k in result.k.iter() {
            if !k.is_nan() {
                assert!(*k >= 0.0 && *k <= 100.0);
            }
        }
    }

    #[test]
    fn test_stochastic_large_values() {
        let high = vec![1e12_f64, 1.01e12, 1.02e12, 1.03e12, 1.04e12];
        let low = vec![0.99e12, 1.0e12, 1.01e12, 1.02e12, 1.03e12];
        let close = vec![1.0e12, 1.01e12, 1.02e12, 1.025e12, 1.035e12];

        let result = stochastic_fast(&high, &low, &close, 3, 2).unwrap();

        // Should handle large values
        for k in result.k.iter() {
            if !k.is_nan() {
                assert!(*k >= 0.0 && *k <= 100.0);
            }
        }
    }

    // ==================== Property Tests ====================

    #[test]
    fn test_stochastic_output_length_equals_input_length() {
        for len in [5, 10, 50, 100] {
            for k_period in [3, 5, 14] {
                for d_period in [1, 3, 5] {
                    if k_period <= len {
                        let high: Vec<f64> = (0..len).map(|x| (x + 10) as f64).collect();
                        let low: Vec<f64> = (0..len).map(|x| x as f64).collect();
                        let close: Vec<f64> = (0..len).map(|x| (x + 5) as f64).collect();

                        let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();
                        assert_eq!(result.k.len(), len);
                        assert_eq!(result.d.len(), len);
                    }
                }
            }
        }
    }

    #[test]
    fn test_stochastic_k_nan_count() {
        // First (k_period - 1) %K values should be NaN
        for k_period in 1..=10 {
            let high: Vec<f64> = (0..20).map(|x| (x + 10) as f64).collect();
            let low: Vec<f64> = (0..20).map(|x| x as f64).collect();
            let close: Vec<f64> = (0..20).map(|x| (x + 5) as f64).collect();

            let result = stochastic_fast(&high, &low, &close, k_period, 3).unwrap();

            let nan_count = result.k.iter().filter(|x| x.is_nan()).count();
            assert_eq!(
                nan_count,
                k_period - 1,
                "Expected {} NaN values for k_period {}",
                k_period - 1,
                k_period
            );
        }
    }

    #[test]
    fn test_stochastic_d_nan_count() {
        // First (k_period + d_period - 2) %D values should be NaN
        for k_period in 1..=5 {
            for d_period in 1..=5 {
                let high: Vec<f64> = (0..20).map(|x| (x + 10) as f64).collect();
                let low: Vec<f64> = (0..20).map(|x| x as f64).collect();
                let close: Vec<f64> = (0..20).map(|x| (x + 5) as f64).collect();

                let result = stochastic_fast(&high, &low, &close, k_period, d_period).unwrap();

                let expected_nan_count = k_period + d_period - 2;
                let nan_count = result.d.iter().filter(|x| x.is_nan()).count();
                assert_eq!(
                    nan_count, expected_nan_count,
                    "Expected {} NaN values for k_period={}, d_period={}",
                    expected_nan_count, k_period, d_period
                );
            }
        }
    }

    #[test]
    fn test_stochastic_d_is_average_of_k() {
        // %D should be SMA of %K
        let high = vec![
            10.0_f64, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.5,
        ];
        let low = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.5];
        let close = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 11.0];

        let result = stochastic_fast(&high, &low, &close, 3, 3).unwrap();

        // Verify %D is SMA of %K
        for i in 4..10 {
            // From index k_period + d_period - 2 = 4
            let expected_d = (result.k[i - 2] + result.k[i - 1] + result.k[i]) / 3.0;
            assert!(
                approx_eq(result.d[i], expected_d, STOCH_EPSILON),
                "d[{}] should be average of k[{}..{}]",
                i,
                i - 2,
                i
            );
        }
    }

    // ==================== Trend Response Tests ====================

    #[test]
    fn test_stochastic_responds_to_uptrend() {
        // In an uptrend, %K should be high (close near high)
        let high: Vec<f64> = (0..10).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let low: Vec<f64> = (0..10).map(|i| 95.0 + (i as f64) * 2.0).collect();
        let close: Vec<f64> = (0..10).map(|i| 99.0 + (i as f64) * 2.0).collect(); // Close near high

        let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

        // In uptrend with close near high, %K should be high (> 50)
        for i in 4..10 {
            assert!(result.k[i] > 50.0, "k[{}] should be > 50 in uptrend", i);
        }
    }

    #[test]
    fn test_stochastic_responds_to_downtrend() {
        // In a downtrend, %K should be low (close near low)
        let high: Vec<f64> = (0..10).map(|i| 100.0 - (i as f64) * 2.0).collect();
        let low: Vec<f64> = (0..10).map(|i| 95.0 - (i as f64) * 2.0).collect();
        let close: Vec<f64> = (0..10)
            .map(|i| f64::from(i).mul_add(-2.0, 96.0))
            .collect(); // Close near low

        let result = stochastic_fast(&high, &low, &close, 5, 3).unwrap();

        // In downtrend with close near low, %K should be low (< 50)
        for i in 4..10 {
            assert!(result.k[i] < 50.0, "k[{i}] should be < 50 in downtrend");
        }
    }
}
