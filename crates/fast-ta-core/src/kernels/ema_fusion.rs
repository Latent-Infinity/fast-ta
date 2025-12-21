//! EMA fusion kernel for computing multiple EMA-family indicators in a single pass.
//!
//! This module provides optimized implementations for computing multiple
//! EMA-based indicators simultaneously, reducing memory bandwidth and
//! improving cache utilization.
//!
//! # Indicators
//!
//! - **EMA (Exponential Moving Average)**: Single exponential smoothing
//! - **DEMA (Double EMA)**: 2×EMA(price) - EMA(EMA(price))
//! - **TEMA (Triple EMA)**: 3×EMA(price) - 3×EMA(EMA(price)) + EMA(EMA(EMA(price)))
//! - **MACD**: Fast EMA - Slow EMA with signal line
//!
//! # Performance
//!
//! When computing multiple EMAs or EMA-derived indicators, fusing them into
//! a single pass over the data provides significant performance benefits:
//!
//! 1. Data is loaded into cache only once
//! 2. The same input values are reused for multiple calculations
//! 3. Memory bandwidth is reduced proportionally to the number of indicators
//!
//! # Example
//!
//! ```
//! use fast_ta_core::kernels::ema_fusion::{ema_multi, ema_fusion, EmaFusionOutput};
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//!
//! // Compute multiple EMAs with different periods in a single pass
//! let periods = vec![2, 3, 5];
//! let result = ema_multi(&data, &periods).unwrap();
//! assert_eq!(result.len(), 3);
//!
//! // Compute EMA, DEMA, and TEMA together
//! let fused = ema_fusion(&data, 3).unwrap();
//! assert!(!fused.ema[2].is_nan());
//! ```
//!
//! # Algorithm
//!
//! All EMA computations use the standard smoothing formula:
//!
//! ```text
//! α = 2 / (period + 1)
//! EMA[0..period-2] = NaN (insufficient lookback)
//! EMA[period-1] = SMA(prices[0..period])  // SMA seed
//! EMA[i] = α × Price[i] + (1 - α) × EMA[i-1]
//! ```
//!
//! For derived indicators:
//!
//! ```text
//! DEMA[i] = 2 × EMA[i] - EMA(EMA)[i]
//! TEMA[i] = 3 × EMA[i] - 3 × EMA(EMA)[i] + EMA(EMA(EMA))[i]
//! ```

use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::{SeriesElement, ValidatedInput};

/// Output structure for fused EMA computations.
///
/// Contains EMA, DEMA (Double EMA), and TEMA (Triple EMA) computed
/// in a single pass over the data.
#[derive(Debug, Clone)]
pub struct EmaFusionOutput<T> {
    /// The EMA (Exponential Moving Average).
    ///
    /// First `period - 1` values are NaN.
    pub ema: Vec<T>,

    /// The DEMA (Double Exponential Moving Average).
    ///
    /// Formula: 2 × EMA - EMA(EMA)
    /// First `2 × period - 2` values are NaN.
    pub dema: Vec<T>,

    /// The TEMA (Triple Exponential Moving Average).
    ///
    /// Formula: 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
    /// First `3 × period - 3` values are NaN.
    pub tema: Vec<T>,
}

impl<T: SeriesElement> EmaFusionOutput<T> {
    /// Returns the length of the output vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ema.len()
    }

    /// Returns true if the output vectors are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ema.is_empty()
    }

    /// Returns the index of the first valid EMA value.
    #[must_use]
    pub fn first_valid_ema_index(period: usize) -> usize {
        period.saturating_sub(1)
    }

    /// Returns the index of the first valid DEMA value.
    #[must_use]
    pub fn first_valid_dema_index(period: usize) -> usize {
        (2 * period).saturating_sub(2)
    }

    /// Returns the index of the first valid TEMA value.
    #[must_use]
    pub fn first_valid_tema_index(period: usize) -> usize {
        (3 * period).saturating_sub(3)
    }
}

/// Output structure for fused MACD computation.
///
/// Contains MACD line, signal line, and histogram computed together.
#[derive(Debug, Clone)]
pub struct MacdFusionOutput<T> {
    /// The MACD line (fast EMA - slow EMA).
    pub macd_line: Vec<T>,

    /// The signal line (EMA of MACD line).
    pub signal_line: Vec<T>,

    /// The histogram (MACD line - signal line).
    pub histogram: Vec<T>,
}

impl<T: SeriesElement> MacdFusionOutput<T> {
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
}

/// Computes multiple EMAs with different periods in a single pass over the data.
///
/// This function is more efficient than calling `ema()` multiple times when
/// you need EMAs for several periods, as it only reads the input data once.
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `periods` - The periods for each EMA to compute
///
/// # Returns
///
/// A `Result` containing a vector of EMA outputs, one for each period,
/// in the same order as the input periods.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - The input data is shorter than the maximum period (`Error::InsufficientData`)
///
/// # Performance
///
/// - Time complexity: O(n × k) where n is data length and k is number of periods
/// - Space complexity: O(n × k) for output vectors
/// - Cache efficiency: Data is loaded once, reused k times
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::ema_fusion::ema_multi;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let periods = vec![2, 3, 5];
/// let result = ema_multi(&data, &periods).unwrap();
///
/// // Each result is an EMA with the corresponding period
/// assert_eq!(result.len(), 3);
/// assert!(!result[0][1].is_nan()); // EMA(2) valid from index 1
/// assert!(!result[1][2].is_nan()); // EMA(3) valid from index 2
/// assert!(!result[2][4].is_nan()); // EMA(5) valid from index 4
/// ```
pub fn ema_multi<T: SeriesElement>(data: &[T], periods: &[usize]) -> Result<Vec<Vec<T>>> {
    // Validate inputs
    data.validate_not_empty()?;

    if periods.is_empty() {
        return Ok(vec![]);
    }

    // Find maximum period and validate all periods
    let mut max_period = 0usize;
    for &period in periods {
        if period == 0 {
            return Err(Error::InvalidPeriod {
                period,
                reason: "period must be at least 1",
            });
        }
        if period > max_period {
            max_period = period;
        }
    }

    if data.len() < max_period {
        return Err(Error::InsufficientData {
            required: max_period,
            actual: data.len(),
        });
    }

    let n = data.len();
    let k = periods.len();

    // Pre-compute alphas and initialize state
    let mut alphas = Vec::with_capacity(k);
    let mut one_minus_alphas = Vec::with_capacity(k);
    let mut period_ts = Vec::with_capacity(k);
    let mut sums = vec![T::zero(); k];
    let mut ema_values = vec![T::nan(); k];
    let mut results: Vec<Vec<T>> = periods.iter().map(|_| vec![T::nan(); n]).collect();

    for &period in periods {
        let period_plus_one = T::from_usize(period + 1)?;
        let alpha = T::two() / period_plus_one;
        alphas.push(alpha);
        one_minus_alphas.push(T::one() - alpha);
        period_ts.push(T::from_usize(period)?);
    }

    // Single pass over the data
    for i in 0..n {
        let value = data[i];

        for j in 0..k {
            let period = periods[j];

            if i < period {
                // Building up the SMA seed
                sums[j] = sums[j] + value;

                if i == period - 1 {
                    // SMA seed is complete
                    ema_values[j] = sums[j] / period_ts[j];
                    results[j][i] = ema_values[j];
                }
            } else {
                // Apply EMA formula
                let ema_current = alphas[j] * value + one_minus_alphas[j] * ema_values[j];
                ema_values[j] = ema_current;
                results[j][i] = ema_current;
            }
        }
    }

    Ok(results)
}

/// Computes multiple EMAs into pre-allocated output buffers.
///
/// This variant allows reusing existing buffers to avoid allocations.
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `periods` - The periods for each EMA to compute
/// * `outputs` - Pre-allocated output buffers (each must be at least as long as input)
///
/// # Returns
///
/// A `Result` containing the number of valid values for each EMA.
pub fn ema_multi_into<T: SeriesElement>(
    data: &[T],
    periods: &[usize],
    outputs: &mut [&mut [T]],
) -> Result<Vec<usize>> {
    // Validate inputs
    data.validate_not_empty()?;

    if periods.len() != outputs.len() {
        return Err(Error::InsufficientData {
            required: periods.len(),
            actual: outputs.len(),
        });
    }

    if periods.is_empty() {
        return Ok(vec![]);
    }

    // Find maximum period and validate
    let mut max_period = 0usize;
    for (i, &period) in periods.iter().enumerate() {
        if period == 0 {
            return Err(Error::InvalidPeriod {
                period,
                reason: "period must be at least 1",
            });
        }
        if period > max_period {
            max_period = period;
        }
        if outputs[i].len() < data.len() {
            return Err(Error::InsufficientData {
                required: data.len(),
                actual: outputs[i].len(),
            });
        }
    }

    if data.len() < max_period {
        return Err(Error::InsufficientData {
            required: max_period,
            actual: data.len(),
        });
    }

    let n = data.len();
    let k = periods.len();

    // Pre-compute alphas and initialize state
    let mut alphas = Vec::with_capacity(k);
    let mut one_minus_alphas = Vec::with_capacity(k);
    let mut period_ts = Vec::with_capacity(k);
    let mut sums = vec![T::zero(); k];
    let mut ema_values = vec![T::nan(); k];

    for &period in periods {
        let period_plus_one = T::from_usize(period + 1)?;
        let alpha = T::two() / period_plus_one;
        alphas.push(alpha);
        one_minus_alphas.push(T::one() - alpha);
        period_ts.push(T::from_usize(period)?);
    }

    // Initialize outputs with NaN
    for (j, output) in outputs.iter_mut().enumerate() {
        for i in 0..(periods[j] - 1) {
            output[i] = T::nan();
        }
    }

    // Single pass over the data
    for i in 0..n {
        let value = data[i];

        for j in 0..k {
            let period = periods[j];

            if i < period {
                // Building up the SMA seed
                sums[j] = sums[j] + value;

                if i == period - 1 {
                    // SMA seed is complete
                    ema_values[j] = sums[j] / period_ts[j];
                    outputs[j][i] = ema_values[j];
                }
            } else {
                // Apply EMA formula
                let ema_current = alphas[j] * value + one_minus_alphas[j] * ema_values[j];
                ema_values[j] = ema_current;
                outputs[j][i] = ema_current;
            }
        }
    }

    // Calculate valid counts
    let valid_counts: Vec<usize> = periods.iter().map(|&p| n - p + 1).collect();

    Ok(valid_counts)
}

/// Computes EMA, DEMA, and TEMA in a single fused pass.
///
/// This function computes all three indicators simultaneously:
/// - EMA: Exponential Moving Average
/// - DEMA: Double EMA (2×EMA - EMA(EMA))
/// - TEMA: Triple EMA (3×EMA - 3×EMA(EMA) + EMA(EMA(EMA)))
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `period` - The period for all EMA calculations
///
/// # Returns
///
/// A `Result` containing an `EmaFusionOutput` with EMA, DEMA, and TEMA.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - The period is zero (`Error::InvalidPeriod`)
/// - The input data is shorter than 3 × period (`Error::InsufficientData`)
///
/// # Performance
///
/// Computing all three indicators together is more efficient than computing
/// them separately because:
/// 1. Data is read only once
/// 2. EMA values are computed once and reused for EMA(EMA) and EMA(EMA(EMA))
/// 3. All multiplications and additions are fused
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::ema_fusion::ema_fusion;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = ema_fusion(&data, 3).unwrap();
///
/// // EMA valid from index 2 (period - 1)
/// assert!(result.ema[1].is_nan());
/// assert!(!result.ema[2].is_nan());
///
/// // DEMA valid from index 4 (2 × period - 2)
/// assert!(result.dema[3].is_nan());
/// assert!(!result.dema[4].is_nan());
///
/// // TEMA valid from index 6 (3 × period - 3)
/// assert!(result.tema[5].is_nan());
/// assert!(!result.tema[6].is_nan());
/// ```
pub fn ema_fusion<T: SeriesElement>(data: &[T], period: usize) -> Result<EmaFusionOutput<T>> {
    // Validate inputs
    if period == 0 {
        return Err(Error::InvalidPeriod {
            period,
            reason: "period must be at least 1",
        });
    }

    data.validate_not_empty()?;

    // Need at least 3 × period - 2 for TEMA to have one valid value
    let min_required = 3 * period - 2;
    if data.len() < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            actual: data.len(),
        });
    }

    let n = data.len();

    // Pre-compute alpha
    let period_plus_one = T::from_usize(period + 1)?;
    let alpha = T::two() / period_plus_one;
    let one_minus_alpha = T::one() - alpha;
    let period_t = T::from_usize(period)?;

    // Initialize output vectors
    let mut ema = vec![T::nan(); n];
    let mut dema = vec![T::nan(); n];
    let mut tema = vec![T::nan(); n];

    // First pass: compute EMA of price
    let mut sum = T::zero();
    let mut ema_value = T::nan();

    for i in 0..n {
        if i < period {
            sum = sum + data[i];
            if i == period - 1 {
                ema_value = sum / period_t;
                ema[i] = ema_value;
            }
        } else {
            ema_value = alpha * data[i] + one_minus_alpha * ema_value;
            ema[i] = ema_value;
        }
    }

    // Second pass: compute EMA of EMA (for DEMA and TEMA)
    let first_valid_ema = period - 1;
    let mut ema2 = vec![T::nan(); n];
    sum = T::zero();
    let mut ema2_value = T::nan();

    for i in first_valid_ema..n {
        let offset = i - first_valid_ema;
        if offset < period {
            sum = sum + ema[i];
            if offset == period - 1 {
                ema2_value = sum / period_t;
                ema2[i] = ema2_value;
            }
        } else {
            ema2_value = alpha * ema[i] + one_minus_alpha * ema2_value;
            ema2[i] = ema2_value;
        }
    }

    // Compute DEMA: 2 × EMA - EMA(EMA)
    let first_valid_dema = 2 * period - 2;
    let two = T::two();
    for i in first_valid_dema..n {
        dema[i] = two * ema[i] - ema2[i];
    }

    // Third pass: compute EMA of EMA of EMA (for TEMA)
    let first_valid_ema2 = 2 * period - 2;
    let mut ema3 = vec![T::nan(); n];
    sum = T::zero();
    let mut ema3_value = T::nan();

    for i in first_valid_ema2..n {
        let offset = i - first_valid_ema2;
        if offset < period {
            sum = sum + ema2[i];
            if offset == period - 1 {
                ema3_value = sum / period_t;
                ema3[i] = ema3_value;
            }
        } else {
            ema3_value = alpha * ema2[i] + one_minus_alpha * ema3_value;
            ema3[i] = ema3_value;
        }
    }

    // Compute TEMA: 3 × EMA - 3 × EMA(EMA) + EMA(EMA(EMA))
    let first_valid_tema = 3 * period - 3;
    let three = T::from_usize(3)?;
    for i in first_valid_tema..n {
        tema[i] = three * ema[i] - three * ema2[i] + ema3[i];
    }

    Ok(EmaFusionOutput { ema, dema, tema })
}

/// Computes EMA, DEMA, and TEMA into pre-allocated output buffers.
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `period` - The period for all EMA calculations
/// * `output` - Pre-allocated output structure
///
/// # Returns
///
/// A `Result` containing a tuple of (valid_ema_count, valid_dema_count, valid_tema_count).
pub fn ema_fusion_into<T: SeriesElement>(
    data: &[T],
    period: usize,
    output: &mut EmaFusionOutput<T>,
) -> Result<(usize, usize, usize)> {
    // Validate inputs
    if period == 0 {
        return Err(Error::InvalidPeriod {
            period,
            reason: "period must be at least 1",
        });
    }

    data.validate_not_empty()?;

    let min_required = 3 * period - 2;
    if data.len() < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            actual: data.len(),
        });
    }

    let n = data.len();

    // Validate output buffer sizes
    if output.ema.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.ema.len(),
        });
    }
    if output.dema.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.dema.len(),
        });
    }
    if output.tema.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.tema.len(),
        });
    }

    // Pre-compute alpha
    let period_plus_one = T::from_usize(period + 1)?;
    let alpha = T::two() / period_plus_one;
    let one_minus_alpha = T::one() - alpha;
    let period_t = T::from_usize(period)?;

    // Initialize with NaN
    for i in 0..(period - 1) {
        output.ema[i] = T::nan();
    }
    for i in 0..(2 * period - 2) {
        output.dema[i] = T::nan();
    }
    for i in 0..(3 * period - 3) {
        output.tema[i] = T::nan();
    }

    // First pass: compute EMA of price
    let mut sum = T::zero();
    let mut ema_value = T::nan();

    for i in 0..n {
        if i < period {
            sum = sum + data[i];
            if i == period - 1 {
                ema_value = sum / period_t;
                output.ema[i] = ema_value;
            }
        } else {
            ema_value = alpha * data[i] + one_minus_alpha * ema_value;
            output.ema[i] = ema_value;
        }
    }

    // Second pass: compute EMA of EMA
    let first_valid_ema = period - 1;
    let mut ema2: Vec<T> = vec![T::nan(); n];
    sum = T::zero();
    let mut ema2_value = T::nan();

    for i in first_valid_ema..n {
        let offset = i - first_valid_ema;
        if offset < period {
            sum = sum + output.ema[i];
            if offset == period - 1 {
                ema2_value = sum / period_t;
                ema2[i] = ema2_value;
            }
        } else {
            ema2_value = alpha * output.ema[i] + one_minus_alpha * ema2_value;
            ema2[i] = ema2_value;
        }
    }

    // Compute DEMA
    let first_valid_dema = 2 * period - 2;
    let two = T::two();
    for i in first_valid_dema..n {
        output.dema[i] = two * output.ema[i] - ema2[i];
    }

    // Third pass: compute EMA of EMA of EMA
    let first_valid_ema2 = 2 * period - 2;
    sum = T::zero();
    let mut ema3_value = T::nan();

    for i in first_valid_ema2..n {
        let offset = i - first_valid_ema2;
        if offset < period {
            sum = sum + ema2[i];
            if offset == period - 1 {
                ema3_value = sum / period_t;
            }
        } else {
            ema3_value = alpha * ema2[i] + one_minus_alpha * ema3_value;
        }

        // Compute TEMA when we have all values
        if offset >= period - 1 {
            let three = T::from_usize(3).unwrap();
            output.tema[i] = three * output.ema[i] - three * ema2[i] + ema3_value;
        }
    }

    // Return valid counts
    let valid_ema = n - (period - 1);
    let valid_dema = n - (2 * period - 2);
    let valid_tema = n - (3 * period - 3);

    Ok((valid_ema, valid_dema, valid_tema))
}

/// Computes MACD (Moving Average Convergence Divergence) using fused EMA computation.
///
/// This is a fusion-optimized version of the MACD indicator that computes
/// the fast EMA, slow EMA, and signal EMA in an efficient manner.
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
/// A `Result` containing a `MacdFusionOutput` with MACD line, signal line, and histogram.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - Any period is zero (`Error::InvalidPeriod`)
/// - The fast period >= slow period (`Error::InvalidPeriod`)
/// - The input data is shorter than required (`Error::InsufficientData`)
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::ema_fusion::macd_fusion;
///
/// let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
/// let result = macd_fusion(&data, 12, 26, 9).unwrap();
///
/// // MACD line valid from slow_period - 1 = 25
/// assert!(result.macd_line[24].is_nan());
/// assert!(!result.macd_line[25].is_nan());
///
/// // Signal line valid from slow_period + signal_period - 2 = 33
/// assert!(result.signal_line[32].is_nan());
/// assert!(!result.signal_line[33].is_nan());
/// ```
pub fn macd_fusion<T: SeriesElement>(
    data: &[T],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<MacdFusionOutput<T>> {
    // Validate inputs
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

    data.validate_not_empty()?;

    let min_required = slow_period + signal_period - 1;
    if data.len() < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            actual: data.len(),
        });
    }

    let n = data.len();

    // Compute fast and slow EMAs together using ema_multi
    let periods = vec![fast_period, slow_period];
    let emas = ema_multi(data, &periods)?;
    let fast_ema = &emas[0];
    let slow_ema = &emas[1];

    // Compute MACD line = fast EMA - slow EMA
    let mut macd_line = vec![T::nan(); n];
    let first_valid_macd = slow_period - 1;
    for i in first_valid_macd..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Compute signal line = EMA of MACD line
    let mut signal_line = vec![T::nan(); n];
    let first_valid_signal = first_valid_macd + signal_period - 1;

    // Pre-compute alpha for signal EMA
    let signal_period_plus_one = T::from_usize(signal_period + 1)?;
    let signal_alpha = T::two() / signal_period_plus_one;
    let signal_one_minus_alpha = T::one() - signal_alpha;
    let signal_period_t = T::from_usize(signal_period)?;

    // SMA seed for signal line
    let mut sum = T::zero();
    for i in first_valid_macd..(first_valid_macd + signal_period) {
        sum = sum + macd_line[i];
    }
    let mut signal_value = sum / signal_period_t;
    signal_line[first_valid_signal] = signal_value;

    // Apply EMA formula for remaining signal values
    for i in (first_valid_signal + 1)..n {
        signal_value = signal_alpha * macd_line[i] + signal_one_minus_alpha * signal_value;
        signal_line[i] = signal_value;
    }

    // Compute histogram = MACD line - signal line
    let mut histogram = vec![T::nan(); n];
    for i in first_valid_signal..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok(MacdFusionOutput {
        macd_line,
        signal_line,
        histogram,
    })
}

/// Computes MACD into pre-allocated output buffers.
///
/// # Arguments
///
/// * `data` - The input price data series
/// * `fast_period` - The period for the fast EMA
/// * `slow_period` - The period for the slow EMA
/// * `signal_period` - The period for the signal line EMA
/// * `output` - Pre-allocated output structure
///
/// # Returns
///
/// A `Result` containing a tuple of (valid_macd_count, valid_signal_count).
pub fn macd_fusion_into<T: SeriesElement>(
    data: &[T],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    output: &mut MacdFusionOutput<T>,
) -> Result<(usize, usize)> {
    // Validate inputs
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

    data.validate_not_empty()?;

    let min_required = slow_period + signal_period - 1;
    if data.len() < min_required {
        return Err(Error::InsufficientData {
            required: min_required,
            actual: data.len(),
        });
    }

    let n = data.len();

    // Validate output buffer sizes
    if output.macd_line.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.macd_line.len(),
        });
    }
    if output.signal_line.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.signal_line.len(),
        });
    }
    if output.histogram.len() < n {
        return Err(Error::InsufficientData {
            required: n,
            actual: output.histogram.len(),
        });
    }

    // Compute fused result and copy to output
    let result = macd_fusion(data, fast_period, slow_period, signal_period)?;

    for i in 0..n {
        output.macd_line[i] = result.macd_line[i];
        output.signal_line[i] = result.signal_line[i];
        output.histogram[i] = result.histogram[i];
    }

    let first_valid_macd = slow_period - 1;
    let first_valid_signal = first_valid_macd + signal_period - 1;

    let valid_macd = n - first_valid_macd;
    let valid_signal = n - first_valid_signal;

    Ok((valid_macd, valid_signal))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    const EPSILON_F32: f32 = 1e-5;
    // Looser epsilon for complex calculations
    const FUSION_EPSILON: f64 = 1e-8;

    // ==================== ema_multi Basic Tests ====================

    #[test]
    fn test_ema_multi_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let periods = vec![2, 3, 5];
        let result = ema_multi(&data, &periods).unwrap();

        assert_eq!(result.len(), 3);

        // EMA(2) should be valid from index 1
        assert!(result[0][0].is_nan());
        assert!(!result[0][1].is_nan());

        // EMA(3) should be valid from index 2
        assert!(result[1][1].is_nan());
        assert!(!result[1][2].is_nan());

        // EMA(5) should be valid from index 4
        assert!(result[2][3].is_nan());
        assert!(!result[2][4].is_nan());
    }

    #[test]
    fn test_ema_multi_single_period() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let periods = vec![3];
        let result = ema_multi(&data, &periods).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0][0].is_nan());
        assert!(result[0][1].is_nan());
        // SMA seed: (1+2+3)/3 = 2
        assert!(approx_eq(result[0][2], 2.0, EPSILON));
    }

    #[test]
    fn test_ema_multi_empty_periods() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let periods: Vec<usize> = vec![];
        let result = ema_multi(&data, &periods).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_ema_multi_f32() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let periods = vec![2, 3];
        let result = ema_multi(&data, &periods).unwrap();

        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0][1], 1.5_f32, EPSILON_F32)); // SMA(2) of [1,2] = 1.5
    }

    #[test]
    fn test_ema_multi_matches_individual_ema() {
        use crate::indicators::ema::ema;

        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let periods = vec![3, 5, 7];

        let fused = ema_multi(&data, &periods).unwrap();

        // Compare with individual EMA calculations
        for (i, &period) in periods.iter().enumerate() {
            let unfused = ema(&data, period).unwrap();

            for j in 0..data.len() {
                assert!(
                    approx_eq(fused[i][j], unfused[j], EPSILON),
                    "Mismatch at period {} index {}: fused={}, unfused={}",
                    period,
                    j,
                    fused[i][j],
                    unfused[j]
                );
            }
        }
    }

    // ==================== ema_multi Error Tests ====================

    #[test]
    fn test_ema_multi_empty_input() {
        let data: Vec<f64> = vec![];
        let periods = vec![3];
        let result = ema_multi(&data, &periods);

        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_ema_multi_zero_period() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let periods = vec![2, 0, 3];
        let result = ema_multi(&data, &periods);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_ema_multi_insufficient_data() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let periods = vec![5]; // Need at least 5 data points
        let result = ema_multi(&data, &periods);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    // ==================== ema_fusion Basic Tests ====================

    #[test]
    fn test_ema_fusion_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema_fusion(&data, 3).unwrap();

        assert_eq!(result.len(), 10);

        // EMA valid from index 2
        assert!(result.ema[1].is_nan());
        assert!(!result.ema[2].is_nan());

        // DEMA valid from index 4 (2*3 - 2)
        assert!(result.dema[3].is_nan());
        assert!(!result.dema[4].is_nan());

        // TEMA valid from index 6 (3*3 - 3)
        assert!(result.tema[5].is_nan());
        assert!(!result.tema[6].is_nan());
    }

    #[test]
    fn test_ema_fusion_output_helpers() {
        assert_eq!(EmaFusionOutput::<f64>::first_valid_ema_index(3), 2);
        assert_eq!(EmaFusionOutput::<f64>::first_valid_dema_index(3), 4);
        assert_eq!(EmaFusionOutput::<f64>::first_valid_tema_index(3), 6);

        assert_eq!(EmaFusionOutput::<f64>::first_valid_ema_index(5), 4);
        assert_eq!(EmaFusionOutput::<f64>::first_valid_dema_index(5), 8);
        assert_eq!(EmaFusionOutput::<f64>::first_valid_tema_index(5), 12);
    }

    #[test]
    fn test_ema_fusion_ema_matches_unfused() {
        use crate::indicators::ema::ema;

        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let period = 3;

        let fused = ema_fusion(&data, period).unwrap();
        let unfused = ema(&data, period).unwrap();

        for i in 0..data.len() {
            assert!(
                approx_eq(fused.ema[i], unfused[i], EPSILON),
                "EMA mismatch at index {}: fused={}, unfused={}",
                i,
                fused.ema[i],
                unfused[i]
            );
        }
    }

    #[test]
    fn test_ema_fusion_dema_formula() {
        // DEMA = 2 * EMA - EMA(EMA)
        // We verify by manually computing EMA(EMA) and checking the formula
        use crate::indicators::ema::ema;

        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let period = 3;

        let fused = ema_fusion(&data, period).unwrap();
        let ema1 = ema(&data, period).unwrap();
        let ema2 = ema(&ema1, period).unwrap();

        let first_valid_dema = 2 * period - 2;
        for i in first_valid_dema..data.len() {
            let expected_dema = 2.0 * ema1[i] - ema2[i];
            assert!(
                approx_eq(fused.dema[i], expected_dema, FUSION_EPSILON),
                "DEMA mismatch at index {}: fused={}, expected={}",
                i,
                fused.dema[i],
                expected_dema
            );
        }
    }

    #[test]
    fn test_ema_fusion_tema_formula() {
        // TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
        use crate::indicators::ema::ema;

        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let period = 3;

        let fused = ema_fusion(&data, period).unwrap();
        let ema1 = ema(&data, period).unwrap();
        let ema2 = ema(&ema1, period).unwrap();
        let ema3 = ema(&ema2, period).unwrap();

        let first_valid_tema = 3 * period - 3;
        for i in first_valid_tema..data.len() {
            let expected_tema = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i];
            assert!(
                approx_eq(fused.tema[i], expected_tema, FUSION_EPSILON),
                "TEMA mismatch at index {}: fused={}, expected={}",
                i,
                fused.tema[i],
                expected_tema
            );
        }
    }

    #[test]
    fn test_ema_fusion_constant_values() {
        // For constant values, EMA = DEMA = TEMA = constant
        let data = vec![5.0_f64; 15];
        let result = ema_fusion(&data, 3).unwrap();

        for i in 2..result.len() {
            if !result.ema[i].is_nan() {
                assert!(approx_eq(result.ema[i], 5.0, EPSILON));
            }
        }
        for i in 4..result.len() {
            if !result.dema[i].is_nan() {
                assert!(approx_eq(result.dema[i], 5.0, EPSILON));
            }
        }
        for i in 6..result.len() {
            if !result.tema[i].is_nan() {
                assert!(approx_eq(result.tema[i], 5.0, EPSILON));
            }
        }
    }

    #[test]
    fn test_ema_fusion_f32() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema_fusion(&data, 3).unwrap();

        assert!(!result.ema[2].is_nan());
        assert!(!result.dema[4].is_nan());
        assert!(!result.tema[6].is_nan());
    }

    // ==================== ema_fusion Error Tests ====================

    #[test]
    fn test_ema_fusion_empty_input() {
        let data: Vec<f64> = vec![];
        let result = ema_fusion(&data, 3);

        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_ema_fusion_zero_period() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let result = ema_fusion(&data, 0);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_ema_fusion_insufficient_data() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]; // Need 3*3-2 = 7
        let result = ema_fusion(&data, 3);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    // ==================== macd_fusion Basic Tests ====================

    #[test]
    fn test_macd_fusion_basic() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd_fusion(&data, 12, 26, 9).unwrap();

        assert_eq!(result.len(), 50);

        // MACD line valid from slow_period - 1 = 25
        assert!(result.macd_line[24].is_nan());
        assert!(!result.macd_line[25].is_nan());

        // Signal line valid from 25 + 9 - 1 = 33
        assert!(result.signal_line[32].is_nan());
        assert!(!result.signal_line[33].is_nan());

        // Histogram same as signal
        assert!(result.histogram[32].is_nan());
        assert!(!result.histogram[33].is_nan());
    }

    #[test]
    fn test_macd_fusion_matches_unfused() {
        use crate::indicators::macd::macd;

        let data: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 0.5).collect();

        let fused = macd_fusion(&data, 12, 26, 9).unwrap();
        let unfused = macd(&data, 12, 26, 9).unwrap();

        for i in 0..data.len() {
            assert!(
                approx_eq(fused.macd_line[i], unfused.macd_line[i], FUSION_EPSILON),
                "MACD line mismatch at index {}: fused={}, unfused={}",
                i,
                fused.macd_line[i],
                unfused.macd_line[i]
            );
            assert!(
                approx_eq(fused.signal_line[i], unfused.signal_line[i], FUSION_EPSILON),
                "Signal line mismatch at index {}: fused={}, unfused={}",
                i,
                fused.signal_line[i],
                unfused.signal_line[i]
            );
            assert!(
                approx_eq(fused.histogram[i], unfused.histogram[i], FUSION_EPSILON),
                "Histogram mismatch at index {}: fused={}, unfused={}",
                i,
                fused.histogram[i],
                unfused.histogram[i]
            );
        }
    }

    #[test]
    fn test_macd_fusion_uptrend() {
        let data: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 2.0).collect();
        let result = macd_fusion(&data, 5, 10, 3).unwrap();

        // In uptrend, MACD line should be positive
        for i in 15..result.len() {
            if !result.macd_line[i].is_nan() {
                assert!(
                    result.macd_line[i] > 0.0,
                    "MACD should be positive in uptrend at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_macd_fusion_downtrend() {
        let data: Vec<f64> = (0..50).map(|i| 150.0 - (i as f64) * 2.0).collect();
        let result = macd_fusion(&data, 5, 10, 3).unwrap();

        // In downtrend, MACD line should be negative
        for i in 15..result.len() {
            if !result.macd_line[i].is_nan() {
                assert!(
                    result.macd_line[i] < 0.0,
                    "MACD should be negative in downtrend at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_macd_fusion_histogram_equals_macd_minus_signal() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd_fusion(&data, 5, 10, 3).unwrap();

        for i in 0..result.len() {
            if !result.histogram[i].is_nan() {
                let expected = result.macd_line[i] - result.signal_line[i];
                assert!(
                    approx_eq(result.histogram[i], expected, EPSILON),
                    "Histogram mismatch at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_macd_fusion_f32() {
        let data: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32) * 0.5).collect();
        let result = macd_fusion(&data, 12, 26, 9).unwrap();

        assert!(!result.macd_line[25].is_nan());
        assert!(!result.signal_line[33].is_nan());
    }

    // ==================== macd_fusion Error Tests ====================

    #[test]
    fn test_macd_fusion_empty_input() {
        let data: Vec<f64> = vec![];
        let result = macd_fusion(&data, 12, 26, 9);

        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_macd_fusion_zero_fast_period() {
        let data = vec![1.0_f64; 50];
        let result = macd_fusion(&data, 0, 26, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_fusion_zero_slow_period() {
        let data = vec![1.0_f64; 50];
        let result = macd_fusion(&data, 12, 0, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_fusion_zero_signal_period() {
        let data = vec![1.0_f64; 50];
        let result = macd_fusion(&data, 12, 26, 0);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_macd_fusion_fast_ge_slow() {
        let data = vec![1.0_f64; 50];
        let result = macd_fusion(&data, 26, 12, 9);

        assert!(matches!(result, Err(Error::InvalidPeriod { .. })));
    }

    #[test]
    fn test_macd_fusion_insufficient_data() {
        let data = vec![1.0_f64; 30]; // Need at least 26 + 9 - 1 = 34
        let result = macd_fusion(&data, 12, 26, 9);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    // ==================== Into Variant Tests ====================

    #[test]
    fn test_ema_multi_into_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let periods = vec![2, 3];
        let mut output1 = vec![0.0_f64; 10];
        let mut output2 = vec![0.0_f64; 10];
        let mut outputs: Vec<&mut [f64]> = vec![&mut output1, &mut output2];

        let valid_counts = ema_multi_into(&data, &periods, &mut outputs).unwrap();

        assert_eq!(valid_counts.len(), 2);
        assert_eq!(valid_counts[0], 9); // 10 - 2 + 1
        assert_eq!(valid_counts[1], 8); // 10 - 3 + 1

        // Verify values match ema_multi
        let fused = ema_multi(&data, &periods).unwrap();
        for i in 0..10 {
            assert!(approx_eq(output1[i], fused[0][i], EPSILON));
            assert!(approx_eq(output2[i], fused[1][i], EPSILON));
        }
    }

    #[test]
    fn test_ema_fusion_into_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = EmaFusionOutput {
            ema: vec![0.0_f64; 10],
            dema: vec![0.0_f64; 10],
            tema: vec![0.0_f64; 10],
        };

        let (valid_ema, valid_dema, valid_tema) = ema_fusion_into(&data, 3, &mut output).unwrap();

        assert_eq!(valid_ema, 8); // 10 - 2
        assert_eq!(valid_dema, 6); // 10 - 4
        assert_eq!(valid_tema, 4); // 10 - 6

        // Verify values match ema_fusion
        let fused = ema_fusion(&data, 3).unwrap();
        for i in 0..10 {
            assert!(approx_eq(output.ema[i], fused.ema[i], EPSILON));
            assert!(approx_eq(output.dema[i], fused.dema[i], EPSILON));
            assert!(approx_eq(output.tema[i], fused.tema[i], EPSILON));
        }
    }

    #[test]
    fn test_macd_fusion_into_basic() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let mut output = MacdFusionOutput {
            macd_line: vec![0.0_f64; 50],
            signal_line: vec![0.0_f64; 50],
            histogram: vec![0.0_f64; 50],
        };

        let (valid_macd, valid_signal) =
            macd_fusion_into(&data, 12, 26, 9, &mut output).unwrap();

        assert_eq!(valid_macd, 25); // 50 - 25
        assert_eq!(valid_signal, 17); // 50 - 33

        // Verify values match macd_fusion
        let fused = macd_fusion(&data, 12, 26, 9).unwrap();
        for i in 0..50 {
            assert!(approx_eq(output.macd_line[i], fused.macd_line[i], EPSILON));
            assert!(approx_eq(output.signal_line[i], fused.signal_line[i], EPSILON));
            assert!(approx_eq(output.histogram[i], fused.histogram[i], EPSILON));
        }
    }

    // ==================== Output Struct Tests ====================

    #[test]
    fn test_ema_fusion_output_len_is_empty() {
        let output = EmaFusionOutput {
            ema: vec![1.0_f64, 2.0, 3.0],
            dema: vec![1.0_f64, 2.0, 3.0],
            tema: vec![1.0_f64, 2.0, 3.0],
        };

        assert_eq!(output.len(), 3);
        assert!(!output.is_empty());

        let empty_output: EmaFusionOutput<f64> = EmaFusionOutput {
            ema: vec![],
            dema: vec![],
            tema: vec![],
        };

        assert_eq!(empty_output.len(), 0);
        assert!(empty_output.is_empty());
    }

    #[test]
    fn test_macd_fusion_output_len_is_empty() {
        let output = MacdFusionOutput {
            macd_line: vec![1.0_f64, 2.0, 3.0],
            signal_line: vec![1.0_f64, 2.0, 3.0],
            histogram: vec![1.0_f64, 2.0, 3.0],
        };

        assert_eq!(output.len(), 3);
        assert!(!output.is_empty());

        let empty_output: MacdFusionOutput<f64> = MacdFusionOutput {
            macd_line: vec![],
            signal_line: vec![],
            histogram: vec![],
        };

        assert_eq!(empty_output.len(), 0);
        assert!(empty_output.is_empty());
    }

    #[test]
    fn test_ema_fusion_output_clone() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema_fusion(&data, 3).unwrap();
        let cloned = result.clone();

        for i in 0..result.len() {
            assert!(approx_eq(result.ema[i], cloned.ema[i], EPSILON));
            assert!(approx_eq(result.dema[i], cloned.dema[i], EPSILON));
            assert!(approx_eq(result.tema[i], cloned.tema[i], EPSILON));
        }
    }

    #[test]
    fn test_macd_fusion_output_clone() {
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = macd_fusion(&data, 12, 26, 9).unwrap();
        let cloned = result.clone();

        for i in 0..result.len() {
            assert!(approx_eq(result.macd_line[i], cloned.macd_line[i], EPSILON));
            assert!(approx_eq(result.signal_line[i], cloned.signal_line[i], EPSILON));
            assert!(approx_eq(result.histogram[i], cloned.histogram[i], EPSILON));
        }
    }

    // ==================== Property Tests ====================

    #[test]
    fn test_ema_multi_output_lengths() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let periods = vec![5, 10, 20];
        let result = ema_multi(&data, &periods).unwrap();

        for (i, output) in result.iter().enumerate() {
            assert_eq!(
                output.len(),
                data.len(),
                "Output {} should have same length as input",
                i
            );
        }
    }

    #[test]
    fn test_ema_fusion_nan_counts() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        for period in [2, 3, 5, 10] {
            let result = ema_fusion(&data, period).unwrap();

            let ema_nan_count = result.ema.iter().filter(|x| x.is_nan()).count();
            let dema_nan_count = result.dema.iter().filter(|x| x.is_nan()).count();
            let tema_nan_count = result.tema.iter().filter(|x| x.is_nan()).count();

            assert_eq!(
                ema_nan_count,
                period - 1,
                "EMA NaN count for period {}",
                period
            );
            assert_eq!(
                dema_nan_count,
                2 * period - 2,
                "DEMA NaN count for period {}",
                period
            );
            assert_eq!(
                tema_nan_count,
                3 * period - 3,
                "TEMA NaN count for period {}",
                period
            );
        }
    }

    #[test]
    fn test_macd_fusion_nan_counts() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        for (fast, slow, signal) in [(3, 5, 2), (5, 10, 3), (12, 26, 9)] {
            let result = macd_fusion(&data, fast, slow, signal).unwrap();

            let macd_nan_count = result.macd_line.iter().filter(|x| x.is_nan()).count();
            let signal_nan_count = result.signal_line.iter().filter(|x| x.is_nan()).count();

            assert_eq!(
                macd_nan_count,
                slow - 1,
                "MACD NaN count for ({}, {}, {})",
                fast,
                slow,
                signal
            );
            assert_eq!(
                signal_nan_count,
                slow + signal - 2,
                "Signal NaN count for ({}, {}, {})",
                fast,
                slow,
                signal
            );
        }
    }

    // ==================== Trend Response Tests ====================

    #[test]
    fn test_dema_responds_faster_than_ema() {
        // DEMA should track trends more closely than EMA
        let data: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = ema_fusion(&data, 5).unwrap();

        // For an uptrend, DEMA should be closer to the current price than EMA
        for i in 10..data.len() {
            if !result.dema[i].is_nan() && !result.ema[i].is_nan() {
                let ema_lag = data[i] - result.ema[i];
                let dema_lag = data[i] - result.dema[i];
                assert!(
                    dema_lag < ema_lag,
                    "DEMA should lag less than EMA at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_tema_responds_faster_than_dema() {
        // TEMA should track trends even more closely than DEMA
        let data: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = ema_fusion(&data, 5).unwrap();

        // For an uptrend, TEMA should be closer to the current price than DEMA
        for i in 15..data.len() {
            if !result.tema[i].is_nan() && !result.dema[i].is_nan() {
                let dema_lag = data[i] - result.dema[i];
                let tema_lag = data[i] - result.tema[i];
                assert!(
                    tema_lag < dema_lag,
                    "TEMA should lag less than DEMA at index {}",
                    i
                );
            }
        }
    }
}
