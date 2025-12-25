//! `HT_TRENDLINE` (Hilbert Transform - Instantaneous Trendline) indicator.
//!
//! The Hilbert Transform Trendline uses signal processing techniques to
//! compute an adaptive trendline based on the dominant cycle period in the data.
//!
//! This implementation is based on John Ehlers' work on applying the Hilbert
//! Transform to financial market data.
//!
//! # Algorithm
//!
//! 1. Compute smoothed price using a weighted moving average
//! 2. Apply Hilbert Transform to extract in-phase (I) and quadrature (Q) components
//! 3. Estimate the dominant cycle period from the phase relationship
//! 4. Use the period to compute an adaptive smoothed trendline
//!
//! # Lookback
//!
//! The lookback period is 63 bars (warm-up period for the Hilbert Transform).

use crate::error::{Error, Result};
use crate::traits::SeriesElement;
use std::f64::consts::PI;

/// Computes the lookback period for `HT_TRENDLINE`.
///
/// The Hilbert Transform requires a warm-up period of 63 bars.
#[inline]
#[must_use]
pub const fn ht_trendline_lookback() -> usize {
    63
}

/// Returns the minimum input length required for `HT_TRENDLINE` calculation.
#[inline]
#[must_use]
pub const fn ht_trendline_min_len() -> usize {
    64
}

/// Computes Hilbert Transform Trendline and stores results in output.
///
/// # Arguments
///
/// * `data` - Input price data (typically close or HL2)
/// * `output` - Pre-allocated output slice
///
/// # Returns
///
/// * `Ok(())` on success
/// * `Err(Error)` if inputs are invalid
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - The period is invalid (`Error::InvalidPeriod`)
/// - There is insufficient data for the lookback (`Error::InsufficientData`)
/// - The output buffer is too small (`Error::BufferTooSmall`)
pub fn ht_trendline_into<T: SeriesElement>(data: &[T], output: &mut [T]) -> Result<()> {
    if data.is_empty() {
        return Err(Error::EmptyInput);
    }

    let n = data.len();
    let lookback = ht_trendline_lookback();
    let min_len = ht_trendline_min_len();

    if n < min_len {
        return Err(Error::InsufficientData {
            indicator: "ht_trendline",
            required: min_len,
            actual: n,
        });
    }

    if output.len() < n {
        return Err(Error::BufferTooSmall {
            indicator: "ht_trendline",
            required: n,
            actual: output.len(),
        });
    }

    // Fill lookback period with NaN
    for i in 0..lookback {
        output[i] = T::nan();
    }

    // Constants for the algorithm
    let two_pi = T::from_f64(2.0 * PI)?;

    // Smoothing coefficients
    let a = T::from_f64(0.0962)?;
    let b = T::from_f64(0.5769)?;

    // State variables
    let mut smooth = vec![T::zero(); n];
    let mut detrender = vec![T::zero(); n];
    let mut i1 = vec![T::zero(); n];
    let mut q1 = vec![T::zero(); n];
    let mut ji = vec![T::zero(); n];
    let mut jq = vec![T::zero(); n];
    let mut i2 = vec![T::zero(); n];
    let mut q2 = vec![T::zero(); n];
    let mut re = vec![T::zero(); n];
    let mut im = vec![T::zero(); n];
    let mut period = vec![T::zero(); n];
    let mut smooth_period = vec![T::zero(); n];
    let mut trendline = vec![T::zero(); n];

    // Initial period estimate
    let six = T::from_usize(6)?;
    for i in 0..n.min(12) {
        period[i] = six;
        smooth_period[i] = six;
    }

    // WMA coefficients
    let c1 = T::from_f64(4.0)?;
    let c2 = T::from_f64(3.0)?;
    let c3 = T::from_f64(2.0)?;
    let c4 = T::one();
    let c_sum = T::from_f64(10.0)?;

    // Main calculation loop
    for i in 6..n {
        // Compute smoothed price (4-bar WMA)
        if i >= 3 {
            smooth[i] =
                (c1 * data[i] + c2 * data[i - 1] + c3 * data[i - 2] + c4 * data[i - 3]) / c_sum;
        } else {
            smooth[i] = data[i];
        }

        // Compute Hilbert Transform components
        if i >= 6 {
            let adj_prev_period = T::from_f64(0.075)? * period[i - 1] + T::from_f64(0.54)?;

            // Detrender
            detrender[i] =
                (a * smooth[i] + b * smooth[i - 2] - b * smooth[i - 4] - a * smooth[i - 6])
                    * adj_prev_period;

            // Compute InPhase and Quadrature components
            q1[i] = (a * detrender[i] + b * detrender[i - 2]
                - b * detrender[i - 4]
                - a * detrender[i - 6])
                * adj_prev_period;
            i1[i] = detrender[i - 3];

            // Advance the phase of I1 and Q1 by 90 degrees
            ji[i] = (a * i1[i] + b * i1[i - 2] - b * i1[i - 4] - a * i1[i - 6]) * adj_prev_period;
            jq[i] = (a * q1[i] + b * q1[i - 2] - b * q1[i - 4] - a * q1[i - 6]) * adj_prev_period;

            // Phasor addition for 3-bar averaging
            i2[i] = i1[i] - jq[i];
            q2[i] = q1[i] + ji[i];

            // Smooth the I and Q components
            let smooth_coef = T::from_f64(0.2)?;
            i2[i] = smooth_coef * i2[i] + (T::one() - smooth_coef) * i2[i - 1];
            q2[i] = smooth_coef * q2[i] + (T::one() - smooth_coef) * q2[i - 1];

            // Homodyne Discriminator
            re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
            im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];

            re[i] = smooth_coef * re[i] + (T::one() - smooth_coef) * re[i - 1];
            im[i] = smooth_coef * im[i] + (T::one() - smooth_coef) * im[i - 1];

            // Compute period
            if im[i] != T::zero() && re[i] != T::zero() {
                period[i] = two_pi / (im[i] / re[i]).atan();
            }

            // Limit period to reasonable range
            let min_period = T::from_f64(6.0)?;
            let max_period = T::from_f64(50.0)?;
            if period[i] > max_period {
                period[i] = max_period;
            }
            if period[i] < min_period {
                period[i] = min_period;
            }

            // Smooth the period
            let period_smooth = T::from_f64(0.33)?;
            smooth_period[i] =
                period_smooth * period[i] + (T::one() - period_smooth) * smooth_period[i - 1];

            // Compute the trendline using the dominant cycle period
            let dc_period = smooth_period[i];
            let half_period = (dc_period / T::from_f64(2.0)?).floor();
            let hp_int = half_period.to_f64().unwrap_or(3.0) as usize;
            let hp_int = hp_int.max(1).min(25);

            // WMA-style trendline based on dominant cycle
            if i >= hp_int {
                let mut sum = T::zero();
                let mut weight_sum = T::zero();
                for j in 0..hp_int {
                    let weight = T::from_usize(hp_int - j)?;
                    if i >= j {
                        sum = sum + weight * data[i - j];
                        weight_sum = weight_sum + weight;
                    }
                }
                if weight_sum > T::zero() {
                    trendline[i] = sum / weight_sum;
                } else {
                    trendline[i] = data[i];
                }
            } else {
                trendline[i] = data[i];
            }
        } else {
            // Initialize with simple values
            period[i] = six;
            smooth_period[i] = six;
            trendline[i] = data[i];
        }
    }

    // Copy trendline to output
    for i in lookback..n {
        output[i] = trendline[i];
    }

    Ok(())
}

/// Computes Hilbert Transform Trendline.
///
/// # Arguments
///
/// * `data` - Input price data (typically close or HL2)
///
/// # Returns
///
/// * `Ok(Vec<T>)` - Vector of trendline values
/// * `Err(Error)` if inputs are invalid
///
/// # Example
///
/// ```
/// use fast_ta::indicators::ht_trendline;
///
/// let mut prices: Vec<f64> = Vec::with_capacity(100);
/// for x in 1..=100 {
///     prices.push(50.0 + (x as f64 * 0.1).sin() * 10.0);
/// }
/// let result = ht_trendline(&prices).unwrap();
/// assert!(result[0].is_nan()); // First 63 values are NaN
/// assert!(result[63].is_finite());
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - The period is invalid (`Error::InvalidPeriod`)
/// - There is insufficient data for the lookback (`Error::InsufficientData`)
pub fn ht_trendline<T: SeriesElement>(data: &[T]) -> Result<Vec<T>> {
    let mut output = vec![T::nan(); data.len()];
    ht_trendline_into(data, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::all, clippy::pedantic, clippy::nursery)]
    use super::*;

    #[test]
    fn test_ht_trendline_lookback() {
        assert_eq!(ht_trendline_lookback(), 63);
    }

    #[test]
    fn test_ht_trendline_min_len() {
        assert_eq!(ht_trendline_min_len(), 64);
    }

    #[test]
    fn test_ht_trendline_empty_input() {
        let data: Vec<f64> = vec![];
        let result = ht_trendline(&data);
        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_ht_trendline_insufficient_data() {
        let data: Vec<f64> = vec![1.0; 50];
        let result = ht_trendline(&data);
        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    #[test]
    fn test_ht_trendline_output_length() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ht_trendline(&data).unwrap();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_ht_trendline_nan_count() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ht_trendline(&data).unwrap();

        let lookback = ht_trendline_lookback();
        let nan_count = result.iter().filter(|x| x.is_nan()).count();
        assert_eq!(nan_count, lookback);
    }

    #[test]
    fn test_ht_trendline_valid_values() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ht_trendline(&data).unwrap();

        let lookback = ht_trendline_lookback();
        for i in lookback..result.len() {
            assert!(result[i].is_finite(), "result[{}] should be finite", i);
        }
    }

    #[test]
    fn test_ht_trendline_trending_data() {
        // Linear uptrend
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = ht_trendline(&data).unwrap();

        let lookback = ht_trendline_lookback();
        // Trendline should generally increase in uptrend
        let mut increasing_count = 0;
        for i in (lookback + 1)..result.len() {
            if result[i] > result[i - 1] {
                increasing_count += 1;
            }
        }
        // Most values should be increasing
        let total = result.len() - lookback - 1;
        assert!(
            increasing_count > total / 2,
            "Trendline should mostly increase in uptrend: {} of {} increasing",
            increasing_count,
            total
        );
    }

    #[test]
    fn test_ht_trendline_cyclic_data() {
        // Sinusoidal data to test cycle detection
        let data: Vec<f64> = (0..200)
            .map(|x| 50.0 + (x as f64 * 0.2).sin() * 10.0)
            .collect();
        let result = ht_trendline(&data).unwrap();

        let lookback = ht_trendline_lookback();
        for i in lookback..result.len() {
            assert!(result[i].is_finite());
            // Trendline should be within reasonable range of the data
            assert!(result[i] > 30.0 && result[i] < 70.0);
        }
    }

    #[test]
    fn test_ht_trendline_constant_data() {
        let data: Vec<f64> = vec![50.0; 100];
        let result = ht_trendline(&data).unwrap();

        let lookback = ht_trendline_lookback();
        for i in lookback..result.len() {
            assert!(result[i].is_finite());
            // For constant data, trendline should be close to the constant value
            assert!(
                (result[i] - 50.0).abs() < 1.0,
                "result[{}]={} should be close to 50.0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_ht_trendline_into() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let mut output = vec![0.0_f64; data.len()];
        ht_trendline_into(&data, &mut output).unwrap();

        let lookback = ht_trendline_lookback();
        for i in 0..lookback {
            assert!(output[i].is_nan());
        }
        for i in lookback..output.len() {
            assert!(output[i].is_finite());
        }
    }

    #[test]
    fn test_ht_trendline_into_buffer_too_small() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let mut output = vec![0.0_f64; 50]; // Too small
        let result = ht_trendline_into(&data, &mut output);
        assert!(matches!(result, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn test_ht_trendline_f32() {
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let result = ht_trendline(&data).unwrap();

        assert_eq!(result.len(), data.len());
        let lookback = ht_trendline_lookback();
        for i in 0..lookback {
            assert!(result[i].is_nan());
        }
        for i in lookback..result.len() {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_ht_trendline_minimum_length() {
        let data: Vec<f64> = (1..=64).map(|x| x as f64).collect();
        let result = ht_trendline(&data).unwrap();

        assert_eq!(result.len(), 64);
        // First 63 should be NaN, last one valid
        for i in 0..63 {
            assert!(result[i].is_nan());
        }
        assert!(result[63].is_finite());
    }
}
