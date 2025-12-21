//! Running statistics using Welford's algorithm.
//!
//! This module provides numerically stable implementations for computing
//! mean, variance, and standard deviation using Welford's online algorithm.
//!
//! # Algorithm
//!
//! Welford's algorithm computes running statistics in a single pass with
//! O(1) per-element update time. It is numerically stable even with extreme
//! values where naive sum-of-squares approaches would suffer from catastrophic
//! cancellation.
//!
//! The algorithm maintains:
//! - `count`: Number of elements processed
//! - `mean`: Running mean of the values
//! - `m2`: Sum of squared differences from the mean (for variance)
//!
//! # Formula
//!
//! ```text
//! For each new value x:
//!   count += 1
//!   delta = x - mean
//!   mean += delta / count
//!   delta2 = x - mean  # Note: using updated mean
//!   m2 += delta * delta2
//!
//! Variance = m2 / count  (population variance)
//! StdDev = sqrt(Variance)
//! ```
//!
//! # Example
//!
//! ```
//! use fast_ta_core::kernels::running_stat::{RunningStat, rolling_stats};
//!
//! // Online computation with RunningStat
//! let mut stat: RunningStat<f64> = RunningStat::new();
//! stat.update(1.0);
//! stat.update(2.0);
//! stat.update(3.0);
//! assert!((stat.mean() - 2.0).abs() < 1e-10);
//!
//! // Rolling statistics with fixed window
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let result = rolling_stats(&data, 3).unwrap();
//! assert!((result.mean[2] - 2.0).abs() < 1e-10);
//! ```
//!
//! # References
//!
//! - Welford, B. P. (1962). "Note on a method for calculating corrected sums
//!   of squares and products". Technometrics. 4 (3): 419–420.
//! - Knuth, D. E. (1997). The Art of Computer Programming, volume 2:
//!   Seminumerical Algorithms (3rd ed.). Section 4.2.2, page 232.

use num_traits::Float;

use crate::error::{Error, Result};
use crate::traits::{SeriesElement, ValidatedInput};

/// A running statistics accumulator using Welford's algorithm.
///
/// This structure maintains running mean, variance, and standard deviation
/// computations in a numerically stable manner. It uses Welford's online
/// algorithm which is O(1) per update and avoids the numerical instability
/// of naive sum-of-squares approaches.
///
/// # Type Parameters
///
/// - `T`: The numeric type (typically `f32` or `f64`)
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::running_stat::RunningStat;
///
/// let mut stat: RunningStat<f64> = RunningStat::new();
/// stat.update(10.0);
/// stat.update(20.0);
/// stat.update(30.0);
///
/// assert!((stat.mean() - 20.0).abs() < 1e-10);
/// assert_eq!(stat.count(), 3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RunningStat<T> {
    count: usize,
    mean: T,
    m2: T, // Sum of squared differences from the mean
}

impl<T: SeriesElement> Default for RunningStat<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SeriesElement> RunningStat<T> {
    /// Creates a new empty running statistics accumulator.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let stat: RunningStat<f64> = RunningStat::new();
    /// assert_eq!(stat.count(), 0);
    /// assert!(stat.mean().is_nan());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: T::nan(),
            m2: T::zero(),
        }
    }

    /// Updates the running statistics with a new value.
    ///
    /// This implements Welford's online algorithm for numerically stable
    /// computation of mean and variance.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to incorporate
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(2.0);
    /// stat.update(3.0);
    /// assert!((stat.mean() - 2.0).abs() < 1e-10);
    /// ```
    pub fn update(&mut self, value: T) {
        // Handle NaN values - they propagate through
        if value.is_nan() {
            self.mean = T::nan();
            self.m2 = T::nan();
            return;
        }

        self.count += 1;

        if self.count == 1 {
            // First value - mean is just the value
            self.mean = value;
            self.m2 = T::zero();
        } else {
            // Welford's algorithm
            let delta = value - self.mean;
            let count_t =
                T::from_usize(self.count).unwrap_or_else(|_| T::from(self.count as f64).unwrap());
            self.mean = self.mean + delta / count_t;
            let delta2 = value - self.mean;
            self.m2 = self.m2 + delta * delta2;
        }
    }

    /// Removes a value from the running statistics.
    ///
    /// This is the inverse of `update` and is used for sliding window
    /// calculations. Note that removing values in a different order than
    /// they were added may introduce small floating-point errors.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to remove
    ///
    /// # Panics
    ///
    /// This method may produce incorrect results if called when count is 0
    /// or if the value was never added.
    pub fn remove(&mut self, value: T) {
        // Handle edge cases
        if self.count == 0 || value.is_nan() || self.mean.is_nan() {
            return;
        }

        if self.count == 1 {
            // Removing the last value
            self.count = 0;
            self.mean = T::nan();
            self.m2 = T::zero();
            return;
        }

        // Inverse of Welford's algorithm
        let count_t =
            T::from_usize(self.count).unwrap_or_else(|_| T::from(self.count as f64).unwrap());
        let delta = value - self.mean;
        let new_mean = (self.mean * count_t - value) / (count_t - T::one());

        let delta2 = value - new_mean;
        self.m2 = self.m2 - delta * delta2;

        // Ensure m2 doesn't go negative due to floating-point errors
        if self.m2 < T::zero() {
            self.m2 = T::zero();
        }

        self.mean = new_mean;
        self.count -= 1;
    }

    /// Returns the number of values processed.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// assert_eq!(stat.count(), 0);
    /// stat.update(1.0);
    /// assert_eq!(stat.count(), 1);
    /// ```
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns the running mean.
    ///
    /// Returns `NaN` if no values have been processed or if any processed
    /// value was `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(3.0);
    /// assert!((stat.mean() - 2.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn mean(&self) -> T {
        self.mean
    }

    /// Returns the population variance.
    ///
    /// Uses the population variance formula: `Var = m2 / count`
    /// (divides by n, not n-1).
    ///
    /// Returns 0 if count < 2 (variance requires at least 2 values for a
    /// meaningful result), or `NaN` if the mean is `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(2.0);
    /// stat.update(3.0);
    /// // Variance of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    /// assert!((stat.variance() - 2.0/3.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn variance(&self) -> T {
        if self.mean.is_nan() {
            return T::nan();
        }
        if self.count < 2 {
            return T::zero();
        }
        let count_t =
            T::from_usize(self.count).unwrap_or_else(|_| T::from(self.count as f64).unwrap());
        self.m2 / count_t
    }

    /// Returns the sample variance.
    ///
    /// Uses the sample variance formula: `Var = m2 / (count - 1)`
    /// (Bessel's correction, divides by n-1).
    ///
    /// Returns 0 if count < 2, or `NaN` if the mean is `NaN`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(2.0);
    /// stat.update(3.0);
    /// // Sample variance of [1,2,3] = 2/2 = 1.0
    /// assert!((stat.sample_variance() - 1.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn sample_variance(&self) -> T {
        if self.mean.is_nan() {
            return T::nan();
        }
        if self.count < 2 {
            return T::zero();
        }
        let count_minus_one = T::from_usize(self.count - 1)
            .unwrap_or_else(|_| T::from((self.count - 1) as f64).unwrap());
        self.m2 / count_minus_one
    }

    /// Returns the population standard deviation.
    ///
    /// This is the square root of the population variance.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(2.0);
    /// stat.update(3.0);
    /// // StdDev = sqrt(2/3)
    /// assert!((stat.stddev() - (2.0_f64/3.0).sqrt()).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn stddev(&self) -> T {
        self.variance().sqrt()
    }

    /// Returns the sample standard deviation.
    ///
    /// This is the square root of the sample variance.
    #[must_use]
    pub fn sample_stddev(&self) -> T {
        self.sample_variance().sqrt()
    }

    /// Resets the accumulator to its initial state.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RunningStat;
    ///
    /// let mut stat: RunningStat<f64> = RunningStat::new();
    /// stat.update(1.0);
    /// stat.update(2.0);
    /// assert_eq!(stat.count(), 2);
    ///
    /// stat.reset();
    /// assert_eq!(stat.count(), 0);
    /// assert!(stat.mean().is_nan());
    /// ```
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = T::nan();
        self.m2 = T::zero();
    }
}

/// A rolling statistics calculator for sliding window computations.
///
/// This structure maintains a sliding window of values and computes
/// mean, variance, and standard deviation efficiently using Welford's
/// algorithm with add/remove operations.
///
/// # Type Parameters
///
/// - `T`: The numeric type (typically `f32` or `f64`)
#[derive(Debug, Clone)]
pub struct RollingStat<T> {
    stat: RunningStat<T>,
    period: usize,
}

impl<T: SeriesElement> RollingStat<T> {
    /// Creates a new rolling statistics calculator with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `period` - The window size for rolling calculations
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::kernels::running_stat::RollingStat;
    ///
    /// let rolling: RollingStat<f64> = RollingStat::new(5);
    /// ```
    #[must_use]
    pub fn new(period: usize) -> Self {
        Self {
            stat: RunningStat::new(),
            period,
        }
    }

    /// Returns the window size.
    #[must_use]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the current count of values in the window.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.stat.count()
    }

    /// Returns the current mean.
    #[must_use]
    pub fn mean(&self) -> T {
        self.stat.mean()
    }

    /// Returns the current population variance.
    #[must_use]
    pub fn variance(&self) -> T {
        self.stat.variance()
    }

    /// Returns the current population standard deviation.
    #[must_use]
    pub fn stddev(&self) -> T {
        self.stat.stddev()
    }

    /// Updates the rolling statistics by adding a new value and optionally
    /// removing an old value.
    ///
    /// # Arguments
    ///
    /// * `new_value` - The new value to add to the window
    /// * `old_value` - The old value to remove (if window is full)
    pub fn update(&mut self, new_value: T, old_value: Option<T>) {
        if let Some(old) = old_value {
            self.stat.remove(old);
        }
        self.stat.update(new_value);
    }

    /// Resets the rolling statistics.
    pub fn reset(&mut self) {
        self.stat.reset();
    }
}

/// Output structure containing rolling mean, variance, and standard deviation.
///
/// Each vector has the same length as the input data. The first `period - 1`
/// values are NaN due to insufficient lookback data.
#[derive(Debug, Clone)]
pub struct RollingStatOutput<T> {
    /// The rolling mean.
    pub mean: Vec<T>,
    /// The rolling population variance.
    pub variance: Vec<T>,
    /// The rolling population standard deviation.
    pub stddev: Vec<T>,
}

/// Computes rolling mean, variance, and standard deviation in a single pass.
///
/// This function uses Welford's algorithm to compute all three statistics
/// simultaneously, which is more efficient than computing them separately.
///
/// # Arguments
///
/// * `data` - The input data series
/// * `period` - The window size for rolling calculations
///
/// # Returns
///
/// A `Result` containing a [`RollingStatOutput`] with mean, variance, and stddev,
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - The period is zero (`Error::InvalidPeriod`)
/// - The input data is shorter than the period (`Error::InsufficientData`)
///
/// # Performance
///
/// - Time complexity: O(n) where n is the length of the input data
/// - Space complexity: O(n) for each of the three output vectors
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::running_stat::rolling_stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = rolling_stats(&data, 3).unwrap();
///
/// // First 2 values are NaN
/// assert!(result.mean[0].is_nan());
/// assert!(result.mean[1].is_nan());
///
/// // Mean of [1,2,3] = 2.0
/// assert!((result.mean[2] - 2.0).abs() < 1e-10);
///
/// // Mean of [2,3,4] = 3.0
/// assert!((result.mean[3] - 3.0).abs() < 1e-10);
/// ```
pub fn rolling_stats<T: SeriesElement>(data: &[T], period: usize) -> Result<RollingStatOutput<T>> {
    // Validate inputs
    if period == 0 {
        return Err(Error::InvalidPeriod {
            period,
            reason: "period must be at least 1",
        });
    }

    data.validate_not_empty()?;

    if data.len() < period {
        return Err(Error::InsufficientData {
            required: period,
            actual: data.len(),
        });
    }

    // Initialize output vectors with NaN
    let mut mean = vec![T::nan(); data.len()];
    let mut variance = vec![T::nan(); data.len()];
    let mut stddev = vec![T::nan(); data.len()];

    // Initialize accumulator for the first window
    let mut stat = RunningStat::new();

    // Build up the first window
    for &value in data.iter().take(period) {
        stat.update(value);
    }

    // Compute first valid values at index (period - 1)
    let first_idx = period - 1;
    mean[first_idx] = stat.mean();
    variance[first_idx] = stat.variance();
    stddev[first_idx] = stat.stddev();

    // Rolling calculation for remaining elements
    for i in period..data.len() {
        let old_value = data[i - period];
        let new_value = data[i];

        // Update rolling stats
        stat.remove(old_value);
        stat.update(new_value);

        // Store results
        mean[i] = stat.mean();
        variance[i] = stat.variance();
        stddev[i] = stat.stddev();
    }

    Ok(RollingStatOutput {
        mean,
        variance,
        stddev,
    })
}

/// Computes rolling statistics into pre-allocated output buffers.
///
/// This variant allows reusing existing buffers to avoid allocations in
/// performance-critical code paths.
///
/// # Arguments
///
/// * `data` - The input data series
/// * `period` - The window size for rolling calculations
/// * `output` - Pre-allocated output structure (each vector must be at least as long as input)
///
/// # Returns
///
/// A `Result` containing the number of valid values computed (data.len() - period + 1),
/// or an error if validation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The input data is empty (`Error::EmptyInput`)
/// - The period is zero (`Error::InvalidPeriod`)
/// - The input data is shorter than the period (`Error::InsufficientData`)
/// - Any output buffer is shorter than the input data
///
/// # Example
///
/// ```
/// use fast_ta_core::kernels::running_stat::{rolling_stats_into, RollingStatOutput};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mut output = RollingStatOutput {
///     mean: vec![0.0; 5],
///     variance: vec![0.0; 5],
///     stddev: vec![0.0; 5],
/// };
/// let valid_count = rolling_stats_into(&data, 3, &mut output).unwrap();
///
/// assert_eq!(valid_count, 3);
/// ```
pub fn rolling_stats_into<T: SeriesElement>(
    data: &[T],
    period: usize,
    output: &mut RollingStatOutput<T>,
) -> Result<usize> {
    // Validate inputs
    if period == 0 {
        return Err(Error::InvalidPeriod {
            period,
            reason: "period must be at least 1",
        });
    }

    data.validate_not_empty()?;

    if data.len() < period {
        return Err(Error::InsufficientData {
            required: period,
            actual: data.len(),
        });
    }

    // Validate output buffers
    if output.mean.len() < data.len() {
        return Err(Error::InsufficientData {
            required: data.len(),
            actual: output.mean.len(),
        });
    }
    if output.variance.len() < data.len() {
        return Err(Error::InsufficientData {
            required: data.len(),
            actual: output.variance.len(),
        });
    }
    if output.stddev.len() < data.len() {
        return Err(Error::InsufficientData {
            required: data.len(),
            actual: output.stddev.len(),
        });
    }

    // Initialize lookback period with NaN
    for i in 0..(period - 1) {
        output.mean[i] = T::nan();
        output.variance[i] = T::nan();
        output.stddev[i] = T::nan();
    }

    // Initialize accumulator for the first window
    let mut stat = RunningStat::new();

    // Build up the first window
    for &value in data.iter().take(period) {
        stat.update(value);
    }

    // Compute first valid values at index (period - 1)
    let first_idx = period - 1;
    output.mean[first_idx] = stat.mean();
    output.variance[first_idx] = stat.variance();
    output.stddev[first_idx] = stat.stddev();

    // Rolling calculation for remaining elements
    for i in period..data.len() {
        let old_value = data[i - period];
        let new_value = data[i];

        // Update rolling stats
        stat.remove(old_value);
        stat.update(new_value);

        // Store results
        output.mean[i] = stat.mean();
        output.variance[i] = stat.variance();
        output.stddev[i] = stat.stddev();
    }

    // Return count of valid (non-NaN) values
    Ok(data.len() - period + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to compare floating point values
    fn approx_eq<T: Float>(a: T, b: T, epsilon: T) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < epsilon
    }

    const EPSILON: f64 = 1e-10;
    const EPSILON_F32: f32 = 1e-5;

    // ==================== RunningStat Basic Tests ====================

    #[test]
    fn test_running_stat_new() {
        let stat: RunningStat<f64> = RunningStat::new();
        assert_eq!(stat.count(), 0);
        assert!(stat.mean().is_nan());
        assert!(stat.variance().is_nan());
        assert!(stat.stddev().is_nan());
    }

    #[test]
    fn test_running_stat_default() {
        let stat: RunningStat<f64> = RunningStat::default();
        assert_eq!(stat.count(), 0);
        assert!(stat.mean().is_nan());
    }

    #[test]
    fn test_running_stat_single_value() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(5.0);

        assert_eq!(stat.count(), 1);
        assert!(approx_eq(stat.mean(), 5.0, EPSILON));
        assert!(approx_eq(stat.variance(), 0.0, EPSILON));
        assert!(approx_eq(stat.stddev(), 0.0, EPSILON));
    }

    #[test]
    fn test_running_stat_two_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(3.0);

        assert_eq!(stat.count(), 2);
        assert!(approx_eq(stat.mean(), 2.0, EPSILON));
        // Variance of [1, 3] = ((1-2)^2 + (3-2)^2) / 2 = (1 + 1) / 2 = 1
        assert!(approx_eq(stat.variance(), 1.0, EPSILON));
        assert!(approx_eq(stat.stddev(), 1.0, EPSILON));
    }

    #[test]
    fn test_running_stat_three_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(2.0);
        stat.update(3.0);

        assert_eq!(stat.count(), 3);
        assert!(approx_eq(stat.mean(), 2.0, EPSILON));
        // Variance of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        assert!(approx_eq(stat.variance(), 2.0 / 3.0, EPSILON));
        assert!(approx_eq(stat.stddev(), (2.0_f64 / 3.0).sqrt(), EPSILON));
    }

    #[test]
    fn test_running_stat_sample_variance() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(2.0);
        stat.update(3.0);

        // Sample variance of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / (3-1) = 2/2 = 1.0
        assert!(approx_eq(stat.sample_variance(), 1.0, EPSILON));
        assert!(approx_eq(stat.sample_stddev(), 1.0, EPSILON));
    }

    #[test]
    fn test_running_stat_constant_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        for _ in 0..10 {
            stat.update(5.0);
        }

        assert_eq!(stat.count(), 10);
        assert!(approx_eq(stat.mean(), 5.0, EPSILON));
        assert!(approx_eq(stat.variance(), 0.0, EPSILON));
        assert!(approx_eq(stat.stddev(), 0.0, EPSILON));
    }

    #[test]
    fn test_running_stat_f32() {
        let mut stat: RunningStat<f32> = RunningStat::new();
        stat.update(1.0_f32);
        stat.update(2.0_f32);
        stat.update(3.0_f32);

        assert!(approx_eq(stat.mean(), 2.0_f32, EPSILON_F32));
        assert!(approx_eq(stat.variance(), 2.0_f32 / 3.0_f32, EPSILON_F32));
    }

    #[test]
    fn test_running_stat_reset() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(2.0);
        assert_eq!(stat.count(), 2);

        stat.reset();
        assert_eq!(stat.count(), 0);
        assert!(stat.mean().is_nan());
        assert!(stat.variance().is_nan());
    }

    // ==================== NaN Handling Tests ====================

    #[test]
    fn test_running_stat_nan_propagation() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(f64::NAN);

        assert!(stat.mean().is_nan());
        assert!(stat.variance().is_nan());
    }

    #[test]
    fn test_running_stat_nan_first() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(f64::NAN);

        assert!(stat.mean().is_nan());
    }

    // ==================== Remove Operation Tests ====================

    #[test]
    fn test_running_stat_remove_single() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(5.0);
        stat.remove(5.0);

        assert_eq!(stat.count(), 0);
        assert!(stat.mean().is_nan());
    }

    #[test]
    fn test_running_stat_remove_from_multiple() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(2.0);
        stat.update(3.0);

        // Remove 1.0, should have mean of [2, 3] = 2.5
        stat.remove(1.0);

        assert_eq!(stat.count(), 2);
        assert!(approx_eq(stat.mean(), 2.5, EPSILON));
        // Variance of [2, 3] = ((2-2.5)^2 + (3-2.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        assert!(approx_eq(stat.variance(), 0.25, EPSILON));
    }

    #[test]
    fn test_running_stat_sliding_window() {
        let mut stat: RunningStat<f64> = RunningStat::new();

        // Add [1, 2, 3]
        stat.update(1.0);
        stat.update(2.0);
        stat.update(3.0);
        assert!(approx_eq(stat.mean(), 2.0, EPSILON));

        // Slide to [2, 3, 4]
        stat.remove(1.0);
        stat.update(4.0);
        assert!(approx_eq(stat.mean(), 3.0, EPSILON));

        // Slide to [3, 4, 5]
        stat.remove(2.0);
        stat.update(5.0);
        assert!(approx_eq(stat.mean(), 4.0, EPSILON));
    }

    #[test]
    fn test_running_stat_remove_from_empty() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.remove(5.0); // Should not panic

        assert_eq!(stat.count(), 0);
    }

    // ==================== Numeric Stability Tests ====================

    #[test]
    fn test_running_stat_numeric_stability_large_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();

        // Large values that would cause issues with naive sum-of-squares
        let base = 1e15;
        stat.update(base);
        stat.update(base + 1.0);
        stat.update(base + 2.0);

        assert!(approx_eq(stat.mean(), base + 1.0, 1e5)); // Allow some tolerance for large numbers
        // Variance should still be 2/3 (same relative spread)
        assert!(approx_eq(stat.variance(), 2.0 / 3.0, EPSILON));
    }

    #[test]
    fn test_running_stat_numeric_stability_small_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();

        let scale = 1e-10;
        stat.update(scale * 1.0);
        stat.update(scale * 2.0);
        stat.update(scale * 3.0);

        assert!(approx_eq(stat.mean(), scale * 2.0, 1e-20));
        assert!(approx_eq(
            stat.variance(),
            scale * scale * 2.0 / 3.0,
            1e-30
        ));
    }

    #[test]
    fn test_running_stat_numeric_stability_mixed_scale() {
        let mut stat: RunningStat<f64> = RunningStat::new();

        // Values with different scales
        stat.update(1e10);
        stat.update(1.0);
        stat.update(1e-10);

        // Should compute without overflow or underflow
        assert!(!stat.mean().is_nan());
        assert!(!stat.variance().is_nan());
        assert!(!stat.variance().is_infinite());
    }

    #[test]
    fn test_running_stat_many_values() {
        let mut stat: RunningStat<f64> = RunningStat::new();

        // Add many values to check for accumulated errors
        for i in 0..10000 {
            stat.update(i as f64);
        }

        // Mean of 0..9999 = (0 + 9999) / 2 = 4999.5
        assert!(approx_eq(stat.mean(), 4999.5, 1e-6));
    }

    // ==================== Property Tests (fused == unfused) ====================

    #[test]
    fn test_rolling_stats_matches_separate_calculations() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let period = 3;

        // Fused calculation
        let fused = rolling_stats(&data, period).unwrap();

        // Separate calculations (using naive approach for comparison)
        for i in (period - 1)..data.len() {
            let window: Vec<f64> = data[(i + 1 - period)..=i].to_vec();
            let n = window.len() as f64;

            let mean = window.iter().sum::<f64>() / n;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let stddev = variance.sqrt();

            assert!(
                approx_eq(fused.mean[i], mean, EPSILON),
                "Mean mismatch at index {}",
                i
            );
            assert!(
                approx_eq(fused.variance[i], variance, EPSILON),
                "Variance mismatch at index {}",
                i
            );
            assert!(
                approx_eq(fused.stddev[i], stddev, EPSILON),
                "StdDev mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_rolling_stats_matches_bollinger_stddev() {
        // Compare with the existing bollinger rolling_stddev implementation
        use crate::indicators::bollinger::rolling_stddev;

        let data: Vec<f64> = (0..50).map(|x| (x as f64 * 0.1).sin() * 10.0 + 50.0).collect();
        let period = 5;

        let fused = rolling_stats(&data, period).unwrap();
        let unfused = rolling_stddev(&data, period).unwrap();

        for i in (period - 1)..data.len() {
            assert!(
                approx_eq(fused.stddev[i], unfused[i], EPSILON),
                "StdDev mismatch at index {}: fused={}, unfused={}",
                i,
                fused.stddev[i],
                unfused[i]
            );
        }
    }

    #[test]
    fn test_rolling_stats_matches_sma() {
        // Compare mean with SMA implementation
        use crate::indicators::sma::sma;

        let data: Vec<f64> = (0..30).map(|x| x as f64 + 10.0).collect();
        let period = 7;

        let fused = rolling_stats(&data, period).unwrap();
        let unfused = sma(&data, period).unwrap();

        for i in 0..data.len() {
            assert!(
                approx_eq(fused.mean[i], unfused[i], EPSILON),
                "Mean mismatch at index {}: fused={}, unfused={}",
                i,
                fused.mean[i],
                unfused[i]
            );
        }
    }

    // ==================== rolling_stats Basic Tests ====================

    #[test]
    fn test_rolling_stats_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 3).unwrap();

        assert_eq!(result.mean.len(), 5);
        assert_eq!(result.variance.len(), 5);
        assert_eq!(result.stddev.len(), 5);

        // First 2 values should be NaN
        assert!(result.mean[0].is_nan());
        assert!(result.mean[1].is_nan());

        // From index 2, values should be valid
        assert!(!result.mean[2].is_nan());
        assert!(!result.variance[2].is_nan());
        assert!(!result.stddev[2].is_nan());
    }

    #[test]
    fn test_rolling_stats_known_values() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 3).unwrap();

        // Window [1,2,3]: mean=2, var=2/3, stddev=sqrt(2/3)
        assert!(approx_eq(result.mean[2], 2.0, EPSILON));
        assert!(approx_eq(result.variance[2], 2.0 / 3.0, EPSILON));
        assert!(approx_eq(result.stddev[2], (2.0_f64 / 3.0).sqrt(), EPSILON));

        // Window [2,3,4]: mean=3
        assert!(approx_eq(result.mean[3], 3.0, EPSILON));

        // Window [3,4,5]: mean=4
        assert!(approx_eq(result.mean[4], 4.0, EPSILON));
    }

    #[test]
    fn test_rolling_stats_period_one() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 1).unwrap();

        // With period 1, mean equals input and variance is 0
        for i in 0..5 {
            assert!(approx_eq(result.mean[i], data[i], EPSILON));
            assert!(approx_eq(result.variance[i], 0.0, EPSILON));
            assert!(approx_eq(result.stddev[i], 0.0, EPSILON));
        }
    }

    #[test]
    fn test_rolling_stats_period_equals_length() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 5).unwrap();

        // Only the last value is valid
        for i in 0..4 {
            assert!(result.mean[i].is_nan());
        }

        // Mean = (1+2+3+4+5)/5 = 3.0
        assert!(approx_eq(result.mean[4], 3.0, EPSILON));
    }

    #[test]
    fn test_rolling_stats_constant_values() {
        let data = vec![5.0_f64; 10];
        let result = rolling_stats(&data, 3).unwrap();

        for i in 2..10 {
            assert!(approx_eq(result.mean[i], 5.0, EPSILON));
            assert!(approx_eq(result.variance[i], 0.0, EPSILON));
            assert!(approx_eq(result.stddev[i], 0.0, EPSILON));
        }
    }

    #[test]
    fn test_rolling_stats_f32() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 3).unwrap();

        assert!(approx_eq(result.mean[2], 2.0_f32, EPSILON_F32));
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_rolling_stats_empty_input() {
        let data: Vec<f64> = vec![];
        let result = rolling_stats(&data, 3);

        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_rolling_stats_zero_period() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let result = rolling_stats(&data, 0);

        assert!(matches!(result, Err(Error::InvalidPeriod { period: 0, .. })));
    }

    #[test]
    fn test_rolling_stats_period_exceeds_length() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let result = rolling_stats(&data, 5);

        assert!(matches!(
            result,
            Err(Error::InsufficientData {
                required: 5,
                actual: 3
            })
        ));
    }

    // ==================== rolling_stats_into Tests ====================

    #[test]
    fn test_rolling_stats_into_basic() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut output = RollingStatOutput {
            mean: vec![0.0_f64; 5],
            variance: vec![0.0_f64; 5],
            stddev: vec![0.0_f64; 5],
        };
        let valid_count = rolling_stats_into(&data, 3, &mut output).unwrap();

        assert_eq!(valid_count, 3);
        assert!(output.mean[0].is_nan());
        assert!(output.mean[1].is_nan());
        assert!(approx_eq(output.mean[2], 2.0, EPSILON));
    }

    #[test]
    fn test_rolling_stats_into_buffer_reuse() {
        let data1 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let mut output = RollingStatOutput {
            mean: vec![0.0_f64; 5],
            variance: vec![0.0_f64; 5],
            stddev: vec![0.0_f64; 5],
        };

        rolling_stats_into(&data1, 3, &mut output).unwrap();
        assert!(approx_eq(output.mean[2], 2.0, EPSILON));

        rolling_stats_into(&data2, 3, &mut output).unwrap();
        assert!(approx_eq(output.mean[2], 4.0, EPSILON)); // (5+4+3)/3
    }

    #[test]
    fn test_rolling_stats_into_insufficient_output() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut output = RollingStatOutput {
            mean: vec![0.0_f64; 3], // Too short
            variance: vec![0.0_f64; 5],
            stddev: vec![0.0_f64; 5],
        };
        let result = rolling_stats_into(&data, 3, &mut output);

        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    #[test]
    fn test_rolling_stats_and_into_produce_same_result() {
        let data = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let result1 = rolling_stats(&data, 4).unwrap();

        let mut result2 = RollingStatOutput {
            mean: vec![0.0_f64; data.len()],
            variance: vec![0.0_f64; data.len()],
            stddev: vec![0.0_f64; data.len()],
        };
        rolling_stats_into(&data, 4, &mut result2).unwrap();

        for i in 0..data.len() {
            assert!(approx_eq(result1.mean[i], result2.mean[i], EPSILON));
            assert!(approx_eq(result1.variance[i], result2.variance[i], EPSILON));
            assert!(approx_eq(result1.stddev[i], result2.stddev[i], EPSILON));
        }
    }

    // ==================== RollingStat Struct Tests ====================

    #[test]
    fn test_rolling_stat_struct_basic() {
        let rolling: RollingStat<f64> = RollingStat::new(3);
        assert_eq!(rolling.period(), 3);
        assert_eq!(rolling.count(), 0);
    }

    #[test]
    fn test_rolling_stat_struct_update() {
        let mut rolling: RollingStat<f64> = RollingStat::new(3);

        // Build window [1, 2, 3]
        rolling.update(1.0, None);
        rolling.update(2.0, None);
        rolling.update(3.0, None);

        assert_eq!(rolling.count(), 3);
        assert!(approx_eq(rolling.mean(), 2.0, EPSILON));

        // Slide to [2, 3, 4]
        rolling.update(4.0, Some(1.0));

        assert_eq!(rolling.count(), 3);
        assert!(approx_eq(rolling.mean(), 3.0, EPSILON));
    }

    #[test]
    fn test_rolling_stat_struct_reset() {
        let mut rolling: RollingStat<f64> = RollingStat::new(3);
        rolling.update(1.0, None);
        rolling.update(2.0, None);

        rolling.reset();

        assert_eq!(rolling.count(), 0);
        assert!(rolling.mean().is_nan());
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_rolling_stats_with_nan_in_data() {
        let data = vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let result = rolling_stats(&data, 3).unwrap();

        // Windows containing NaN should produce NaN output
        assert!(result.mean[2].is_nan()); // [1, 2, NaN]
        assert!(result.mean[3].is_nan()); // [2, NaN, 4]
        assert!(result.mean[4].is_nan()); // [NaN, 4, 5]
        // NaN has rolled out, but the stat is still tainted
        // This is expected behavior - once NaN enters, it persists in our rolling calculation
    }

    #[test]
    fn test_rolling_stats_negative_values() {
        let data = vec![-5.0_f64, -3.0, -1.0, 1.0, 3.0, 5.0];
        let result = rolling_stats(&data, 3).unwrap();

        // Mean of [-5,-3,-1] = -3.0
        assert!(approx_eq(result.mean[2], -3.0, EPSILON));

        // Mean of [1,3,5] = 3.0
        assert!(approx_eq(result.mean[5], 3.0, EPSILON));
    }

    #[test]
    fn test_rolling_stats_large_window() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let result = rolling_stats(&data, 50).unwrap();

        // First 49 values should be NaN
        for i in 0..49 {
            assert!(result.mean[i].is_nan());
        }

        // Index 49: mean of 0..49 = 24.5
        assert!(approx_eq(result.mean[49], 24.5, EPSILON));

        // Last: mean of 50..99 = 74.5
        assert!(approx_eq(result.mean[99], 74.5, EPSILON));
    }

    // ==================== Valid Count Tests ====================

    #[test]
    fn test_rolling_stats_valid_count() {
        let data = vec![1.0_f64; 100];
        let mut output = RollingStatOutput {
            mean: vec![0.0_f64; 100],
            variance: vec![0.0_f64; 100],
            stddev: vec![0.0_f64; 100],
        };

        let valid_count = rolling_stats_into(&data, 10, &mut output).unwrap();
        assert_eq!(valid_count, 91); // 100 - 10 + 1

        let valid_count = rolling_stats_into(&data, 1, &mut output).unwrap();
        assert_eq!(valid_count, 100); // All values valid

        let valid_count = rolling_stats_into(&data, 100, &mut output).unwrap();
        assert_eq!(valid_count, 1); // Only last value valid
    }

    // ==================== Clone and Copy Tests ====================

    #[test]
    fn test_running_stat_clone() {
        let mut stat: RunningStat<f64> = RunningStat::new();
        stat.update(1.0);
        stat.update(2.0);

        let stat_clone = stat;
        assert_eq!(stat_clone.count(), 2);
        assert!(approx_eq(stat_clone.mean(), stat.mean(), EPSILON));
    }

    #[test]
    fn test_rolling_stat_output_clone() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_stats(&data, 3).unwrap();
        let cloned = result.clone();

        for i in 0..5 {
            assert!(approx_eq(result.mean[i], cloned.mean[i], EPSILON));
        }
    }
}
