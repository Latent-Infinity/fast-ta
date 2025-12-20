//! Core traits for fast-ta numeric operations.
//!
//! This module defines the traits used throughout the fast-ta library
//! for generic numeric operations on data series.

use num_traits::{Float, NumCast};

use crate::error::{Error, Result};

/// A trait for types that can be used as elements in a data series.
///
/// This trait provides a common interface for numeric operations on series data,
/// abstracting over `f32` and `f64` types. It extends `num_traits::Float` with
/// additional methods specific to time series operations.
///
/// # Type Bounds
///
/// The trait requires:
/// - `Float`: Standard floating-point operations (NaN handling, infinity, arithmetic)
/// - `NumCast`: Safe conversion between numeric types
/// - `Copy`: Values can be copied (required for efficient iteration)
/// - `Default`: A default value exists (typically zero)
///
/// # Example
///
/// ```
/// use fast_ta_core::traits::SeriesElement;
/// use num_traits::Float;
///
/// fn compute_sum<T: SeriesElement>(data: &[T]) -> T {
///     data.iter().fold(T::zero(), |acc, &x| {
///         if x.is_nan() { acc } else { acc + x }
///     })
/// }
///
/// let data = vec![1.0_f64, 2.0, f64::NAN, 4.0];
/// let sum = compute_sum(&data);
/// assert!((sum - 7.0).abs() < 1e-10);
/// ```
pub trait SeriesElement: Float + NumCast + Copy + Default + Send + Sync + 'static {
    /// Creates a series element from a `usize` value.
    ///
    /// This is commonly used for converting period parameters to the series element type.
    ///
    /// # Errors
    ///
    /// Returns `Error::NumericConversion` if the value cannot be represented in this type.
    fn from_usize(value: usize) -> Result<Self> {
        <Self as NumCast>::from(value).ok_or(Error::NumericConversion {
            context: "usize to series element",
        })
    }

    /// Creates a series element from an `i32` value.
    ///
    /// # Errors
    ///
    /// Returns `Error::NumericConversion` if the value cannot be represented in this type.
    fn from_i32(value: i32) -> Result<Self> {
        <Self as NumCast>::from(value).ok_or(Error::NumericConversion {
            context: "i32 to series element",
        })
    }

    /// Creates a series element from an `f64` value.
    ///
    /// # Errors
    ///
    /// Returns `Error::NumericConversion` if the value cannot be represented in this type.
    fn from_f64(value: f64) -> Result<Self> {
        <Self as NumCast>::from(value).ok_or(Error::NumericConversion {
            context: "f64 to series element",
        })
    }

    /// Returns the constant 2 as this type.
    ///
    /// This is commonly used in EMA calculations: `alpha = 2 / (period + 1)`.
    fn two() -> Self {
        // Safe unwrap: 2 is always representable in Float types
        <Self as NumCast>::from(2).unwrap()
    }
}

// Blanket implementation for all types that satisfy the bounds
impl<T: Float + NumCast + Copy + Default + Send + Sync + 'static> SeriesElement for T {}

/// Trait for validating input data before indicator computation.
///
/// This trait provides validation methods to check that input data meets
/// the requirements of an indicator before computation begins.
pub trait ValidatedInput {
    /// The element type of the series.
    type Element: SeriesElement;

    /// Returns the length of the series.
    fn len(&self) -> usize;

    /// Returns true if the series is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validates that the series has at least `min_length` elements.
    ///
    /// # Errors
    ///
    /// Returns `Error::InsufficientData` if the series is shorter than `min_length`.
    fn validate_min_length(&self, min_length: usize) -> Result<()> {
        if self.len() < min_length {
            Err(Error::InsufficientData {
                required: min_length,
                actual: self.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Validates that the series is not empty.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptyInput` if the series is empty.
    fn validate_not_empty(&self) -> Result<()> {
        if self.is_empty() {
            Err(Error::EmptyInput)
        } else {
            Ok(())
        }
    }
}

// Implementation for slices
impl<T: SeriesElement> ValidatedInput for [T] {
    type Element = T;

    fn len(&self) -> usize {
        self.len()
    }
}

// Implementation for Vec
impl<T: SeriesElement> ValidatedInput for Vec<T> {
    type Element = T;

    fn len(&self) -> usize {
        self.len()
    }
}

/// Validates that a period is valid for indicator computation.
///
/// # Errors
///
/// Returns `Error::InvalidPeriod` if the period is zero.
pub fn validate_period(period: usize) -> Result<()> {
    if period == 0 {
        Err(Error::InvalidPeriod {
            period,
            reason: "period must be at least 1",
        })
    } else {
        Ok(())
    }
}

/// Validates that input data is suitable for indicator computation.
///
/// This function performs the following checks:
/// 1. The data is not empty
/// 2. The data has at least `min_length` elements
/// 3. The period is valid (non-zero)
///
/// # Errors
///
/// Returns an appropriate error if any validation fails.
pub fn validate_indicator_input<T: SeriesElement>(
    data: &[T],
    period: usize,
) -> Result<()> {
    validate_period(period)?;
    data.validate_not_empty()?;
    data.validate_min_length(period)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_element_from_usize() {
        let val: f64 = SeriesElement::from_usize(42).unwrap();
        assert!((val - 42.0).abs() < 1e-10);

        let val_f32: f32 = SeriesElement::from_usize(100).unwrap();
        assert!((val_f32 - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_series_element_from_i32() {
        let val: f64 = SeriesElement::from_i32(-5).unwrap();
        assert!((val - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_series_element_from_f64() {
        let val: f64 = SeriesElement::from_f64(3.14159).unwrap();
        assert!((val - 3.14159).abs() < 1e-10);

        // Test conversion from f64 to f32 (may lose precision)
        let val_f32: f32 = SeriesElement::from_f64(3.14159).unwrap();
        assert!((val_f32 - 3.14159_f32).abs() < 1e-5);
    }

    #[test]
    fn test_series_element_two() {
        let two_f64: f64 = SeriesElement::two();
        assert!((two_f64 - 2.0).abs() < 1e-10);

        let two_f32: f32 = SeriesElement::two();
        assert!((two_f32 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_validated_input_len() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(ValidatedInput::len(&data), 3);

        let slice: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        assert_eq!(ValidatedInput::len(slice), 4);
    }

    #[test]
    fn test_validated_input_is_empty() {
        let empty: Vec<f64> = vec![];
        assert!(ValidatedInput::is_empty(&empty));

        let non_empty: Vec<f64> = vec![1.0];
        assert!(!ValidatedInput::is_empty(&non_empty));
    }

    #[test]
    fn test_validate_min_length_success() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(data.validate_min_length(5).is_ok());
        assert!(data.validate_min_length(3).is_ok());
    }

    #[test]
    fn test_validate_min_length_failure() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = data.validate_min_length(5);
        assert!(result.is_err());
        match result {
            Err(Error::InsufficientData { required, actual }) => {
                assert_eq!(required, 5);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_validate_not_empty_success() {
        let data: Vec<f64> = vec![1.0];
        assert!(data.validate_not_empty().is_ok());
    }

    #[test]
    fn test_validate_not_empty_failure() {
        let data: Vec<f64> = vec![];
        let result = data.validate_not_empty();
        assert!(matches!(result, Err(Error::EmptyInput)));
    }

    #[test]
    fn test_validate_period_success() {
        assert!(validate_period(1).is_ok());
        assert!(validate_period(100).is_ok());
    }

    #[test]
    fn test_validate_period_zero() {
        let result = validate_period(0);
        assert!(result.is_err());
        match result {
            Err(Error::InvalidPeriod { period, reason }) => {
                assert_eq!(period, 0);
                assert!(!reason.is_empty());
            }
            _ => panic!("Expected InvalidPeriod error"),
        }
    }

    #[test]
    fn test_validate_indicator_input_success() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(validate_indicator_input(&data, 3).is_ok());
        assert!(validate_indicator_input(&data, 5).is_ok());
    }

    #[test]
    fn test_validate_indicator_input_empty() {
        let data: Vec<f64> = vec![];
        let result = validate_indicator_input(&data, 3);
        assert!(matches!(result, Err(Error::InvalidPeriod { .. }) | Err(Error::EmptyInput)));
    }

    #[test]
    fn test_validate_indicator_input_insufficient() {
        let data: Vec<f64> = vec![1.0, 2.0];
        let result = validate_indicator_input(&data, 5);
        assert!(matches!(result, Err(Error::InsufficientData { .. })));
    }

    #[test]
    fn test_validate_indicator_input_zero_period() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = validate_indicator_input(&data, 0);
        assert!(matches!(result, Err(Error::InvalidPeriod { .. })));
    }

    #[test]
    fn test_series_element_nan_handling() {
        // Test that NaN values work correctly
        let nan: f64 = f64::NAN;
        assert!(nan.is_nan());

        let data: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_series_element_infinity_handling() {
        // Test that infinity values are representable
        let inf: f64 = f64::INFINITY;
        let neg_inf: f64 = f64::NEG_INFINITY;

        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(inf.is_sign_positive());
        assert!(neg_inf.is_sign_negative());
    }

    #[test]
    fn test_slice_validated_input() {
        let slice: &[f64] = &[1.0, 2.0, 3.0];
        assert!(slice.validate_min_length(2).is_ok());
        assert!(slice.validate_not_empty().is_ok());
    }

    #[test]
    fn test_series_element_default() {
        // Test that Default is implemented (returns zero)
        let default: f64 = f64::default();
        assert!((default - 0.0).abs() < 1e-10);

        let default_f32: f32 = f32::default();
        assert!((default_f32 - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_series_element_send_sync() {
        // Compile-time test that SeriesElement types are Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<f64>();
        assert_send_sync::<f32>();
    }
}
