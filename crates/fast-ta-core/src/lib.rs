//! fast-ta-core: High-performance technical analysis library
//!
//! This crate provides core functionality for computing technical analysis indicators
//! with a focus on performance through kernel fusion and efficient algorithms.
//!
//! # Features
//!
//! - Generic numeric operations supporting both `f32` and `f64`
//! - Comprehensive error handling for edge cases
//! - NaN-aware computations for handling missing data
//! - O(n) algorithms for all indicators
//!
//! # Modules
//!
//! - [`error`]: Error types for handling computation failures
//! - [`traits`]: Core traits for numeric operations on data series
//!
//! # Example
//!
//! ```
//! use fast_ta_core::traits::{SeriesElement, ValidatedInput, validate_indicator_input};
//! use fast_ta_core::error::{Error, Result};
//!
//! fn simple_sum<T: SeriesElement>(data: &[T]) -> Result<T> {
//!     data.validate_not_empty()?;
//!     Ok(data.iter().copied().fold(T::zero(), |a, b| a + b))
//! }
//!
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let sum = simple_sum(&data).unwrap();
//! assert!((sum - 15.0).abs() < 1e-10);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod traits;

// Re-export commonly used types at crate root
pub use error::{Error, Result};
pub use traits::{SeriesElement, ValidatedInput};
