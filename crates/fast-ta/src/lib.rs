//! fast-ta: High-performance technical analysis library
//!
//! This crate provides fast and accurate implementations of common technical
//! analysis indicators used in financial trading and analysis.
//!
//! # Features
//!
//! - **Performance**: O(n) algorithms with optimized memory access patterns
//! - **Accuracy**: Validated against TA-Lib golden reference outputs
//! - **Generics**: Works with both `f32` and `f64` data types
//! - **Safety**: Comprehensive error handling for edge cases
//!
//! # Quick Start
//!
//! ```
//! use fast_ta::prelude::*;
//! use fast_ta::indicators::sma;
//!
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let result = sma(&data, 3).unwrap();
//!
//! // First 2 values are NaN (lookback period)
//! assert!(result[0].is_nan());
//! assert!(result[1].is_nan());
//! // SMA values start from index 2
//! assert!((result[2] - 2.0).abs() < 1e-10);
//! ```
//!
//! # Available Indicators
//!
//! ## Moving Averages
//! - [`indicators::sma()`]: Simple Moving Average
//! - [`indicators::ema()`]: Exponential Moving Average
//!
//! ## Momentum
//! - [`indicators::rsi()`]: Relative Strength Index
//! - [`indicators::macd()`]: Moving Average Convergence Divergence
//! - [`indicators::stochastic_fast()`]: Stochastic Oscillator (fast, slow, full)
//!
//! ## Volatility
//! - [`indicators::atr()`]: Average True Range
//! - [`indicators::bollinger()`]: Bollinger Bands
//!
//! # Error Handling
//!
//! All indicator functions return [`Result<T, Error>`] to handle edge cases:
//!
//! ```
//! use fast_ta::prelude::*;
//! use fast_ta::indicators::sma;
//!
//! // Period too long for data
//! let short_data = vec![1.0_f64, 2.0];
//! let result = sma(&short_data, 10);
//! assert!(result.is_err());
//!
//! // Empty data
//! let empty: Vec<f64> = vec![];
//! let result = sma(&empty, 5);
//! assert!(result.is_err());
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::perf)]
#![warn(clippy::nursery)]
#![warn(clippy::needless_collect)]
#![warn(clippy::or_fun_call)]
#![warn(clippy::inefficient_to_string)]
#![warn(clippy::useless_conversion)]
#![allow(clippy::module_name_repetitions)]

pub mod batch;
pub mod error;
pub mod indicators;
pub mod kernels;
pub mod prelude;
pub mod traits;
pub mod utils;

// Re-export commonly used types at crate root
pub use error::{Error, Result};
pub use indicators::sma;
pub use traits::{SeriesElement, ValidatedInput};
pub use utils::{approx_eq, approx_eq_relative, count_nan_prefix, count_nans, EPSILON, LOOSE_EPSILON};
