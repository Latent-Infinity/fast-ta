//! Technical analysis indicators.
//!
//! This module provides implementations of common technical analysis indicators.
//! All indicators are:
//!
//! - Generic over `f32` and `f64` types via the [`SeriesElement`](crate::traits::SeriesElement) trait
//! - O(n) time complexity with efficient algorithms
//! - NaN-aware for handling missing data in lookback periods
//! - Error-handling for edge cases (insufficient data, invalid periods)
//!
//! # Indicators
//!
//! - [`sma`]: Simple Moving Average

pub mod sma;

// Re-export indicator functions for convenient access
pub use sma::sma;
