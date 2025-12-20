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
//! - [`ema`]: Exponential Moving Average (standard and Wilder smoothing variants)
//! - [`rsi`]: Relative Strength Index (momentum oscillator using Wilder smoothing)

pub mod ema;
pub mod rsi;
pub mod sma;

// Re-export indicator functions for convenient access
pub use ema::{ema, ema_wilder, ema_with_alpha};
pub use rsi::rsi;
pub use sma::sma;
