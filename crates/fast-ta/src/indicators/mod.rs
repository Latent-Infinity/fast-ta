//! Technical analysis indicators.
//!
//! This module provides implementations of common technical analysis indicators
//! used for analyzing price data and identifying trading signals.
//!
//! # Overview
//!
//! All indicators in this module share the following properties:
//!
//! - **Generic**: Work with both `f32` and `f64` types via the
//!   [`SeriesElement`](crate::traits::SeriesElement) trait
//! - **Efficient**: O(n) time complexity using optimized rolling algorithms
//! - **NaN-aware**: Handle missing data in lookback periods gracefully
//! - **Error-safe**: Return typed errors for edge cases (insufficient data, invalid periods)
//!
//! # Indicator Categories
//!
//! ## Trend Indicators
//!
//! - [`sma`] - Simple Moving Average: arithmetic mean over a rolling window
//! - [`ema`] - Exponential Moving Average: weighted average emphasizing recent data
//! - [`ema_wilder`] - Wilder's EMA: uses smoothing factor α = 1/period
//! - [`macd`] - MACD: trend-following momentum using EMA differences
//!
//! ## Momentum Indicators
//!
//! - [`rsi`] - Relative Strength Index: measures speed and magnitude of price changes
//! - [`stochastic_fast`] - Fast Stochastic: compares closing price to price range
//! - [`stochastic_slow`] - Slow Stochastic: smoothed version of fast stochastic
//! - [`stochastic_full`] - Full Stochastic: configurable smoothing for both %K and %D
//!
//! ## Volatility Indicators
//!
//! - [`atr`] - Average True Range: measures market volatility using price ranges
//! - [`true_range`] - True Range: single-period volatility component
//! - [`bollinger`] - Bollinger Bands: price envelope based on standard deviation
//! - [`rolling_stddev`] - Rolling Standard Deviation: statistical dispersion measure
//!
//! # Example
//!
//! ```
//! use fast_ta::indicators::{sma, ema, rsi};
//!
//! let prices = vec![44.0_f64, 44.5, 43.5, 44.5, 44.0, 43.0, 42.5, 43.5, 44.5, 45.0];
//!
//! // Calculate a 5-period Simple Moving Average
//! let sma_result = sma(&prices, 5).unwrap();
//!
//! // Calculate a 5-period Exponential Moving Average
//! let ema_result = ema(&prices, 5).unwrap();
//!
//! // Calculate the 5-period RSI
//! let rsi_result = rsi(&prices, 5).unwrap();
//! ```
//!
//! # NaN Handling
//!
//! All indicators return NaN values for the lookback period (typically `period - 1`
//! elements at the start). This design ensures output arrays have the same length
//! as input arrays, simplifying alignment with original data.
//!
//! # Error Handling
//!
//! Indicators return [`Result<T, Error>`](crate::error::Error) to handle:
//!
//! - Empty input data ([`EmptyInput`](crate::error::Error::EmptyInput))
//! - Invalid period values ([`InvalidPeriod`](crate::error::Error::InvalidPeriod))
//! - Insufficient data for the requested period
//!   ([`InsufficientData`](crate::error::Error::InsufficientData))

pub mod atr;
pub mod bollinger;
pub mod ema;
pub mod macd;
pub mod rsi;
pub mod sma;
pub mod stochastic;

// Re-export indicator functions for convenient access.
//
// These re-exports allow users to import directly from `indicators` without
// needing to specify the submodule, e.g., `use fast_ta::indicators::sma;`

pub use atr::{atr, atr_lookback, atr_min_len, true_range, true_range_lookback};
pub use bollinger::{bollinger, bollinger_lookback, bollinger_min_len, rolling_stddev, BollingerOutput};
pub use ema::{ema, ema_lookback, ema_min_len, ema_wilder, ema_with_alpha};
pub use macd::{macd, macd_line_lookback, macd_min_len, macd_signal_lookback, MacdOutput};
pub use rsi::{rsi, rsi_lookback, rsi_min_len};
pub use sma::{sma, sma_lookback, sma_min_len};
pub use stochastic::{
    stochastic_d_lookback, stochastic_fast, stochastic_full, stochastic_k_lookback,
    stochastic_min_len, stochastic_slow, StochasticOutput,
};
