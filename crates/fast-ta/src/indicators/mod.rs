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
//! - [`adx`] - Average Directional Index: measures trend strength (not direction)
//! - [`donchian`] - Donchian Channels: price channels for breakout identification
//!
//! ## Momentum Indicators
//!
//! - [`rsi`] - Relative Strength Index: measures speed and magnitude of price changes
//! - [`stochastic`] - Stochastic Oscillator: compares closing price to price range (canonical API)
//! - [`stochastic_fast`], [`stochastic_slow`], [`stochastic_full`] - Convenience variants
//! - [`williams_r`] - Williams %R: momentum oscillator comparing close to high-low range
//!
//! ## Volatility Indicators
//!
//! - [`atr`] - Average True Range: measures market volatility using price ranges
//! - [`true_range`] - True Range: single-period volatility component
//! - [`bollinger`] - Bollinger Bands: price envelope based on standard deviation
//! - [`rolling_stddev`] - Rolling Standard Deviation: statistical dispersion measure
//!
//! ## Volume Indicators
//!
//! - [`obv`] - On-Balance Volume: cumulative volume flow to predict price changes
//! - [`vwap`] - Volume Weighted Average Price: average price weighted by volume
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

pub mod adx;
pub mod atr;
pub mod bollinger;
pub mod donchian;
pub mod ema;
pub mod macd;
pub mod obv;
pub mod rsi;
pub mod sma;
pub mod stochastic;
pub mod vwap;
pub mod williams_r;

// Re-export indicator functions for convenient access.
//
// These re-exports allow users to import directly from `indicators` without
// needing to specify the submodule, e.g., `use fast_ta::indicators::sma;`

// ADX (Average Directional Index)
pub use adx::{adx, adx_into, adx_lookback, adx_min_len, di_lookback, AdxOutput};

// ATR and True Range
pub use atr::{
    atr, atr_into, atr_lookback, atr_min_len, true_range, true_range_into, true_range_lookback,
};

// Bollinger Bands
pub use bollinger::{
    bollinger, bollinger_into, bollinger_lookback, bollinger_min_len, rolling_stddev,
    rolling_stddev_into, Bollinger, BollingerOutput,
};

// Donchian Channels
pub use donchian::{donchian, donchian_into, donchian_lookback, donchian_min_len, DonchianOutput};

// Exponential Moving Average
pub use ema::{
    ema, ema_into, ema_lookback, ema_min_len, ema_wilder, ema_wilder_into, ema_with_alpha,
    ema_with_alpha_into,
};

// MACD
pub use macd::{
    macd, macd_into, macd_line_lookback, macd_min_len, macd_signal_lookback, Macd, MacdOutput,
};

// OBV (On-Balance Volume)
pub use obv::{obv, obv_into, obv_lookback, obv_min_len};

// RSI
pub use rsi::{rsi, rsi_into, rsi_lookback, rsi_min_len};

// Simple Moving Average
pub use sma::{sma, sma_into, sma_lookback, sma_min_len};

// Stochastic Oscillator
pub use stochastic::{
    stochastic, stochastic_d_lookback, stochastic_fast, stochastic_fast_into, stochastic_full,
    stochastic_full_into, stochastic_into, stochastic_k_lookback, stochastic_min_len,
    stochastic_slow, stochastic_slow_into, Stochastic, StochasticOutput,
};

// VWAP (Volume Weighted Average Price)
pub use vwap::{vwap, vwap_into, vwap_lookback, vwap_min_len};

// Williams %R
pub use williams_r::{williams_r, williams_r_into, williams_r_lookback, williams_r_min_len};
