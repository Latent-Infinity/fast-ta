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
//! - [`macd`]: Moving Average Convergence Divergence (trend-following momentum indicator)
//! - [`atr`]: Average True Range (volatility indicator using Wilder smoothing)
//! - [`bollinger`]: Bollinger Bands (volatility indicator with middle, upper, and lower bands)
//! - [`stochastic`]: Stochastic Oscillator (momentum oscillator using rolling extrema)

pub mod atr;
pub mod bollinger;
pub mod ema;
pub mod macd;
pub mod rsi;
pub mod sma;
pub mod stochastic;

// Re-export indicator functions for convenient access
pub use atr::{atr, true_range};
pub use bollinger::{bollinger, rolling_stddev, BollingerOutput};
pub use ema::{ema, ema_wilder, ema_with_alpha};
pub use macd::{macd, MacdOutput};
pub use rsi::rsi;
pub use sma::sma;
pub use stochastic::{stochastic_fast, stochastic_full, stochastic_slow, StochasticOutput};
