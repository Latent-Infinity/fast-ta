//! Commonly used types and traits for convenient importing.
//!
//! This prelude provides the most frequently used types and traits from `fast-ta`
//! to simplify imports in typical usage scenarios.
//!
//! # Usage
//!
//! ```
//! use fast_ta::prelude::*;
//! ```
//!
//! # Contents
//!
//! This prelude re-exports:
//!
//! - [`Error`]: The main error type for indicator computation failures
//! - [`Result`]: Type alias for `std::result::Result<T, Error>`
//! - [`SeriesElement`]: Trait for numeric types usable in indicators
//! - [`ValidatedInput`]: Extension trait for input validation
//! - Lookback functions: `*_lookback()` for all indicators
//! - Minimum length functions: `*_min_len()` for all indicators

pub use crate::error::{Error, Result};
pub use crate::traits::{SeriesElement, ValidatedInput};

// Lookback and min_len functions for all indicators (PRD §4.4)
pub use crate::indicators::atr::{atr_lookback, atr_min_len, true_range_lookback};
pub use crate::indicators::bollinger::{bollinger_lookback, bollinger_min_len};
pub use crate::indicators::ema::{ema_lookback, ema_min_len};
pub use crate::indicators::macd::{macd_line_lookback, macd_min_len, macd_signal_lookback};
pub use crate::indicators::rsi::{rsi_lookback, rsi_min_len};
pub use crate::indicators::sma::{sma_lookback, sma_min_len};
pub use crate::indicators::stochastic::{
    stochastic_d_lookback, stochastic_k_lookback, stochastic_min_len,
};
pub use crate::kernels::rolling_extrema::{rolling_extrema_lookback, rolling_extrema_min_len};
