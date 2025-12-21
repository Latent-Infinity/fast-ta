//! Direct mode multi-indicator computation.
//!
//! This module provides functionality for computing multiple indicators independently
//! without using the plan-based DAG execution. Each indicator is computed separately,
//! making this the baseline for comparison with fused/plan-based execution modes.
//!
//! # Overview
//!
//! Direct mode is the simplest approach to computing multiple indicators:
//! - Each indicator is computed independently
//! - No dependency resolution or optimization
//! - No kernel fusion or shared computation
//! - Serves as baseline for E07 end-to-end comparison
//!
//! # Example
//!
//! ```
//! use fast_ta_core::plan::direct_mode::{DirectExecutor, IndicatorRequest};
//! use fast_ta_core::plan::IndicatorKind;
//!
//! // Create price data
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
//!
//! // Define indicators to compute
//! let requests = vec![
//!     IndicatorRequest::simple(IndicatorKind::Sma, 20),
//!     IndicatorRequest::simple(IndicatorKind::Ema, 20),
//!     IndicatorRequest::simple(IndicatorKind::Rsi, 14),
//! ];
//!
//! // Compute all indicators independently
//! let executor = DirectExecutor::new();
//! let results = executor.execute(&prices, &requests).unwrap();
//!
//! assert_eq!(results.len(), 3);
//! ```
//!
//! # Performance Characteristics
//!
//! Direct mode has the following performance characteristics:
//! - O(n × k) total time where k is the number of indicators
//! - No plan compilation overhead
//! - No opportunity for kernel fusion
//! - Each indicator reads the input data independently
//!
//! This mode is useful for:
//! - Small number of indicators (where plan overhead exceeds benefits)
//! - Benchmarking and comparison with plan mode
//! - Simple use cases without complex dependencies

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::indicators::{
    atr, bollinger, ema, ema_wilder, macd, rsi, sma, stochastic_fast, stochastic_full,
    stochastic_slow, BollingerOutput, MacdOutput, StochasticOutput,
};
use crate::plan::spec::IndicatorKind;
use crate::traits::SeriesElement;

/// A request to compute an indicator.
///
/// Encapsulates all the parameters needed to compute a single indicator.
#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorRequest {
    /// The type of indicator to compute
    pub kind: IndicatorKind,
    /// Primary period parameter
    pub period: usize,
    /// Secondary period (for MACD slow period, Stochastic D period, etc.)
    pub secondary_period: Option<usize>,
    /// Tertiary period (for MACD signal period, Stochastic Full K smoothing)
    pub tertiary_period: Option<usize>,
    /// Multiplier (for Bollinger Bands)
    pub multiplier: Option<f64>,
    /// Optional identifier for the result
    pub id: Option<String>,
}

impl IndicatorRequest {
    /// Creates a simple indicator request with just a period.
    ///
    /// Suitable for SMA, EMA, RSI, ATR, and other single-period indicators.
    ///
    /// # Arguments
    ///
    /// * `kind` - The type of indicator
    /// * `period` - The lookback period
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::direct_mode::IndicatorRequest;
    /// use fast_ta_core::plan::IndicatorKind;
    ///
    /// let request = IndicatorRequest::simple(IndicatorKind::Sma, 20);
    /// assert_eq!(request.period, 20);
    /// ```
    #[must_use]
    pub fn simple(kind: IndicatorKind, period: usize) -> Self {
        Self {
            kind,
            period,
            secondary_period: None,
            tertiary_period: None,
            multiplier: None,
            id: None,
        }
    }

    /// Creates a Bollinger Bands request.
    ///
    /// # Arguments
    ///
    /// * `period` - The SMA period (typically 20)
    /// * `multiplier` - Standard deviation multiplier (typically 2.0)
    #[must_use]
    pub fn bollinger(period: usize, multiplier: f64) -> Self {
        Self {
            kind: IndicatorKind::BollingerBands,
            period,
            secondary_period: None,
            tertiary_period: None,
            multiplier: Some(multiplier),
            id: None,
        }
    }

    /// Creates a MACD request with standard parameters.
    ///
    /// # Arguments
    ///
    /// * `fast_period` - Fast EMA period (typically 12)
    /// * `slow_period` - Slow EMA period (typically 26)
    /// * `signal_period` - Signal line EMA period (typically 9)
    #[must_use]
    pub fn macd(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            kind: IndicatorKind::Macd,
            period: fast_period,
            secondary_period: Some(slow_period),
            tertiary_period: Some(signal_period),
            multiplier: None,
            id: None,
        }
    }

    /// Creates a Stochastic Fast request.
    ///
    /// # Arguments
    ///
    /// * `k_period` - %K lookback period
    /// * `d_period` - %D smoothing period
    #[must_use]
    pub fn stochastic_fast(k_period: usize, d_period: usize) -> Self {
        Self {
            kind: IndicatorKind::StochasticFast,
            period: k_period,
            secondary_period: Some(d_period),
            tertiary_period: None,
            multiplier: None,
            id: None,
        }
    }

    /// Creates a Stochastic Slow request.
    ///
    /// # Arguments
    ///
    /// * `k_period` - %K lookback period
    /// * `d_period` - %D smoothing period
    #[must_use]
    pub fn stochastic_slow(k_period: usize, d_period: usize) -> Self {
        Self {
            kind: IndicatorKind::StochasticSlow,
            period: k_period,
            secondary_period: Some(d_period),
            tertiary_period: None,
            multiplier: None,
            id: None,
        }
    }

    /// Creates a Stochastic Full request.
    ///
    /// # Arguments
    ///
    /// * `k_period` - %K lookback period
    /// * `k_smooth` - %K smoothing period
    /// * `d_period` - %D smoothing period
    #[must_use]
    pub fn stochastic_full(k_period: usize, k_smooth: usize, d_period: usize) -> Self {
        Self {
            kind: IndicatorKind::StochasticFull,
            period: k_period,
            secondary_period: Some(k_smooth),
            tertiary_period: Some(d_period),
            multiplier: None,
            id: None,
        }
    }

    /// Sets an identifier for this request.
    ///
    /// The identifier is used to look up results in the output map.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns a unique key for this request configuration.
    #[must_use]
    pub fn config_key(&self) -> String {
        let mut key = format!("{}_{}", self.kind.name(), self.period);
        if let Some(secondary) = self.secondary_period {
            key.push_str(&format!("_{secondary}"));
        }
        if let Some(tertiary) = self.tertiary_period {
            key.push_str(&format!("_{tertiary}"));
        }
        if let Some(mult) = self.multiplier {
            key.push_str(&format!("_{mult:.2}"));
        }
        key
    }
}

/// Result of computing an indicator.
///
/// Different indicators produce different output structures:
/// - Single output indicators produce a single `Vec<T>`
/// - MACD produces three vectors (line, signal, histogram)
/// - Bollinger Bands produces three vectors (middle, upper, lower)
/// - Stochastic produces two vectors (%K, %D)
#[derive(Debug, Clone)]
pub enum IndicatorResult<T: SeriesElement> {
    /// Single output vector (SMA, EMA, RSI, ATR, etc.)
    Single(Vec<T>),
    /// MACD output (line, signal, histogram)
    Macd(MacdOutput<T>),
    /// Bollinger Bands output (middle, upper, lower)
    Bollinger(BollingerOutput<T>),
    /// Stochastic output (%K, %D)
    Stochastic(StochasticOutput<T>),
}

impl<T: SeriesElement> IndicatorResult<T> {
    /// Returns the number of output series.
    #[must_use]
    pub fn output_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Macd(_) | Self::Bollinger(_) => 3,
            Self::Stochastic(_) => 2,
        }
    }

    /// Returns the length of the output series.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Single(v) => v.len(),
            Self::Macd(m) => m.macd_line.len(),
            Self::Bollinger(b) => b.middle.len(),
            Self::Stochastic(s) => s.k.len(),
        }
    }

    /// Returns true if the output is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extracts the single output vector.
    ///
    /// # Panics
    ///
    /// Panics if this is not a `Single` variant.
    #[must_use]
    pub fn unwrap_single(self) -> Vec<T> {
        match self {
            Self::Single(v) => v,
            _ => panic!("Expected Single variant"),
        }
    }

    /// Extracts the MACD output.
    ///
    /// # Panics
    ///
    /// Panics if this is not a `Macd` variant.
    #[must_use]
    pub fn unwrap_macd(self) -> MacdOutput<T> {
        match self {
            Self::Macd(m) => m,
            _ => panic!("Expected Macd variant"),
        }
    }

    /// Extracts the Bollinger Bands output.
    ///
    /// # Panics
    ///
    /// Panics if this is not a `Bollinger` variant.
    #[must_use]
    pub fn unwrap_bollinger(self) -> BollingerOutput<T> {
        match self {
            Self::Bollinger(b) => b,
            _ => panic!("Expected Bollinger variant"),
        }
    }

    /// Extracts the Stochastic output.
    ///
    /// # Panics
    ///
    /// Panics if this is not a `Stochastic` variant.
    #[must_use]
    pub fn unwrap_stochastic(self) -> StochasticOutput<T> {
        match self {
            Self::Stochastic(s) => s,
            _ => panic!("Expected Stochastic variant"),
        }
    }

    /// Returns the single output as a reference, if applicable.
    #[must_use]
    pub fn as_single(&self) -> Option<&Vec<T>> {
        match self {
            Self::Single(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the MACD output as a reference, if applicable.
    #[must_use]
    pub fn as_macd(&self) -> Option<&MacdOutput<T>> {
        match self {
            Self::Macd(m) => Some(m),
            _ => None,
        }
    }

    /// Returns the Bollinger output as a reference, if applicable.
    #[must_use]
    pub fn as_bollinger(&self) -> Option<&BollingerOutput<T>> {
        match self {
            Self::Bollinger(b) => Some(b),
            _ => None,
        }
    }

    /// Returns the Stochastic output as a reference, if applicable.
    #[must_use]
    pub fn as_stochastic(&self) -> Option<&StochasticOutput<T>> {
        match self {
            Self::Stochastic(s) => Some(s),
            _ => None,
        }
    }
}

/// OHLCV data for indicators that require high, low, close (and optionally open, volume).
#[derive(Debug, Clone)]
pub struct OhlcvData<'a, T: SeriesElement> {
    /// High prices
    pub high: &'a [T],
    /// Low prices
    pub low: &'a [T],
    /// Close prices
    pub close: &'a [T],
}

impl<'a, T: SeriesElement> OhlcvData<'a, T> {
    /// Creates new OHLCV data from slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    #[must_use]
    pub fn new(high: &'a [T], low: &'a [T], close: &'a [T]) -> Self {
        assert_eq!(high.len(), low.len(), "High and low must have same length");
        assert_eq!(
            high.len(),
            close.len(),
            "High and close must have same length"
        );
        Self { high, low, close }
    }

    /// Returns the length of the data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Returns true if the data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// Direct mode executor for computing indicators independently.
///
/// This executor computes each indicator separately without any optimization
/// or fusion. It serves as the baseline for comparison with plan-based execution.
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::direct_mode::{DirectExecutor, IndicatorRequest};
/// use fast_ta_core::plan::IndicatorKind;
///
/// let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
/// let executor = DirectExecutor::new();
///
/// let results = executor.execute(&prices, &[
///     IndicatorRequest::simple(IndicatorKind::Sma, 20),
///     IndicatorRequest::simple(IndicatorKind::Ema, 10),
/// ]).unwrap();
///
/// assert_eq!(results.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct DirectExecutor {
    // Placeholder for future configuration options
    _config: (),
}

impl DirectExecutor {
    /// Creates a new direct executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Executes multiple indicator requests on price data.
    ///
    /// Each indicator is computed independently in sequence.
    ///
    /// # Arguments
    ///
    /// * `data` - The price series to compute indicators on
    /// * `requests` - The indicators to compute
    ///
    /// # Returns
    ///
    /// A vector of results in the same order as the requests.
    ///
    /// # Errors
    ///
    /// Returns an error if any indicator computation fails (e.g., insufficient data).
    pub fn execute<T: SeriesElement>(
        &self,
        data: &[T],
        requests: &[IndicatorRequest],
    ) -> Result<Vec<IndicatorResult<T>>> {
        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            let result = self.compute_single(data, request)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Executes multiple indicator requests and returns results as a map.
    ///
    /// # Arguments
    ///
    /// * `data` - The price series to compute indicators on
    /// * `requests` - The indicators to compute (each should have an id set)
    ///
    /// # Returns
    ///
    /// A map from indicator ID to result. If no ID is set, uses config_key.
    ///
    /// # Errors
    ///
    /// Returns an error if any indicator computation fails.
    pub fn execute_named<T: SeriesElement>(
        &self,
        data: &[T],
        requests: &[IndicatorRequest],
    ) -> Result<HashMap<String, IndicatorResult<T>>> {
        let mut results = HashMap::with_capacity(requests.len());

        for request in requests {
            let result = self.compute_single(data, request)?;
            let key = request.id.clone().unwrap_or_else(|| request.config_key());
            results.insert(key, result);
        }

        Ok(results)
    }

    /// Executes indicator requests that require OHLCV data.
    ///
    /// This is used for ATR, Stochastic, and other indicators that need
    /// high/low/close prices.
    ///
    /// # Arguments
    ///
    /// * `ohlcv` - The OHLCV data
    /// * `requests` - The indicators to compute
    ///
    /// # Returns
    ///
    /// A vector of results in the same order as the requests.
    ///
    /// # Errors
    ///
    /// Returns an error if any indicator computation fails.
    pub fn execute_ohlcv<T: SeriesElement>(
        &self,
        ohlcv: &OhlcvData<'_, T>,
        requests: &[IndicatorRequest],
    ) -> Result<Vec<IndicatorResult<T>>> {
        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            let result = self.compute_single_ohlcv(ohlcv, request)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Computes all 7 baseline indicators at once.
    ///
    /// This is a convenience method for E07 benchmarking that computes:
    /// - SMA(20)
    /// - EMA(20)
    /// - RSI(14)
    /// - MACD(12, 26, 9)
    /// - ATR(14)
    /// - Bollinger Bands(20, 2.0)
    /// - Stochastic Fast(14, 3)
    ///
    /// # Arguments
    ///
    /// * `ohlcv` - The OHLCV data
    ///
    /// # Returns
    ///
    /// A map of indicator names to results.
    ///
    /// # Errors
    ///
    /// Returns an error if any indicator computation fails.
    pub fn execute_all_baseline<T: SeriesElement>(
        &self,
        ohlcv: &OhlcvData<'_, T>,
    ) -> Result<HashMap<String, IndicatorResult<T>>> {
        let mut results = HashMap::with_capacity(7);

        // SMA(20) on close
        results.insert(
            "sma_20".to_string(),
            IndicatorResult::Single(sma(ohlcv.close, 20)?),
        );

        // EMA(20) on close
        results.insert(
            "ema_20".to_string(),
            IndicatorResult::Single(ema(ohlcv.close, 20)?),
        );

        // RSI(14) on close
        results.insert(
            "rsi_14".to_string(),
            IndicatorResult::Single(rsi(ohlcv.close, 14)?),
        );

        // MACD(12, 26, 9) on close
        results.insert(
            "macd_12_26_9".to_string(),
            IndicatorResult::Macd(macd(ohlcv.close, 12, 26, 9)?),
        );

        // ATR(14)
        results.insert(
            "atr_14".to_string(),
            IndicatorResult::Single(atr(ohlcv.high, ohlcv.low, ohlcv.close, 14)?),
        );

        // Bollinger Bands(20, 2.0) on close
        results.insert(
            "bollinger_20_2".to_string(),
            IndicatorResult::Bollinger(bollinger(ohlcv.close, 20, T::from(2.0).unwrap())?),
        );

        // Stochastic Fast(14, 3)
        results.insert(
            "stochastic_14_3".to_string(),
            IndicatorResult::Stochastic(stochastic_fast(ohlcv.high, ohlcv.low, ohlcv.close, 14, 3)?),
        );

        Ok(results)
    }

    /// Computes a single indicator.
    fn compute_single<T: SeriesElement>(
        &self,
        data: &[T],
        request: &IndicatorRequest,
    ) -> Result<IndicatorResult<T>> {
        match request.kind {
            IndicatorKind::Sma => Ok(IndicatorResult::Single(sma(data, request.period)?)),

            IndicatorKind::Ema => Ok(IndicatorResult::Single(ema(data, request.period)?)),

            IndicatorKind::EmaWilder => {
                Ok(IndicatorResult::Single(ema_wilder(data, request.period)?))
            }

            IndicatorKind::Rsi => Ok(IndicatorResult::Single(rsi(data, request.period)?)),

            IndicatorKind::Macd => {
                let slow = request.secondary_period.unwrap_or(26);
                let signal = request.tertiary_period.unwrap_or(9);
                Ok(IndicatorResult::Macd(macd(data, request.period, slow, signal)?))
            }

            IndicatorKind::BollingerBands => {
                let mult = T::from(request.multiplier.unwrap_or(2.0))
                    .ok_or_else(|| Error::NumericConversion {
                        context: "Bollinger multiplier".to_string(),
                    })?;
                Ok(IndicatorResult::Bollinger(bollinger(data, request.period, mult)?))
            }

            IndicatorKind::Dema | IndicatorKind::Tema => {
                // These would require the ema_fusion kernel, but for direct mode
                // we compute them using the basic approach
                Err(Error::InvalidPeriod {
                    period: request.period,
                    reason: format!(
                        "{} requires DEMA/TEMA kernel which is not available in direct mode for single series",
                        request.kind.name()
                    ),
                })
            }

            // OHLCV indicators require the OHLCV execute path
            IndicatorKind::Atr
            | IndicatorKind::TrueRange
            | IndicatorKind::StochasticFast
            | IndicatorKind::StochasticSlow
            | IndicatorKind::StochasticFull => Err(Error::InvalidPeriod {
                period: request.period,
                reason: format!(
                    "{} requires OHLCV data, use execute_ohlcv instead",
                    request.kind.name()
                ),
            }),

            // Kernel operations
            IndicatorKind::RollingStdDev | IndicatorKind::RollingMax | IndicatorKind::RollingMin => {
                // These are kernel operations, not typically used as standalone indicators
                Err(Error::InvalidPeriod {
                    period: request.period,
                    reason: format!(
                        "{} is a kernel operation, not a standalone indicator",
                        request.kind.name()
                    ),
                })
            }

            IndicatorKind::Custom => Err(Error::InvalidPeriod {
                period: request.period,
                reason: "Custom indicators are not supported in direct mode".to_string(),
            }),
        }
    }

    /// Computes a single indicator that requires OHLCV data.
    fn compute_single_ohlcv<T: SeriesElement>(
        &self,
        ohlcv: &OhlcvData<'_, T>,
        request: &IndicatorRequest,
    ) -> Result<IndicatorResult<T>> {
        match request.kind {
            // Price-based indicators use close price
            IndicatorKind::Sma => Ok(IndicatorResult::Single(sma(ohlcv.close, request.period)?)),

            IndicatorKind::Ema => Ok(IndicatorResult::Single(ema(ohlcv.close, request.period)?)),

            IndicatorKind::EmaWilder => Ok(IndicatorResult::Single(ema_wilder(
                ohlcv.close,
                request.period,
            )?)),

            IndicatorKind::Rsi => Ok(IndicatorResult::Single(rsi(ohlcv.close, request.period)?)),

            IndicatorKind::Macd => {
                let slow = request.secondary_period.unwrap_or(26);
                let signal = request.tertiary_period.unwrap_or(9);
                Ok(IndicatorResult::Macd(macd(
                    ohlcv.close,
                    request.period,
                    slow,
                    signal,
                )?))
            }

            IndicatorKind::BollingerBands => {
                let mult = T::from(request.multiplier.unwrap_or(2.0))
                    .ok_or_else(|| Error::NumericConversion {
                        context: "Bollinger multiplier".to_string(),
                    })?;
                Ok(IndicatorResult::Bollinger(bollinger(
                    ohlcv.close,
                    request.period,
                    mult,
                )?))
            }

            IndicatorKind::Atr => Ok(IndicatorResult::Single(atr(
                ohlcv.high,
                ohlcv.low,
                ohlcv.close,
                request.period,
            )?)),

            IndicatorKind::TrueRange => {
                use crate::indicators::atr::true_range;
                Ok(IndicatorResult::Single(true_range(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                )?))
            }

            IndicatorKind::StochasticFast => {
                let d_period = request.secondary_period.unwrap_or(3);
                Ok(IndicatorResult::Stochastic(stochastic_fast(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    d_period,
                )?))
            }

            IndicatorKind::StochasticSlow => {
                let d_period = request.secondary_period.unwrap_or(3);
                Ok(IndicatorResult::Stochastic(stochastic_slow(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    d_period,
                )?))
            }

            IndicatorKind::StochasticFull => {
                let k_smooth = request.secondary_period.unwrap_or(3);
                let d_period = request.tertiary_period.unwrap_or(3);
                Ok(IndicatorResult::Stochastic(stochastic_full(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    k_smooth,
                    d_period,
                )?))
            }

            // Kernels and custom not supported
            IndicatorKind::Dema
            | IndicatorKind::Tema
            | IndicatorKind::RollingStdDev
            | IndicatorKind::RollingMax
            | IndicatorKind::RollingMin
            | IndicatorKind::Custom => self.compute_single(ohlcv.close, request),
        }
    }
}

/// Computes all 7 baseline indicators using direct mode.
///
/// This is a convenience function for benchmarking that computes all indicators
/// independently without any optimization.
///
/// # Arguments
///
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
///
/// # Returns
///
/// A map of indicator names to results.
///
/// # Errors
///
/// Returns an error if any indicator computation fails.
pub fn compute_all_direct<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
) -> Result<HashMap<String, IndicatorResult<T>>> {
    let ohlcv = OhlcvData::new(high, low, close);
    DirectExecutor::new().execute_all_baseline(&ohlcv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_prices(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect()
    }

    fn generate_test_ohlcv(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_indicator_request_simple() {
        let request = IndicatorRequest::simple(IndicatorKind::Sma, 20);
        assert_eq!(request.kind, IndicatorKind::Sma);
        assert_eq!(request.period, 20);
        assert!(request.secondary_period.is_none());
        assert!(request.tertiary_period.is_none());
        assert!(request.multiplier.is_none());
        assert!(request.id.is_none());
    }

    #[test]
    fn test_indicator_request_bollinger() {
        let request = IndicatorRequest::bollinger(20, 2.5);
        assert_eq!(request.kind, IndicatorKind::BollingerBands);
        assert_eq!(request.period, 20);
        assert_eq!(request.multiplier, Some(2.5));
    }

    #[test]
    fn test_indicator_request_macd() {
        let request = IndicatorRequest::macd(12, 26, 9);
        assert_eq!(request.kind, IndicatorKind::Macd);
        assert_eq!(request.period, 12);
        assert_eq!(request.secondary_period, Some(26));
        assert_eq!(request.tertiary_period, Some(9));
    }

    #[test]
    fn test_indicator_request_stochastic_fast() {
        let request = IndicatorRequest::stochastic_fast(14, 3);
        assert_eq!(request.kind, IndicatorKind::StochasticFast);
        assert_eq!(request.period, 14);
        assert_eq!(request.secondary_period, Some(3));
    }

    #[test]
    fn test_indicator_request_stochastic_slow() {
        let request = IndicatorRequest::stochastic_slow(14, 3);
        assert_eq!(request.kind, IndicatorKind::StochasticSlow);
        assert_eq!(request.period, 14);
        assert_eq!(request.secondary_period, Some(3));
    }

    #[test]
    fn test_indicator_request_stochastic_full() {
        let request = IndicatorRequest::stochastic_full(14, 3, 3);
        assert_eq!(request.kind, IndicatorKind::StochasticFull);
        assert_eq!(request.period, 14);
        assert_eq!(request.secondary_period, Some(3));
        assert_eq!(request.tertiary_period, Some(3));
    }

    #[test]
    fn test_indicator_request_with_id() {
        let request = IndicatorRequest::simple(IndicatorKind::Sma, 20).with_id("my_sma");
        assert_eq!(request.id, Some("my_sma".to_string()));
    }

    #[test]
    fn test_indicator_request_config_key() {
        assert_eq!(
            IndicatorRequest::simple(IndicatorKind::Sma, 20).config_key(),
            "SMA_20"
        );
        assert_eq!(
            IndicatorRequest::macd(12, 26, 9).config_key(),
            "MACD_12_26_9"
        );
        assert_eq!(
            IndicatorRequest::bollinger(20, 2.0).config_key(),
            "Bollinger_20_2.00"
        );
    }

    #[test]
    fn test_indicator_result_single() {
        let result: IndicatorResult<f64> = IndicatorResult::Single(vec![1.0, 2.0, 3.0]);
        assert_eq!(result.output_count(), 1);
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert!(result.as_single().is_some());
        assert!(result.as_macd().is_none());
    }

    #[test]
    fn test_indicator_result_unwrap_single() {
        let result: IndicatorResult<f64> = IndicatorResult::Single(vec![1.0, 2.0, 3.0]);
        let values = result.unwrap_single();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ohlcv_data_new() {
        let high = vec![102.0, 103.0, 104.0];
        let low = vec![98.0, 99.0, 100.0];
        let close = vec![100.0, 101.0, 102.0];

        let ohlcv = OhlcvData::new(&high, &low, &close);
        assert_eq!(ohlcv.len(), 3);
        assert!(!ohlcv.is_empty());
    }

    #[test]
    #[should_panic(expected = "High and low must have same length")]
    fn test_ohlcv_data_length_mismatch() {
        let high = vec![102.0, 103.0];
        let low = vec![98.0, 99.0, 100.0];
        let close = vec![100.0, 101.0, 102.0];

        let _ohlcv = OhlcvData::new(&high, &low, &close);
    }

    #[test]
    fn test_direct_executor_new() {
        let executor = DirectExecutor::new();
        let _debug = format!("{:?}", executor);
    }

    #[test]
    fn test_direct_executor_sma() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_direct_executor_ema() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Ema, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_direct_executor_rsi() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Rsi, 14)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_direct_executor_macd() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::macd(12, 26, 9)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 3);
        assert!(results[0].as_macd().is_some());
    }

    #[test]
    fn test_direct_executor_bollinger() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::bollinger(20, 2.0)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 3);
        assert!(results[0].as_bollinger().is_some());
    }

    #[test]
    fn test_direct_executor_multiple_indicators() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Sma, 20),
            IndicatorRequest::simple(IndicatorKind::Ema, 20),
            IndicatorRequest::simple(IndicatorKind::Rsi, 14),
            IndicatorRequest::macd(12, 26, 9),
            IndicatorRequest::bollinger(20, 2.0),
        ];

        let results = executor.execute(&prices, &requests).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_direct_executor_named() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Sma, 20).with_id("fast_sma"),
            IndicatorRequest::simple(IndicatorKind::Sma, 50).with_id("slow_sma"),
        ];

        let results = executor.execute_named(&prices, &requests).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains_key("fast_sma"));
        assert!(results.contains_key("slow_sma"));
    }

    #[test]
    fn test_direct_executor_ohlcv() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Atr, 14),
            IndicatorRequest::stochastic_fast(14, 3),
        ];

        let results = executor.execute_ohlcv(&ohlcv, &requests).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].output_count(), 1); // ATR
        assert_eq!(results[1].output_count(), 2); // Stochastic
    }

    #[test]
    fn test_direct_executor_all_baseline() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        let results = executor.execute_all_baseline(&ohlcv).unwrap();

        assert_eq!(results.len(), 7);
        assert!(results.contains_key("sma_20"));
        assert!(results.contains_key("ema_20"));
        assert!(results.contains_key("rsi_14"));
        assert!(results.contains_key("macd_12_26_9"));
        assert!(results.contains_key("atr_14"));
        assert!(results.contains_key("bollinger_20_2"));
        assert!(results.contains_key("stochastic_14_3"));
    }

    #[test]
    fn test_compute_all_direct() {
        let (high, low, close) = generate_test_ohlcv(100);

        let results = compute_all_direct(&high, &low, &close).unwrap();

        assert_eq!(results.len(), 7);
    }

    #[test]
    fn test_direct_executor_insufficient_data() {
        let prices = generate_test_prices(10);
        let executor = DirectExecutor::new();

        let result = executor.execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 50)]);

        assert!(result.is_err());
    }

    #[test]
    fn test_direct_executor_ohlcv_required() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        // ATR requires OHLCV data
        let result = executor.execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Atr, 14)]);

        assert!(result.is_err());
    }

    #[test]
    fn test_direct_executor_stochastic_slow_ohlcv() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        let results = executor
            .execute_ohlcv(&ohlcv, &[IndicatorRequest::stochastic_slow(14, 3)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].as_stochastic().is_some());
    }

    #[test]
    fn test_direct_executor_stochastic_full_ohlcv() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        let results = executor
            .execute_ohlcv(&ohlcv, &[IndicatorRequest::stochastic_full(14, 3, 3)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].as_stochastic().is_some());
    }

    #[test]
    fn test_direct_executor_true_range_ohlcv() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        let results = executor
            .execute_ohlcv(&ohlcv, &[IndicatorRequest::simple(IndicatorKind::TrueRange, 1)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }

    #[test]
    fn test_direct_executor_ema_wilder() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::EmaWilder, 14)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_direct_executor_price_indicators_from_ohlcv() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = DirectExecutor::new();

        // These should use close price from OHLCV
        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Sma, 20),
            IndicatorRequest::simple(IndicatorKind::Ema, 20),
            IndicatorRequest::simple(IndicatorKind::Rsi, 14),
            IndicatorRequest::macd(12, 26, 9),
            IndicatorRequest::bollinger(20, 2.0),
        ];

        let results = executor.execute_ohlcv(&ohlcv, &requests).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_indicator_result_macd_accessors() {
        let macd_output = MacdOutput {
            macd_line: vec![1.0, 2.0, 3.0],
            signal_line: vec![0.5, 1.0, 1.5],
            histogram: vec![0.5, 1.0, 1.5],
        };
        let result: IndicatorResult<f64> = IndicatorResult::Macd(macd_output);

        assert_eq!(result.output_count(), 3);
        assert_eq!(result.len(), 3);
        assert!(result.as_macd().is_some());

        let macd = result.unwrap_macd();
        assert_eq!(macd.macd_line.len(), 3);
    }

    #[test]
    fn test_indicator_result_bollinger_accessors() {
        let bb_output = BollingerOutput {
            middle: vec![100.0, 101.0, 102.0],
            upper: vec![102.0, 103.0, 104.0],
            lower: vec![98.0, 99.0, 100.0],
        };
        let result: IndicatorResult<f64> = IndicatorResult::Bollinger(bb_output);

        assert_eq!(result.output_count(), 3);
        assert!(result.as_bollinger().is_some());

        let bb = result.unwrap_bollinger();
        assert_eq!(bb.middle.len(), 3);
    }

    #[test]
    fn test_indicator_result_stochastic_accessors() {
        let stoch_output = StochasticOutput {
            k: vec![50.0, 60.0, 70.0],
            d: vec![55.0, 62.0, 68.0],
        };
        let result: IndicatorResult<f64> = IndicatorResult::Stochastic(stoch_output);

        assert_eq!(result.output_count(), 2);
        assert!(result.as_stochastic().is_some());

        let stoch = result.unwrap_stochastic();
        assert_eq!(stoch.k.len(), 3);
    }

    #[test]
    fn test_indicator_result_clone() {
        let result: IndicatorResult<f64> = IndicatorResult::Single(vec![1.0, 2.0, 3.0]);
        let cloned = result.clone();

        assert_eq!(result.len(), cloned.len());
    }

    #[test]
    fn test_indicator_request_clone_eq() {
        let request = IndicatorRequest::macd(12, 26, 9);
        let cloned = request.clone();

        assert_eq!(request, cloned);
    }

    #[test]
    fn test_direct_mode_f32() {
        let prices: Vec<f32> = (0..100).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let executor = DirectExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_direct_executor_empty_requests() {
        let prices = generate_test_prices(100);
        let executor = DirectExecutor::new();

        let results = executor.execute(&prices, &[]).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_indicator_result_empty() {
        let result: IndicatorResult<f64> = IndicatorResult::Single(vec![]);
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }
}
