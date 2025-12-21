//! Plan mode multi-indicator computation using fused kernels.
//!
//! This module provides functionality for computing multiple indicators using
//! the plan-based DAG execution with fused kernel optimizations from E02-E04.
//!
//! # Overview
//!
//! Plan mode uses several optimization strategies:
//! - **Kernel Fusion**: Multiple EMAs computed in single pass (ema_fusion kernel)
//! - **Running Statistics**: Mean and stddev computed together (running_stat kernel)
//! - **Rolling Extrema**: Max and min computed together (rolling_extrema kernel)
//! - **Dependency Resolution**: DAG-based ordering to share intermediate results
//!
//! # Example
//!
//! ```
//! use fast_ta_core::plan::plan_mode::{PlanExecutor, compute_all_plan};
//! use fast_ta_core::plan::direct_mode::OhlcvData;
//!
//! // Generate test data
//! let n = 100;
//! let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect();
//! let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
//! let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
//!
//! // Compute all baseline indicators using plan mode
//! let results = compute_all_plan(&high, &low, &close).unwrap();
//!
//! assert_eq!(results.len(), 7);
//! ```
//!
//! # Performance Characteristics
//!
//! Plan mode provides performance benefits over direct mode when:
//! - Computing many EMA-based indicators (ema_multi fusion)
//! - Computing Bollinger Bands (running_stat fusion for mean+stddev)
//! - Computing Stochastic (rolling_extrema fusion for max+min)
//! - Indicators share dependencies
//!
//! The overhead of plan building is amortized over many indicator computations.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::indicators::{
    atr, bollinger, macd, rsi, stochastic_fast, stochastic_full, stochastic_slow, BollingerOutput,
    MacdOutput, StochasticOutput,
};
use crate::kernels::ema_fusion::{ema_fusion, ema_multi, macd_fusion};
use crate::kernels::rolling_extrema::rolling_extrema;
use crate::kernels::running_stat::rolling_stats;
use crate::plan::direct_mode::{IndicatorRequest, IndicatorResult, OhlcvData};
use crate::plan::spec::IndicatorKind;
use crate::traits::SeriesElement;

/// Plan mode executor for computing indicators using fused kernels.
///
/// This executor analyzes indicator requests and uses optimized fused kernels
/// where possible to reduce data access and improve cache efficiency.
///
/// # Optimizations Applied
///
/// - **EMA Fusion**: Multiple EMA periods computed in single pass
/// - **MACD Fusion**: Fast/slow EMA and signal computed together
/// - **Running Stats Fusion**: Mean and stddev for Bollinger Bands
/// - **Rolling Extrema Fusion**: Highest high and lowest low for Stochastic
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::plan_mode::PlanExecutor;
/// use fast_ta_core::plan::direct_mode::IndicatorRequest;
/// use fast_ta_core::plan::IndicatorKind;
///
/// let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
/// let executor = PlanExecutor::new();
///
/// let results = executor.execute(&prices, &[
///     IndicatorRequest::simple(IndicatorKind::Ema, 10),
///     IndicatorRequest::simple(IndicatorKind::Ema, 20),
///     IndicatorRequest::simple(IndicatorKind::Ema, 50),
/// ]).unwrap();
///
/// assert_eq!(results.len(), 3);
/// ```
#[derive(Debug, Default)]
pub struct PlanExecutor {
    // Placeholder for future configuration options
    _config: (),
}

impl PlanExecutor {
    /// Creates a new plan executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Executes multiple indicator requests using fused kernels where possible.
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
    /// Returns an error if any indicator computation fails.
    pub fn execute<T: SeriesElement>(
        &self,
        data: &[T],
        requests: &[IndicatorRequest],
    ) -> Result<Vec<IndicatorResult<T>>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Group requests by type for fusion opportunities
        // Pre-allocate with capacity - worst case all requests are one type
        let mut ema_periods: Vec<(usize, usize)> = Vec::with_capacity(requests.len()); // (request_idx, period)
        let mut other_requests: Vec<(usize, &IndicatorRequest)> = Vec::with_capacity(requests.len());

        for (idx, request) in requests.iter().enumerate() {
            match request.kind {
                IndicatorKind::Ema => {
                    ema_periods.push((idx, request.period));
                }
                _ => {
                    other_requests.push((idx, request));
                }
            }
        }

        let mut results: Vec<Option<IndicatorResult<T>>> = vec![None; requests.len()];

        // Fused EMA computation
        if !ema_periods.is_empty() {
            let mut periods: Vec<usize> = Vec::with_capacity(ema_periods.len());
            periods.extend(ema_periods.iter().map(|(_, p)| *p));
            let ema_results = ema_multi(data, &periods)?;

            for (i, (request_idx, _)) in ema_periods.iter().enumerate() {
                results[*request_idx] = Some(IndicatorResult::Single(ema_results[i].clone()));
            }
        }

        // Compute other indicators individually (with fusion where applicable)
        for (request_idx, request) in other_requests {
            let result = self.compute_single(data, request)?;
            results[request_idx] = Some(result);
        }

        // Convert to final results
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
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
        let results = self.execute(data, requests)?;
        let mut map = HashMap::with_capacity(results.len());

        for (i, result) in results.into_iter().enumerate() {
            let key = requests[i]
                .id
                .clone()
                .unwrap_or_else(|| requests[i].config_key());
            map.insert(key, result);
        }

        Ok(map)
    }

    /// Executes indicator requests that require OHLCV data.
    ///
    /// This version uses fused kernels for OHLCV-based indicators:
    /// - Rolling extrema for Stochastic highest high/lowest low
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
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Group EMA requests for fusion
        // Pre-allocate with capacity - worst case all requests are one type
        let mut ema_periods: Vec<(usize, usize)> = Vec::with_capacity(requests.len());
        let mut other_requests: Vec<(usize, &IndicatorRequest)> = Vec::with_capacity(requests.len());

        for (idx, request) in requests.iter().enumerate() {
            match request.kind {
                IndicatorKind::Ema => {
                    ema_periods.push((idx, request.period));
                }
                _ => {
                    other_requests.push((idx, request));
                }
            }
        }

        let mut results: Vec<Option<IndicatorResult<T>>> = vec![None; requests.len()];

        // Fused EMA computation on close prices
        if !ema_periods.is_empty() {
            let mut periods: Vec<usize> = Vec::with_capacity(ema_periods.len());
            periods.extend(ema_periods.iter().map(|(_, p)| *p));
            let ema_results = ema_multi(ohlcv.close, &periods)?;

            for (i, (request_idx, _)) in ema_periods.iter().enumerate() {
                results[*request_idx] = Some(IndicatorResult::Single(ema_results[i].clone()));
            }
        }

        // Compute other indicators
        for (request_idx, request) in other_requests {
            let result = self.compute_single_ohlcv(ohlcv, request)?;
            results[request_idx] = Some(result);
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Computes all 7 baseline indicators using fused kernels.
    ///
    /// This method demonstrates the full power of plan mode:
    /// - Fused MACD (fast/slow EMA + signal)
    /// - Fused Bollinger (mean + stddev via running_stats)
    /// - Fused Stochastic (highest high + lowest low via rolling_extrema)
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

        // Use fused EMA multi for SMA and EMA together
        // Note: SMA is not EMA, but we compute them separately here
        let sma_result = crate::indicators::sma::sma(ohlcv.close, 20)?;
        results.insert("sma_20".to_string(), IndicatorResult::Single(sma_result));

        // EMA(20)
        let ema_result = crate::indicators::ema::ema(ohlcv.close, 20)?;
        results.insert("ema_20".to_string(), IndicatorResult::Single(ema_result));

        // RSI(14) - uses Wilder smoothing internally
        let rsi_result = rsi(ohlcv.close, 14)?;
        results.insert("rsi_14".to_string(), IndicatorResult::Single(rsi_result));

        // MACD(12, 26, 9) - using fused kernel
        let macd_fused = macd_fusion(ohlcv.close, 12, 26, 9)?;
        let macd_output = MacdOutput {
            macd_line: macd_fused.macd_line,
            signal_line: macd_fused.signal_line,
            histogram: macd_fused.histogram,
        };
        results.insert("macd_12_26_9".to_string(), IndicatorResult::Macd(macd_output));

        // ATR(14) - uses true range and Wilder smoothing
        let atr_result = atr(ohlcv.high, ohlcv.low, ohlcv.close, 14)?;
        results.insert("atr_14".to_string(), IndicatorResult::Single(atr_result));

        // Bollinger Bands(20, 2.0) - using running_stats fusion
        let stats = rolling_stats(ohlcv.close, 20)?;
        let two = T::from(2.0).unwrap();
        let n = ohlcv.close.len();
        let mut upper = vec![T::nan(); n];
        let mut lower = vec![T::nan(); n];

        for i in 19..n {
            upper[i] = stats.mean[i] + two * stats.stddev[i];
            lower[i] = stats.mean[i] - two * stats.stddev[i];
        }

        let bb_output = BollingerOutput {
            middle: stats.mean,
            upper,
            lower,
        };
        results.insert(
            "bollinger_20_2".to_string(),
            IndicatorResult::Bollinger(bb_output),
        );

        // Stochastic Fast(14, 3) - using rolling_extrema fusion
        let extrema = rolling_extrema(ohlcv.high, 14)?;
        let low_extrema = crate::kernels::rolling_extrema::rolling_min(ohlcv.low, 14)?;

        // Compute %K: (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        let mut k = vec![T::nan(); n];
        let hundred = T::from(100.0).unwrap();

        for i in 13..n {
            let highest = extrema.max[i];
            let lowest = low_extrema[i];
            let range = highest - lowest;

            if range > T::zero() {
                k[i] = (ohlcv.close[i] - lowest) / range * hundred;
            } else {
                k[i] = T::from(50.0).unwrap(); // No movement
            }
        }

        // Compute %D: SMA of %K with period 3
        let d = crate::indicators::sma::sma(&k[13..].to_vec(), 3)?;
        let mut d_full = vec![T::nan(); n];
        for i in 0..d.len() {
            d_full[i + 13] = d[i];
        }

        let stoch_output = StochasticOutput { k, d: d_full };
        results.insert(
            "stochastic_14_3".to_string(),
            IndicatorResult::Stochastic(stoch_output),
        );

        Ok(results)
    }

    /// Computes a single indicator (with fusion where applicable).
    fn compute_single<T: SeriesElement>(
        &self,
        data: &[T],
        request: &IndicatorRequest,
    ) -> Result<IndicatorResult<T>> {
        match request.kind {
            IndicatorKind::Sma => {
                let result = crate::indicators::sma::sma(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Ema => {
                // Single EMA - use basic ema
                let result = crate::indicators::ema::ema(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::EmaWilder => {
                let result = crate::indicators::ema::ema_wilder(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Rsi => {
                let result = rsi(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Macd => {
                // Use fused MACD kernel
                let slow = request.secondary_period.unwrap_or(26);
                let signal = request.tertiary_period.unwrap_or(9);
                let fused = macd_fusion(data, request.period, slow, signal)?;
                let output = MacdOutput {
                    macd_line: fused.macd_line,
                    signal_line: fused.signal_line,
                    histogram: fused.histogram,
                };
                Ok(IndicatorResult::Macd(output))
            }

            IndicatorKind::BollingerBands => {
                // Use running_stats fusion for mean + stddev
                let mult = T::from(request.multiplier.unwrap_or(2.0))
                    .ok_or_else(|| Error::NumericConversion {
                        context: "Bollinger multiplier",
                    })?;

                let stats = rolling_stats(data, request.period)?;
                let n = data.len();
                let mut upper = vec![T::nan(); n];
                let mut lower = vec![T::nan(); n];

                let first_valid = request.period - 1;
                for i in first_valid..n {
                    upper[i] = stats.mean[i] + mult * stats.stddev[i];
                    lower[i] = stats.mean[i] - mult * stats.stddev[i];
                }

                Ok(IndicatorResult::Bollinger(BollingerOutput {
                    middle: stats.mean,
                    upper,
                    lower,
                }))
            }

            IndicatorKind::Dema => {
                // Use ema_fusion for DEMA
                let fused = ema_fusion(data, request.period)?;
                Ok(IndicatorResult::Single(fused.dema))
            }

            IndicatorKind::Tema => {
                // Use ema_fusion for TEMA
                let fused = ema_fusion(data, request.period)?;
                Ok(IndicatorResult::Single(fused.tema))
            }

            // OHLCV indicators require the OHLCV execute path
            IndicatorKind::Atr
            | IndicatorKind::TrueRange
            | IndicatorKind::StochasticFast
            | IndicatorKind::StochasticSlow
            | IndicatorKind::StochasticFull => Err(Error::InvalidPeriod {
                period: request.period,
                reason: "requires OHLCV data, use execute_ohlcv instead",
            }),

            // Kernel operations
            IndicatorKind::RollingStdDev => {
                let stats = rolling_stats(data, request.period)?;
                Ok(IndicatorResult::Single(stats.stddev))
            }

            IndicatorKind::RollingMax => {
                let result = crate::kernels::rolling_extrema::rolling_max(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::RollingMin => {
                let result = crate::kernels::rolling_extrema::rolling_min(data, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Custom => Err(Error::InvalidPeriod {
                period: request.period,
                reason: "Custom indicators are not supported in plan mode",
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
            IndicatorKind::Sma => {
                let result = crate::indicators::sma::sma(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Ema => {
                let result = crate::indicators::ema::ema(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::EmaWilder => {
                let result = crate::indicators::ema::ema_wilder(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Rsi => {
                let result = rsi(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::Macd => {
                let slow = request.secondary_period.unwrap_or(26);
                let signal = request.tertiary_period.unwrap_or(9);
                let fused = macd_fusion(ohlcv.close, request.period, slow, signal)?;
                let output = MacdOutput {
                    macd_line: fused.macd_line,
                    signal_line: fused.signal_line,
                    histogram: fused.histogram,
                };
                Ok(IndicatorResult::Macd(output))
            }

            IndicatorKind::BollingerBands => {
                let mult = T::from(request.multiplier.unwrap_or(2.0))
                    .ok_or_else(|| Error::NumericConversion {
                        context: "Bollinger multiplier",
                    })?;

                let stats = rolling_stats(ohlcv.close, request.period)?;
                let n = ohlcv.close.len();
                let mut upper = vec![T::nan(); n];
                let mut lower = vec![T::nan(); n];

                let first_valid = request.period - 1;
                for i in first_valid..n {
                    upper[i] = stats.mean[i] + mult * stats.stddev[i];
                    lower[i] = stats.mean[i] - mult * stats.stddev[i];
                }

                Ok(IndicatorResult::Bollinger(BollingerOutput {
                    middle: stats.mean,
                    upper,
                    lower,
                }))
            }

            IndicatorKind::Atr => {
                let result = atr(ohlcv.high, ohlcv.low, ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::TrueRange => {
                let result =
                    crate::indicators::atr::true_range(ohlcv.high, ohlcv.low, ohlcv.close)?;
                Ok(IndicatorResult::Single(result))
            }

            IndicatorKind::StochasticFast => {
                let d_period = request.secondary_period.unwrap_or(3);
                let result = stochastic_fast(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    d_period,
                )?;
                Ok(IndicatorResult::Stochastic(result))
            }

            IndicatorKind::StochasticSlow => {
                let d_period = request.secondary_period.unwrap_or(3);
                let result = stochastic_slow(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    d_period,
                )?;
                Ok(IndicatorResult::Stochastic(result))
            }

            IndicatorKind::StochasticFull => {
                let k_smooth = request.secondary_period.unwrap_or(3);
                let d_period = request.tertiary_period.unwrap_or(3);
                let result = stochastic_full(
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    request.period,
                    k_smooth,
                    d_period,
                )?;
                Ok(IndicatorResult::Stochastic(result))
            }

            IndicatorKind::Dema => {
                let fused = ema_fusion(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(fused.dema))
            }

            IndicatorKind::Tema => {
                let fused = ema_fusion(ohlcv.close, request.period)?;
                Ok(IndicatorResult::Single(fused.tema))
            }

            // Other indicators use close price
            _ => self.compute_single(ohlcv.close, request),
        }
    }
}

/// Computes all 7 baseline indicators using plan mode with fused kernels.
///
/// This is a convenience function for benchmarking that computes all indicators
/// using the optimized fused kernel approach.
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
pub fn compute_all_plan<T: SeriesElement>(
    high: &[T],
    low: &[T],
    close: &[T],
) -> Result<HashMap<String, IndicatorResult<T>>> {
    let ohlcv = OhlcvData::new(high, low, close);
    PlanExecutor::new().execute_all_baseline(&ohlcv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

    fn generate_test_prices(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect()
    }

    fn generate_test_ohlcv(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    fn approx_eq<T: Float>(a: T, b: T, epsilon: T) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_nan() || b.is_nan() {
            return false;
        }
        (a - b).abs() < epsilon
    }

    const EPSILON: f64 = 1e-10;
    // Looser epsilon for fused calculations that may have tiny floating-point differences
    const FUSION_EPSILON: f64 = 1e-8;

    // ==================== Basic Execution Tests ====================

    #[test]
    fn test_plan_executor_new() {
        let executor = PlanExecutor::new();
        let _debug = format!("{:?}", executor);
    }

    #[test]
    fn test_plan_executor_empty_requests() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor.execute(&prices, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_plan_executor_single_sma() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_plan_executor_single_ema() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Ema, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_plan_executor_multiple_emas_fused() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Ema, 10),
            IndicatorRequest::simple(IndicatorKind::Ema, 20),
            IndicatorRequest::simple(IndicatorKind::Ema, 50),
        ];

        let results = executor.execute(&prices, &requests).unwrap();

        assert_eq!(results.len(), 3);
        for result in &results {
            assert_eq!(result.len(), 100);
        }
    }

    #[test]
    fn test_plan_executor_rsi() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Rsi, 14)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    #[test]
    fn test_plan_executor_macd_fused() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::macd(12, 26, 9)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 3);
        assert!(results[0].as_macd().is_some());
    }

    #[test]
    fn test_plan_executor_bollinger_fused() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::bollinger(20, 2.0)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 3);
        assert!(results[0].as_bollinger().is_some());
    }

    #[test]
    fn test_plan_executor_dema() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Dema, 10)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }

    #[test]
    fn test_plan_executor_tema() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Tema, 10)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }

    // ==================== OHLCV Tests ====================

    #[test]
    fn test_plan_executor_ohlcv_basic() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = PlanExecutor::new();

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
    fn test_plan_executor_all_baseline() {
        let (high, low, close) = generate_test_ohlcv(100);
        let ohlcv = OhlcvData::new(&high, &low, &close);
        let executor = PlanExecutor::new();

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
    fn test_compute_all_plan() {
        let (high, low, close) = generate_test_ohlcv(100);

        let results = compute_all_plan(&high, &low, &close).unwrap();

        assert_eq!(results.len(), 7);
    }

    // ==================== Plan Mode vs Direct Mode Comparison ====================

    #[test]
    fn test_plan_mode_matches_direct_mode_sma() {
        let prices = generate_test_prices(100);
        let plan_executor = PlanExecutor::new();
        let direct_executor = crate::plan::direct_mode::DirectExecutor::new();

        let request = IndicatorRequest::simple(IndicatorKind::Sma, 20);

        let plan_results = plan_executor.execute(&prices, &[request.clone()]).unwrap();
        let direct_results = direct_executor.execute(&prices, &[request]).unwrap();

        let plan_sma = plan_results[0].as_single().unwrap();
        let direct_sma = direct_results[0].as_single().unwrap();

        for i in 0..plan_sma.len() {
            assert!(
                approx_eq(plan_sma[i], direct_sma[i], EPSILON),
                "SMA mismatch at index {}: plan={}, direct={}",
                i,
                plan_sma[i],
                direct_sma[i]
            );
        }
    }

    #[test]
    fn test_plan_mode_matches_direct_mode_ema() {
        let prices = generate_test_prices(100);
        let plan_executor = PlanExecutor::new();
        let direct_executor = crate::plan::direct_mode::DirectExecutor::new();

        let request = IndicatorRequest::simple(IndicatorKind::Ema, 20);

        let plan_results = plan_executor.execute(&prices, &[request.clone()]).unwrap();
        let direct_results = direct_executor.execute(&prices, &[request]).unwrap();

        let plan_ema = plan_results[0].as_single().unwrap();
        let direct_ema = direct_results[0].as_single().unwrap();

        for i in 0..plan_ema.len() {
            assert!(
                approx_eq(plan_ema[i], direct_ema[i], EPSILON),
                "EMA mismatch at index {}: plan={}, direct={}",
                i,
                plan_ema[i],
                direct_ema[i]
            );
        }
    }

    #[test]
    fn test_plan_mode_matches_direct_mode_rsi() {
        let prices = generate_test_prices(100);
        let plan_executor = PlanExecutor::new();
        let direct_executor = crate::plan::direct_mode::DirectExecutor::new();

        let request = IndicatorRequest::simple(IndicatorKind::Rsi, 14);

        let plan_results = plan_executor.execute(&prices, &[request.clone()]).unwrap();
        let direct_results = direct_executor.execute(&prices, &[request]).unwrap();

        let plan_rsi = plan_results[0].as_single().unwrap();
        let direct_rsi = direct_results[0].as_single().unwrap();

        for i in 0..plan_rsi.len() {
            assert!(
                approx_eq(plan_rsi[i], direct_rsi[i], EPSILON),
                "RSI mismatch at index {}: plan={}, direct={}",
                i,
                plan_rsi[i],
                direct_rsi[i]
            );
        }
    }

    #[test]
    fn test_plan_mode_matches_direct_mode_macd() {
        let prices = generate_test_prices(100);
        let plan_executor = PlanExecutor::new();
        let direct_executor = crate::plan::direct_mode::DirectExecutor::new();

        let request = IndicatorRequest::macd(12, 26, 9);

        let plan_results = plan_executor.execute(&prices, &[request.clone()]).unwrap();
        let direct_results = direct_executor.execute(&prices, &[request]).unwrap();

        let plan_macd = plan_results[0].as_macd().unwrap();
        let direct_macd = direct_results[0].as_macd().unwrap();

        for i in 0..plan_macd.macd_line.len() {
            assert!(
                approx_eq(plan_macd.macd_line[i], direct_macd.macd_line[i], FUSION_EPSILON),
                "MACD line mismatch at index {}: plan={}, direct={}",
                i,
                plan_macd.macd_line[i],
                direct_macd.macd_line[i]
            );
            assert!(
                approx_eq(plan_macd.signal_line[i], direct_macd.signal_line[i], FUSION_EPSILON),
                "MACD signal mismatch at index {}: plan={}, direct={}",
                i,
                plan_macd.signal_line[i],
                direct_macd.signal_line[i]
            );
            assert!(
                approx_eq(plan_macd.histogram[i], direct_macd.histogram[i], FUSION_EPSILON),
                "MACD histogram mismatch at index {}: plan={}, direct={}",
                i,
                plan_macd.histogram[i],
                direct_macd.histogram[i]
            );
        }
    }

    #[test]
    fn test_plan_mode_matches_direct_mode_bollinger() {
        let prices = generate_test_prices(100);
        let plan_executor = PlanExecutor::new();
        let direct_executor = crate::plan::direct_mode::DirectExecutor::new();

        let request = IndicatorRequest::bollinger(20, 2.0);

        let plan_results = plan_executor.execute(&prices, &[request.clone()]).unwrap();
        let direct_results = direct_executor.execute(&prices, &[request]).unwrap();

        let plan_bb = plan_results[0].as_bollinger().unwrap();
        let direct_bb = direct_results[0].as_bollinger().unwrap();

        for i in 0..plan_bb.middle.len() {
            assert!(
                approx_eq(plan_bb.middle[i], direct_bb.middle[i], FUSION_EPSILON),
                "Bollinger middle mismatch at index {}: plan={}, direct={}",
                i,
                plan_bb.middle[i],
                direct_bb.middle[i]
            );
            assert!(
                approx_eq(plan_bb.upper[i], direct_bb.upper[i], FUSION_EPSILON),
                "Bollinger upper mismatch at index {}: plan={}, direct={}",
                i,
                plan_bb.upper[i],
                direct_bb.upper[i]
            );
            assert!(
                approx_eq(plan_bb.lower[i], direct_bb.lower[i], FUSION_EPSILON),
                "Bollinger lower mismatch at index {}: plan={}, direct={}",
                i,
                plan_bb.lower[i],
                direct_bb.lower[i]
            );
        }
    }

    #[test]
    fn test_plan_mode_matches_direct_mode_all_baseline() {
        let (high, low, close) = generate_test_ohlcv(100);

        let plan_results = compute_all_plan(&high, &low, &close).unwrap();
        let direct_results =
            crate::plan::direct_mode::compute_all_direct(&high, &low, &close).unwrap();

        // Check all 7 indicators match
        assert_eq!(plan_results.len(), direct_results.len());

        // SMA
        let plan_sma = plan_results["sma_20"].as_single().unwrap();
        let direct_sma = direct_results["sma_20"].as_single().unwrap();
        for i in 0..plan_sma.len() {
            assert!(
                approx_eq(plan_sma[i], direct_sma[i], EPSILON),
                "SMA mismatch at index {}",
                i
            );
        }

        // EMA
        let plan_ema = plan_results["ema_20"].as_single().unwrap();
        let direct_ema = direct_results["ema_20"].as_single().unwrap();
        for i in 0..plan_ema.len() {
            assert!(
                approx_eq(plan_ema[i], direct_ema[i], EPSILON),
                "EMA mismatch at index {}",
                i
            );
        }

        // RSI
        let plan_rsi = plan_results["rsi_14"].as_single().unwrap();
        let direct_rsi = direct_results["rsi_14"].as_single().unwrap();
        for i in 0..plan_rsi.len() {
            assert!(
                approx_eq(plan_rsi[i], direct_rsi[i], EPSILON),
                "RSI mismatch at index {}",
                i
            );
        }

        // MACD
        let plan_macd = plan_results["macd_12_26_9"].as_macd().unwrap();
        let direct_macd = direct_results["macd_12_26_9"].as_macd().unwrap();
        for i in 0..plan_macd.macd_line.len() {
            assert!(
                approx_eq(plan_macd.macd_line[i], direct_macd.macd_line[i], FUSION_EPSILON),
                "MACD line mismatch at index {}",
                i
            );
            assert!(
                approx_eq(plan_macd.signal_line[i], direct_macd.signal_line[i], FUSION_EPSILON),
                "MACD signal mismatch at index {}",
                i
            );
        }

        // ATR
        let plan_atr = plan_results["atr_14"].as_single().unwrap();
        let direct_atr = direct_results["atr_14"].as_single().unwrap();
        for i in 0..plan_atr.len() {
            assert!(
                approx_eq(plan_atr[i], direct_atr[i], EPSILON),
                "ATR mismatch at index {}",
                i
            );
        }

        // Bollinger Bands
        let plan_bb = plan_results["bollinger_20_2"].as_bollinger().unwrap();
        let direct_bb = direct_results["bollinger_20_2"].as_bollinger().unwrap();
        for i in 0..plan_bb.middle.len() {
            assert!(
                approx_eq(plan_bb.middle[i], direct_bb.middle[i], FUSION_EPSILON),
                "Bollinger middle mismatch at index {}",
                i
            );
            assert!(
                approx_eq(plan_bb.upper[i], direct_bb.upper[i], FUSION_EPSILON),
                "Bollinger upper mismatch at index {}",
                i
            );
            assert!(
                approx_eq(plan_bb.lower[i], direct_bb.lower[i], FUSION_EPSILON),
                "Bollinger lower mismatch at index {}",
                i
            );
        }

        // Stochastic - note: may have slightly different implementation
        let plan_stoch = plan_results["stochastic_14_3"].as_stochastic().unwrap();
        let direct_stoch = direct_results["stochastic_14_3"].as_stochastic().unwrap();
        for i in 0..plan_stoch.k.len() {
            assert!(
                approx_eq(plan_stoch.k[i], direct_stoch.k[i], FUSION_EPSILON),
                "Stochastic K mismatch at index {}: plan={}, direct={}",
                i,
                plan_stoch.k[i],
                direct_stoch.k[i]
            );
        }
    }

    // ==================== Named Results Tests ====================

    #[test]
    fn test_plan_executor_named() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Sma, 20).with_id("fast_sma"),
            IndicatorRequest::simple(IndicatorKind::Sma, 50).with_id("slow_sma"),
        ];

        let results = executor.execute_named(&prices, &requests).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains_key("fast_sma"));
        assert!(results.contains_key("slow_sma"));
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn test_plan_executor_insufficient_data() {
        let prices = generate_test_prices(10);
        let executor = PlanExecutor::new();

        let result =
            executor.execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 50)]);

        assert!(result.is_err());
    }

    #[test]
    fn test_plan_executor_ohlcv_required() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        // ATR requires OHLCV data
        let result =
            executor.execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Atr, 14)]);

        assert!(result.is_err());
    }

    // ==================== F32 Tests ====================

    #[test]
    fn test_plan_mode_f32() {
        let prices: Vec<f32> = (0..100).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let executor = PlanExecutor::new();

        let results = executor
            .execute(&prices, &[IndicatorRequest::simple(IndicatorKind::Sma, 20)])
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 100);
    }

    // ==================== Mixed Request Tests ====================

    #[test]
    fn test_plan_executor_mixed_requests() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let requests = vec![
            IndicatorRequest::simple(IndicatorKind::Sma, 20),
            IndicatorRequest::simple(IndicatorKind::Ema, 20),
            IndicatorRequest::simple(IndicatorKind::Ema, 10), // Should be fused with above
            IndicatorRequest::simple(IndicatorKind::Rsi, 14),
            IndicatorRequest::macd(12, 26, 9),
            IndicatorRequest::bollinger(20, 2.0),
        ];

        let results = executor.execute(&prices, &requests).unwrap();

        assert_eq!(results.len(), 6);
        assert_eq!(results[0].output_count(), 1); // SMA
        assert_eq!(results[1].output_count(), 1); // EMA
        assert_eq!(results[2].output_count(), 1); // EMA
        assert_eq!(results[3].output_count(), 1); // RSI
        assert_eq!(results[4].output_count(), 3); // MACD
        assert_eq!(results[5].output_count(), 3); // Bollinger
    }

    // ==================== Kernel Operations Tests ====================

    #[test]
    fn test_plan_executor_rolling_stddev() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(
                &prices,
                &[IndicatorRequest::simple(IndicatorKind::RollingStdDev, 20)],
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }

    #[test]
    fn test_plan_executor_rolling_max() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(
                &prices,
                &[IndicatorRequest::simple(IndicatorKind::RollingMax, 20)],
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }

    #[test]
    fn test_plan_executor_rolling_min() {
        let prices = generate_test_prices(100);
        let executor = PlanExecutor::new();

        let results = executor
            .execute(
                &prices,
                &[IndicatorRequest::simple(IndicatorKind::RollingMin, 20)],
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_count(), 1);
    }
}
