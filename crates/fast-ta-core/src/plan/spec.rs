//! Indicator specification types.
//!
//! This module defines the [`IndicatorSpec`] struct that describes an indicator's
//! configuration, including its type, parameters, dependencies, and outputs.
//!
//! # Overview
//!
//! An indicator specification contains all the information needed to:
//! - Identify the type of indicator (SMA, EMA, RSI, etc.)
//! - Configure the indicator's parameters (period, multipliers, etc.)
//! - Declare dependencies on other indicators
//! - Describe the output(s) produced
//!
//! # Example
//!
//! ```
//! use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind, OutputSpec};
//!
//! // Create a simple SMA specification
//! let sma_spec = IndicatorSpec::new(IndicatorKind::Sma, 20);
//! assert_eq!(sma_spec.kind(), IndicatorKind::Sma);
//! assert_eq!(sma_spec.period(), 20);
//!
//! // Create a Bollinger Bands specification with custom multiplier
//! let bb_spec = IndicatorSpec::bollinger(20, 2.0);
//! assert_eq!(bb_spec.kind(), IndicatorKind::BollingerBands);
//! assert_eq!(bb_spec.outputs().len(), 3); // middle, upper, lower
//! ```

use std::borrow::Cow;

/// Enumeration of supported indicator types.
///
/// Each variant represents a different technical analysis indicator that can
/// be computed by the plan infrastructure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndicatorKind {
    /// Simple Moving Average
    Sma,
    /// Exponential Moving Average (standard smoothing)
    Ema,
    /// Exponential Moving Average with Wilder's smoothing
    EmaWilder,
    /// Relative Strength Index
    Rsi,
    /// Moving Average Convergence Divergence
    Macd,
    /// Average True Range
    Atr,
    /// Bollinger Bands
    BollingerBands,
    /// Stochastic Oscillator (Fast)
    StochasticFast,
    /// Stochastic Oscillator (Slow)
    StochasticSlow,
    /// Stochastic Oscillator (Full)
    StochasticFull,
    /// Double Exponential Moving Average
    Dema,
    /// Triple Exponential Moving Average
    Tema,
    /// True Range (component of ATR)
    TrueRange,
    /// Rolling Standard Deviation
    RollingStdDev,
    /// Rolling Maximum
    RollingMax,
    /// Rolling Minimum
    RollingMin,
    /// Custom indicator (user-defined)
    Custom,
}

impl IndicatorKind {
    /// Returns the name of the indicator kind as a string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sma => "SMA",
            Self::Ema => "EMA",
            Self::EmaWilder => "EMA_Wilder",
            Self::Rsi => "RSI",
            Self::Macd => "MACD",
            Self::Atr => "ATR",
            Self::BollingerBands => "Bollinger",
            Self::StochasticFast => "Stochastic_Fast",
            Self::StochasticSlow => "Stochastic_Slow",
            Self::StochasticFull => "Stochastic_Full",
            Self::Dema => "DEMA",
            Self::Tema => "TEMA",
            Self::TrueRange => "TrueRange",
            Self::RollingStdDev => "RollingStdDev",
            Self::RollingMax => "RollingMax",
            Self::RollingMin => "RollingMin",
            Self::Custom => "Custom",
        }
    }

    /// Returns the number of output series produced by this indicator.
    ///
    /// Most indicators produce a single output, but some (like MACD, Bollinger Bands,
    /// and Stochastic) produce multiple outputs.
    #[must_use]
    pub fn output_count(self) -> usize {
        match self {
            Self::Sma
            | Self::Ema
            | Self::EmaWilder
            | Self::Rsi
            | Self::Atr
            | Self::Dema
            | Self::Tema
            | Self::TrueRange
            | Self::RollingStdDev
            | Self::RollingMax
            | Self::RollingMin
            | Self::Custom => 1,
            Self::StochasticFast | Self::StochasticSlow | Self::StochasticFull => 2, // %K, %D
            Self::Macd | Self::BollingerBands => 3, // line/signal/histogram or middle/upper/lower
        }
    }

    /// Returns whether this indicator requires OHLCV data.
    ///
    /// Some indicators (like ATR, Stochastic) need access to High, Low, and Close
    /// prices rather than just a single price series.
    #[must_use]
    pub fn requires_ohlcv(self) -> bool {
        matches!(
            self,
            Self::Atr
                | Self::TrueRange
                | Self::StochasticFast
                | Self::StochasticSlow
                | Self::StochasticFull
        )
    }
}

/// Description of an output produced by an indicator.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    /// Name of the output (e.g., "middle", "upper", "lower" for Bollinger Bands)
    name: Cow<'static, str>,
    /// Human-readable description of the output
    description: Cow<'static, str>,
}

impl OutputSpec {
    /// Creates a new output specification.
    ///
    /// # Arguments
    ///
    /// * `name` - Short identifier for the output
    /// * `description` - Human-readable description
    #[must_use]
    pub fn new(name: impl Into<Cow<'static, str>>, description: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }

    /// Returns the name of the output.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the description of the output.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Specification for an indicator instance.
///
/// An `IndicatorSpec` describes a specific configuration of an indicator,
/// including its type, parameters, dependencies, and expected outputs.
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
///
/// // Simple moving average with period 20
/// let spec = IndicatorSpec::new(IndicatorKind::Sma, 20);
/// assert_eq!(spec.period(), 20);
/// assert!(spec.dependencies().is_empty());
///
/// // MACD with custom periods
/// let macd = IndicatorSpec::macd(12, 26, 9);
/// assert_eq!(macd.outputs().len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorSpec {
    /// The type of indicator
    kind: IndicatorKind,
    /// Primary period parameter (if applicable)
    period: usize,
    /// Secondary period (for indicators like MACD with fast/slow)
    secondary_period: Option<usize>,
    /// Tertiary period (for indicators like Stochastic Full)
    tertiary_period: Option<usize>,
    /// Multiplier parameter (for Bollinger Bands)
    multiplier: Option<f64>,
    /// IDs of indicators this one depends on
    dependencies: Vec<String>,
    /// Descriptions of outputs produced
    outputs: Vec<OutputSpec>,
    /// Optional custom name for this indicator instance
    name: Option<Cow<'static, str>>,
}

impl IndicatorSpec {
    /// Creates a new indicator specification with a single period parameter.
    ///
    /// This is the most common constructor for simple indicators like SMA, EMA, and RSI.
    ///
    /// # Arguments
    ///
    /// * `kind` - The type of indicator
    /// * `period` - The lookback period
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
    ///
    /// let sma = IndicatorSpec::new(IndicatorKind::Sma, 20);
    /// assert_eq!(sma.kind(), IndicatorKind::Sma);
    /// assert_eq!(sma.period(), 20);
    /// ```
    #[must_use]
    pub fn new(kind: IndicatorKind, period: usize) -> Self {
        let outputs = Self::default_outputs_for_kind(kind);
        Self {
            kind,
            period,
            secondary_period: None,
            tertiary_period: None,
            multiplier: None,
            dependencies: Vec::new(),
            outputs,
            name: None,
        }
    }

    /// Creates a Bollinger Bands specification.
    ///
    /// # Arguments
    ///
    /// * `period` - The moving average period (typically 20)
    /// * `multiplier` - Standard deviation multiplier (typically 2.0)
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
    ///
    /// let bb = IndicatorSpec::bollinger(20, 2.0);
    /// assert_eq!(bb.kind(), IndicatorKind::BollingerBands);
    /// assert_eq!(bb.multiplier(), Some(2.0));
    /// ```
    #[must_use]
    pub fn bollinger(period: usize, multiplier: f64) -> Self {
        Self {
            kind: IndicatorKind::BollingerBands,
            period,
            secondary_period: None,
            tertiary_period: None,
            multiplier: Some(multiplier),
            dependencies: Vec::new(),
            outputs: vec![
                OutputSpec::new("middle", "Middle band (SMA)"),
                OutputSpec::new("upper", "Upper band (SMA + k*stddev)"),
                OutputSpec::new("lower", "Lower band (SMA - k*stddev)"),
            ],
            name: None,
        }
    }

    /// Creates a MACD specification.
    ///
    /// # Arguments
    ///
    /// * `fast_period` - Fast EMA period (typically 12)
    /// * `slow_period` - Slow EMA period (typically 26)
    /// * `signal_period` - Signal line EMA period (typically 9)
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
    ///
    /// let macd = IndicatorSpec::macd(12, 26, 9);
    /// assert_eq!(macd.kind(), IndicatorKind::Macd);
    /// assert_eq!(macd.period(), 12);
    /// assert_eq!(macd.secondary_period(), Some(26));
    /// assert_eq!(macd.tertiary_period(), Some(9));
    /// ```
    #[must_use]
    pub fn macd(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            kind: IndicatorKind::Macd,
            period: fast_period,
            secondary_period: Some(slow_period),
            tertiary_period: Some(signal_period),
            multiplier: None,
            dependencies: Vec::new(),
            outputs: vec![
                OutputSpec::new("line", "MACD line (fast EMA - slow EMA)"),
                OutputSpec::new("signal", "Signal line (EMA of MACD line)"),
                OutputSpec::new("histogram", "MACD histogram (line - signal)"),
            ],
            name: None,
        }
    }

    /// Creates a Stochastic Fast specification.
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
            dependencies: Vec::new(),
            outputs: vec![
                OutputSpec::new("k", "Fast %K (raw stochastic)"),
                OutputSpec::new("d", "Fast %D (SMA of %K)"),
            ],
            name: None,
        }
    }

    /// Creates a Stochastic Slow specification.
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
            dependencies: Vec::new(),
            outputs: vec![
                OutputSpec::new("k", "Slow %K (SMA of raw stochastic)"),
                OutputSpec::new("d", "Slow %D (SMA of slow %K)"),
            ],
            name: None,
        }
    }

    /// Creates a Stochastic Full specification.
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
            dependencies: Vec::new(),
            outputs: vec![
                OutputSpec::new("k", "Full %K (smoothed stochastic)"),
                OutputSpec::new("d", "Full %D (SMA of full %K)"),
            ],
            name: None,
        }
    }

    /// Returns the default outputs for a given indicator kind.
    fn default_outputs_for_kind(kind: IndicatorKind) -> Vec<OutputSpec> {
        match kind {
            IndicatorKind::Sma => vec![OutputSpec::new("sma", "Simple Moving Average")],
            IndicatorKind::Ema => vec![OutputSpec::new("ema", "Exponential Moving Average")],
            IndicatorKind::EmaWilder => {
                vec![OutputSpec::new("ema_wilder", "Wilder's Exponential Moving Average")]
            }
            IndicatorKind::Rsi => vec![OutputSpec::new("rsi", "Relative Strength Index")],
            IndicatorKind::Atr => vec![OutputSpec::new("atr", "Average True Range")],
            IndicatorKind::Dema => vec![OutputSpec::new("dema", "Double Exponential Moving Average")],
            IndicatorKind::Tema => vec![OutputSpec::new("tema", "Triple Exponential Moving Average")],
            IndicatorKind::TrueRange => vec![OutputSpec::new("tr", "True Range")],
            IndicatorKind::RollingStdDev => {
                vec![OutputSpec::new("stddev", "Rolling Standard Deviation")]
            }
            IndicatorKind::RollingMax => vec![OutputSpec::new("max", "Rolling Maximum")],
            IndicatorKind::RollingMin => vec![OutputSpec::new("min", "Rolling Minimum")],
            IndicatorKind::Custom => vec![OutputSpec::new("custom", "Custom indicator output")],
            // Complex indicators with multiple outputs use their own constructors
            IndicatorKind::Macd
            | IndicatorKind::BollingerBands
            | IndicatorKind::StochasticFast
            | IndicatorKind::StochasticSlow
            | IndicatorKind::StochasticFull => Vec::new(),
        }
    }

    /// Returns the indicator kind.
    #[must_use]
    pub fn kind(&self) -> IndicatorKind {
        self.kind
    }

    /// Returns the primary period parameter.
    #[must_use]
    pub fn period(&self) -> usize {
        self.period
    }

    /// Returns the secondary period parameter, if any.
    #[must_use]
    pub fn secondary_period(&self) -> Option<usize> {
        self.secondary_period
    }

    /// Returns the tertiary period parameter, if any.
    #[must_use]
    pub fn tertiary_period(&self) -> Option<usize> {
        self.tertiary_period
    }

    /// Returns the multiplier parameter, if any.
    #[must_use]
    pub fn multiplier(&self) -> Option<f64> {
        self.multiplier
    }

    /// Returns the dependencies of this indicator.
    #[must_use]
    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }

    /// Returns the output specifications.
    #[must_use]
    pub fn outputs(&self) -> &[OutputSpec] {
        &self.outputs
    }

    /// Returns the custom name, if set.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sets a custom name for this indicator instance.
    ///
    /// # Arguments
    ///
    /// * `name` - The custom name
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
    ///
    /// let spec = IndicatorSpec::new(IndicatorKind::Sma, 20)
    ///     .with_name("fast_sma");
    /// assert_eq!(spec.name(), Some("fast_sma"));
    /// ```
    #[must_use]
    pub fn with_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Adds a dependency on another indicator.
    ///
    /// Dependencies are indicator IDs that must be computed before this indicator.
    ///
    /// # Arguments
    ///
    /// * `dependency` - The ID of the dependency indicator
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::spec::{IndicatorSpec, IndicatorKind};
    ///
    /// // Create a custom indicator that depends on SMA
    /// let spec = IndicatorSpec::new(IndicatorKind::Custom, 20)
    ///     .with_dependency("sma_20");
    /// assert_eq!(spec.dependencies(), &["sma_20"]);
    /// ```
    #[must_use]
    pub fn with_dependency(mut self, dependency: impl Into<String>) -> Self {
        self.dependencies.push(dependency.into());
        self
    }

    /// Adds multiple dependencies.
    #[must_use]
    pub fn with_dependencies(mut self, dependencies: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.dependencies.extend(dependencies.into_iter().map(Into::into));
        self
    }

    /// Returns a unique key for this indicator configuration.
    ///
    /// This key can be used to identify equivalent indicator configurations
    /// for deduplication and caching purposes.
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
            // Format multiplier with limited precision to avoid floating-point issues
            key.push_str(&format!("_{mult:.2}"));
        }
        key
    }

    /// Returns the minimum data length required to compute this indicator.
    ///
    /// This is the number of data points needed before the first valid
    /// (non-NaN) output can be produced.
    #[must_use]
    pub fn min_data_length(&self) -> usize {
        match self.kind {
            IndicatorKind::Sma
            | IndicatorKind::Ema
            | IndicatorKind::EmaWilder
            | IndicatorKind::Rsi
            | IndicatorKind::Atr
            | IndicatorKind::Dema
            | IndicatorKind::Tema
            | IndicatorKind::TrueRange
            | IndicatorKind::RollingStdDev
            | IndicatorKind::RollingMax
            | IndicatorKind::RollingMin
            | IndicatorKind::Custom => self.period,
            IndicatorKind::BollingerBands => self.period,
            IndicatorKind::Macd => {
                // Need slow_period for MACD line, then signal_period for signal
                let slow = self.secondary_period.unwrap_or(26);
                let signal = self.tertiary_period.unwrap_or(9);
                slow + signal - 1
            }
            IndicatorKind::StochasticFast | IndicatorKind::StochasticSlow => {
                let d_period = self.secondary_period.unwrap_or(3);
                self.period + d_period - 1
            }
            IndicatorKind::StochasticFull => {
                let k_smooth = self.secondary_period.unwrap_or(3);
                let d_period = self.tertiary_period.unwrap_or(3);
                self.period + k_smooth + d_period - 2
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_kind_name() {
        assert_eq!(IndicatorKind::Sma.name(), "SMA");
        assert_eq!(IndicatorKind::Ema.name(), "EMA");
        assert_eq!(IndicatorKind::Rsi.name(), "RSI");
        assert_eq!(IndicatorKind::Macd.name(), "MACD");
        assert_eq!(IndicatorKind::BollingerBands.name(), "Bollinger");
    }

    #[test]
    fn test_indicator_kind_output_count() {
        assert_eq!(IndicatorKind::Sma.output_count(), 1);
        assert_eq!(IndicatorKind::Ema.output_count(), 1);
        assert_eq!(IndicatorKind::Rsi.output_count(), 1);
        assert_eq!(IndicatorKind::Macd.output_count(), 3);
        assert_eq!(IndicatorKind::BollingerBands.output_count(), 3);
        assert_eq!(IndicatorKind::StochasticFast.output_count(), 2);
    }

    #[test]
    fn test_indicator_kind_requires_ohlcv() {
        assert!(!IndicatorKind::Sma.requires_ohlcv());
        assert!(!IndicatorKind::Ema.requires_ohlcv());
        assert!(IndicatorKind::Atr.requires_ohlcv());
        assert!(IndicatorKind::TrueRange.requires_ohlcv());
        assert!(IndicatorKind::StochasticFast.requires_ohlcv());
    }

    #[test]
    fn test_output_spec_creation() {
        let output = OutputSpec::new("test", "Test description");
        assert_eq!(output.name(), "test");
        assert_eq!(output.description(), "Test description");
    }

    #[test]
    fn test_indicator_spec_new() {
        let spec = IndicatorSpec::new(IndicatorKind::Sma, 20);
        assert_eq!(spec.kind(), IndicatorKind::Sma);
        assert_eq!(spec.period(), 20);
        assert_eq!(spec.secondary_period(), None);
        assert_eq!(spec.tertiary_period(), None);
        assert_eq!(spec.multiplier(), None);
        assert!(spec.dependencies().is_empty());
        assert_eq!(spec.outputs().len(), 1);
    }

    #[test]
    fn test_indicator_spec_bollinger() {
        let spec = IndicatorSpec::bollinger(20, 2.0);
        assert_eq!(spec.kind(), IndicatorKind::BollingerBands);
        assert_eq!(spec.period(), 20);
        assert_eq!(spec.multiplier(), Some(2.0));
        assert_eq!(spec.outputs().len(), 3);
        assert_eq!(spec.outputs()[0].name(), "middle");
        assert_eq!(spec.outputs()[1].name(), "upper");
        assert_eq!(spec.outputs()[2].name(), "lower");
    }

    #[test]
    fn test_indicator_spec_macd() {
        let spec = IndicatorSpec::macd(12, 26, 9);
        assert_eq!(spec.kind(), IndicatorKind::Macd);
        assert_eq!(spec.period(), 12);
        assert_eq!(spec.secondary_period(), Some(26));
        assert_eq!(spec.tertiary_period(), Some(9));
        assert_eq!(spec.outputs().len(), 3);
        assert_eq!(spec.outputs()[0].name(), "line");
        assert_eq!(spec.outputs()[1].name(), "signal");
        assert_eq!(spec.outputs()[2].name(), "histogram");
    }

    #[test]
    fn test_indicator_spec_stochastic_fast() {
        let spec = IndicatorSpec::stochastic_fast(14, 3);
        assert_eq!(spec.kind(), IndicatorKind::StochasticFast);
        assert_eq!(spec.period(), 14);
        assert_eq!(spec.secondary_period(), Some(3));
        assert_eq!(spec.outputs().len(), 2);
    }

    #[test]
    fn test_indicator_spec_stochastic_slow() {
        let spec = IndicatorSpec::stochastic_slow(14, 3);
        assert_eq!(spec.kind(), IndicatorKind::StochasticSlow);
        assert_eq!(spec.period(), 14);
        assert_eq!(spec.secondary_period(), Some(3));
    }

    #[test]
    fn test_indicator_spec_stochastic_full() {
        let spec = IndicatorSpec::stochastic_full(14, 3, 3);
        assert_eq!(spec.kind(), IndicatorKind::StochasticFull);
        assert_eq!(spec.period(), 14);
        assert_eq!(spec.secondary_period(), Some(3));
        assert_eq!(spec.tertiary_period(), Some(3));
    }

    #[test]
    fn test_indicator_spec_with_name() {
        let spec = IndicatorSpec::new(IndicatorKind::Sma, 20).with_name("fast_sma");
        assert_eq!(spec.name(), Some("fast_sma"));
    }

    #[test]
    fn test_indicator_spec_with_dependency() {
        let spec = IndicatorSpec::new(IndicatorKind::Custom, 20).with_dependency("sma_20");
        assert_eq!(spec.dependencies(), &["sma_20"]);
    }

    #[test]
    fn test_indicator_spec_with_dependencies() {
        let spec = IndicatorSpec::new(IndicatorKind::Custom, 20)
            .with_dependencies(["sma_20", "ema_20"]);
        assert_eq!(spec.dependencies(), &["sma_20", "ema_20"]);
    }

    #[test]
    fn test_indicator_spec_config_key() {
        let sma = IndicatorSpec::new(IndicatorKind::Sma, 20);
        assert_eq!(sma.config_key(), "SMA_20");

        let macd = IndicatorSpec::macd(12, 26, 9);
        assert_eq!(macd.config_key(), "MACD_12_26_9");

        let bb = IndicatorSpec::bollinger(20, 2.0);
        assert_eq!(bb.config_key(), "Bollinger_20_2.00");
    }

    #[test]
    fn test_indicator_spec_min_data_length() {
        let sma = IndicatorSpec::new(IndicatorKind::Sma, 20);
        assert_eq!(sma.min_data_length(), 20);

        let macd = IndicatorSpec::macd(12, 26, 9);
        assert_eq!(macd.min_data_length(), 34); // 26 + 9 - 1

        let stoch_fast = IndicatorSpec::stochastic_fast(14, 3);
        assert_eq!(stoch_fast.min_data_length(), 16); // 14 + 3 - 1

        let stoch_full = IndicatorSpec::stochastic_full(14, 3, 3);
        assert_eq!(stoch_full.min_data_length(), 18); // 14 + 3 + 3 - 2
    }

    #[test]
    fn test_indicator_kind_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(IndicatorKind::Sma);
        set.insert(IndicatorKind::Ema);
        set.insert(IndicatorKind::Sma); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&IndicatorKind::Sma));
        assert!(set.contains(&IndicatorKind::Ema));
    }

    #[test]
    fn test_output_spec_clone() {
        let output = OutputSpec::new("test", "description");
        let cloned = output.clone();
        assert_eq!(output, cloned);
    }

    #[test]
    fn test_indicator_spec_clone() {
        let spec = IndicatorSpec::macd(12, 26, 9).with_name("my_macd");
        let cloned = spec.clone();
        assert_eq!(spec, cloned);
        assert_eq!(cloned.name(), Some("my_macd"));
    }

    #[test]
    fn test_indicator_kind_copy() {
        let kind = IndicatorKind::Rsi;
        let kind2 = kind; // Copy
        assert_eq!(kind, kind2);
    }

    #[test]
    fn test_all_single_output_indicators() {
        let single_output_kinds = [
            IndicatorKind::Sma,
            IndicatorKind::Ema,
            IndicatorKind::EmaWilder,
            IndicatorKind::Rsi,
            IndicatorKind::Atr,
            IndicatorKind::Dema,
            IndicatorKind::Tema,
            IndicatorKind::TrueRange,
            IndicatorKind::RollingStdDev,
            IndicatorKind::RollingMax,
            IndicatorKind::RollingMin,
            IndicatorKind::Custom,
        ];

        for kind in single_output_kinds {
            assert_eq!(kind.output_count(), 1, "Expected {} to have 1 output", kind.name());
            let spec = IndicatorSpec::new(kind, 14);
            assert!(!spec.outputs().is_empty(), "Expected {} to have outputs", kind.name());
        }
    }

    #[test]
    fn test_output_spec_with_string() {
        // Test with owned String instead of &str
        let output = OutputSpec::new(String::from("dynamic"), String::from("dynamic desc"));
        assert_eq!(output.name(), "dynamic");
        assert_eq!(output.description(), "dynamic desc");
    }

    #[test]
    fn test_indicator_spec_with_string_name() {
        let spec = IndicatorSpec::new(IndicatorKind::Sma, 20)
            .with_name(String::from("owned_name"));
        assert_eq!(spec.name(), Some("owned_name"));
    }

    #[test]
    fn test_indicator_spec_chain_builders() {
        let spec = IndicatorSpec::new(IndicatorKind::Custom, 20)
            .with_name("custom_indicator")
            .with_dependency("dep1")
            .with_dependency("dep2");

        assert_eq!(spec.name(), Some("custom_indicator"));
        assert_eq!(spec.dependencies(), &["dep1", "dep2"]);
    }
}
