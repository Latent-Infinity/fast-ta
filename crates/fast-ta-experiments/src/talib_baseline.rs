//! TA-Lib comparison baseline using Golden Files strategy.
//!
//! This module provides utilities for comparing fast-ta indicator outputs
//! against pre-computed TA-Lib reference values stored in golden files.
//!
//! # Strategy Overview (ADR-001)
//!
//! We use the **Golden Files** approach for TA-Lib comparison:
//!
//! 1. **Correctness Validation**: Compare fast-ta outputs against pre-computed
//!    TA-Lib reference values stored in JSON golden files.
//!
//! 2. **Performance Benchmarking**: Benchmark fast-ta in isolation using Criterion,
//!    with comparative claims based on documented TA-Lib baselines.
//!
//! # Golden File Format
//!
//! Golden files are stored in `benches/golden/` with the following JSON format:
//!
//! ```json
//! {
//!   "indicator": "SMA",
//!   "parameters": { "period": 14 },
//!   "talib_version": "0.4.32",
//!   "generated_at": "2024-12-20T00:00:00Z",
//!   "test_cases": [
//!     {
//!       "name": "random_walk_1k",
//!       "input_seed": 42,
//!       "input_length": 1000,
//!       "output": [null, null, ..., 100.234, 100.456, ...]
//!     }
//!   ]
//! }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use fast_ta_experiments::talib_baseline::{GoldenFile, load_golden_file, compare_outputs};
//!
//! // Load golden file
//! let golden = load_golden_file("benches/golden/sma.json")?;
//!
//! // Compute fast-ta output
//! let input = generate_random_walk(1000, 42);
//! let fast_ta_output = fast_ta_core::indicators::sma(&input, 14)?;
//!
//! // Compare against golden reference
//! let result = compare_outputs(&fast_ta_output, &golden.test_cases[0].output, 1e-10);
//! assert!(result.is_ok());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during golden file operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GoldenFileError {
    /// Failed to read the golden file from disk.
    IoError(String),
    /// Failed to parse the golden file JSON.
    ParseError(String),
    /// The golden file has an invalid format.
    InvalidFormat(String),
    /// The test case was not found in the golden file.
    TestCaseNotFound(String),
    /// The output comparison failed.
    ComparisonFailed {
        /// Index of the first mismatch.
        index: usize,
        /// Expected value from golden file.
        expected: Option<f64>,
        /// Actual value from fast-ta.
        actual: f64,
        /// Relative error between values.
        relative_error: f64,
    },
    /// The output lengths don't match.
    LengthMismatch {
        /// Expected length from golden file.
        expected: usize,
        /// Actual length from fast-ta.
        actual: usize,
    },
}

impl std::fmt::Display for GoldenFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            Self::TestCaseNotFound(name) => write!(f, "Test case not found: {name}"),
            Self::ComparisonFailed {
                index,
                expected,
                actual,
                relative_error,
            } => write!(
                f,
                "Comparison failed at index {index}: expected {expected:?}, got {actual}, relative error: {relative_error}"
            ),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "Length mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for GoldenFileError {}

/// Result type for golden file operations.
pub type GoldenResult<T> = Result<T, GoldenFileError>;

// ============================================================================
// Golden File Data Structures
// ============================================================================

/// A single test case within a golden file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCase {
    /// Name of the test case (e.g., "random_walk_1k").
    pub name: String,
    /// Seed used to generate the input data.
    pub input_seed: u64,
    /// Length of the input data.
    pub input_length: usize,
    /// Expected output values. `None` represents NaN values.
    pub output: Vec<Option<f64>>,
}

impl TestCase {
    /// Creates a new test case.
    #[must_use]
    pub fn new(name: impl Into<String>, input_seed: u64, input_length: usize) -> Self {
        Self {
            name: name.into(),
            input_seed,
            input_length,
            output: Vec::new(),
        }
    }

    /// Sets the expected output values.
    #[must_use]
    pub fn with_output(mut self, output: Vec<Option<f64>>) -> Self {
        self.output = output;
        self
    }
}

/// Indicator parameters stored in golden files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndicatorParams {
    /// Period for most indicators (e.g., SMA, EMA, RSI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period: Option<usize>,
    /// Fast period for MACD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_period: Option<usize>,
    /// Slow period for MACD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slow_period: Option<usize>,
    /// Signal period for MACD and Stochastic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal_period: Option<usize>,
    /// K period for Stochastic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k_period: Option<usize>,
    /// D period for Stochastic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d_period: Option<usize>,
    /// Number of standard deviations for Bollinger Bands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_std_dev: Option<f64>,
    /// Additional parameters as key-value pairs.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for IndicatorParams {
    fn default() -> Self {
        Self {
            period: None,
            fast_period: None,
            slow_period: None,
            signal_period: None,
            k_period: None,
            d_period: None,
            num_std_dev: None,
            extra: HashMap::new(),
        }
    }
}

impl IndicatorParams {
    /// Creates new indicator parameters with a period.
    #[must_use]
    pub fn with_period(period: usize) -> Self {
        Self {
            period: Some(period),
            ..Default::default()
        }
    }

    /// Creates MACD-style parameters.
    #[must_use]
    pub fn macd(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            signal_period: Some(signal_period),
            ..Default::default()
        }
    }

    /// Creates Bollinger Bands parameters.
    #[must_use]
    pub fn bollinger(period: usize, num_std_dev: f64) -> Self {
        Self {
            period: Some(period),
            num_std_dev: Some(num_std_dev),
            ..Default::default()
        }
    }

    /// Creates Stochastic parameters.
    #[must_use]
    pub fn stochastic(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period: Some(k_period),
            d_period: Some(d_period),
            ..Default::default()
        }
    }
}

/// A golden file containing TA-Lib reference outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldenFile {
    /// Name of the indicator (e.g., "SMA", "EMA", "RSI").
    pub indicator: String,
    /// Parameters used for the indicator.
    pub parameters: IndicatorParams,
    /// Version of TA-Lib used to generate the reference values.
    pub talib_version: String,
    /// ISO 8601 timestamp of when the golden file was generated.
    pub generated_at: String,
    /// List of test cases with reference outputs.
    pub test_cases: Vec<TestCase>,
}

impl GoldenFile {
    /// Creates a new golden file.
    #[must_use]
    pub fn new(indicator: impl Into<String>, parameters: IndicatorParams) -> Self {
        Self {
            indicator: indicator.into(),
            parameters,
            talib_version: String::new(),
            generated_at: String::new(),
            test_cases: Vec::new(),
        }
    }

    /// Sets the TA-Lib version.
    #[must_use]
    pub fn with_talib_version(mut self, version: impl Into<String>) -> Self {
        self.talib_version = version.into();
        self
    }

    /// Sets the generation timestamp.
    #[must_use]
    pub fn with_generated_at(mut self, timestamp: impl Into<String>) -> Self {
        self.generated_at = timestamp.into();
        self
    }

    /// Adds a test case.
    #[must_use]
    pub fn with_test_case(mut self, test_case: TestCase) -> Self {
        self.test_cases.push(test_case);
        self
    }

    /// Finds a test case by name.
    #[must_use]
    pub fn find_test_case(&self, name: &str) -> Option<&TestCase> {
        self.test_cases.iter().find(|tc| tc.name == name)
    }
}

// ============================================================================
// Golden File I/O
// ============================================================================

/// Loads a golden file from disk.
///
/// # Arguments
///
/// * `path` - Path to the golden file JSON.
///
/// # Errors
///
/// Returns `GoldenFileError::IoError` if the file cannot be read.
/// Returns `GoldenFileError::ParseError` if the JSON is invalid.
pub fn load_golden_file<P: AsRef<Path>>(path: P) -> GoldenResult<GoldenFile> {
    let content = fs::read_to_string(path.as_ref())
        .map_err(|e| GoldenFileError::IoError(e.to_string()))?;

    serde_json::from_str(&content)
        .map_err(|e| GoldenFileError::ParseError(e.to_string()))
}

/// Saves a golden file to disk.
///
/// # Arguments
///
/// * `path` - Path where the golden file will be saved.
/// * `golden` - The golden file data to save.
///
/// # Errors
///
/// Returns `GoldenFileError::IoError` if the file cannot be written.
pub fn save_golden_file<P: AsRef<Path>>(path: P, golden: &GoldenFile) -> GoldenResult<()> {
    let content = serde_json::to_string_pretty(golden)
        .map_err(|e| GoldenFileError::ParseError(e.to_string()))?;

    fs::write(path.as_ref(), content)
        .map_err(|e| GoldenFileError::IoError(e.to_string()))
}

// ============================================================================
// Output Comparison
// ============================================================================

/// Comparison result between fast-ta and golden reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonResult {
    /// Whether all values matched within tolerance.
    pub passed: bool,
    /// Total number of values compared.
    pub total_values: usize,
    /// Number of values that matched.
    pub matched_values: usize,
    /// Number of NaN values (matching NaNs count as matched).
    pub nan_values: usize,
    /// Maximum relative error observed.
    pub max_relative_error: f64,
    /// Index of maximum relative error.
    pub max_error_index: Option<usize>,
    /// List of mismatched indices (if any).
    pub mismatches: Vec<usize>,
}

impl ComparisonResult {
    /// Creates a successful comparison result.
    fn success(total: usize, nan_count: usize, max_error: f64, max_error_idx: Option<usize>) -> Self {
        Self {
            passed: true,
            total_values: total,
            matched_values: total,
            nan_values: nan_count,
            max_relative_error: max_error,
            max_error_index: max_error_idx,
            mismatches: Vec::new(),
        }
    }

    /// Creates a failed comparison result.
    fn failure(
        total: usize,
        matched: usize,
        nan_count: usize,
        max_error: f64,
        max_error_idx: Option<usize>,
        mismatches: Vec<usize>,
    ) -> Self {
        Self {
            passed: false,
            total_values: total,
            matched_values: matched,
            nan_values: nan_count,
            max_relative_error: max_error,
            max_error_index: max_error_idx,
            mismatches,
        }
    }
}

/// Default tolerance for floating-point comparisons (1e-10 relative error).
pub const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Compares fast-ta output against golden reference values.
///
/// # Arguments
///
/// * `actual` - The fast-ta computed output.
/// * `expected` - The golden reference values (`None` represents NaN).
/// * `tolerance` - Maximum allowed relative error between values.
///
/// # Returns
///
/// A `ComparisonResult` indicating whether the outputs match.
///
/// # Errors
///
/// Returns `GoldenFileError::LengthMismatch` if the output lengths differ.
pub fn compare_outputs(
    actual: &[f64],
    expected: &[Option<f64>],
    tolerance: f64,
) -> GoldenResult<ComparisonResult> {
    if actual.len() != expected.len() {
        return Err(GoldenFileError::LengthMismatch {
            expected: expected.len(),
            actual: actual.len(),
        });
    }

    let mut max_error = 0.0_f64;
    let mut max_error_idx: Option<usize> = None;
    let mut nan_count = 0_usize;
    let mut mismatches = Vec::new();

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        match e {
            None => {
                // Expected NaN
                nan_count += 1;
                if !a.is_nan() {
                    mismatches.push(i);
                }
            }
            Some(exp_val) => {
                if a.is_nan() {
                    // Expected value but got NaN
                    mismatches.push(i);
                } else {
                    // Compare floating-point values
                    let error = relative_error(*a, *exp_val);
                    if error > max_error {
                        max_error = error;
                        max_error_idx = Some(i);
                    }
                    if error > tolerance {
                        mismatches.push(i);
                    }
                }
            }
        }
    }

    let total = actual.len();
    let matched = total - mismatches.len();

    if mismatches.is_empty() {
        Ok(ComparisonResult::success(total, nan_count, max_error, max_error_idx))
    } else {
        Ok(ComparisonResult::failure(
            total, matched, nan_count, max_error, max_error_idx, mismatches,
        ))
    }
}

/// Computes the relative error between two floating-point values.
///
/// For values close to zero, uses absolute error instead.
#[must_use]
pub fn relative_error(actual: f64, expected: f64) -> f64 {
    if expected.abs() < f64::EPSILON {
        (actual - expected).abs()
    } else {
        ((actual - expected) / expected).abs()
    }
}

/// Converts an f64 slice to the golden file format (Option<f64>).
///
/// NaN values are converted to `None`.
#[must_use]
pub fn to_golden_format(values: &[f64]) -> Vec<Option<f64>> {
    values
        .iter()
        .map(|&v| if v.is_nan() { None } else { Some(v) })
        .collect()
}

/// Converts golden file format to f64 slice.
///
/// `None` values are converted to `NaN`.
#[must_use]
pub fn from_golden_format(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect()
}

// ============================================================================
// Reference Timing Data
// ============================================================================

/// Reference timing data for TA-Lib indicators.
///
/// These are documented baseline timings for comparison purposes.
/// Actual TA-Lib performance may vary based on system configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceTimings {
    /// TA-Lib version these timings were measured with.
    pub talib_version: String,
    /// System description (e.g., "Intel Core i7-10700 @ 2.9GHz").
    pub system: String,
    /// Date the timings were measured.
    pub measured_at: String,
    /// Timing data for each indicator.
    pub timings: HashMap<String, IndicatorTimings>,
}

/// Timing data for a single indicator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndicatorTimings {
    /// Indicator name.
    pub indicator: String,
    /// Parameters used.
    pub parameters: IndicatorParams,
    /// Timings for different data sizes.
    pub data_sizes: Vec<DataSizeTiming>,
}

/// Timing for a specific data size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataSizeTiming {
    /// Number of data points.
    pub size: usize,
    /// Mean execution time in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation of execution time in nanoseconds.
    pub stddev_ns: f64,
    /// Throughput in millions of elements per second.
    pub throughput_meps: f64,
}

impl ReferenceTimings {
    /// Creates new reference timings.
    #[must_use]
    pub fn new(talib_version: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            talib_version: talib_version.into(),
            system: system.into(),
            measured_at: String::new(),
            timings: HashMap::new(),
        }
    }

    /// Loads reference timings from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> GoldenResult<Self> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| GoldenFileError::IoError(e.to_string()))?;

        serde_json::from_str(&content)
            .map_err(|e| GoldenFileError::ParseError(e.to_string()))
    }

    /// Gets timing data for an indicator.
    #[must_use]
    pub fn get(&self, indicator: &str) -> Option<&IndicatorTimings> {
        self.timings.get(indicator)
    }
}

// ============================================================================
// Speedup Calculation
// ============================================================================

/// Calculates the speedup factor between fast-ta and TA-Lib.
///
/// # Arguments
///
/// * `fast_ta_ns` - Fast-ta execution time in nanoseconds.
/// * `talib_ns` - TA-Lib reference execution time in nanoseconds.
///
/// # Returns
///
/// The speedup factor (e.g., 2.0 means fast-ta is 2x faster).
#[must_use]
pub fn calculate_speedup(fast_ta_ns: f64, talib_ns: f64) -> f64 {
    if fast_ta_ns <= 0.0 {
        return f64::INFINITY;
    }
    talib_ns / fast_ta_ns
}

/// Formats a speedup factor for display.
#[must_use]
pub fn format_speedup(speedup: f64) -> String {
    if speedup >= 1.0 {
        format!("{:.2}x faster", speedup)
    } else if speedup > 0.0 {
        format!("{:.2}x slower", 1.0 / speedup)
    } else {
        "N/A".to_string()
    }
}

// ============================================================================
// Standard Test Data Configuration
// ============================================================================

/// Standard data sizes for golden file generation and benchmarks.
pub const GOLDEN_DATA_SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];

/// Standard seeds for reproducible test data.
pub const GOLDEN_SEEDS: [u64; 3] = [42, 123, 456];

/// Standard indicator configurations for golden file generation.
#[must_use]
pub fn standard_indicator_configs() -> Vec<(&'static str, IndicatorParams)> {
    vec![
        ("SMA", IndicatorParams::with_period(14)),
        ("EMA", IndicatorParams::with_period(14)),
        ("RSI", IndicatorParams::with_period(14)),
        ("MACD", IndicatorParams::macd(12, 26, 9)),
        ("ATR", IndicatorParams::with_period(14)),
        ("BBANDS", IndicatorParams::bollinger(20, 2.0)),
        ("STOCH", IndicatorParams::stochastic(14, 3)),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // GoldenFile Tests
    // ========================================================================

    #[test]
    fn test_golden_file_creation() {
        let golden = GoldenFile::new("SMA", IndicatorParams::with_period(14))
            .with_talib_version("0.4.32")
            .with_generated_at("2024-12-20T00:00:00Z");

        assert_eq!(golden.indicator, "SMA");
        assert_eq!(golden.parameters.period, Some(14));
        assert_eq!(golden.talib_version, "0.4.32");
        assert!(golden.test_cases.is_empty());
    }

    #[test]
    fn test_golden_file_with_test_cases() {
        let test_case = TestCase::new("random_walk_1k", 42, 1000)
            .with_output(vec![None, None, Some(100.0), Some(101.0)]);

        let golden = GoldenFile::new("SMA", IndicatorParams::with_period(14))
            .with_test_case(test_case.clone());

        assert_eq!(golden.test_cases.len(), 1);
        assert_eq!(golden.test_cases[0].name, "random_walk_1k");
        assert_eq!(golden.test_cases[0].input_seed, 42);
        assert_eq!(golden.test_cases[0].output.len(), 4);
    }

    #[test]
    fn test_find_test_case() {
        let tc1 = TestCase::new("test_1", 42, 100);
        let tc2 = TestCase::new("test_2", 123, 200);

        let golden = GoldenFile::new("SMA", IndicatorParams::with_period(14))
            .with_test_case(tc1)
            .with_test_case(tc2);

        assert!(golden.find_test_case("test_1").is_some());
        assert!(golden.find_test_case("test_2").is_some());
        assert!(golden.find_test_case("nonexistent").is_none());
    }

    // ========================================================================
    // IndicatorParams Tests
    // ========================================================================

    #[test]
    fn test_indicator_params_period() {
        let params = IndicatorParams::with_period(20);
        assert_eq!(params.period, Some(20));
        assert!(params.fast_period.is_none());
    }

    #[test]
    fn test_indicator_params_macd() {
        let params = IndicatorParams::macd(12, 26, 9);
        assert_eq!(params.fast_period, Some(12));
        assert_eq!(params.slow_period, Some(26));
        assert_eq!(params.signal_period, Some(9));
    }

    #[test]
    fn test_indicator_params_bollinger() {
        let params = IndicatorParams::bollinger(20, 2.0);
        assert_eq!(params.period, Some(20));
        assert!((params.num_std_dev.unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_indicator_params_stochastic() {
        let params = IndicatorParams::stochastic(14, 3);
        assert_eq!(params.k_period, Some(14));
        assert_eq!(params.d_period, Some(3));
    }

    // ========================================================================
    // Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_outputs_exact_match() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];

        let result = compare_outputs(&actual, &expected, DEFAULT_TOLERANCE).unwrap();
        assert!(result.passed);
        assert_eq!(result.total_values, 3);
        assert_eq!(result.matched_values, 3);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn test_compare_outputs_with_nans() {
        let actual = vec![f64::NAN, f64::NAN, 3.0];
        let expected = vec![None, None, Some(3.0)];

        let result = compare_outputs(&actual, &expected, DEFAULT_TOLERANCE).unwrap();
        assert!(result.passed);
        assert_eq!(result.nan_values, 2);
    }

    #[test]
    fn test_compare_outputs_within_tolerance() {
        let actual = vec![1.0, 2.0, 3.0000000001];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];

        let result = compare_outputs(&actual, &expected, 1e-9).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_compare_outputs_outside_tolerance() {
        let actual = vec![1.0, 2.0, 3.1];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];

        let result = compare_outputs(&actual, &expected, 1e-10).unwrap();
        assert!(!result.passed);
        assert_eq!(result.mismatches, vec![2]);
    }

    #[test]
    fn test_compare_outputs_length_mismatch() {
        let actual = vec![1.0, 2.0];
        let expected = vec![Some(1.0), Some(2.0), Some(3.0)];

        let result = compare_outputs(&actual, &expected, DEFAULT_TOLERANCE);
        assert!(matches!(result, Err(GoldenFileError::LengthMismatch { .. })));
    }

    #[test]
    fn test_compare_outputs_nan_mismatch() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![Some(1.0), None, Some(3.0)]; // Expected NaN at index 1

        let result = compare_outputs(&actual, &expected, DEFAULT_TOLERANCE).unwrap();
        assert!(!result.passed);
        assert_eq!(result.mismatches, vec![1]);
    }

    // ========================================================================
    // Utility Function Tests
    // ========================================================================

    #[test]
    fn test_relative_error() {
        assert!((relative_error(1.0, 1.0) - 0.0).abs() < f64::EPSILON);
        assert!((relative_error(1.1, 1.0) - 0.1).abs() < f64::EPSILON);
        assert!((relative_error(0.9, 1.0) - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_relative_error_near_zero() {
        // When expected is near zero, use absolute error
        let error = relative_error(0.0001, 0.0);
        assert!((error - 0.0001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_golden_format() {
        let values = vec![1.0, f64::NAN, 3.0];
        let golden = to_golden_format(&values);

        assert_eq!(golden[0], Some(1.0));
        assert_eq!(golden[1], None);
        assert_eq!(golden[2], Some(3.0));
    }

    #[test]
    fn test_from_golden_format() {
        let golden = vec![Some(1.0), None, Some(3.0)];
        let values = from_golden_format(&golden);

        assert!((values[0] - 1.0).abs() < f64::EPSILON);
        assert!(values[1].is_nan());
        assert!((values[2] - 3.0).abs() < f64::EPSILON);
    }

    // ========================================================================
    // Speedup Calculation Tests
    // ========================================================================

    #[test]
    fn test_calculate_speedup() {
        assert!((calculate_speedup(100.0, 200.0) - 2.0).abs() < f64::EPSILON);
        assert!((calculate_speedup(200.0, 100.0) - 0.5).abs() < f64::EPSILON);
        assert!((calculate_speedup(100.0, 100.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_speedup_edge_cases() {
        assert_eq!(calculate_speedup(0.0, 100.0), f64::INFINITY);
    }

    #[test]
    fn test_format_speedup() {
        assert_eq!(format_speedup(2.0), "2.00x faster");
        assert_eq!(format_speedup(0.5), "2.00x slower");
        assert_eq!(format_speedup(1.0), "1.00x faster");
    }

    // ========================================================================
    // Standard Config Tests
    // ========================================================================

    #[test]
    fn test_golden_data_sizes() {
        assert_eq!(GOLDEN_DATA_SIZES.len(), 4);
        assert_eq!(GOLDEN_DATA_SIZES[0], 1_000);
        assert_eq!(GOLDEN_DATA_SIZES[3], 1_000_000);
    }

    #[test]
    fn test_golden_seeds() {
        assert_eq!(GOLDEN_SEEDS.len(), 3);
        assert_eq!(GOLDEN_SEEDS[0], 42);
    }

    #[test]
    fn test_standard_indicator_configs() {
        let configs = standard_indicator_configs();
        assert_eq!(configs.len(), 7);

        // Check all 7 baseline indicators are included
        let names: Vec<_> = configs.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"SMA"));
        assert!(names.contains(&"EMA"));
        assert!(names.contains(&"RSI"));
        assert!(names.contains(&"MACD"));
        assert!(names.contains(&"ATR"));
        assert!(names.contains(&"BBANDS"));
        assert!(names.contains(&"STOCH"));
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_golden_file_serialization() {
        let test_case = TestCase::new("test", 42, 100)
            .with_output(vec![None, Some(1.0), Some(2.0)]);

        let golden = GoldenFile::new("SMA", IndicatorParams::with_period(14))
            .with_talib_version("0.4.32")
            .with_generated_at("2024-12-20T00:00:00Z")
            .with_test_case(test_case);

        // Serialize to JSON
        let json = serde_json::to_string(&golden).expect("serialization failed");

        // Deserialize back
        let parsed: GoldenFile = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(parsed, golden);
    }

    #[test]
    fn test_reference_timings_serialization() {
        let mut timings = ReferenceTimings::new("0.4.32", "Test System");
        timings.measured_at = "2024-12-20".to_string();

        let indicator_timing = IndicatorTimings {
            indicator: "SMA".to_string(),
            parameters: IndicatorParams::with_period(14),
            data_sizes: vec![DataSizeTiming {
                size: 1000,
                mean_ns: 1000.0,
                stddev_ns: 50.0,
                throughput_meps: 1.0,
            }],
        };
        timings.timings.insert("SMA".to_string(), indicator_timing);

        // Serialize and deserialize
        let json = serde_json::to_string(&timings).expect("serialization failed");
        let parsed: ReferenceTimings = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(parsed.talib_version, "0.4.32");
        assert!(parsed.get("SMA").is_some());
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_golden_file_error_display() {
        let err = GoldenFileError::IoError("file not found".to_string());
        assert!(err.to_string().contains("IO error"));
        assert!(err.to_string().contains("file not found"));

        let err = GoldenFileError::LengthMismatch { expected: 100, actual: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));

        let err = GoldenFileError::ComparisonFailed {
            index: 5,
            expected: Some(1.0),
            actual: 2.0,
            relative_error: 1.0,
        };
        assert!(err.to_string().contains("index 5"));
    }

    #[test]
    fn test_comparison_result() {
        let result = ComparisonResult::success(100, 10, 1e-12, Some(5));
        assert!(result.passed);
        assert_eq!(result.total_values, 100);
        assert_eq!(result.nan_values, 10);

        let result = ComparisonResult::failure(100, 95, 10, 0.1, Some(5), vec![1, 2, 3, 4, 5]);
        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 5);
    }

    // ========================================================================
    // TestCase Tests
    // ========================================================================

    #[test]
    fn test_test_case_creation() {
        let tc = TestCase::new("my_test", 42, 1000);
        assert_eq!(tc.name, "my_test");
        assert_eq!(tc.input_seed, 42);
        assert_eq!(tc.input_length, 1000);
        assert!(tc.output.is_empty());
    }

    #[test]
    fn test_test_case_with_output() {
        let output = vec![None, Some(1.0), Some(2.0)];
        let tc = TestCase::new("test", 42, 100).with_output(output.clone());
        assert_eq!(tc.output, output);
    }
}
