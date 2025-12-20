//! fast-ta-experiments: Benchmark suite and micro-experiments
//!
//! This crate contains synthetic data generators, benchmark utilities,
//! and micro-experiments (E01-E07) for validating performance hypotheses.
//!
//! # Modules
//!
//! - [`data`] - Synthetic data generators with seeded RNG for reproducible benchmarks
//! - [`talib_baseline`] - TA-Lib comparison utilities using golden files
//!
//! # Example
//!
//! ```
//! use fast_ta_experiments::data::{generate_random_walk, generate_ohlcv, inject_nans};
//!
//! // Generate reproducible price series
//! let prices = generate_random_walk(1000, 42);
//!
//! // Generate OHLCV data with proper invariants
//! let ohlcv = generate_ohlcv(1000, 42);
//! assert!(ohlcv.validate_invariants());
//!
//! // Inject NaN values for testing sparse data handling
//! let sparse = inject_nans(&prices, 0.05, 42);
//! ```
//!
//! # TA-Lib Comparison
//!
//! ```rust,ignore
//! use fast_ta_experiments::talib_baseline::{load_golden_file, compare_outputs, DEFAULT_TOLERANCE};
//!
//! // Load golden file with TA-Lib reference values
//! let golden = load_golden_file("benches/golden/sma.json")?;
//!
//! // Compare fast-ta output against reference
//! let result = compare_outputs(&fast_ta_output, &golden.test_cases[0].output, DEFAULT_TOLERANCE);
//! assert!(result?.passed);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod data;
pub mod talib_baseline;
