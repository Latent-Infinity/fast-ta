//! fast-ta-experiments: Benchmark suite and micro-experiments
//!
//! This crate contains synthetic data generators, benchmark utilities,
//! and micro-experiments (E01-E07) for validating performance hypotheses.
//!
//! # Modules
//!
//! - [`data`] - Synthetic data generators with seeded RNG for reproducible benchmarks
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

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod data;
