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

pub use crate::error::{Error, Result};
pub use crate::traits::{SeriesElement, ValidatedInput};
