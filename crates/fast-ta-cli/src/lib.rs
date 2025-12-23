//! fast-ta CLI library
//!
//! This module exposes the CLI components for testing and reuse.

pub mod args;
pub mod csv_parser;
pub mod csv_writer;
pub mod error;

pub use error::{CliError, Result};
