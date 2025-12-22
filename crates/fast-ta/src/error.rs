//! Error types for fast-ta-core.
//!
//! This module defines the error types used throughout the fast-ta library
//! for handling various failure conditions.

use thiserror::Error;

/// The main error type for fast-ta operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// The input data series is too short for the requested operation.
    ///
    /// This error is returned when the input data has fewer elements than
    /// required by the indicator's period or lookback window.
    #[error("insufficient data: required {required} elements, got {actual}")]
    InsufficientData {
        /// The number of data points required.
        required: usize,
        /// The number of data points provided.
        actual: usize,
    },

    /// Failed to convert a numeric value to the target type.
    ///
    /// This error occurs when using `NumCast::from()` to convert values
    /// (e.g., converting a `usize` period to a generic `Float` type) and
    /// the conversion fails.
    #[error("numeric conversion failed: {context}")]
    NumericConversion {
        /// Description of the conversion that failed.
        context: &'static str,
    },

    /// A cyclic dependency was detected in the indicator DAG.
    ///
    /// This error is returned when the dependency graph contains a cycle,
    /// which would result in infinite recursion during execution.
    #[error("cyclic dependency detected involving node {node_id}")]
    CyclicDependency {
        /// Identifier of the node that participates in the cycle.
        node_id: usize,
    },

    /// The input data series is empty.
    ///
    /// This is a special case of insufficient data where no data was provided.
    #[error("empty input: no data provided")]
    EmptyInput,

    /// The period parameter is invalid.
    ///
    /// This error is returned when the period is zero or otherwise invalid
    /// for the requested operation.
    #[error("invalid period {period}: {reason}")]
    InvalidPeriod {
        /// The invalid period value that was provided.
        period: usize,
        /// Description of why the period is invalid.
        reason: &'static str,
    },
}

/// Convenience type alias for Results using the fast-ta Error type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_error() {
        let err = Error::InsufficientData {
            required: 20,
            actual: 10,
        };
        assert_eq!(
            err.to_string(),
            "insufficient data: required 20 elements, got 10"
        );
    }

    #[test]
    fn test_numeric_conversion_error() {
        let err = Error::NumericConversion {
            context: "converting period to float",
        };
        assert_eq!(
            err.to_string(),
            "numeric conversion failed: converting period to float"
        );
    }

    #[test]
    fn test_cyclic_dependency_error() {
        let err = Error::CyclicDependency { node_id: 42 };
        assert_eq!(
            err.to_string(),
            "cyclic dependency detected involving node 42"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let err = Error::EmptyInput;
        assert_eq!(err.to_string(), "empty input: no data provided");
    }

    #[test]
    fn test_invalid_period_error() {
        let err = Error::InvalidPeriod {
            period: 0,
            reason: "period must be at least 1",
        };
        assert_eq!(err.to_string(), "invalid period 0: period must be at least 1");
    }

    #[test]
    fn test_error_equality() {
        let err1 = Error::InsufficientData {
            required: 20,
            actual: 10,
        };
        let err2 = Error::InsufficientData {
            required: 20,
            actual: 10,
        };
        let err3 = Error::InsufficientData {
            required: 30,
            actual: 10,
        };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_clone() {
        let err = Error::CyclicDependency { node_id: 5 };
        let err_clone = err.clone();
        assert_eq!(err, err_clone);
    }

    #[test]
    fn test_error_debug() {
        let err = Error::NumericConversion {
            context: "test context",
        };
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("NumericConversion"));
        assert!(debug_str.contains("test context"));
    }

    #[test]
    fn test_result_type_alias() {
        fn test_fn(succeed: bool) -> Result<i32> {
            if succeed {
                Ok(42)
            } else {
                Err(Error::EmptyInput)
            }
        }

        assert_eq!(test_fn(true).unwrap(), 42);
        assert!(test_fn(false).is_err());
    }

    #[test]
    fn test_error_is_std_error() {
        fn accepts_std_error<E: std::error::Error>(_: E) {}
        let err = Error::EmptyInput;
        accepts_std_error(err);
    }
}
