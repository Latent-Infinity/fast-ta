//! Fusion kernels for efficient computation of multiple statistics.
//!
//! This module provides optimized kernel implementations for performance-critical
//! operations that benefit from algorithmic optimization.
//!
//! # Kernels
//!
//! - [`rolling_extrema`]: Monotonic deque algorithm for O(n) rolling max/min
//!
//! # Performance
//!
//! The rolling extrema kernel uses a monotonic deque data structure to compute
//! rolling maximum and minimum values in O(n) time, compared to the O(n*k) naive
//! approach. This provides 10-100x speedups for larger window sizes.

pub mod rolling_extrema;

// Re-export kernel types for convenient access.
//
// These re-exports allow users to import directly from `kernels` without
// needing to specify the submodule, e.g., `use fast_ta::kernels::rolling_max;`

// Rolling extrema kernel exports: O(n) rolling max/min using monotonic deque
pub use rolling_extrema::{
    rolling_extrema, rolling_extrema_into, rolling_extrema_lookback, rolling_extrema_min_len,
    rolling_max, rolling_max_into, rolling_max_naive, rolling_min, rolling_min_into,
    rolling_min_naive, MonotonicDeque, RollingExtremaOutput,
};
