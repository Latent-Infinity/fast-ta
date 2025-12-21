//! Fusion kernels for efficient computation of multiple statistics.
//!
//! This module provides optimized kernel implementations that compute multiple
//! related statistics in a single pass over the data, reducing memory bandwidth
//! and improving cache utilization.
//!
//! # Kernels
//!
//! - [`running_stat`]: Welford's algorithm for numerically stable mean, variance, and standard deviation
//! - [`ema_fusion`]: Fused computation of EMA-family indicators (EMA, DEMA, TEMA, MACD)
//! - [`rolling_extrema`]: Monotonic deque algorithm for O(n) rolling max/min
//!
//! # Performance
//!
//! Fusion kernels are designed to be more efficient than computing statistics
//! independently when multiple related values are needed. For example, computing
//! both mean and standard deviation in a single pass is faster than computing
//! them separately because:
//!
//! 1. Data is loaded into cache only once
//! 2. Intermediate values are reused
//! 3. Memory bandwidth is reduced
//!
//! # Numeric Stability
//!
//! All kernels use numerically stable algorithms (e.g., Welford's algorithm for
//! variance) to handle extreme values and prevent catastrophic cancellation.

pub mod ema_fusion;
pub mod rolling_extrema;
pub mod running_stat;

// Re-export kernel types for convenient access
pub use ema_fusion::{
    ema_fusion, ema_fusion_into, ema_multi, ema_multi_into, macd_fusion, macd_fusion_into,
    EmaFusionOutput, MacdFusionOutput,
};
pub use rolling_extrema::{
    rolling_extrema, rolling_extrema_into, rolling_max, rolling_max_into, rolling_max_naive,
    rolling_min, rolling_min_into, rolling_min_naive, MonotonicDeque, RollingExtremaOutput,
};
pub use running_stat::{
    rolling_stats, rolling_stats_into, RollingStat, RollingStatOutput, RunningStat,
};
