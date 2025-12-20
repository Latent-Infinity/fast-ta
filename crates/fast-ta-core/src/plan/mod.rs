//! Plan infrastructure for indicator computation.
//!
//! This module provides the core infrastructure for planning and executing
//! indicator computations efficiently. The plan system enables:
//!
//! - Registration of indicator specifications
//! - Dependency resolution between indicators
//! - DAG-based execution ordering
//! - Potential fusion of compatible computations
//!
//! # Architecture
//!
//! The plan infrastructure consists of three main components:
//!
//! 1. **IndicatorSpec**: Describes an indicator's parameters, dependencies, and outputs
//! 2. **Registry**: Central storage for indicator specifications and factory functions
//! 3. **DAG**: Dependency graph for resolving execution order (implemented separately)
//!
//! # Example
//!
//! ```ignore
//! use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
//!
//! let mut registry = Registry::new();
//!
//! // Register a simple SMA indicator
//! let spec = IndicatorSpec::new(IndicatorKind::Sma, 20);
//! registry.register("sma_20", spec);
//!
//! // Query registered indicators
//! let registered = registry.get("sma_20");
//! assert!(registered.is_some());
//! ```
//!
//! # Performance Considerations
//!
//! The plan infrastructure adds a small overhead for building and resolving the
//! execution plan, but enables significant optimizations when computing many
//! indicators together:
//!
//! - Shared computations are identified and computed once
//! - Memory bandwidth is optimized through data locality
//! - Fused kernels can be applied when applicable

pub mod registry;
pub mod spec;

// Re-export commonly used types for convenient access
pub use registry::Registry;
pub use spec::{IndicatorKind, IndicatorSpec, OutputSpec};
