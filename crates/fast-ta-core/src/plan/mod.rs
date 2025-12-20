//! Plan infrastructure for indicator computation.
//!
//! This module provides the core infrastructure for planning and executing
//! indicator computations efficiently. The plan system enables:
//!
//! - Registration of indicator specifications
//! - Dependency resolution between indicators
//! - DAG-based execution ordering
//! - Potential fusion of compatible computations
//! - Direct mode for independent indicator computation
//!
//! # Architecture
//!
//! The plan infrastructure consists of four main components:
//!
//! 1. **IndicatorSpec**: Describes an indicator's parameters, dependencies, and outputs
//! 2. **Registry**: Central storage for indicator specifications and factory functions
//! 3. **DAG**: Dependency graph for resolving execution order
//! 4. **DirectMode**: Independent indicator computation without plan overhead
//!
//! # Execution Modes
//!
//! ## Direct Mode
//!
//! Direct mode computes each indicator independently without any optimization.
//! This is useful for:
//! - Small number of indicators
//! - Benchmarking and comparison
//! - Simple use cases without dependencies
//!
//! ```
//! use fast_ta_core::plan::direct_mode::{DirectExecutor, IndicatorRequest};
//! use fast_ta_core::plan::IndicatorKind;
//!
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
//! let executor = DirectExecutor::new();
//!
//! let results = executor.execute(&prices, &[
//!     IndicatorRequest::simple(IndicatorKind::Sma, 20),
//!     IndicatorRequest::simple(IndicatorKind::Ema, 10),
//! ]).unwrap();
//! ```
//!
//! ## Plan Mode
//!
//! Plan mode uses DAG-based execution for optimized computation:
//!
//! ```
//! use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
//! use fast_ta_core::plan::dag::DagBuilder;
//!
//! let mut registry = Registry::new();
//!
//! // Register indicators with dependencies
//! registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
//! registry.register("custom",
//!     IndicatorSpec::new(IndicatorKind::Custom, 10)
//!         .with_dependency("sma_20"));
//!
//! // Build execution plan
//! let plan = DagBuilder::from_registry(&registry).build().unwrap();
//!
//! // Execute in topologically sorted order
//! for id in plan.iter() {
//!     println!("Execute: {}", id);
//! }
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
//!
//! For small numbers of indicators, direct mode may be faster due to no plan overhead.

pub mod dag;
pub mod direct_mode;
pub mod registry;
pub mod spec;

// Re-export commonly used types for convenient access
pub use registry::Registry;
pub use spec::{IndicatorKind, IndicatorSpec, OutputSpec};

// Re-export direct mode types
pub use direct_mode::{compute_all_direct, DirectExecutor, IndicatorRequest, IndicatorResult, OhlcvData};
