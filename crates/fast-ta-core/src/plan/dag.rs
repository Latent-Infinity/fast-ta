//! DAG builder for indicator dependency resolution and execution ordering.
//!
//! This module provides a directed acyclic graph (DAG) implementation for
//! resolving indicator dependencies and determining valid execution order.
//!
//! # Overview
//!
//! When computing multiple indicators that depend on each other, we need to
//! ensure that dependencies are computed before the indicators that use them.
//! This module uses [petgraph](https://docs.rs/petgraph/) to build a dependency
//! graph and perform topological sorting to find a valid execution order.
//!
//! # Cycle Detection
//!
//! Cyclic dependencies (e.g., A depends on B, B depends on A) are detected
//! during the topological sort and result in a [`CyclicDependency`] error.
//!
//! [`CyclicDependency`]: crate::Error::CyclicDependency
//!
//! # Example
//!
//! ```
//! use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
//! use fast_ta_core::plan::dag::{DagBuilder, ExecutionPlan};
//!
//! // Create a registry with indicators
//! let mut registry = Registry::new();
//! registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
//! registry.register("sma_50", IndicatorSpec::new(IndicatorKind::Sma, 50));
//! registry.register("custom",
//!     IndicatorSpec::new(IndicatorKind::Custom, 10)
//!         .with_dependencies(["sma_20", "sma_50"]));
//!
//! // Build the DAG and get execution order
//! let plan = DagBuilder::from_registry(&registry).build().unwrap();
//! let order = plan.execution_order();
//!
//! // Custom indicator should come after its dependencies
//! let sma_20_pos = order.iter().position(|id| *id == "sma_20").unwrap();
//! let sma_50_pos = order.iter().position(|id| *id == "sma_50").unwrap();
//! let custom_pos = order.iter().position(|id| *id == "custom").unwrap();
//!
//! assert!(sma_20_pos < custom_pos);
//! assert!(sma_50_pos < custom_pos);
//! ```

use std::collections::HashMap;

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use crate::error::{Error, Result};
use crate::plan::Registry;

/// A node in the indicator dependency graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagNode {
    /// The indicator ID as registered in the Registry.
    id: String,
}

impl DagNode {
    /// Creates a new DAG node with the given indicator ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// Returns the indicator ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// Builder for constructing an indicator dependency DAG.
///
/// The builder collects indicator specifications and their dependencies,
/// then constructs a directed graph that can be topologically sorted to
/// determine execution order.
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
/// use fast_ta_core::plan::dag::DagBuilder;
///
/// let mut registry = Registry::new();
/// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
///
/// let plan = DagBuilder::from_registry(&registry).build().unwrap();
/// assert_eq!(plan.len(), 1);
/// ```
#[derive(Debug)]
pub struct DagBuilder {
    /// The directed graph storing indicator dependencies.
    graph: DiGraph<DagNode, ()>,
    /// Map from indicator ID to node index for fast lookup.
    node_indices: HashMap<String, NodeIndex>,
}

impl DagBuilder {
    /// Creates a new empty DAG builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_indices: HashMap::new(),
        }
    }

    /// Creates a new DAG builder with pre-allocated capacity.
    ///
    /// Use this when you know approximately how many indicators will be added.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(nodes, edges),
            node_indices: HashMap::with_capacity(nodes),
        }
    }

    /// Creates a DAG builder from a registry.
    ///
    /// This is the most common way to create a DAG builder. All indicators
    /// in the registry are added as nodes, and their declared dependencies
    /// are added as edges.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    /// use fast_ta_core::plan::dag::DagBuilder;
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));
    ///
    /// let builder = DagBuilder::from_registry(&registry);
    /// let plan = builder.build().unwrap();
    /// assert_eq!(plan.len(), 2);
    /// ```
    #[must_use]
    pub fn from_registry(registry: &Registry) -> Self {
        let capacity = registry.len();
        let mut builder = Self::with_capacity(capacity, capacity);

        // First pass: add all nodes
        for (id, _spec) in registry.iter() {
            builder.add_node(id);
        }

        // Second pass: add all edges (dependencies)
        for (id, spec) in registry.iter() {
            for dep in spec.dependencies() {
                // Only add edge if the dependency exists in the registry
                if builder.node_indices.contains_key(dep) {
                    // Edge goes from dependency to dependent (dep -> id)
                    // This means: dep must be computed before id
                    builder.add_edge(dep, id);
                }
            }
        }

        builder
    }

    /// Adds a node to the graph.
    ///
    /// If a node with the same ID already exists, this is a no-op.
    ///
    /// # Returns
    ///
    /// The node index of the (possibly existing) node.
    pub fn add_node(&mut self, id: impl Into<String>) -> NodeIndex {
        let id = id.into();
        if let Some(&index) = self.node_indices.get(&id) {
            return index;
        }

        let node = DagNode::new(id.clone());
        let index = self.graph.add_node(node);
        self.node_indices.insert(id, index);
        index
    }

    /// Adds a directed edge from `from` to `to`.
    ///
    /// This represents that `from` must be computed before `to`.
    /// Both nodes must already exist in the graph.
    ///
    /// # Returns
    ///
    /// `true` if the edge was added, `false` if either node doesn't exist.
    pub fn add_edge(&mut self, from: &str, to: &str) -> bool {
        let Some(&from_idx) = self.node_indices.get(from) else {
            return false;
        };
        let Some(&to_idx) = self.node_indices.get(to) else {
            return false;
        };

        self.graph.add_edge(from_idx, to_idx, ());
        true
    }

    /// Adds a dependency relationship.
    ///
    /// This is a convenience method that's equivalent to `add_edge(dependency, dependent)`.
    /// It expresses that `dependent` depends on `dependency`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::dag::DagBuilder;
    ///
    /// let mut builder = DagBuilder::new();
    /// builder.add_node("sma_20");
    /// builder.add_node("custom");
    ///
    /// // custom depends on sma_20 (sma_20 must be computed first)
    /// builder.add_dependency("custom", "sma_20");
    /// ```
    pub fn add_dependency(&mut self, dependent: &str, dependency: &str) -> bool {
        self.add_edge(dependency, dependent)
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Checks if a node with the given ID exists.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.node_indices.contains_key(id)
    }

    /// Builds the execution plan.
    ///
    /// This performs a topological sort on the graph to determine a valid
    /// execution order. If the graph contains a cycle, a `CyclicDependency`
    /// error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CyclicDependency`] if the graph contains a cycle.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    /// use fast_ta_core::plan::dag::DagBuilder;
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    ///
    /// let plan = DagBuilder::from_registry(&registry).build().unwrap();
    /// assert_eq!(plan.execution_order().len(), 1);
    /// ```
    pub fn build(self) -> Result<ExecutionPlan> {
        // Perform topological sort
        match toposort(&self.graph, None) {
            Ok(order) => {
                let execution_order: Vec<String> = order
                    .into_iter()
                    .map(|idx| self.graph[idx].id.clone())
                    .collect();

                Ok(ExecutionPlan {
                    graph: self.graph,
                    node_indices: self.node_indices,
                    execution_order,
                })
            }
            Err(cycle) => {
                // Convert NodeIndex to usize for the error
                Err(Error::CyclicDependency {
                    node_id: cycle.node_id().index(),
                })
            }
        }
    }
}

impl Default for DagBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// An execution plan for computing indicators.
///
/// The execution plan contains a valid topological ordering of indicators,
/// ensuring that all dependencies are computed before the indicators that
/// depend on them.
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
/// use fast_ta_core::plan::dag::DagBuilder;
///
/// let mut registry = Registry::new();
/// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
/// registry.register("bb", IndicatorSpec::bollinger(20, 2.0));
///
/// let plan = DagBuilder::from_registry(&registry).build().unwrap();
///
/// // Iterate through the execution order
/// for id in plan.execution_order() {
///     println!("Compute: {}", id);
/// }
/// ```
#[derive(Debug)]
pub struct ExecutionPlan {
    /// The underlying directed graph.
    graph: DiGraph<DagNode, ()>,
    /// Map from indicator ID to node index.
    node_indices: HashMap<String, NodeIndex>,
    /// The topologically sorted execution order.
    execution_order: Vec<String>,
}

impl ExecutionPlan {
    /// Returns the execution order as a slice of indicator IDs.
    ///
    /// The order is guaranteed to satisfy all dependency constraints:
    /// if indicator A depends on indicator B, then B will appear before A
    /// in the returned slice.
    #[must_use]
    pub fn execution_order(&self) -> &[String] {
        &self.execution_order
    }

    /// Returns an iterator over the execution order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.execution_order.iter().map(String::as_str)
    }

    /// Returns the number of indicators in the plan.
    #[must_use]
    pub fn len(&self) -> usize {
        self.execution_order.len()
    }

    /// Returns `true` if the plan is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.execution_order.is_empty()
    }

    /// Returns the dependencies of a given indicator.
    ///
    /// # Arguments
    ///
    /// * `id` - The indicator ID to query
    ///
    /// # Returns
    ///
    /// An iterator over the IDs of indicators that `id` depends on,
    /// or `None` if the indicator is not in the plan.
    pub fn dependencies(&self, id: &str) -> Option<impl Iterator<Item = &str>> {
        let idx = self.node_indices.get(id)?;

        // Incoming edges represent dependencies (things this node depends on)
        Some(
            self.graph
                .edges_directed(*idx, Direction::Incoming)
                .map(|edge| self.graph[edge.source()].id.as_str()),
        )
    }

    /// Returns the dependents of a given indicator.
    ///
    /// These are the indicators that depend on the given indicator.
    ///
    /// # Arguments
    ///
    /// * `id` - The indicator ID to query
    ///
    /// # Returns
    ///
    /// An iterator over the IDs of indicators that depend on `id`,
    /// or `None` if the indicator is not in the plan.
    pub fn dependents(&self, id: &str) -> Option<impl Iterator<Item = &str>> {
        let idx = self.node_indices.get(id)?;

        // Outgoing edges represent dependents (things that depend on this node)
        Some(
            self.graph
                .edges_directed(*idx, Direction::Outgoing)
                .map(|edge| self.graph[edge.target()].id.as_str()),
        )
    }

    /// Returns `true` if the indicator with `id` is in the plan.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.node_indices.contains_key(id)
    }

    /// Returns the number of dependencies for a given indicator.
    ///
    /// This is the in-degree of the node in the graph.
    #[must_use]
    pub fn dependency_count(&self, id: &str) -> Option<usize> {
        let idx = self.node_indices.get(id)?;
        Some(
            self.graph
                .edges_directed(*idx, Direction::Incoming)
                .count(),
        )
    }

    /// Returns the number of dependents for a given indicator.
    ///
    /// This is the out-degree of the node in the graph.
    #[must_use]
    pub fn dependent_count(&self, id: &str) -> Option<usize> {
        let idx = self.node_indices.get(id)?;
        Some(
            self.graph
                .edges_directed(*idx, Direction::Outgoing)
                .count(),
        )
    }

    /// Returns indicators that have no dependencies (roots of the DAG).
    ///
    /// These indicators can be computed first since they don't depend on
    /// any other indicators.
    pub fn roots(&self) -> impl Iterator<Item = &str> {
        self.execution_order.iter().filter_map(|id| {
            let idx = self.node_indices.get(id)?;
            if self
                .graph
                .edges_directed(*idx, Direction::Incoming)
                .count()
                == 0
            {
                Some(id.as_str())
            } else {
                None
            }
        })
    }

    /// Returns indicators that have no dependents (leaves of the DAG).
    ///
    /// These are the "final" indicators that no other indicator depends on.
    pub fn leaves(&self) -> impl Iterator<Item = &str> {
        self.execution_order.iter().filter_map(|id| {
            let idx = self.node_indices.get(id)?;
            if self
                .graph
                .edges_directed(*idx, Direction::Outgoing)
                .count()
                == 0
            {
                Some(id.as_str())
            } else {
                None
            }
        })
    }
}

/// Converts the execution plan into an iterator over indicator IDs.
impl IntoIterator for ExecutionPlan {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.execution_order.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{IndicatorKind, IndicatorSpec};

    #[test]
    fn test_dag_node_creation() {
        let node = DagNode::new("sma_20");
        assert_eq!(node.id(), "sma_20");
    }

    #[test]
    fn test_dag_node_with_string() {
        let node = DagNode::new(String::from("ema_10"));
        assert_eq!(node.id(), "ema_10");
    }

    #[test]
    fn test_dag_builder_new() {
        let builder = DagBuilder::new();
        assert_eq!(builder.node_count(), 0);
        assert_eq!(builder.edge_count(), 0);
    }

    #[test]
    fn test_dag_builder_default() {
        let builder = DagBuilder::default();
        assert_eq!(builder.node_count(), 0);
    }

    #[test]
    fn test_dag_builder_with_capacity() {
        let builder = DagBuilder::with_capacity(10, 20);
        assert_eq!(builder.node_count(), 0);
    }

    #[test]
    fn test_dag_builder_add_node() {
        let mut builder = DagBuilder::new();
        let idx1 = builder.add_node("sma_20");
        let idx2 = builder.add_node("ema_10");

        assert_eq!(builder.node_count(), 2);
        assert!(builder.contains("sma_20"));
        assert!(builder.contains("ema_10"));
        assert!(!builder.contains("rsi_14"));
        assert_ne!(idx1, idx2);
    }

    #[test]
    fn test_dag_builder_add_node_duplicate() {
        let mut builder = DagBuilder::new();
        let idx1 = builder.add_node("sma_20");
        let idx2 = builder.add_node("sma_20");

        assert_eq!(builder.node_count(), 1);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_dag_builder_add_edge() {
        let mut builder = DagBuilder::new();
        builder.add_node("sma_20");
        builder.add_node("custom");

        let added = builder.add_edge("sma_20", "custom");
        assert!(added);
        assert_eq!(builder.edge_count(), 1);
    }

    #[test]
    fn test_dag_builder_add_edge_missing_node() {
        let mut builder = DagBuilder::new();
        builder.add_node("sma_20");

        let added = builder.add_edge("sma_20", "nonexistent");
        assert!(!added);
        assert_eq!(builder.edge_count(), 0);
    }

    #[test]
    fn test_dag_builder_add_dependency() {
        let mut builder = DagBuilder::new();
        builder.add_node("sma_20");
        builder.add_node("custom");

        // custom depends on sma_20
        let added = builder.add_dependency("custom", "sma_20");
        assert!(added);
        assert_eq!(builder.edge_count(), 1);
    }

    #[test]
    fn test_dag_builder_from_registry_empty() {
        let registry = Registry::new();
        let builder = DagBuilder::from_registry(&registry);

        assert_eq!(builder.node_count(), 0);
        assert_eq!(builder.edge_count(), 0);
    }

    #[test]
    fn test_dag_builder_from_registry_single() {
        let mut registry = Registry::new();
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));

        let builder = DagBuilder::from_registry(&registry);
        assert_eq!(builder.node_count(), 1);
        assert_eq!(builder.edge_count(), 0);
        assert!(builder.contains("sma_20"));
    }

    #[test]
    fn test_dag_builder_from_registry_with_dependencies() {
        let mut registry = Registry::new();
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema_20", IndicatorSpec::new(IndicatorKind::Ema, 20));
        registry.register(
            "custom",
            IndicatorSpec::new(IndicatorKind::Custom, 10)
                .with_dependencies(["sma_20", "ema_20"]),
        );

        let builder = DagBuilder::from_registry(&registry);
        assert_eq!(builder.node_count(), 3);
        assert_eq!(builder.edge_count(), 2);
    }

    #[test]
    fn test_dag_builder_from_registry_missing_dependency() {
        let mut registry = Registry::new();
        registry.register(
            "custom",
            IndicatorSpec::new(IndicatorKind::Custom, 10)
                .with_dependency("nonexistent"),
        );

        let builder = DagBuilder::from_registry(&registry);
        assert_eq!(builder.node_count(), 1);
        assert_eq!(builder.edge_count(), 0); // Edge not added for missing dependency
    }

    #[test]
    fn test_dag_builder_build_empty() {
        let builder = DagBuilder::new();
        let plan = builder.build().unwrap();

        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_dag_builder_build_single() {
        let mut builder = DagBuilder::new();
        builder.add_node("sma_20");

        let plan = builder.build().unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.execution_order(), &["sma_20"]);
    }

    #[test]
    fn test_dag_builder_build_linear_dependencies() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        // a -> b -> c (a must be computed first, then b, then c)
        builder.add_edge("a", "b");
        builder.add_edge("b", "c");

        let plan = builder.build().unwrap();
        let order = plan.execution_order();

        // Verify order: a before b, b before c
        let a_pos = order.iter().position(|x| x == "a").unwrap();
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let c_pos = order.iter().position(|x| x == "c").unwrap();

        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_dag_builder_build_diamond_dependencies() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");
        builder.add_node("d");

        // Diamond pattern:
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        builder.add_edge("a", "b");
        builder.add_edge("a", "c");
        builder.add_edge("b", "d");
        builder.add_edge("c", "d");

        let plan = builder.build().unwrap();
        let order = plan.execution_order();

        let a_pos = order.iter().position(|x| x == "a").unwrap();
        let b_pos = order.iter().position(|x| x == "b").unwrap();
        let c_pos = order.iter().position(|x| x == "c").unwrap();
        let d_pos = order.iter().position(|x| x == "d").unwrap();

        // a must be first
        assert!(a_pos < b_pos);
        assert!(a_pos < c_pos);

        // d must be last
        assert!(b_pos < d_pos);
        assert!(c_pos < d_pos);
    }

    #[test]
    fn test_dag_builder_build_cycle_detection() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");

        // Create a cycle: a -> b -> a
        builder.add_edge("a", "b");
        builder.add_edge("b", "a");

        let result = builder.build();
        assert!(result.is_err());

        match result.unwrap_err() {
            Error::CyclicDependency { .. } => {}
            other => panic!("Expected CyclicDependency error, got {:?}", other),
        }
    }

    #[test]
    fn test_dag_builder_build_self_cycle() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");

        // Self-cycle: a -> a
        builder.add_edge("a", "a");

        let result = builder.build();
        assert!(result.is_err());

        match result.unwrap_err() {
            Error::CyclicDependency { .. } => {}
            other => panic!("Expected CyclicDependency error, got {:?}", other),
        }
    }

    #[test]
    fn test_dag_builder_build_three_node_cycle() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        // Cycle: a -> b -> c -> a
        builder.add_edge("a", "b");
        builder.add_edge("b", "c");
        builder.add_edge("c", "a");

        let result = builder.build();
        assert!(result.is_err());

        match result.unwrap_err() {
            Error::CyclicDependency { .. } => {}
            other => panic!("Expected CyclicDependency error, got {:?}", other),
        }
    }

    #[test]
    fn test_execution_plan_iter() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");

        let plan = builder.build().unwrap();
        let ids: Vec<&str> = plan.iter().collect();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_execution_plan_into_iter() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");

        let plan = builder.build().unwrap();
        let ids: Vec<String> = plan.into_iter().collect();

        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_execution_plan_contains() {
        let mut builder = DagBuilder::new();
        builder.add_node("sma_20");
        builder.add_node("ema_10");

        let plan = builder.build().unwrap();

        assert!(plan.contains("sma_20"));
        assert!(plan.contains("ema_10"));
        assert!(!plan.contains("rsi_14"));
    }

    #[test]
    fn test_execution_plan_dependencies() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        builder.add_edge("a", "c");
        builder.add_edge("b", "c");

        let plan = builder.build().unwrap();

        // c depends on a and b
        let deps: Vec<&str> = plan.dependencies("c").unwrap().collect();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"a"));
        assert!(deps.contains(&"b"));

        // a has no dependencies
        let deps: Vec<&str> = plan.dependencies("a").unwrap().collect();
        assert!(deps.is_empty());

        // Nonexistent node
        assert!(plan.dependencies("nonexistent").is_none());
    }

    #[test]
    fn test_execution_plan_dependents() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        builder.add_edge("a", "b");
        builder.add_edge("a", "c");

        let plan = builder.build().unwrap();

        // a has two dependents: b and c
        let deps: Vec<&str> = plan.dependents("a").unwrap().collect();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"b"));
        assert!(deps.contains(&"c"));

        // b has no dependents
        let deps: Vec<&str> = plan.dependents("b").unwrap().collect();
        assert!(deps.is_empty());

        // Nonexistent node
        assert!(plan.dependents("nonexistent").is_none());
    }

    #[test]
    fn test_execution_plan_dependency_count() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        builder.add_edge("a", "c");
        builder.add_edge("b", "c");

        let plan = builder.build().unwrap();

        assert_eq!(plan.dependency_count("a"), Some(0));
        assert_eq!(plan.dependency_count("b"), Some(0));
        assert_eq!(plan.dependency_count("c"), Some(2));
        assert_eq!(plan.dependency_count("nonexistent"), None);
    }

    #[test]
    fn test_execution_plan_dependent_count() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        builder.add_edge("a", "b");
        builder.add_edge("a", "c");

        let plan = builder.build().unwrap();

        assert_eq!(plan.dependent_count("a"), Some(2));
        assert_eq!(plan.dependent_count("b"), Some(0));
        assert_eq!(plan.dependent_count("c"), Some(0));
        assert_eq!(plan.dependent_count("nonexistent"), None);
    }

    #[test]
    fn test_execution_plan_roots() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");
        builder.add_node("d");

        builder.add_edge("a", "c");
        builder.add_edge("b", "c");
        builder.add_edge("c", "d");

        let plan = builder.build().unwrap();
        let roots: Vec<&str> = plan.roots().collect();

        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&"a"));
        assert!(roots.contains(&"b"));
    }

    #[test]
    fn test_execution_plan_leaves() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        builder.add_edge("a", "b");
        builder.add_edge("a", "c");

        let plan = builder.build().unwrap();
        let leaves: Vec<&str> = plan.leaves().collect();

        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&"b"));
        assert!(leaves.contains(&"c"));
    }

    #[test]
    fn test_execution_plan_all_independent() {
        let mut builder = DagBuilder::new();
        builder.add_node("a");
        builder.add_node("b");
        builder.add_node("c");

        // No edges - all nodes are independent
        let plan = builder.build().unwrap();

        // All nodes are roots and leaves
        let roots: Vec<&str> = plan.roots().collect();
        let leaves: Vec<&str> = plan.leaves().collect();

        assert_eq!(roots.len(), 3);
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_full_integration_with_registry() {
        let mut registry = Registry::new();

        // SMA and EMA are independent
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema_20", IndicatorSpec::new(IndicatorKind::Ema, 20));

        // Bollinger depends on SMA conceptually (though not declared in spec here)
        // For testing, let's make a custom indicator depend on both
        registry.register(
            "combined",
            IndicatorSpec::new(IndicatorKind::Custom, 10)
                .with_dependencies(["sma_20", "ema_20"]),
        );

        let plan = DagBuilder::from_registry(&registry).build().unwrap();

        assert_eq!(plan.len(), 3);

        // Verify combined comes after its dependencies
        let order = plan.execution_order();
        let sma_pos = order.iter().position(|x| x == "sma_20").unwrap();
        let ema_pos = order.iter().position(|x| x == "ema_20").unwrap();
        let combined_pos = order.iter().position(|x| x == "combined").unwrap();

        assert!(sma_pos < combined_pos);
        assert!(ema_pos < combined_pos);

        // Verify roots and leaves
        let roots: Vec<&str> = plan.roots().collect();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&"sma_20"));
        assert!(roots.contains(&"ema_20"));

        let leaves: Vec<&str> = plan.leaves().collect();
        assert_eq!(leaves.len(), 1);
        assert!(leaves.contains(&"combined"));
    }

    #[test]
    fn test_complex_dag() {
        let mut registry = Registry::new();

        // Build a more complex DAG:
        //
        //   sma_10 ─────────────┐
        //                       ▼
        //   sma_20 ──► ema_on_sma ──► final
        //                       ▲
        //   ema_10 ─────────────┘
        //

        registry.register("sma_10", IndicatorSpec::new(IndicatorKind::Sma, 10));
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema_10", IndicatorSpec::new(IndicatorKind::Ema, 10));

        registry.register(
            "ema_on_sma",
            IndicatorSpec::new(IndicatorKind::Custom, 5)
                .with_dependencies(["sma_10", "sma_20", "ema_10"]),
        );

        registry.register(
            "final",
            IndicatorSpec::new(IndicatorKind::Custom, 1).with_dependency("ema_on_sma"),
        );

        let plan = DagBuilder::from_registry(&registry).build().unwrap();
        let order = plan.execution_order();

        // Verify ema_on_sma comes after all its dependencies
        let sma_10_pos = order.iter().position(|x| x == "sma_10").unwrap();
        let sma_20_pos = order.iter().position(|x| x == "sma_20").unwrap();
        let ema_10_pos = order.iter().position(|x| x == "ema_10").unwrap();
        let ema_on_sma_pos = order.iter().position(|x| x == "ema_on_sma").unwrap();
        let final_pos = order.iter().position(|x| x == "final").unwrap();

        assert!(sma_10_pos < ema_on_sma_pos);
        assert!(sma_20_pos < ema_on_sma_pos);
        assert!(ema_10_pos < ema_on_sma_pos);
        assert!(ema_on_sma_pos < final_pos);
    }

    #[test]
    fn test_dag_node_clone_eq() {
        let node1 = DagNode::new("test");
        let node2 = node1.clone();
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_dag_builder_debug() {
        let builder = DagBuilder::new();
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("DagBuilder"));
    }

    #[test]
    fn test_execution_plan_debug() {
        let builder = DagBuilder::new();
        let plan = builder.build().unwrap();
        let debug_str = format!("{:?}", plan);
        assert!(debug_str.contains("ExecutionPlan"));
    }

    #[test]
    fn test_multiple_paths_to_same_node() {
        let mut builder = DagBuilder::new();
        builder.add_node("root");
        builder.add_node("mid1");
        builder.add_node("mid2");
        builder.add_node("leaf");

        // Multiple paths from root to leaf:
        // root -> mid1 -> leaf
        // root -> mid2 -> leaf
        builder.add_edge("root", "mid1");
        builder.add_edge("root", "mid2");
        builder.add_edge("mid1", "leaf");
        builder.add_edge("mid2", "leaf");

        let plan = builder.build().unwrap();
        let order = plan.execution_order();

        let root_pos = order.iter().position(|x| x == "root").unwrap();
        let mid1_pos = order.iter().position(|x| x == "mid1").unwrap();
        let mid2_pos = order.iter().position(|x| x == "mid2").unwrap();
        let leaf_pos = order.iter().position(|x| x == "leaf").unwrap();

        assert!(root_pos < mid1_pos);
        assert!(root_pos < mid2_pos);
        assert!(mid1_pos < leaf_pos);
        assert!(mid2_pos < leaf_pos);
    }

    #[test]
    fn test_wide_dag() {
        let mut builder = DagBuilder::new();

        // One root with many leaves
        builder.add_node("root");
        for i in 0..10 {
            let leaf_id = format!("leaf_{}", i);
            builder.add_node(&leaf_id);
            builder.add_edge("root", &leaf_id);
        }

        let plan = builder.build().unwrap();
        assert_eq!(plan.len(), 11);

        let roots: Vec<&str> = plan.roots().collect();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], "root");

        let leaves: Vec<&str> = plan.leaves().collect();
        assert_eq!(leaves.len(), 10);
    }

    #[test]
    fn test_deep_dag() {
        let mut builder = DagBuilder::new();

        // Deep chain of dependencies
        let depth = 10;
        for i in 0..depth {
            let node_id = format!("node_{}", i);
            builder.add_node(&node_id);
            if i > 0 {
                let prev_id = format!("node_{}", i - 1);
                builder.add_edge(&prev_id, &node_id);
            }
        }

        let plan = builder.build().unwrap();
        assert_eq!(plan.len(), depth);

        // Verify linear order
        for i in 0..depth - 1 {
            let curr_id = format!("node_{}", i);
            let next_id = format!("node_{}", i + 1);
            let curr_pos = plan.execution_order().iter().position(|x| *x == curr_id).unwrap();
            let next_pos = plan.execution_order().iter().position(|x| *x == next_id).unwrap();
            assert!(curr_pos < next_pos);
        }
    }
}
