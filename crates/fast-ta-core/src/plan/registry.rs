//! Indicator registry for managing indicator specifications.
//!
//! The registry provides a central location for storing and retrieving
//! indicator specifications. It supports:
//!
//! - Registration of indicator specifications by ID
//! - Lookup of registered specifications
//! - Listing all registered indicators
//! - Configuration-based deduplication
//!
//! # Example
//!
//! ```
//! use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
//!
//! let mut registry = Registry::new();
//!
//! // Register indicators
//! registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
//! registry.register("ema_20", IndicatorSpec::new(IndicatorKind::Ema, 20));
//! registry.register("macd", IndicatorSpec::macd(12, 26, 9));
//!
//! // Query registered indicators
//! assert!(registry.contains("sma_20"));
//! assert_eq!(registry.len(), 3);
//!
//! // Get a specification
//! let sma_spec = registry.get("sma_20").unwrap();
//! assert_eq!(sma_spec.period(), 20);
//! ```

use std::collections::HashMap;

use super::spec::IndicatorSpec;

/// Registry for indicator specifications.
///
/// The `Registry` provides a central location for managing indicator specifications.
/// Indicators are registered with a unique string ID and can be queried by that ID.
///
/// # Thread Safety
///
/// The `Registry` is not thread-safe. If you need to share a registry across
/// threads, wrap it in an `Arc<Mutex<Registry>>` or `Arc<RwLock<Registry>>`.
///
/// # Example
///
/// ```
/// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
///
/// let mut registry = Registry::new();
/// registry.register("rsi_14", IndicatorSpec::new(IndicatorKind::Rsi, 14));
///
/// if let Some(spec) = registry.get("rsi_14") {
///     println!("RSI period: {}", spec.period());
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct Registry {
    /// Map from indicator ID to specification
    indicators: HashMap<String, IndicatorSpec>,
    /// Map from config key to indicator ID for deduplication
    config_index: HashMap<String, String>,
}

impl Registry {
    /// Creates a new empty registry.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::Registry;
    ///
    /// let registry = Registry::new();
    /// assert!(registry.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry with pre-allocated capacity.
    ///
    /// Use this when you know approximately how many indicators will be registered
    /// to avoid reallocation during registration.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The number of indicators to pre-allocate space for
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::Registry;
    ///
    /// let registry = Registry::with_capacity(100);
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indicators: HashMap::with_capacity(capacity),
            config_index: HashMap::with_capacity(capacity),
        }
    }

    /// Registers an indicator specification with the given ID.
    ///
    /// If an indicator with the same ID already exists, it will be replaced
    /// and the old specification will be returned.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this indicator
    /// * `spec` - The indicator specification
    ///
    /// # Returns
    ///
    /// The previous specification if one existed with the same ID, or `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    ///
    /// // First registration
    /// let old = registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// assert!(old.is_none());
    ///
    /// // Replace existing
    /// let old = registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 50));
    /// assert!(old.is_some());
    /// assert_eq!(old.unwrap().period(), 20);
    /// ```
    pub fn register(&mut self, id: impl Into<String>, spec: IndicatorSpec) -> Option<IndicatorSpec> {
        let id = id.into();
        let config_key = spec.config_key();

        // Update config index
        self.config_index.insert(config_key, id.clone());

        // Insert and return old value
        self.indicators.insert(id, spec)
    }

    /// Removes an indicator from the registry.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the indicator to remove
    ///
    /// # Returns
    ///
    /// The specification if it existed, or `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    ///
    /// let removed = registry.unregister("sma");
    /// assert!(removed.is_some());
    /// assert!(!registry.contains("sma"));
    /// ```
    pub fn unregister(&mut self, id: &str) -> Option<IndicatorSpec> {
        if let Some(spec) = self.indicators.remove(id) {
            let config_key = spec.config_key();
            // Only remove from config_index if it points to this ID
            if let Some(indexed_id) = self.config_index.get(&config_key) {
                if indexed_id == id {
                    self.config_index.remove(&config_key);
                }
            }
            Some(spec)
        } else {
            None
        }
    }

    /// Gets a reference to an indicator specification by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the indicator to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the specification if found, or `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("ema_10", IndicatorSpec::new(IndicatorKind::Ema, 10));
    ///
    /// if let Some(spec) = registry.get("ema_10") {
    ///     assert_eq!(spec.kind(), IndicatorKind::Ema);
    /// }
    /// ```
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&IndicatorSpec> {
        self.indicators.get(id)
    }

    /// Gets a mutable reference to an indicator specification by ID.
    ///
    /// **Note:** Modifying the specification may invalidate the config index.
    /// Use with caution.
    #[must_use]
    pub fn get_mut(&mut self, id: &str) -> Option<&mut IndicatorSpec> {
        self.indicators.get_mut(id)
    }

    /// Checks if an indicator with the given ID is registered.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID to check
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("rsi", IndicatorSpec::new(IndicatorKind::Rsi, 14));
    ///
    /// assert!(registry.contains("rsi"));
    /// assert!(!registry.contains("macd"));
    /// ```
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.indicators.contains_key(id)
    }

    /// Returns the number of registered indicators.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// assert_eq!(registry.len(), 0);
    ///
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// assert_eq!(registry.len(), 1);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.indicators.len()
    }

    /// Returns `true` if no indicators are registered.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::Registry;
    ///
    /// let registry = Registry::new();
    /// assert!(registry.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indicators.is_empty()
    }

    /// Returns an iterator over all registered indicator IDs.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));
    ///
    /// let ids: Vec<&str> = registry.ids().collect();
    /// assert_eq!(ids.len(), 2);
    /// ```
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.indicators.keys().map(String::as_str)
    }

    /// Returns an iterator over all registered specifications.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));
    ///
    /// for (id, spec) in registry.iter() {
    ///     println!("{}: {} period {}", id, spec.kind().name(), spec.period());
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&str, &IndicatorSpec)> {
        self.indicators.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Finds an indicator by its configuration key.
    ///
    /// This is useful for finding indicators with equivalent configurations,
    /// which can be used for deduplication or caching.
    ///
    /// # Arguments
    ///
    /// * `config_key` - The configuration key to search for
    ///
    /// # Returns
    ///
    /// The ID of the indicator with matching configuration, if found.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("my_sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    ///
    /// // Find by config key
    /// let id = registry.find_by_config("SMA_20");
    /// assert_eq!(id, Some("my_sma"));
    /// ```
    #[must_use]
    pub fn find_by_config(&self, config_key: &str) -> Option<&str> {
        self.config_index.get(config_key).map(String::as_str)
    }

    /// Returns an existing indicator ID if an equivalent specification exists,
    /// or registers the new specification and returns its ID.
    ///
    /// This is useful for automatically deduplicating indicator computations.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID to use if registration is needed
    /// * `spec` - The specification to register
    ///
    /// # Returns
    ///
    /// The ID of the (possibly existing) indicator.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    ///
    /// // First registration
    /// let id1 = registry.get_or_register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// assert_eq!(id1, "sma_20");
    ///
    /// // Second registration with same config but different ID - returns existing
    /// let id2 = registry.get_or_register("another_sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// assert_eq!(id2, "sma_20"); // Returns existing ID
    /// ```
    pub fn get_or_register(&mut self, id: impl Into<String>, spec: IndicatorSpec) -> &str {
        let config_key = spec.config_key();

        // Check if an equivalent indicator already exists
        if let Some(existing_id) = self.config_index.get(&config_key) {
            return existing_id;
        }

        // Register new indicator
        let id = id.into();
        self.config_index.insert(config_key, id.clone());
        self.indicators.insert(id.clone(), spec);

        // Return a reference to the ID stored in config_index
        // SAFETY: We just inserted this key, so it must exist
        self.config_index.get(&self.indicators.get(&id).unwrap().config_key()).unwrap()
    }

    /// Clears all registered indicators.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// assert!(!registry.is_empty());
    ///
    /// registry.clear();
    /// assert!(registry.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.indicators.clear();
        self.config_index.clear();
    }

    /// Returns all indicator IDs that match the given kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The indicator kind to filter by
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("sma_10", IndicatorSpec::new(IndicatorKind::Sma, 10));
    /// registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
    /// registry.register("ema_10", IndicatorSpec::new(IndicatorKind::Ema, 10));
    ///
    /// let sma_ids: Vec<&str> = registry.find_by_kind(IndicatorKind::Sma).collect();
    /// assert_eq!(sma_ids.len(), 2);
    /// ```
    pub fn find_by_kind(&self, kind: super::spec::IndicatorKind) -> impl Iterator<Item = &str> {
        self.indicators
            .iter()
            .filter(move |(_, spec)| spec.kind() == kind)
            .map(|(id, _)| id.as_str())
    }

    /// Validates all registered indicators have their dependencies satisfied.
    ///
    /// Returns a list of (indicator_id, missing_dependency) pairs for any
    /// indicators with unsatisfied dependencies.
    ///
    /// # Example
    ///
    /// ```
    /// use fast_ta_core::plan::{Registry, IndicatorSpec, IndicatorKind};
    ///
    /// let mut registry = Registry::new();
    /// registry.register("custom", IndicatorSpec::new(IndicatorKind::Custom, 20)
    ///     .with_dependency("missing_dep"));
    ///
    /// let missing = registry.validate_dependencies();
    /// assert_eq!(missing.len(), 1);
    /// assert_eq!(missing[0], ("custom", "missing_dep"));
    /// ```
    #[must_use]
    pub fn validate_dependencies(&self) -> Vec<(&str, &str)> {
        let mut missing = Vec::new();
        for (id, spec) in &self.indicators {
            for dep in spec.dependencies() {
                if !self.indicators.contains_key(dep) {
                    missing.push((id.as_str(), dep.as_str()));
                }
            }
        }
        missing
    }
}

/// Converts the registry into an iterator over (id, spec) pairs.
impl IntoIterator for Registry {
    type Item = (String, IndicatorSpec);
    type IntoIter = std::collections::hash_map::IntoIter<String, IndicatorSpec>;

    fn into_iter(self) -> Self::IntoIter {
        self.indicators.into_iter()
    }
}

/// Collects (id, spec) pairs into a registry.
impl<S: Into<String>> FromIterator<(S, IndicatorSpec)> for Registry {
    fn from_iter<I: IntoIterator<Item = (S, IndicatorSpec)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut registry = Registry::with_capacity(lower);
        for (id, spec) in iter {
            registry.register(id, spec);
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::spec::IndicatorKind;

    #[test]
    fn test_registry_new() {
        let registry = Registry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_with_capacity() {
        let registry = Registry::with_capacity(100);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = Registry::new();
        let old = registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        assert!(old.is_none());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("sma_20"));
    }

    #[test]
    fn test_registry_register_replace() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        let old = registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 50));

        assert!(old.is_some());
        assert_eq!(old.unwrap().period(), 20);
        assert_eq!(registry.get("sma").unwrap().period(), 50);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));

        let removed = registry.unregister("sma");
        assert!(removed.is_some());
        assert!(!registry.contains("sma"));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_unregister_nonexistent() {
        let mut registry = Registry::new();
        let removed = registry.unregister("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_registry_get() {
        let mut registry = Registry::new();
        registry.register("ema_10", IndicatorSpec::new(IndicatorKind::Ema, 10));

        let spec = registry.get("ema_10");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().kind(), IndicatorKind::Ema);
        assert_eq!(spec.unwrap().period(), 10);
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = Registry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_get_mut() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));

        // Get mutable reference (though we shouldn't modify in practice)
        let spec = registry.get_mut("sma");
        assert!(spec.is_some());
    }

    #[test]
    fn test_registry_contains() {
        let mut registry = Registry::new();
        registry.register("rsi", IndicatorSpec::new(IndicatorKind::Rsi, 14));

        assert!(registry.contains("rsi"));
        assert!(!registry.contains("macd"));
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let mut registry = Registry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_registry_ids() {
        let mut registry = Registry::new();
        registry.register("a", IndicatorSpec::new(IndicatorKind::Sma, 10));
        registry.register("b", IndicatorSpec::new(IndicatorKind::Ema, 10));

        let mut ids: Vec<&str> = registry.ids().collect();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_registry_iter() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));

        let mut entries: Vec<_> = registry.iter().collect();
        entries.sort_by_key(|(id, _)| *id);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "ema");
        assert_eq!(entries[1].0, "sma");
    }

    #[test]
    fn test_registry_find_by_config() {
        let mut registry = Registry::new();
        registry.register("my_sma", IndicatorSpec::new(IndicatorKind::Sma, 20));

        let id = registry.find_by_config("SMA_20");
        assert_eq!(id, Some("my_sma"));

        let id = registry.find_by_config("EMA_20");
        assert!(id.is_none());
    }

    #[test]
    fn test_registry_get_or_register_new() {
        let mut registry = Registry::new();
        let id = registry.get_or_register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        assert_eq!(id, "sma_20");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_get_or_register_existing() {
        let mut registry = Registry::new();
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));

        // Try to register equivalent config with different ID
        let id = registry.get_or_register("another_sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        assert_eq!(id, "sma_20"); // Should return existing ID
        assert_eq!(registry.len(), 1); // Should not add new entry
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));

        registry.clear();
        assert!(registry.is_empty());
        assert!(registry.find_by_config("SMA_20").is_none());
    }

    #[test]
    fn test_registry_find_by_kind() {
        let mut registry = Registry::new();
        registry.register("sma_10", IndicatorSpec::new(IndicatorKind::Sma, 10));
        registry.register("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema_10", IndicatorSpec::new(IndicatorKind::Ema, 10));
        registry.register("rsi_14", IndicatorSpec::new(IndicatorKind::Rsi, 14));

        let mut sma_ids: Vec<&str> = registry.find_by_kind(IndicatorKind::Sma).collect();
        sma_ids.sort();
        assert_eq!(sma_ids, vec!["sma_10", "sma_20"]);

        let ema_ids: Vec<&str> = registry.find_by_kind(IndicatorKind::Ema).collect();
        assert_eq!(ema_ids, vec!["ema_10"]);

        let macd_ids: Vec<&str> = registry.find_by_kind(IndicatorKind::Macd).collect();
        assert!(macd_ids.is_empty());
    }

    #[test]
    fn test_registry_validate_dependencies_empty() {
        let registry = Registry::new();
        assert!(registry.validate_dependencies().is_empty());
    }

    #[test]
    fn test_registry_validate_dependencies_satisfied() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register(
            "custom",
            IndicatorSpec::new(IndicatorKind::Custom, 10).with_dependency("sma"),
        );

        assert!(registry.validate_dependencies().is_empty());
    }

    #[test]
    fn test_registry_validate_dependencies_missing() {
        let mut registry = Registry::new();
        registry.register(
            "custom",
            IndicatorSpec::new(IndicatorKind::Custom, 10).with_dependency("missing"),
        );

        let missing = registry.validate_dependencies();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], ("custom", "missing"));
    }

    #[test]
    fn test_registry_validate_dependencies_multiple_missing() {
        let mut registry = Registry::new();
        registry.register(
            "custom1",
            IndicatorSpec::new(IndicatorKind::Custom, 10).with_dependency("missing1"),
        );
        registry.register(
            "custom2",
            IndicatorSpec::new(IndicatorKind::Custom, 10)
                .with_dependencies(["missing2", "missing3"]),
        );

        let missing = registry.validate_dependencies();
        assert_eq!(missing.len(), 3);
    }

    #[test]
    fn test_registry_into_iterator() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));

        let mut entries: Vec<_> = registry.into_iter().collect();
        entries.sort_by_key(|(id, _)| id.clone());

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "ema");
        assert_eq!(entries[1].0, "sma");
    }

    #[test]
    fn test_registry_from_iterator() {
        let specs = vec![
            ("sma_20", IndicatorSpec::new(IndicatorKind::Sma, 20)),
            ("ema_20", IndicatorSpec::new(IndicatorKind::Ema, 20)),
        ];

        let registry: Registry = specs.into_iter().collect();
        assert_eq!(registry.len(), 2);
        assert!(registry.contains("sma_20"));
        assert!(registry.contains("ema_20"));
    }

    #[test]
    fn test_registry_clone() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));

        let cloned = registry.clone();
        assert_eq!(cloned.len(), 1);
        assert!(cloned.contains("sma"));
    }

    #[test]
    fn test_registry_debug() {
        let mut registry = Registry::new();
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));

        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("Registry"));
    }

    #[test]
    fn test_registry_complex_specs() {
        let mut registry = Registry::new();

        // Register various complex indicators
        registry.register("macd", IndicatorSpec::macd(12, 26, 9));
        registry.register("bb", IndicatorSpec::bollinger(20, 2.0));
        registry.register("stoch", IndicatorSpec::stochastic_full(14, 3, 3));

        assert_eq!(registry.len(), 3);

        // Verify config keys work correctly
        assert_eq!(registry.find_by_config("MACD_12_26_9"), Some("macd"));
        assert_eq!(registry.find_by_config("Bollinger_20_2.00"), Some("bb"));
    }

    #[test]
    fn test_registry_string_id() {
        let mut registry = Registry::new();

        // Test with String instead of &str
        let id = String::from("dynamic_id");
        registry.register(id, IndicatorSpec::new(IndicatorKind::Sma, 20));

        assert!(registry.contains("dynamic_id"));
    }

    #[test]
    fn test_registry_config_index_update_on_replace() {
        let mut registry = Registry::new();

        // Register with one config
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        assert_eq!(registry.find_by_config("SMA_20"), Some("sma"));

        // Replace with different config
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 50));
        assert_eq!(registry.find_by_config("SMA_50"), Some("sma"));
        // Note: The old config key may still exist but point to the same ID
    }

    #[test]
    fn test_registry_multiple_kinds() {
        let mut registry = Registry::new();

        // Register one of each kind
        registry.register("sma", IndicatorSpec::new(IndicatorKind::Sma, 20));
        registry.register("ema", IndicatorSpec::new(IndicatorKind::Ema, 20));
        registry.register("rsi", IndicatorSpec::new(IndicatorKind::Rsi, 14));
        registry.register("atr", IndicatorSpec::new(IndicatorKind::Atr, 14));
        registry.register("macd", IndicatorSpec::macd(12, 26, 9));
        registry.register("bb", IndicatorSpec::bollinger(20, 2.0));
        registry.register("stoch", IndicatorSpec::stochastic_fast(14, 3));

        assert_eq!(registry.len(), 7);

        // Verify each can be found by kind
        assert_eq!(registry.find_by_kind(IndicatorKind::Sma).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::Ema).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::Rsi).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::Atr).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::Macd).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::BollingerBands).count(), 1);
        assert_eq!(registry.find_by_kind(IndicatorKind::StochasticFast).count(), 1);
    }
}
