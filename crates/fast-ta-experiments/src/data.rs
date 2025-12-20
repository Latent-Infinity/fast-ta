//! Synthetic data generators for reproducible benchmarks.
//!
//! This module provides deterministic data generators using seeded RNG
//! to ensure reproducibility across benchmark runs.
//!
//! # Key Features
//!
//! - **Deterministic**: Same seed always produces identical output
//! - **OHLCV Invariants**: Generated OHLCV data maintains proper relationships
//!   (High ≥ Close ≥ Low, High ≥ Open ≥ Low)
//! - **Configurable NaN injection**: For testing sparse data handling
//!
//! # Example Usage
//!
//! ```
//! use fast_ta_experiments::data::{generate_random_walk, generate_ohlcv, inject_nans};
//!
//! // Generate reproducible price series
//! let prices = generate_random_walk(1000, 42);
//! assert_eq!(prices.len(), 1000);
//!
//! // Generate OHLCV data
//! let ohlcv = generate_ohlcv(1000, 42);
//! assert_eq!(ohlcv.open.len(), 1000);
//!
//! // Inject NaN values at 5% ratio
//! let sparse_prices = inject_nans(&prices, 0.05, 42);
//! ```

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// OHLCV Data Structure
// ============================================================================

/// OHLCV (Open, High, Low, Close, Volume) candlestick data.
///
/// This structure holds separate vectors for each component,
/// following a columnar data layout for cache efficiency.
#[derive(Debug, Clone, PartialEq)]
pub struct Ohlcv {
    /// Opening prices for each bar.
    pub open: Vec<f64>,
    /// High prices for each bar (always ≥ open, close, low).
    pub high: Vec<f64>,
    /// Low prices for each bar (always ≤ open, close, high).
    pub low: Vec<f64>,
    /// Closing prices for each bar.
    pub close: Vec<f64>,
    /// Volume for each bar.
    pub volume: Vec<f64>,
}

impl Ohlcv {
    /// Creates a new OHLCV struct with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for all vectors
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            open: Vec::with_capacity(capacity),
            high: Vec::with_capacity(capacity),
            low: Vec::with_capacity(capacity),
            close: Vec::with_capacity(capacity),
            volume: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of bars in the OHLCV data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.open.len()
    }

    /// Returns true if the OHLCV data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    /// Validates that OHLCV invariants are maintained for all bars.
    ///
    /// Invariants checked:
    /// - High ≥ max(Open, Close)
    /// - Low ≤ min(Open, Close)
    /// - High ≥ Low
    /// - All vectors have the same length
    ///
    /// Returns `true` if all invariants are satisfied.
    #[must_use]
    pub fn validate_invariants(&self) -> bool {
        // Check all vectors have the same length
        let len = self.open.len();
        if self.high.len() != len
            || self.low.len() != len
            || self.close.len() != len
            || self.volume.len() != len
        {
            return false;
        }

        // Check OHLCV invariants for each bar
        for i in 0..len {
            let open = self.open[i];
            let high = self.high[i];
            let low = self.low[i];
            let close = self.close[i];
            let volume = self.volume[i];

            // Skip NaN values - they don't need to satisfy invariants
            if open.is_nan() || high.is_nan() || low.is_nan() || close.is_nan() {
                continue;
            }

            // High must be >= max(Open, Close)
            if high < open || high < close {
                return false;
            }

            // Low must be <= min(Open, Close)
            if low > open || low > close {
                return false;
            }

            // High must be >= Low
            if high < low {
                return false;
            }

            // Volume must be non-negative
            if volume < 0.0 {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Random Walk Generator
// ============================================================================

/// Generates a random walk price series with deterministic output.
///
/// The random walk simulates realistic price movement using geometric
/// Brownian motion with configurable drift and volatility.
///
/// # Arguments
///
/// * `n` - Number of data points to generate
/// * `seed` - Seed for the random number generator (same seed = same output)
///
/// # Returns
///
/// A vector of `n` f64 values representing a random walk price series.
///
/// # Example
///
/// ```
/// use fast_ta_experiments::data::generate_random_walk;
///
/// let prices1 = generate_random_walk(100, 42);
/// let prices2 = generate_random_walk(100, 42);
/// assert_eq!(prices1, prices2); // Same seed = same output
///
/// let prices3 = generate_random_walk(100, 123);
/// assert_ne!(prices1, prices3); // Different seed = different output
/// ```
#[must_use]
pub fn generate_random_walk(n: usize, seed: u64) -> Vec<f64> {
    generate_random_walk_with_params(n, seed, 100.0, 0.0001, 0.02)
}

/// Generates a random walk with configurable parameters.
///
/// # Arguments
///
/// * `n` - Number of data points to generate
/// * `seed` - Seed for the random number generator
/// * `initial_price` - Starting price for the random walk
/// * `drift` - Average return per step (e.g., 0.0001 for 0.01% drift)
/// * `volatility` - Standard deviation of returns (e.g., 0.02 for 2%)
///
/// # Returns
///
/// A vector of `n` f64 values representing the random walk.
#[must_use]
pub fn generate_random_walk_with_params(
    n: usize,
    seed: u64,
    initial_price: f64,
    drift: f64,
    volatility: f64,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut prices = Vec::with_capacity(n);
    let mut price = initial_price;

    prices.push(price);

    for _ in 1..n {
        // Generate standard normal using Box-Muller transform
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Apply geometric Brownian motion
        let return_pct = drift + volatility * z;
        price *= 1.0 + return_pct;

        // Ensure price doesn't go negative
        price = price.max(0.01);
        prices.push(price);
    }

    prices
}

// ============================================================================
// OHLCV Generator
// ============================================================================

/// Generates OHLCV candlestick data with deterministic output.
///
/// The generator creates realistic candlestick patterns while maintaining
/// proper OHLCV invariants:
/// - High ≥ max(Open, Close)
/// - Low ≤ min(Open, Close)
/// - High ≥ Low
///
/// # Arguments
///
/// * `n` - Number of bars to generate
/// * `seed` - Seed for the random number generator
///
/// # Returns
///
/// An [`Ohlcv`] struct containing `n` bars of data.
///
/// # Example
///
/// ```
/// use fast_ta_experiments::data::generate_ohlcv;
///
/// let ohlcv = generate_ohlcv(100, 42);
/// assert_eq!(ohlcv.len(), 100);
/// assert!(ohlcv.validate_invariants());
/// ```
#[must_use]
pub fn generate_ohlcv(n: usize, seed: u64) -> Ohlcv {
    generate_ohlcv_with_params(n, seed, 100.0, 0.0001, 0.02)
}

/// Generates OHLCV data with configurable parameters.
///
/// # Arguments
///
/// * `n` - Number of bars to generate
/// * `seed` - Seed for the random number generator
/// * `initial_price` - Starting price for the series
/// * `drift` - Average return per bar
/// * `volatility` - Standard deviation of returns
///
/// # Returns
///
/// An [`Ohlcv`] struct containing `n` bars of data.
#[must_use]
pub fn generate_ohlcv_with_params(
    n: usize,
    seed: u64,
    initial_price: f64,
    drift: f64,
    volatility: f64,
) -> Ohlcv {
    if n == 0 {
        return Ohlcv::with_capacity(0);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ohlcv = Ohlcv::with_capacity(n);
    let mut prev_close = initial_price;

    for _ in 0..n {
        // Generate multiple random values for the bar
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let u3: f64 = rng.random();
        let u4: f64 = rng.random();
        let u5: f64 = rng.random();

        // Generate return using Box-Muller
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Open is typically near previous close with small gap
        let gap_factor = (u3 - 0.5) * volatility * 0.5;
        let open = (prev_close * (1.0 + gap_factor)).max(0.01);

        // Close based on drift + volatility
        let return_pct = drift + volatility * z1;
        let close = (open * (1.0 + return_pct)).max(0.01);

        // High and Low based on intraday volatility
        let intraday_vol = volatility * (0.5 + u4 * 0.5);
        let max_oc = open.max(close);
        let min_oc = open.min(close);

        // High is above max(open, close)
        let high_extension = max_oc * intraday_vol * u4;
        let high = max_oc + high_extension;

        // Low is below min(open, close)
        let low_extension = min_oc * intraday_vol * (1.0 - u4);
        let low = (min_oc - low_extension).max(0.01);

        // Volume with some randomness
        let base_volume = 1_000_000.0;
        let volume = base_volume * (0.5 + u5);

        ohlcv.open.push(open);
        ohlcv.high.push(high);
        ohlcv.low.push(low);
        ohlcv.close.push(close);
        ohlcv.volume.push(volume);

        prev_close = close;
    }

    ohlcv
}

// ============================================================================
// NaN Injection
// ============================================================================

/// Injects NaN values into a series at random positions.
///
/// This function is useful for testing how indicators handle sparse data
/// with missing values.
///
/// # Arguments
///
/// * `series` - The input series to inject NaNs into
/// * `ratio` - The ratio of values to replace with NaN (0.0 to 1.0)
/// * `seed` - Seed for reproducible NaN placement
///
/// # Returns
///
/// A new vector with NaN values injected at random positions.
///
/// # Panics
///
/// Panics if `ratio` is not in the range [0.0, 1.0].
///
/// # Example
///
/// ```
/// use fast_ta_experiments::data::{generate_random_walk, inject_nans};
///
/// let prices = generate_random_walk(100, 42);
/// let sparse = inject_nans(&prices, 0.1, 42); // 10% NaN
///
/// // Count NaN values
/// let nan_count = sparse.iter().filter(|x| x.is_nan()).count();
/// // Approximately 10% should be NaN (random, so not exact)
/// assert!(nan_count > 0);
/// ```
#[must_use]
pub fn inject_nans(series: &[f64], ratio: f64, seed: u64) -> Vec<f64> {
    assert!(
        (0.0..=1.0).contains(&ratio),
        "ratio must be between 0.0 and 1.0"
    );

    if ratio == 0.0 {
        return series.to_vec();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(series.len());

    for &value in series {
        let r: f64 = rng.random();
        if r < ratio {
            result.push(f64::NAN);
        } else {
            result.push(value);
        }
    }

    result
}

/// Injects NaN values into OHLCV data at random positions.
///
/// When a bar is selected for NaN injection, all five values
/// (Open, High, Low, Close, Volume) are set to NaN for that bar.
///
/// # Arguments
///
/// * `ohlcv` - The input OHLCV data
/// * `ratio` - The ratio of bars to replace with NaN (0.0 to 1.0)
/// * `seed` - Seed for reproducible NaN placement
///
/// # Returns
///
/// A new [`Ohlcv`] struct with NaN values injected at random bars.
#[must_use]
pub fn inject_nans_ohlcv(ohlcv: &Ohlcv, ratio: f64, seed: u64) -> Ohlcv {
    assert!(
        (0.0..=1.0).contains(&ratio),
        "ratio must be between 0.0 and 1.0"
    );

    if ratio == 0.0 {
        return ohlcv.clone();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut result = Ohlcv::with_capacity(ohlcv.len());

    for i in 0..ohlcv.len() {
        let r: f64 = rng.random();
        if r < ratio {
            result.open.push(f64::NAN);
            result.high.push(f64::NAN);
            result.low.push(f64::NAN);
            result.close.push(f64::NAN);
            result.volume.push(f64::NAN);
        } else {
            result.open.push(ohlcv.open[i]);
            result.high.push(ohlcv.high[i]);
            result.low.push(ohlcv.low[i]);
            result.close.push(ohlcv.close[i]);
            result.volume.push(ohlcv.volume[i]);
        }
    }

    result
}

// ============================================================================
// Additional Utility Functions
// ============================================================================

/// Generates a simple trending series (for testing trend indicators).
///
/// Creates a series with a clear uptrend, useful for verifying
/// trend-following indicators.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `seed` - Seed for reproducibility
/// * `trend_strength` - How strong the trend is (0.0 = no trend, 1.0 = strong)
///
/// # Returns
///
/// A vector of trending prices.
#[must_use]
pub fn generate_trending_series(n: usize, seed: u64, trend_strength: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;
    let base_drift = 0.001 * trend_strength;
    let volatility = 0.01;

    prices.push(price);

    for _ in 1..n {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let return_pct = base_drift + volatility * z;
        price *= 1.0 + return_pct;
        price = price.max(0.01);
        prices.push(price);
    }

    prices
}

/// Generates a mean-reverting series (for testing oscillators).
///
/// Creates a series that oscillates around a mean, useful for
/// verifying oscillator indicators like RSI or Stochastic.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `seed` - Seed for reproducibility
/// * `mean` - The mean to revert to
/// * `reversion_speed` - How quickly it reverts (0.0 = no reversion, 1.0 = fast)
///
/// # Returns
///
/// A vector of mean-reverting prices.
#[must_use]
pub fn generate_mean_reverting_series(
    n: usize,
    seed: u64,
    mean: f64,
    reversion_speed: f64,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut prices = Vec::with_capacity(n);
    let mut price = mean;
    let volatility = mean * 0.02;
    let reversion = reversion_speed.clamp(0.0, 1.0) * 0.1;

    prices.push(price);

    for _ in 1..n {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Mean reversion component
        let mean_reversion = (mean - price) * reversion;

        // Random component
        let random_shock = volatility * z;

        price += mean_reversion + random_shock;
        price = price.max(0.01);
        prices.push(price);
    }

    prices
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Random Walk Tests
    // ========================================================================

    #[test]
    fn test_generate_random_walk_deterministic() {
        let series1 = generate_random_walk(100, 42);
        let series2 = generate_random_walk(100, 42);
        assert_eq!(series1, series2, "Same seed should produce identical output");
    }

    #[test]
    fn test_generate_random_walk_different_seeds() {
        let series1 = generate_random_walk(100, 42);
        let series2 = generate_random_walk(100, 123);
        assert_ne!(series1, series2, "Different seeds should produce different output");
    }

    #[test]
    fn test_generate_random_walk_length() {
        for n in [0, 1, 10, 100, 1000] {
            let series = generate_random_walk(n, 42);
            assert_eq!(series.len(), n, "Generated series should have length {}", n);
        }
    }

    #[test]
    fn test_generate_random_walk_positive_prices() {
        let series = generate_random_walk(10_000, 42);
        for price in &series {
            assert!(*price > 0.0, "All prices should be positive");
        }
    }

    #[test]
    fn test_generate_random_walk_no_nan() {
        let series = generate_random_walk(1000, 42);
        for price in &series {
            assert!(!price.is_nan(), "Random walk should not produce NaN");
            assert!(!price.is_infinite(), "Random walk should not produce Inf");
        }
    }

    #[test]
    fn test_generate_random_walk_with_params() {
        let series = generate_random_walk_with_params(100, 42, 50.0, 0.001, 0.05);
        assert_eq!(series.len(), 100);
        assert_eq!(series[0], 50.0, "First price should be initial_price");
    }

    // ========================================================================
    // OHLCV Tests
    // ========================================================================

    #[test]
    fn test_generate_ohlcv_deterministic() {
        let ohlcv1 = generate_ohlcv(100, 42);
        let ohlcv2 = generate_ohlcv(100, 42);
        assert_eq!(ohlcv1, ohlcv2, "Same seed should produce identical OHLCV");
    }

    #[test]
    fn test_generate_ohlcv_different_seeds() {
        let ohlcv1 = generate_ohlcv(100, 42);
        let ohlcv2 = generate_ohlcv(100, 123);
        assert_ne!(ohlcv1.close, ohlcv2.close, "Different seeds should produce different output");
    }

    #[test]
    fn test_generate_ohlcv_length() {
        for n in [0, 1, 10, 100, 1000] {
            let ohlcv = generate_ohlcv(n, 42);
            assert_eq!(ohlcv.len(), n);
            assert_eq!(ohlcv.open.len(), n);
            assert_eq!(ohlcv.high.len(), n);
            assert_eq!(ohlcv.low.len(), n);
            assert_eq!(ohlcv.close.len(), n);
            assert_eq!(ohlcv.volume.len(), n);
        }
    }

    #[test]
    fn test_generate_ohlcv_invariants() {
        // Test with multiple seeds and sizes
        for seed in [42, 123, 456, 789] {
            for n in [100, 1000, 10000] {
                let ohlcv = generate_ohlcv(n, seed);
                assert!(
                    ohlcv.validate_invariants(),
                    "OHLCV invariants should hold for seed={}, n={}",
                    seed,
                    n
                );
            }
        }
    }

    #[test]
    fn test_generate_ohlcv_high_ge_low() {
        let ohlcv = generate_ohlcv(10_000, 42);
        for i in 0..ohlcv.len() {
            assert!(
                ohlcv.high[i] >= ohlcv.low[i],
                "High should be >= Low at index {}",
                i
            );
        }
    }

    #[test]
    fn test_generate_ohlcv_high_ge_open_close() {
        let ohlcv = generate_ohlcv(10_000, 42);
        for i in 0..ohlcv.len() {
            assert!(
                ohlcv.high[i] >= ohlcv.open[i],
                "High should be >= Open at index {}",
                i
            );
            assert!(
                ohlcv.high[i] >= ohlcv.close[i],
                "High should be >= Close at index {}",
                i
            );
        }
    }

    #[test]
    fn test_generate_ohlcv_low_le_open_close() {
        let ohlcv = generate_ohlcv(10_000, 42);
        for i in 0..ohlcv.len() {
            assert!(
                ohlcv.low[i] <= ohlcv.open[i],
                "Low should be <= Open at index {}",
                i
            );
            assert!(
                ohlcv.low[i] <= ohlcv.close[i],
                "Low should be <= Close at index {}",
                i
            );
        }
    }

    #[test]
    fn test_generate_ohlcv_volume_positive() {
        let ohlcv = generate_ohlcv(1000, 42);
        for volume in &ohlcv.volume {
            assert!(*volume > 0.0, "Volume should be positive");
        }
    }

    #[test]
    fn test_generate_ohlcv_no_nan() {
        let ohlcv = generate_ohlcv(1000, 42);
        for i in 0..ohlcv.len() {
            assert!(!ohlcv.open[i].is_nan(), "Open should not be NaN");
            assert!(!ohlcv.high[i].is_nan(), "High should not be NaN");
            assert!(!ohlcv.low[i].is_nan(), "Low should not be NaN");
            assert!(!ohlcv.close[i].is_nan(), "Close should not be NaN");
            assert!(!ohlcv.volume[i].is_nan(), "Volume should not be NaN");
        }
    }

    #[test]
    fn test_ohlcv_is_empty() {
        let empty = generate_ohlcv(0, 42);
        assert!(empty.is_empty());

        let non_empty = generate_ohlcv(10, 42);
        assert!(!non_empty.is_empty());
    }

    // ========================================================================
    // NaN Injection Tests
    // ========================================================================

    #[test]
    fn test_inject_nans_deterministic() {
        let series = generate_random_walk(1000, 42);
        let sparse1 = inject_nans(&series, 0.1, 42);
        let sparse2 = inject_nans(&series, 0.1, 42);

        // Check that NaN positions are identical
        for (a, b) in sparse1.iter().zip(sparse2.iter()) {
            assert_eq!(
                a.is_nan(),
                b.is_nan(),
                "NaN positions should be deterministic"
            );
            if !a.is_nan() {
                assert_eq!(*a, *b, "Non-NaN values should be identical");
            }
        }
    }

    #[test]
    fn test_inject_nans_preserves_length() {
        let series = generate_random_walk(1000, 42);
        let sparse = inject_nans(&series, 0.2, 42);
        assert_eq!(sparse.len(), series.len());
    }

    #[test]
    fn test_inject_nans_zero_ratio() {
        let series = generate_random_walk(1000, 42);
        let sparse = inject_nans(&series, 0.0, 42);
        assert_eq!(sparse, series, "Zero ratio should produce identical series");
    }

    #[test]
    fn test_inject_nans_approximate_ratio() {
        let series = generate_random_walk(10_000, 42);
        let ratio = 0.1;
        let sparse = inject_nans(&series, ratio, 42);

        let nan_count = sparse.iter().filter(|x| x.is_nan()).count();
        let actual_ratio = nan_count as f64 / series.len() as f64;

        // Allow ±2% tolerance for statistical variation
        assert!(
            (actual_ratio - ratio).abs() < 0.02,
            "NaN ratio {} should be close to target {}",
            actual_ratio,
            ratio
        );
    }

    #[test]
    fn test_inject_nans_high_ratio() {
        let series = generate_random_walk(1000, 42);
        let sparse = inject_nans(&series, 0.9, 42);

        let nan_count = sparse.iter().filter(|x| x.is_nan()).count();
        assert!(nan_count > 800, "High ratio should produce many NaNs");
    }

    #[test]
    fn test_inject_nans_full_ratio() {
        let series = generate_random_walk(100, 42);
        let sparse = inject_nans(&series, 1.0, 42);

        let nan_count = sparse.iter().filter(|x| x.is_nan()).count();
        assert_eq!(nan_count, 100, "Full ratio should produce all NaNs");
    }

    #[test]
    #[should_panic(expected = "ratio must be between 0.0 and 1.0")]
    fn test_inject_nans_invalid_ratio_negative() {
        let series = generate_random_walk(100, 42);
        inject_nans(&series, -0.1, 42);
    }

    #[test]
    #[should_panic(expected = "ratio must be between 0.0 and 1.0")]
    fn test_inject_nans_invalid_ratio_over_one() {
        let series = generate_random_walk(100, 42);
        inject_nans(&series, 1.1, 42);
    }

    #[test]
    fn test_inject_nans_ohlcv() {
        let ohlcv = generate_ohlcv(1000, 42);
        let sparse = inject_nans_ohlcv(&ohlcv, 0.1, 42);

        assert_eq!(sparse.len(), ohlcv.len());

        // Check that all values in a bar are NaN together
        for i in 0..sparse.len() {
            let open_nan = sparse.open[i].is_nan();
            let high_nan = sparse.high[i].is_nan();
            let low_nan = sparse.low[i].is_nan();
            let close_nan = sparse.close[i].is_nan();
            let volume_nan = sparse.volume[i].is_nan();

            // All should be NaN or all should be non-NaN
            assert!(
                (open_nan && high_nan && low_nan && close_nan && volume_nan)
                    || (!open_nan && !high_nan && !low_nan && !close_nan && !volume_nan),
                "All OHLCV values should be NaN together at bar {}",
                i
            );
        }
    }

    #[test]
    fn test_inject_nans_ohlcv_zero_ratio() {
        let ohlcv = generate_ohlcv(100, 42);
        let sparse = inject_nans_ohlcv(&ohlcv, 0.0, 42);
        assert_eq!(sparse, ohlcv);
    }

    // ========================================================================
    // Trending Series Tests
    // ========================================================================

    #[test]
    fn test_generate_trending_series_deterministic() {
        let series1 = generate_trending_series(100, 42, 0.5);
        let series2 = generate_trending_series(100, 42, 0.5);
        assert_eq!(series1, series2);
    }

    #[test]
    fn test_generate_trending_series_length() {
        for n in [0, 1, 100, 1000] {
            let series = generate_trending_series(n, 42, 0.5);
            assert_eq!(series.len(), n);
        }
    }

    #[test]
    fn test_generate_trending_series_uptrend() {
        let series = generate_trending_series(1000, 42, 1.0);
        // With strong uptrend, end should be higher than start on average
        // Allow some variation due to randomness
        let first_avg = series[..100].iter().sum::<f64>() / 100.0;
        let last_avg = series[900..].iter().sum::<f64>() / 100.0;
        assert!(
            last_avg > first_avg,
            "Strong uptrend should show higher end prices"
        );
    }

    // ========================================================================
    // Mean Reverting Series Tests
    // ========================================================================

    #[test]
    fn test_generate_mean_reverting_deterministic() {
        let series1 = generate_mean_reverting_series(100, 42, 100.0, 0.5);
        let series2 = generate_mean_reverting_series(100, 42, 100.0, 0.5);
        assert_eq!(series1, series2);
    }

    #[test]
    fn test_generate_mean_reverting_length() {
        for n in [0, 1, 100, 1000] {
            let series = generate_mean_reverting_series(n, 42, 100.0, 0.5);
            assert_eq!(series.len(), n);
        }
    }

    #[test]
    fn test_generate_mean_reverting_stays_near_mean() {
        let mean = 100.0;
        let series = generate_mean_reverting_series(10_000, 42, mean, 0.8);

        // With strong mean reversion, average should be close to mean
        let avg: f64 = series.iter().sum::<f64>() / series.len() as f64;
        let deviation = (avg - mean).abs() / mean;

        assert!(
            deviation < 0.1,
            "Mean-reverting series average {} should be close to mean {}",
            avg,
            mean
        );
    }

    // ========================================================================
    // Ohlcv Struct Tests
    // ========================================================================

    #[test]
    fn test_ohlcv_with_capacity() {
        let ohlcv = Ohlcv::with_capacity(100);
        assert!(ohlcv.is_empty());
        assert_eq!(ohlcv.len(), 0);
    }

    #[test]
    fn test_ohlcv_validate_invariants_valid() {
        let ohlcv = Ohlcv {
            open: vec![10.0, 11.0],
            high: vec![12.0, 13.0],
            low: vec![9.0, 10.0],
            close: vec![11.0, 12.0],
            volume: vec![1000.0, 2000.0],
        };
        assert!(ohlcv.validate_invariants());
    }

    #[test]
    fn test_ohlcv_validate_invariants_high_too_low() {
        let ohlcv = Ohlcv {
            open: vec![10.0],
            high: vec![9.0], // Invalid: high < open
            low: vec![8.0],
            close: vec![9.5],
            volume: vec![1000.0],
        };
        assert!(!ohlcv.validate_invariants());
    }

    #[test]
    fn test_ohlcv_validate_invariants_low_too_high() {
        let ohlcv = Ohlcv {
            open: vec![10.0],
            high: vec![12.0],
            low: vec![11.0], // Invalid: low > open
            close: vec![11.5],
            volume: vec![1000.0],
        };
        assert!(!ohlcv.validate_invariants());
    }

    #[test]
    fn test_ohlcv_validate_invariants_mismatched_lengths() {
        let ohlcv = Ohlcv {
            open: vec![10.0, 11.0],
            high: vec![12.0], // Only one element
            low: vec![9.0, 10.0],
            close: vec![11.0, 12.0],
            volume: vec![1000.0, 2000.0],
        };
        assert!(!ohlcv.validate_invariants());
    }

    #[test]
    fn test_ohlcv_validate_invariants_negative_volume() {
        let ohlcv = Ohlcv {
            open: vec![10.0],
            high: vec![12.0],
            low: vec![9.0],
            close: vec![11.0],
            volume: vec![-1000.0], // Invalid: negative volume
        };
        assert!(!ohlcv.validate_invariants());
    }

    #[test]
    fn test_ohlcv_validate_invariants_with_nan() {
        // NaN values should be skipped in validation
        let ohlcv = Ohlcv {
            open: vec![f64::NAN, 10.0],
            high: vec![f64::NAN, 12.0],
            low: vec![f64::NAN, 9.0],
            close: vec![f64::NAN, 11.0],
            volume: vec![f64::NAN, 1000.0],
        };
        assert!(ohlcv.validate_invariants());
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_series() {
        let series = generate_random_walk(0, 42);
        assert!(series.is_empty());

        let ohlcv = generate_ohlcv(0, 42);
        assert!(ohlcv.is_empty());
    }

    #[test]
    fn test_single_element() {
        let series = generate_random_walk(1, 42);
        assert_eq!(series.len(), 1);
        assert_eq!(series[0], 100.0); // Initial price

        let ohlcv = generate_ohlcv(1, 42);
        assert_eq!(ohlcv.len(), 1);
        assert!(ohlcv.validate_invariants());
    }

    #[test]
    fn test_large_series() {
        let series = generate_random_walk(1_000_000, 42);
        assert_eq!(series.len(), 1_000_000);

        // Verify no NaN or Inf in large series
        for price in &series {
            assert!(price.is_finite(), "Large series should have finite values");
        }
    }
}
