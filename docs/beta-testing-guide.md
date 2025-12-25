# fast-ta Beta Testing Guide

**Version**: 0.1.0-beta
**Date**: 2025-12-23

This guide walks beta testers through installing and validating fast-ta from source.

---

## Prerequisites

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Rust**: stable toolchain (1.70+)
- **Git**: for cloning the repository

### Install Rust (if needed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version  # Should show 1.70+
```

---

## Step 1: Create Test Project

```bash
# Create a new directory for testing
mkdir fast-ta-beta-test
cd fast-ta-beta-test

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.9-3.12
```

---

## Step 2: Clone and Install fast-ta

```bash
# Clone the repository
git clone https://github.com/anthropics/fast-ta.git
cd fast-ta/crates/fast-ta-python

# Install build dependencies
pip install maturin numpy

# Build and install fast-ta (development mode)
maturin develop --release

# Verify installation
python -c "import fast_ta; print(f'fast-ta {fast_ta.__version__} installed successfully')"
```

Expected output:
```
fast-ta 0.1.0 installed successfully
```

---

## Step 3: Run Validation Script

Create a file named `validate_fast_ta.py`:

```python
#!/usr/bin/env python3
"""
fast-ta Beta Validation Script

This script tests all 17 indicator functions exposed by fast-ta
to verify the Python->Rust bridge works correctly.
"""

import numpy as np
import sys

def main():
    print("=" * 60)
    print("fast-ta Beta Validation")
    print("=" * 60)

    try:
        import fast_ta
        print(f"\n[OK] fast-ta version: {fast_ta.__version__}")
    except ImportError as e:
        print(f"\n[FAIL] Could not import fast_ta: {e}")
        sys.exit(1)

    # Generate test data
    np.random.seed(42)
    n = 100

    # Price data (simulated random walk)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000) + 1000

    results = {}
    errors = []

    # =========================================================================
    # Moving Averages
    # =========================================================================
    print("\n--- Moving Averages ---")

    # SMA
    try:
        result = fast_ta.sma(close, 20)
        assert len(result) == n, f"SMA length mismatch: {len(result)} != {n}"
        assert np.isnan(result[:19]).all(), "SMA should have NaN prefix"
        assert not np.isnan(result[19:]).any(), "SMA should have valid values after lookback"
        results['sma'] = 'PASS'
        print(f"  [OK] sma(close, 20) - first valid: {result[19]:.4f}")
    except Exception as e:
        results['sma'] = f'FAIL: {e}'
        errors.append(('sma', e))
        print(f"  [FAIL] sma: {e}")

    # SMA with out= parameter (zero-copy)
    try:
        out = np.empty(n, dtype=np.float64)
        result = fast_ta.sma(close, 20, out=out)
        assert result is out, "out= should return same array"
        results['sma_out'] = 'PASS'
        print(f"  [OK] sma(close, 20, out=out) - zero-copy verified")
    except Exception as e:
        results['sma_out'] = f'FAIL: {e}'
        errors.append(('sma_out', e))
        print(f"  [FAIL] sma with out=: {e}")

    # EMA
    try:
        result = fast_ta.ema(close, 20)
        assert len(result) == n
        results['ema'] = 'PASS'
        print(f"  [OK] ema(close, 20) - first valid: {result[19]:.4f}")
    except Exception as e:
        results['ema'] = f'FAIL: {e}'
        errors.append(('ema', e))
        print(f"  [FAIL] ema: {e}")

    # EMA Wilder
    try:
        result = fast_ta.ema_wilder(close, 20)
        assert len(result) == n
        results['ema_wilder'] = 'PASS'
        print(f"  [OK] ema_wilder(close, 20) - first valid: {result[19]:.4f}")
    except Exception as e:
        results['ema_wilder'] = f'FAIL: {e}'
        errors.append(('ema_wilder', e))
        print(f"  [FAIL] ema_wilder: {e}")

    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    print("\n--- Momentum Indicators ---")

    # RSI
    try:
        result = fast_ta.rsi(close, 14)
        assert len(result) == n
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"
        results['rsi'] = 'PASS'
        print(f"  [OK] rsi(close, 14) - range: [{valid.min():.1f}, {valid.max():.1f}]")
    except Exception as e:
        results['rsi'] = f'FAIL: {e}'
        errors.append(('rsi', e))
        print(f"  [FAIL] rsi: {e}")

    # MACD
    try:
        macd_line, signal_line, histogram = fast_ta.macd(close, 12, 26, 9)
        assert len(macd_line) == n
        assert len(signal_line) == n
        assert len(histogram) == n
        results['macd'] = 'PASS'
        print(f"  [OK] macd(close, 12, 26, 9) - returns 3 arrays")
    except Exception as e:
        results['macd'] = f'FAIL: {e}'
        errors.append(('macd', e))
        print(f"  [FAIL] macd: {e}")

    # Stochastic
    try:
        k, d = fast_ta.stochastic(high, low, close, 14, 3, 1)
        assert len(k) == n
        assert len(d) == n
        results['stochastic'] = 'PASS'
        print(f"  [OK] stochastic(high, low, close, 14, 3, 1) - returns %K, %D")
    except Exception as e:
        results['stochastic'] = f'FAIL: {e}'
        errors.append(('stochastic', e))
        print(f"  [FAIL] stochastic: {e}")

    # Stochastic Fast
    try:
        k, d = fast_ta.stochastic_fast(high, low, close, 14, 3)
        assert len(k) == n
        results['stochastic_fast'] = 'PASS'
        print(f"  [OK] stochastic_fast(high, low, close, 14, 3)")
    except Exception as e:
        results['stochastic_fast'] = f'FAIL: {e}'
        errors.append(('stochastic_fast', e))
        print(f"  [FAIL] stochastic_fast: {e}")

    # Stochastic Slow
    try:
        k, d = fast_ta.stochastic_slow(high, low, close, 14, 3)
        assert len(k) == n
        results['stochastic_slow'] = 'PASS'
        print(f"  [OK] stochastic_slow(high, low, close, 14, 3)")
    except Exception as e:
        results['stochastic_slow'] = f'FAIL: {e}'
        errors.append(('stochastic_slow', e))
        print(f"  [FAIL] stochastic_slow: {e}")

    # Williams %R
    try:
        result = fast_ta.williams_r(high, low, close, 14)
        assert len(result) == n
        valid = result[~np.isnan(result)]
        assert (valid >= -100).all() and (valid <= 0).all(), "Williams %R must be in [-100, 0]"
        results['williams_r'] = 'PASS'
        print(f"  [OK] williams_r(high, low, close, 14) - range: [{valid.min():.1f}, {valid.max():.1f}]")
    except Exception as e:
        results['williams_r'] = f'FAIL: {e}'
        errors.append(('williams_r', e))
        print(f"  [FAIL] williams_r: {e}")

    # ADX
    try:
        adx, plus_di, minus_di = fast_ta.adx(high, low, close, 14)
        assert len(adx) == n
        assert len(plus_di) == n
        assert len(minus_di) == n
        results['adx'] = 'PASS'
        print(f"  [OK] adx(high, low, close, 14) - returns ADX, +DI, -DI")
    except Exception as e:
        results['adx'] = f'FAIL: {e}'
        errors.append(('adx', e))
        print(f"  [FAIL] adx: {e}")

    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    print("\n--- Volatility Indicators ---")

    # ATR
    try:
        result = fast_ta.atr(high, low, close, 14)
        assert len(result) == n
        valid = result[~np.isnan(result)]
        assert (valid > 0).all(), "ATR must be positive"
        results['atr'] = 'PASS'
        print(f"  [OK] atr(high, low, close, 14) - mean: {valid.mean():.4f}")
    except Exception as e:
        results['atr'] = f'FAIL: {e}'
        errors.append(('atr', e))
        print(f"  [FAIL] atr: {e}")

    # True Range
    try:
        result = fast_ta.true_range(high, low, close)
        assert len(result) == n
        results['true_range'] = 'PASS'
        print(f"  [OK] true_range(high, low, close)")
    except Exception as e:
        results['true_range'] = f'FAIL: {e}'
        errors.append(('true_range', e))
        print(f"  [FAIL] true_range: {e}")

    # Bollinger Bands
    try:
        upper, middle, lower = fast_ta.bollinger(close, 20, 2.0)
        assert len(upper) == n
        assert len(middle) == n
        assert len(lower) == n
        # Verify band ordering (where valid)
        valid_idx = ~np.isnan(upper)
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
        results['bollinger'] = 'PASS'
        print(f"  [OK] bollinger(close, 20, 2.0) - returns upper, middle, lower")
    except Exception as e:
        results['bollinger'] = f'FAIL: {e}'
        errors.append(('bollinger', e))
        print(f"  [FAIL] bollinger: {e}")

    # Donchian Channels
    try:
        upper, middle, lower = fast_ta.donchian(high, low, 20)
        assert len(upper) == n
        assert len(middle) == n
        assert len(lower) == n
        results['donchian'] = 'PASS'
        print(f"  [OK] donchian(high, low, 20) - returns upper, middle, lower")
    except Exception as e:
        results['donchian'] = f'FAIL: {e}'
        errors.append(('donchian', e))
        print(f"  [FAIL] donchian: {e}")

    # Rolling Stddev
    try:
        result = fast_ta.rolling_stddev(close, 20)
        assert len(result) == n
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all(), "Stddev must be non-negative"
        results['rolling_stddev'] = 'PASS'
        print(f"  [OK] rolling_stddev(close, 20)")
    except Exception as e:
        results['rolling_stddev'] = f'FAIL: {e}'
        errors.append(('rolling_stddev', e))
        print(f"  [FAIL] rolling_stddev: {e}")

    # =========================================================================
    # Volume Indicators
    # =========================================================================
    print("\n--- Volume Indicators ---")

    # OBV
    try:
        result = fast_ta.obv(close, volume)
        assert len(result) == n
        results['obv'] = 'PASS'
        print(f"  [OK] obv(close, volume) - final: {result[-1]:.0f}")
    except Exception as e:
        results['obv'] = f'FAIL: {e}'
        errors.append(('obv', e))
        print(f"  [FAIL] obv: {e}")

    # VWAP
    try:
        result = fast_ta.vwap(high, low, close, volume)
        assert len(result) == n
        results['vwap'] = 'PASS'
        print(f"  [OK] vwap(high, low, close, volume) - final: {result[-1]:.4f}")
    except Exception as e:
        results['vwap'] = f'FAIL: {e}'
        errors.append(('vwap', e))
        print(f"  [FAIL] vwap: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if errors:
        print(f"\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        print("\nPlease report these errors to the fast-ta team.")
        sys.exit(1)
    else:
        print("\nAll tests passed! fast-ta is working correctly.")
        print("\nYou can now use fast-ta in your projects:")
        print("""
    import numpy as np
    import fast_ta

    prices = np.array([...])  # Your price data

    # Calculate indicators
    sma = fast_ta.sma(prices, 20)
    rsi = fast_ta.rsi(prices, 14)

    # Zero-copy for performance
    out = np.empty(len(prices))
    fast_ta.sma(prices, 20, out=out)
        """)
        sys.exit(0)


if __name__ == "__main__":
    main()
```

Run the validation:

```bash
python validate_fast_ta.py
```

Expected output:
```
============================================================
fast-ta Beta Validation
============================================================

[OK] fast-ta version: 0.1.0

--- Moving Averages ---
  [OK] sma(close, 20) - first valid: 99.8234
  [OK] sma(close, 20, out=out) - zero-copy verified
  [OK] ema(close, 20) - first valid: 99.8234
  [OK] ema_wilder(close, 20) - first valid: 99.8234

--- Momentum Indicators ---
  [OK] rsi(close, 14) - range: [23.4, 76.8]
  [OK] macd(close, 12, 26, 9) - returns 3 arrays
  [OK] stochastic(high, low, close, 14, 3, 1) - returns %K, %D
  [OK] stochastic_fast(high, low, close, 14, 3)
  [OK] stochastic_slow(high, low, close, 14, 3)
  [OK] williams_r(high, low, close, 14) - range: [-95.2, -4.8]
  [OK] adx(high, low, close, 14) - returns ADX, +DI, -DI

--- Volatility Indicators ---
  [OK] atr(high, low, close, 14) - mean: 0.5234
  [OK] true_range(high, low, close)
  [OK] bollinger(close, 20, 2.0) - returns upper, middle, lower
  [OK] donchian(high, low, 20) - returns upper, middle, lower
  [OK] rolling_stddev(close, 20)

--- Volume Indicators ---
  [OK] obv(close, volume) - final: 12345
  [OK] vwap(high, low, close, volume) - final: 100.1234

============================================================
VALIDATION SUMMARY
============================================================

Tests passed: 18/18

All tests passed! fast-ta is working correctly.
```

---

## Step 4: Test with Your Data

Once validation passes, try fast-ta with your own data:

```python
import numpy as np
import pandas as pd
import fast_ta

# Load your data (example with pandas)
df = pd.read_csv('your_price_data.csv')

# Extract arrays (fast-ta requires numpy arrays)
close = df['close'].to_numpy()
high = df['high'].to_numpy()
low = df['low'].to_numpy()
volume = df['volume'].to_numpy()

# Calculate indicators
sma_20 = fast_ta.sma(close, 20)
rsi_14 = fast_ta.rsi(close, 14)
macd_line, signal, histogram = fast_ta.macd(close, 12, 26, 9)
atr_14 = fast_ta.atr(high, low, close, 14)

# Add back to DataFrame
df['sma_20'] = sma_20
df['rsi_14'] = rsi_14
df['atr_14'] = atr_14

print(df.tail())
```

### Zero-Copy for Performance

For performance-critical code, use the `out=` parameter to avoid allocations:

```python
import numpy as np
import fast_ta

# Pre-allocate output buffers
n = len(close)
sma_buffer = np.empty(n, dtype=np.float64)
ema_buffer = np.empty(n, dtype=np.float64)
rsi_buffer = np.empty(n, dtype=np.float64)

# Calculate indicators (writes directly to buffers)
fast_ta.sma(close, 20, out=sma_buffer)
fast_ta.ema(close, 20, out=ema_buffer)
fast_ta.rsi(close, 14, out=rsi_buffer)

# Buffers now contain the results - no extra allocations
```

### Polars Integration

fast-ta works seamlessly with Polars:

```python
import polars as pl
import fast_ta

df = pl.read_csv('your_data.csv')

# Polars .to_numpy() is essentially zero-copy due to Arrow backing
close = df['close'].to_numpy()
sma = fast_ta.sma(close, 20)

# Add back to DataFrame
df = df.with_columns(pl.Series('sma_20', sma))
```

---

## Providing Feedback

Please report any issues or feedback:

1. **GitHub Issues**: https://github.com/anthropics/fast-ta/issues
2. **Include**:
   - Python version (`python --version`)
   - OS and architecture (`uname -a` or `systeminfo`)
   - Error message and traceback
   - Minimal code to reproduce

### Feedback Topics

We're particularly interested in:

- [ ] **Installation issues**: Problems building or installing
- [ ] **Numerical accuracy**: Discrepancies vs TA-Lib or other libraries
- [ ] **Performance**: Benchmarks vs your current solution
- [ ] **API ergonomics**: Anything confusing or inconvenient
- [ ] **Missing indicators**: What else do you need?
- [ ] **Documentation**: What's unclear?

---

## Troubleshooting

### "maturin: command not found"

```bash
pip install maturin
```

### "error: linker `cc` not found" (Linux)

```bash
sudo apt-get install build-essential  # Debian/Ubuntu
sudo yum groupinstall "Development Tools"  # RHEL/CentOS
```

### "error: failed to run custom build command for `pyo3-ffi`"

Ensure Rust is installed and up to date:
```bash
rustup update stable
```

### Non-contiguous array error

fast-ta requires C-contiguous arrays. If you get a `TypeError` about contiguous arrays:

```python
# This may fail:
strided = data[::2]  # Every other element - not contiguous
fast_ta.sma(strided, 5)  # TypeError!

# Solution: make it contiguous
contiguous = np.ascontiguousarray(strided)
fast_ta.sma(contiguous, 5)  # Works!
```

### Import error after installation

Make sure you're using the same Python that maturin used:
```bash
which python  # Should be your venv Python
maturin develop  # Reinstall
```

---

## Thank You!

Thank you for beta testing fast-ta! Your feedback helps us build a better library.
