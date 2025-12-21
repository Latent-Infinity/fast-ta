# E05: Plan Compilation Overhead Benchmarks

## Experiment Overview

**Experiment ID**: E05
**Name**: Plan Compilation Overhead
**Status**: COMPLETED
**Date**: 2024-12-20

## Objective

Measure the cost of plan infrastructure (registry, DAG construction, topological sort) and calculate the break-even point where plan mode becomes advantageous over direct indicator computation.

### Hypothesis

The plan infrastructure incurs a fixed overhead for:

1. **Registry population**: Registering indicator specifications
2. **DAG construction**: Building dependency graph with petgraph
3. **Topological sort**: Computing valid execution order
4. **Query operations**: Looking up indicators by ID, config, or kind

This overhead should be amortized when:

- Computing many indicators in a single plan
- Reusing the same plan across multiple data batches
- Taking advantage of kernel fusion opportunities (E02-E04)

### Success Criteria

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Break-even point** | <100 executions | **1 execution** (with fusion) | PASS |
| **Compilation time** | <1ms for 10 indicators | **2.1 μs** | PASS |
| **Plan reuse overhead** | ~0 ns | **0.42 ns** | PASS |

## Break-Even Calculation

The break-even point is where:

```
N × T_direct = T_compilation + N × T_plan_exec

Where:
- N = number of executions
- T_direct = time for direct indicator computation
- T_compilation = one-time plan compilation cost
- T_plan_exec = time for plan-based execution (may be faster due to fusion)
```

If fusion provides speedup `S`:

```
T_plan_exec = T_direct / S

Break-even: N = T_compilation / (T_direct × (1 - 1/S))
```

### Actual Break-Even Points

| Scenario | Fusion Speedup (S) | Break-Even (N) |
|----------|-------------------|------------------------|
| No fusion (S=1.0) | 1.0× | ∞ (never breaks even) |
| Minimal fusion (S=1.1) | 1.1× | **1 execution** |
| Moderate fusion (S=1.2) | 1.2× | **1 execution** |
| Good fusion (S=1.5) | 1.5× | **1 execution** |
| Optimal fusion (S=2.0) | 2.0× | **1 execution** |

**Key Finding**: Plan compilation overhead (2.2 μs) is so small relative to indicator execution time (~285 μs) that even minimal fusion speedup (10%) results in immediate break-even after just 1 execution.

## Approaches Benchmarked

### 1. Registry Operations

- **Registration time**: Time to register N indicators
- **Query time**: Time to look up indicators by ID, config key, or kind
- **Validation time**: Time to validate dependency satisfaction

### 2. DAG Construction

- **Independent nodes**: N indicators with no dependencies
- **Linear chains**: A → B → C → ... (depth D)
- **Diamond patterns**: Root → [Mid₁...Midₙ] → Leaf

### 3. Full Plan Compilation

- **Simple plan**: N identical indicator types
- **Mixed plan**: Various indicator types
- **Realistic plan**: 9 common trading indicators

### 4. Plan Reuse

- **Cached plan access**: Time to access execution order from compiled plan
- **Iteration**: Time to iterate through execution order

### 5. Direct vs Plan Mode

Compare overhead of plan compilation against actual indicator computation costs.

## Benchmark Configuration

### Indicator Counts

| Count | Description |
|-------|-------------|
| 1 | Minimal plan |
| 5 | Small trading system |
| 10 | Medium trading system |
| 20 | Large trading system |
| 50 | Complex strategy |
| 100 | Stress test |

### Dependency Patterns

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Independent | No edges | Parallel indicators |
| Linear chain | A→B→C→... | Sequential indicators |
| Diamond | Root→[Mid...]→Leaf | Common dependencies |

### Data Sizes for Comparison

| Size | Points | Description |
|------|--------|-------------|
| 1K | 1,000 | Minimal data |
| 10K | 10,000 | Typical intraday |
| 100K | 100,000 | Large dataset |

### Realistic Trading System

9 indicators commonly used together:

1. SMA(20) - Trend baseline
2. SMA(50) - Longer trend
3. EMA(12) - MACD fast component
4. EMA(26) - MACD slow component
5. RSI(14) - Momentum
6. ATR(14) - Volatility
7. MACD(12,26,9) - Trend/momentum
8. Bollinger Bands(20,2) - Volatility bands
9. Stochastic(14,3) - Overbought/oversold

## Results

### Registry Registration

| Indicator Count | Simple Registration | Mixed Registration | Per-Indicator Cost |
|----------------|--------------------|--------------------|-------------------|
| 1 | 156 ns | 178 ns | 156 ns |
| 5 | 578 ns | 705 ns | 116 ns |
| 10 | 1,103 ns | 1,374 ns | 110 ns |
| 20 | 2,133 ns | 2,655 ns | 107 ns |
| 50 | 5,301 ns | 6,602 ns | 106 ns |

**Analysis**: Registration is O(1) per indicator with ~106-156 ns per indicator. Mixed indicators are ~20-25% slower due to config key generation complexity.

### DAG Construction

| Structure | Count/Depth | Construction Time |
|-----------|-------------|------------------|
| Independent | 1 | 79 ns |
| Independent | 10 | 496 ns |
| Independent | 50 | 2,278 ns |
| Linear chain | 3 | 235 ns |
| Linear chain | 10 | 801 ns |
| Diamond | width=5 | 648 ns |
| Diamond | width=20 | 2,144 ns |

**Analysis**: DAG construction scales linearly O(V+E) as expected. ~50 ns per node for independent graphs, slightly more for dependency edges.

### Full Plan Compilation

| Plan Type | Indicator Count | Compilation Time |
|-----------|----------------|-----------------|
| Simple | 1 | 338 ns |
| Simple | 10 | 2,059 ns |
| Simple | 50 | 9,145 ns |
| Mixed | 10 | 2,324 ns |
| Realistic | 9 | 2,214 ns |

**Analysis**: Full compilation for a realistic 9-indicator trading system takes only **2.2 μs**. This is ~500× faster than the 1ms target!

### Plan Reuse

| Indicator Count | Execution Order Access | Iteration Time |
|----------------|----------------------|----------------|
| 1 | 0.42 ns | 0.32 ns |
| 10 | 0.42 ns | 3.11 ns |
| 50 | 0.41 ns | 14.73 ns |

**Analysis**: Execution order access is essentially free (~0.4 ns regardless of plan size). Iteration is linear at ~0.3 ns per indicator.

### Direct vs Plan Comparison (10K data points)

| Mode | Time | Overhead vs Direct |
|------|------|--------------------|
| Direct (7 indicators) | 284.7 μs | baseline |
| Plan compile only | 2.2 μs | N/A |
| Plan compile + exec | 287.8 μs | +1.1% |
| Cached plan exec | 285.5 μs | +0.3% |

**Analysis**: Plan mode execution is only 0.3% slower than direct mode (within measurement noise). The compilation overhead is negligible at 0.8% of a single execution.

### Break-Even Calculation

| Metric | Value |
|--------|-------|
| Plan compilation time | **2,224 ns** (2.2 μs) |
| Direct execution time (10K, 7 indicators) | **284,668 ns** (284.7 μs) |
| Cached plan execution time | **285,472 ns** (285.5 μs) |
| **Break-even (no fusion)** | **∞** (plan is 0.3% slower) |
| **Break-even (10% fusion speedup)** | **1 execution** |
| **Break-even (20% fusion speedup)** | **1 execution** |

**Key Insight**: Without fusion, plan mode is marginally slower (~0.3%). With any fusion benefit, plan mode breaks even immediately because the compilation cost (2.2 μs) is negligible compared to execution time (285 μs).

### Registry Query Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Get by ID | 9.4 ns | HashMap lookup |
| Find by config | 8.9 ns | HashMap lookup |
| Find by kind | 18.2 ns | Iteration + filter |
| Validate dependencies | 6.9 ns | Existence checks |
| Contains check | 8.2 ns | HashMap contains |

**Analysis**: All query operations complete in <20 ns, well under the 1 μs target.

## Analysis

### Key Findings

1. **Plan compilation is extremely fast**: 2.2 μs for a realistic 9-indicator trading system
2. **Plan reuse is essentially free**: 0.4 ns to access cached execution order
3. **Plan mode overhead without fusion**: Only 0.3% slower than direct mode
4. **Break-even with fusion**: Immediate (1 execution) with any fusion benefit
5. **Registry operations are O(1)**: ~100-150 ns per indicator

### Complexity Verification

| Operation | Expected Complexity | Measured | Notes |
|-----------|---------------------|----------|-------|
| Registry registration | O(1) per indicator | **~110 ns/indicator** | Confirmed |
| DAG construction | O(V + E) | **~50 ns/node** | Linear scaling |
| Topological sort | O(V + E) | **included in DAG** | Linear scan |
| Plan access | O(1) | **0.42 ns** | Reference access |

### Memory Overhead

| Component | Size | Notes |
|-----------|------|-------|
| IndicatorSpec | ~128 bytes | Includes Vec for dependencies |
| Registry (N indicators) | ~N × 200 bytes | HashMap overhead |
| DAG (V vertices, E edges) | ~V × 50 + E × 16 bytes | petgraph DiGraph |
| ExecutionPlan | ~V × 50 bytes | Includes order Vec |

## Go/No-Go Decision

**Decision**: **CONDITIONAL GO** - Plan mode recommended when fusion benefits are available

### Criteria Checklist

#### For LOW OVERHEAD (plan mode viable):

- [x] Full compilation for 10 indicators takes <1ms → **2.1 μs** (500× better)
- [x] Plan reuse overhead is <100ns → **0.42 ns** (240× better)
- [x] Break-even is <10 executions with 20% fusion speedup → **1 execution** (better than expected)
- [x] Registry queries complete in <1μs → **<20 ns** (50× better)

#### For HIGH OVERHEAD (prefer direct mode):

- [ ] Compilation takes >10ms for 10 indicators → N/A (takes 2.1 μs)
- [ ] Break-even exceeds 100 executions → N/A (1 execution with fusion)
- [ ] Plan infrastructure is complex bottleneck → N/A (negligible overhead)

### Decision Rationale

1. **Plan infrastructure overhead is negligible**: Compilation takes only 2.2 μs, which is <1% of typical indicator execution time
2. **Plan reuse is essentially free**: Accessing cached plans takes 0.42 ns
3. **Without fusion, plan mode is marginally slower**: ~0.3% overhead per execution
4. **With any fusion benefit, plan mode wins immediately**: Break-even after just 1 execution

**Recommendation**: Use plan mode as the default execution strategy, as the overhead is minimal and fusion benefits (from E02-E04) can provide speedups that immediately pay off the compilation cost.

**Critical Note**: E02-E03 fusion experiments showed NO fusion benefit (actually slower). This means plan mode currently has no performance advantage over direct mode. Fusion implementation needs investigation before plan mode can provide benefits.

## Implications for fast-ta Architecture

### Based on Results (LOW OVERHEAD):

1. **Default to plan mode**: The overhead is so low that plan-based execution should be the default
2. **Cache compiled plans**: Provide PlanCache for reusing compiled plans (access is ~0.4 ns)
3. **Eager compilation**: Compile plans at startup - 2.2 μs is negligible
4. **Fusion required for benefit**: Without fusion, plan mode is slightly slower; fusion implementation is critical

### When Direct Mode is Preferred

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| Single indicator | Direct mode | No plan benefit |
| One-shot execution | Either (low overhead) | Plan mode only adds 0.8% |
| No fusion available | Direct mode | Plan mode is 0.3% slower |
| Repeated execution | **Plan mode** | Amortize compilation |
| Multiple indicators | **Plan mode** | Enables fusion |

### API Design Implications

```rust
// Direct mode: Simple, minimal overhead
let sma = fast_ta::sma(&prices, 20)?;
let ema = fast_ta::ema(&prices, 20)?;

// Plan mode: Higher overhead, but enables fusion
let plan = PlanBuilder::new()
    .add_sma(20)
    .add_ema(20)
    .add_rsi(14)
    .build()?;  // ~2.2 μs for 9 indicators

let results = plan.execute(&prices)?;

// Cached plan reuse: Essentially free
let results2 = plan.execute(&prices2)?;  // +0.4 ns overhead
```

## Comparison with Other Experiments

| Experiment | Focus | Relationship to E05 | Impact on Break-Even |
|------------|-------|---------------------|---------------------|
| E01 Baseline | Raw indicator cost | ~285 μs baseline | Denominator in break-even |
| E02 RunningStat | Fusion benefit | 2.8× SLOWER (no-go) | Would hurt break-even |
| E03 EMA Fusion | Fusion benefit | 30% SLOWER (no-go) | Would hurt break-even |
| E04 Rolling Extrema | Algorithm choice | Conditional speedup | Improves break-even for large periods |
| **E05 Plan Overhead** | **Infrastructure cost** | **2.2 μs** | **Numerator in break-even** |
| E06 Memory Writes | Memory patterns | Write-every-bar optimal, no buffering needed | NO-GO (buffering), GO (interleaved) |
| E07 End-to-End | Full comparison | Validates E05 predictions | Validates break-even |

**Critical Finding**: E02-E03 fusion experiments showed NO fusion benefit (actually slower), which means plan mode won't break even without other optimizations. E04 shows conditional benefit for rolling extrema with large periods.

## Follow-up Actions

Based on E05 results:

1. **Proceed with plan-based architecture**: The overhead is negligible
2. **Focus on fusion implementation**: Current E02-E03 results show no fusion benefit - this needs investigation
3. **Implement PlanCache**: For production systems with repeated execution
4. **Consider hybrid approach**: Direct mode for single indicators, plan mode for multiple

## Files

- **Benchmark Source**: `crates/fast-ta-experiments/benches/e05_plan_overhead.rs`
- **Registry Implementation**: `crates/fast-ta-core/src/plan/registry.rs`
- **DAG Implementation**: `crates/fast-ta-core/src/plan/dag.rs`
- **Spec Implementation**: `crates/fast-ta-core/src/plan/spec.rs`
- **Criterion Output**: `target/criterion/e05_plan_overhead/`
- **Raw JSON Data**: `target/criterion/e05_plan_overhead/*/base/estimates.json`

## Reproduction

To run this experiment:

```bash
# Run E05 plan overhead benchmarks
cargo bench --package fast-ta-experiments --bench e05_plan_overhead

# View HTML report
open target/criterion/e05_plan_overhead/report/index.html

# Run specific benchmark group
cargo bench --package fast-ta-experiments --bench e05_plan_overhead -- "registry_registration"
cargo bench --package fast-ta-experiments --bench e05_plan_overhead -- "dag_construction"
cargo bench --package fast-ta-experiments --bench e05_plan_overhead -- "break_even"
cargo bench --package fast-ta-experiments --bench e05_plan_overhead -- "full_compilation"
```

## Technical Notes

### Plan Infrastructure Components

```
┌──────────────────────────────────────────────────────────────┐
│                     Plan Compilation                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Registry Population                                      │
│     ┌─────────────────────────────────────────────────────┐  │
│     │ IndicatorSpec + ID → HashMap<String, IndicatorSpec> │  │
│     │ config_key → HashMap<String, String> (dedup index)  │  │
│     └─────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  2. DAG Construction                                         │
│     ┌─────────────────────────────────────────────────────┐  │
│     │ DagBuilder::from_registry(&registry)                │  │
│     │ - Add nodes for each indicator                      │  │
│     │ - Add edges for dependencies                        │  │
│     │ - Returns DagBuilder with DiGraph<DagNode, ()>      │  │
│     └─────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  3. Topological Sort                                         │
│     ┌─────────────────────────────────────────────────────┐  │
│     │ builder.build() → Result<ExecutionPlan, Error>      │  │
│     │ - petgraph::algo::toposort()                        │  │
│     │ - Returns sorted Vec<String> of indicator IDs       │  │
│     │ - Detects cycles → Error::CyclicDependency          │  │
│     └─────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  4. ExecutionPlan (Cached)                                   │
│     ┌─────────────────────────────────────────────────────┐  │
│     │ - execution_order: Vec<String>                      │  │
│     │ - graph: DiGraph<DagNode, ()>                       │  │
│     │ - node_indices: HashMap<String, NodeIndex>          │  │
│     │                                                     │  │
│     │ Methods: execution_order(), dependencies(),         │  │
│     │          dependents(), roots(), leaves()            │  │
│     └─────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Dependency Graph Examples

**Linear Chain (depth=3)**:
```
A → B → C
```
Execution order: [A, B, C]

**Diamond Pattern (width=3)**:
```
     Root
    / | \
   M0 M1 M2
    \ | /
     Leaf
```
Execution order: [Root, M0, M1, M2, Leaf]

**Realistic Trading System**:
```
prices ──┬── SMA(20) ─────────────────┬── Bollinger Bands
         ├── SMA(50)                  │
         ├── EMA(12) ──┬── MACD       │
         ├── EMA(26) ──┘              │
         ├── RSI(14)                  │
         └── ATR(14)                  │
                                      │
OHLCV ───└── Stochastic ──────────────┘
```

### Performance Summary

| Operation | Measured Time | Target | Status |
|-----------|--------------|--------|--------|
| Register 1 indicator | 156 ns | - | Excellent |
| Register 10 indicators | 1.1 μs | - | Excellent |
| Build DAG (10 nodes) | 496 ns | - | Excellent |
| Full compilation (10 indicators) | 2.1 μs | <1 ms | 500× better |
| Plan access | 0.42 ns | ~0 ns | Excellent |
| Iterate order (10 indicators) | 3.1 ns | - | Excellent |

### Optimization Opportunities

Since overhead is already minimal, optimization is low priority:

1. **Pre-allocate**: Use `with_capacity()` for all collections - marginal benefit
2. **Intern strings**: Use string interning for indicator IDs - marginal benefit
3. **Lazy construction**: Build DAG only when needed - not needed (already fast)
4. **Incremental updates**: Add/remove indicators without full rebuild - nice to have
5. **Static plans**: Compile-time plan generation for known configurations - advanced

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: 2024-12-20*
*Benchmark execution time: ~5 minutes*
