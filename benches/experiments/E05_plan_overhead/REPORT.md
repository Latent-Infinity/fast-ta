# E05: Plan Compilation Overhead Benchmarks

## Experiment Overview

**Experiment ID**: E05
**Name**: Plan Compilation Overhead
**Status**: PENDING (awaiting benchmark execution)
**Date**: TBD

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

| Metric | Target | Reasoning |
|--------|--------|-----------|
| **Break-even point** | <100 executions | Plan compilation should amortize quickly |
| **Compilation time** | <1ms for 10 indicators | Negligible vs data processing |
| **Plan reuse overhead** | ~0 ns | Cached plan access should be instant |

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

### Expected Break-Even Points

| Scenario | Fusion Speedup (S) | Expected Break-Even (N) |
|----------|-------------------|------------------------|
| No fusion (S=1.0) | 1.0× | ∞ (never breaks even) |
| Minimal fusion (S=1.1) | 1.1× | TBD executions |
| Moderate fusion (S=1.2) | 1.2× | TBD executions |
| Good fusion (S=1.5) | 1.5× | TBD executions |
| Optimal fusion (S=2.0) | 2.0× | TBD executions |

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

*Results will be populated after running: `cargo bench --package fast-ta-experiments --bench e05_plan_overhead`*

### Registry Registration

| Indicator Count | Simple Registration | Mixed Registration | Per-Indicator Cost |
|----------------|--------------------|--------------------|-------------------|
| 1 | TBD ns | TBD ns | TBD ns |
| 5 | TBD ns | TBD ns | TBD ns |
| 10 | TBD ns | TBD ns | TBD ns |
| 20 | TBD ns | TBD ns | TBD ns |
| 50 | TBD ns | TBD ns | TBD ns |

### DAG Construction

| Structure | Count/Depth | Construction Time |
|-----------|-------------|------------------|
| Independent | 1 | TBD ns |
| Independent | 10 | TBD ns |
| Independent | 50 | TBD ns |
| Linear chain | 3 | TBD ns |
| Linear chain | 10 | TBD ns |
| Diamond | width=5 | TBD ns |
| Diamond | width=20 | TBD ns |

### Full Plan Compilation

| Plan Type | Indicator Count | Compilation Time |
|-----------|----------------|-----------------|
| Simple | 1 | TBD ns |
| Simple | 10 | TBD ns |
| Simple | 50 | TBD ns |
| Mixed | 10 | TBD ns |
| Realistic | 9 | TBD ns |

### Plan Reuse

| Indicator Count | Execution Order Access | Iteration Time |
|----------------|----------------------|----------------|
| 1 | TBD ns | TBD ns |
| 10 | TBD ns | TBD ns |
| 50 | TBD ns | TBD ns |

### Direct vs Plan Comparison (10K data points)

| Mode | Time | Overhead vs Direct |
|------|------|--------------------|
| Direct (7 indicators) | TBD μs | baseline |
| Plan compile only | TBD ns | N/A |
| Plan compile + exec | TBD μs | +TBD% |
| Cached plan exec | TBD μs | ~0% |

### Break-Even Calculation

| Metric | Value |
|--------|-------|
| Plan compilation time | TBD ns |
| Direct execution time (10K, 7 indicators) | TBD μs |
| Cached plan execution time | TBD μs |
| **Break-even (no fusion)** | **TBD executions** |
| **Break-even (20% fusion speedup)** | **TBD executions** |
| **Break-even (50% fusion speedup)** | **TBD executions** |

## Analysis

### Expected Results

Based on algorithm analysis:

1. **Registry registration should be O(1) per indicator**:
   - HashMap insert is amortized O(1)
   - Config key generation is O(1)

2. **DAG construction should be O(V + E)**:
   - V = number of indicators
   - E = number of dependencies (typically O(V) or less)

3. **Topological sort is O(V + E)**:
   - petgraph's toposort is linear

4. **Plan reuse should be nearly free**:
   - Just returning a reference to cached Vec

5. **Break-even should be low with fusion**:
   - 20% fusion speedup → ~5-10 executions
   - 50% fusion speedup → ~2-3 executions

### Complexity Verification

| Operation | Expected Complexity | Notes |
|-----------|---------------------|-------|
| Registry registration | O(1) per indicator | HashMap insert |
| DAG construction | O(V + E) | Graph building |
| Topological sort | O(V + E) | Linear scan |
| Plan access | O(1) | Reference access |

### Memory Overhead

| Component | Size | Notes |
|-----------|------|-------|
| IndicatorSpec | ~128 bytes | Includes Vec for dependencies |
| Registry (N indicators) | ~N × 200 bytes | HashMap overhead |
| DAG (V vertices, E edges) | ~V × 50 + E × 16 bytes | petgraph DiGraph |
| ExecutionPlan | ~V × 50 bytes | Includes order Vec |

## Go/No-Go Decision

**Decision**: PENDING

### Criteria Checklist

#### For LOW OVERHEAD (plan mode recommended):

- [ ] Full compilation for 10 indicators takes <1ms
- [ ] Plan reuse overhead is <100ns
- [ ] Break-even is <10 executions with 20% fusion speedup
- [ ] Registry queries complete in <1μs

#### For HIGH OVERHEAD (prefer direct mode):

- [ ] Compilation takes >10ms for 10 indicators
- [ ] Break-even exceeds 100 executions
- [ ] Plan infrastructure is complex bottleneck

## Implications for fast-ta Architecture

### If LOW OVERHEAD:

1. **Default to plan mode**: Recommend plan-based execution for typical use cases
2. **Cache compiled plans**: Provide PlanCache for reusing compiled plans
3. **Eager compilation**: Compile plans at startup, not per-request
4. **Fusion by default**: Enable kernel fusion in plan execution

### If HIGH OVERHEAD:

1. **Direct mode default**: Use direct calls for small indicator sets
2. **Lazy compilation**: Only compile when explicitly requested
3. **Plan threshold**: Only use plans for ≥N indicators
4. **Simple API**: Hide plan complexity behind simple facade

## Recommendations

### When to Use Plan Mode

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| Single indicator | Direct mode | No overhead benefit |
| 2-5 indicators | Consider direct | Low fusion opportunity |
| 6-20 indicators | **Plan mode** | Good fusion, amortized overhead |
| 20+ indicators | **Plan mode** | Significant fusion potential |
| Repeated execution | **Plan mode** | Cache and reuse plan |
| One-shot execution | Direct mode | No amortization opportunity |

### API Design Implications

```rust
// Direct mode: Simple, low overhead
let sma = fast_ta::sma(&prices, 20)?;
let ema = fast_ta::ema(&prices, 20)?;

// Plan mode: Higher overhead, but enables fusion
let plan = PlanBuilder::new()
    .add_sma(20)
    .add_ema(20)
    .add_rsi(14)
    .build()?;

let results = plan.execute(&prices)?;
```

## Comparison with Other Experiments

| Experiment | Focus | Relationship to E05 |
|------------|-------|---------------------|
| E01 Baseline | Raw indicator cost | Baseline for break-even |
| E02 RunningStat | Fusion benefit | Improves break-even |
| E03 EMA Fusion | Fusion benefit | Improves break-even |
| E04 Rolling Extrema | Algorithm choice | Independent of plan |
| **E05 Plan Overhead** | **Infrastructure cost** | **Break-even calculation** |
| E06 Memory Writes | Memory patterns | May affect plan execution |
| E07 End-to-End | Full comparison | Validates E05 predictions |

## Follow-up Actions

After E05 completes:

1. **If LOW OVERHEAD**:
   - Proceed with plan-based architecture
   - Implement PlanCache for production
   - Document recommended usage patterns

2. **If HIGH OVERHEAD**:
   - Consider simpler alternatives
   - Profile for optimization opportunities
   - Consider hybrid approach

3. **E06 (Memory Writes)**: Next infrastructure experiment

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

### Performance Expectations

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Register 1 indicator | ~50-100 ns | HashMap insert |
| Register 10 indicators | ~500 ns - 1 μs | Linear scaling |
| Build DAG (10 nodes) | ~200-500 ns | Graph construction |
| Topological sort (10 nodes) | ~100-300 ns | Linear algorithm |
| Full compilation (10 indicators) | ~1-5 μs | All steps combined |
| Plan access | ~10-20 ns | Reference return |
| Iterate order (10 indicators) | ~50-100 ns | Simple iteration |

### Optimization Opportunities

If overhead is too high:

1. **Pre-allocate**: Use `with_capacity()` for all collections
2. **Intern strings**: Use string interning for indicator IDs
3. **Lazy construction**: Build DAG only when needed
4. **Incremental updates**: Add/remove indicators without full rebuild
5. **Static plans**: Compile-time plan generation for known configurations

---

*Report generated for fast-ta micro-experiments framework*
*Last updated: Pending benchmark execution*
