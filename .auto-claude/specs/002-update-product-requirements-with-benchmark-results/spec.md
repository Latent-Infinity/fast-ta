# Specification: Update Product Requirements with Benchmark Results

## Overview

This task involves executing the complete benchmark suite (E01-E07 experiments) for the fast-ta technical analysis library, populating the experiment reports with actual performance data, and updating the product requirements document with validated hypothesis results. The benchmarks are fully implemented but have never been executed, leaving all performance claims as "PENDING VALIDATION" with "TBD" values throughout the documentation.

## Workflow Type

**Type**: feature

**Rationale**: While the code infrastructure exists, this task requires executing benchmarks, collecting data, analyzing results, and updating documentation - a multi-step feature completion workflow that produces concrete deliverables (benchmark data and updated documentation).

## Task Scope

### Services Involved
- **main** (primary) - Rust workspace containing fast-ta-core and fast-ta-experiments crates

### This Task Will:
- [ ] Execute all 7 benchmark experiments (E01-E07) using Criterion
- [ ] Populate benchmark results in all experiment REPORT.md files
- [ ] Update docs/experiments/SUMMARY.md with consolidated results
- [ ] Update docs/product-requirements.md hypothesis validation status
- [ ] Record go/no-go decisions based on actual performance data
- [ ] Verify O(n) complexity claims for all indicators

### Out of Scope:
- Implementing new indicators or kernels
- Modifying benchmark code
- Changing the plan infrastructure architecture
- Adding new experiments beyond E01-E07
- Creating CI/CD benchmark automation

## Service Context

### Main (fast-ta Workspace)

**Tech Stack:**
- Language: Rust
- Framework: None (library crate)
- Key directories:
  - `crates/fast-ta-core/` - Core library with indicators, kernels, plan infrastructure
  - `crates/fast-ta-experiments/` - Benchmarking suite with E01-E07
  - `benches/experiments/` - Experiment reports (E01-E07 subdirectories)
  - `docs/` - Documentation including PRD and experiment summary

**Entry Point:** `crates/fast-ta-core/src/lib.rs`

**How to Run Benchmarks:**
```bash
# Run all experiments
cargo bench --workspace

# Run individual experiments
cargo bench --package fast-ta-experiments --bench e01_baseline
cargo bench --package fast-ta-experiments --bench e02_running_stat
cargo bench --package fast-ta-experiments --bench e03_ema_fusion
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema
cargo bench --package fast-ta-experiments --bench e05_plan_overhead
cargo bench --package fast-ta-experiments --bench e06_memory_writes
cargo bench --package fast-ta-experiments --bench e07_end_to_end
```

**Package Manager:** cargo

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `benches/experiments/E01_baseline/REPORT.md` | main | Replace all TBD values with actual benchmark results |
| `benches/experiments/E02_running_stat/REPORT.md` | main | Replace TBD with Welford's fusion benchmark results |
| `benches/experiments/E03_ema_fusion/REPORT.md` | main | Replace TBD with multi-EMA fusion benchmark results |
| `benches/experiments/E04_rolling_extrema/REPORT.md` | main | Replace TBD with deque vs naive benchmark results |
| `benches/experiments/E05_plan_overhead/REPORT.md` | main | Replace TBD with plan compilation overhead results |
| `benches/experiments/E06_memory_writes/REPORT.md` | main | Replace TBD with memory write pattern results |
| `benches/experiments/E07_end_to_end/REPORT.md` | main | Replace TBD with direct vs plan mode comparison |
| `docs/experiments/SUMMARY.md` | main | Update executive summary table with actual results/decisions |
| `docs/product-requirements.md` | main | Update hypothesis validation status (Section 1.3) from PENDING to VALIDATED/INVALIDATED |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `docs/development-plan.md` | Task structure and acceptance criteria format |
| `docs/experiments/SUMMARY.md` | Result table format and decision criteria |
| `benches/experiments/E01_baseline/REPORT.md` | Report structure with results tables |
| `crates/fast-ta-experiments/benches/e01_baseline.rs` | Benchmark configuration and data sizes |

## Patterns to Follow

### Result Table Format

From `benches/experiments/E01_baseline/REPORT.md`:

```markdown
| Indicator | 1K (ns/op) | 10K (ns/op) | 100K (ns/op) | ns/element @ 100K |
|-----------|------------|-------------|--------------|-------------------|
| SMA | 1,234 | 12,345 | 123,456 | 1.23 |
```

**Key Points:**
- Use consistent units (ns/op for time, ns/element for throughput)
- Include data sizes: 1K, 10K, 100K (1M for extended tests)
- Calculate per-element cost for complexity verification

### Go/No-Go Decision Format

From `docs/experiments/SUMMARY.md`:

```markdown
| Decision | Condition | Speedup Observed | Recommendation |
|----------|-----------|------------------|----------------|
| **GO** | ≥20% speedup | 25% | Adopt fused kernels |
```

**Key Points:**
- Reference target speedup from development plan
- Compare actual vs expected
- Provide clear recommendation

### Hypothesis Validation Format

From `docs/product-requirements.md`:

```markdown
| ID | Hypothesis | Target | Experiment | Status |
|----|------------|--------|------------|--------|
| H1 | All indicators O(n) | Linear scaling | E01 | **VALIDATED** ✓ |
```

**Key Points:**
- Change PENDING VALIDATION to VALIDATED or INVALIDATED
- Add checkmark (✓) or X for visual clarity
- Reference specific benchmark data

## Requirements

### Functional Requirements

1. **Execute Complete Benchmark Suite**
   - Description: Run all E01-E07 benchmarks using Criterion
   - Acceptance: All benchmarks complete without errors, HTML reports generated in target/criterion/

2. **Populate E01 Baseline Results**
   - Description: Record individual indicator performance at 1K, 10K, 100K data points
   - Acceptance: All 7 indicators have timing data; O(n) complexity verified via 10×/100× scaling ratios

3. **Populate E02-E04 Kernel Results**
   - Description: Record fusion kernel performance vs unfused alternatives
   - Acceptance: Speedup percentages calculated; go/no-go decisions recorded

4. **Populate E05-E06 Infrastructure Results**
   - Description: Record plan overhead and memory write pattern results
   - Acceptance: Break-even point calculated; write pattern recommendations made

5. **Populate E07 End-to-End Results**
   - Description: Compare direct mode vs plan mode for full workloads
   - Acceptance: Speedup measured for 7/14/21/28 indicators; final architecture recommendation made

6. **Update Product Requirements Document**
   - Description: Mark each hypothesis as validated or invalidated with supporting data
   - Acceptance: All 7 hypotheses (H1-H7) have final status; version bumped if needed

### Edge Cases

1. **Benchmark Variance** - Run multiple times if results show >20% variance; use Criterion's statistical analysis
2. **Compilation Failures** - Ensure `cargo build --workspace` succeeds before benchmarking
3. **Thermal Throttling** - Run on stable system; discard first run if anomalous
4. **Memory Pressure** - 1M data point tests may require adequate system RAM

## Implementation Notes

### DO
- Run `cargo build --workspace` first to ensure compilation succeeds
- Use `cargo bench --workspace` for complete suite
- Extract timing data from Criterion JSON output in `target/criterion/`
- Use Criterion HTML reports for visual verification
- Round timings to appropriate precision (μs for >1ms, ns otherwise)
- Document hardware specs where benchmarks were run

### DON'T
- Modify benchmark source code
- Cherry-pick favorable results
- Skip experiments - all 7 must be run
- Use release mode without proper optimization flags (Criterion handles this)
- Run benchmarks under heavy system load

## Development Environment

### Start Benchmarks

```bash
# Compile workspace
cargo build --workspace --release

# Run all benchmarks (may take 10-30 minutes)
cargo bench --workspace

# Run quick decision benchmarks only
cargo bench --package fast-ta-experiments --bench e02_running_stat -- "comparison"
cargo bench --package fast-ta-experiments --bench e03_ema_fusion -- "ema_count_scaling"
cargo bench --package fast-ta-experiments --bench e04_rolling_extrema -- "period_scaling"
cargo bench --package fast-ta-experiments --bench e07_end_to_end -- "go_no_go"
```

### Output Locations
- Criterion HTML: `target/criterion/report/index.html`
- Per-experiment: `target/criterion/e0[1-7]_*/report/index.html`
- JSON data: `target/criterion/*/base/estimates.json`

### Required Environment Variables
- None required for benchmarking

## Success Criteria

The task is complete when:

1. [ ] All 7 experiments (E01-E07) have been executed successfully
2. [ ] All REPORT.md files have actual benchmark numbers instead of TBD
3. [ ] docs/experiments/SUMMARY.md has consolidated results with decisions
4. [ ] docs/product-requirements.md Section 1.3 has validated hypothesis status
5. [ ] Go/no-go decisions recorded for each fusion strategy:
   - E02: RunningStat fusion (target: ≥20% speedup)
   - E03: EMA fusion (target: ≥15% speedup for ≥10 EMAs)
   - E04: Rolling extrema (target: ≥5× speedup at k≥50)
   - E05: Plan overhead (target: <100 executions break-even)
   - E06: Memory writes (target: ≥10% improvement)
   - E07: Plan mode (target: ≥1.5× speedup for ≥20 indicators)
6. [ ] No console errors during benchmark execution
7. [ ] Existing tests still pass (`cargo test --workspace`)

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| Workspace builds | All crates | `cargo build --workspace` succeeds |
| Tests pass | All crates | `cargo test --workspace` passes |
| Clippy clean | All crates | `cargo clippy --workspace` no warnings |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| Benchmark execution | fast-ta-experiments | All 7 benchmarks run without panic |
| Result extraction | Criterion output | JSON files contain valid timing data |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Full benchmark suite | 1. `cargo bench --workspace` | All benchmarks complete, HTML reports generated |
| Quick decision benchmarks | 1. Run subset benchmarks | Core comparison data available |

### Documentation Verification
| Check | File | Expected |
|-------|------|----------|
| No TBD values | E01-E07 REPORT.md | All tables have numeric values |
| Hypothesis status | product-requirements.md | All H1-H7 have final status |
| Summary complete | experiments/SUMMARY.md | Executive table has decisions |

### Benchmark Data Verification
| Check | Query/Command | Expected |
|-------|---------------|----------|
| Criterion output exists | `ls target/criterion/` | E01-E07 directories present |
| JSON data valid | `cat target/criterion/e01_baseline/*/base/estimates.json` | Valid JSON with point_estimate |
| O(n) verified | E01 scaling ratios | 10K→100K ratio near 10× |

### QA Sign-off Requirements
- [ ] All benchmarks executed successfully
- [ ] All REPORT.md files populated with actual results
- [ ] All hypotheses have validated/invalidated status
- [ ] Go/no-go decisions documented for each experiment
- [ ] No regressions in existing tests
- [ ] Documentation is internally consistent (same numbers in reports and summary)
- [ ] PRD version updated if significant findings
