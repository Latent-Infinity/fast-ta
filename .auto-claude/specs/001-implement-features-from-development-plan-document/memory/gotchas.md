# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-20 21:01]
The `cargo` command is not in the allowed commands for this project. Verification steps requiring cargo build/test/bench cannot be run directly.

_Context: Discovered during subtask-0-1 when attempting to run `cargo build --workspace` for verification. Files must be created correctly without direct verification, or verification must be done externally._

## [2025-12-20 21:49]
The `cargo` command is not in the allowed commands for this project. Verification steps requiring cargo build/test/bench cannot be run directly.

_Context: Discovered during subtask-1-7 when attempting to run `cargo test --package fast-ta-core -- stochastic` for verification. Files must be created correctly without direct verification, or verification must be done externally._

## [2025-12-20 22:41]
The `cargo` command is not in the allowed commands for this project. Verification steps requiring cargo build/test/bench cannot be run directly.

_Context: Discovered during subtask-3-3 when attempting to run `cargo bench --package fast-ta-experiments --bench e05_plan_overhead --no-run` for verification. Files must be created correctly without direct verification, or verification must be done externally._
