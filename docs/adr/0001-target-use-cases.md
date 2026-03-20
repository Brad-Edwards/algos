# ADR-0001: Target Use Cases

**Status:** Accepted

## Context

RL has a wide range of application domains. Algorithm selection, interface design, and performance tradeoffs depend on which domains the library is designed for. Attempting to serve all domains equally results in a library that serves none well.

## Decision

The library targets four primary use cases:

1. **Game AI / simulation environments in Rust** — environments already implemented in Rust (e.g. Bevy); no FFI overhead; deterministic, high-frequency simulation.

2. **Systems and infrastructure optimization** — the RL environment is a Rust system (query planner, scheduler, compiler, router); the library must integrate without FFI or subprocess overhead.

3. **Robotics — simulation and edge deployment** — continuous control with real-time constraints; no garbage collection pauses; small memory footprint on device.

4. **High-frequency / latency-sensitive domains** — environments where Python interpreter overhead is a liability.

Out of scope as primary targets: large-scale distributed LLM training, academic benchmarking against Python-ecosystem baselines, any workload where the environment is not in Rust.

## Consequences

- Algorithm selection is driven by what is useful in these four domains, not by general RL completeness.
- The library does not need to interop with Python RL environments (Gymnasium, etc.) as a primary concern.
- Offline RL (IQL) is elevated in priority because systems domains commonly have logged data and cannot afford live exploration.
- Continuous control algorithms (SAC, TD3) are elevated in priority for robotics.
- LLM RL algorithms (GRPO, PPO-RLHF) are not a primary target but are included for completeness (Wave 3).
