# ADR-0002: Compute Backend

**Status:** Accepted

## Context

RL algorithm implementations require numerical computation (matrix multiply, softmax, gradient updates). Several options exist: ndarray (CPU, pure Rust), candle (Hugging Face, CUDA-capable), burn (modular backends), tch (LibTorch bindings), or no backend dependency at all.

The library currently uses ndarray throughout. The algorithm layer and the function approximator layer have distinct responsibilities and different change rates.

## Decision

The library separates algorithm logic from function approximator execution via traits (`Policy`, `ValueFunction`). The library owns the algorithm layer. The function approximator layer is user-supplied.

- **Default implementations** use ndarray (CPU, no additional dependencies).
- **No CUDA, candle, or burn dependency** in the library itself.
- Users who want GPU execution implement the traits against their chosen backend.

The library does not provide GPU-backed implementations. It does not vendor or wrap any GPU framework.

## Consequences

- No `cuda`, `candle`, or `burn` in `Cargo.toml` for the core library.
- Users running LLM or large-model workloads must implement the `Policy`/`ValueFunction` traits themselves.
- Default ndarray implementations are sufficient for small networks and all tabular methods.
- The algorithm implementations (loss computation, advantage estimation, buffer sampling) remain backend-agnostic and testable without hardware.
- Adding a candle or burn feature-flag backend is possible later without changing the algorithm layer.
