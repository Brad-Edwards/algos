# ADR-0003: Trait Interface Design

**Status:** Accepted

## Context

The existing codebase defines network structs inline per algorithm with no shared interface. This makes it impossible to swap function approximators, compose algorithms, or test algorithm logic in isolation from network execution.

Four integration boundaries need formal interfaces: the environment, the policy, the value function, and the experience buffer.

## Decision

Four core traits define the integration boundaries:

**`Environment`**
- `fn step(&mut self, action: &Action) -> (Observation, f64, bool)` — (obs, reward, done)
- `fn reset(&mut self) -> Observation`
- `fn observation_space(&self) -> SpaceSpec`
- `fn action_space(&self) -> SpaceSpec`

**`Policy`**
- `fn log_probs(&self, obs: &Observation) -> Vec<f64>`
- `fn sample(&self, obs: &Observation) -> (Action, f64)` — (action, log_prob)
- `fn update(&mut self, grads: &PolicyGradient)`

**`ValueFunction`**
- `fn value(&self, obs: &Observation) -> f64`
- `fn action_value(&self, obs: &Observation, action: &Action) -> f64`
- `fn update(&mut self, grads: &ValueGradient)`

**`ReplayBuffer`**
- `fn push(&mut self, transition: Transition)`
- `fn sample(&self, n: usize) -> Vec<Transition>`
- `fn len(&self) -> usize`
- `fn capacity(&self) -> usize`

Concrete types for `Action`, `Observation`, `SpaceSpec`, `Transition` are defined in `infra/`. Algorithm implementations depend only on these traits, not on concrete network types.

## Consequences

- All existing algorithm implementations must be refactored to use these traits.
- Algorithm logic can be tested with stub implementations of the traits.
- Users can provide any function approximator backend by implementing `Policy` and `ValueFunction`.
- The `ReplayBuffer` trait allows PER and HER to be drop-in replacements for standard replay.
- `SpaceSpec` must cover at minimum: discrete (n actions), continuous (bounds, shape), and multi-discrete.
