# ADR-0004: Algorithm Delivery in Waves

**Status:** Accepted

## Context

Adding algorithms on top of the current codebase before the infrastructure is stabilized produces algorithms that don't compose, can't share buffers, and can't be tested in isolation. Ordering matters.

## Decision

Delivery is structured in three waves with a strict dependency: Wave N+1 does not begin until Wave N is complete.

**Wave 1 — Infrastructure**
- Define and implement `Environment`, `Policy`, `ValueFunction`, `ReplayBuffer` traits (ADR-0003)
- Standard replay buffer, rollout buffer
- Observation and reward normalization utilities
- Refactor existing algorithms (Q-Learning, SARSA, DQN, PPO, Actor-Critic, TRPO, AlphaZero MCTS) to use the traits
- Audit existing implementations for correctness

**Wave 2 — Algorithm Expansion**
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic, synchronous)
- IQN (Implicit Quantile Networks)
- IQL (Implicit Q-Learning, offline)
- PER (Prioritized Experience Replay — `ReplayBuffer` impl)
- HER (Hindsight Experience Replay — `ReplayBuffer` impl)
- RND (Random Network Distillation — exploration module)
- KL divergence utility and reference policy abstraction (shared with Wave 3)
- Reward normalization (group-level and running statistics)

**Wave 3 — LLM RL**
- GRPO (Group Relative Policy Optimization)
- PPO-RLHF (PPO with reward model and KL penalty against reference)
- DPO (Direct Preference Optimization)
- REINFORCE++

Wave 3 algorithms depend on the KL divergence utility and reference policy abstraction delivered in Wave 2.

## Consequences

- No new algorithm code merges until the Wave 1 trait refactor is complete.
- Wave 2 and Wave 3 algorithms are not designed or scoped until Wave 1 is stable.
- KL divergence and reference policy land in Wave 2 because they are infrastructure shared by Wave 3; they are not LLM-specific.
