# ADR-0005: Wave 2 Algorithm Selection

**Status:** Accepted

## Context

Wave 2 expands beyond the existing algorithm set. Selection criteria: coverage of the four target use cases (ADR-0001), no redundancy with existing algorithms, and dependency on Wave 1 infrastructure being in place.

## Decision

**Algorithms added:**

| Algorithm | Rationale |
|-----------|-----------|
| SAC | Standard continuous control algorithm; covers robotics target use case; entropy regularization improves stability |
| TD3 | Standard continuous control baseline alongside SAC; simpler critic structure; both are needed as they suit different settings |
| A2C | Synchronous advantage actor-critic; fills the gap between Actor-Critic and PPO; simpler than PPO and faster to converge on some tasks |
| IQN | Distributional RL; models return distribution rather than expectation; improves performance in stochastic environments |
| IQL | Offline RL; learn from logged data without live environment interaction; high priority for systems/infra target use case |
| PER | Prioritized Experience Replay; implemented as a `ReplayBuffer` trait impl; used by DQN, SAC, TD3, IQN |
| HER | Hindsight Experience Replay; implemented as a `ReplayBuffer` trait impl; required for goal-conditioned tasks in robotics |
| RND | Random Network Distillation; intrinsic motivation for sparse-reward environments; implemented as an exploration module, not a standalone algorithm |

**Algorithms not included in Wave 2:**

| Algorithm | Reason |
|-----------|--------|
| DDPG | TD3 supersedes it; adding both provides no coverage benefit |
| CQL | IQL covers the offline RL use case with lower implementation complexity |
| Rainbow DQN | PER and the existing DQN cover the practical gains; full Rainbow adds complexity for marginal benefit |
| MADDPG / QMIX | Multi-agent is a valid target but adds significant interface complexity; deferred |

## Consequences

- PER and HER are buffer implementations, not standalone algorithms; they must implement `ReplayBuffer` from ADR-0003.
- RND is an exploration module that wraps any algorithm; the module interface must be defined in Wave 1 or early Wave 2.
- SAC and TD3 both require continuous action spaces; `SpaceSpec` must support continuous spaces before either can be implemented.
- IQL requires a static dataset interface in addition to `ReplayBuffer`; this interface must be defined.
