# Architecture

## Layer Model

The library owns the algorithm layer. Users provide the environment above
and (optionally) function approximators below.

```
┌──────────────────────────────────┐
│  User: Environment              │
│  step(), reset(), spaces        │
├──────────────────────────────────┤
│  Library: Algorithm             │
│  update rules, loss functions,  │
│  experience buffers, exploration│
├──────────────────────────────────┤
│  User: Function Approximator    │
│  (ndarray defaults provided)    │
└──────────────────────────────────┘
```

## Module Structure

```
src/
├── lib.rs
├── error.rs
└── rl/
    ├── env.rs              # Environment trait, SpaceSpec, Transition
    ├── policy.rs           # Policy trait
    ├── value.rs            # ValueFunction trait
    ├── buffer/             # Replay and rollout buffers
    │   ├── replay.rs
    │   └── rollout.rs
    ├── explore/            # Exploration strategies
    │   └── epsilon_greedy.rs
    ├── tabular/            # Q-Learning, SARSA, Double-Q, Monte Carlo ES
    ├── value_based/        # DQN
    ├── policy_gradient/    # REINFORCE, PPO, TRPO
    ├── actor_critic/       # Actor-Critic
    └── planning/           # AlphaZero-style MCTS
```

## Trait Interfaces

| Trait | Responsibility |
|-------|---------------|
| `Environment` | `step`, `reset`, observation space, action space |
| `Policy` | maps observation → action distribution (sample + log probs) |
| `ValueFunction` | maps observation → scalar value |
| `Buffer` | `push`, `sample`, `len`, `capacity` |

## Algorithm Waves

| Wave | Content |
|------|---------|
| 1 (current) | Trait infrastructure, buffers, exploration, reorganize existing algorithms |
| 2 | SAC, TD3, A2C, IQN, IQL, PER, HER, RND |
| 3 | GRPO, PPO-RLHF, DPO, REINFORCE++ |
