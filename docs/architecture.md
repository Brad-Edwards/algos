# Architecture

## Layer Model

The library owns the algorithm layer. The layers above and below it are user-provided via traits.

```
┌──────────────────────────────────────────┐
│  User: Environment                       │
│  step(), reset(), observation/action     │
│  spaces, domain-specific simulation      │
├──────────────────────────────────────────┤
│  Library: Algorithm                      │
│  Update rules, loss computation,         │
│  experience buffers, exploration,        │
│  advantage estimation                    │
├──────────────────────────────────────────┤
│  User: Function Approximator             │
│  forward(), backward(), any backend      │
│  (ndarray built-ins provided)            │
└──────────────────────────────────────────┘
```

## Module Structure

```
src/ml/rl/
├── infra/           # traits, buffers, normalizers, exploration strategies
├── tabular/         # Q-Learning, SARSA, Double-Q, Monte Carlo ES
├── value/           # DQN, IQN, distributional RL
├── policy/          # REINFORCE, A2C, PPO, TRPO
├── actor_critic/    # Actor-Critic, SAC, TD3
├── offline/         # IQL
└── planning/        # MCTS, AlphaZero-style
```

## Data Flow

```
Environment
  └─ (obs, reward, done) ──→ Algorithm
                              ├──→ ReplayBuffer | RolloutBuffer
                              ├──→ Policy / ValueFunction (via trait)
                              │     └─ loss, gradients
                              └──→ action ──→ Environment
```

## Scope

**Library owns:**
- Algorithm implementations — update rules and loss computation
- Experience buffers — standard replay, prioritized (PER), rollout, hindsight (HER)
- Exploration — ε-greedy, noise processes, intrinsic reward (RND)
- Advantage estimation — Monte Carlo, n-step TD, GAE, group-relative
- Shared utilities — KL divergence, reward normalization, observation normalization

**Library does not own:**
- Neural network architecture or parameter storage
- GPU / hardware execution
- Environment simulation

## Trait Interfaces

| Trait | Responsibility |
|-------|---------------|
| `Environment` | `step`, `reset`, observation space, action space |
| `Policy` | maps observation → action distribution (sample + log probs) |
| `ValueFunction` | maps (observation, optional action) → scalar |
| `ReplayBuffer` | `push`, `sample`, `len`, `capacity` |

Default ndarray-based implementations of `Policy` and `ValueFunction` are provided. Users may supply their own (e.g. candle, burn).

## Algorithm Waves

| Wave | Content |
|------|---------|
| 1 | Infrastructure (traits, buffers, normalizers, exploration) + audit/stabilize existing tabular and deep RL |
| 2 | SAC, TD3, A2C, IQN, IQL (offline), PER, HER, RND; KL divergence and reward normalization utilities |
| 3 | GRPO, PPO-RLHF, DPO, REINFORCE++ |
