# algos

Reinforcement learning algorithms in Rust.

## What this is

A focused RL library providing composable, tested implementations of
core RL algorithms — from tabular methods to deep policy gradient and
planning algorithms. Designed for Rust-native environments: game AI,
systems optimization, robotics simulation, and latency-sensitive control.

## Algorithms

### Tabular
- **Q-Learning** — off-policy TD control
- **SARSA** — on-policy TD control
- **Double Q-Learning** — bias-reduced Q-learning with dual tables
- **Monte Carlo ES** — first-visit Monte Carlo with exploring starts

### Value-Based (Deep)
- **DQN** — Deep Q-Network with experience replay and target network

### Policy Gradient
- **REINFORCE** — Monte Carlo policy gradient with value baseline
- **PPO** — Proximal Policy Optimization with clipped surrogate objective
- **TRPO** — Trust Region Policy Optimization with conjugate gradient

### Actor-Critic
- **Actor-Critic** — online actor-critic with TD(0) advantage

### Planning
- **AlphaZero MCTS** — Monte Carlo Tree Search with dual-head network

## Architecture

The library owns the algorithm layer. Users provide:
1. An **Environment** (`step`, `reset`, observation/action spaces)
2. Optionally, custom **function approximators** via `Policy` and `ValueFunction` traits

Default ndarray-based approximators are built into each algorithm.

```
User: Environment  →  Library: Algorithm  →  User: Function Approximator
```

See [docs/architecture.md](docs/architecture.md) for details.

## Infrastructure

- **Replay buffer** — uniform sampling for off-policy methods
- **Rollout buffer** — sequential storage for on-policy methods
- **Epsilon-greedy** — configurable exploration with decay

## Usage

```toml
[dependencies]
algos = "0.7"
```

```rust
use algos::rl::tabular::QLearning;

let mut agent = QLearning::new(
    0.1,   // learning rate
    0.99,  // discount factor
    0.1,   // epsilon
    4,     // number of actions
);

let action = agent.select_action(state);
agent.update(state, action, reward, next_state, done);
```

## Roadmap

- **Wave 2**: SAC, TD3, A2C, IQN, IQL, PER, HER, RND
- **Wave 3**: GRPO, PPO-RLHF, DPO, REINFORCE++

## License

BSD 3-Clause. See [LICENSE](LICENSE).
