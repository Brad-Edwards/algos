//! Reinforcement learning algorithms for Rust.
//!
//! # Architecture
//!
//! The library owns the **algorithm layer**. Users provide the environment
//! (via [`env::Environment`]) and, for deep RL, the function approximator
//! (via [`policy::Policy`] / [`value::ValueFunction`]).
//!
//! ```text
//! ┌──────────────────────────────────┐
//! │  User: Environment              │
//! │  step(), reset(), spaces        │
//! ├──────────────────────────────────┤
//! │  Library: Algorithm             │
//! │  update rules, buffers,         │
//! │  exploration, advantage est.    │
//! ├──────────────────────────────────┤
//! │  User: Function Approximator    │
//! │  (ndarray defaults provided)    │
//! └──────────────────────────────────┘
//! ```
//!
//! # Modules
//!
//! - [`tabular`] — Q-Learning, SARSA, Double Q-Learning, Monte Carlo ES
//! - [`value_based`] — DQN
//! - [`policy_gradient`] — REINFORCE, PPO, TRPO
//! - [`actor_critic`] — Actor-Critic
//! - [`planning`] — AlphaZero-style MCTS
//! - [`buffer`] — Replay and rollout buffers
//! - [`explore`] — Exploration strategies (epsilon-greedy)
//! - [`env`] — Environment trait, space specs, transition types
//! - [`policy`] — Policy trait for function approximators
//! - [`value`] — Value function trait for function approximators

pub mod actor_critic;
pub mod buffer;
pub mod env;
pub mod explore;
pub mod planning;
pub mod policy;
pub mod policy_gradient;
pub mod tabular;
pub mod value;
pub mod value_based;
