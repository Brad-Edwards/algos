/// Specification of an observation or action space.
#[derive(Debug, Clone)]
pub enum SpaceSpec {
    /// Discrete space with `n` possible values (0..n).
    Discrete(usize),
    /// Continuous space defined by lower and upper bounds per dimension.
    Continuous { low: Vec<f64>, high: Vec<f64> },
    /// Multiple independent discrete spaces.
    MultiDiscrete(Vec<usize>),
}

/// A single environment transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Trait for RL environments.
///
/// Environments produce observations, accept actions, and return rewards.
/// Implement this trait to connect any simulation or game to the RL algorithms.
pub trait Environment {
    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Vec<f64>;

    /// Take an action and return (observation, reward, done).
    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool);

    /// Describe the observation space.
    fn observation_space(&self) -> SpaceSpec;

    /// Describe the action space.
    fn action_space(&self) -> SpaceSpec;
}
