use ndarray::Array1;

/// Gradient update for a policy network.
pub struct PolicyGradient {
    pub params: Array1<f64>,
}

/// Trait for policy function approximators.
///
/// A policy maps observations to action distributions. Implement this trait
/// to provide custom neural network backends (e.g. candle, burn) or to use
/// the built-in ndarray-based defaults.
pub trait Policy {
    /// Return log-probabilities over actions for the given observation.
    fn log_probs(&self, obs: &Array1<f64>) -> Array1<f64>;

    /// Sample an action and return (action, log_prob).
    fn sample(&self, obs: &Array1<f64>) -> (usize, f64);

    /// Apply a gradient update to the policy parameters.
    fn update(&mut self, grads: &PolicyGradient);
}
