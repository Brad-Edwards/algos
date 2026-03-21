use ndarray::Array1;

/// Gradient update for a value function.
pub struct ValueGradient {
    pub params: Array1<f64>,
}

/// Trait for value function approximators.
///
/// Maps observations (and optionally actions) to scalar values.
/// Implement this trait to provide custom backends.
pub trait ValueFunction {
    /// Estimate the state value V(s).
    fn value(&self, obs: &Array1<f64>) -> f64;

    /// Estimate the action value Q(s, a).
    fn action_value(&self, obs: &Array1<f64>, action: usize) -> f64;

    /// Apply a gradient update to the value function parameters.
    fn update(&mut self, grads: &ValueGradient);
}
