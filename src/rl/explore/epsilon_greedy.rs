use rand::Rng;

/// Epsilon-greedy exploration strategy.
pub struct EpsilonGreedy {
    epsilon: f64,
    min_epsilon: f64,
    decay: f64,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f64, min_epsilon: f64, decay: f64) -> Self {
        Self {
            epsilon,
            min_epsilon,
            decay,
        }
    }

    /// Returns true if the agent should explore (take a random action).
    pub fn should_explore(&self) -> bool {
        rand::thread_rng().gen::<f64>() < self.epsilon
    }

    /// Select a random action from `n_actions`.
    pub fn random_action(&self, n_actions: usize) -> usize {
        rand::thread_rng().gen_range(0..n_actions)
    }

    /// Decay epsilon by the configured rate, clamped to min_epsilon.
    pub fn step(&mut self) {
        self.epsilon = (self.epsilon * self.decay).max(self.min_epsilon);
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay() {
        let mut eg = EpsilonGreedy::new(1.0, 0.01, 0.99);
        eg.step();
        assert!((eg.epsilon() - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_min_clamp() {
        let mut eg = EpsilonGreedy::new(0.02, 0.01, 0.1);
        eg.step();
        assert!((eg.epsilon() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_random_action_in_range() {
        let eg = EpsilonGreedy::new(1.0, 0.0, 1.0);
        for _ in 0..100 {
            let a = eg.random_action(5);
            assert!(a < 5);
        }
    }
}
