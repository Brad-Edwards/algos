use rand::Rng;
use std::collections::HashMap;

/// Tabular Q-Learning.
///
/// Off-policy TD control that learns the optimal action-value function
/// directly using the Bellman optimality equation:
/// Q(s,a) <- Q(s,a) + α [R + γ max_a' Q(s',a') - Q(s,a)]
pub struct QLearning {
    q_table: HashMap<(usize, usize), f64>,
    learning_rate: f64,
    gamma: f64,
    epsilon: f64,
    n_actions: usize,
}

impl QLearning {
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, n_actions: usize) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            gamma,
            epsilon,
            n_actions,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.n_actions)
        } else {
            self.best_action(state)
        }
    }

    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        done: bool,
    ) {
        let current_q = self.q_value(state, action);
        let next_max_q = if done {
            0.0
        } else {
            self.max_q_value(next_state)
        };
        let new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q);
        self.q_table.insert((state, action), new_q);
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    pub fn q_value(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    pub fn policy(&self) -> HashMap<usize, usize> {
        let states: std::collections::HashSet<usize> =
            self.q_table.keys().map(|(s, _)| *s).collect();
        states
            .into_iter()
            .map(|s| (s, self.best_action(s)))
            .collect()
    }

    fn max_q_value(&self, state: usize) -> f64 {
        (0..self.n_actions)
            .map(|a| self.q_value(state, a))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }

    fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                self.q_value(state, a)
                    .partial_cmp(&self.q_value(state, b))
                    .unwrap()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_increases_q_for_positive_reward() {
        let mut agent = QLearning::new(0.1, 0.99, 0.0, 4);
        agent.update(0, 1, 1.0, 1, false);
        assert!(agent.q_value(0, 1) > 0.0);
    }

    #[test]
    fn test_policy_reflects_learned_values() {
        let mut agent = QLearning::new(0.5, 0.99, 0.0, 2);
        // Teach that action 1 in state 0 is better
        for _ in 0..10 {
            agent.update(0, 1, 1.0, 0, true);
            agent.update(0, 0, -1.0, 0, true);
        }
        assert_eq!(*agent.policy().get(&0).unwrap(), 1);
    }

    #[test]
    fn test_epsilon_decay() {
        let mut agent = QLearning::new(0.1, 0.99, 1.0, 4);
        agent.decay_epsilon(0.5);
        assert!((agent.epsilon - 0.5).abs() < 1e-10);
    }
}
