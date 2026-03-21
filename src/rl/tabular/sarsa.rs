use rand::Rng;
use std::collections::HashMap;

/// Tabular SARSA (State-Action-Reward-State-Action).
///
/// On-policy TD control that updates Q-values using the action actually
/// taken in the next state:
/// Q(s,a) <- Q(s,a) + α [R + γ Q(s',a') - Q(s,a)]
pub struct Sarsa {
    q_table: HashMap<(usize, usize), f64>,
    learning_rate: f64,
    gamma: f64,
    epsilon: f64,
    n_actions: usize,
}

impl Sarsa {
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
        next_action: usize,
        done: bool,
    ) {
        let current_q = self.q_value(state, action);
        let next_q = if done {
            0.0
        } else {
            self.q_value(next_state, next_action)
        };
        let new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q);
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
    fn test_on_policy_update() {
        let mut agent = Sarsa::new(0.1, 0.99, 0.0, 4);
        agent.update(0, 1, 1.0, 1, 2, false);
        assert!(agent.q_value(0, 1) > 0.0);
    }

    #[test]
    fn test_terminal_update() {
        let mut agent = Sarsa::new(0.5, 0.99, 0.0, 2);
        agent.update(0, 0, 5.0, 1, 0, true);
        assert!((agent.q_value(0, 0) - 2.5).abs() < 1e-10);
    }
}
