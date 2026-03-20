use rand::Rng;
use std::collections::HashMap;

/// Double Q-Learning to reduce overestimation bias.
///
/// Maintains two Q-tables and randomly selects which to update.
/// Uses one table's argmax to index into the other's values,
/// decoupling action selection from evaluation.
pub struct DoubleQLearning {
    q1: HashMap<(usize, usize), f64>,
    q2: HashMap<(usize, usize), f64>,
    learning_rate: f64,
    gamma: f64,
    epsilon: f64,
    n_actions: usize,
}

impl DoubleQLearning {
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, n_actions: usize) -> Self {
        Self {
            q1: HashMap::new(),
            q2: HashMap::new(),
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
            self.best_action_avg(state)
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
        let mut rng = rand::thread_rng();
        if rng.gen_bool(0.5) {
            let best_a = self.best_action_from(&self.q1, next_state);
            let next_q = if done {
                0.0
            } else {
                self.q2_value(next_state, best_a)
            };
            let current = self.q1_value(state, action);
            let new_q = current + self.learning_rate * (reward + self.gamma * next_q - current);
            self.q1.insert((state, action), new_q);
        } else {
            let best_a = self.best_action_from(&self.q2, next_state);
            let next_q = if done {
                0.0
            } else {
                self.q1_value(next_state, best_a)
            };
            let current = self.q2_value(state, action);
            let new_q = current + self.learning_rate * (reward + self.gamma * next_q - current);
            self.q2.insert((state, action), new_q);
        }
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    pub fn policy(&self) -> HashMap<usize, usize> {
        let states: std::collections::HashSet<usize> = self
            .q1
            .keys()
            .chain(self.q2.keys())
            .map(|(s, _)| *s)
            .collect();
        states
            .into_iter()
            .map(|s| (s, self.best_action_avg(s)))
            .collect()
    }

    fn q1_value(&self, state: usize, action: usize) -> f64 {
        *self.q1.get(&(state, action)).unwrap_or(&0.0)
    }

    fn q2_value(&self, state: usize, action: usize) -> f64 {
        *self.q2.get(&(state, action)).unwrap_or(&0.0)
    }

    fn best_action_avg(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                let avg_a = (self.q1_value(state, a) + self.q2_value(state, a)) / 2.0;
                let avg_b = (self.q1_value(state, b) + self.q2_value(state, b)) / 2.0;
                avg_a.partial_cmp(&avg_b).unwrap()
            })
            .unwrap_or(0)
    }

    fn best_action_from(&self, table: &HashMap<(usize, usize), f64>, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                let va = table.get(&(state, a)).unwrap_or(&0.0);
                let vb = table.get(&(state, b)).unwrap_or(&0.0);
                va.partial_cmp(vb).unwrap()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_q_update() {
        let mut agent = DoubleQLearning::new(0.1, 0.99, 0.0, 4);
        // Run enough updates to populate both tables
        for _ in 0..20 {
            agent.update(0, 1, 1.0, 1, false);
        }
        // At least one table should have a positive value
        let v1 = agent.q1_value(0, 1);
        let v2 = agent.q2_value(0, 1);
        assert!(v1 > 0.0 || v2 > 0.0);
    }
}
