use rand::Rng;
use std::collections::HashMap;

/// SARSA (State-Action-Reward-State-Action) implementation
pub struct SARSA {
    q_table: HashMap<(usize, usize), f64>,
    learning_rate: f64,
    gamma: f64,
    epsilon: f64,
    n_actions: usize,
}

impl SARSA {
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, n_actions: usize) -> Self {
        SARSA {
            q_table: HashMap::new(),
            learning_rate,
            gamma,
            epsilon,
            n_actions,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();

        // Epsilon-greedy action selection
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.n_actions)
        } else {
            self.get_best_action(state)
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
        let current_q = self.get_q_value(state, action);
        let next_q = if done {
            0.0
        } else {
            self.get_q_value(next_state, next_action)
        };

        // SARSA update rule: Q(s,a) = Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        let new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q);

        self.q_table.insert((state, action), new_q);
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    fn get_q_value(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn get_best_action(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut max_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q_value = self.get_q_value(state, action);
            if q_value > max_q {
                max_q = q_value;
                best_action = action;
            }
        }

        best_action
    }

    pub fn get_policy(&self) -> HashMap<usize, usize> {
        let mut policy = HashMap::new();
        let states: Vec<usize> = self
            .q_table
            .keys()
            .map(|(s, _)| *s)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for state in states {
            policy.insert(state, self.get_best_action(state));
        }

        policy
    }

    pub fn get_q_table(&self) -> &HashMap<(usize, usize), f64> {
        &self.q_table
    }
}
