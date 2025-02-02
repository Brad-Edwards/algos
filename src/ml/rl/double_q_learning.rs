use rand::Rng;
use std::collections::HashMap;

/// Double Q-Learning implementation to reduce overestimation bias
pub struct DoubleQLearning {
    q1_table: HashMap<(usize, usize), f64>,
    q2_table: HashMap<(usize, usize), f64>,
    learning_rate: f64,
    gamma: f64,
    epsilon: f64,
    n_actions: usize,
}

impl DoubleQLearning {
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, n_actions: usize) -> Self {
        DoubleQLearning {
            q1_table: HashMap::new(),
            q2_table: HashMap::new(),
            learning_rate,
            gamma,
            epsilon,
            n_actions,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();

        // Epsilon-greedy action selection using average of Q1 and Q2
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
        done: bool,
    ) {
        let mut rng = rand::thread_rng();

        // Randomly update either Q1 or Q2
        if rng.gen_bool(0.5) {
            self.update_q1(state, action, reward, next_state, done);
        } else {
            self.update_q2(state, action, reward, next_state, done);
        }
    }

    fn update_q1(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        done: bool,
    ) {
        let current_q = self.get_q1_value(state, action);

        let next_best_action = self.get_best_action_q1(next_state);
        let next_q = if done {
            0.0
        } else {
            self.get_q2_value(next_state, next_best_action)
        };

        let new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q);

        self.q1_table.insert((state, action), new_q);
    }

    fn update_q2(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        done: bool,
    ) {
        let current_q = self.get_q2_value(state, action);

        let next_best_action = self.get_best_action_q2(next_state);
        let next_q = if done {
            0.0
        } else {
            self.get_q1_value(next_state, next_best_action)
        };

        let new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q);

        self.q2_table.insert((state, action), new_q);
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    fn get_q1_value(&self, state: usize, action: usize) -> f64 {
        *self.q1_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn get_q2_value(&self, state: usize, action: usize) -> f64 {
        *self.q2_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn get_average_q_value(&self, state: usize, action: usize) -> f64 {
        (self.get_q1_value(state, action) + self.get_q2_value(state, action)) / 2.0
    }

    fn get_best_action(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut max_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q_value = self.get_average_q_value(state, action);
            if q_value > max_q {
                max_q = q_value;
                best_action = action;
            }
        }

        best_action
    }

    fn get_best_action_q1(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut max_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q_value = self.get_q1_value(state, action);
            if q_value > max_q {
                max_q = q_value;
                best_action = action;
            }
        }

        best_action
    }

    fn get_best_action_q2(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut max_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q_value = self.get_q2_value(state, action);
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
            .q1_table
            .keys()
            .chain(self.q2_table.keys())
            .map(|(s, _)| *s)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for state in states {
            policy.insert(state, self.get_best_action(state));
        }

        policy
    }
}
