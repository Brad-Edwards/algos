use rand::Rng;
use std::collections::{HashMap, VecDeque};

/// Monte Carlo with Exploring Starts implementation
pub struct MonteCarloES {
    q_table: HashMap<(usize, usize), f64>,
    returns: HashMap<(usize, usize), VecDeque<f64>>,
    policy: HashMap<usize, usize>,
    n_actions: usize,
    epsilon: f64,
}

impl MonteCarloES {
    pub fn new(n_actions: usize, epsilon: f64) -> Self {
        MonteCarloES {
            q_table: HashMap::new(),
            returns: HashMap::new(),
            policy: HashMap::new(),
            n_actions,
            epsilon,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();

        // Epsilon-greedy policy
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.n_actions)
        } else {
            *self.policy.get(&state).unwrap_or(&0)
        }
    }

    pub fn update(&mut self, episode: Vec<(usize, usize, f64)>) {
        let mut g = 0.0;
        let mut visited = HashMap::new();

        // Process episode in reverse order
        for (_t, (state, action, reward)) in episode.iter().enumerate().rev() {
            g += reward; // No discount factor in basic Monte Carlo

            // First-visit Monte Carlo
            if let std::collections::hash_map::Entry::Vacant(e) = visited.entry((*state, *action)) {
                e.insert(true);

                // Update returns
                self.returns
                    .entry((*state, *action))
                    .or_default()
                    .push_back(g);

                // Update Q-value with mean return
                let returns = self.returns.get(&(*state, *action)).unwrap();
                let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                self.q_table.insert((*state, *action), mean_return);

                // Update policy
                self.update_policy(*state);
            }
        }
    }

    fn update_policy(&mut self, state: usize) {
        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let value = self.get_q_value(state, action);
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        self.policy.insert(state, best_action);
    }

    pub fn generate_episode(&self, env: &mut impl Environment) -> Vec<(usize, usize, f64)> {
        let mut episode = Vec::new();
        let mut state = env.reset();
        let mut done = false;

        while !done {
            let action = self.select_action(state);
            let (next_state, reward, is_done) = env.step(action);

            episode.push((state, action, reward));

            if is_done {
                done = true;
            }

            state = next_state;
        }

        episode
    }

    fn get_q_value(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    pub fn get_policy(&self) -> &HashMap<usize, usize> {
        &self.policy
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }
}

/// Environment trait that must be implemented by any environment used with MonteCarloES
pub trait Environment {
    fn reset(&mut self) -> usize;
    fn step(&mut self, action: usize) -> (usize, f64, bool);
}
