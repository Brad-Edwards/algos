use rand::Rng;
use std::collections::{HashMap, VecDeque};

use crate::rl::env::Environment;

/// Monte Carlo with Exploring Starts.
///
/// First-visit MC control: collects full episodes, computes returns for
/// each (state, action) pair, and updates Q-values using the mean return.
pub struct MonteCarloES {
    q_table: HashMap<(usize, usize), f64>,
    returns: HashMap<(usize, usize), VecDeque<f64>>,
    learned_policy: HashMap<usize, usize>,
    n_actions: usize,
    epsilon: f64,
}

impl MonteCarloES {
    pub fn new(n_actions: usize, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            returns: HashMap::new(),
            learned_policy: HashMap::new(),
            n_actions,
            epsilon,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.n_actions)
        } else {
            *self.learned_policy.get(&state).unwrap_or(&0)
        }
    }

    /// Update Q-values from a complete episode of (state, action, reward) triples.
    pub fn update(&mut self, episode: &[(usize, usize, f64)]) {
        let mut g = 0.0;
        let mut visited = HashMap::new();

        for &(state, action, reward) in episode.iter().rev() {
            g += reward;
            if let std::collections::hash_map::Entry::Vacant(e) = visited.entry((state, action)) {
                e.insert(true);
                self.returns
                    .entry((state, action))
                    .or_default()
                    .push_back(g);
                let returns = self.returns.get(&(state, action)).unwrap();
                let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                self.q_table.insert((state, action), mean_return);
                self.update_policy(state);
            }
        }
    }

    /// Generate an episode using the current policy on the given environment.
    pub fn generate_episode(&self, env: &mut impl Environment) -> Vec<(usize, usize, f64)> {
        let mut episode = Vec::new();
        let mut obs = env.reset();
        loop {
            let state = obs[0] as usize;
            let action = self.select_action(state);
            let (next_obs, reward, done) = env.step(action);
            episode.push((state, action, reward));
            if done {
                break;
            }
            obs = next_obs;
        }
        episode
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    pub fn policy(&self) -> &HashMap<usize, usize> {
        &self.learned_policy
    }

    pub fn q_value(&self, state: usize, action: usize) -> f64 {
        *self.q_table.get(&(state, action)).unwrap_or(&0.0)
    }

    fn update_policy(&mut self, state: usize) {
        let best = (0..self.n_actions)
            .max_by(|&a, &b| {
                self.q_value(state, a)
                    .partial_cmp(&self.q_value(state, b))
                    .unwrap()
            })
            .unwrap_or(0);
        self.learned_policy.insert(state, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_from_episode() {
        let mut agent = MonteCarloES::new(2, 0.0);
        let episode = vec![(0, 1, 1.0), (1, 0, 0.5)];
        agent.update(&episode);
        // State 1, action 0 should have return 0.5
        assert!((agent.q_value(1, 0) - 0.5).abs() < 1e-10);
        // State 0, action 1 should have return 1.5 (1.0 + 0.5)
        assert!((agent.q_value(0, 1) - 1.5).abs() < 1e-10);
    }
}
