use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;

/// Policy Gradients (REINFORCE) implementation with baseline
pub struct PolicyGradients {
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    memory: VecDeque<Experience>,
    learning_rate: f64,
    gamma: f64,
}

struct PolicyNetwork {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct ValueNetwork {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

pub struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
}

impl PolicyGradients {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, gamma: f64) -> Self {
        PolicyGradients {
            policy_network: PolicyNetwork::new(state_dim, action_dim),
            value_network: ValueNetwork::new(state_dim),
            memory: VecDeque::new(),
            learning_rate,
            gamma,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let action_probs = self.policy_network.forward(state);
        self.sample_action(&action_probs)
    }

    pub fn train(&mut self, episode: Vec<Experience>) {
        let mut returns = Vec::new();
        let mut running_return = 0.0;

        // Calculate returns for each step
        for experience in episode.iter().rev() {
            running_return = experience.reward + self.gamma * running_return;
            returns.push(running_return);
        }
        returns.reverse();

        // Convert returns to array
        let returns = Array1::from(returns);

        // Update networks
        for (i, experience) in episode.iter().enumerate() {
            // Calculate advantage
            let value = self.value_network.forward(&experience.state);
            let advantage = returns[i] - value;

            // Update policy network
            let action_probs = self.policy_network.forward(&experience.state);
            let mut policy_gradient = action_probs.clone();
            policy_gradient[experience.action] -= 1.0;

            self.policy_network.backward(
                &experience.state,
                &policy_gradient,
                advantage,
                self.learning_rate,
            );

            // Update value network (baseline)
            self.value_network
                .backward(&experience.state, returns[i], self.learning_rate);
        }
    }

    fn sample_action(&self, probs: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();
        let sample = rng.gen::<f64>();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample < cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    pub fn store_experience(&mut self, state: Array1<f64>, action: usize, reward: f64) {
        self.memory.push_back(Experience {
            state,
            action,
            reward,
        });
    }

    pub fn get_episode(&mut self) -> Vec<Experience> {
        self.memory.drain(..).collect()
    }
}

impl PolicyNetwork {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        PolicyNetwork {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let logits = state.dot(&self.weights) + &self.biases;
        self.softmax(logits)
    }

    fn backward(
        &mut self,
        state: &Array1<f64>,
        policy_gradient: &Array1<f64>,
        advantage: f64,
        learning_rate: f64,
    ) {
        // Policy gradient update
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= learning_rate * advantage * state[i] * policy_gradient[j];
            }
        }

        for j in 0..self.biases.len() {
            self.biases[j] -= learning_rate * advantage * policy_gradient[j];
        }
    }

    fn softmax(&self, x: Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|a| (a - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl ValueNetwork {
    fn new(input_dim: usize) -> Self {
        ValueNetwork {
            weights: Array2::zeros((input_dim, 1)),
            biases: Array1::zeros(1),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> f64 {
        (state.dot(&self.weights) + &self.biases)[0]
    }

    fn backward(&mut self, state: &Array1<f64>, target: f64, learning_rate: f64) {
        let prediction = self.forward(state);
        let error = target - prediction;

        // Value network update
        for i in 0..self.weights.nrows() {
            self.weights[[i, 0]] += learning_rate * error * state[i];
        }
        self.biases[0] += learning_rate * error;
    }
}
