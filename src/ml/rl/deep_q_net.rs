use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Deep Q-Network
#[derive(Clone)]
pub struct DQNConfig {
    pub state_dim: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub epsilon: f64,
    pub gamma: f64,
    pub learning_rate: f64,
    pub memory_size: usize,
    pub batch_size: usize,
    pub target_update_freq: usize,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            hidden_dim: 24,
            action_dim: 2,
            epsilon: 1.0,
            gamma: 0.99,
            learning_rate: 0.001,
            memory_size: 10000,
            batch_size: 32,
            target_update_freq: 100,
        }
    }
}

/// Deep Q-Network implementation using neural networks for Q-function approximation
pub struct DQN {
    network: QNetwork,
    target_network: QNetwork,
    memory: VecDeque<Experience>,
    epsilon: f64,
    gamma: f64,
    learning_rate: f64,
    batch_size: usize,
    target_update_freq: usize,
    update_counter: usize,
}

struct QNetwork {
    hidden_weights: Array2<f64>,
    hidden_biases: Array1<f64>,
    output_weights: Array2<f64>,
    output_biases: Array1<f64>,
}

struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    next_state: Array1<f64>,
    done: bool,
}

impl DQN {
    pub fn new(config: DQNConfig) -> Self {
        let network = QNetwork::new(config.state_dim, config.hidden_dim, config.action_dim);
        let target_network = network.clone();

        DQN {
            network,
            target_network,
            memory: VecDeque::with_capacity(config.memory_size),
            epsilon: config.epsilon,
            gamma: config.gamma,
            learning_rate: config.learning_rate,
            batch_size: config.batch_size,
            target_update_freq: config.target_update_freq,
            update_counter: 0,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.network.output_weights.ncols())
        } else {
            let q_values = self.network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        }
    }

    pub fn train(
        &mut self,
        state: Array1<f64>,
        action: usize,
        reward: f64,
        next_state: Array1<f64>,
        done: bool,
    ) {
        // Store experience
        self.memory.push_back(Experience {
            state,
            action,
            reward,
            next_state,
            done,
        });

        if self.memory.len() > self.memory.capacity() {
            self.memory.pop_front();
        }

        if self.memory.len() >= self.batch_size {
            self.update_network();

            // Update target network periodically
            self.update_counter += 1;
            if self.update_counter % self.target_update_freq == 0 {
                self.target_network = self.network.clone();
            }
        }
    }

    fn update_network(&mut self) {
        let mut rng = rand::thread_rng();
        let batch_indices: Vec<usize> = (0..self.memory.len()).collect();
        let batch_indices: Vec<usize> = batch_indices
            .choose_multiple(&mut rng, self.batch_size)
            .copied()
            .collect();

        let mut _total_loss = 0.0;

        for idx in batch_indices {
            let experience = &self.memory[idx];

            // Get current Q value
            let mut current_q_values = self.network.forward(&experience.state);
            let current_q = current_q_values[experience.action];

            // Get target Q value
            let next_q_values = self.target_network.forward(&experience.next_state);
            let next_max_q = if experience.done {
                0.0
            } else {
                next_q_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            };

            let target_q = experience.reward + self.gamma * next_max_q;

            // Compute loss and gradients
            let td_error = target_q - current_q;
            _total_loss += td_error * td_error;

            // Update current Q value
            current_q_values[experience.action] = current_q + self.learning_rate * td_error;

            // Backpropagate
            self.network
                .backward(&experience.state, &current_q_values, self.learning_rate);
        }
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }
}

impl QNetwork {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let hidden_weights = Array2::zeros((input_dim, hidden_dim));
        let hidden_biases = Array1::zeros(hidden_dim);
        let output_weights = Array2::zeros((hidden_dim, output_dim));
        let output_biases = Array1::zeros(output_dim);

        QNetwork {
            hidden_weights,
            hidden_biases,
            output_weights,
            output_biases,
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        // Hidden layer with ReLU activation
        let hidden = state.dot(&self.hidden_weights) + &self.hidden_biases;
        let hidden = hidden.mapv(|x| x.max(0.0));

        // Output layer (no activation, raw Q-values)
        hidden.dot(&self.output_weights) + &self.output_biases
    }

    fn backward(&mut self, state: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        // Forward pass to get activations
        let hidden = state.dot(&self.hidden_weights) + &self.hidden_biases;
        let hidden_activated = hidden.mapv(|x| x.max(0.0));
        let output = hidden_activated.dot(&self.output_weights) + &self.output_biases;

        // Output layer gradients
        let output_delta = output - target;
        let output_delta = output_delta.to_owned();

        // Hidden layer gradients
        let hidden_delta = self.output_weights.dot(&output_delta);
        let hidden_delta = &hidden_delta * &hidden.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let hidden_delta = hidden_delta.to_owned();

        // Update weights and biases
        let output_delta_view = output_delta.view();
        let output_update = learning_rate
            * hidden_activated
                .insert_axis(ndarray::Axis(1))
                .dot(&output_delta_view.insert_axis(ndarray::Axis(0)));
        self.output_weights -= &output_update;
        self.output_biases -= &(learning_rate * output_delta);

        let hidden_delta_view = hidden_delta.view();
        let hidden_update = learning_rate
            * state
                .to_owned()
                .insert_axis(ndarray::Axis(1))
                .dot(&hidden_delta_view.insert_axis(ndarray::Axis(0)));
        self.hidden_weights -= &hidden_update;
        self.hidden_biases -= &(learning_rate * hidden_delta);
    }
}

impl Clone for QNetwork {
    fn clone(&self) -> Self {
        QNetwork {
            hidden_weights: self.hidden_weights.clone(),
            hidden_biases: self.hidden_biases.clone(),
            output_weights: self.output_weights.clone(),
            output_biases: self.output_biases.clone(),
        }
    }
}
