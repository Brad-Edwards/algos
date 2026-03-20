use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Deep Q-Network.
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

/// Deep Q-Network with experience replay and target network.
///
/// Uses a two-layer neural network (ReLU hidden, linear output) to
/// approximate Q(s,a). A frozen target network provides stable TD targets,
/// updated every `target_update_freq` steps.
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
        Self {
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
                .map(|(i, _)| i)
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
            self.update_counter += 1;
            if self.update_counter.is_multiple_of(self.target_update_freq) {
                self.target_network = self.network.clone();
            }
        }
    }

    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon *= decay_rate;
    }

    fn update_network(&mut self) {
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.memory.len()).collect();
        let batch: Vec<usize> = indices
            .choose_multiple(&mut rng, self.batch_size)
            .copied()
            .collect();

        for idx in batch {
            let exp = &self.memory[idx];
            let mut current_q = self.network.forward(&exp.state);
            let current = current_q[exp.action];

            let next_q = self.target_network.forward(&exp.next_state);
            let next_max = if exp.done {
                0.0
            } else {
                next_q.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            };
            let target = exp.reward + self.gamma * next_max;
            let td_error = target - current;
            current_q[exp.action] = current + self.learning_rate * td_error;
            self.network
                .backward(&exp.state, &current_q, self.learning_rate);
        }
    }
}

#[derive(Clone)]
struct QNetwork {
    hidden_weights: Array2<f64>,
    hidden_biases: Array1<f64>,
    output_weights: Array2<f64>,
    output_biases: Array1<f64>,
}

impl QNetwork {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            hidden_weights: Array2::zeros((input_dim, hidden_dim)),
            hidden_biases: Array1::zeros(hidden_dim),
            output_weights: Array2::zeros((hidden_dim, output_dim)),
            output_biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let hidden = (state.dot(&self.hidden_weights) + &self.hidden_biases).mapv(|x| x.max(0.0));
        hidden.dot(&self.output_weights) + &self.output_biases
    }

    fn backward(&mut self, state: &Array1<f64>, target: &Array1<f64>, lr: f64) {
        let hidden_raw = state.dot(&self.hidden_weights) + &self.hidden_biases;
        let hidden = hidden_raw.mapv(|x| x.max(0.0));
        let output = hidden.dot(&self.output_weights) + &self.output_biases;

        let output_delta = &output - target;
        let hidden_delta = &self.output_weights.dot(&output_delta)
            * &hidden_raw.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        self.output_weights -= &(lr
            * hidden
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&output_delta.view().insert_axis(ndarray::Axis(0))));
        self.output_biases -= &(lr * &output_delta);

        self.hidden_weights -= &(lr
            * state
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&hidden_delta.view().insert_axis(ndarray::Axis(0))));
        self.hidden_biases -= &(lr * &hidden_delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqn_creates_with_default_config() {
        let dqn = DQN::new(DQNConfig::default());
        assert_eq!(dqn.batch_size, 32);
    }

    #[test]
    fn test_dqn_forward_output_shape() {
        let net = QNetwork::new(4, 8, 2);
        let state = Array1::zeros(4);
        let q = net.forward(&state);
        assert_eq!(q.len(), 2);
    }
}
