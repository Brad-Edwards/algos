use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Actor-Critic implementation combining policy and value function approximation
pub struct ActorCritic {
    actor_network: ActorNetwork,
    critic_network: CriticNetwork,
    memory: VecDeque<Experience>,
    gamma: f64,
    actor_lr: f64,
    critic_lr: f64,
}

struct ActorNetwork {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct CriticNetwork {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    next_state: Array1<f64>,
    done: bool,
}

impl ActorCritic {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        gamma: f64,
        actor_lr: f64,
        critic_lr: f64,
    ) -> Self {
        let actor_network = ActorNetwork::new(state_dim, action_dim);
        let critic_network = CriticNetwork::new(state_dim);

        ActorCritic {
            actor_network,
            critic_network,
            memory: VecDeque::new(),
            gamma,
            actor_lr,
            critic_lr,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let action_probs = self.actor_network.forward(state);
        self.sample_action(&action_probs)
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
            next_state: next_state.clone(),
            done,
        });

        if self.memory.len() >= 32 {
            // Mini-batch size
            self.update();
        }
    }

    fn update(&mut self) {
        let batch: Vec<_> = self.memory.drain(..32).collect();

        for experience in batch {
            // Calculate TD error
            let current_value = self.critic_network.forward(&experience.state);
            let next_value = if experience.done {
                0.0
            } else {
                self.critic_network.forward(&experience.next_state)
            };

            let td_target = experience.reward + self.gamma * next_value;
            let td_error = td_target - current_value;

            // Update critic
            self.critic_network
                .backward(&experience.state, td_error, self.critic_lr);

            // Update actor using advantage
            self.actor_network.backward(
                &experience.state,
                experience.action,
                td_error,
                self.actor_lr,
            );
        }
    }

    fn sample_action(&self, probs: &Array1<f64>) -> usize {
        let mut cumsum = 0.0;
        let sample = rand::random::<f64>();

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample < cumsum {
                return i;
            }
        }

        probs.len() - 1
    }
}

impl ActorNetwork {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let weights = Array2::zeros((input_dim, output_dim));
        let biases = Array1::zeros(output_dim);

        ActorNetwork { weights, biases }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let logits = state.dot(&self.weights) + &self.biases;
        self.softmax(logits)
    }

    fn backward(&mut self, state: &Array1<f64>, action: usize, advantage: f64, learning_rate: f64) {
        let probs = self.forward(state);
        let mut grad = probs.clone();
        grad[action] -= 1.0;

        // Policy gradient update
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= learning_rate * advantage * state[i] * grad[j];
            }
        }

        for j in 0..self.biases.len() {
            self.biases[j] -= learning_rate * advantage * grad[j];
        }
    }

    fn softmax(&self, x: Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|a| (a - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl CriticNetwork {
    fn new(input_dim: usize) -> Self {
        let weights = Array2::zeros((input_dim, 1));
        let biases = Array1::zeros(1);

        CriticNetwork { weights, biases }
    }

    fn forward(&self, state: &Array1<f64>) -> f64 {
        (state.dot(&self.weights) + &self.biases)[0]
    }

    fn backward(&mut self, state: &Array1<f64>, td_error: f64, learning_rate: f64) {
        // Value function gradient update
        for i in 0..self.weights.nrows() {
            self.weights[[i, 0]] += learning_rate * td_error * state[i];
        }
        self.biases[0] += learning_rate * td_error;
    }
}
