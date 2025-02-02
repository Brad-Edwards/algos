use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Proximal Policy Optimization
#[derive(Clone)]
pub struct PPOConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub clip_epsilon: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub gamma: f64,
    pub batch_size: usize,
    pub n_epochs: usize,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            clip_epsilon: 0.2,
            actor_lr: 0.001,
            critic_lr: 0.001,
            gamma: 0.99,
            batch_size: 32,
            n_epochs: 10,
        }
    }
}

/// Proximal Policy Optimization implementation with clipped objective
pub struct PPO {
    actor: Actor,
    critic: Critic,
    memory: VecDeque<Experience>,
    clip_epsilon: f64,
    actor_lr: f64,
    critic_lr: f64,
    gamma: f64,
    batch_size: usize,
    n_epochs: usize,
}

struct Actor {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct Critic {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    done: bool,
    log_prob: f64,
    value: f64,
}

impl PPO {
    pub fn new(config: PPOConfig) -> Self {
        PPO {
            actor: Actor::new(config.state_dim, config.action_dim),
            critic: Critic::new(config.state_dim),
            memory: VecDeque::new(),
            clip_epsilon: config.clip_epsilon,
            actor_lr: config.actor_lr,
            critic_lr: config.critic_lr,
            gamma: config.gamma,
            batch_size: config.batch_size,
            n_epochs: config.n_epochs,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> (usize, f64, f64) {
        let action_probs = self.actor.forward(state);
        let value = self.critic.forward(state);
        let action = self.sample_action(&action_probs);
        let log_prob = (action_probs[action] + 1e-10).ln();

        (action, log_prob, value)
    }

    pub fn store_experience(
        &mut self,
        state: Array1<f64>,
        action: usize,
        reward: f64,
        done: bool,
        log_prob: f64,
        value: f64,
    ) {
        self.memory.push_back(Experience {
            state,
            action,
            reward,
            done,
            log_prob,
            value,
        });
    }

    pub fn train(&mut self) {
        if self.memory.len() < self.batch_size {
            return;
        }

        let experiences: Vec<_> = self.memory.drain(..).collect();
        let mut returns = Vec::new();
        let mut advantages = Vec::new();

        // Calculate returns and advantages
        let mut running_return = 0.0;
        let mut running_advantage = 0.0;

        for experience in experiences.iter().rev() {
            running_return =
                experience.reward + self.gamma * running_return * (!experience.done as i32 as f64);
            let delta = experience.reward
                + self.gamma * running_advantage * (!experience.done as i32 as f64)
                - experience.value;
            running_advantage =
                delta + self.gamma * 0.95 * running_advantage * (!experience.done as i32 as f64);

            returns.push(running_return);
            advantages.push(running_advantage);
        }

        returns.reverse();
        advantages.reverse();

        // Normalize advantages
        let mean_adv = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let std_adv = (advantages
            .iter()
            .map(|x| (x - mean_adv).powi(2))
            .sum::<f64>()
            / advantages.len() as f64)
            .sqrt();

        for advantage in advantages.iter_mut() {
            *advantage = (*advantage - mean_adv) / (std_adv + 1e-8);
        }

        // Training epochs
        for _ in 0..self.n_epochs {
            let mut indices: Vec<usize> = (0..experiences.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            for chunk in indices.chunks(self.batch_size) {
                // Update actor
                for &idx in chunk {
                    let experience = &experiences[idx];
                    let old_log_prob = experience.log_prob;
                    let advantage = advantages[idx];

                    let action_probs = self.actor.forward(&experience.state);
                    let new_log_prob = (action_probs[experience.action] + 1e-10).ln();

                    let ratio = (new_log_prob - old_log_prob).exp();
                    let surr1 = ratio * advantage;
                    let surr2 =
                        ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage;

                    let actor_loss = -surr1.min(surr2);

                    // Update actor network
                    let mut policy_gradient = action_probs;
                    policy_gradient[experience.action] -= 1.0;
                    self.actor.backward(
                        &experience.state,
                        &policy_gradient,
                        actor_loss,
                        self.actor_lr,
                    );

                    // Update critic
                    self.critic
                        .backward(&experience.state, returns[idx], self.critic_lr);
                }
            }
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
}

impl Actor {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Actor {
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
        loss: f64,
        learning_rate: f64,
    ) {
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= learning_rate * loss * state[i] * policy_gradient[j];
            }
        }

        for j in 0..self.biases.len() {
            self.biases[j] -= learning_rate * loss * policy_gradient[j];
        }
    }

    fn softmax(&self, x: Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|a| (a - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl Critic {
    fn new(input_dim: usize) -> Self {
        Critic {
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

        for i in 0..self.weights.nrows() {
            self.weights[[i, 0]] += learning_rate * error * state[i];
        }
        self.biases[0] += learning_rate * error;
    }
}
