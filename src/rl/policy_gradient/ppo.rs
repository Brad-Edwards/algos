use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Proximal Policy Optimization.
#[derive(Clone)]
pub struct PPOConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub clip_epsilon: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
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
            gae_lambda: 0.95,
            batch_size: 32,
            n_epochs: 10,
        }
    }
}

/// Proximal Policy Optimization with clipped surrogate objective.
///
/// Collects on-policy rollouts, computes GAE advantages, and performs
/// multiple epochs of minibatch updates with a clipped probability ratio
/// to prevent destructively large policy changes.
pub struct PPO {
    actor: Actor,
    critic: Critic,
    memory: VecDeque<Experience>,
    clip_epsilon: f64,
    actor_lr: f64,
    critic_lr: f64,
    gamma: f64,
    gae_lambda: f64,
    batch_size: usize,
    n_epochs: usize,
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
        Self {
            actor: Actor::new(config.state_dim, config.action_dim),
            critic: Critic::new(config.state_dim),
            memory: VecDeque::new(),
            clip_epsilon: config.clip_epsilon,
            actor_lr: config.actor_lr,
            critic_lr: config.critic_lr,
            gamma: config.gamma,
            gae_lambda: config.gae_lambda,
            batch_size: config.batch_size,
            n_epochs: config.n_epochs,
        }
    }

    /// Select an action and return (action, log_prob, value_estimate).
    pub fn select_action(&self, state: &Array1<f64>) -> (usize, f64, f64) {
        let probs = self.actor.forward(state);
        let value = self.critic.forward(state);
        let action = sample_categorical(&probs);
        let log_prob = (probs[action] + 1e-10).ln();
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
        let (returns, advantages) = self.compute_gae(&experiences);

        for _ in 0..self.n_epochs {
            let mut indices: Vec<usize> = (0..experiences.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            for chunk in indices.chunks(self.batch_size) {
                for &idx in chunk {
                    let exp = &experiences[idx];
                    let probs = self.actor.forward(&exp.state);
                    let new_log_prob = (probs[exp.action] + 1e-10).ln();

                    let ratio = (new_log_prob - exp.log_prob).exp();
                    let surr1 = ratio * advantages[idx];
                    let surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                        * advantages[idx];
                    let actor_loss = -surr1.min(surr2);

                    let mut grad = probs;
                    grad[exp.action] -= 1.0;
                    self.actor
                        .backward(&exp.state, &grad, actor_loss, self.actor_lr);
                    self.critic
                        .backward(&exp.state, returns[idx], self.critic_lr);
                }
            }
        }
    }

    fn compute_gae(&self, experiences: &[Experience]) -> (Vec<f64>, Vec<f64>) {
        let n = experiences.len();
        let mut returns = vec![0.0; n];
        let mut advantages = vec![0.0; n];
        let mut running_return = 0.0;
        let mut running_adv = 0.0;

        for i in (0..n).rev() {
            let mask = if experiences[i].done { 0.0 } else { 1.0 };
            running_return = experiences[i].reward + self.gamma * running_return * mask;
            let delta =
                experiences[i].reward + self.gamma * running_adv * mask - experiences[i].value;
            running_adv = delta + self.gamma * self.gae_lambda * running_adv * mask;
            returns[i] = running_return;
            advantages[i] = running_adv;
        }

        // Normalize advantages
        let mean = advantages.iter().sum::<f64>() / n as f64;
        let std = (advantages.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        for a in advantages.iter_mut() {
            *a = (*a - mean) / (std + 1e-8);
        }

        (returns, advantages)
    }
}

fn sample_categorical(probs: &Array1<f64>) -> usize {
    let sample = rand::thread_rng().gen::<f64>();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if sample < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|a| (a - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

struct Actor {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Actor {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        softmax(&(state.dot(&self.weights) + &self.biases))
    }

    fn backward(&mut self, state: &Array1<f64>, grad: &Array1<f64>, loss: f64, lr: f64) {
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= lr * loss * state[i] * grad[j];
            }
        }
        for j in 0..self.biases.len() {
            self.biases[j] -= lr * loss * grad[j];
        }
    }
}

struct Critic {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Critic {
    fn new(input_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, 1)),
            biases: Array1::zeros(1),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> f64 {
        (state.dot(&self.weights) + &self.biases)[0]
    }

    fn backward(&mut self, state: &Array1<f64>, target: f64, lr: f64) {
        let error = target - self.forward(state);
        for i in 0..self.weights.nrows() {
            self.weights[[i, 0]] += lr * error * state[i];
        }
        self.biases[0] += lr * error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_default_config() {
        let ppo = PPO::new(PPOConfig::default());
        assert_eq!(ppo.batch_size, 32);
        assert!((ppo.clip_epsilon - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_properties() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let p = softmax(&x);
        assert!((p.sum() - 1.0).abs() < 1e-10);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }
}
