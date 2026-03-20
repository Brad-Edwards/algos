use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Online Actor-Critic with TD(0) updates.
///
/// Combines a policy network (actor) with a value network (critic).
/// The critic provides a one-step TD error used as the advantage
/// signal for the actor, enabling online (step-by-step) learning.
pub struct ActorCritic {
    actor: ActorNet,
    critic: CriticNet,
    memory: VecDeque<Experience>,
    gamma: f64,
    actor_lr: f64,
    critic_lr: f64,
    batch_size: usize,
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
        Self {
            actor: ActorNet::new(state_dim, action_dim),
            critic: CriticNet::new(state_dim),
            memory: VecDeque::new(),
            gamma,
            actor_lr,
            critic_lr,
            batch_size: 32,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let probs = self.actor.forward(state);
        sample_categorical(&probs)
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
            next_state: next_state.clone(),
            done,
        });
        if self.memory.len() >= self.batch_size {
            self.update();
        }
    }

    fn update(&mut self) {
        let batch: Vec<_> = self.memory.drain(..self.batch_size).collect();
        for exp in &batch {
            let current_v = self.critic.forward(&exp.state);
            let next_v = if exp.done {
                0.0
            } else {
                self.critic.forward(&exp.next_state)
            };
            let td_error = exp.reward + self.gamma * next_v - current_v;

            self.critic.backward(&exp.state, td_error, self.critic_lr);
            self.actor
                .backward(&exp.state, exp.action, td_error, self.actor_lr);
        }
    }
}

fn sample_categorical(probs: &Array1<f64>) -> usize {
    let sample = rand::random::<f64>();
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
    exp_x.clone() / exp_x.sum()
}

struct ActorNet {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl ActorNet {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        softmax(&(state.dot(&self.weights) + &self.biases))
    }

    fn backward(&mut self, state: &Array1<f64>, action: usize, advantage: f64, lr: f64) {
        let probs = self.forward(state);
        let mut grad = probs;
        grad[action] -= 1.0;

        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                self.weights[[i, j]] -= lr * advantage * state[i] * grad[j];
            }
        }
        for j in 0..self.biases.len() {
            self.biases[j] -= lr * advantage * grad[j];
        }
    }
}

struct CriticNet {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl CriticNet {
    fn new(input_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, 1)),
            biases: Array1::zeros(1),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> f64 {
        (state.dot(&self.weights) + &self.biases)[0]
    }

    fn backward(&mut self, state: &Array1<f64>, td_error: f64, lr: f64) {
        for i in 0..self.weights.nrows() {
            self.weights[[i, 0]] += lr * td_error * state[i];
        }
        self.biases[0] += lr * td_error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_forward_is_distribution() {
        let actor = ActorNet::new(4, 3);
        let state = Array1::from(vec![1.0, 0.0, -1.0, 0.5]);
        let probs = actor.forward(&state);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
    }
}
