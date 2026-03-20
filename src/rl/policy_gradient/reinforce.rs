use ndarray::{Array1, Array2};
use rand::Rng;

/// REINFORCE (Monte Carlo policy gradient) with value baseline.
///
/// Collects full episodes, computes discounted returns, subtracts a
/// learned baseline to reduce variance, and updates the policy using
/// the policy gradient theorem.
pub struct Reinforce {
    policy_net: PolicyNet,
    value_net: ValueNet,
    learning_rate: f64,
    gamma: f64,
}

pub struct Experience {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
}

impl Reinforce {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, gamma: f64) -> Self {
        Self {
            policy_net: PolicyNet::new(state_dim, action_dim),
            value_net: ValueNet::new(state_dim),
            learning_rate,
            gamma,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let probs = self.policy_net.forward(state);
        sample_categorical(&probs)
    }

    pub fn train(&mut self, episode: &[Experience]) {
        let returns = compute_returns(episode, self.gamma);

        for (i, exp) in episode.iter().enumerate() {
            let value = self.value_net.forward(&exp.state);
            let advantage = returns[i] - value;

            let probs = self.policy_net.forward(&exp.state);
            let mut grad = probs.clone();
            grad[exp.action] -= 1.0;

            self.policy_net
                .backward(&exp.state, &grad, advantage, self.learning_rate);
            self.value_net
                .backward(&exp.state, returns[i], self.learning_rate);
        }
    }
}

fn compute_returns(episode: &[Experience], gamma: f64) -> Vec<f64> {
    let mut returns = vec![0.0; episode.len()];
    let mut running = 0.0;
    for i in (0..episode.len()).rev() {
        running = episode[i].reward + gamma * running;
        returns[i] = running;
    }
    returns
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

struct PolicyNet {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl PolicyNet {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        softmax(&(state.dot(&self.weights) + &self.biases))
    }

    fn backward(&mut self, state: &Array1<f64>, grad: &Array1<f64>, advantage: f64, lr: f64) {
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

struct ValueNet {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl ValueNet {
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
    fn test_compute_returns() {
        let episode = vec![
            Experience {
                state: Array1::zeros(2),
                action: 0,
                reward: 1.0,
            },
            Experience {
                state: Array1::zeros(2),
                action: 0,
                reward: 1.0,
            },
            Experience {
                state: Array1::zeros(2),
                action: 0,
                reward: 1.0,
            },
        ];
        let returns = compute_returns(&episode, 0.99);
        assert!((returns[2] - 1.0).abs() < 1e-10);
        assert!((returns[1] - 1.99).abs() < 1e-10);
        assert!((returns[0] - 2.9701).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let p = softmax(&x);
        assert!((p.sum() - 1.0).abs() < 1e-10);
    }
}
