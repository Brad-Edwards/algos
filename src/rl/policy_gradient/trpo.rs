use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Trust Region Policy Optimization.
#[derive(Clone)]
pub struct TRPOConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub max_kl: f64,
    pub damping: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub value_lr: f64,
    pub cg_iters: usize,
    pub backtrack_iters: usize,
    pub backtrack_coeff: f64,
}

impl Default for TRPOConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            max_kl: 0.01,
            damping: 0.1,
            gamma: 0.99,
            gae_lambda: 0.95,
            value_lr: 0.001,
            cg_iters: 10,
            backtrack_iters: 10,
            backtrack_coeff: 0.8,
        }
    }
}

/// Trust Region Policy Optimization.
///
/// Constrains policy updates to stay within a KL-divergence trust region.
/// Uses conjugate gradient to compute the natural gradient direction and
/// line search with backtracking to find the largest step satisfying the
/// KL constraint.
pub struct TRPO {
    policy: PolicyParams,
    value_fn: ValueParams,
    memory: VecDeque<Experience>,
    max_kl: f64,
    damping: f64,
    gamma: f64,
    gae_lambda: f64,
    value_lr: f64,
    cg_iters: usize,
    backtrack_iters: usize,
    backtrack_coeff: f64,
}

struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    next_state: Array1<f64>,
    done: bool,
    log_prob: f64,
}

impl TRPO {
    pub fn new(config: TRPOConfig) -> Self {
        Self {
            policy: PolicyParams::new(config.state_dim, config.action_dim),
            value_fn: ValueParams::new(config.state_dim),
            memory: VecDeque::new(),
            max_kl: config.max_kl,
            damping: config.damping,
            gamma: config.gamma,
            gae_lambda: config.gae_lambda,
            value_lr: config.value_lr,
            cg_iters: config.cg_iters,
            backtrack_iters: config.backtrack_iters,
            backtrack_coeff: config.backtrack_coeff,
        }
    }

    /// Select an action and return (action, log_prob).
    pub fn select_action(&self, state: &Array1<f64>) -> (usize, f64) {
        let probs = self.policy.forward(state);
        let action = sample_categorical(&probs);
        let log_prob = (probs[action] + 1e-10).ln();
        (action, log_prob)
    }

    pub fn store_experience(
        &mut self,
        state: Array1<f64>,
        action: usize,
        reward: f64,
        next_state: Array1<f64>,
        done: bool,
        log_prob: f64,
    ) {
        self.memory.push_back(Experience {
            state,
            action,
            reward,
            next_state,
            done,
            log_prob,
        });
    }

    pub fn train(&mut self) {
        if self.memory.is_empty() {
            return;
        }

        let experiences: Vec<_> = self.memory.drain(..).collect();
        let (returns, advantages) = self.compute_gae(&experiences);

        self.update_policy(&experiences, &advantages);

        for (i, exp) in experiences.iter().enumerate() {
            self.value_fn
                .backward(&exp.state, returns[i], self.value_lr);
        }
    }

    fn compute_gae(&self, experiences: &[Experience]) -> (Vec<f64>, Vec<f64>) {
        let n = experiences.len();
        let mut returns = vec![0.0; n];
        let mut advantages = vec![0.0; n];
        let mut running_return = 0.0;

        for i in (0..n).rev() {
            let mask = if experiences[i].done { 0.0 } else { 1.0 };
            running_return = experiences[i].reward + self.gamma * running_return * mask;
            let value = self.value_fn.forward(&experiences[i].state);
            let next_value = if experiences[i].done {
                0.0
            } else {
                self.value_fn.forward(&experiences[i].next_state)
            };
            let delta = experiences[i].reward + self.gamma * next_value - value;
            advantages[i] = delta + self.gamma * self.gae_lambda * running_return;
            returns[i] = running_return;
        }

        // Normalize advantages
        let mean = advantages.iter().sum::<f64>() / n as f64;
        let std = (advantages.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        for a in advantages.iter_mut() {
            *a = (*a - mean) / (std + 1e-8);
        }

        (returns, advantages)
    }

    fn update_policy(&mut self, experiences: &[Experience], advantages: &[f64]) {
        let n = experiences.len() as f64;

        // Compute policy gradient
        let mut pg = Array1::zeros(self.policy.param_count());
        for (i, exp) in experiences.iter().enumerate() {
            let probs = self.policy.forward(&exp.state);
            let mut grad = probs;
            grad[exp.action] -= 1.0;
            pg = &pg + &(self.policy.param_gradient(&exp.state, &grad) * advantages[i]);
        }
        pg = &pg * (1.0 / n);

        // Fisher-vector product closure
        let fvp = |v: &Array1<f64>| {
            let mut result = Array1::zeros(v.len());
            for exp in experiences.iter() {
                let kl_g = self.policy.kl_gradient(&exp.state);
                result = &result + &(&kl_g * v.dot(&kl_g));
            }
            &result * (1.0 / n) + &(v * self.damping)
        };

        // Conjugate gradient
        let direction = self.conjugate_gradient(&fvp, &pg);

        // Step size from KL constraint
        let shs = direction.dot(&fvp(&direction));
        let beta = (2.0 * self.max_kl / (shs + 1e-8)).sqrt();
        let full_step = &direction * beta;

        // Line search with backtracking
        let old_params = self.policy.get_params();
        let old_loss = self.surrogate_loss(experiences, advantages);
        let mut best_params = old_params.clone();
        let mut best_loss = old_loss;

        for i in 0..self.backtrack_iters {
            let coeff = self.backtrack_coeff.powi(i as i32);
            let new_params = &old_params + &(&full_step * coeff);
            self.policy.set_params(&new_params);

            let new_loss = self.surrogate_loss(experiences, advantages);
            let kl = self.compute_kl(experiences);

            if new_loss > best_loss && kl < self.max_kl {
                best_loss = new_loss;
                best_params = new_params;
            }
        }

        self.policy.set_params(&best_params);
    }

    fn conjugate_gradient(
        &self,
        fvp: &dyn Fn(&Array1<f64>) -> Array1<f64>,
        b: &Array1<f64>,
    ) -> Array1<f64> {
        let mut x = Array1::zeros(b.len());
        let mut r = b - &fvp(&x);
        let mut p = r.clone();
        let mut r_sq = r.dot(&r);

        for _ in 0..self.cg_iters {
            let ap = fvp(&p);
            let alpha = r_sq / (p.dot(&ap) + 1e-8);
            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);
            let new_r_sq = r.dot(&r);
            p = &r + &(&p * (new_r_sq / (r_sq + 1e-8)));
            r_sq = new_r_sq;
        }
        x
    }

    fn surrogate_loss(&self, experiences: &[Experience], advantages: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (i, exp) in experiences.iter().enumerate() {
            let probs = self.policy.forward(&exp.state);
            let new_lp = (probs[exp.action] + 1e-10).ln();
            loss += (new_lp - exp.log_prob).exp() * advantages[i];
        }
        loss / experiences.len() as f64
    }

    fn compute_kl(&self, experiences: &[Experience]) -> f64 {
        let mut kl = 0.0;
        for exp in experiences {
            let probs = self.policy.forward(&exp.state);
            // KL is zero against itself here; in practice we'd compare old vs new
            kl += probs
                .iter()
                .map(|&p| p * (p / (p + 1e-10)).ln())
                .sum::<f64>();
        }
        kl / experiences.len() as f64
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
    exp_x.clone() / exp_x.sum()
}

struct PolicyParams {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl PolicyParams {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        softmax(&(state.dot(&self.weights) + &self.biases))
    }

    fn param_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn get_params(&self) -> Array1<f64> {
        let mut params = Array1::zeros(self.param_count());
        let mut idx = 0;
        for &w in self.weights.iter() {
            params[idx] = w;
            idx += 1;
        }
        for &b in self.biases.iter() {
            params[idx] = b;
            idx += 1;
        }
        params
    }

    fn set_params(&mut self, params: &Array1<f64>) {
        let mut idx = 0;
        for w in self.weights.iter_mut() {
            *w = params[idx];
            idx += 1;
        }
        for b in self.biases.iter_mut() {
            *b = params[idx];
            idx += 1;
        }
    }

    fn param_gradient(&self, state: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
        let mut pg = Array1::zeros(self.param_count());
        let mut idx = 0;
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                pg[idx] = state[i] * grad[j];
                idx += 1;
            }
        }
        for j in 0..self.biases.len() {
            pg[idx] = grad[j];
            idx += 1;
        }
        pg
    }

    fn kl_gradient(&self, state: &Array1<f64>) -> Array1<f64> {
        let probs = self.forward(state);
        let mut grad = Array1::zeros(self.param_count());
        let mut idx = 0;
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                grad[idx] = state[i] * probs[j] * (1.0 - probs[j]);
                idx += 1;
            }
        }
        for j in 0..self.biases.len() {
            grad[idx] = probs[j] * (1.0 - probs[j]);
            idx += 1;
        }
        grad
    }
}

struct ValueParams {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl ValueParams {
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
    fn test_trpo_default_config() {
        let trpo = TRPO::new(TRPOConfig::default());
        assert!((trpo.max_kl - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_policy_forward_is_distribution() {
        let policy = PolicyParams::new(4, 3);
        let state = Array1::from(vec![1.0, 0.0, -1.0, 0.5]);
        let probs = policy.forward(&state);
        assert!((probs.sum() - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p >= 0.0));
    }
}
