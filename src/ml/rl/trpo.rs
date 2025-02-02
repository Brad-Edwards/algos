use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for Trust Region Policy Optimization
#[derive(Clone)]
pub struct TRPOConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub max_kl: f64,
    pub damping: f64,
    pub gamma: f64,
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
            value_lr: 0.001,
            cg_iters: 10,
            backtrack_iters: 10,
            backtrack_coeff: 0.8,
        }
    }
}

/// Trust Region Policy Optimization implementation
pub struct TRPO {
    policy: Policy,
    value_fn: ValueFunction,
    memory: VecDeque<Experience>,
    max_kl: f64,
    damping: f64,
    gamma: f64,
    value_lr: f64,
    cg_iters: usize,
    backtrack_iters: usize,
    backtrack_coeff: f64,
}

struct Policy {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

struct ValueFunction {
    weights: Array2<f64>,
    biases: Array1<f64>,
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
        TRPO {
            policy: Policy::new(config.state_dim, config.action_dim),
            value_fn: ValueFunction::new(config.state_dim),
            memory: VecDeque::new(),
            max_kl: config.max_kl,
            damping: config.damping,
            gamma: config.gamma,
            value_lr: config.value_lr,
            cg_iters: config.cg_iters,
            backtrack_iters: config.backtrack_iters,
            backtrack_coeff: config.backtrack_coeff,
        }
    }

    pub fn select_action(&self, state: &Array1<f64>) -> (usize, f64) {
        let action_probs = self.policy.forward(state);
        let action = self.sample_action(&action_probs);
        let log_prob = (action_probs[action] + 1e-10).ln();

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
        let mut returns = Vec::new();
        let mut advantages = Vec::new();

        // Compute returns and advantages
        let mut running_return = 0.0;
        for experience in experiences.iter().rev() {
            running_return =
                experience.reward + self.gamma * running_return * (!experience.done as i32 as f64);
            let value = self.value_fn.forward(&experience.state);
            let next_value = if experience.done {
                0.0
            } else {
                self.value_fn.forward(&experience.next_state)
            };

            let delta = experience.reward + self.gamma * next_value - value;
            let advantage = delta + self.gamma * 0.95 * running_return;

            returns.push(running_return);
            advantages.push(advantage);
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

        // Update policy using TRPO
        self.update_policy(&experiences, &advantages);

        // Update value function
        for (i, experience) in experiences.iter().enumerate() {
            self.value_fn
                .backward(&experience.state, returns[i], self.value_lr);
        }
    }

    fn update_policy(&mut self, experiences: &[Experience], advantages: &[f64]) {
        // Compute policy gradient
        let mut policy_gradient = Array1::zeros(self.policy.parameter_size());
        for (i, experience) in experiences.iter().enumerate() {
            let action_probs = self.policy.forward(&experience.state);
            let mut grad = action_probs.clone();
            grad[experience.action] -= 1.0;

            let param_grad = self.policy.parameter_gradient(&experience.state, &grad);
            policy_gradient = &policy_gradient + &(param_grad * advantages[i]);
        }
        policy_gradient = &policy_gradient * (1.0 / experiences.len() as f64);

        // Compute Fisher matrix vector product
        let fisher_vector_product = |v: &Array1<f64>| {
            let mut fvp = Array1::zeros(v.len());
            for experience in experiences.iter() {
                let kl_grad = self.policy.kl_gradient(&experience.state);
                let dot_product = v.dot(&kl_grad);
                fvp = &fvp + &(&kl_grad * dot_product);
            }
            fvp = &fvp * (1.0 / experiences.len() as f64);
            fvp = &fvp + &(v * self.damping);
            fvp
        };

        // Compute search direction using conjugate gradient
        let search_direction = self.conjugate_gradient(fisher_vector_product, &policy_gradient);

        // Compute step size
        let shs = search_direction.dot(&fisher_vector_product(&search_direction));
        let beta = (2.0 * self.max_kl / (shs + 1e-8)).sqrt();
        let step_size = beta * search_direction;

        // Line search
        let old_params = self.policy.get_parameters();
        let mut best_params = old_params.clone();
        let old_loss = self.compute_surrogate_loss(experiences, advantages);
        let mut best_loss = old_loss;

        for i in 0..self.backtrack_iters {
            let coeff = self.backtrack_coeff.powi(i as i32);
            let new_params = &old_params + &(&step_size * coeff);
            self.policy.set_parameters(&new_params);

            let new_loss = self.compute_surrogate_loss(experiences, advantages);
            let kl = self.compute_kl_divergence(experiences);

            if new_loss > best_loss && kl < self.max_kl {
                best_loss = new_loss;
                best_params = new_params.clone();
            }
        }

        // Set best parameters
        self.policy.set_parameters(&best_params);
    }

    fn conjugate_gradient<F>(&self, fvp: F, b: &Array1<f64>) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let mut x = Array1::zeros(b.len());
        let mut r = b - &fvp(&x);
        let mut p = r.clone();
        let mut r_norm_sq = r.dot(&r);

        for _ in 0..self.cg_iters {
            let ap = fvp(&p);
            let alpha = r_norm_sq / (p.dot(&ap) + 1e-8);
            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);
            let r_norm_sq_new = r.dot(&r);
            let beta = r_norm_sq_new / (r_norm_sq + 1e-8);
            r_norm_sq = r_norm_sq_new;
            p = &r + &(&p * beta);
        }

        x
    }

    fn compute_surrogate_loss(&self, experiences: &[Experience], advantages: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (i, experience) in experiences.iter().enumerate() {
            let action_probs = self.policy.forward(&experience.state);
            let new_log_prob = (action_probs[experience.action] + 1e-10).ln();
            let ratio = (new_log_prob - experience.log_prob).exp();
            loss += ratio * advantages[i];
        }
        loss / experiences.len() as f64
    }

    fn compute_kl_divergence(&self, experiences: &[Experience]) -> f64 {
        let mut kl = 0.0;
        for experience in experiences {
            let old_probs = self.policy.forward(&experience.state);
            let new_probs = self.policy.forward(&experience.state);

            kl += old_probs
                .iter()
                .zip(new_probs.iter())
                .map(|(&p_old, &p_new)| p_old * (p_old / (p_new + 1e-10)).ln())
                .sum::<f64>();
        }
        kl / experiences.len() as f64
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

impl Policy {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Policy {
            weights: Array2::zeros((input_dim, output_dim)),
            biases: Array1::zeros(output_dim),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let logits = state.dot(&self.weights) + &self.biases;
        self.softmax(logits)
    }

    fn parameter_size(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn get_parameters(&self) -> Array1<f64> {
        let mut params = Array1::zeros(self.parameter_size());
        let mut idx = 0;

        for w in self.weights.iter() {
            params[idx] = *w;
            idx += 1;
        }

        for b in self.biases.iter() {
            params[idx] = *b;
            idx += 1;
        }

        params
    }

    fn set_parameters(&mut self, params: &Array1<f64>) {
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

    fn parameter_gradient(
        &self,
        state: &Array1<f64>,
        policy_gradient: &Array1<f64>,
    ) -> Array1<f64> {
        let mut grad = Array1::zeros(self.parameter_size());
        let mut idx = 0;

        // Weight gradients
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                grad[idx] = state[i] * policy_gradient[j];
                idx += 1;
            }
        }

        // Bias gradients
        for j in 0..self.biases.len() {
            grad[idx] = policy_gradient[j];
            idx += 1;
        }

        grad
    }

    fn kl_gradient(&self, state: &Array1<f64>) -> Array1<f64> {
        let probs = self.forward(state);
        let mut grad = Array1::zeros(self.parameter_size());
        let mut idx = 0;

        // Weight gradients
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                grad[idx] = state[i] * probs[j] * (1.0 - probs[j]);
                idx += 1;
            }
        }

        // Bias gradients
        for j in 0..self.biases.len() {
            grad[idx] = probs[j] * (1.0 - probs[j]);
            idx += 1;
        }

        grad
    }

    fn softmax(&self, x: Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|a| (a - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }
}

impl ValueFunction {
    fn new(input_dim: usize) -> Self {
        ValueFunction {
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
