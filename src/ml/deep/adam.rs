use std::f64;

/// A library-grade implementation of the Adam optimizer.
#[derive(Debug, Clone)]
pub struct Adam {
    /// The learning rate (step size).
    pub learning_rate: f64,
    /// Exponential decay rate for the first moment estimates.
    pub beta1: f64,
    /// Exponential decay rate for the second moment estimates.
    pub beta2: f64,
    /// A small constant for numerical stability.
    pub epsilon: f64,
    /// Time step counter.
    pub t: usize,
    /// First moment vector.
    pub m: Vec<f64>,
    /// Second moment vector.
    pub v: Vec<f64>,
}

impl Adam {
    /// Creates a new Adam optimizer instance.
    ///
    /// # Arguments
    /// 
    /// * `learning_rate` - The learning rate.
    /// * `beta1` - The exponential decay rate for the first moment estimates.
    /// * `beta2` - The exponential decay rate for the second moment estimates.
    /// * `epsilon` - A small constant for numerical stability.
    /// * `param_size` - The number of parameters to optimize.
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::adam::Adam;
    /// let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8, 10);
    /// ```
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, param_size: usize) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: vec![0.0; param_size],
            v: vec![0.0; param_size],
        }
    }

    /// Updates the parameters using the Adam optimization rule.
    ///
    /// The update rule is as follows:
    ///
    /// m = beta1 * m + (1 - beta1) * grad
    /// v = beta2 * v + (1 - beta2) * grad^2
    /// m̂ = m / (1 - beta1^t)
    /// v̂ = v / (1 - beta2^t)
    /// param = param - learning_rate * m̂ / (sqrt(v̂) + epsilon)
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable slice of parameters to be updated.
    /// * `grads` - Slice of gradients corresponding to each parameter.
    pub fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len(), "Parameters and gradients must be the same length");
        self.t += 1;
        let t_f64 = self.t as f64;
        
        for (i, (param, &grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t_f64));
            // Compute bias-corrected second moment estimate
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t_f64));
            
            // Update parameter
            *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests a single update step to ensure parameters are updated correctly.
    #[test]
    fn test_adam_single_update() {
        let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8, 1);
        let mut params = vec![1.0];
        let grads = vec![0.5];
        adam.update(&mut params, &grads);
        // With a positive gradient, the parameter should decrease.
        assert!(params[0] < 1.0);
    }

    /// Tests that a mismatch in lengths between parameters and gradients results in a panic.
    #[test]
    #[should_panic(expected = "Parameters and gradients must be the same length")]
    fn test_adam_mismatched_lengths() {
        let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8, 2);
        let mut params = vec![1.0];
        let grads = vec![0.5, 0.3];
        adam.update(&mut params, &grads);
    }

    /// Tests multiple update steps to observe the cumulative effect of the optimizer.
    #[test]
    fn test_adam_multiple_updates() {
        let mut adam = Adam::new(0.01, 0.9, 0.999, 1e-8, 2);
        let mut params = vec![0.0, 0.0];
        let grads = vec![1.0, 2.0];
        for _ in 0..100 {
            adam.update(&mut params, &grads);
        }
        // With constant positive gradients, parameters should become negative over time.
        assert!(params[0] < 0.0 && params[1] < 0.0);
    }
}
