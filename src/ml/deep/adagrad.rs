use std::f64;

/// A library-grade implementation of the AdaGrad optimizer.
///
/// AdaGrad adapts the learning rate to the parameters, performing larger updates
/// for infrequent parameters and smaller updates for frequent parameters.
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// The base learning rate.
    pub learning_rate: f64,
    /// A small constant for numerical stability.
    pub epsilon: f64,
    /// Accumulated squared gradients.
    pub accumulated_grad_sq: Vec<f64>,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer instance.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The base learning rate.
    /// * `epsilon` - A small constant for numerical stability.
    /// * `param_size` - The number of parameters to optimize.
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::adagrad::AdaGrad;
    /// let optimizer = AdaGrad::new(0.01, 1e-8, 10);
    /// ```
    pub fn new(learning_rate: f64, epsilon: f64, param_size: usize) -> Self {
        AdaGrad {
            learning_rate,
            epsilon,
            accumulated_grad_sq: vec![0.0; param_size],
        }
    }

    /// Updates the parameters using the AdaGrad optimization rule.
    ///
    /// The update rule is as follows:
    ///
    /// acc_grad_sq += grad^2
    /// param = param - learning_rate * grad / sqrt(acc_grad_sq + epsilon)
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable slice of parameters to be updated.
    /// * `grads` - Slice of gradients corresponding to each parameter.
    pub fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "Parameters and gradients must be the same length"
        );
        assert_eq!(
            params.len(),
            self.accumulated_grad_sq.len(),
            "Parameter size mismatch with initialization"
        );

        for (i, (param, &grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Accumulate squared gradient
            self.accumulated_grad_sq[i] += grad * grad;

            // Compute adaptive learning rate and update parameter
            let adaptive_lr =
                self.learning_rate / (self.accumulated_grad_sq[i].sqrt() + self.epsilon);
            *param -= adaptive_lr * grad;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests initialization with correct parameter sizes.
    #[test]
    fn test_adagrad_initialization() {
        let optimizer = AdaGrad::new(0.01, 1e-8, 5);
        assert_eq!(optimizer.accumulated_grad_sq.len(), 5);
        assert!(optimizer.accumulated_grad_sq.iter().all(|&x| x == 0.0));
    }

    /// Tests a single update step to ensure parameters are updated correctly.
    #[test]
    fn test_adagrad_single_update() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8, 1);
        let mut params = vec![1.0];
        let grads = vec![0.5];
        optimizer.update(&mut params, &grads);
        // With positive gradient, parameter should decrease
        assert!(params[0] < 1.0);
        // Accumulated gradient should be positive
        assert!(optimizer.accumulated_grad_sq[0] > 0.0);
    }

    /// Tests that mismatched parameter and gradient lengths cause a panic.
    #[test]
    #[should_panic(expected = "Parameters and gradients must be the same length")]
    fn test_adagrad_mismatched_lengths() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8, 2);
        let mut params = vec![1.0];
        let grads = vec![0.5, 0.3];
        optimizer.update(&mut params, &grads);
    }

    /// Tests that parameter size mismatch with initialization causes a panic.
    #[test]
    #[should_panic(expected = "Parameter size mismatch with initialization")]
    fn test_adagrad_size_mismatch() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8, 1);
        let mut params = vec![1.0, 2.0];
        let grads = vec![0.5, 0.3];
        optimizer.update(&mut params, &grads);
    }

    /// Tests multiple update steps to verify adaptive learning rate behavior.
    #[test]
    fn test_adagrad_multiple_updates() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8, 2);
        let mut params = vec![0.0, 0.0];
        let grads = vec![1.0, 0.1];

        // Perform multiple updates
        for _ in 0..100 {
            optimizer.update(&mut params, &grads);
        }

        // Parameter with larger gradient should have smaller accumulated changes
        // due to adaptive learning rate
        let change_ratio = params[0].abs() / params[1].abs();
        assert!(change_ratio < 10.0);
    }

    /// Tests that updates still work with very small gradients.
    #[test]
    fn test_adagrad_small_gradients() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8, 1);
        let mut params = vec![1.0];
        let grads = vec![1e-6];

        optimizer.update(&mut params, &grads);
        assert!(params[0] != 1.0); // Should still update despite small gradient
    }
}
