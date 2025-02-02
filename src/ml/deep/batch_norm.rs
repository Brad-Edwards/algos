use std::f64;

/// A library-grade implementation of Batch Normalization.
///
/// Batch Normalization normalizes the input by adjusting and scaling the activations,
/// making network training more stable and faster. It maintains running statistics
/// for inference time normalization.
#[derive(Debug, Clone)]
pub struct BatchNorm {
    /// Number of features/channels to normalize
    pub num_features: usize,
    /// Small constant added to variance for numerical stability
    pub epsilon: f64,
    /// Exponential moving average factor for running statistics
    pub momentum: f64,
    /// Learnable scale parameter
    pub gamma: Vec<f64>,
    /// Learnable shift parameter
    pub beta: Vec<f64>,
    /// Running mean for inference
    pub running_mean: Vec<f64>,
    /// Running variance for inference
    pub running_var: Vec<f64>,
    /// Training mode flag
    pub training: bool,
}

impl BatchNorm {
    /// Creates a new BatchNorm instance.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features/channels to normalize
    /// * `epsilon` - Small constant added to variance for numerical stability
    /// * `momentum` - Exponential moving average factor for running statistics
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::batch_norm::BatchNorm;
    /// let batch_norm = BatchNorm::new(64, 1e-5, 0.1);
    /// ```
    pub fn new(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        BatchNorm {
            num_features,
            epsilon,
            momentum,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            training: true,
        }
    }

    /// Sets the training mode of the layer.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Sets the evaluation mode of the layer.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Forward pass of batch normalization.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data with shape [batch_size, num_features]
    ///
    /// # Returns
    ///
    /// * Normalized output with same shape as input
    /// * Cache containing intermediate values for backward pass (if in training mode)
    pub fn forward(&mut self, input: &[Vec<f64>]) -> (Vec<Vec<f64>>, Option<BatchNormCache>) {
        assert!(!input.is_empty(), "Input cannot be empty");
        assert_eq!(
            input[0].len(),
            self.num_features,
            "Input feature dimension mismatch"
        );

        let batch_size = input.len();
        let mut output = vec![vec![0.0; self.num_features]; batch_size];

        if self.training {
            // Calculate batch statistics
            let mut batch_mean = vec![0.0; self.num_features];
            let mut batch_var = vec![0.0; self.num_features];

            // Compute mean
            for input_batch in input.iter().take(batch_size) {
                for (j, mean) in batch_mean.iter_mut().enumerate().take(self.num_features) {
                    *mean += input_batch[j];
                }
            }
            for mean in batch_mean.iter_mut().take(self.num_features) {
                *mean /= batch_size as f64;
            }

            // Compute variance
            for input_batch in input.iter().take(batch_size) {
                for (j, var) in batch_var.iter_mut().enumerate().take(self.num_features) {
                    let diff = input_batch[j] - batch_mean[j];
                    *var += diff * diff;
                }
            }
            for var in batch_var.iter_mut().take(self.num_features) {
                *var /= batch_size as f64;
            }

            // Update running statistics
            for j in 0..self.num_features {
                self.running_mean[j] =
                    self.momentum * self.running_mean[j] + (1.0 - self.momentum) * batch_mean[j];
                self.running_var[j] =
                    self.momentum * self.running_var[j] + (1.0 - self.momentum) * batch_var[j];
            }

            // Normalize and scale
            for i in 0..batch_size {
                for j in 0..self.num_features {
                    let normalized =
                        (input[i][j] - batch_mean[j]) / (batch_var[j] + self.epsilon).sqrt();
                    output[i][j] = self.gamma[j] * normalized + self.beta[j];
                }
            }

            // Cache values for backward pass
            let cache = BatchNormCache {
                input: input.to_vec(),
                batch_mean,
                batch_var,
                normalized: output.clone(),
            };

            (output, Some(cache))
        } else {
            // Use running statistics for inference
            for i in 0..batch_size {
                for j in 0..self.num_features {
                    let normalized = (input[i][j] - self.running_mean[j])
                        / (self.running_var[j] + self.epsilon).sqrt();
                    output[i][j] = self.gamma[j] * normalized + self.beta[j];
                }
            }
            (output, None)
        }
    }

    /// Backward pass of batch normalization.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to output
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    /// * Gradient with respect to gamma
    /// * Gradient with respect to beta
    pub fn backward(
        &self,
        grad_output: &[Vec<f64>],
        cache: &BatchNormCache,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let batch_size = grad_output.len();
        let mut grad_input = vec![vec![0.0; self.num_features]; batch_size];
        let mut grad_gamma = vec![0.0; self.num_features];
        let mut grad_beta = vec![0.0; self.num_features];

        // Compute gradients
        for (i, grad_output_batch) in grad_output.iter().enumerate().take(batch_size) {
            for j in 0..self.num_features {
                grad_beta[j] += grad_output_batch[j];
                grad_gamma[j] +=
                    grad_output_batch[j] * (cache.normalized[i][j] - self.beta[j]) / self.gamma[j];
            }
        }

        // Compute gradient with respect to input
        for i in 0..batch_size {
            for j in 0..self.num_features {
                let std = (cache.batch_var[j] + self.epsilon).sqrt();
                let centered = cache.input[i][j] - cache.batch_mean[j];

                let grad_norm = grad_output[i][j] * self.gamma[j];
                let grad_var =
                    -0.5 * grad_norm * centered / (cache.batch_var[j] + self.epsilon).powf(1.5);
                let grad_mean = -grad_norm / std;

                grad_input[i][j] = grad_norm / std
                    + 2.0 * grad_var * centered / batch_size as f64
                    + grad_mean / batch_size as f64;
            }
        }

        (grad_input, grad_gamma, grad_beta)
    }
}

/// Cache structure for storing intermediate values needed in backward pass
#[derive(Debug, Clone)]
pub struct BatchNormCache {
    /// Input data
    pub input: Vec<Vec<f64>>,
    /// Batch mean
    pub batch_mean: Vec<f64>,
    /// Batch variance
    pub batch_var: Vec<f64>,
    /// Normalized values before scaling and shifting
    pub normalized: Vec<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests basic initialization of BatchNorm
    #[test]
    fn test_batch_norm_initialization() {
        let bn = BatchNorm::new(4, 1e-5, 0.1);
        assert_eq!(bn.num_features, 4);
        assert_eq!(bn.gamma.len(), 4);
        assert_eq!(bn.beta.len(), 4);
        assert!(bn.gamma.iter().all(|&x| x == 1.0));
        assert!(bn.beta.iter().all(|&x| x == 0.0));
    }

    /// Tests forward pass in training mode
    #[test]
    fn test_batch_norm_forward_train() {
        let mut bn = BatchNorm::new(2, 1e-5, 0.1);
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let (output, cache) = bn.forward(&input);

        assert!(cache.is_some());
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());

        // Check that output is normalized
        let _cache = cache.unwrap();
        for j in 0..bn.num_features {
            let mut mean = 0.0;
            let mut var = 0.0;
            for i in 0..output.len() {
                mean += output[i][j];
            }
            mean /= output.len() as f64;
            for i in 0..output.len() {
                var += (output[i][j] - mean).powi(2);
            }
            var /= output.len() as f64;

            // Due to gamma and beta, mean might not be exactly 0 and var might not be exactly 1
            assert!((mean - bn.beta[j]).abs() < 1e-5);
            assert!((var - bn.gamma[j].powi(2)).abs() < 1e-5);
        }
    }

    /// Tests forward pass in evaluation mode
    #[test]
    fn test_batch_norm_forward_eval() {
        let mut bn = BatchNorm::new(2, 1e-5, 0.1);
        bn.eval();
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let (output, cache) = bn.forward(&input);

        assert!(cache.is_none());
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
    }

    /// Tests backward pass computation
    #[test]
    fn test_batch_norm_backward() {
        let mut bn = BatchNorm::new(2, 1e-5, 0.1);
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let (_, cache) = bn.forward(&input);
        let grad_output = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let (grad_input, grad_gamma, grad_beta) = bn.backward(&grad_output, &cache.unwrap());

        assert_eq!(grad_input.len(), input.len());
        assert_eq!(grad_input[0].len(), input[0].len());
        assert_eq!(grad_gamma.len(), bn.num_features);
        assert_eq!(grad_beta.len(), bn.num_features);
    }

    /// Tests handling of empty input
    #[test]
    #[should_panic(expected = "Input cannot be empty")]
    fn test_batch_norm_empty_input() {
        let mut bn = BatchNorm::new(2, 1e-5, 0.1);
        let input: Vec<Vec<f64>> = vec![];
        bn.forward(&input);
    }

    /// Tests handling of input with wrong feature dimension
    #[test]
    #[should_panic(expected = "Input feature dimension mismatch")]
    fn test_batch_norm_wrong_feature_dim() {
        let mut bn = BatchNorm::new(3, 1e-5, 0.1);
        let input = vec![
            vec![1.0, 2.0], // Only 2 features when 3 expected
            vec![3.0, 4.0],
        ];
        bn.forward(&input);
    }
}
