use rand::Rng;

/// Dropout layer for regularization
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Probability of dropping a neuron (between 0 and 1)
    pub p: f64,
    /// Whether the layer is in training mode
    pub training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of dropping a neuron (between 0 and 1)
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::dropout::Dropout;
    /// let dropout = Dropout::new(0.5);
    /// ```
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be between 0 and 1"
        );
        Dropout { p, training: true }
    }

    /// Sets the layer's mode to training or evaluation
    ///
    /// # Arguments
    ///
    /// * `training` - Whether to set the layer to training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward pass of the dropout layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    ///
    /// # Returns
    ///
    /// * Output tensor
    /// * Cache for backward pass
    pub fn forward(&self, input: &[Vec<f64>]) -> (Vec<Vec<f64>>, DropoutCache) {
        let mut rng = rand::thread_rng();
        let batch_size = input.len();
        let features = input[0].len();

        // Initialize mask and output
        let mut mask = vec![vec![1.0; features]; batch_size];
        let mut output = input.to_owned();

        if self.training {
            let scale = 1.0 / (1.0 - self.p); // Scale factor for training

            // Generate dropout mask and apply it
            for i in 0..batch_size {
                for j in 0..features {
                    if rng.gen::<f64>() < self.p {
                        mask[i][j] = 0.0;
                        output[i][j] = 0.0;
                    } else {
                        output[i][j] *= scale;
                    }
                }
            }
        }

        let cache = DropoutCache {
            mask,
            scale: if self.training {
                1.0 / (1.0 - self.p)
            } else {
                1.0
            },
        };

        (output, cache)
    }

    /// Forward pass for 4D tensors (used in convolutional networks)
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch_size, channels, height, width)
    ///
    /// # Returns
    ///
    /// * Output tensor
    /// * Cache for backward pass
    pub fn forward_4d(
        &self,
        input: &[Vec<Vec<Vec<f64>>>],
    ) -> (Vec<Vec<Vec<Vec<f64>>>>, Dropout4DCache) {
        let mut rng = rand::thread_rng();
        let batch_size = input.len();
        let channels = input[0].len();
        let height = input[0][0].len();
        let width = input[0][0][0].len();

        // Initialize mask and output
        let mut mask = vec![vec![vec![vec![1.0; width]; height]; channels]; batch_size];
        let mut output = input.to_owned();

        if self.training {
            let scale = 1.0 / (1.0 - self.p);

            // Generate dropout mask and apply it
            for b in 0..batch_size {
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            if rng.gen::<f64>() < self.p {
                                mask[b][c][h][w] = 0.0;
                                output[b][c][h][w] = 0.0;
                            } else {
                                output[b][c][h][w] *= scale;
                            }
                        }
                    }
                }
            }
        }

        let cache = Dropout4DCache {
            mask,
            scale: if self.training {
                1.0 / (1.0 - self.p)
            } else {
                1.0
            },
        };

        (output, cache)
    }

    /// Backward pass of the dropout layer
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to the output
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    pub fn backward(&self, grad_output: &[Vec<f64>], cache: &DropoutCache) -> Vec<Vec<f64>> {
        let mut grad_input = grad_output.to_owned();

        if self.training {
            // Apply dropout mask and scaling to gradients
            for i in 0..grad_input.len() {
                for j in 0..grad_input[0].len() {
                    grad_input[i][j] *= cache.mask[i][j] * cache.scale;
                }
            }
        }

        grad_input
    }

    /// Backward pass for 4D tensors
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to the output
    /// * `cache` - Cache from forward pass
    ///
    /// # Returns
    ///
    /// * Gradient with respect to input
    pub fn backward_4d(
        &self,
        grad_output: &[Vec<Vec<Vec<f64>>>],
        cache: &Dropout4DCache,
    ) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut grad_input = grad_output.to_owned();

        if self.training {
            // Apply dropout mask and scaling to gradients
            for b in 0..grad_input.len() {
                for c in 0..grad_input[0].len() {
                    for h in 0..grad_input[0][0].len() {
                        for w in 0..grad_input[0][0][0].len() {
                            grad_input[b][c][h][w] *= cache.mask[b][c][h][w] * cache.scale;
                        }
                    }
                }
            }
        }

        grad_input
    }
}

/// Cache for Dropout forward pass
#[derive(Debug, Clone)]
pub struct DropoutCache {
    /// Dropout mask
    pub mask: Vec<Vec<f64>>,
    /// Scale factor
    pub scale: f64,
}

/// Cache for Dropout forward pass (4D version)
#[derive(Debug, Clone)]
pub struct Dropout4DCache {
    /// Dropout mask
    pub mask: Vec<Vec<Vec<Vec<f64>>>>,
    /// Scale factor
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests dropout initialization
    #[test]
    fn test_dropout_initialization() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.p, 0.5);
        assert!(dropout.training);
    }

    /// Tests invalid dropout probability
    #[test]
    #[should_panic(expected = "Dropout probability must be between 0 and 1")]
    fn test_invalid_dropout_probability() {
        Dropout::new(1.5);
    }

    /// Tests forward pass in training mode
    #[test]
    fn test_forward_training() {
        let dropout = Dropout::new(0.5);
        let input = vec![vec![1.0; 10]; 5];
        let (output, cache) = dropout.forward(&input);

        // Check dimensions
        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
        assert_eq!(cache.mask.len(), input.len());
        assert_eq!(cache.mask[0].len(), input[0].len());

        // Check scaling
        assert!((cache.scale - 2.0).abs() < 1e-6);
    }

    /// Tests forward pass in evaluation mode
    #[test]
    fn test_forward_eval() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let input = vec![vec![1.0; 10]; 5];
        let (output, cache) = dropout.forward(&input);

        // In eval mode, output should equal input
        assert_eq!(output, input);
        assert_eq!(cache.scale, 1.0);
    }

    /// Tests backward pass
    #[test]
    fn test_backward() {
        let dropout = Dropout::new(0.5);
        let input = vec![vec![1.0; 10]; 5];
        let (_, cache) = dropout.forward(&input);

        let grad_output = vec![vec![1.0; 10]; 5];
        let grad_input = dropout.backward(&grad_output, &cache);

        assert_eq!(grad_input.len(), input.len());
        assert_eq!(grad_input[0].len(), input[0].len());
    }

    /// Tests 4D forward pass
    #[test]
    fn test_forward_4d() {
        let dropout = Dropout::new(0.5);
        let input = vec![vec![vec![vec![1.0; 32]; 32]; 3]; 1];
        let (output, cache) = dropout.forward_4d(&input);

        assert_eq!(output.len(), input.len());
        assert_eq!(output[0].len(), input[0].len());
        assert_eq!(output[0][0].len(), input[0][0].len());
        assert_eq!(output[0][0][0].len(), input[0][0][0].len());

        assert_eq!(cache.mask.len(), input.len());
        assert!((cache.scale - 2.0).abs() < 1e-6);
    }

    /// Tests 4D backward pass
    #[test]
    fn test_backward_4d() {
        let dropout = Dropout::new(0.5);
        let input = vec![vec![vec![vec![1.0; 32]; 32]; 3]; 1];
        let (_, cache) = dropout.forward_4d(&input);

        let grad_output = vec![vec![vec![vec![1.0; 32]; 32]; 3]; 1];
        let grad_input = dropout.backward_4d(&grad_output, &cache);

        assert_eq!(grad_input.len(), input.len());
        assert_eq!(grad_input[0].len(), input[0].len());
        assert_eq!(grad_input[0][0].len(), input[0][0].len());
        assert_eq!(grad_input[0][0][0].len(), input[0][0][0].len());
    }
}
