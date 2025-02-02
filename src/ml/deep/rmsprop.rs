/// RMSprop optimizer implementation
/// 
/// RMSprop maintains a moving average of squared gradients and divides the gradient
/// by the square root of this average. This helps handle different scales of gradients
/// and can lead to faster convergence than standard SGD.
#[derive(Debug, Clone)]
pub struct RMSprop {
    /// Learning rate
    pub learning_rate: f64,
    /// Decay rate for moving average
    pub decay_rate: f64,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Moving average of squared gradients for each parameter
    cache: Vec<f64>,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size for updates
    /// * `decay_rate` - Rate for moving average (typically 0.9)
    /// * `epsilon` - Small constant for numerical stability
    ///
    /// # Example
    ///
    /// ```
    /// use algos::ml::deep::rmsprop::RMSprop;
    /// let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
    /// ```
    pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64) -> Self {
        assert!(learning_rate > 0.0, "Learning rate must be positive");
        assert!(decay_rate > 0.0 && decay_rate < 1.0, "Decay rate must be between 0 and 1");
        assert!(epsilon > 0.0, "Epsilon must be positive");

        RMSprop {
            learning_rate,
            decay_rate,
            epsilon,
            cache: Vec::new(),
        }
    }

    /// Initializes the optimizer for a given number of parameters
    ///
    /// # Arguments
    ///
    /// * `param_count` - Number of parameters to optimize
    pub fn initialize(&mut self, param_count: usize) {
        self.cache = vec![0.0; param_count];
    }

    /// Updates parameters using RMSprop
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters to update
    /// * `grads` - Gradients for each parameter
    ///
    /// # Returns
    ///
    /// * Updated parameters
    pub fn update(&mut self, params: &[f64], grads: &[f64]) -> Vec<f64> {
        assert_eq!(params.len(), grads.len(), "Parameters and gradients must have same length");
        
        if self.cache.is_empty() {
            self.initialize(params.len());
        }

        let mut updated_params = params.to_vec();
        
        for i in 0..params.len() {
            // Update moving average of squared gradients
            self.cache[i] = self.decay_rate * self.cache[i] + 
                           (1.0 - self.decay_rate) * grads[i].powi(2);
            
            // Update parameters
            updated_params[i] -= self.learning_rate * grads[i] / 
                               (self.cache[i] + self.epsilon).sqrt();
        }
        
        updated_params
    }

    /// Updates 2D parameters (e.g., weight matrices)
    ///
    /// # Arguments
    ///
    /// * `params` - 2D parameters to update
    /// * `grads` - 2D gradients for each parameter
    ///
    /// # Returns
    ///
    /// * Updated 2D parameters
    pub fn update_2d(&mut self, params: &[Vec<f64>], grads: &[Vec<f64>]) -> Vec<Vec<f64>> {
        assert_eq!(params.len(), grads.len(), "Parameters and gradients must have same dimensions");
        
        let total_params: usize = params.iter().map(|row| row.len()).sum();
        if self.cache.is_empty() {
            self.initialize(total_params);
        }

        let mut updated_params = params.to_vec();
        let mut cache_idx = 0;
        
        for i in 0..params.len() {
            assert_eq!(params[i].len(), grads[i].len(), 
                      "Parameter and gradient rows must have same length");
            
            for j in 0..params[i].len() {
                // Update moving average of squared gradients
                self.cache[cache_idx] = self.decay_rate * self.cache[cache_idx] + 
                                      (1.0 - self.decay_rate) * grads[i][j].powi(2);
                
                // Update parameters
                updated_params[i][j] -= self.learning_rate * grads[i][j] / 
                                      (self.cache[cache_idx] + self.epsilon).sqrt();
                
                cache_idx += 1;
            }
        }
        
        updated_params
    }

    /// Updates 4D parameters (e.g., convolutional filters)
    ///
    /// # Arguments
    ///
    /// * `params` - 4D parameters to update
    /// * `grads` - 4D gradients for each parameter
    ///
    /// # Returns
    ///
    /// * Updated 4D parameters
    pub fn update_4d(&mut self, params: &[Vec<Vec<Vec<f64>>>], grads: &[Vec<Vec<Vec<f64>>>]) 
        -> Vec<Vec<Vec<Vec<f64>>>> 
    {
        assert_eq!(params.len(), grads.len(), "Parameters and gradients must have same dimensions");
        
        let total_params: usize = params.iter()
            .flat_map(|x| x.iter())
            .flat_map(|x| x.iter())
            .map(|x| x.len())
            .sum();
            
        if self.cache.is_empty() {
            self.initialize(total_params);
        }

        let mut updated_params = params.to_vec();
        let mut cache_idx = 0;
        
        for i in 0..params.len() {
            assert_eq!(params[i].len(), grads[i].len());
            for j in 0..params[i].len() {
                assert_eq!(params[i][j].len(), grads[i][j].len());
                for k in 0..params[i][j].len() {
                    assert_eq!(params[i][j][k].len(), grads[i][j][k].len());
                    for l in 0..params[i][j][k].len() {
                        // Update moving average of squared gradients
                        self.cache[cache_idx] = self.decay_rate * self.cache[cache_idx] + 
                                              (1.0 - self.decay_rate) * grads[i][j][k][l].powi(2);
                        
                        // Update parameters
                        updated_params[i][j][k][l] -= self.learning_rate * grads[i][j][k][l] / 
                                                    (self.cache[cache_idx] + self.epsilon).sqrt();
                        
                        cache_idx += 1;
                    }
                }
            }
        }
        
        updated_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests RMSprop initialization
    #[test]
    fn test_rmsprop_initialization() {
        let optimizer = RMSprop::new(0.001, 0.9, 1e-8);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.decay_rate, 0.9);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert!(optimizer.cache.is_empty());
    }

    /// Tests invalid learning rate
    #[test]
    #[should_panic(expected = "Learning rate must be positive")]
    fn test_invalid_learning_rate() {
        RMSprop::new(-0.001, 0.9, 1e-8);
    }

    /// Tests invalid decay rate
    #[test]
    #[should_panic(expected = "Decay rate must be between 0 and 1")]
    fn test_invalid_decay_rate() {
        RMSprop::new(0.001, 1.5, 1e-8);
    }

    /// Tests parameter update
    #[test]
    fn test_parameter_update() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        
        let updated = optimizer.update(&params, &grads);
        
        assert_eq!(updated.len(), params.len());
        for i in 0..params.len() {
            assert!(updated[i] != params[i]); // Parameters should change
        }
    }

    /// Tests 2D parameter update
    #[test]
    fn test_2d_parameter_update() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let params = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let grads = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        
        let updated = optimizer.update_2d(&params, &grads);
        
        assert_eq!(updated.len(), params.len());
        assert_eq!(updated[0].len(), params[0].len());
        for i in 0..params.len() {
            for j in 0..params[i].len() {
                assert!(updated[i][j] != params[i][j]); // Parameters should change
            }
        }
    }

    /// Tests 4D parameter update
    #[test]
    fn test_4d_parameter_update() {
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        let params = vec![vec![vec![vec![1.0; 2]; 2]; 2]; 2];
        let grads = vec![vec![vec![vec![0.1; 2]; 2]; 2]; 2];
        
        let updated = optimizer.update_4d(&params, &grads);
        
        assert_eq!(updated.len(), params.len());
        assert_eq!(updated[0].len(), params[0].len());
        assert_eq!(updated[0][0].len(), params[0][0].len());
        assert_eq!(updated[0][0][0].len(), params[0][0][0].len());
        
        // Check that parameters have been updated
        assert!(updated[0][0][0][0] != params[0][0][0][0]);
    }
}
