use std::f64;

/// A simple logistic regression model with L2 regularization (optional).
/// It uses batch gradient descent for training.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Coefficients, including intercept as the first element:
    ///   `coefficients[0]` = intercept
    ///   `coefficients[1..]` = feature weights
    pub coefficients: Vec<f64>,
    pub fit_intercept: bool,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    /// L2 regularization parameter (0.0 = no regularization).
    pub lambda: f64,
}

impl LogisticRegression {
    /// Creates a new LogisticRegression model with default settings.
    /// - `fit_intercept`: whether to fit an intercept term
    /// - `learning_rate`: step size for gradient descent
    /// - `max_iterations`: maximum number of gradient descent steps
    /// - `tolerance`: stopping criterion on coefficient updates
    /// - `lambda`: L2 regularization strength (0.0 = no regularization)
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        lambda: f64,
    ) -> Self {
        Self {
            coefficients: Vec::new(),
            fit_intercept,
            learning_rate,
            max_iterations,
            tolerance,
            lambda,
        }
    }

    /// Fit the logistic regression model on the given data and binary labels (0 or 1).
    ///
    /// # Arguments
    /// - `features`: NxD data, N samples, D features each
    /// - `labels`: Nx1 binary labels (0.0 or 1.0)
    ///
    /// # Panics
    /// - If `features` is empty or `labels` length does not match `features` length.
    /// - If any feature row has different length from the others.
    /// - If labels are not 0.0 or 1.0 (though minor floating tolerances are allowed).
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        let n = features.len();
        if n == 0 {
            panic!("No training samples provided.");
        }
        if labels.len() != n {
            panic!("Mismatch in features and labels length.");
        }
        let d = features[0].len();
        for (i, row) in features.iter().enumerate() {
            if row.len() != d {
                panic!("Feature dimension mismatch at row {}", i);
            }
        }

        // Check labels are in {0, 1}, within floating tolerance
        for &lbl in labels {
            if !(-f64::EPSILON..=1.0 + f64::EPSILON).contains(&lbl) {
                panic!("Label out of [0,1] range: {}", lbl);
            }
        }

        // Build design matrix X if intercept is used (augment with 1.0 in first column).
        let effective_dim = if self.fit_intercept { d + 1 } else { d };
        self.coefficients = vec![0.0; effective_dim];

        // Batch gradient descent
        for _iter in 0..self.max_iterations {
            // Compute gradient
            let mut gradient = vec![0.0; effective_dim];

            // For each sample
            for i in 0..n {
                let xi = &features[i];
                let yi = labels[i];
                let predicted = self.predict_proba_one(xi);
                let error = predicted - yi; // (h_theta(xi) - yi)
                if self.fit_intercept {
                    // gradient for intercept
                    gradient[0] += error;
                    // gradient for features
                    for (j, val) in xi.iter().enumerate().take(d) {
                        gradient[j + 1] += error * val;
                    }
                } else {
                    for (j, val) in xi.iter().enumerate().take(d) {
                        gradient[j] += error * val;
                    }
                }
            }

            // Average gradient & add regularization term
            for (j, val) in gradient.iter_mut().enumerate().take(effective_dim) {
                *val /= n as f64; // average
                if self.lambda > 0.0 && j > 0 {
                    // do not regularize intercept
                    *val += (self.lambda / n as f64) * self.coefficients[j];
                }
            }

            // Update step
            let mut max_update = 0.0;
            for (j, &val) in gradient.iter().enumerate().take(effective_dim) {
                let update = self.learning_rate * val;
                self.coefficients[j] -= update;
                let abs_update = update.abs();
                if abs_update > max_update {
                    max_update = abs_update;
                }
            }

            // Check convergence
            if max_update < self.tolerance {
                break;
            }
        }
    }

    /// Predict the probability of label=1 for a single feature vector.
    ///
    /// # Panics
    /// - If `features` dimension doesn't match the trained model.
    pub fn predict_proba_one(&self, features: &[f64]) -> f64 {
        let d = if self.fit_intercept {
            self.coefficients.len() - 1
        } else {
            self.coefficients.len()
        };
        if features.len() != d {
            panic!("Expected {} features, got {}", d, features.len());
        }

        let mut z = if self.fit_intercept {
            // intercept
            self.coefficients[0]
        } else {
            0.0
        };
        if self.fit_intercept {
            for (j, val) in features.iter().enumerate().take(d) {
                z += self.coefficients[j + 1] * val;
            }
        } else {
            for (j, val) in features.iter().enumerate().take(d) {
                z += self.coefficients[j] * val;
            }
        }

        sigmoid(z)
    }

    /// Predict probabilities for multiple rows of features.
    pub fn predict_proba_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|row| self.predict_proba_one(row))
            .collect()
    }

    /// Predict a binary label (0 or 1) for a single feature vector, using threshold=0.5.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        if self.predict_proba_one(features) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    /// Predict binary labels for multiple rows.
    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|row| self.predict_one(row)).collect()
    }
}

/// The logistic sigmoid function.
fn sigmoid(z: f64) -> f64 {
    // Numerically stable approach
    if z >= 0.0 {
        let exp_neg = (-z).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = z.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_logistic_regression() {
        // We'll fit a simple logistic regression for a linearly separable dataset in 1D.
        // y=1 if x>2.0 else 0, with some margin.
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![
            0.0, //  x=0
            0.0, //  x=1
            0.0, //  x=2
            1.0, //  x=3
            1.0, //  x=4
        ];

        let mut clf = LogisticRegression::new(true, 0.5, 500, 1e-6, 0.0);
        clf.fit(&x, &y);

        // Check predictions
        let preds = clf.predict_batch(&x);
        // Expect roughly [0, 0, 0, 1, 1].
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
    }
}
