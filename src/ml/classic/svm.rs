use std::f64::consts::E;

/// A trait for kernel functions used by the SVM.
pub trait Kernel {
    /// Compute the kernel value between two feature vectors.
    fn compute(&self, x: &[f64], y: &[f64]) -> f64;
}

/// A linear kernel K(x,y) = x · y
#[derive(Debug, Clone)]
pub struct LinearKernel;

impl Kernel for LinearKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}

/// An RBF (Gaussian) kernel K(x,y) = exp(-gamma * ||x - y||^2)
#[derive(Debug, Clone)]
pub struct RBFKernel {
    pub gamma: f64,
}

impl Kernel for RBFKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff = xi - yi;
            sum_sq += diff * diff;
        }
        (-(self.gamma) * sum_sq).exp()
    }
}

/// A polynomial kernel K(x,y) = (x · y + coef0)^degree
#[derive(Debug, Clone)]
pub struct PolynomialKernel {
    pub degree: u32,
    pub coef0: f64,
}

impl Kernel for PolynomialKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        (dot + self.coef0).powi(self.degree as i32)
    }
}

/// Configuration for the SVM training process.
#[derive(Debug, Clone)]
pub struct SVMConfig {
    /// Regularization parameter (often called `C` in SVM formulations).
    pub c: f64,
    /// Tolerance for stopping criterion (SMO iteration).
    pub tolerance: f64,
    /// Maximum number of iterations for the SMO loop.
    pub max_iter: usize,
    /// Epsilon for floating comparisons or alpha changes.
    pub eps: f64,
}

/// A Support Vector Machine for **binary classification** using the SMO algorithm.
/// Labels must be +1 or -1.
#[derive(Debug, Clone)]
pub struct SVM<K: Kernel> {
    /// The kernel used to transform data or compute similarity.
    pub kernel: K,
    /// The learned Lagrange multipliers, one per training sample.
    pub alphas: Vec<f64>,
    /// The learned bias term.
    pub b: f64,
    /// The training data used to build the model.
    pub support_vectors: Vec<Vec<f64>>,
    /// The labels corresponding to the training data (+1 or -1).
    pub labels: Vec<f64>,
    /// Cache of kernel evaluations if needed for speed (optional).
    kernel_cache: Option<Vec<Vec<f64>>>,
    /// The SVM configuration (C, tolerance, etc.).
    pub config: SVMConfig,
}

/// Implementation of the SVM.
impl<K: Kernel> SVM<K> {
    /// Creates a new, untrained SVM with the given kernel and config.
    pub fn new(kernel: K, config: SVMConfig) -> Self {
        Self {
            kernel,
            alphas: Vec::new(),
            b: 0.0,
            support_vectors: Vec::new(),
            labels: Vec::new(),
            kernel_cache: None,
            config,
        }
    }

    /// Fit (train) the SVM on the provided data using a simplified SMO approach.
    ///
    /// # Arguments
    /// - `x`: NxD dataset (N samples, D features).
    /// - `y`: Nx1 vector of labels (+1.0 or -1.0).
    ///
    /// # Panics
    /// - If `x.len() != y.len()`.
    /// - If any label is not +1 or -1.
    /// - If dataset is empty.
    ///
    /// This method modifies the SVM in-place, setting `alphas`, `b`, and storing the training set
    /// in `support_vectors` and `labels`.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        if n == 0 {
            panic!("No training data provided.");
        }
        if y.len() != n {
            panic!("Mismatch in x.len() and y.len().");
        }
        for &lbl in y {
            if (lbl - 1.0).abs() > 1e-12 && (lbl + 1.0).abs() > 1e-12 {
                panic!("Labels must be +1 or -1, got {}", lbl);
            }
        }

        // Initialize internal data
        self.support_vectors = x.to_vec();
        self.labels = y.to_vec();
        self.alphas = vec![0.0; n];
        self.b = 0.0;
        self.init_kernel_cache();

        // SMO training
        self.smo_solve();
    }

    /// Predict label (+1 or -1) for a single feature vector.
    /// Uses the sign of the decision function.
    ///
    /// # Example
    /// ```
    /// // after calling .fit(...) on the SVM:
    /// // let label = svm.predict(&[1.2, 3.4]);
    /// ```
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let decision_value = self.decision_function(sample);
        if decision_value >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Predict probabilities or decision scores for a single sample
    /// by returning the margin (distance from boundary) = w · x + b in kernel space.
    /// This is not a true probability but can be used as a confidence measure.
    pub fn decision_function(&self, sample: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.support_vectors.len() {
            if self.alphas[i].abs() > self.config.eps {
                let k_val = self.kernel.compute(&self.support_vectors[i], sample);
                sum += self.alphas[i] * self.labels[i] * k_val;
            }
        }
        sum + self.b
    }

    /// Predict multiple samples at once.
    pub fn predict_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|row| self.predict(row)).collect()
    }

    /// Initialize a kernel cache for faster kernel lookups if desired.
    fn init_kernel_cache(&mut self) {
        let n = self.support_vectors.len();
        self.kernel_cache = Some(vec![vec![0.0; n]; n]);
        let cache = self.kernel_cache.as_mut().unwrap();
        for i in 0..n {
            for j in 0..n {
                cache[i][j] =
                    self.kernel.compute(&self.support_vectors[i], &self.support_vectors[j]);
            }
        }
    }

    /// Get kernel value K(i,j) from cache or compute on the fly.
    fn kernel_value(&self, i: usize, j: usize) -> f64 {
        if let Some(ref cache) = self.kernel_cache {
            cache[i][j]
        } else {
            self.kernel
                .compute(&self.support_vectors[i], &self.support_vectors[j])
        }
    }

    /// The main SMO iteration. This is a simplified version, not optimized for large-scale.
    fn smo_solve(&mut self) {
        let n = self.support_vectors.len();
        let c = self.config.c;
        let tol = self.config.tolerance;
        let max_iter = self.config.max_iter;

        let mut iter_count = 0;
        let mut alpha_changed = 0;

        // Precompute errors: E[i] = f(x_i) - y_i
        let mut errors = vec![0.0; n];
        for i in 0..n {
            errors[i] = self.compute_error(i);
        }

        while iter_count < max_iter {
            alpha_changed = 0;
            for i in 0..n {
                let e_i = errors[i];
                let r_i = e_i * self.labels[i];
                // Check KKT conditions
                // If outside boundary => can attempt to optimize alpha_i
                if (r_i < -tol && self.alphas[i] < c - self.config.eps)
                    || (r_i > tol && self.alphas[i] > self.config.eps)
                {
                    // pick a j != i
                    let j = self.select_second_index(i, &errors);

                    let e_j = errors[j];
                    let alpha_i_old = self.alphas[i];
                    let alpha_j_old = self.alphas[j];
                    let (l, h) = self.compute_l_h(i, j);

                    if (l - h).abs() < self.config.eps {
                        continue;
                    }

                    let eta = 2.0 * self.kernel_value(i, j)
                        - self.kernel_value(i, i)
                        - self.kernel_value(j, j);
                    if eta >= 0.0 {
                        continue;
                    }
                    // new alpha_j
                    self.alphas[j] = alpha_j_old - (self.labels[j] * (e_i - e_j) / eta);

                    // clip to [L, H]
                    if self.alphas[j] > h {
                        self.alphas[j] = h;
                    } else if self.alphas[j] < l {
                        self.alphas[j] = l;
                    }

                    if (self.alphas[j] - alpha_j_old).abs() < self.config.eps {
                        continue;
                    }

                    // alpha_i
                    self.alphas[i] = alpha_i_old
                        + self.labels[i] * self.labels[j] * (alpha_j_old - self.alphas[j]);

                    // update b
                    let b1 = self.b
                        - e_i
                        - self.labels[i] * (self.alphas[i] - alpha_i_old)
                            * self.kernel_value(i, i)
                        - self.labels[j] * (self.alphas[j] - alpha_j_old)
                            * self.kernel_value(i, j);
                    let b2 = self.b
                        - e_j
                        - self.labels[i] * (self.alphas[i] - alpha_i_old)
                            * self.kernel_value(i, j)
                        - self.labels[j] * (self.alphas[j] - alpha_j_old)
                            * self.kernel_value(j, j);

                    if self.alphas[i] > 0.0 && self.alphas[i] < c {
                        self.b = b1;
                    } else if self.alphas[j] > 0.0 && self.alphas[j] < c {
                        self.b = b2;
                    } else {
                        self.b = 0.5 * (b1 + b2);
                    }

                    // update errors
                    errors[i] = self.compute_error(i);
                    errors[j] = self.compute_error(j);

                    alpha_changed += 1;
                }
            } // end for i
            if alpha_changed == 0 {
                break;
            }
            iter_count += 1;
        } // end while
    }

    /// Compute error E[i] = f(x_i) - y_i
    fn compute_error(&self, i: usize) -> f64 {
        let fx_i = self.decision_function(&self.support_vectors[i]);
        fx_i - self.labels[i]
    }

    /// Heuristic to select a second index j different from i.
    fn select_second_index(&self, i: usize, errors: &[f64]) -> usize {
        // a naive approach: pick the index with the largest error difference
        let mut best_j = i;
        let mut max_diff = 0.0;
        for (idx, &err) in errors.iter().enumerate() {
            if idx == i {
                continue;
            }
            let diff = (errors[i] - err).abs();
            if diff > max_diff {
                max_diff = diff;
                best_j = idx;
            }
        }
        best_j
    }

    /// Compute L and H for the alpha_j update.
    fn compute_l_h(&self, i: usize, j: usize) -> (f64, f64) {
        let c = self.config.c;
        if self.labels[i] == self.labels[j] {
            let gamma = self.alphas[i] + self.alphas[j];
            let l = f64::max(0.0, gamma - c);
            let h = f64::min(c, gamma);
            (l, h)
        } else {
            let gamma = self.alphas[j] - self.alphas[i];
            let l = f64::max(0.0, -gamma);
            let h = f64::min(c, c - gamma);
            (l, h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_svm_separable() {
        // We'll build a small linearly-separable dataset in 2D:
        // Points with x1 + x2 > 2 => label +1
        // Otherwise => label -1
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![1.0, 1.0],
            vec![2.0, 2.5],
            vec![2.5, 2.0],
            vec![3.0, 3.0],
        ];
        let labels = vec![
            -1.0, // sum=0.0
            -1.0, // sum=1.5
            -1.0, // sum=2.0 borderline
            +1.0, // sum=4.5
            +1.0, // sum=4.5
            +1.0, // sum=6.0
        ];

        let config = SVMConfig {
            c: 1.0,
            tolerance: 1e-3,
            max_iter: 100,
            eps: 1e-6,
        };
        let kernel = LinearKernel;
        let mut svm = SVM::new(kernel, config);
        svm.fit(&data, &labels);

        // Test predictions
        let test1 = svm.predict(&[1.0, 1.0]); // sum=2.0 => borderline => might be -1 or +1 depending on margin
        let test2 = svm.predict(&[2.0, 2.0]); // sum=4.0 => definitely +1
        let test3 = svm.predict(&[0.5, 0.5]); // sum=1.0 => -1

        // We won't enforce a strict label for the sum=2 boundary, but let's see what we get.
        assert!(test1 == -1.0 || test1 == 1.0);
        assert_eq!(test2, 1.0);
        assert_eq!(test3, -1.0);
    }

    #[test]
    fn test_rbf_svm() {
        // A simple test for an RBF SVM with an XOR-like pattern:
        // (0,0) => -1, (1,1) => -1, (1,0) => +1, (0,1) => +1
        // RBF should separate them in kernel space.
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let labels = vec![-1.0, -1.0, 1.0, 1.0];

        let config = SVMConfig {
            c: 10.0,
            tolerance: 1e-3,
            max_iter: 200,
            eps: 1e-6,
        };
        let kernel = RBFKernel { gamma: 1.0 };
        let mut svm = SVM::new(kernel, config);
        svm.fit(&data, &labels);

        // Check training points classification
        for i in 0..data.len() {
            let pred = svm.predict(&data[i]);
            assert_eq!(pred, labels[i], "Mismatch at i={}", i);
        }
    }
}
