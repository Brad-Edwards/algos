/// lib.rs

/// A simple linear regression model using the Ordinary Least Squares (OLS) solution.
/// 
/// # Fields
/// - `coefficients`: The learned parameters (including intercept as the first element).
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// `coefficients[0]` is the intercept (beta_0), 
    /// `coefficients[1..]` are the slopes for each feature (beta_1, ..., beta_d).
    pub coefficients: Vec<f64>,
}

impl LinearRegression {
    /// Fits a linear model using Ordinary Least Squares (analytic solution).
    ///
    /// # Arguments
    /// - `features`: NxD dataset, where N is number of samples and D is number of features.
    /// - `target`:   Nx1 vector of target values.
    /// - `fit_intercept`: If true, will include an intercept term automatically.
    ///
    /// # Returns
    /// - A `LinearRegression` model with `coefficients` computed.
    ///
    /// # Panics
    /// - If `features.len() != target.len()`.
    /// - If there are no samples or no features (when `fit_intercept = false` and `features` is empty).
    /// - If matrix inversion fails (singular matrix).
    ///
    /// # Example
    ///
    /// ```
    /// use linreg::LinearRegression;
    ///
    /// let x = vec![
    ///     vec![1.0, 2.0],
    ///     vec![2.0, 3.0],
    ///     vec![3.0, 4.0],
    ///     vec![4.0, 5.0],
    /// ];
    /// let y = vec![2.0, 3.0, 4.0, 5.0];
    ///
    /// // Fit with intercept
    /// let model = LinearRegression::fit(&x, &y, true);
    /// println!("Coefficients: {:?}", model.coefficients);
    /// // Suppose we do a prediction
    /// let pred = model.predict(&[2.5, 3.5]);
    /// println!("Predicted value: {:?}", pred);
    /// ```
    pub fn fit(features: &[Vec<f64>], target: &[f64], fit_intercept: bool) -> Self {
        let n = features.len();
        if n == 0 {
            panic!("No training samples provided.");
        }
        if n != target.len() {
            panic!("features and target must have the same number of rows.");
        }
        let d = if n > 0 { features[0].len() } else { 0 };
        if d == 0 && !fit_intercept {
            panic!("No features provided and fit_intercept = false.");
        }

        // Construct design matrix X of size (n x (d+1)) if fit_intercept=true, else (n x d).
        // X[i,0]=1 if intercept is used, then the features follow.
        let x_cols = if fit_intercept { d + 1 } else { d };
        let mut x_matrix = vec![0.0; n * x_cols];
        for i in 0..n {
            let row = &features[i];
            if row.len() != d {
                panic!("Inconsistent feature dimension on row {}.", i);
            }
            if fit_intercept {
                x_matrix[i * x_cols + 0] = 1.0; // intercept
                for (j, val) in row.iter().enumerate() {
                    x_matrix[i * x_cols + (j + 1)] = *val;
                }
            } else {
                for (j, val) in row.iter().enumerate() {
                    x_matrix[i * x_cols + j] = *val;
                }
            }
        }

        // Convert target to a vector y of length n
        let mut y_vec = vec![0.0; n];
        y_vec.copy_from_slice(target);

        // Solve for coefficients using the normal equation: beta = (X^T X)^(-1) X^T y
        // 1. Compute X^T X => shape (x_cols x x_cols)
        // 2. Invert it
        // 3. Compute X^T y => shape (x_cols x 1)
        // 4. Multiply the inverted matrix by X^T y

        // 1. X^T X
        let xtx = matmul_transpose_a(&x_matrix, n, x_cols); // (x_cols x x_cols)
        let mut xtx_inv = invert_matrix(xtx, x_cols)
            .unwrap_or_else(|| panic!("Matrix inversion failed (X^T X might be singular)."));

        // 2. X^T y => shape (x_cols)
        let xty = matvec_transpose_a(&x_matrix, n, x_cols, &y_vec);

        // 3. beta = xtx_inv * xty
        let mut beta = vec![0.0; x_cols];
        for i in 0..x_cols {
            let mut sum = 0.0;
            for j in 0..x_cols {
                sum += xtx_inv[i * x_cols + j] * xty[j];
            }
            beta[i] = sum;
        }

        Self { coefficients: beta }
    }

    /// Predict the target value for a single feature vector `x`.
    ///
    /// # Panics
    /// - If the dimension of `x` doesn't match `coefficients.len() - 1` when intercept is included,
    ///   or `coefficients.len()` if no intercept.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let dim = self.coefficients.len();
        // If we have an intercept, the first coefficient is beta_0, 
        // otherwise we treat `coefficients` as purely slope terms.
        let has_intercept = match dim {
            1 => true, // 1D with intercept means just beta0?
            _ => {
                // We'll guess if there's an intercept by seeing if the user expects
                // x.len() == dim-1
                x.len() + 1 == dim
            }
        };

        if has_intercept {
            // pred = beta0 + sum_{j=1..d} beta_j * x_j
            let mut result = self.coefficients[0];
            for (j, &val) in x.iter().enumerate() {
                result += self.coefficients[j + 1] * val;
            }
            result
        } else {
            // pred = sum_{j=0..d-1} beta_j * x_j
            if x.len() != dim {
                panic!("Input feature length mismatch. Expected {}, got {}", dim, x.len());
            }
            let mut result = 0.0;
            for (j, &val) in x.iter().enumerate() {
                result += self.coefficients[j] * val;
            }
            result
        }
    }

    /// Predict for multiple feature vectors at once.
    pub fn predict_batch(&self, xs: &[Vec<f64>]) -> Vec<f64> {
        xs.iter().map(|row| self.predict(row)).collect()
    }
}

/// Multiplies `X^T` (size x_cols x n) by `X` (size n x x_cols) to get an x_cols x x_cols matrix.
/// This effectively computes X^T X without storing X^T explicitly in memory.
fn matmul_transpose_a(x: &[f64], n: usize, x_cols: usize) -> Vec<f64> {
    // X is shape (n x x_cols), so X^T is shape (x_cols x n).
    // (X^T X) => shape (x_cols x x_cols).
    let mut result = vec![0.0; x_cols * x_cols];
    for i in 0..x_cols {
        for j in 0..x_cols {
            let mut sum = 0.0;
            for k in 0..n {
                // X^T(i, k) = X(k, i)
                // X(k, j)
                let xi = x[k * x_cols + i];
                let xj = x[k * x_cols + j];
                sum += xi * xj;
            }
            result[i * x_cols + j] = sum;
        }
    }
    result
}

/// Multiplies `X^T` (x_cols x n) by `y` (n x 1) => (x_cols x 1).
fn matvec_transpose_a(x: &[f64], n: usize, x_cols: usize, y: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; x_cols];
    for i in 0..x_cols {
        let mut sum = 0.0;
        for k in 0..n {
            // X^T(i,k) = X(k,i)
            sum += x[k * x_cols + i] * y[k];
        }
        result[i] = sum;
    }
    result
}

/// Inverts a square matrix `mat` of size (dim x dim) using a naive Gauss-Jordan elimination.
/// Returns None if the matrix is singular.
///
/// # Note
/// This is purely illustrative. For numerical stability and performance, use a dedicated library (e.g. `nalgebra`).
fn invert_matrix(mut mat: Vec<f64>, dim: usize) -> Option<Vec<f64>> {
    // We'll augment with identity and do row ops
    let mut inv = vec![0.0; dim * dim];
    for i in 0..dim {
        inv[i * dim + i] = 1.0;
    }

    // Perform Gauss-Jordan
    for i in 0..dim {
        // Find pivot
        let mut pivot_row = i;
        let mut pivot_val = mat[i * dim + i].abs();
        for r in (i + 1)..dim {
            let val = mat[r * dim + i].abs();
            if val > pivot_val {
                pivot_row = r;
                pivot_val = val;
            }
        }
        if pivot_val < 1e-15 {
            // Singular or extremely close
            return None;
        }
        // Swap pivot row into place
        if pivot_row != i {
            for c in 0..dim {
                mat.swap(i * dim + c, pivot_row * dim + c);
                inv.swap(i * dim + c, pivot_row * dim + c);
            }
        }

        // Normalize pivot row
        let pivot = mat[i * dim + i];
        for c in 0..dim {
            mat[i * dim + c] /= pivot;
            inv[i * dim + c] /= pivot;
        }

        // Eliminate column in other rows
        for r in 0..dim {
            if r != i {
                let factor = mat[r * dim + i];
                for c in 0..dim {
                    mat[r * dim + c] -= factor * mat[i * dim + c];
                    inv[r * dim + c] -= factor * inv[i * dim + c];
                }
            }
        }
    }

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_regression() {
        // We'll fit y = 2 + 3x, with a small dataset.
        // x = 0 => y=2, x=1 => y=5, x=2 => y=8, ...
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];
        let y = vec![2.0, 5.0, 8.0, 11.0];
        let model = LinearRegression::fit(&x, &y, true);

        // Expect intercept ~2, slope ~3
        let intercept = model.coefficients[0];
        let slope = model.coefficients[1];
        assert!((intercept - 2.0).abs() < 1e-7);
        assert!((slope - 3.0).abs() < 1e-7);

        // Test a prediction
        let pred = model.predict(&[4.0]); // expect 2 + 3*4 = 14
        assert!((pred - 14.0).abs() < 1e-7);
    }

    #[test]
    fn test_no_intercept() {
        // Suppose actual relationship y = 2*x
        // We'll fit without intercept => slope should be ~2
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let model = LinearRegression::fit(&x, &y, false);
        assert_eq!(model.coefficients.len(), 1);
        let slope = model.coefficients[0];
        assert!((slope - 2.0).abs() < 1e-7);

        // Predict for x=4 => y=8
        let pred = model.predict(&[4.0]);
        assert!((pred - 8.0).abs() < 1e-7);
    }
}
