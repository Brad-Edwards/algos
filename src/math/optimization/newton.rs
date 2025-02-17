use num_traits::Float;
use std::fmt::Debug;

use crate::math::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using Newton's method.
///
/// Newton's method uses both first and second derivatives to find the minimum of a function.
/// It typically converges faster than gradient descent for well-behaved functions.
///
/// # Arguments
///
/// * `f` - The objective function to minimize
/// * `initial_point` - The starting point for optimization
/// * `config` - Configuration options for the optimization process
///
/// # Returns
///
/// Returns an `OptimizationResult` containing the optimal point found and optimization statistics.
///
/// # Examples
///
/// ```
/// use algos::math::optimization::{ObjectiveFunction, OptimizationConfig};
/// use algos::math::optimization::newton::minimize;
///
/// // Define a simple quadratic function
/// struct Quadratic;
///
/// impl ObjectiveFunction<f64> for Quadratic {
///     fn evaluate(&self, point: &[f64]) -> f64 {
///         point.iter().map(|x| x * x).sum()
///     }
///
///     fn gradient(&self, point: &[f64]) -> Option<Vec<f64>> {
///         Some(point.iter().map(|x| 2.0 * x).collect())
///     }
///
///     fn hessian(&self, point: &[f64]) -> Option<Vec<Vec<f64>>> {
///         let n = point.len();
///         let mut h = vec![vec![0.0; n]; n];
///         for i in 0..n {
///             h[i][i] = 2.0;
///         }
///         Some(h)
///     }
/// }
///
/// let f = Quadratic;
/// let initial_point = vec![1.0, 1.0];
/// let config = OptimizationConfig::default();
///
/// let result = minimize(&f, &initial_point, &config);
/// assert!(result.converged);
/// ```
pub fn minimize<T, F>(
    f: &F,
    initial_point: &[T],
    config: &OptimizationConfig<T>,
) -> OptimizationResult<T>
where
    T: Float + Debug,
    F: ObjectiveFunction<T>,
{
    let mut current_point = initial_point.to_vec();
    let mut iterations = 0;
    let mut converged = false;
    let n = initial_point.len();

    while iterations < config.max_iterations {
        // Get gradient and Hessian at current point
        let gradient = match f.gradient(&current_point) {
            Some(g) => g,
            None => break, // If gradient is not available, we can't continue
        };

        let hessian = match f.hessian(&current_point) {
            Some(h) => h,
            None => break, // If Hessian is not available, we can't continue
        };

        // Check for convergence using gradient norm
        let gradient_norm = gradient
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
            .sqrt();
        if gradient_norm < config.tolerance {
            converged = true;
            break;
        }

        // Solve H * delta = -gradient using Gaussian elimination
        let mut augmented = vec![vec![T::zero(); n + 1]; n];
        for (i, row) in augmented.iter_mut().enumerate().take(n) {
            for (j, val) in row.iter_mut().enumerate().take(n) {
                *val = hessian[i][j];
            }
            row[n] = -gradient[i];
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            let mut max_val = augmented[i][i].abs();
            for (j, row) in augmented.iter().enumerate().skip(i + 1).take(n - i - 1) {
                let val = row[i].abs();
                if val > max_val {
                    max_idx = j;
                    max_val = val;
                }
            }

            // Swap rows if necessary
            if max_idx != i {
                augmented.swap(i, max_idx);
            }

            // Check for singular matrix
            if augmented[i][i].abs() < T::from(1e-10).unwrap() {
                break;
            }

            // Store the i-th row values we need
            let pivot_row_vals: Vec<T> = augmented[i].clone();
            let pivot = pivot_row_vals[i];

            // Update lower triangular part
            for (_j, row) in augmented.iter_mut().enumerate().skip(i + 1).take(n - i - 1) {
                let factor = row[i] / pivot;
                for (k, val) in row.iter_mut().enumerate().skip(i).take(n - i + 1) {
                    *val = *val - factor * pivot_row_vals[k];
                }
            }
        }

        // Back substitution
        let mut delta = vec![T::zero(); n];
        for (i, row) in augmented.iter().enumerate().rev().take(n) {
            let mut sum = row[n];
            for (j, &val) in row.iter().enumerate().skip(i + 1).take(n - i - 1) {
                sum = sum - val * delta[j];
            }
            delta[i] = sum / row[i];
        }

        // Update point using Newton step with line search
        let mut step_size = T::one();
        let mut new_point = vec![T::zero(); n];
        let current_value = f.evaluate(&current_point);

        // Simple backtracking line search
        for _ in 0..20 {
            for (i, new_val) in new_point.iter_mut().enumerate() {
                *new_val = current_point[i] + step_size * delta[i];
            }
            let new_value = f.evaluate(&new_point);
            if new_value < current_value {
                break;
            }
            step_size = step_size * T::from(0.5).unwrap();
        }

        current_point = new_point;
        iterations += 1;
    }

    OptimizationResult {
        optimal_point: current_point.clone(),
        optimal_value: f.evaluate(&current_point),
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test function: f(x, y) = x^2 + y^2
    struct Quadratic;

    impl ObjectiveFunction<f64> for Quadratic {
        fn evaluate(&self, point: &[f64]) -> f64 {
            point.iter().map(|x| x * x).sum()
        }

        fn gradient(&self, point: &[f64]) -> Option<Vec<f64>> {
            Some(point.iter().map(|x| 2.0 * x).collect())
        }

        fn hessian(&self, point: &[f64]) -> Option<Vec<Vec<f64>>> {
            let n = point.len();
            let mut h = vec![vec![0.0; n]; n];
            for i in 0..n {
                h[i][i] = 2.0;
            }
            Some(h)
        }
    }

    #[test]
    fn test_newton_quadratic() {
        let f = Quadratic;
        let initial_point = vec![1.0, 1.0];
        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!(result.optimal_value < 1e-10);
        for x in result.optimal_point {
            assert!(x.abs() < 1e-5);
        }
    }

    // Test function: f(x) = (x - 2)^2
    struct QuadraticWithMinimum;

    impl ObjectiveFunction<f64> for QuadraticWithMinimum {
        fn evaluate(&self, point: &[f64]) -> f64 {
            let x = point[0];
            (x - 2.0).powi(2)
        }

        fn gradient(&self, point: &[f64]) -> Option<Vec<f64>> {
            let x = point[0];
            Some(vec![2.0 * (x - 2.0)])
        }

        fn hessian(&self, _point: &[f64]) -> Option<Vec<Vec<f64>>> {
            Some(vec![vec![2.0]])
        }
    }

    #[test]
    fn test_newton_quadratic_with_minimum() {
        let f = QuadraticWithMinimum;
        let initial_point = vec![0.0];
        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 2.0).abs() < 1e-5);
        assert!(result.optimal_value < 1e-10);
    }
}
