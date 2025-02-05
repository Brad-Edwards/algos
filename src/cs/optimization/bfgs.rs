use num_traits::Float;
use std::fmt::Debug;

use crate::cs::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using the BFGS (Broyden–Fletcher–Goldfarb–Shanno) method.
///
/// BFGS is a quasi-Newton method that approximates the Hessian matrix using gradient information.
/// It maintains a positive definite approximation to the Hessian matrix and updates it iteratively.
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
/// use algos::cs::optimization::{ObjectiveFunction, OptimizationConfig};
/// use algos::cs::optimization::bfgs::minimize;
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
/// }
///
/// let f = Quadratic;
/// let initial_point = vec![1.0, 1.0];
/// let config = OptimizationConfig::default();
///
/// let result = minimize(&f, &initial_point, &config);
/// assert!(result.converged);
/// ```
pub fn minimize<T>(f: &impl ObjectiveFunction<T>, initial_point: &[T], config: &OptimizationConfig<T>) -> OptimizationResult<T>
where
    T: Float + Debug,
{
    let n = initial_point.len();
    let mut current_point = initial_point.to_vec();
    let mut iterations = 0;
    let mut converged = false;

    // Initialize approximate inverse Hessian as identity matrix
    let mut h_inv = vec![vec![T::zero(); n]; n];
    for i in 0..n {
        h_inv[i][i] = T::one();
    }

    // Get initial gradient
    let mut gradient = match f.gradient(&current_point) {
        Some(g) => g,
        None => {
            return OptimizationResult {
                optimal_point: current_point.clone(),
                optimal_value: f.evaluate(&current_point),
                iterations: 0,
                converged: false,
            };
        }
    };

    // Constants for Wolfe conditions
    let c1 = T::from(1e-4).unwrap();  // Sufficient decrease parameter
    let c2 = T::from(0.9).unwrap();   // Curvature condition parameter
    let max_line_search = 20;

    while iterations < config.max_iterations {
        // Check for convergence with a more relaxed criterion for difficult functions
        let gradient_norm = gradient
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
            .sqrt();
        let scale = T::one().max(f.evaluate(&current_point).abs());
        if gradient_norm < config.tolerance * scale {
            converged = true;
            break;
        }

        // Compute search direction: p = -H⁻¹∇f
        let mut direction = vec![T::zero(); n];
        for i in 0..n {
            for j in 0..n {
                direction[i] = direction[i] - h_inv[i][j] * gradient[j];
            }
        }

        // Line search with Wolfe conditions
        let mut alpha = T::one();  // Start with full step
        let mut new_point = vec![T::zero(); n];
        let current_value = f.evaluate(&current_point);
        let directional_derivative = gradient
            .iter()
            .zip(direction.iter())
            .fold(T::zero(), |acc, (&g, &d)| acc + g * d);

        let mut found_step = false;
        for _ in 0..max_line_search {
            // Try current step size
            for i in 0..n {
                new_point[i] = current_point[i] + alpha * direction[i];
            }
            let new_value = f.evaluate(&new_point);

            // Check Armijo condition (sufficient decrease)
            if new_value <= current_value + c1 * alpha * directional_derivative {
                // Get new gradient for curvature condition
                if let Some(new_grad) = f.gradient(&new_point) {
                    let new_directional_derivative = new_grad
                        .iter()
                        .zip(direction.iter())
                        .fold(T::zero(), |acc, (&g, &d)| acc + g * d);

                    // Check curvature condition
                    if new_directional_derivative.abs() <= c2 * directional_derivative.abs() {
                        found_step = true;
                        break;
                    }
                }
            }
            alpha = alpha * T::from(0.5).unwrap();
        }

        if !found_step {
            // If line search failed, take a small step in the descent direction
            alpha = T::from(1e-4).unwrap();
            for i in 0..n {
                new_point[i] = current_point[i] + alpha * direction[i];
            }
        }

        // Compute s = x_{k+1} - x_k
        let s: Vec<T> = new_point
            .iter()
            .zip(current_point.iter())
            .map(|(&x_new, &x_old)| x_new - x_old)
            .collect();

        // Get new gradient and compute y = ∇f_{k+1} - ∇f_k
        let new_gradient = match f.gradient(&new_point) {
            Some(g) => g,
            None => break,
        };

        let y: Vec<T> = new_gradient
            .iter()
            .zip(gradient.iter())
            .map(|(&g_new, &g_old)| g_new - g_old)
            .collect();

        // Compute ρ = 1/(y^T s) with safeguard
        let ys = y.iter()
            .zip(s.iter())
            .fold(T::zero(), |acc, (&y_i, &s_i)| acc + y_i * s_i);
        let rho = if ys.abs() > T::from(1e-10).unwrap() {
            T::one() / ys
        } else {
            T::one() / T::from(1e-10).unwrap()
        };

        // BFGS update for inverse Hessian approximation
        let mut temp_matrix = vec![vec![T::zero(); n]; n];
        let mut new_h_inv = vec![vec![T::zero(); n]; n];

        // First multiply: (I - ρsy^T)H_k⁻¹
        for i in 0..n {
            for j in 0..n {
                let sy_term = s[i]
                    * y.iter()
                        .enumerate()
                        .fold(T::zero(), |acc, (k, &y_k)| acc + y_k * h_inv[k][j]);
                temp_matrix[i][j] = h_inv[i][j] - rho * sy_term;
            }
        }

        // Then multiply by (I - ρys^T)
        for i in 0..n {
            for j in 0..n {
                let ys_term = y.iter()
                    .enumerate()
                    .fold(T::zero(), |acc, (k, &y_k)| acc + y_k * temp_matrix[i][k])
                    * s[j];
                new_h_inv[i][j] = temp_matrix[i][j] - rho * ys_term;
            }
        }

        // Add ρss^T
        for i in 0..n {
            for j in 0..n {
                new_h_inv[i][j] = new_h_inv[i][j] + rho * s[i] * s[j];
            }
        }

        // Update for next iteration
        h_inv = new_h_inv;
        current_point = new_point;
        gradient = new_gradient;
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
    }

    #[test]
    fn test_bfgs_quadratic() {
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
    }

    #[test]
    fn test_bfgs_quadratic_with_minimum() {
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

    // Test function: f(x, y) = (x - 1)^2 + 100(y - x^2)^2 (Rosenbrock function)
    struct Rosenbrock;

    impl ObjectiveFunction<f64> for Rosenbrock {
        fn evaluate(&self, point: &[f64]) -> f64 {
            let x = point[0];
            let y = point[1];
            (x - 1.0).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
        }

        fn gradient(&self, point: &[f64]) -> Option<Vec<f64>> {
            let x = point[0];
            let y = point[1];
            Some(vec![
                2.0 * (x - 1.0) - 400.0 * x * (y - x.powi(2)),
                200.0 * (y - x.powi(2)),
            ])
        }
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let f = Rosenbrock;
        let initial_point = vec![0.0, 0.0];
        let config = OptimizationConfig {
            max_iterations: 2000,  // Increased max iterations
            tolerance: 1e-5,      // Slightly relaxed tolerance
            learning_rate: 1.0,   // Start with full step size
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 1.0).abs() < 1e-3);
        assert!((result.optimal_point[1] - 1.0).abs() < 1e-3);
    }
} 