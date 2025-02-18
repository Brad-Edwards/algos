use num_traits::Float;
use std::fmt::Debug;

use crate::math::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using the nonlinear conjugate gradient method.
///
/// The conjugate gradient method is particularly effective for large-scale optimization problems
/// and combines the advantages of steepest descent and Newton's method.
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
/// use algos::math::optimization::conjugate_gradient::minimize;
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

    // Initialize direction as negative gradient
    let mut direction: Vec<T> = gradient.iter().map(|&x| -x).collect();
    let mut prev_gradient_norm_sq = gradient.iter().fold(T::zero(), |acc, &x| acc + x * x);

    while iterations < config.max_iterations {
        // Line search to find step size
        let mut alpha = config.learning_rate;
        let mut new_point = vec![T::zero(); n];
        let current_value = f.evaluate(&current_point);

        // Simple backtracking line search
        for _ in 0..20 {
            new_point
                .iter_mut()
                .zip(current_point.iter().zip(direction.iter()))
                .for_each(|(new, (&curr, &dir))| *new = curr + alpha * dir);
            let new_value = f.evaluate(&new_point);
            if new_value < current_value {
                break;
            }
            alpha = alpha * T::from(0.5).unwrap();
        }

        // Update point
        current_point = new_point;

        // Get new gradient
        let new_gradient = match f.gradient(&current_point) {
            Some(g) => g,
            None => break,
        };

        // Check for convergence
        let gradient_norm = new_gradient
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
            .sqrt();
        if gradient_norm < config.tolerance {
            converged = true;
            break;
        }

        // Compute beta using Polak-Ribière formula
        let new_gradient_norm_sq = new_gradient.iter().fold(T::zero(), |acc, &x| acc + x * x);
        let beta_numerator = new_gradient
            .iter()
            .zip(gradient.iter())
            .fold(T::zero(), |acc, (&new_g, &old_g)| {
                acc + new_g * (new_g - old_g)
            });
        let beta = (beta_numerator / prev_gradient_norm_sq).max(T::zero());

        // Update direction using conjugate gradient formula
        direction
            .iter_mut()
            .zip(new_gradient.iter())
            .for_each(|(d, &g)| *d = -g + beta * *d);

        // Update for next iteration
        gradient = new_gradient;
        prev_gradient_norm_sq = new_gradient_norm_sq;
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
    fn test_conjugate_gradient_quadratic() {
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
    fn test_conjugate_gradient_quadratic_with_minimum() {
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
    fn test_conjugate_gradient_rosenbrock() {
        let f = Rosenbrock;
        let initial_point = vec![0.0, 0.0];
        let config = OptimizationConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 1.0).abs() < 1e-3);
        assert!((result.optimal_point[1] - 1.0).abs() < 1e-3);
    }
}
