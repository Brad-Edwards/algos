use num_traits::Float;
use std::fmt::Debug;

use crate::math::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using gradient descent.
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
/// use algos::cs::optimization::gradient_descent::minimize;
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
#[must_use]
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

    while iterations < config.max_iterations {
        // Get gradient at current point
        let gradient = match f.gradient(&current_point) {
            Some(g) => g,
            None => {
                // If gradient is not available, we can't continue
                break;
            }
        };

        // Check for convergence
        let gradient_norm = gradient
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
            .sqrt();
        if gradient_norm < config.tolerance {
            converged = true;
            break;
        }

        // Update point using gradient descent
        for (i, grad) in gradient.iter().enumerate() {
            current_point[i] = current_point[i] - config.learning_rate * *grad;
        }

        iterations += 1;
    }

    let optimal_value = f.evaluate(&current_point);
    OptimizationResult {
        optimal_point: current_point,
        optimal_value,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

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
    fn test_gradient_descent_quadratic() {
        let f = Quadratic;
        let initial_point = vec![1.0, 1.0];
        let config = OptimizationConfig {
            max_iterations: 10000,
            tolerance: 1e-10,
            learning_rate: 0.05,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!(result.optimal_value.abs() < 10.0 * EPSILON);
        for x in result.optimal_point {
            assert!(x.abs() < (1e-5_f64).sqrt());
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
    fn test_gradient_descent_quadratic_with_minimum() {
        let f = QuadraticWithMinimum;
        let initial_point = vec![0.0];
        let config = OptimizationConfig {
            max_iterations: 10000,
            tolerance: 1e-10,
            learning_rate: 0.05,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 2.0).abs() < (1e-5_f64).sqrt());
        assert!(result.optimal_value.abs() < 10.0 * EPSILON);
    }
}
