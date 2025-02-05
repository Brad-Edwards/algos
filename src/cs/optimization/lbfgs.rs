use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::cs::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using the L-BFGS (Limited-memory BFGS) method.
///
/// L-BFGS is a memory-efficient variant of BFGS that maintains a limited history
/// of position and gradient differences to approximate the inverse Hessian matrix.
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
/// use algos::cs::optimization::lbfgs::minimize;
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
    const M: usize = 10; // Number of corrections to store
    let n = initial_point.len();
    let mut current_point = initial_point.to_vec();
    let mut iterations = 0;
    let mut converged = false;

    // Storage for the last M corrections
    let mut s_list: VecDeque<Vec<T>> = VecDeque::with_capacity(M);
    let mut y_list: VecDeque<Vec<T>> = VecDeque::with_capacity(M);
    let mut rho_list: VecDeque<T> = VecDeque::with_capacity(M);

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

    while iterations < config.max_iterations {
        // Check for convergence
        let gradient_norm = gradient
            .iter()
            .fold(T::zero(), |acc, &x| acc + x * x)
            .sqrt();
        if gradient_norm < config.tolerance {
            converged = true;
            break;
        }

        // Compute search direction using L-BFGS two-loop recursion
        let mut q = gradient.clone();
        let mut alpha_list = Vec::with_capacity(s_list.len());

        // First loop
        for i in (0..s_list.len()).rev() {
            let alpha = rho_list[i]
                * s_list[i]
                    .iter()
                    .zip(q.iter())
                    .fold(T::zero(), |acc, (&s, &q)| acc + s * q);
            alpha_list.push(alpha);
            for (q_j, y_j) in q.iter_mut().zip(y_list[i].iter()) {
                *q_j = *q_j - alpha * *y_j;
            }
        }

        // Scale the initial Hessian approximation
        let mut r = if !s_list.is_empty() {
            let i = s_list.len() - 1;
            let yy = y_list[i].iter().fold(T::zero(), |acc, &y| acc + y * y);
            let ys = y_list[i]
                .iter()
                .zip(s_list[i].iter())
                .fold(T::zero(), |acc, (&y, &s)| acc + y * s);
            q.iter_mut().for_each(|r_j| *r_j = *r_j * (ys / yy));
            q
        } else {
            q.iter_mut()
                .for_each(|r_j| *r_j = *r_j * config.learning_rate);
            q
        };

        // Second loop
        for i in 0..s_list.len() {
            let beta = rho_list[i]
                * y_list[i]
                    .iter()
                    .zip(r.iter())
                    .fold(T::zero(), |acc, (&y, &r)| acc + y * r);
            let alpha = alpha_list[s_list.len() - 1 - i];
            for (r_j, s_j) in r.iter_mut().zip(s_list[i].iter()) {
                *r_j = *r_j + (alpha - beta) * *s_j;
            }
        }

        // r now contains the search direction
        let direction: Vec<T> = r.iter().map(|&x| -x).collect();

        // Line search to find step size
        let mut alpha = T::one();
        let mut new_point = vec![T::zero(); n];
        let current_value = f.evaluate(&current_point);

        // Simple backtracking line search
        for _ in 0..20 {
            for i in 0..n {
                new_point[i] = current_point[i] + alpha * direction[i];
            }
            let new_value = f.evaluate(&new_point);
            if new_value < current_value {
                break;
            }
            alpha = alpha * T::from(0.5).unwrap();
        }

        // Get new gradient
        let new_gradient = match f.gradient(&new_point) {
            Some(g) => g,
            None => break,
        };

        // Update the correction vectors
        let s = new_point
            .iter()
            .zip(current_point.iter())
            .map(|(&x_new, &x_old)| x_new - x_old)
            .collect::<Vec<T>>();
        let y = new_gradient
            .iter()
            .zip(gradient.iter())
            .map(|(&g_new, &g_old)| g_new - g_old)
            .collect::<Vec<T>>();

        let ys = y
            .iter()
            .zip(s.iter())
            .fold(T::zero(), |acc, (&y_i, &s_i)| acc + y_i * s_i);
        let rho = T::one() / ys;

        if s_list.len() == M {
            s_list.pop_front();
            y_list.pop_front();
            rho_list.pop_front();
        }
        s_list.push_back(s);
        y_list.push_back(y);
        rho_list.push_back(rho);

        // Update for next iteration
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
    fn test_lbfgs_quadratic() {
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
    fn test_lbfgs_quadratic_with_minimum() {
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
    fn test_lbfgs_rosenbrock() {
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

    // Test high-dimensional optimization
    struct HighDimensionalQuadratic;

    impl ObjectiveFunction<f64> for HighDimensionalQuadratic {
        fn evaluate(&self, point: &[f64]) -> f64 {
            point
                .iter()
                .enumerate()
                .map(|(i, &x)| (i + 1) as f64 * x * x)
                .sum()
        }

        fn gradient(&self, point: &[f64]) -> Option<Vec<f64>> {
            Some(
                point
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| 2.0 * (i + 1) as f64 * x)
                    .collect(),
            )
        }
    }

    #[test]
    fn test_lbfgs_high_dimensional() {
        let f = HighDimensionalQuadratic;
        let initial_point = vec![1.0; 100];
        let config = OptimizationConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!(result.optimal_value < 1e-6);
        for x in result.optimal_point {
            assert!(x.abs() < 1e-3);
        }
    }
}
