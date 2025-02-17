use num_traits::Float;
use std::fmt::Debug;

use crate::math::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Minimizes an objective function using the Nelder-Mead simplex method.
///
/// The Nelder-Mead method is a derivative-free optimization algorithm that uses
/// a simplex of n+1 points to explore the n-dimensional space and find a minimum.
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
/// use algos::math::optimization::nelder_mead::minimize;
///
/// // Define a simple quadratic function
/// struct Quadratic;
///
/// impl ObjectiveFunction<f64> for Quadratic {
///     fn evaluate(&self, point: &[f64]) -> f64 {
///         point.iter().map(|x| x * x).sum()
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
    // Nelder-Mead parameters (adjusted for better performance)
    let alpha = T::from(1.0).unwrap(); // reflection coefficient
    let gamma = T::from(2.0).unwrap(); // expansion coefficient
    let rho = T::from(0.5).unwrap(); // contraction coefficient
    let sigma = T::from(0.5).unwrap(); // shrink coefficient

    let n = initial_point.len();
    let mut iterations = 0;
    let mut converged = false;

    // Initialize simplex
    let mut simplex = initialize_simplex(initial_point);
    let mut values = evaluate_simplex(f, &simplex);
    let mut best_value = values[0];
    let mut best_point = simplex[0].clone();

    while iterations < config.max_iterations {
        // Order vertices by function value
        order_simplex(&mut simplex, &mut values);

        // Update best point if we found a better one
        if values[0] < best_value {
            best_value = values[0];
            best_point = simplex[0].clone();
        }

        // Compute centroid of all points except worst
        let centroid = compute_centroid(&simplex[..n]);

        // Check for convergence using multiple criteria
        let size_measure = compute_simplex_size(&simplex);
        let value_range = values[n] - values[0];

        if size_measure < config.tolerance && value_range < config.tolerance {
            converged = true;
            break;
        }

        // Reflection
        let reflected = reflect(&centroid, &simplex[n], alpha);
        let reflected_value = f.evaluate(&reflected);

        if values[0] <= reflected_value && reflected_value < values[n - 1] {
            // Accept reflection
            simplex[n] = reflected;
            values[n] = reflected_value;
        } else if reflected_value < values[0] {
            // Try expansion
            let expanded = reflect(&centroid, &simplex[n], gamma);
            let expanded_value = f.evaluate(&expanded);

            if expanded_value < reflected_value {
                simplex[n] = expanded;
                values[n] = expanded_value;
            } else {
                simplex[n] = reflected;
                values[n] = reflected_value;
            }
        } else {
            // Try contraction
            let contracted = reflect(&centroid, &simplex[n], -rho);
            let contracted_value = f.evaluate(&contracted);

            if contracted_value < values[n] {
                simplex[n] = contracted;
                values[n] = contracted_value;
            } else {
                // Shrink towards best point
                let mut best = simplex[0].clone();
                let mut best_value = values[0];
                for i in 1..n + 1 {
                    if values[i] < best_value {
                        best = simplex[i].clone();
                        best_value = values[i];
                    }
                }
                for i in 1..=n {
                    for (j, val) in best.iter_mut().enumerate().take(n) {
                        *val = centroid[j] + sigma * (centroid[j] - simplex[i][j]);
                    }
                    values[i] = f.evaluate(&best);
                }
            }
        }

        iterations += 1;
    }

    // Return best point found
    OptimizationResult {
        optimal_point: best_point,
        optimal_value: best_value,
        iterations,
        converged,
    }
}

// Initialize simplex around initial point with improved scaling
fn initialize_simplex<T>(initial_point: &[T]) -> Vec<Vec<T>>
where
    T: Float + Debug,
{
    let n = initial_point.len();
    let mut simplex = vec![initial_point.to_vec()];

    // Create n additional vertices with better scaling
    let scale = T::from(0.1).unwrap(); // Increased scale for better exploration
    for i in 0..n {
        let mut vertex = initial_point.to_vec();
        if vertex[i] == T::zero() {
            vertex[i] = scale;
        } else {
            vertex[i] = vertex[i] * (T::one() + scale);
        }
        simplex.push(vertex);
    }

    simplex
}

// Evaluate function at all simplex vertices
fn evaluate_simplex<T, F>(f: &F, simplex: &[Vec<T>]) -> Vec<T>
where
    T: Float + Debug,
    F: ObjectiveFunction<T>,
{
    simplex.iter().map(|x| f.evaluate(x)).collect()
}

// Order simplex vertices by function value
fn order_simplex<T>(simplex: &mut [Vec<T>], values: &mut [T])
where
    T: Float + Debug,
{
    let n = values.len() - 1;
    for i in 0..n {
        for j in 0..n - i {
            if values[j] > values[j + 1] {
                values.swap(j, j + 1);
                simplex.swap(j, j + 1);
            }
        }
    }
}

// Compute centroid of points
fn compute_centroid<T>(points: &[Vec<T>]) -> Vec<T>
where
    T: Float + Debug,
{
    let n = points[0].len();
    let m = points.len();
    let mut centroid = vec![T::zero(); n];

    for i in 0..n {
        for point in points.iter() {
            centroid[i] = centroid[i] + point[i];
        }
        centroid[i] = centroid[i] / T::from(m).unwrap();
    }

    centroid
}

// Reflect point through centroid
fn reflect<T>(centroid: &[T], point: &[T], coefficient: T) -> Vec<T>
where
    T: Float + Debug,
{
    centroid
        .iter()
        .zip(point.iter())
        .map(|(&c, &p)| c + coefficient * (c - p))
        .collect()
}

// Compute size of simplex as maximum distance between any two vertices
fn compute_simplex_size<T>(simplex: &[Vec<T>]) -> T
where
    T: Float + Debug,
{
    let n = simplex.len();
    let mut max_dist = T::zero();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = simplex[i]
                .iter()
                .zip(simplex[j].iter())
                .fold(T::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y))
                .sqrt();
            max_dist = max_dist.max(dist);
        }
    }

    max_dist
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
    }

    #[test]
    fn test_nelder_mead_quadratic() {
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
    }

    #[test]
    fn test_nelder_mead_quadratic_with_minimum() {
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
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let f = Rosenbrock;
        let initial_point = vec![0.0, 0.0];
        let config = OptimizationConfig {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&f, &initial_point, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 1.0).abs() < 1e-3);
        assert!((result.optimal_point[1] - 1.0).abs() < 1e-3);
    }
}
