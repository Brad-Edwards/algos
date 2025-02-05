use num_traits::Float;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fmt::Debug;

use crate::cs::optimization::{ObjectiveFunction, OptimizationConfig, OptimizationResult};

/// Configuration specific to simulated annealing.
#[derive(Debug, Clone)]
pub struct AnnealingConfig<T>
where
    T: Float + Debug,
{
    /// Initial temperature
    pub initial_temperature: T,
    /// Temperature reduction factor
    pub cooling_rate: T,
    /// Number of iterations at each temperature
    pub iterations_per_temp: usize,
    /// Lower bounds for each dimension
    pub lower_bounds: Vec<T>,
    /// Upper bounds for each dimension
    pub upper_bounds: Vec<T>,
}

impl<T> Default for AnnealingConfig<T>
where
    T: Float + Debug,
{
    fn default() -> Self {
        Self {
            initial_temperature: T::from(100.0).unwrap(),
            cooling_rate: T::from(0.95).unwrap(),
            iterations_per_temp: 50,
            lower_bounds: vec![T::from(-10.0).unwrap()],
            upper_bounds: vec![T::from(10.0).unwrap()],
        }
    }
}

/// Minimizes an objective function using Simulated Annealing.
///
/// Simulated Annealing is a probabilistic optimization method that mimics the
/// physical process of annealing in metallurgy. It can escape local minima by
/// occasionally accepting worse solutions based on a temperature parameter.
///
/// # Arguments
///
/// * `f` - The objective function to minimize
/// * `initial_point` - The starting point for optimization
/// * `config` - Configuration options for the optimization process
/// * `sa_config` - Configuration specific to simulated annealing
///
/// # Returns
///
/// Returns an `OptimizationResult` containing the optimal point found and optimization statistics.
///
/// # Examples
///
/// ```
/// use algos::cs::optimization::{ObjectiveFunction, OptimizationConfig};
/// use algos::cs::optimization::simulated_annealing::{AnnealingConfig, minimize};
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
/// let sa_config = AnnealingConfig::default();
///
/// let result = minimize(&f, &initial_point, &config, &sa_config);
/// assert!(result.converged);
/// ```
pub fn minimize<T, F>(
    f: &F,
    initial_point: &[T],
    config: &OptimizationConfig<T>,
    sa_config: &AnnealingConfig<T>,
) -> OptimizationResult<T>
where
    T: Float + Debug,
    F: ObjectiveFunction<T>,
{
    let mut rng = rand::thread_rng();
    let _n = initial_point.len();

    let mut current_point = initial_point.to_vec();
    let mut current_value = f.evaluate(&current_point);

    let mut best_point = current_point.clone();
    let mut best_value = current_value;

    let mut temperature = sa_config.initial_temperature;
    let mut iterations = 0;
    let mut converged = false;

    while iterations < config.max_iterations {
        let mut improved = false;

        for _ in 0..sa_config.iterations_per_temp {
            // Generate neighbor
            let neighbor = generate_neighbor(
                &current_point,
                temperature,
                &sa_config.lower_bounds,
                &sa_config.upper_bounds,
                &mut rng,
            );
            let neighbor_value = f.evaluate(&neighbor);

            // Compute acceptance probability
            let delta = neighbor_value - current_value;
            let accept = if delta <= T::zero() {
                true
            } else {
                let probability = (-delta / temperature).exp();
                rng.gen::<f64>() < probability.to_f64().unwrap()
            };

            // Update current solution
            if accept {
                current_point = neighbor;
                current_value = neighbor_value;

                // Update best solution
                if current_value < best_value {
                    best_point = current_point.clone();
                    best_value = current_value;
                    improved = true;
                }
            }
        }

        // Check for convergence
        if temperature < config.tolerance || (iterations > 0 && !improved) {
            converged = true;
            break;
        }

        // Cool down
        temperature = temperature * sa_config.cooling_rate;
        iterations += 1;
    }

    OptimizationResult {
        optimal_point: best_point,
        optimal_value: best_value,
        iterations,
        converged,
    }
}

// Generate neighbor solution
fn generate_neighbor<T, R: Rng>(
    point: &[T],
    temperature: T,
    lower_bounds: &[T],
    upper_bounds: &[T],
    rng: &mut R,
) -> Vec<T>
where
    T: Float + Debug,
{
    let n = point.len();
    let mut neighbor = Vec::with_capacity(n);

    for i in 0..n {
        let lower = lower_bounds[i.min(lower_bounds.len() - 1)].to_f64().unwrap();
        let upper = upper_bounds[i.min(upper_bounds.len() - 1)].to_f64().unwrap();
        let current = point[i].to_f64().unwrap();

        // Scale perturbation with temperature
        let range = (upper - lower) * temperature.to_f64().unwrap();
        let dist = Uniform::new(-range, range);
        let perturbation = dist.sample(rng);

        // Ensure new point is within bounds
        let new_value = (current + perturbation).max(lower).min(upper);
        neighbor.push(T::from(new_value).unwrap());
    }

    neighbor
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
    fn test_simulated_annealing_quadratic() {
        let f = Quadratic;
        let initial_point = vec![1.0, 1.0];
        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };
        let sa_config = AnnealingConfig {
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            iterations_per_temp: 50,
            lower_bounds: vec![-10.0, -10.0],
            upper_bounds: vec![10.0, 10.0],
        };

        let result = minimize(&f, &initial_point, &config, &sa_config);

        assert!(result.converged);
        assert!(result.optimal_value < 1e-4);
        for x in result.optimal_point {
            assert!(x.abs() < 1e-2);
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
    fn test_simulated_annealing_quadratic_with_minimum() {
        let f = QuadraticWithMinimum;
        let initial_point = vec![0.0];
        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };
        let sa_config = AnnealingConfig {
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            iterations_per_temp: 50,
            lower_bounds: vec![-10.0],
            upper_bounds: vec![10.0],
        };

        let result = minimize(&f, &initial_point, &config, &sa_config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 2.0).abs() < 1e-2);
        assert!(result.optimal_value < 1e-4);
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
    fn test_simulated_annealing_rosenbrock() {
        let f = Rosenbrock;
        let initial_point = vec![0.0, 0.0];
        let config = OptimizationConfig {
            max_iterations: 200,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };
        let sa_config = AnnealingConfig {
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            iterations_per_temp: 100,
            lower_bounds: vec![-10.0, -10.0],
            upper_bounds: vec![10.0, 10.0],
        };

        let result = minimize(&f, &initial_point, &config, &sa_config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 1.0).abs() < 1e-1);
        assert!((result.optimal_point[1] - 1.0).abs() < 1e-1);
    }

    // Test function with multiple local minima
    struct MultiModal;

    impl ObjectiveFunction<f64> for MultiModal {
        fn evaluate(&self, point: &[f64]) -> f64 {
            let x = point[0];
            let y = point[1];
            (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2)
        }
    }

    #[test]
    fn test_simulated_annealing_multimodal() {
        let f = MultiModal;
        let initial_point = vec![0.0, 0.0];
        let config = OptimizationConfig {
            max_iterations: 200,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };
        let sa_config = AnnealingConfig {
            initial_temperature: 100.0,
            cooling_rate: 0.95,
            iterations_per_temp: 100,
            lower_bounds: vec![-10.0, -10.0],
            upper_bounds: vec![10.0, 10.0],
        };

        let result = minimize(&f, &initial_point, &config, &sa_config);

        assert!(result.converged);
        assert!(result.optimal_value < 1.0); // Multiple global minima with value 0
    }
} 