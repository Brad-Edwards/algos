pub mod gradient_descent;
pub mod newton;
pub mod conjugate_gradient;
pub mod bfgs;
pub mod lbfgs;
pub mod simplex;
pub mod interior_point;
pub mod nelder_mead;
pub mod genetic;
pub mod simulated_annealing;

use num_traits::{Float, Zero};
use std::fmt::Debug;

pub use gradient_descent::minimize as gradient_descent_minimize;
pub use newton::minimize as newton_minimize;
pub use conjugate_gradient::minimize as conjugate_gradient_minimize;
pub use bfgs::minimize as bfgs_minimize;
pub use lbfgs::minimize as lbfgs_minimize;
pub use simplex::minimize as simplex_minimize;
pub use interior_point::minimize as interior_point_minimize;
pub use nelder_mead::minimize as nelder_mead_minimize;
pub use genetic::minimize as genetic_minimize;
pub use simulated_annealing::minimize as simulated_annealing_minimize;

/// A trait for objective functions that can be optimized.
pub trait ObjectiveFunction<T>
where
    T: Float + Debug,
{
    /// Evaluates the objective function at the given point.
    fn evaluate(&self, point: &[T]) -> T;

    /// Computes the gradient of the objective function at the given point.
    /// Returns None if the gradient is not available.
    fn gradient(&self, _point: &[T]) -> Option<Vec<T>> {
        None
    }

    /// Computes the Hessian matrix at the given point.
    /// Returns None if the Hessian is not available.
    fn hessian(&self, _point: &[T]) -> Option<Vec<Vec<T>>> {
        None
    }
}

/// Configuration options for optimization algorithms.
#[derive(Debug, Clone)]
pub struct OptimizationConfig<T>
where
    T: Float + Debug,
{
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: T,
    /// Learning rate (step size) for gradient-based methods
    pub learning_rate: T,
}

impl<T> Default for OptimizationConfig<T>
where
    T: Float + Debug + Zero,
{
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: T::from(1e-6).unwrap(),
            learning_rate: T::from(0.01).unwrap(),
        }
    }
}

/// Result of an optimization process.
#[derive(Debug, Clone)]
pub struct OptimizationResult<T>
where
    T: Float + Debug,
{
    /// The optimal point found
    pub optimal_point: Vec<T>,
    /// The value of the objective function at the optimal point
    pub optimal_value: T,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the optimization converged
    pub converged: bool,
}
