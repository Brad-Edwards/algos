use num_traits::Float;
use std::fmt::Debug;

use crate::math::optimization::{OptimizationConfig, OptimizationResult};

/// A linear programming problem in standard form.
#[derive(Debug, Clone)]
pub struct LinearProgram<T>
where
    T: Float + Debug,
{
    /// The objective function coefficients (c in min c^T x)
    pub objective: Vec<T>,
    /// The constraint matrix (A in Ax ≤ b)
    pub constraints: Vec<Vec<T>>,
    /// The right-hand side vector (b in Ax ≤ b)
    pub rhs: Vec<T>,
}

const EPSILON: f64 = 1e-10;

/// Minimizes a linear program using the Simplex Method.
///
/// The Simplex Method solves linear programming problems in standard form:
/// minimize c^T x
/// subject to Ax ≤ b
///           x ≥ 0
///
/// # Arguments
///
/// * `lp` - The linear program to solve
/// * `config` - Configuration options for the optimization process
///
/// # Returns
///
/// Returns an `OptimizationResult` containing the optimal point found and optimization statistics.
///
/// # Examples
///
/// ```
/// use algos::cs::optimization::{OptimizationConfig};
/// use algos::cs::optimization::simplex::{LinearProgram, minimize};
///
/// // Solve the linear program:
/// // minimize -x - y
/// // subject to:
/// //   x + y ≤ 1
/// //   x, y ≥ 0
///
/// let lp = LinearProgram {
///     objective: vec![-1.0, -1.0],
///     constraints: vec![vec![1.0, 1.0]],
///     rhs: vec![1.0],
/// };
///
/// let config = OptimizationConfig::default();
/// let result = minimize(&lp, &config);
/// assert!(result.converged);
/// ```
pub fn minimize<T>(lp: &LinearProgram<T>, config: &OptimizationConfig<T>) -> OptimizationResult<T>
where
    T: Float + Debug,
{
    // Convert minimization to maximization by negating objective
    let max_lp = LinearProgram {
        objective: lp.objective.iter().map(|&c| -c).collect(),
        constraints: lp.constraints.clone(),
        rhs: lp.rhs.clone(),
    };

    println!("Simplex solver: Starting minimization");
    println!("Original objective: {:?}", lp.objective);
    println!("Negated objective: {:?}", max_lp.objective);
    println!("Constraints: {:?}", lp.constraints);
    println!("RHS: {:?}", lp.rhs);

    // Solve maximization problem
    let result = maximize(&max_lp, config);

    println!("Maximization result:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Optimal value: {:?}", result.optimal_value);
    println!("  Solution: {:?}", result.optimal_point);

    // Convert result back to minimization
    OptimizationResult {
        optimal_point: result.optimal_point,
        optimal_value: -result.optimal_value,
        iterations: result.iterations,
        converged: result.converged,
    }
}

fn maximize<T>(lp: &LinearProgram<T>, config: &OptimizationConfig<T>) -> OptimizationResult<T>
where
    T: Float + Debug,
{
    let m = lp.constraints.len(); // Number of constraints
    let n = lp.objective.len(); // Number of original variables

    let mut tableau = initialize_tableau(lp);
    let mut iterations = 0;
    let mut converged = false;
    let eps = T::from(EPSILON).unwrap();

    println!("Initial tableau: {:?}", tableau);

    // Phase II: Optimize
    while iterations < config.max_iterations {
        // Find entering variable (most negative reduced cost)
        let mut entering_col = None;
        let mut min_reduced_cost = T::zero();
        for j in 0..n {
            if tableau[0][j] < min_reduced_cost {
                min_reduced_cost = tableau[0][j];
                entering_col = Some(j);
            }
        }

        // Check optimality
        if entering_col.is_none() {
            converged = true;
            break;
        }
        let entering_col = entering_col.unwrap();

        // Find leaving variable (minimum ratio test)
        let mut leaving_row = None;
        let mut min_ratio = T::infinity();
        for i in 1..=m {
            let coef = tableau[i][entering_col];
            if coef > eps {
                let ratio = tableau[i][n + m] / coef;
                if ratio < min_ratio {
                    min_ratio = ratio;
                    leaving_row = Some(i);
                }
            }
        }

        // Check unboundedness
        if leaving_row.is_none() {
            break;
        }
        let leaving_row = leaving_row.unwrap();

        println!("Iteration {}: pivot ({}, {})", iterations, leaving_row, entering_col);
        println!("Before pivot: {:?}", tableau);

        // Pivot
        pivot(&mut tableau, leaving_row, entering_col);

        println!("After pivot: {:?}", tableau);

        iterations += 1;
    }

    println!("Final tableau: {:?}", tableau);

    // Extract solution
    let optimal_point = extract_solution(&tableau, m, n);
    let optimal_value = -tableau[0][n + m]; // Negate since we're maximizing

    OptimizationResult {
        optimal_point,
        optimal_value,
        iterations,
        converged,
    }
}

// Initialize the simplex tableau with slack variables
fn initialize_tableau<T>(lp: &LinearProgram<T>) -> Vec<Vec<T>>
where
    T: Float + Debug,
{
    let m = lp.constraints.len();
    let n = lp.objective.len();
    let mut tableau = vec![vec![T::zero(); n + m + 1]; m + 1];

    // Set objective row (negated for maximization)
    for j in 0..n {
        tableau[0][j] = -lp.objective[j];
    }

    // Set constraint rows
    for i in 0..m {
        // Set original variables
        for j in 0..n {
            tableau[i + 1][j] = lp.constraints[i][j];
        }
        // Set slack variables
        tableau[i + 1][n + i] = T::one();
        // Set RHS
        tableau[i + 1][n + m] = lp.rhs[i];
    }

    tableau
}

// Pivot the tableau around the given element
fn pivot<T>(tableau: &mut [Vec<T>], pivot_row: usize, pivot_col: usize)
where
    T: Float + Debug,
{
    let m = tableau.len() - 1;
    let n = tableau[0].len() - 1;

    // Get pivot element
    let pivot_element = tableau[pivot_row][pivot_col];

    // Normalize pivot row
    for j in 0..=n {
        tableau[pivot_row][j] = tableau[pivot_row][j] / pivot_element;
    }

    // Update other rows
    for i in 0..=m {
        if i != pivot_row {
            let factor = tableau[i][pivot_col];
            for j in 0..=n {
                tableau[i][j] = tableau[i][j] - factor * tableau[pivot_row][j];
            }
        }
    }
}

// Extract solution from tableau
fn extract_solution<T>(tableau: &[Vec<T>], m: usize, n: usize) -> Vec<T>
where
    T: Float + Debug,
{
    let mut solution = vec![T::zero(); n];
    for j in 0..n {
        let mut value = T::zero();
        let mut basic_row = None;
        for i in 1..=m {
            if (tableau[i][j] - T::one()).abs() < T::from(EPSILON).unwrap() {
                if basic_row.is_none() {
                    basic_row = Some(i);
                } else {
                    // Not a basic variable
                    basic_row = None;
                    break;
                }
            } else if tableau[i][j].abs() > T::from(EPSILON).unwrap() {
                // Not a basic variable
                basic_row = None;
                break;
            }
        }
        if let Some(row) = basic_row {
            value = tableau[row][n + m];
        }
        solution[j] = value;
    }
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_lp() {
        // Solve:
        // minimize -x - y
        // subject to:
        //   x + y ≤ 1
        //   x, y ≥ 0
        let lp = LinearProgram {
            objective: vec![-1.0, -1.0],
            constraints: vec![vec![1.0, 1.0]],
            rhs: vec![1.0],
        };

        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&lp, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 0.5).abs() < 1e-6);
        assert!((result.optimal_point[1] - 0.5).abs() < 1e-6);
        assert!((result.optimal_value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounded_lp() {
        // Solve:
        // minimize -2x - y
        // subject to:
        //   x + y ≤ 2
        //   x ≤ 1
        //   x, y ≥ 0
        let lp = LinearProgram {
            objective: vec![-2.0, -1.0],
            constraints: vec![vec![1.0, 1.0], vec![1.0, 0.0]],
            rhs: vec![2.0, 1.0],
        };

        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&lp, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 1.0).abs() < 1e-6);
        assert!((result.optimal_point[1] - 1.0).abs() < 1e-6);
        assert!((result.optimal_value + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_degenerate_lp() {
        // Solve:
        // minimize -x - y
        // subject to:
        //   x + y ≤ 1
        //   x ≤ 0.5
        //   y ≤ 0.5
        //   x, y ≥ 0
        let lp = LinearProgram {
            objective: vec![-1.0, -1.0],
            constraints: vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            rhs: vec![1.0, 0.5, 0.5],
        };

        let config = OptimizationConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 1.0,
        };

        let result = minimize(&lp, &config);

        assert!(result.converged);
        assert!((result.optimal_point[0] - 0.5).abs() < 1e-6);
        assert!((result.optimal_point[1] - 0.5).abs() < 1e-6);
        assert!((result.optimal_value + 1.0).abs() < 1e-6);
    }
}
