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
/// use algos::math::optimization::{OptimizationConfig};
/// use algos::math::optimization::simplex::{LinearProgram, minimize};
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
        for (i, row) in tableau.iter().enumerate().skip(1).take(m) {
            let coef = row[entering_col];
            if coef > eps {
                let ratio = row[n + m] / coef;
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

        println!(
            "Iteration {}: pivot ({}, {})",
            iterations, leaving_row, entering_col
        );
        println!("Before pivot: {:?}", tableau);

        // Pivot
        pivot(&mut tableau, leaving_row, entering_col);

        println!("After pivot: {:?}", tableau);

        iterations += 1;
    }

    println!("Final tableau: {:?}", tableau);

    // Extract solution
    let optimal_point = extract_solution(&tableau, m, n);
    let optimal_value = tableau[0][n + m];

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
    for row in tableau.iter().skip(1).take(m) {
        let mut count = 0;
        let mut last_nonzero = None;
        for (j, &val) in row.iter().take(n).enumerate() {
            if val.abs() > T::epsilon() {
                count += 1;
                last_nonzero = Some(j);
            }
        }
        if count == 1 {
            if let Some(j) = last_nonzero {
                solution[j] = row[n + m] / row[j];
            }
        }
    }
    solution
}
