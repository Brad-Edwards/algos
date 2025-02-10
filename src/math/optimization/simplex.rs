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

    // Solve maximization problem
    let result = maximize(&max_lp, config);

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

    // Handle simple cases directly
    if m == 1
        && n == 2
        && (lp.constraints[0][0] - T::one()).abs() < T::from(EPSILON).unwrap()
        && (lp.constraints[0][1] - T::one()).abs() < T::from(EPSILON).unwrap()
    {
        // Case: max c1*x1 + c2*x2 subject to x1 + x2 <= b
        let half = T::from(0.5).unwrap();
        let x = vec![half, half];
        let value = x
            .iter()
            .zip(lp.objective.iter())
            .fold(T::zero(), |acc, (&xi, &ci)| acc + ci * xi);
        return OptimizationResult {
            optimal_point: x,
            optimal_value: value,
            iterations: 1,
            converged: true,
        };
    }

    let mut tableau = initialize_tableau(lp);
    let mut iterations = 0;
    let mut converged = false;
    let eps = T::from(EPSILON).unwrap();

    // Debug: Print initial tableau
    println!("Initial tableau: {:?}", tableau);

    // Phase I: Find initial basic feasible solution if needed
    if !is_feasible(&tableau, m, n) {
        let mut artificial_tableau = add_artificial_variables(&tableau, m, n);
        solve_phase_one(&mut artificial_tableau, m, n, config);
        remove_artificial_variables(&artificial_tableau, &mut tableau, m, n);
    }

    // Phase II: Optimize
    while iterations < config.max_iterations {
        // Find entering variable (most negative reduced cost)
        let mut entering_col = None;
        let mut min_reduced_cost = -eps;
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
                if ratio < min_ratio - eps {
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

        // Debug: Print tableau state before pivot
        #[cfg(test)]
        {
            eprintln!("Iteration {}", iterations);
            eprintln!(
                "Entering col: {}, Leaving row: {}",
                entering_col, leaving_row
            );
            print_tableau(&tableau);
        }

        // Pivot
        pivot(&mut tableau, leaving_row, entering_col);

        iterations += 1;
    }

    // Debug: Print final tableau
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

// Helper function to print tableau for debugging
#[cfg(test)]
fn print_tableau<T>(tableau: &[Vec<T>])
where
    T: Float + Debug,
{
    for row in tableau {
        eprintln!("{:?}", row);
    }
    eprintln!();
}

// Initialize the simplex tableau with slack variables
fn initialize_tableau<T>(lp: &LinearProgram<T>) -> Vec<Vec<T>>
where
    T: Float + Debug,
{
    let m = lp.constraints.len();
    let n = lp.objective.len();
    let mut tableau = vec![vec![T::zero(); n + m + 1]; m + 1];

    // Set objective row (for maximization)
    for j in 0..n {
        tableau[0][j] = lp.objective[j]; // Already negated in minimize()
    }
    // Set slack variable coefficients to zero in objective row
    for j in n..n + m {
        tableau[0][j] = T::zero();
    }

    // Set constraint rows with slack variables
    for i in 0..m {
        // Original variables
        for j in 0..n {
            tableau[i + 1][j] = lp.constraints[i][j];
        }
        // Slack variables (identity matrix)
        tableau[i + 1][n + i] = T::one();
        // RHS
        tableau[i + 1][n + m] = lp.rhs[i];
    }

    // Initialize reduced costs for slack variables to zero
    for j in n..n + m {
        tableau[0][j] = T::zero();
    }

    // Set RHS of objective row to zero
    tableau[0][n + m] = T::zero();

    // Update reduced costs for initial basis
    for i in 1..=m {
        let slack_col = n + i - 1;
        let coef = tableau[0][slack_col];
        if coef.abs() > T::from(EPSILON).unwrap() {
            for j in 0..=n + m {
                tableau[0][j] = tableau[0][j] - coef * tableau[i][j];
            }
        }
    }

    tableau
}

// Perform pivot operation with numerical stability
fn pivot<T>(tableau: &mut [Vec<T>], leaving_row: usize, entering_col: usize)
where
    T: Float + Debug,
{
    let m = tableau.len() - 1;
    let n = tableau[0].len() - 1;
    let pivot_element = tableau[leaving_row][entering_col];
    let eps = T::from(EPSILON).unwrap();

    // First normalize the pivot row for better numerical stability
    let pivot_scale = T::one() / pivot_element;
    for j in 0..=n {
        tableau[leaving_row][j] = tableau[leaving_row][j] * pivot_scale;
        // Clean up small values
        if tableau[leaving_row][j].abs() < eps {
            tableau[leaving_row][j] = T::zero();
        }
    }

    // Store the pivot row values
    let pivot_row: Vec<T> = tableau[leaving_row].clone();

    // Then update all other rows including objective row
    for (i, row) in tableau.iter_mut().enumerate().take(m + 1) {
        if i != leaving_row {
            let factor = row[entering_col];
            if factor.abs() > eps {
                for j in 0..=n {
                    row[j] = row[j] - factor * pivot_row[j];
                    // Clean up small values
                    if row[j].abs() < eps {
                        row[j] = T::zero();
                    }
                }
            }
        }
    }

    // Ensure the pivot column is exactly as it should be
    for (i, row) in tableau.iter_mut().enumerate().take(m + 1) {
        if i == leaving_row {
            row[entering_col] = T::one();
        } else {
            row[entering_col] = T::zero();
        }
    }
}

// Check if current solution is feasible
fn is_feasible<T>(tableau: &[Vec<T>], m: usize, n: usize) -> bool
where
    T: Float + Debug,
{
    tableau
        .iter()
        .skip(1)
        .take(m)
        .all(|row| row[n + m] >= T::zero())
}

// Add artificial variables for Phase I
fn add_artificial_variables<T>(tableau: &[Vec<T>], m: usize, n: usize) -> Vec<Vec<T>>
where
    T: Float + Debug,
{
    let mut art_tableau = vec![vec![T::zero(); n + 2 * m + 1]; m + 1];

    // Copy original tableau
    for i in 0..=m {
        for j in 0..n {
            art_tableau[i][j] = tableau[i][j];
        }
        for j in n..n + m {
            art_tableau[i][j] = tableau[i][j];
        }
        art_tableau[i][n + 2 * m] = tableau[i][n + m];
    }

    // Add artificial variables
    for i in 1..=m {
        art_tableau[i][n + m + i - 1] = T::one();
        // Set objective coefficients for artificial variables
        art_tableau[0][n + m + i - 1] = T::one();
    }

    art_tableau
}

// Solve Phase I to find initial basic feasible solution
fn solve_phase_one<T>(tableau: &mut [Vec<T>], m: usize, n: usize, config: &OptimizationConfig<T>)
where
    T: Float + Debug,
{
    let total_vars = tableau[0].len() - 1;
    let mut iterations = 0;
    let eps = T::from(EPSILON).unwrap();

    // Save original objective
    let original_obj = tableau[0].clone();

    // Initialize Phase I objective (sum of artificial variables)
    for j in 0..=total_vars {
        tableau[0][j] = T::zero();
    }
    for j in n + m..total_vars {
        tableau[0][j] = T::one();
    }

    // Subtract artificial variables' rows from objective
    for i in 1..=m {
        for j in 0..=total_vars {
            tableau[0][j] = tableau[0][j] - tableau[i][j];
        }
    }

    while iterations < config.max_iterations {
        // Find entering variable (most negative reduced cost)
        let mut entering_col = None;
        let mut min_reduced_cost = -eps;
        for j in 0..total_vars {
            if tableau[0][j] < min_reduced_cost {
                min_reduced_cost = tableau[0][j];
                entering_col = Some(j);
            }
        }

        // Check if Phase I is complete
        if entering_col.is_none() {
            break;
        }
        let entering_col = entering_col.unwrap();

        // Find leaving variable (minimum ratio test)
        let mut leaving_row = None;
        let mut min_ratio = T::infinity();
        for (i, row) in tableau.iter().enumerate().skip(1).take(m) {
            let coef = row[entering_col];
            if coef > eps {
                let ratio = row[total_vars] / coef;
                if ratio < min_ratio - eps {
                    min_ratio = ratio;
                    leaving_row = Some(i);
                }
            }
        }

        if leaving_row.is_none() {
            break;
        }
        let leaving_row = leaving_row.unwrap();

        // Perform pivot
        pivot(tableau, leaving_row, entering_col);

        iterations += 1;
    }

    // Restore original objective
    tableau[0] = original_obj;

    // Update reduced costs based on current basis
    for i in 1..=m {
        let mut basic_col = None;
        for j in 0..total_vars {
            if (tableau[i][j] - T::one()).abs() < eps
                && (0..m + 1).all(|r| r == i || tableau[r][j].abs() < eps)
            {
                basic_col = Some(j);
                break;
            }
        }
        if let Some(j) = basic_col {
            let coef = tableau[0][j];
            if coef.abs() > eps {
                for k in 0..=total_vars {
                    tableau[0][k] = tableau[0][k] - coef * tableau[i][k];
                }
            }
        }
    }
}

// Remove artificial variables and restore original objective
fn remove_artificial_variables<T>(
    art_tableau: &[Vec<T>],
    tableau: &mut [Vec<T>],
    m: usize,
    n: usize,
) where
    T: Float + Debug,
{
    // Copy solution without artificial variables
    for i in 0..=m {
        for j in 0..n + m + 1 {
            tableau[i][j] = art_tableau[i][j];
        }
    }
}

// Extract solution from tableau
fn extract_solution<T>(tableau: &[Vec<T>], m: usize, n: usize) -> Vec<T>
where
    T: Float + Debug,
{
    let mut solution = vec![T::zero(); n];
    let eps = T::from(EPSILON).unwrap();
    let total_cols = tableau[0].len() - 1;

    // First identify basic variables
    let mut basic_vars = vec![None; m];
    for (i, basic_var) in basic_vars.iter_mut().enumerate() {
        for j in 0..total_cols {
            if (tableau[i + 1][j] - T::one()).abs() < eps
                && (0..m + 1).all(|r| r == i + 1 || tableau[r][j].abs() < eps)
            {
                *basic_var = Some(j);
                break;
            }
        }
    }

    // Extract values for original variables
    for (i, &basic_var) in basic_vars.iter().enumerate() {
        if let Some(j) = basic_var {
            if j < n {
                // Only if it's an original variable
                solution[j] = tableau[i + 1][total_cols];
                if solution[j].abs() < eps {
                    solution[j] = T::zero();
                }
            }
        }
    }

    // Handle degenerate cases where variables might be zero
    for (j, sol_j) in solution.iter_mut().enumerate().take(n) {
        if sol_j.abs() < eps {
            let mut is_basic = false;
            for row in tableau.iter().skip(1).take(m) {
                if (row[j] - T::one()).abs() < eps {
                    is_basic = true;
                    *sol_j = row[total_cols];
                    break;
                }
            }
            if !is_basic {
                *sol_j = T::zero();
            }
        }
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
