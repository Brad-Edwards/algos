#[derive(Debug, Clone, PartialEq)]
pub enum ILPStatus {
    Optimal,
    Infeasible,
    MaxIterationsReached,
}

#[derive(Debug, Clone)]
pub struct ILPSolution {
    pub values: Vec<f64>,
    pub objective_value: f64,
    pub status: ILPStatus,
}

pub struct IntegerLinearProgram {
    pub objective: Vec<f64>,
    pub constraints: Vec<Vec<f64>>,
    pub bounds: Vec<f64>,
    /// List of indices which are integer variables (the rest are continuous)
    pub integer_vars: Vec<usize>,
}

pub trait ILPSolver {
    fn solve(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn std::error::Error>>;
}

// -----------------------------------------------------------------------
// A simple local "minimize" implementation for a linear program
// with no global or static data. We do two-phase or any basic approach.
//
// LinearProgram: we want to minimize c^T x subject to A x <= b, x >= 0
// If your constraints have negative bounds or any advanced pivoting, you'll
// need a two-phase approach. This example is enough to illustrate the idea.
// -----------------------------------------------------------------------
#[derive(Debug)]
pub struct LinearProgram {
    pub objective: Vec<f64>,        // c
    pub constraints: Vec<Vec<f64>>, // rows of A
    pub rhs: Vec<f64>,              // b
}

/// Configuration for the optimizer
#[derive(Debug)]
pub struct OptimizationConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub learning_rate: f64, // not really used here, just a placeholder
}

#[derive(Debug)]
pub struct MinimizeResult {
    pub converged: bool,
    pub optimal_value: f64,
    pub optimal_point: Vec<f64>,
}

/// A minimal local simplex‐style routine.  
/// Only handles "A x <= b, x >= 0, minimize c^T x".
/// If the problem is unbounded or infeasible under these assumptions, we might detect it
/// or just mark "converged: false".
pub fn minimize(lp: &LinearProgram, cfg: &OptimizationConfig) -> MinimizeResult {
    let n = lp.objective.len();
    let m = lp.constraints.len();

    // Build an augmented tableau for standard simplex:
    //   We add slack variables s_i for each inequality: A x + s = b
    //   Then we have n + m variables in total: x_1..x_n, s_1..s_m
    //   Basic variables initially are the s_i's, each row has one s_i = b_i
    // The tableau has shape (m+1) x (n+m+1)
    //   last column is the RHS
    //   last row is the objective row
    // (We do not do a formal 2‐phase method, so if b_i < 0 or other edge cases appear,
    //  this code might fail.)
    let mut tableau = vec![vec![0.0; n + m + 1]; m + 1];

    // Fill the top m rows with A and slack identity
    for i in 0..m {
        for j in 0..n {
            tableau[i][j] = lp.constraints[i][j];
        }
        // slack variable for row i
        tableau[i][n + i] = 1.0;
        // RHS
        tableau[i][n + m] = lp.rhs[i];
    }

    // Objective row: we want to minimize c^T x
    for j in 0..n {
        tableau[m][j] = lp.objective[j];
    }
    // Slack columns in the objective row are 0
    // RHS is 0

    // Basic iterative pivoting
    let mut iterations = 0;
    loop {
        // 1) Find entering column (most negative entry in objective row => largest drive upwards)
        let mut pivot_col = None;
        let mut min_val = -cfg.tolerance;
        for col in 0..(n + m) {
            let val = tableau[m][col];
            if val < min_val {
                min_val = val;
                pivot_col = Some(col);
            }
        }
        if pivot_col.is_none() {
            // No negative entries => optimal solution found
            break;
        }
        let pivot_col = pivot_col.unwrap();

        // 2) Find leaving row by min ratio test
        let mut pivot_row = None;
        let mut best_ratio = f64::MAX;
        for row in 0..m {
            let coeff = tableau[row][pivot_col];
            if coeff > cfg.tolerance {
                let ratio = tableau[row][n + m] / coeff;
                if ratio < best_ratio {
                    best_ratio = ratio;
                    pivot_row = Some(row);
                }
            }
        }
        if pivot_row.is_none() {
            // Unbounded in pivot_col direction
            // We'll just declare "did not converge"
            return MinimizeResult {
                converged: false,
                optimal_value: 0.0,
                optimal_point: vec![0.0; n],
            };
        }
        let pivot_row = pivot_row.unwrap();

        // 3) Pivot around (pivot_row, pivot_col)
        let pivot_val = tableau[pivot_row][pivot_col];
        if pivot_val.abs() < 1e-15 {
            // degenerate pivot => skip or break
            return MinimizeResult {
                converged: false,
                optimal_value: 0.0,
                optimal_point: vec![0.0; n],
            };
        }
        // Normalize pivot row
        for col in 0..(n + m + 1) {
            tableau[pivot_row][col] /= pivot_val;
        }
        // Eliminate pivot_col in all other rows
        for row in 0..(m + 1) {
            if row != pivot_row {
                let factor = tableau[row][pivot_col];
                for col in 0..(n + m + 1) {
                    if col == pivot_col {
                        tableau[row][col] = 0.0;
                    } else {
                        tableau[row][col] -= factor * tableau[pivot_row][col];
                    }
                }
            }
        }

        iterations += 1;
        if iterations >= cfg.max_iterations {
            return MinimizeResult {
                converged: false,
                optimal_value: 0.0,
                optimal_point: vec![0.0; n],
            };
        }
    }

    // If we reach here, we have an optimal tableau. Extract solution:
    // Among x_1..x_n and s_1..s_m, whichever columns correspond to basic variables with a single 1
    // in that column => read from the RHS. Others => 0.
    let mut x = vec![0.0; n];
    for row in 0..m {
        // try to see if row is a basic variable for some column
        // find the column that has a 1 in this row and 0 in others
        let mut pivot_col = None;
        for col in 0..(n + m) {
            if (tableau[row][col] - 1.0).abs() < 1e-12 {
                // check if the rest of that column is 0 in other rows
                let mut is_pivot = true;
                for r2 in 0..m {
                    if r2 != row && tableau[r2][col].abs() > 1e-12 {
                        is_pivot = false;
                        break;
                    }
                }
                if is_pivot {
                    pivot_col = Some(col);
                    break;
                }
            }
        }
        if let Some(col_idx) = pivot_col {
            // if it's among the original x variables, record in x
            if col_idx < n {
                x[col_idx] = tableau[row][n + m];
            }
        }
    }

    // The objective row last cell is the value of the *transformed* objective:
    // We were maximizing -c^T x, so that final row's RHS is -optimal_value
    let final_obj = tableau[m][n + m];

    MinimizeResult {
        converged: true,
        optimal_value: final_obj,
        optimal_point: x,
    }
}

// -----------------------------------------------------------------------
// Benders Decomposition Implementation
// -----------------------------------------------------------------------
use std::collections::HashSet;
use std::error::Error;

pub struct BendersDecomposition {
    max_iterations: usize,
    tolerance: f64,
    max_cuts_per_iteration: usize,
}

impl BendersDecomposition {
    pub fn new(max_iterations: usize, tolerance: f64, max_cuts_per_iteration: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_cuts_per_iteration,
        }
    }

    fn only_contains_integer_vars(coeffs: &[f64], int_vars: &[usize]) -> bool {
        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff.abs() > 1e-14 && !int_vars.contains(&i) {
                return false;
            }
        }
        true
    }

    // Partition constraints: those that involve only integer (master) variables go to the master.
    // If any constraint references a continuous variable, it goes to the subproblem set.
    fn partition_constraints(
        &self,
        problem: &IntegerLinearProgram,
    ) -> (Vec<(Vec<f64>, f64)>, Vec<(Vec<f64>, f64)>) {
        let int_var_set: HashSet<usize> = problem.integer_vars.iter().copied().collect();
        let mut master_cons = Vec::new();
        let mut sub_cons = Vec::new();

        for (coeffs, &rhs) in problem.constraints.iter().zip(problem.bounds.iter()) {
            let mut has_continuous = false;
            for (j, &c) in coeffs.iter().enumerate() {
                if !int_var_set.contains(&j) && c.abs() > 1e-14 {
                    has_continuous = true;
                    break;
                }
            }
            if has_continuous {
                sub_cons.push((coeffs.clone(), rhs));
            } else {
                master_cons.push((coeffs.clone(), rhs));
            }
        }
        (master_cons, sub_cons)
    }

    fn solve_master_problem(
        &self,
        cuts: &[(Vec<f64>, f64)],
        problem: &IntegerLinearProgram,
        master_constraints: &[(Vec<f64>, f64)],
    ) -> Result<ILPSolution, Box<dyn Error>> {
        eprintln!("\nSolving master problem with {} cuts", cuts.len());

        // Combine the original master constraints with Benders cuts
        let mut m_cons = Vec::new();
        let mut m_rhs = Vec::new();
        for (coeffs, rhs) in master_constraints {
            m_cons.push(coeffs.clone());
            m_rhs.push(*rhs);
        }
        for (coeffs, rhs) in cuts {
            m_cons.push(coeffs.clone());
            m_rhs.push(*rhs);
        }

        // Fix: Properly handle maximization by negating objective coefficients
        let mut objective = vec![0.0; problem.objective.len()];
        for i in 0..problem.objective.len() {
            if problem.integer_vars.contains(&i) {
                // Negate because we're converting max to min
                objective[i] = -problem.objective[i];
            }
        }

        // Add objective term for continuous variables (eta)
        if !problem
            .integer_vars
            .contains(&(problem.objective.len() - 1))
        {
            objective.push(1.0);
        }

        let lp = LinearProgram {
            objective,
            constraints: m_cons,
            rhs: m_rhs,
        };

        eprintln!("Master problem formulation:");
        eprintln!("Objective: {:?}", lp.objective);
        for (i, (cc, rr)) in lp.constraints.iter().zip(lp.rhs.iter()).enumerate() {
            eprintln!("  {}: {:?} <= {}", i, cc, rr);
        }

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let res = minimize(&lp, &config);

        eprintln!("Master problem solution:");
        eprintln!("  Converged: {}", res.converged);
        eprintln!("  Objective: {}", -res.optimal_value);
        eprintln!("  Solution: {:?}", res.optimal_point);

        if !res.converged {
            eprintln!("Master problem did not converge => returning infeasible placeholder.");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        // Check feasibility in original constraints
        let x = &res.optimal_point;
        let mut feasible = true;
        for (i, (coeffs, &rhs)) in problem
            .constraints
            .iter()
            .zip(problem.bounds.iter())
            .enumerate()
        {
            let lhs: f64 = coeffs.iter().zip(x.iter()).map(|(&a, &xx)| a * xx).sum();
            if lhs > rhs + self.tolerance {
                eprintln!("Original constraint {} violated: {} > {}", i, lhs, rhs);
                feasible = false;
                break;
            }
        }
        if !feasible {
            eprintln!("Master solution is infeasible w.r.t. the original constraints.");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        Ok(ILPSolution {
            values: x.clone(),
            objective_value: res.optimal_value,
            status: ILPStatus::Optimal,
        })
    }

    fn solve_subproblem(
        &self,
        fixed_vars: &[f64],
        problem: &IntegerLinearProgram,
        subproblem_constraints: &[(Vec<f64>, f64)],
    ) -> Result<(f64, Vec<f64>, f64), Box<dyn Error>> {
        eprintln!(
            "\nSolving subproblem with fixed integer variables: {:?}",
            fixed_vars
        );

        // Fix: Correct objective handling for subproblem
        let mut sub_objective = vec![0.0; problem.objective.len()];
        for i in 0..problem.objective.len() {
            if !problem.integer_vars.contains(&i) {
                sub_objective[i] = problem.objective[i];
            }
        }

        let mut sub_cons = Vec::new();
        let mut sub_rhs = Vec::new();
        for (coeffs, rhs) in subproblem_constraints {
            if !Self::only_contains_integer_vars(coeffs, &problem.integer_vars) {
                let new_coeffs = coeffs.clone();
                let mut new_rhs = *rhs;
                // Adjust RHS based on fixed integer variables
                for (i, &val) in fixed_vars.iter().enumerate() {
                    if problem.integer_vars.contains(&i) {
                        new_rhs -= coeffs[i] * val;
                    }
                }
                sub_cons.push(new_coeffs);
                sub_rhs.push(new_rhs);
            }
        }

        let lp = LinearProgram {
            objective: sub_objective,
            constraints: sub_cons,
            rhs: sub_rhs,
        };
        eprintln!("Subproblem formulation:");
        eprintln!("Objective: {:?}", lp.objective);
        for (i, (cc, rr)) in lp.constraints.iter().zip(lp.rhs.iter()).enumerate() {
            eprintln!("  {}: {:?} <= {}", i, cc, rr);
        }

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let res = minimize(&lp, &config);
        eprintln!("Subproblem solution:");
        eprintln!("  Converged: {}", res.converged);
        eprintln!("  Objective: {}", -res.optimal_value);
        eprintln!("  Solution: {:?}", res.optimal_point);

        if !res.converged {
            return Err("Subproblem did not converge or was unbounded.".into());
        }

        // Evaluate subproblem objective in the original objective sense
        let sub_x = &res.optimal_point;
        let sub_obj: f64 = sub_x
            .iter()
            .zip(problem.objective.iter())
            .map(|(&xx, &c)| xx * c)
            .sum();

        eprintln!("Calculated subproblem objective: {}", sub_obj);

        // Benders cut:
        //   We want a linear cut of the form: sum_i cut_coeffs[i] * x_i <= cut_rhs
        //   For each integer var i, cut_coeffs[i] = -C_i (the objective coeff).
        //   cut_rhs = sub_obj + sum_i( C_i * fixed_vars[i] ) for integer i
        let mut cut_coeffs = vec![0.0; problem.objective.len()];
        let mut cut_rhs = sub_obj;
        for i in problem.integer_vars.iter().copied() {
            let c_i = problem.objective[i];
            cut_coeffs[i] = -c_i;
            cut_rhs += c_i * fixed_vars[i];
        }

        Ok((sub_obj, cut_coeffs, cut_rhs))
    }
}

impl ILPSolver for BendersDecomposition {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        eprintln!("\nSolving ILP with Benders Decomposition:");
        eprintln!("Objective: {:?}", problem.objective);
        for (i, (con, &rhs)) in problem
            .constraints
            .iter()
            .zip(problem.bounds.iter())
            .enumerate()
        {
            eprintln!("  {}: {:?} <= {}", i, con, rhs);
        }
        eprintln!("Integer vars: {:?}", problem.integer_vars);

        let (master_cons, sub_cons) = self.partition_constraints(problem);

        let mut cuts = Vec::new();
        let mut best_sol = None;
        let mut best_obj = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut no_improvement = 0;

        while iterations < self.max_iterations && no_improvement < 5 {
            iterations += 1;
            eprintln!("\nIteration {}", iterations);

            // Solve master
            let master_sol = match self.solve_master_problem(&cuts, problem, &master_cons) {
                Ok(sol) => sol,
                Err(_) => {
                    eprintln!("Master solve failed unexpectedly.");
                    break;
                }
            };

            if master_sol.status == ILPStatus::Infeasible {
                // If first iteration with no cuts is infeasible => overall infeasible
                if iterations == 1 && cuts.is_empty() {
                    eprintln!("Initial master problem infeasible => entire problem infeasible.");
                    return Ok(ILPSolution {
                        values: vec![],
                        objective_value: 0.0,
                        status: ILPStatus::Infeasible,
                    });
                }
                // otherwise just continue
                continue;
            }

            // Check integer feasibility
            let mut is_int = true;
            for &i_idx in &problem.integer_vars {
                let val = master_sol.values[i_idx];
                if (val - val.round()).abs() > self.tolerance {
                    eprintln!("Master solution var {} not integer: {}", i_idx, val);
                    is_int = false;
                    break;
                }
            }

            let master_obj = master_sol.objective_value;
            eprintln!("Master objective = {}", master_obj);

            if is_int {
                eprintln!("Master is integer-feasible solution!");
                if master_obj > best_obj + self.tolerance {
                    eprintln!(" -> Found a new best integer solution!");
                    best_obj = master_obj;
                    best_sol = Some(ILPSolution {
                        values: master_sol.values.clone(),
                        objective_value: master_obj,
                        status: ILPStatus::Optimal,
                    });
                    no_improvement = 0;
                } else {
                    no_improvement += 1;
                }
            }

            // Solve the subproblem with the integer variables from the master
            match self.solve_subproblem(&master_sol.values, problem, &sub_cons) {
                Ok((sub_obj, cut_coeffs, cut_rhs)) => {
                    eprintln!("Subproblem objective = {}", sub_obj);
                    // If sub_obj ~ master_obj => we are done
                    if (master_obj - sub_obj).abs() < self.tolerance {
                        eprintln!("Master & subproblem objectives match => converged.");
                        break;
                    }
                    // Check cut violation
                    let lhs: f64 = master_sol
                        .values
                        .iter()
                        .zip(&cut_coeffs)
                        .map(|(&xv, &cc)| xv * cc)
                        .sum();
                    let violation = lhs - cut_rhs;
                    if violation > self.tolerance {
                        eprintln!("Cut is violated ({}), adding cut!", violation);
                        cuts.push((cut_coeffs, cut_rhs));
                        no_improvement = 0;
                        if cuts.len() >= self.max_cuts_per_iteration {
                            eprintln!("Reached max cuts per iteration, next iteration...");
                            continue;
                        }
                    } else {
                        eprintln!("No significant cut violation ({}).", violation);
                        no_improvement += 1;
                    }
                }
                Err(_) => {
                    eprintln!("Subproblem infeasible => original problem likely infeasible");
                    if iterations == 1 && cuts.is_empty() {
                        return Ok(ILPSolution {
                            values: vec![],
                            objective_value: 0.0,
                            status: ILPStatus::Infeasible,
                        });
                    }
                    no_improvement += 1;
                }
            }
        }

        // End: return best known solution if any
        match best_sol {
            Some(sol) => Ok(sol),
            None => {
                let status = if iterations >= self.max_iterations {
                    ILPStatus::MaxIterationsReached
                } else {
                    ILPStatus::Infeasible
                };
                Ok(ILPSolution {
                    values: vec![],
                    objective_value: 0.0,
                    status,
                })
            }
        }
    }
}

// -----------------------------------------------------------------------
// Unit Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // Maximize x + y subject to:
        //   x + y <= 5
        //   x >= 0  => -x <= 0
        //   y >= 0  => -y <= 0
        // x,y ∈ Z
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],  // x + y <= 5
                vec![-1.0, 0.0], // x >= 0
                vec![0.0, -1.0], // y >= 0
            ],
            bounds: vec![5.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = BendersDecomposition::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Optimal);
        // Expect best objective is 5
        assert!((solution.objective_value - 5.0).abs() < 1e-6);
        // Both x,y should be integers
        for &i in &problem.integer_vars {
            let val = solution.values[i];
            assert!((val - val.round()).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // Maximize x + y subject to:
        //   x + y <= 5
        //   x + y >= 6  => -x - y <= -6
        //   x >= 0
        //   y >= 0
        // x,y ∈ Z
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],   // x + y <= 5
                vec![-1.0, -1.0], // x + y >= 6
                vec![-1.0, 0.0],  // x >= 0
                vec![0.0, -1.0],  // y >= 0
            ],
            bounds: vec![5.0, -6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = BendersDecomposition::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
