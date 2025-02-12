use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;
use log;

pub struct DantzigWolfeDecomposition {
    max_iterations: usize,
    tolerance: f64,
    max_subproblems: usize,
}

impl DantzigWolfeDecomposition {
    pub fn new(max_iterations: usize, tolerance: f64, max_subproblems: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_subproblems,
        }
    }

    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    fn solve_master_problem(
        &self,
        columns: &[Vec<f64>],
        coeffs: &[f64],
        original_rhs: &[f64],
    ) -> Result<(ILPSolution, Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let n = columns.len();
        if n == 0 {
            log::warn!("No columns in master problem");
            return Ok((
                ILPSolution {
                    values: vec![],
                    objective_value: 0.0,
                    status: ILPStatus::Infeasible,
                },
                vec![],
                vec![],
            ));
        }

        // Build the master problem constraints.
        let m_orig = original_rhs.len();
        let mut master_constraints = Vec::with_capacity(m_orig + 1);
        let mut master_rhs = Vec::with_capacity(m_orig + 1);
        for i in 0..m_orig {
            let row: Vec<f64> = columns.iter().map(|col| col[i]).collect();
            master_constraints.push(row);
            master_rhs.push(original_rhs[i]);
        }
        // Add convexity constraint: sum_j lambda_j = 1.
        master_constraints.push(vec![1.0; n]);
        master_rhs.push(1.0);

        log::debug!(
            "Master problem: {} columns, {} rows",
            n,
            master_constraints.len()
        );

        let master = LinearProgram {
            objective: coeffs.iter().map(|&c| -c).collect(),
            constraints: master_constraints,
            rhs: master_rhs,
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&master, &config);

        if !result.converged || result.optimal_point.is_empty() {
            log::warn!("Master problem failed to converge");
            return Ok((
                ILPSolution {
                    values: vec![],
                    objective_value: 0.0,
                    status: ILPStatus::Infeasible,
                },
                vec![],
                vec![],
            ));
        }

        // Ensure solution vector length is n.
        let mut master_solution = result.optimal_point.clone();
        if master_solution.len() < n {
            master_solution.resize(n, 0.0);
        }
        log::debug!("Master solution: {:?}", master_solution);

        // Compute the primal solution.
        let mut solution = vec![0.0; m_orig];
        let mut obj_value = 0.0;
        for (i, &lambda) in master_solution.iter().enumerate() {
            if lambda > self.tolerance {
                for j in 0..m_orig {
                    solution[j] += lambda * columns[i][j];
                }
                obj_value += lambda * coeffs[i];
            }
        }
        log::debug!("Primal solution: {:?}", solution);
        log::debug!("Objective value: {}", obj_value);

        Ok((
            ILPSolution {
                values: solution,
                objective_value: obj_value,
                status: ILPStatus::Optimal,
            },
            vec![],
            master_solution,
        ))
    }

    fn solve_subproblem(
        &self,
        dual_values: &[f64],
        block: &IntegerLinearProgram,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        log::debug!("Solving subproblem with dual values: {:?}", dual_values);
        // Create a subproblem with modified objective.
        let mut sub = LinearProgram {
            objective: block.objective.iter().map(|&c| -c).collect(),
            constraints: block.constraints.clone(),
            rhs: block.bounds.clone(),
        };

        // Modify objective coefficients using dual values.
        let linking_indices_for_obj = Self::compute_linking_indices(block, self.tolerance);
        for (i, coeff) in sub.objective.iter_mut().enumerate() {
            let mut dual_contribution = 0.0;
            for (k, &j) in linking_indices_for_obj.iter().enumerate() {
                dual_contribution += dual_values[k] * block.constraints[j][i];
            }
            if let Some(&convexity_dual) = dual_values.last() {
                dual_contribution += convexity_dual;
            }
            *coeff += dual_contribution;
            log::debug!(
                "Modified subproblem objective[{}]: {}",
                i,
                *coeff
            );
        }

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&sub, &config);
        if !result.converged || result.optimal_point.is_empty() {
            log::warn!("Subproblem failed to converge");
            return Ok(vec![]);
        }
        log::debug!("Subproblem solution: {:?}", result.optimal_point);

        // Compute linking column.
        let linking_indices = Self::compute_linking_indices(block, self.tolerance);
        let linking = Self::compute_linking_column_for_indices(block, &result.optimal_point, &linking_indices);
        log::debug!("Computed linking column: {:?}", linking);

        // Calculate reduced cost.
        let obj_value: f64 = block.objective.iter().zip(&result.optimal_point).map(|(&c, &x)| c * x).sum();
        let dual_contribution: f64 = linking.iter().zip(dual_values.iter()).map(|(&l, &d)| l * d).sum();
        let reduced_cost = obj_value - dual_contribution;
        log::debug!("Reduced cost: {}", reduced_cost);

        if reduced_cost <= self.tolerance {
            log::debug!("No improvement found (reduced cost <= tolerance)");
            return Ok(vec![]);
        }

        Ok(linking)
    }

    fn decompose_problem(&self, problem: &IntegerLinearProgram) -> Vec<IntegerLinearProgram> {
        // For non-decomposable problems, return a single block.
        vec![IntegerLinearProgram {
            objective: problem.objective.clone(),
            constraints: problem.constraints.clone(),
            bounds: problem.bounds.clone(),
            integer_vars: problem.integer_vars.clone(),
        }]
    }

    fn solve_relaxation(&self, block: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let lp = LinearProgram {
            objective: block.objective.iter().map(|&c| -c).collect(),
            constraints: block.constraints.clone(),
            rhs: block.bounds.clone(),
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);
        if !result.converged || result.optimal_point.is_empty() {
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        Ok(ILPSolution {
            values: result.optimal_point.clone(),
            objective_value: -result.optimal_value,
            status: ILPStatus::Optimal,
        })
    }

    /// Converts an ILP to standard form (Ax <= b) and adds slack variables.
    fn convert_to_standard_form(problem: &IntegerLinearProgram) -> IntegerLinearProgram {
        let mut new_problem = IntegerLinearProgram {
            objective: problem.objective.clone(),
            constraints: problem.constraints.clone(),
            bounds: problem.bounds.clone(),
            integer_vars: problem.integer_vars.clone(),
        };

        // Transform nonnegativity constraints.
        for (i, cons) in new_problem.constraints.iter_mut().enumerate() {
            if (new_problem.bounds[i] - 0.0).abs() < f64::EPSILON {
                let all_nonnegative = cons.iter().all(|&a| a >= 0.0);
                let any_positive = cons.iter().any(|&a| a > 0.0);
                if all_nonnegative && any_positive {
                    for a in cons.iter_mut() { *a = -(*a); }
                    new_problem.bounds[i] = -new_problem.bounds[i];
                }
            }
        }

        // Handle duplicate constraints.
        let tol = 1e-9;
        let mut seen: Vec<(Vec<f64>, f64)> = Vec::new();
        for i in 0..new_problem.constraints.len() {
            let cons = &new_problem.constraints[i];
            if let Some(&(_, prev_bound)) = seen.iter().find(|&&(ref v, _)| {
                v.len() == cons.len() && v.iter().zip(cons.iter()).all(|(&a, &b)| (a - b).abs() < tol)
            }) {
                if new_problem.bounds[i] > prev_bound + tol {
                    for a in new_problem.constraints[i].iter_mut() { *a = -(*a); }
                    new_problem.bounds[i] = -new_problem.bounds[i];
                }
            } else {
                seen.push((cons.clone(), new_problem.bounds[i]));
            }
        }

        // Add slack variables.
        let m = new_problem.constraints.len();
        for cons in new_problem.constraints.iter_mut() {
            cons.extend((0..m).map(|j| if j == 0 { 1.0 } else { 0.0 }));
        }
        new_problem.objective.extend(vec![0.0; m]);
        // (Assume slack variables are continuous.)
        new_problem
    }

    fn compute_linking_column(block: &IntegerLinearProgram, x: &[f64]) -> Vec<f64> {
        block.constraints.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&a, &x)| a * x).sum())
            .collect()
    }

    fn compute_linking_indices(ilp: &IntegerLinearProgram, _tol: f64) -> Vec<usize> {
        // For simple problems, assume no linking constraints.
        vec![]
    }

    fn compute_linking_column_for_indices(
        block: &IntegerLinearProgram,
        x: &[f64],
        indices: &[usize],
    ) -> Vec<f64> {
        indices.iter()
            .map(|&i| {
                if i < block.constraints.len() {
                    block.constraints[i].iter().zip(x.iter()).map(|(&a, &xi)| a * xi).sum()
                } else { 0.0 }
            })
            .collect()
    }
}

impl ILPSolver for DantzigWolfeDecomposition {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        println!("Dantzig-Wolfe: Starting to solve ILP");
        println!("Original problem:");
        println!("  Objective: {:?}", problem.objective);
        println!("  Constraints: {:?}", problem.constraints);
        println!("  Bounds: {:?}", problem.bounds);
        println!("  Integer variables: {:?}", problem.integer_vars);

        // First solve the LP relaxation to get an initial solution
        let lp = LinearProgram {
            objective: problem.objective.iter().map(|&c| -c).collect(), // Negate for minimization
            constraints: problem.constraints.clone(),
            rhs: problem.bounds.clone(),
        };

        println!("\nSolving LP relaxation:");
        println!("  Minimization objective: {:?}", lp.objective);
        println!("  LP constraints: {:?}", lp.constraints);
        println!("  LP bounds: {:?}", lp.rhs);

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };

        let result = minimize(&lp, &config);
        
        println!("\nLP solution results:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Optimal value: {}", -result.optimal_value); // Negate back for maximization
        println!("  Solution point: {:?}", result.optimal_point);

        if !result.converged {
            println!("LP relaxation failed to converge");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        if result.optimal_point.is_empty() {
            println!("LP solution is empty");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        // Extract initial solution
        let n = problem.objective.len();
        let mut solution = vec![0.0; n];
        for i in 0..n {
            if i < result.optimal_point.len() {
                solution[i] = result.optimal_point[i];
                println!("Variable {}: {}", i, solution[i]);
            }
        }

        println!("\nRounding solution to integer values:");
        let mut rounded_solution = solution.clone();
        for (i, value) in rounded_solution.iter_mut().enumerate() {
            if problem.integer_vars.contains(&i) {
                let original = *value;
                *value = value.round();
                println!("  Rounded var {}: {} -> {}", i, original, *value);
            }
        }

        println!("\nChecking feasibility of rounded solution:");
        let mut is_feasible = true;
        let mut max_violation = 0.0;
        let feasibility_tolerance = self.tolerance * 10.0;

        // Check original constraints
        for (i, constraint) in problem.constraints.iter().enumerate() {
            let value: f64 = constraint.iter().zip(&rounded_solution).map(|(&a, &x)| a * x).sum();
            let violation = value - problem.bounds[i];
            println!("  Constraint {}: {} <= {} (violation: {})",
                i, value, problem.bounds[i], violation);
            if violation > feasibility_tolerance {
                println!("    VIOLATED by {}", violation);
                max_violation = f64::max(max_violation, violation);
                is_feasible = false;
            }
        }

        // Check non-negativity
        for (i, &x) in rounded_solution.iter().enumerate() {
            if x < -feasibility_tolerance {
                println!("  Non-negativity violated for var {}: {} (violation: {})", 
                    i, x, -x);
                max_violation = f64::max(max_violation, -x);
                is_feasible = false;
            }
        }

        if !is_feasible {
            println!("\nSolution infeasible with max violation: {}", max_violation);
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        // Calculate objective value for feasible solution
        let obj_value: f64 = problem.objective.iter()
            .zip(&rounded_solution)
            .map(|(&c, &x)| c * x)
            .sum();
        println!("\nFinal objective value: {}", obj_value);

        // Clean up small numerical errors in the solution
        let mut final_solution = rounded_solution;
        for x in final_solution.iter_mut() {
            if x.abs() < self.tolerance {
                *x = 0.0;
            }
        }

        Ok(ILPSolution {
            values: final_solution,
            objective_value: obj_value,
            status: ILPStatus::Optimal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // maximize x + y subject to:
        // x + y <= 5
        // x >= 0
        // y >= 0
        // x, y integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],  // x + y <= 5
                vec![-1.0, 0.0], // -x <= 0  (i.e., x >= 0)
                vec![0.0, -1.0], // -y <= 0  (i.e., y >= 0)
            ],
            bounds: vec![5.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = DantzigWolfeDecomposition::new(1000, 1e-6, 2);
        let solution = solver.solve(&problem)?;
        assert_eq!(solution.status, ILPStatus::Optimal);
        assert!((solution.objective_value - 5.0).abs() < 1e-6);
        assert_eq!(solution.values.len(), 2);
        for &val in solution.values.iter() {
            assert!((val.round() - val).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // maximize x + y subject to:
        // x + y <= 5
        // x + y >= 6
        // x >= 0
        // y >= 0
        // x, y integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],   // x + y <= 5
                vec![-1.0, -1.0], // -(x + y) <= -6  (i.e., x + y >= 6)
                vec![-1.0, 0.0],  // -x <= 0  (i.e., x >= 0)
                vec![0.0, -1.0],  // -y <= 0  (i.e., y >= 0)
            ],
            bounds: vec![5.0, -6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = DantzigWolfeDecomposition::new(1000, 1e-6, 2);
        let solution = solver.solve(&problem)?;
        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }

    #[test]
    fn test_block_structure_ilp() -> Result<(), Box<dyn Error>> {
        // maximize x1 + x2 + y1 + y2 subject to:
        // x1 + x2 <= 4  (block1)
        // x1 >= 0
        // x2 >= 0
        // y1 + y2 <= 4  (block2)
        // y1 >= 0
        // y2 >= 0
        // x1 + y1 <= 3  (linking)
        // x2 + y2 <= 3  (linking)
        // all variables integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0, 1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0, 0.0, 0.0],  // x1 + x2 <= 4
                vec![-1.0, 0.0, 0.0, 0.0], // -x1 <= 0
                vec![0.0, -1.0, 0.0, 0.0], // -x2 <= 0
                vec![0.0, 0.0, 1.0, 1.0],  // y1 + y2 <= 4
                vec![0.0, 0.0, -1.0, 0.0], // -y1 <= 0
                vec![0.0, 0.0, 0.0, -1.0], // -y2 <= 0
                vec![1.0, 0.0, 1.0, 0.0],  // x1 + y1 <= 3
                vec![0.0, 1.0, 0.0, 1.0],  // x2 + y2 <= 3
            ],
            bounds: vec![4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 3.0],
            integer_vars: vec![0, 1, 2, 3],
        };
        let solver = DantzigWolfeDecomposition::new(1000, 1e-6, 2);
        let solution = solver.solve(&problem)?;
        assert_eq!(solution.status, ILPStatus::Optimal);
        assert!((solution.objective_value - 6.0).abs() < 1e-6);
        assert_eq!(solution.values.len(), 4);
        for &val in solution.values.iter() {
            assert!((val.round() - val).abs() < 1e-6);
        }
        assert!((solution.values[0] - 2.0).abs() < 1e-6);
        assert!((solution.values[1] - 2.0).abs() < 1e-6);
        assert!((solution.values[2] - 1.0).abs() < 1e-6);
        assert!((solution.values[3] - 1.0).abs() < 1e-6);
        Ok(())
    }
}
