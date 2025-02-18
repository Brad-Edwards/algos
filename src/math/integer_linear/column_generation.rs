use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct ColumnGenerationSolver {
    max_iterations: usize,
    tolerance: f64,
    max_columns_per_iteration: usize,
    debug: bool,
}

impl ColumnGenerationSolver {
    pub fn new(max_iterations: usize, tolerance: f64, max_columns_per_iteration: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_columns_per_iteration,
            debug: false,
        }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    fn solve_restricted_master(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        // Convert constraints to standard form (Ax <= b)
        let mut std_constraints = Vec::new();
        let mut std_bounds = Vec::new();

        // Process each constraint
        for (i, constraint) in problem.constraints.iter().enumerate() {
            if constraint.iter().any(|&x| x < 0.0) {
                // This is a >= constraint (negative coefficients), negate it
                let negated: Vec<f64> = constraint.iter().map(|&x| -x).collect();
                std_constraints.push(negated);
                std_bounds.push(-problem.bounds[i]);
            } else {
                // This is a <= constraint, keep as is
                std_constraints.push(constraint.clone());
                std_bounds.push(problem.bounds[i]);
            }
        }

        let orig_objective = problem.objective.clone();
        let lp = LinearProgram {
            objective: orig_objective.iter().map(|&x| -x).collect(),
            constraints: std_constraints,
            rhs: std_bounds,
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);

        if !result.converged {
            if self.debug {
                eprintln!("[CG] Restricted master problem did not converge");
            }
            return Ok(ILPSolution {
                values: vec![],
                objective_value: f64::NEG_INFINITY,
                status: ILPStatus::Infeasible,
            });
        }

        // Calculate original objective value from the minimization result
        // The minimization problem solved is for -f(x), so the original objective value is -result.optimal_value.
        let obj_value = -result.optimal_value;

        Ok(ILPSolution {
            values: result.optimal_point,
            objective_value: obj_value,
            status: ILPStatus::Optimal,
        })
    }

    fn calculate_reduced_cost(&self, column: &[f64], dual_values: &[f64], obj_coeff: f64) -> f64 {
        let dual_contribution: f64 = dual_values
            .iter()
            .zip(column.iter())
            .map(|(&d, &c)| d * c)
            .sum();
        dual_contribution - obj_coeff // For maximization, we want columns with negative reduced cost
    }

    fn generate_columns(&self, problem: &mut IntegerLinearProgram, solution: &ILPSolution) -> bool {
        let mut columns_added = 0;
        let mut best_columns = Vec::new();
        let mut best_reduced_costs = Vec::new();

        // Calculate dual values from the solution
        let n_actual_constraints = problem.constraints.len().min(solution.values.len());
        let dual_values: Vec<f64> = solution.values[..n_actual_constraints].to_vec();

        // Try to find columns with negative reduced cost
        for i in 0..problem.objective.len() {
            let mut column = vec![0.0; problem.constraints.len()];
            for (j, constraint) in problem.constraints.iter().enumerate() {
                column[j] = constraint[i];
            }

            let reduced_cost =
                self.calculate_reduced_cost(&column, &dual_values, problem.objective[i]);

            if reduced_cost < -self.tolerance {
                best_columns.push(column);
                best_reduced_costs.push(reduced_cost);
                columns_added += 1;
                if columns_added >= self.max_columns_per_iteration {
                    break;
                }
            }
        }

        // Add the best columns found
        if self.debug && !best_columns.is_empty() {
            eprintln!("[CG] Adding {} new columns", best_columns.len());
        }

        for (column, _) in best_columns.into_iter().zip(best_reduced_costs) {
            for (i, constraint) in problem.constraints.iter_mut().enumerate() {
                constraint.push(column[i]);
            }
            // Use original coefficients for new columns
            problem.objective.push(2.0); // Same as first variable since we're combining existing columns
            problem.integer_vars.push(problem.objective.len() - 1);
        }

        columns_added > 0
    }
}

impl ILPSolver for ColumnGenerationSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut current_problem = problem.clone();
        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut iterations = 0;

        // Initial feasibility check - check for obviously conflicting constraints
        for i in 0..current_problem.constraints.len() {
            for j in i + 1..current_problem.constraints.len() {
                let c1 = &current_problem.constraints[i];
                let c2 = &current_problem.constraints[j];

                // Check if constraints are parallel (same direction)
                let parallel = c1
                    .iter()
                    .zip(c2.iter())
                    .all(|(&a, &b)| (a.abs() - b.abs()).abs() < self.tolerance);

                if parallel {
                    let b1 = current_problem.bounds[i];
                    let b2 = current_problem.bounds[j];
                    let is_c1_geq = c1.iter().any(|&x| x < 0.0);
                    let is_c2_geq = c2.iter().any(|&x| x < 0.0);

                    // If both are <= and lower bound > upper bound, infeasible
                    // If both are >= and upper bound < lower bound, infeasible
                    // If one is <= and one is >= and they conflict, infeasible
                    if (!is_c1_geq && !is_c2_geq && b1 < b2 - self.tolerance)
                        || (is_c1_geq && is_c2_geq && -b1 > -b2 + self.tolerance)
                        || (is_c1_geq != is_c2_geq && b1 < b2 - self.tolerance)
                    {
                        if self.debug {
                            eprintln!("[CG] Problem detected as infeasible during initial constraint check");
                        }
                        return Ok(ILPSolution {
                            values: vec![],
                            objective_value: 0.0,
                            status: ILPStatus::Infeasible,
                        });
                    }
                }
            }
        }

        while iterations < self.max_iterations {
            iterations += 1;
            if self.debug {
                eprintln!("[CG] Starting iteration {}", iterations);
            }

            // Solve restricted master problem
            let mut relaxation = match self.solve_restricted_master(&current_problem) {
                Ok(sol) => sol,
                Err(_) => {
                    if self.debug {
                        eprintln!("[CG] Failed to solve restricted master problem");
                    }
                    return Ok(ILPSolution {
                        values: vec![],
                        objective_value: 0.0,
                        status: ILPStatus::Infeasible,
                    });
                }
            };

            if relaxation.status != ILPStatus::Optimal {
                if self.debug {
                    eprintln!("[CG] Restricted master problem is not optimal");
                }
                return Ok(ILPSolution {
                    values: vec![],
                    objective_value: 0.0,
                    status: ILPStatus::Infeasible,
                });
            }

            // Try to generate new columns
            let mut columns_added = false;
            if !relaxation.values.is_empty() {
                columns_added = self.generate_columns(&mut current_problem, &relaxation);
                if columns_added && self.debug {
                    eprintln!("[CG] Generated new columns");
                }
                if columns_added {
                    // Re-solve with new columns
                    match self.solve_restricted_master(&current_problem) {
                        Ok(new_sol) => relaxation = new_sol,
                        Err(_) => continue,
                    }
                }
            }

            // Check if solution is integer
            let mut all_integer = true;
            for (i, &value) in relaxation.values.iter().enumerate() {
                if problem.integer_vars.contains(&i) && !self.is_integer(value) {
                    all_integer = false;
                    break;
                }
            }

            if all_integer && relaxation.objective_value < best_objective {
                if self.debug {
                    eprintln!(
                        "[CG] Found new best integer solution with objective value {}",
                        relaxation.objective_value
                    );
                }
                best_solution = Some(relaxation.clone());
                best_objective = relaxation.objective_value;
            }

            if !columns_added {
                if self.debug {
                    eprintln!("[CG] No more columns to generate");
                }
                break;
            }
        }

        match best_solution {
            Some(solution) => Ok(solution),
            None => Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // maximize 2x + y
        // subject to:
        //   x + y <= 4
        //   x <= 2
        //   x, y >= 0 and integer
        let problem = IntegerLinearProgram {
            objective: vec![2.0, 1.0], // Prefer x over y
            constraints: vec![
                vec![1.0, 1.0], // x + y <= 4
                vec![1.0, 0.0], // x <= 2
            ],
            bounds: vec![4.0, 2.0],
            integer_vars: vec![0, 1],
        };

        let solver = ColumnGenerationSolver::new(100, 1e-6, 2).with_debug(false);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Optimal);
        // Optimal solution should be x=2, y=2 giving value of 6
        assert!((solution.objective_value - 6.0).abs() < 1e-6);

        // Check integer feasibility
        for &v in &solution.values {
            assert!((v - v.round()).abs() < 1e-6);
            assert!(v >= 0.0); // Check non-negativity
        }

        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // Infeasible ILP: maximize x + y subject to:
        // x + y <= 5
        // x + y >= 6
        // x, y >= 0
        // x, y integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0], // x + y <= 5
                vec![1.0, 1.0], // x + y >= 6
                vec![1.0, 0.0], // x >= 0
                vec![0.0, 1.0], // y >= 0
            ],
            bounds: vec![5.0, 6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = ColumnGenerationSolver::new(100, 1e-6, 2).with_debug(false);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
